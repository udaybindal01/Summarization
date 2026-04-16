"""
train.py  —  Dual-Tower Dynamic Hypergraph Training Pipeline
=============================================================
Key differences from GraMFormer v2
------------------------------------
  - Primary dataset: MovieSum (1,800 movies vs MENSA's 500)
  - MovieHypergraphDataset replaces MovieGraphDatasetV2:
      builds incidence_matrix [N, S] and edge/entity type tensors per movie
      instead of three [S,S] static adjacency matrices
  - hypergraph_collate_fn replaces movie_collate_fn
  - Model: DualTowerHypergraphSummariser replaces GraMFormerV2
  - Loss: incidence_matrix passed instead of causal_adj for coherence loss
  - Ablation flags updated: --no_hypergraph, --static_hypergraph,
    --no_raft, --no_pointer_head, --no_coherence_loss, --no_contrastive_loss
  - Evaluation: ROUGE + BERTScore + METEOR + entity F1 (all 4 every epoch)
  - Cross-dataset: evaluate MovieSum-trained model on MENSA zero-shot

Paper baseline to implement separately:
  DiscoGraMS ported to screenplay domain (static pairwise graph + dual-tower)
  Run: python train.py --run_name discograms_baseline --static_hypergraph --no_gru
"""

import torch
import json
import os
import re
import math
import gzip
import sys
import wandb
import evaluate as hf_evaluate
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch._dynamo

torch._dynamo.config.cache_size_limit = 512
os.environ["USE_TORCH"] = "1"
os.environ["TMPDIR"]    = "/tmp/uday"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/tmp/uday/lib")

from peft import LoraConfig, get_peft_model

import sum as _sum_module
from sum import (
    DualTowerHypergraphSummariser,
    RelationalEventConsistencyLoss,
    RaftConsensusAttentionV2,
    log_hyperedge_attention,
    log_entity_state_norms,
    HYPEREDGE_TYPE_MAP,
    ENTITY_TYPE_MAP,
    MAX_ENTITIES,
    NUM_HYPEREDGE_TYPES,
    NUM_ENTITY_TYPES,
)
from visualize_graph import log_hypergraph_to_wandb

torch.set_float32_matmul_precision("high")

# =============================================================================
# Argument parsing + ablation config
# =============================================================================
import argparse as _ap
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--run_name",           type=str,   default="full_model")
_p.add_argument("--bart_model",         type=str,
                default=("/tmp/uday/bart-large"
                         if os.path.isdir("/tmp/uday/bart-large")
                         else "facebook/bart-large"))
_p.add_argument("--bart_tokenizer",     type=str,   default="")
_p.add_argument("--d_model",            type=int,   default=1024)
_p.add_argument("--num_layers",         type=int,   default=4)
# Ablation flags
_p.add_argument("--no_hypergraph",      action="store_true",
                help="Disable hypergraph tower entirely (text-only baseline)")
_p.add_argument("--static_hypergraph",  action="store_true",
                help="Freeze entity states after scene 0 (DiscoGraMS-style baseline)")
_p.add_argument("--no_gru",             action="store_true",
                help="Replace GRU update with simple EMA (α=0.7)")
_p.add_argument("--no_raft",            action="store_true",
                help="Disable RAFT cross-modal fusion in text tower")
_p.add_argument("--no_pointer_head",    action="store_true")
_p.add_argument("--no_coherence_loss",  action="store_true")
_p.add_argument("--no_contrastive_loss",action="store_true")
_p.add_argument("--entity_penalty",     type=float, default=3.0)
_p.add_argument("--dataset",            type=str,   default="moviesum",
                choices=["moviesum", "mensa", "both"])
# v4 hypergraph improvements
_p.add_argument("--no_adaptive_streams", action="store_true",
                help="Ablation: use global stream weights instead of scene-conditioned gating")
_p.add_argument("--no_entity_names",    action="store_true",
                help="Ablation: do not use entity name embeddings for initialization")
_p.add_argument("--edge_dropout",       type=float, default=0.1,
                help="Incidence matrix edge dropout rate during training (0 = disabled)")
_args, _ = _p.parse_known_args()

_BART_TOKENIZER = _args.bart_tokenizer or _args.bart_model

ABLATION = {
    "run_name":             _args.run_name,
    "bart_model":           _args.bart_model,
    "bart_tokenizer":       _BART_TOKENIZER,
    "d_model":              _args.d_model,
    "num_layers":           _args.num_layers,
    "no_hypergraph":        _args.no_hypergraph,
    "static_hypergraph":    _args.static_hypergraph,
    "no_gru":               _args.no_gru,
    "no_raft":              _args.no_raft,
    "no_pointer_head":      _args.no_pointer_head,
    "no_coherence_loss":    _args.no_coherence_loss,
    "no_contrastive_loss":  _args.no_contrastive_loss,
    "entity_penalty":       _args.entity_penalty,
    "dataset":              _args.dataset,
    "no_adaptive_streams":  _args.no_adaptive_streams,
    "no_entity_names":      _args.no_entity_names,
    "edge_dropout":         _args.edge_dropout,
}

# ── Paths ─────────────────────────────────────────────────────────────────────
MOVIESUM_JSONL   = "/tmp/uday/moviesum_data.jsonl.gz"
MENSA_JSONL      = "/tmp/uday/mensa_train_data.jsonl.gz"

TRAIN_SPLIT_PATH = f"/tmp/uday/train_{ABLATION['run_name']}.jsonl"
EVAL_SPLIT_PATH  = f"/tmp/uday/eval_{ABLATION['run_name']}.jsonl"

NUM_TRAIN_MOVIES = 1500   # MovieSum has ~2200 movies; keep ~700 for eval/test

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE         = 1
ACCUMULATION_STEPS = 16
EPOCHS_STAGE1      = 2    # RoBERTa + Mamba frozen; train hypergraph + decoder
EPOCHS_STAGE2      = 20   # LoRA on Mamba; full model except frozen RoBERTa
LR_NEW_LAYERS      = 1e-4
LR_LORA            = 1e-5
MAX_SEQ_LEN        = 256
MAX_SCENES         = 64


# =============================================================================
# Scene-level dataset (lazy byte-offset reader)
# =============================================================================

class SceneDataset(Dataset):
    """
    Reads a flat .jsonl file (uncompressed) scene by scene.
    Stores byte offsets for O(1) random access; ~50 MB RAM regardless of size.

    Fields returned per scene (dict):
        input_ids, target_ids, action_mask, dialogue_mask,
        entity_mask, header_mask, graph_triplets, characters,
        ner_entities, hyperedge_type, character_emotions, scene_meta, movie_id
    """

    def __init__(self, jsonl_path, max_seq_len=256):
        if jsonl_path.endswith(".gz"):
            raise ValueError(
                f"SceneDataset requires uncompressed .jsonl, got: {jsonl_path}\n"
                "Run split_dataset_by_movie() first — it writes plain .jsonl splits."
            )
        self.max_seq_len = max_seq_len
        self.jsonl_path  = jsonl_path
        self.tokenizer   = AutoTokenizer.from_pretrained(_BART_TOKENIZER)
        self.movie_ids   = []

        mid_re = re.compile(rb'"movie_id"\s*:\s*"([^"]+)"')
        print(f"Indexing {jsonl_path}...")
        self._offsets = []
        with open(jsonl_path, "rb") as f:
            while True:
                offset = f.tell()
                line   = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                m = mid_re.search(line)
                self._offsets.append(offset)
                self.movie_ids.append(m.group(1).decode() if m else "unknown")
        print(f"Indexed {len(self._offsets):,} scenes.")

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, idx):
        with open(self.jsonl_path, "rb") as f:
            f.seek(self._offsets[idx])
            item = json.loads(f.readline())

        L = self.max_seq_len

        def _trunc_pad(lst, pad=1):
            return (lst + [pad] * L)[:L]

        def _bool_t(key):
            return torch.tensor(_trunc_pad(item.get(key, [0] * L), 0), dtype=torch.bool)

        return {
            "movie_id":           item["movie_id"],
            "input_ids":          torch.tensor(_trunc_pad(item["input_ids"]),  dtype=torch.long),
            "target_ids":         torch.tensor(_trunc_pad(item["target_ids"]), dtype=torch.long),
            "action_mask":        _bool_t("action_mask"),
            "dialogue_mask":      _bool_t("dialogue_mask"),
            "entity_mask":        _bool_t("entity_mask"),
            "header_mask":        _bool_t("header_mask"),
            "graph_triplets":     item.get("graph_triplets", []),
            "characters":         item.get("characters", []),
            "ner_entities":       item.get("ner_entities", []),      # new
            "hyperedge_type":     item.get("hyperedge_type", "NEUTRAL"),  # new
            "character_emotions": item.get("character_emotions", {}),
            "scene_meta":         item.get("scene_meta", {}),
        }


# =============================================================================
# Movie-level hypergraph dataset
# =============================================================================

class MovieHypergraphDataset(Dataset):
    """
    Groups scenes by movie and builds per-movie hypergraph structure:

        incidence_matrix  [MAX_ENTITIES, MAX_SCENES]  float
            B[n, s] = 1.0 if entity n appears in scene s

        edge_type_ids     [MAX_SCENES]   long
            Dominant action type of each scene (0–4)

        entity_type_ids   [MAX_ENTITIES] long
            Semantic type of each entity node (0–4)

        entity_mask       [MAX_ENTITIES] bool
            True = valid entity (not padding)

    Entity registry is built from:
        1. NER entities (ner_entities field) — highest quality
        2. Speaker names (characters field) — screenplay-specific
        3. SVO subjects/objects (graph_triplets) — broader fallback
    """

    # Minimal stop set: only pronouns / determiners that are never entity names.
    # All other filtering is frequency-based (see _build_movie_graph).
    _STOP = frozenset({
        "him", "her", "he", "she", "they", "them", "it", "i",
        "we", "you", "who", "what", "that", "this", "me", "my",
        "his", "its", "our", "your", "their", "the", "an",
    })

    def __init__(self, scene_dataset, max_scenes=MAX_SCENES,
                 max_entities=MAX_ENTITIES):
        self.scene_dataset = scene_dataset
        self.max_scenes    = max_scenes
        self.max_entities  = max_entities
        self._cache        = {}

        print("Building movie index for hypergraph dataset...")
        self.movie_map = defaultdict(list)
        for i in range(len(scene_dataset)):
            raw = scene_dataset.movie_ids[i]
            name = raw.split("_Scene_")[0] if "_Scene_" in raw else raw
            self.movie_map[name].append(i)
        self.movie_names = list(self.movie_map.keys())
        print(f"  {len(self.movie_names):,} movies.")

    def __len__(self):
        return len(self.movie_names)

    # ── Entity extraction helpers ──────────────────────────────────────────

    def _scene_entities(self, scene):
        """
        Returns dict {normalized_name → entity_type_str} for one scene.
        Priority: NER > speaker names > SVO subjects/objects.
        """
        ents = {}

        # 1. NER entities (best quality)
        for e in scene.get("ner_entities", []):
            n = e.get("text", "").strip().lower()
            t = e.get("type", "OTHER")
            if n and len(n) > 1 and n not in self._STOP:
                ents[n] = t

        # 2. Speaker names (PERSON type)
        for char in scene.get("characters", []):
            n = char.strip().lower()
            if n and len(n) > 1 and n not in self._STOP:
                ents.setdefault(n, "PERSON")

        # 3. SVO subjects and objects (strict: only capitalized proper nouns)
        # Common nouns from SVO ("car", "door", "gun") appear in most scenes
        # and destroy incidence matrix sparsity. Only keep names (Title Case).
        for trip in scene.get("graph_triplets", []):
            parts = trip.split("_")
            for field_idx in (0, 2):
                if len(parts) > field_idx:
                    raw = parts[field_idx].replace("NOT ", "").strip()
                    # Only keep if it looks like a proper noun (starts uppercase)
                    if (raw and raw[0].isupper() and raw.isalpha()
                            and len(raw) > 2 and raw.lower() not in self._STOP):
                        ents.setdefault(raw.lower(), "PERSON")
        return ents

    # ── Main item builder ──────────────────────────────────────────────────

    # Frequency thresholds for automatic filtering (fraction of total scenes)
    _MAX_SCENE_FRAC = 0.50   # entities in >50% of scenes are generic noise
    _MIN_SCENES     = 1      # entities in only 1 scene are kept (may be important cameos)

    def __getitem__(self, idx):
        movie_name = self.movie_names[idx]

        if movie_name in self._cache:
            return self._cache[movie_name]

        all_indices = self.movie_map[movie_name]
        S = self.max_scenes
        N = self.max_entities

        # Stride-sample if movie > max_scenes
        if len(all_indices) > S:
            step   = len(all_indices) / S
            sel    = [all_indices[int(i * step)] for i in range(S)]
        else:
            sel    = all_indices
        scenes     = [self.scene_dataset[i] for i in sel]
        num_scenes = len(scenes)

        # ── Pass 1: collect all entity mentions per scene ────────────────
        per_scene_ents = []   # list of {name → type_str} per scene
        all_names      = {}   # name → type_str (first-seen type wins)
        scene_count    = {}   # name → number of scenes this entity appears in

        for scene in scenes:
            ents = self._scene_entities(scene)
            per_scene_ents.append(ents)
            for name, etype in ents.items():
                all_names.setdefault(name, etype)
                scene_count[name] = scene_count.get(name, 0) + 1

        # ── Pass 2: frequency-based filtering ────────────────────────────
        # Remove entities appearing in too many scenes (generic nouns)
        max_allowed = max(3, int(self._MAX_SCENE_FRAC * num_scenes))
        # NER entities (PERSON, ORG, GPE, FACILITY) bypass the upper filter
        # because a protagonist appearing in 80% of scenes is correct.
        _NER_TYPES = {"PERSON", "ORG", "GPE", "FACILITY"}
        keep = {}
        for name, etype in all_names.items():
            sc = scene_count[name]
            if sc < self._MIN_SCENES:
                continue
            if sc > max_allowed and etype not in _NER_TYPES:
                continue   # generic noun like "car" / "door"
            keep[name] = etype

        # ── Build entity registry: prioritize by scene count ─────────────
        # Entities appearing in more scenes are more important to the narrative.
        sorted_ents = sorted(keep.items(), key=lambda x: scene_count[x[0]], reverse=True)

        entity_to_idx = {}
        entity_types  = []
        for name, etype in sorted_ents:
            if len(entity_to_idx) >= N:
                break
            entity_to_idx[name] = len(entity_to_idx)
            entity_types.append(ENTITY_TYPE_MAP.get(etype, 4))

        n_valid = len(entity_to_idx)

        # ── Tensors ───────────────────────────────────────────────────────
        entity_type_ids = torch.zeros(N, dtype=torch.long)
        entity_type_ids[:n_valid] = torch.tensor(entity_types[:n_valid], dtype=torch.long)

        entity_mask = torch.zeros(N, dtype=torch.bool)
        entity_mask[:n_valid] = True

        # Incidence matrix B [N, S] with hierarchical role weights
        incidence = torch.zeros(N, S, dtype=torch.float)
        for s, scene in enumerate(scenes):
            speakers = [c.upper() for c in scene.get("characters", [])]
            triplets = scene.get("graph_triplets", [])

            for name in per_scene_ents[s]:
                if name not in entity_to_idx:
                    continue
                n_idx = entity_to_idx[name]
                name_upper = name.upper()
                name_lower = name.lower()

                if name_upper in speakers:
                    weight = 1.0   # active speaker
                elif any(trip.lower().startswith(name_lower + "_") for trip in triplets):
                    weight = 0.7   # SVO subject
                elif any(trip.lower().endswith("_" + name_lower) for trip in triplets):
                    weight = 0.5   # SVO object
                else:
                    weight = 0.3   # background mention

                incidence[n_idx, s] = weight

        # Hyperedge types [S] — legacy, always NEUTRAL (latent edges in model)
        edge_type_ids = torch.full((S,), 4, dtype=torch.long)
        for s, scene in enumerate(scenes):
            htype = scene.get("hyperedge_type", "NEUTRAL")
            edge_type_ids[s] = HYPEREDGE_TYPE_MAP.get(htype, 4)

        # Invert entity_to_idx to get ordered name list for visualization
        entity_names = [""] * N
        for name, eidx in entity_to_idx.items():
            entity_names[eidx] = name

        result = {
            "movie_name":       movie_name,
            "scenes":           scenes,
            "incidence_matrix": incidence,       # [N, S]
            "edge_type_ids":    edge_type_ids,   # [S]
            "entity_type_ids":  entity_type_ids, # [N]
            "entity_mask":      entity_mask,     # [N]
            "entity_names":     entity_names,    # List[str] length N, "" = padding
        }
        self._cache[movie_name] = result
        return result


# =============================================================================
# Collate function
# =============================================================================

def hypergraph_collate_fn(batch):
    """
    Packs a list of movie items into padded batch tensors.
    Handles variable scene counts (padded to max_scenes in batch).
    """
    max_scenes = max(len(item["scenes"]) for item in batch)
    seq_len    = len(batch[0]["scenes"][0]["input_ids"])
    B          = len(batch)
    N          = MAX_ENTITIES

    # Scene-level tensors
    input_ids   = torch.full((B, max_scenes, seq_len), 1, dtype=torch.long)
    target_ids  = torch.full((B, max_scenes, seq_len), 1, dtype=torch.long)
    action_mask = torch.zeros((B, max_scenes, seq_len), dtype=torch.bool)
    dial_mask   = torch.zeros((B, max_scenes, seq_len), dtype=torch.bool)
    ent_mask    = torch.zeros((B, max_scenes, seq_len), dtype=torch.bool)
    head_mask   = torch.zeros((B, max_scenes, seq_len), dtype=torch.bool)

    # Hypergraph tensors
    incidence_matrix = torch.zeros((B, N, max_scenes), dtype=torch.float)
    edge_type_ids    = torch.zeros((B, max_scenes), dtype=torch.long)
    entity_type_ids  = torch.zeros((B, N), dtype=torch.long)
    entity_mask_b    = torch.zeros((B, N), dtype=torch.bool)

    all_triplets = []

    for b, item in enumerate(batch):
        ns = len(item["scenes"])
        for s in range(max_scenes):
            if s < ns:
                sc = item["scenes"][s]
                SL = seq_len
                input_ids[b, s]    = sc["input_ids"][:SL]
                target_ids[b, s]   = sc["target_ids"][:SL]
                action_mask[b, s]  = sc["action_mask"][:SL].clone()
                dial_mask[b, s]    = sc["dialogue_mask"][:SL].clone()
                ent_mask[b, s]     = sc["entity_mask"][:SL].clone()
                head_mask[b, s]    = sc["header_mask"][:SL].clone()
                all_triplets.append(sc["graph_triplets"])
            else:
                all_triplets.append([])

        incidence_matrix[b, :, :ns] = item["incidence_matrix"][:, :ns]
        edge_type_ids[b, :ns]       = item["edge_type_ids"][:ns]
        entity_type_ids[b]          = item["entity_type_ids"]
        entity_mask_b[b]            = item["entity_mask"]

    return {
        "input_ids":        input_ids,
        "target_ids":       target_ids,
        "action_mask":      action_mask,
        "dial_mask":        dial_mask,
        "ent_mask":         ent_mask,
        "head_mask":        head_mask,
        "incidence_matrix": incidence_matrix,
        "edge_type_ids":    edge_type_ids,
        "entity_type_ids":  entity_type_ids,
        "entity_mask":      entity_mask_b,
        "triplets":         all_triplets,
        "movie_name":       [item["movie_name"] for item in batch],
        "entity_names":     [item.get("entity_names", []) for item in batch],
    }


# Keep old name as alias so existing inference scripts don't break.
movie_collate_fn = hypergraph_collate_fn


# =============================================================================
# Dataset splitting (gzip → plain .jsonl)
# =============================================================================

def split_dataset_by_movie(input_path, train_path, eval_path, num_train=1500):
    """
    Splits a .jsonl[.gz] file into uncompressed train/eval .jsonl files,
    grouped by movie. Uncompressed splits enable byte-offset seeking in
    SceneDataset (constant-memory O(1) access).
    """
    print(f"Splitting {os.path.basename(input_path)} "
          f"(train: {num_train} movies)...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    mid_re       = re.compile(r'"movie_id"\s*:\s*"([^"]+)"')
    train_movies = set()
    eval_movies  = set()
    open_fn      = gzip.open if input_path.endswith(".gz") else open

    with open_fn(input_path, "rt", encoding="utf-8") as inf, \
         open(train_path, "wt", encoding="utf-8") as tr_out, \
         open(eval_path,  "wt", encoding="utf-8") as ev_out:

        for line in tqdm(inf, desc="Splitting", unit="lines"):
            if not line.strip():
                continue
            m         = mid_re.search(line)
            raw_id    = m.group(1) if m else "unknown"
            base      = raw_id.split("_Scene_")[0] if "_Scene_" in raw_id else raw_id

            if base in train_movies:
                tr_out.write(line)
            elif base in eval_movies:
                ev_out.write(line)
            elif len(train_movies) < num_train:
                train_movies.add(base)
                tr_out.write(line)
            else:
                eval_movies.add(base)
                ev_out.write(line)

    print(f"Split done: {len(train_movies)} train / {len(eval_movies)} eval movies.")


# =============================================================================
# Beam-search generation
# =============================================================================

@torch.no_grad()
def generate_summary(model, aligned_memory, enc_attn_mask, tokenizer,
                     device, max_new_tokens=200, beam_size=4):
    """
    Beam search over the BART decoder using pre-computed aligned_memory.
    aligned_memory  : [1, S+N, D]
    enc_attn_mask   : [1, S+N]
    """
    B = aligned_memory.size(0)
    assert B == 1

    beams     = [(0.0, [tokenizer.bos_token_id or 0])]
    completed = []
    eos_id    = tokenizer.eos_token_id or 2
    pad_id    = tokenizer.pad_token_id or 1

    for _ in range(max_new_tokens):
        new_beams = []
        for score, tokens in beams:
            if tokens[-1] == eos_id:
                completed.append((score, tokens))
                continue
            t_ids  = torch.tensor([tokens], dtype=torch.long, device=device)
            t_mask = (t_ids != pad_id).long()
            dec_out = model.bart_decoder(
                input_ids=t_ids,
                attention_mask=t_mask,
                encoder_hidden_states=aligned_memory,
                encoder_attention_mask=enc_attn_mask,
            )
            logits    = model.head(dec_out.last_hidden_state[:, -1, :]).float()
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

            # No-repeat trigram penalty
            if len(tokens) >= 3:
                for k in range(len(tokens) - 2):
                    ng = tuple(tokens[k:k + 3])
                    log_probs[ng[-1]] = -1e4   # avoid -inf in bfloat16

            top_v, top_i = log_probs.topk(beam_size)
            for v, idx in zip(top_v.tolist(), top_i.tolist()):
                new_beams.append((score + v, tokens + [idx]))

        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
        if not beams:
            break

    pool      = completed if completed else beams
    best      = max(pool, key=lambda x: x[0] / max(len(x[1]), 1))
    return tokenizer.decode(best[1], skip_special_tokens=True)


# =============================================================================
# Training
# =============================================================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    torch.cuda.empty_cache()

    # ── 1. Dataset selection ─────────────────────────────────────────────────
    src_path = MOVIESUM_JSONL if ABLATION["dataset"] in ("moviesum", "both") else MENSA_JSONL

    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"Source dataset not found: {src_path}\n"
            f"Run: python emnlp_extractor.py --dataset "
            f"{'moviesum' if 'moviesum' in src_path else 'mensa'} --out {src_path}"
        )

    # Regenerate splits if they don't exist OR if the source file is newer than them
    # (catches the "stale splits from a previous tiny run" failure mode)
    src_mtime = os.path.getmtime(src_path)
    splits_stale = not (
        os.path.exists(TRAIN_SPLIT_PATH) and os.path.exists(EVAL_SPLIT_PATH)
        and os.path.getmtime(TRAIN_SPLIT_PATH) >= src_mtime
        and os.path.getmtime(EVAL_SPLIT_PATH)  >= src_mtime
    )
    if splits_stale:
        split_dataset_by_movie(src_path, TRAIN_SPLIT_PATH, EVAL_SPLIT_PATH,
                               num_train=NUM_TRAIN_MOVIES)

    EPOCHS = EPOCHS_STAGE1 + EPOCHS_STAGE2

    wandb.init(
        project="DualTower-Hypergraph",
        name=ABLATION["run_name"],
        config={
            "lr_new": LR_NEW_LAYERS, "lr_lora": LR_LORA,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "accum_steps": ACCUMULATION_STEPS,
            "max_seq_len": MAX_SEQ_LEN, "max_entities": MAX_ENTITIES,
            "architecture": "DualTowerHypergraphSummariser",
            **ABLATION,
        },
    )

    # ── 2. Datasets ───────────────────────────────────────────────────────────
    base_train = SceneDataset(TRAIN_SPLIT_PATH, max_seq_len=MAX_SEQ_LEN)
    base_eval  = SceneDataset(EVAL_SPLIT_PATH,  max_seq_len=MAX_SEQ_LEN)
    train_ds   = MovieHypergraphDataset(base_train, max_scenes=MAX_SCENES)
    eval_ds    = MovieHypergraphDataset(base_eval,  max_scenes=MAX_SCENES)

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                          num_workers=4, pin_memory=True,
                          persistent_workers=True,
                          collate_fn=hypergraph_collate_fn)
    eval_dl  = DataLoader(eval_ds,  batch_size=1, shuffle=False,
                          num_workers=4, pin_memory=True,
                          persistent_workers=True,
                          collate_fn=hypergraph_collate_fn)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    tokenizer = base_train.tokenizer
    _sum_module.BART_MODEL = ABLATION["bart_model"]

    model = DualTowerHypergraphSummariser(
        vocab_size=len(tokenizer),
        d_model=ABLATION["d_model"],
        num_layers=ABLATION["num_layers"],
        max_entities=MAX_ENTITIES,
        tokenizer=tokenizer,
        use_adaptive_streams=not ABLATION["no_adaptive_streams"],
        use_entity_names=not ABLATION["no_entity_names"],
        edge_dropout=ABLATION["edge_dropout"],
    ).to(device)
    print(f"Model: d_model={ABLATION['d_model']}  num_layers={ABLATION['num_layers']}  "
          f"adaptive_streams={not ABLATION['no_adaptive_streams']}  "
          f"entity_names={not ABLATION['no_entity_names']}  "
          f"edge_dropout={ABLATION['edge_dropout']}")

    # Ablation: disable hypergraph tower (text-only baseline)
    if ABLATION["no_hypergraph"]:
        import types
        def _noop_hyp(self, scene_reps, inc, etype, entype, emask):
            return scene_reps, torch.zeros(
                scene_reps.size(0), self.max_entities, self.d_model,
                device=scene_reps.device
            )
        model.hypergraph_tower.forward = types.MethodType(
            _noop_hyp, model.hypergraph_tower
        )
        print("ABLATION: hypergraph tower disabled (text-only baseline)")

    # Ablation: static hypergraph — freeze entity states after init
    # (DiscoGraMS-style: graph is built once, never updated)
    if ABLATION["static_hypergraph"]:
        for p in model.hypergraph_tower.entity_gru.parameters():
            p.requires_grad = False
        print("ABLATION: GRU frozen → static hypergraph (DiscoGraMS baseline)")

    # Ablation: disable GRU, use EMA instead
    if ABLATION["no_gru"] and not ABLATION["static_hypergraph"]:
        import types
        _alpha = 0.7
        def _ema_update(self, msg_flat, h_flat):
            return _alpha * h_flat + (1 - _alpha) * msg_flat
        model.hypergraph_tower.entity_gru.forward = types.MethodType(
            _ema_update, model.hypergraph_tower.entity_gru
        )
        print(f"ABLATION: GRU replaced with EMA (α={_alpha})")

    # Ablation: disable RAFT
    if ABLATION["no_raft"]:
        import types
        def _raft_noop(self, features, *masks):
            return features
        model.raft.forward = types.MethodType(_raft_noop, model.raft)
        print("ABLATION: RAFT disabled")

    # Ablation: disable pointer head
    if ABLATION["no_pointer_head"]:
        import types
        def _no_ptr(self, dec, mem, trips, tok, emb, dev):
            B, T, D = dec.shape
            return (torch.ones(B, T, 1, device=dev),
                    torch.zeros(B, T, self.vocab_size, device=dev))
        model.pointer_head.forward = types.MethodType(_no_ptr, model.pointer_head)
        print("ABLATION: pointer head disabled")

    # ── 4. Checkpoint detection ───────────────────────────────────────────────
    ckpt_latest = "/tmp/uday/checkpoints/dual_hyp_latest.pt"
    start_epoch = 0
    ckpt_state  = None
    if os.path.exists(ckpt_latest):
        print(f"Resuming from {ckpt_latest}...")
        ckpt_state  = torch.load(ckpt_latest, map_location=device, weights_only=True)
        start_epoch = ckpt_state.get("epoch", 0)

    # ── 5a. Xavier init for new (non-pretrained) layers ───────────────────────
    # Runs BEFORE checkpoint load so checkpoint values overwrite random init on
    # matching keys. New params not present in the checkpoint keep their Xavier init.
    _skip = {"roberta", "bart_decoder", "head", "scene_proj"}
    for name, param in model.named_parameters():
        if param.requires_grad and not any(s in name for s in _skip):
            if len(param.shape) > 1:
                torch.nn.init.xavier_uniform_(param)

    # ── 5b. Checkpoint load — MUST happen before LoRA wrapping ────────────────
    # After get_peft_model(), Mamba key names change:
    #   mamba_tower.layers.*  →  mamba_tower.base_model.model.layers.*
    # Loading the checkpoint after LoRA wrap causes all Mamba weights to be
    # silently skipped (key mismatch under strict=False) → random Mamba → NaN.
    if ckpt_state:
        cur = model.state_dict()
        ckpt_params = ckpt_state["model_state_dict"]
        mismatched = [k for k, v in ckpt_params.items()
                      if k in cur and v.shape != cur[k].shape]
        if mismatched:
            print(f"  Skipping {len(mismatched)} shape-mismatched param(s) from checkpoint "
                  f"(arch change): {mismatched}")
        compatible = {k: v for k, v in ckpt_params.items() if k not in mismatched}

        # Skip encoder_attn from checkpoint: model reinitializes these from scratch
        # via _reinit_encoder_attn() (Xavier init). Old checkpoints contain either
        # pretrained BART values (useless for our encoder) or NaN-corrupted zeros.
        old_enc_attn = [k for k in compatible if "encoder_attn" in k]
        if old_enc_attn:
            print(f"  Skipping {len(old_enc_attn)} encoder_attn param(s) from checkpoint "
                  f"(model reinitializes from scratch via Xavier init)")
            for k in old_enc_attn:
                del compatible[k]

        model.load_state_dict(compatible, strict=False)

    # ── 5c. Stage setup + LoRA — applied AFTER checkpoint load ────────────────
    lora_applied = False

    def apply_lora(r=16, alpha=32):
        nonlocal lora_applied
        lora_cfg = LoraConfig(
            r=r, lora_alpha=alpha,
            target_modules=["in_proj", "x_proj", "out_proj", "dt_proj"],
            lora_dropout=0.05, bias="none",
        )
        model.mamba_tower = get_peft_model(model.mamba_tower, lora_cfg)
        lora_applied = True
        model.enable_gradient_checkpointing()
        print(f"LoRA applied: r={r}, alpha={alpha}")

    def _set_stage1_grads():
        """Stage 1: freeze RoBERTa + Mamba + pos-embed + BART decoder (except encoder_attn) + lm_head.
        Trainable: hypergraph tower, scene_proj, RAFT, fusion, post_scan_adapter,
                   encoder_attn (Xavier-init), pointer_head.
        encoder_attn is freshly initialized (not pretrained) so trains safely at LR_NEW_LAYERS.
        post_scan_adapter is unblocked here so the encoder→decoder bridge learns from epoch 1.
        """
        for name, p in model.named_parameters():
            if any(s in name for s in ["roberta", "mamba_tower", "scene_pos_embed"]):
                p.requires_grad = False
            elif "bart_decoder" in name and "encoder_attn" not in name:
                p.requires_grad = False
            elif name.startswith("head."):
                p.requires_grad = False
            else:
                p.requires_grad = True

    def _set_stage2_grads():
        """Stage 2: freeze RoBERTa only; LoRA on Mamba; train everything else including adapter.
        encoder_attn (Xavier-init) continues training alongside all new layers."""
        for name, p in model.named_parameters():
            if "roberta" in name:
                p.requires_grad = False
            elif "bart_decoder" in name and "encoder_attn" not in name:
                p.requires_grad = False
            elif name.startswith("head."):
                p.requires_grad = False
            else:
                p.requires_grad = True  # covers post_scan_adapter, encoder_attn, hypergraph, raft, etc.

    if start_epoch >= EPOCHS_STAGE1:
        apply_lora(r=64, alpha=128)   # wraps correctly-loaded Mamba weights
        _set_stage2_grads()
    else:
        _set_stage1_grads()

    # Diagnostic: verify freeze policy
    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen_named    = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    trainable_M = sum(p.numel() for _, p in trainable_named) / 1e6
    frozen_M    = sum(p.numel() for _, p in frozen_named)    / 1e6
    print(f"  Trainable: {len(trainable_named)} tensors / {trainable_M:.1f}M params")
    print(f"  Frozen:    {len(frozen_named)} tensors / {frozen_M:.1f}M params")
    print(f"  Trainable groups: {sorted(set(n.split('.')[0] for n, _ in trainable_named))}")
    print(f"  Sample trainable: {sorted(n for n, _ in trainable_named)[:8]}")

    # ── 6. Loss ───────────────────────────────────────────────────────────────
    criterion = RelationalEventConsistencyLoss(
        alpha=0.1 if not ABLATION["no_contrastive_loss"] else 0.0,
        tokenizer=tokenizer,
        entity_penalty=ABLATION["entity_penalty"],
        label_smoothing=0.1,
        coherence_weight=0.05 if not ABLATION["no_coherence_loss"] else 0.0,
    )

    # ── 7. Optimiser ──────────────────────────────────────────────────────────
    def _make_optim():
        lora_p    = [p for n, p in model.named_parameters()
                     if p.requires_grad and "lora_" in n]
        adapter_p = [p for n, p in model.named_parameters()
                     if p.requires_grad and "post_scan_adapter" in n]
        other_p   = [p for n, p in model.named_parameters()
                     if p.requires_grad and "lora_" not in n
                     and "post_scan_adapter" not in n]
        return AdamW([
            {"params": other_p,    "lr": LR_NEW_LAYERS},
            {"params": lora_p,     "lr": LR_LORA},
            {"params": adapter_p,  "lr": LR_NEW_LAYERS},
        ], weight_decay=0.01)

    optimizer = _make_optim()
    # Updated for modern PyTorch to prevent deprecation spam
    # scaler    = torch.amp.GradScaler("cuda", enabled=False)
    wandb.watch(model, criterion, log="all", log_freq=50)

    total_steps  = (len(train_dl) // ACCUMULATION_STEPS) * EPOCHS
    warmup_steps = int(0.05 * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ── 8. Evaluation metrics ─────────────────────────────────────────────────
    rouge_metric     = hf_evaluate.load("rouge")
    bertscore_metric = hf_evaluate.load("bertscore")
    meteor_metric    = hf_evaluate.load("meteor")

    import spacy as _spacy
    _nlp_ner = _spacy.load("en_core_web_sm",
                           disable=["parser", "tagger", "lemmatizer"])

    def _entity_f1(preds, refs):
        tp = fp = fn = 0
        for pred, ref in zip(preds, refs):
            p_e = {e.text.lower() for e in _nlp_ner(pred).ents}
            r_e = {e.text.lower() for e in _nlp_ner(ref).ents}
            tp += len(p_e & r_e)
            fp += len(p_e - r_e)
            fn += len(r_e - p_e)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # ── 9. Training loop ──────────────────────────────────────────────────────
    global_step = 0

    for epoch in range(start_epoch, EPOCHS_STAGE1 + EPOCHS_STAGE2):

        # Curriculum: entity penalty and alpha anneal in stage 2
        if epoch >= EPOCHS_STAGE1:
            s2 = epoch - EPOCHS_STAGE1
            criterion.alpha         = 0.2 + 0.3 * (s2 / max(1, EPOCHS_STAGE2 - 1))
            criterion.entity_penalty = 3.0 + 4.0 * (s2 / max(1, EPOCHS_STAGE2 - 1))
        else:
            criterion.alpha         = 0.1
            criterion.entity_penalty = 3.0

        print(f"\n[Epoch {epoch+1}/{EPOCHS_STAGE1+EPOCHS_STAGE2}]  "
              f"α={criterion.alpha:.3f}  entity_pen={criterion.entity_penalty:.2f}")

        # Stage transition at epoch EPOCHS_STAGE1
        if epoch == EPOCHS_STAGE1 and not lora_applied:
            print("→ Stage 2: applying LoRA to Mamba tower")
            torch.cuda.empty_cache()   # release Stage-1 fragmentation before LoRA expands memory
            apply_lora(r=16, alpha=32)
            _set_stage2_grads()
            optimizer = _make_optim()
            # scaler    = torch.amp.GradScaler("cuda", enabled=False)
            s2_total  = (len(train_dl) // ACCUMULATION_STEPS) * EPOCHS_STAGE2
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=int(0.05 * s2_total),
                num_training_steps=s2_total
            )

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss_sum  = 0.0
        n_valid_batches = 0  # count only non-NaN batches for correct loss average
        optimizer.zero_grad()


        bar = tqdm(train_dl, desc=f"E{epoch+1} Train", unit="batch")
        for bi, batch in enumerate(bar):
            inp  = batch["input_ids"].to(device)
            amsk = batch["action_mask"].to(device)
            dmsk = batch["dial_mask"].to(device)
            emsk = batch["ent_mask"].to(device)
            hmsk = batch["head_mask"].to(device)
            inc  = batch["incidence_matrix"].to(device)
            etid = batch["edge_type_ids"].to(device)
            enid = batch["entity_type_ids"].to(device)
            emk  = batch["entity_mask"].to(device)
            tgt  = batch["target_ids"].to(device)
            trip = batch["triplets"]
            enames = batch.get("entity_names")

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                log_pr, H_text_4d, labels, _, H_hyp = model(
                    inp, amsk, dmsk, emsk, hmsk,
                    inc, etid, enid, emk,
                    target_ids=tgt, triplets=trip, entity_names=enames,
                )
                log_pr = log_pr.float()
                loss   = criterion(
                    log_probs=log_pr.view(-1, log_pr.size(-1)),
                    targets=labels.view(-1),
                    triplets=trip,
                    hidden_states=H_text_4d,
                    head_weight=model.head.weight,
                    incidence_matrix=inc,
                )
                loss = loss / ACCUMULATION_STEPS

            val = loss.item() * ACCUMULATION_STEPS
            if math.isnan(val) or math.isinf(val):
                print(f"  ⚠ NaN/Inf at batch {bi}, skipping")
                # Emergency: if loss is NaN, model weights may be corrupted.
                # Scan and recover immediately — don't wait for the gradient step.
                nan_w = [(n, p) for n, p in model.named_parameters()
                         if not torch.isfinite(p.data).all()]
                if nan_w:
                    print(f"    → {len(nan_w)} weight tensor(s) contain NaN — recovering: "
                          f"{[n for n, _ in nan_w[:3]]}")
                    for _, p in nan_w:
                        p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)
                    for _, p in nan_w:
                        if p in optimizer.state:
                            del optimizer.state[p]
                optimizer.zero_grad()
                continue

            loss.backward()

            if (bi + 1) % ACCUMULATION_STEPS == 0 or (bi + 1) == len(train_dl):
                # --- THE GRADIENT SHIELD ---
                has_nan_grad = False
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    print(f"  ⚠ NaN gradient at batch {bi}, discarding step")
                    optimizer.zero_grad()
                else:
                    # Cap individual gradient elements before computing global norm.
                    # Without this, large-but-finite elements (e.g. 1e20 from encoder_attn
                    # distribution mismatch) cause sum-of-squares overflow in clip_grad_norm_,
                    # which poisons ALL gradients with NaN via Inf*0.
                    nn_utils.clip_grad_value_(model.parameters(), 1.0)
                    nn_utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # ── Weight NaN guard ─────────────────────────────────────
                    # Recover from rare NaN weights (e.g. BF16 precision loss in
                    # Stage 2). Uses nan_to_num + per-param Adam state reset.
                    # Unlike the old version, this does NOT trigger an infinite
                    # loop because encoder_attn now has its own 1e-6 LR group
                    # and clip_grad_value_ prevents gradient overflow.
                    nan_params = [(n, p) for n, p in model.named_parameters()
                                  if p.requires_grad and not torch.isfinite(p.data).all()]
                    if nan_params:
                        names = [n for n, _ in nan_params[:3]]
                        print(f"  ☠ NaN in {len(nan_params)} param(s) at step {global_step}: {names}")
                        for _, p in nan_params:
                            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)
                        # Only clear Adam state for affected params (not all).
                        for _, p in nan_params:
                            if p in optimizer.state:
                                del optimizer.state[p]
                    # ──────────────────────────────────────────────────────────

                # Log entity state norms every 500 steps (extra forward pass — avoid every 100)
                if global_step % 500 == 0:
                    with torch.no_grad():
                        _, H_t4d = model(
                            inp, amsk, dmsk, emsk, hmsk,
                            inc, etid, enid, emk,
                            target_ids=None, triplets=None, entity_names=enames,
                        )
                        H_t = H_t4d.mean(dim=2)
                        _, H_nodes = model.hypergraph_tower(H_t, inc, etid, enid, emk)
                        log_entity_state_norms(H_nodes, emk, global_step)

            train_loss_sum  += val
            n_valid_batches += 1
            lr_now = optimizer.param_groups[0]["lr"]
            bar.set_postfix(loss=f"{val:.4f}", lr=f"{lr_now:.2e}")
            wandb.log({
                "train/batch_loss": val,
                "train/lr":         lr_now,
                "train/alpha":      criterion.alpha,
                "epoch":            epoch + 1,
            })

        avg_train = train_loss_sum / max(n_valid_batches, 1)

        # ── Eval ──────────────────────────────────────────────────────────────
        model.eval()
        eval_loss_sum = 0.0
        all_preds, all_refs = [], []
        # Hypergraph quality accumulators
        hg_entities_per_scene = []   # avg non-zero roles per scene
        hg_scenes_per_entity  = []   # avg scenes each entity appears in
        hg_entity_coverage    = []   # fraction of ref entities found in hypergraph

        bar_e = tqdm(eval_dl, desc=f"E{epoch+1} Eval", unit="batch")
        with torch.no_grad():
            for bi, batch in enumerate(bar_e):
                inp  = batch["input_ids"].to(device)
                amsk = batch["action_mask"].to(device)
                dmsk = batch["dial_mask"].to(device)
                emsk = batch["ent_mask"].to(device)
                hmsk = batch["head_mask"].to(device)
                inc  = batch["incidence_matrix"].to(device)
                etid = batch["edge_type_ids"].to(device)
                enid = batch["entity_type_ids"].to(device)
                emk  = batch["entity_mask"].to(device)
                tgt  = batch["target_ids"].to(device)
                trip = batch["triplets"]
                enames = batch.get("entity_names")

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    log_pr, H_text_4d, labels, _, H_hyp = model(
                        inp, amsk, dmsk, emsk, hmsk,
                        inc, etid, enid, emk,
                        target_ids=tgt, triplets=trip, entity_names=enames,
                    )
                    log_pr = log_pr.float()
                    eloss  = criterion(
                        log_probs=log_pr.view(-1, log_pr.size(-1)),
                        targets=labels.view(-1),
                        triplets=trip,
                        hidden_states=H_text_4d,
                        head_weight=model.head.weight,
                        incidence_matrix=inc,
                    )

                # --- NEW EVAL SAFETY CHECK ---
                val = eloss.item()
                if math.isnan(val) or math.isinf(val):
                    continue # Skip corrupted eval movies
                
                eval_loss_sum += val
                bar_e.set_postfix(eval_loss=f"{val:.4f}")
                # -----------------------------

                # ── Hypergraph quality stats (every batch, cheap) ──────────
                inc_b = inc[0].cpu().float()   # [N, S]
                active = (inc_b > 0).float()
                ents_per_scene = active.sum(dim=0)   # entities per scene
                scenes_per_ent = active.sum(dim=1)   # scenes per entity
                hg_entities_per_scene.append(ents_per_scene[ents_per_scene > 0].mean().item()
                                             if ents_per_scene.sum() > 0 else 0.0)
                hg_scenes_per_entity.append(scenes_per_ent[scenes_per_ent > 0].mean().item()
                                            if scenes_per_ent.sum() > 0 else 0.0)

                # Entity coverage: do hypergraph entities appear in the reference?
                if enames and enames[0]:
                    ref_text = tokenizer.decode(tgt[0, 0].cpu().tolist(), skip_special_tokens=True).lower()
                    hg_names = [n.lower() for n in enames[0] if n]
                    if hg_names:
                        found = sum(1 for n in hg_names if n in ref_text)
                        hg_entity_coverage.append(found / len(hg_names))

                # Fusion gate: 1.0 = all text tower, 0.0 = all hypergraph tower
                if hasattr(model, "_last_gate_mean"):
                    wandb.log({"eval/fusion_gate_mean": model._last_gate_mean,
                               "epoch": epoch + 1})

                # Beam-search generation every 10th batch (~30 samples for stable metrics)
                if bi % 10 == 0:
                    aligned_mem, _ = model(
                        inp, amsk, dmsk, emsk, hmsk,
                        inc, etid, enid, emk,
                        target_ids=None, triplets=None, entity_names=enames,
                    )
                    # enc_attn_mask: scenes + entity nodes
                    mem_pad = torch.zeros(
                        aligned_mem.size(0), aligned_mem.size(1),
                        dtype=torch.bool, device=device
                    )
                    mem_pad[:, :inp.size(1)] = (inp[:, :, 0] == 1)
                    mem_pad[:, inp.size(1):] = ~emk
                    all_masked = mem_pad.all(dim=1)
                    mem_pad[all_masked, 0] = False
                    enc_attn = (~mem_pad).long()

                    # Wrap the generation in the same bfloat16 context
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = generate_summary(
                            model, aligned_mem, enc_attn, tokenizer,
                            device, max_new_tokens=200, beam_size=4,
                        )
                    ref  = tokenizer.decode(
                        tgt[0, 0].cpu().tolist(), skip_special_tokens=True
                    )
                    all_preds.append(pred)
                    all_refs.append(ref)
                    wandb.log({
                        "eval/sample": wandb.Html(
                            f"<b>Pred:</b> {pred}<br><b>Ref:</b> {ref}"
                        ),
                        "epoch": epoch + 1,
                    })

                    # Hyperedge similarity heatmap every 50th batch
                    if bi % 50 == 0:
                        _mname = batch["movie_name"][0] if batch.get("movie_name") else ""
                        log_hyperedge_attention(model, H_hyp, inc, _mname)

                    # Full 4-panel hypergraph visualization once per epoch (first batch)
                    if bi == 0:
                        _mname = batch["movie_name"][0] if batch.get("movie_name") else ""
                        _enames = batch["entity_names"][0] if batch.get("entity_names") else []
                        _save = f"/tmp/uday/hypergraph_epoch{epoch+1}.png"
                        log_hypergraph_to_wandb(
                            inc[0].cpu(), _enames,
                            entity_type_ids=enid[0].cpu(),
                            entity_mask=emk[0].cpu(),
                            movie_name=_mname,
                            step=epoch + 1,
                            save_path=_save,
                        )

        avg_eval = eval_loss_sum / max(len(eval_dl), 1)

        # ── Metrics ───────────────────────────────────────────────────────────
        r1 = r2 = rL = bs_f1 = met = ent_f1 = 0.0
        if all_preds:
            rs   = rouge_metric.compute(predictions=all_preds, references=all_refs,
                                        use_stemmer=True)
            r1, r2, rL = rs["rouge1"], rs["rouge2"], rs["rougeL"]

            # bs_out = bertscore_metric.compute(
            #     predictions=all_preds, references=all_refs,
            #     lang="en", model_type="roberta-large",
            # )
            # bs_f1  = sum(bs_out["f1"]) / max(len(bs_out["f1"]), 1)

            met    = meteor_metric.compute(
                predictions=all_preds, references=all_refs
            )["meteor"]

            ent_f1 = _entity_f1(all_preds, all_refs)

        print(f"Epoch {epoch+1} | "
              f"Train {avg_train:.4f} | Eval {avg_eval:.4f} | "
              f"R1 {r1:.4f} | R2 {r2:.4f} | RL {rL:.4f} | "
              f"BS {bs_f1:.4f} | METEOR {met:.4f} | EntF1 {ent_f1:.4f}")

        # Hypergraph quality summary
        avg_ents_per_scene = sum(hg_entities_per_scene) / max(len(hg_entities_per_scene), 1)
        avg_scenes_per_ent = sum(hg_scenes_per_entity)  / max(len(hg_scenes_per_entity), 1)
        avg_ent_coverage   = sum(hg_entity_coverage)     / max(len(hg_entity_coverage), 1)
        print(f"  Hypergraph: {avg_ents_per_scene:.1f} ents/scene, "
              f"{avg_scenes_per_ent:.1f} scenes/ent, "
              f"entity_coverage={avg_ent_coverage:.3f}")

        wandb.log({
            "epoch/train_loss": avg_train, "epoch/eval_loss": avg_eval,
            "epoch/rouge1": r1, "epoch/rouge2": r2, "epoch/rougeL": rL,
            "epoch/bertscore_f1": bs_f1, "epoch/meteor": met,
            "epoch/entity_f1": ent_f1,
            "epoch/hg_ents_per_scene": avg_ents_per_scene,
            "epoch/hg_scenes_per_entity": avg_scenes_per_ent,
            "epoch/hg_entity_coverage": avg_ent_coverage,
            "epoch": epoch + 1,
        })

        # ── Checkpoint ────────────────────────────────────────────────────────
        save_dir = "/tmp/uday/checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        ckpt_ep = f"{save_dir}/dual_hyp_epoch_{epoch+1}.pt"
        torch.save({
            "epoch":             epoch + 1,
            "model_state_dict":  model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train, "eval_loss": avg_eval,
            "rouge1": r1, "rouge2": r2, "rougeL": rL,
            "bertscore_f1": bs_f1,
        }, ckpt_ep)
        torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict()},
                   ckpt_latest)
        print(f"  Checkpoint saved → {ckpt_ep}")

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    train()
"""
train.py  —  LED + Mamba-Hypergraph Training Pipeline (v4)
============================================================
Key changes from v3:
  - LED encoder replaces RoBERTa + Mamba text tower
  - Mamba-based entity state tracking replaces GRU in hypergraph tower
  - LED decoder replaces BART decoder (no distribution mismatch)
  - Input format: full screenplay as single sequence with <scene> separators
  - Scene boundaries tracked for pooling LED output → scene-level reps
  - dt value logging for narrative turning point interpretability
"""

import torch
import json
import os
import re
import math
import gzip
import sys
import types
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
    LEDMambaHypergraphSummariser,
    RelationalEventConsistencyLoss,
    log_hyperedge_attention,
    log_entity_state_norms,
    log_entity_dt_heatmap,
    ENTITY_TYPE_MAP,
    HYPEREDGE_TYPE_MAP,
    NUM_HYPEREDGE_TYPES,
    MAX_ENTITIES,
    NUM_ENTITY_TYPES,
    LED_MODEL,
)
from visualize_graph import log_hypergraph_to_wandb

torch.set_float32_matmul_precision("high")

# =============================================================================
# Argument parsing + ablation config
# =============================================================================
import argparse as _ap
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--run_name",           type=str,   default="full_model_v4")
_p.add_argument("--led_model",          type=str,
                default=("/tmp/uday/led-large-16384"
                         if os.path.isdir("/tmp/uday/led-large-16384")
                         else "allenai/led-large-16384"))
_p.add_argument("--d_model",            type=int,   default=1024)
_p.add_argument("--mamba_layers",       type=int,   default=2,
                help="Number of Mamba layers for entity temporal dynamics")
# Ablation flags
_p.add_argument("--no_hypergraph",      action="store_true",
                help="Disable hypergraph tower entirely (LED-only baseline)")
_p.add_argument("--static_hypergraph",  action="store_true",
                help="Freeze entity states (static graph baseline)")
_p.add_argument("--no_mamba_entity",    action="store_true",
                help="Replace Mamba with GRU for entity updates (ablation)")
_p.add_argument("--no_coherence_loss",  action="store_true")
_p.add_argument("--no_contrastive_loss",action="store_true")
_p.add_argument("--entity_penalty",     type=float, default=3.0)
_p.add_argument("--dataset",            type=str,   default="moviesum",
                choices=["moviesum", "mensa", "both"])
_p.add_argument("--no_adaptive_streams", action="store_true")
_p.add_argument("--no_entity_names",    action="store_true")
_p.add_argument("--edge_dropout",       type=float, default=0.1)
_args, _ = _p.parse_known_args()

ABLATION = {
    "run_name":             _args.run_name,
    "led_model":            _args.led_model,
    "d_model":              _args.d_model,
    "mamba_layers":         _args.mamba_layers,
    "no_hypergraph":        _args.no_hypergraph,
    "static_hypergraph":    _args.static_hypergraph,
    "no_mamba_entity":      _args.no_mamba_entity,
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
NUM_TRAIN_MOVIES = 1500

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE         = 1
ACCUMULATION_STEPS = 16
EPOCHS_STAGE1      = 3    # LED encoder frozen; train hypergraph + decoder
EPOCHS_STAGE2      = 20   # Unfreeze LED encoder global attn + LoRA entity Mamba
LR_NEW_LAYERS      = 1e-4
LR_DECODER         = 2e-5  # lower LR for pretrained LED decoder
LR_LORA            = 1e-5
MAX_INPUT_TOKENS   = 16384  # LED max input
MAX_TARGET_TOKENS  = 256   # summaries avg ~120 words; 512 wasted decoder memory
MAX_SCENES         = 64


# =============================================================================
# Scene-level dataset (lazy byte-offset reader)
# =============================================================================

class SceneDataset(Dataset):
    """Reads uncompressed .jsonl scene-by-scene with byte-offset seeking."""

    def __init__(self, jsonl_path, max_seq_len=256):
        if jsonl_path.endswith(".gz"):
            raise ValueError(
                f"SceneDataset requires uncompressed .jsonl, got: {jsonl_path}\n"
                "Run split_dataset_by_movie() first."
            )
        self.max_seq_len = max_seq_len
        self.jsonl_path  = jsonl_path
        # BART tokenizer for decoding stored input_ids (legacy data format)
        _bart_name = ("/tmp/uday/bart-large" if os.path.isdir("/tmp/uday/bart-large")
                      else "facebook/bart-large")
        self.tokenizer = AutoTokenizer.from_pretrained(_bart_name)
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
        return item


# =============================================================================
# Movie-level dataset with LED tokenization
# =============================================================================

class MovieHypergraphDataset(Dataset):
    """
    Groups scenes by movie. For each movie:
    1. Concatenates all scene texts with <scene> separator tokens
    2. Tokenizes the full screenplay for LED (up to 16384 tokens)
    3. Records scene boundary positions for pooling
    4. Builds hypergraph incidence matrix from entity mentions
    """

    _STOP = frozenset({
        "him", "her", "he", "she", "they", "them", "it", "i",
        "we", "you", "who", "what", "that", "this", "me", "my",
        "his", "its", "our", "your", "their", "the", "an",
    })
    _MAX_SCENE_FRAC = 0.50
    _MIN_SCENES     = 1

    def __init__(self, scene_dataset, tokenizer, max_scenes=MAX_SCENES,
                 max_entities=MAX_ENTITIES, max_input_tokens=MAX_INPUT_TOKENS,
                 max_target_tokens=MAX_TARGET_TOKENS):
        self.scene_dataset     = scene_dataset
        self.tokenizer         = tokenizer
        self.max_scenes        = max_scenes
        self.max_entities      = max_entities
        self.max_input_tokens  = max_input_tokens
        self.max_target_tokens = max_target_tokens
        self._cache            = {}

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

    def _scene_entities(self, scene):
        # Coreference mapping: mention → canonical name (from extractor)
        coref = scene.get("coref_entities", {})

        def _canon(name):
            """Resolve a mention to its canonical name via coreference."""
            return coref.get(name, name)

        ents = {}
        for e in scene.get("ner_entities", []):
            n = e.get("text", "").strip().lower()
            t = e.get("type", "OTHER")
            if n and len(n) > 1 and n not in self._STOP:
                ents[_canon(n)] = t
        for char in scene.get("characters", []):
            n = char.strip().lower()
            if n and len(n) > 1 and n not in self._STOP:
                ents.setdefault(_canon(n), "PERSON")
        for trip in scene.get("graph_triplets", []):
            parts = trip.split("_")
            for field_idx in (0, 2):
                if len(parts) > field_idx:
                    raw = parts[field_idx].replace("NOT ", "").strip()
                    if (raw and raw[0].isupper() and raw.isalpha()
                            and len(raw) > 2 and raw.lower() not in self._STOP):
                        ents.setdefault(_canon(raw.lower()), "PERSON")
        return ents

    def _get_scene_text(self, scene):
        """Reconstruct clean text from a scene record."""
        # The extractor stores input_ids tokenized by BART. For LED, we need
        # the raw text. We decode from the stored input_ids as a fallback,
        # but prefer the original text if available.
        if "clean_text" in scene:
            return scene["clean_text"]
        # Decode from input_ids (BART tokenizer) — imperfect but works
        bart_tok = self.scene_dataset.tokenizer
        ids = scene.get("input_ids", [])
        if isinstance(ids, list):
            return bart_tok.decode(ids, skip_special_tokens=True)
        return ""

    def __getitem__(self, idx):
        movie_name = self.movie_names[idx]
        if movie_name in self._cache:
            return self._cache[movie_name]

        all_indices = self.movie_map[movie_name]
        S = self.max_scenes
        N = self.max_entities

        # Stride-sample if movie > max_scenes
        if len(all_indices) > S:
            step = len(all_indices) / S
            sel  = [all_indices[int(i * step)] for i in range(S)]
        else:
            sel = all_indices
        scenes     = [self.scene_dataset[i] for i in sel]
        num_scenes = len(scenes)

        # ── Build concatenated screenplay text ──────────────────────────
        scene_texts = []
        for scene in scenes:
            text = self._get_scene_text(scene)
            if text.strip():
                scene_texts.append(text.strip())
            else:
                scene_texts.append("[empty scene]")

        # Join with special separator
        sep = " </s> "  # LED uses </s> as separator
        full_text = sep.join(scene_texts)

        # Tokenize for LED
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_input_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)       # [T]
        attention_mask  = encoding["attention_mask"].squeeze(0)  # [T]

        # Find scene boundaries by locating </s> separator tokens
        sep_id = self.tokenizer.convert_tokens_to_ids("</s>")
        sep_positions = (input_ids == sep_id).nonzero(as_tuple=True)[0].tolist()

        # Build scene boundaries [S, 2] = (start_tok, end_tok)
        scene_boundaries = torch.zeros(S, 2, dtype=torch.long)
        # First scene starts after BOS (position 0 is BOS for LED)
        prev_end = 1  # skip BOS
        for s in range(min(num_scenes, S)):
            start = prev_end
            if s < len(sep_positions):
                end = sep_positions[s]
            else:
                # Last scene: goes to end of real tokens
                end = attention_mask.sum().item()
            scene_boundaries[s] = torch.tensor([start, end])
            prev_end = end + 1  # skip the separator token

        # Global attention mask: first token + scene separator tokens
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[0] = 1  # CLS/BOS token
        for pos in sep_positions:
            if pos < self.max_input_tokens:
                global_attention_mask[pos] = 1

        # ── Target (summary) ────────────────────────────────────────────
        # Prefer raw summary text (from v4 extractor); fall back to BART decode
        summary_str = scenes[0].get("summary_text", "")
        if not summary_str:
            summary_raw = scenes[0].get("target_ids", [])
            bart_tok = self.scene_dataset.tokenizer
            if isinstance(summary_raw, list):
                summary_str = bart_tok.decode(summary_raw, skip_special_tokens=True)
            else:
                summary_str = str(summary_raw)

        target_enc = self.tokenizer(
            summary_str,
            max_length=self.max_target_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target_ids = target_enc["input_ids"].squeeze(0)  # [T_tgt]

        # ── Entity extraction + incidence matrix ────────────────────────
        per_scene_ents = []
        all_names      = {}
        scene_count    = {}
        for scene in scenes:
            ents = self._scene_entities(scene)
            per_scene_ents.append(ents)
            for name, etype in ents.items():
                all_names.setdefault(name, etype)
                scene_count[name] = scene_count.get(name, 0) + 1

        max_allowed = max(3, int(self._MAX_SCENE_FRAC * num_scenes))
        _NER_TYPES = {"PERSON", "ORG", "GPE", "FACILITY"}
        keep = {}
        for name, etype in all_names.items():
            sc = scene_count[name]
            if sc < self._MIN_SCENES:
                continue
            if sc > max_allowed and etype not in _NER_TYPES:
                continue
            keep[name] = etype

        sorted_ents = sorted(keep.items(), key=lambda x: scene_count[x[0]], reverse=True)
        entity_to_idx = {}
        entity_types  = []
        for name, etype in sorted_ents:
            if len(entity_to_idx) >= N:
                break
            entity_to_idx[name] = len(entity_to_idx)
            entity_types.append(ENTITY_TYPE_MAP.get(etype, 4))
        n_valid = len(entity_to_idx)

        entity_type_ids = torch.zeros(N, dtype=torch.long)
        entity_type_ids[:n_valid] = torch.tensor(entity_types[:n_valid], dtype=torch.long)
        entity_mask = torch.zeros(N, dtype=torch.bool)
        entity_mask[:n_valid] = True

        incidence = torch.zeros(N, S, dtype=torch.float)
        for s, scene in enumerate(scenes):
            if s >= S:
                break
            speakers = [c.upper() for c in scene.get("characters", [])]
            triplets = scene.get("graph_triplets", [])
            for name in per_scene_ents[s]:
                if name not in entity_to_idx:
                    continue
                n_idx = entity_to_idx[name]
                name_upper = name.upper()
                name_lower = name.lower()
                if name_upper in speakers:
                    weight = 1.0
                elif any(trip.lower().startswith(name_lower + "_") for trip in triplets):
                    weight = 0.7
                elif any(trip.lower().endswith("_" + name_lower) for trip in triplets):
                    weight = 0.5
                else:
                    weight = 0.3
                incidence[n_idx, s] = weight

        # ── Edge type classification from triplet verbs ─────────────────
        # Classify dominant verb type per scene into 5 event types
        _CONFLICT_VERBS  = frozenset({"attack","fight","shoot","kill","threaten","hit",
                                       "argue","confront","oppose","defeat","destroy",
                                       "chase","stab","punch","beat","clash","struggle"})
        _ALLIANCE_VERBS  = frozenset({"help","support","protect","join","ally","trust",
                                       "save","cooperate","agree","assist","give","share",
                                       "embrace","comfort","unite","recruit","befriend"})
        _DECEPTION_VERBS = frozenset({"lie","betray","trick","deceive","manipulate",
                                       "hide","pretend","mislead","fake","conceal","deny",
                                       "steal","spy","scheme","plot","bribe","forge"})
        _DIALOGUE_VERBS  = frozenset({"say","ask","tell","speak","talk","reply","answer",
                                       "shout","whisper","explain","warn","promise","call",
                                       "inform","request","deny","question","confess"})

        def _classify_verb(verb: str) -> int:
            v = verb.lower()
            if v in _CONFLICT_VERBS:  return HYPEREDGE_TYPE_MAP["CONFLICT"]
            if v in _ALLIANCE_VERBS:  return HYPEREDGE_TYPE_MAP["ALLIANCE"]
            if v in _DECEPTION_VERBS: return HYPEREDGE_TYPE_MAP["DECEPTION"]
            if v in _DIALOGUE_VERBS:  return HYPEREDGE_TYPE_MAP["DIALOGUE"]
            return HYPEREDGE_TYPE_MAP["NEUTRAL"]

        edge_type_ids = torch.full((S,), HYPEREDGE_TYPE_MAP["NEUTRAL"], dtype=torch.long)
        for s, scene in enumerate(scenes[:S]):
            type_votes = [0] * NUM_HYPEREDGE_TYPES
            for trip in scene.get("graph_triplets", []):
                parts = trip.split("_")
                if len(parts) >= 3:
                    verb = parts[1]
                    type_votes[_classify_verb(verb)] += 1
            if any(v > 0 for v in type_votes):
                edge_type_ids[s] = type_votes.index(max(type_votes))

        # ── Emotion matrix [N, S] ────────────────────────────────────────
        emotion_matrix = torch.zeros(N, S, dtype=torch.float)
        for s, scene in enumerate(scenes[:S]):
            coref = scene.get("coref_entities", {})
            char_emotions = scene.get("character_emotions", {})
            for char_name, polarity in char_emotions.items():
                # Resolve mention to canonical name, then to entity index
                canonical = coref.get(char_name, char_name)
                if canonical.lower() in entity_to_idx:
                    n_idx = entity_to_idx[canonical.lower()]
                    emotion_matrix[n_idx, s] = float(polarity)
                elif char_name.lower() in entity_to_idx:
                    n_idx = entity_to_idx[char_name.lower()]
                    emotion_matrix[n_idx, s] = float(polarity)

        entity_names = [""] * N
        for name, eidx in entity_to_idx.items():
            entity_names[eidx] = name

        # Collect all triplets per scene (used in loss, not model)
        all_triplets = []
        for scene in scenes[:S]:
            all_triplets.append(scene.get("graph_triplets", []))
        while len(all_triplets) < S:
            all_triplets.append([])

        result = {
            "movie_name":            movie_name,
            "input_ids":             input_ids,              # [T]
            "attention_mask":        attention_mask,          # [T]
            "global_attention_mask": global_attention_mask,   # [T]
            "scene_boundaries":      scene_boundaries,       # [S, 2]
            "target_ids":            target_ids,             # [T_tgt]
            "incidence_matrix":      incidence,              # [N, S]
            "edge_type_ids":         edge_type_ids,          # [S]
            "entity_type_ids":       entity_type_ids,        # [N]
            "entity_mask":           entity_mask,            # [N]
            "emotion_matrix":        emotion_matrix,         # [N, S]
            "entity_names":          entity_names,           # List[str]
            "triplets":              all_triplets,           # List[List[str]]
            "num_scenes":            num_scenes,
        }
        self._cache[movie_name] = result
        return result


# =============================================================================
# Collate function
# =============================================================================

def hypergraph_collate_fn(batch):
    """Packs movie items into batch tensors. B=1 expected."""
    B = len(batch)
    N = MAX_ENTITIES

    # Find max dimensions
    max_T     = max(item["input_ids"].size(0) for item in batch)
    max_T_tgt = max(item["target_ids"].size(0) for item in batch)
    max_S     = max(item["scene_boundaries"].size(0) for item in batch)

    input_ids      = torch.ones(B, max_T, dtype=torch.long)      # pad with 1
    attention_mask  = torch.zeros(B, max_T, dtype=torch.long)
    global_attn     = torch.zeros(B, max_T, dtype=torch.long)
    scene_bounds    = torch.zeros(B, max_S, 2, dtype=torch.long)
    target_ids      = torch.ones(B, max_T_tgt, dtype=torch.long)
    incidence       = torch.zeros(B, N, max_S, dtype=torch.float)
    edge_type_ids   = torch.zeros(B, max_S, dtype=torch.long)
    entity_type_ids = torch.zeros(B, N, dtype=torch.long)
    entity_mask_b   = torch.zeros(B, N, dtype=torch.bool)
    emotion_matrix  = torch.zeros(B, N, max_S, dtype=torch.float)

    all_triplets  = []
    all_names     = []
    all_movie     = []
    num_scenes_b  = []

    for b, item in enumerate(batch):
        T   = item["input_ids"].size(0)
        Tt  = item["target_ids"].size(0)
        Sb  = item["scene_boundaries"].size(0)
        ns  = item["num_scenes"]

        input_ids[b, :T]          = item["input_ids"]
        attention_mask[b, :T]     = item["attention_mask"]
        global_attn[b, :T]       = item["global_attention_mask"]
        scene_bounds[b, :Sb]     = item["scene_boundaries"]
        target_ids[b, :Tt]       = item["target_ids"]
        incidence[b, :, :Sb]     = item["incidence_matrix"][:, :Sb]
        edge_type_ids[b, :Sb]    = item["edge_type_ids"][:Sb]
        entity_type_ids[b]       = item["entity_type_ids"]
        entity_mask_b[b]         = item["entity_mask"]
        if "emotion_matrix" in item:
            emotion_matrix[b, :, :Sb] = item["emotion_matrix"][:, :Sb]
        all_triplets.append(item["triplets"])
        all_names.append(item["entity_names"])
        all_movie.append(item["movie_name"])
        num_scenes_b.append(ns)

    return {
        "input_ids":             input_ids,
        "attention_mask":        attention_mask,
        "global_attention_mask": global_attn,
        "scene_boundaries":      scene_bounds,
        "target_ids":            target_ids,
        "incidence_matrix":      incidence,
        "edge_type_ids":         edge_type_ids,
        "entity_type_ids":       entity_type_ids,
        "entity_mask":           entity_mask_b,
        "emotion_matrix":        emotion_matrix,
        "triplets":              all_triplets,
        "entity_names":          all_names,
        "movie_name":            all_movie,
        "num_scenes":            num_scenes_b,
    }


movie_collate_fn = hypergraph_collate_fn


# =============================================================================
# Dataset splitting (gzip → plain .jsonl)
# =============================================================================

def split_dataset_by_movie(input_path, train_path, eval_path, num_train=1500):
    print(f"Splitting {os.path.basename(input_path)} (train: {num_train} movies)...")
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
            m    = mid_re.search(line)
            raw  = m.group(1) if m else "unknown"
            base = raw.split("_Scene_")[0] if "_Scene_" in raw else raw
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
    Beam search over the LED decoder using pre-computed aligned_memory.
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
            dec_out = model.led_decoder(
                input_ids=t_ids,
                attention_mask=t_mask,
                encoder_hidden_states=aligned_memory,
                encoder_attention_mask=enc_attn_mask,
            )
            last_hidden = dec_out.last_hidden_state
            logits      = model.head(last_hidden[:, -1, :]).float()
            log_probs   = F.log_softmax(logits, dim=-1).squeeze(0)

            # No-repeat trigram penalty
            if len(tokens) >= 3:
                for k in range(len(tokens) - 2):
                    ng = tuple(tokens[k:k + 3])
                    log_probs[ng[-1]] = -1e4

            top_v, top_i = log_probs.topk(beam_size)
            for v, idx_val in zip(top_v.tolist(), top_i.tolist()):
                new_beams.append((score + v, tokens + [idx_val]))

        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
        if not beams:
            break

    pool = completed if completed else beams
    best = max(pool, key=lambda x: x[0] / max(len(x[1]), 1))
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
            f"{'moviesum' if 'moviesum' in src_path else 'mensa'} --out {src_path}")

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
        project="LED-MambaHypergraph",
        name=ABLATION["run_name"],
        config={
            "lr_new": LR_NEW_LAYERS, "lr_decoder": LR_DECODER, "lr_lora": LR_LORA,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "accum_steps": ACCUMULATION_STEPS,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_target_tokens": MAX_TARGET_TOKENS,
            "max_entities": MAX_ENTITIES,
            "architecture": "LEDMambaHypergraphSummariser",
            **ABLATION,
        },
    )

    # ── 2. Datasets ───────────────────────────────────────────────────────────
    # LED tokenizer for both dataset construction and model
    led_tokenizer = AutoTokenizer.from_pretrained(ABLATION["led_model"])

    base_train = SceneDataset(TRAIN_SPLIT_PATH, max_seq_len=256)
    base_eval  = SceneDataset(EVAL_SPLIT_PATH,  max_seq_len=256)
    train_ds   = MovieHypergraphDataset(base_train, led_tokenizer,
                                         max_scenes=MAX_SCENES)
    eval_ds    = MovieHypergraphDataset(base_eval, led_tokenizer,
                                         max_scenes=MAX_SCENES)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True,
                          persistent_workers=True,
                          collate_fn=hypergraph_collate_fn)
    eval_dl  = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True,
                          persistent_workers=True,
                          collate_fn=hypergraph_collate_fn)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    tokenizer = led_tokenizer
    _sum_module.LED_MODEL = ABLATION["led_model"]

    model = LEDMambaHypergraphSummariser(
        vocab_size=len(tokenizer),
        d_model=ABLATION["d_model"],
        max_entities=MAX_ENTITIES,
        max_scenes=MAX_SCENES,
        tokenizer=tokenizer,
        use_adaptive_streams=not ABLATION["no_adaptive_streams"],
        use_entity_names=not ABLATION["no_entity_names"],
        edge_dropout=ABLATION["edge_dropout"],
        mamba_layers=ABLATION["mamba_layers"],
    ).to(device)
    print(f"Model: d_model={ABLATION['d_model']}  mamba_layers={ABLATION['mamba_layers']}  "
          f"adaptive_streams={not ABLATION['no_adaptive_streams']}  "
          f"entity_names={not ABLATION['no_entity_names']}")

    # ── Ablations ─────────────────────────────────────────────────────────────
    if ABLATION["no_hypergraph"]:
        def _noop_hyp(self, scene_reps, _inc, _etid, _entype, _emask, **kw):
            return scene_reps, torch.zeros(
                scene_reps.size(0), self.max_entities, self.d_model,
                device=scene_reps.device)
        model.hypergraph_tower.forward = types.MethodType(
            _noop_hyp, model.hypergraph_tower)
        print("ABLATION: hypergraph tower disabled (LED-only baseline)")

    if ABLATION["static_hypergraph"]:
        for p in model.hypergraph_tower.entity_mamba.parameters():
            p.requires_grad = False
        print("ABLATION: Entity Mamba frozen → static hypergraph")

    # ── 4. Checkpoint detection ───────────────────────────────────────────────
    ckpt_latest = "/tmp/uday/checkpoints/led_mamba_latest.pt"
    start_epoch = 0
    ckpt_state  = None
    if os.path.exists(ckpt_latest):
        print(f"Resuming from {ckpt_latest}...")
        ckpt_state  = torch.load(ckpt_latest, map_location=device, weights_only=True)
        start_epoch = ckpt_state.get("epoch", 0)

    # ── 5a. Xavier init for new layers ────────────────────────────────────────
    _skip = {"led_encoder", "led_decoder", "head"}
    for name, param in model.named_parameters():
        if param.requires_grad and not any(s in name for s in _skip):
            if len(param.shape) > 1:
                torch.nn.init.xavier_uniform_(param)

    # ── 5b. Checkpoint load ───────────────────────────────────────────────────
    if ckpt_state:
        cur = model.state_dict()
        ckpt_params = ckpt_state["model_state_dict"]
        mismatched = [k for k, v in ckpt_params.items()
                      if k in cur and v.shape != cur[k].shape]
        if mismatched:
            print(f"  Skipping {len(mismatched)} shape-mismatched params: {mismatched}")
        compatible = {k: v for k, v in ckpt_params.items() if k not in mismatched}
        model.load_state_dict(compatible, strict=False)

    # ── 5c. Stage setup ──────────────────────────────────────────────────────
    lora_applied = False

    def apply_lora_entity_mamba(r=16, alpha=32):
        """Apply LoRA to entity Mamba's projection layers for efficient fine-tuning."""
        nonlocal lora_applied
        lora_cfg = LoraConfig(
            r=r, lora_alpha=alpha,
            target_modules=["in_proj", "x_proj", "out_proj", "dt_proj"],
            lora_dropout=0.05, bias="none",
        )
        model.hypergraph_tower.entity_mamba = get_peft_model(
            model.hypergraph_tower.entity_mamba, lora_cfg)
        lora_applied = True
        model.enable_gradient_checkpointing()
        print(f"LoRA applied to entity Mamba: r={r}, alpha={alpha}")

    def _set_stage1_grads():
        """Stage 1: LED encoder frozen. Train hypergraph, decoder, fusion."""
        for p in model.led_encoder.parameters():
            p.requires_grad = False
        for p in model.led_decoder.parameters():
            p.requires_grad = True
        for p in model.head.parameters():
            p.requires_grad = True
        for p in model.hypergraph_tower.parameters():
            p.requires_grad = True
        for p in model.graph_text_fusion.parameters():
            p.requires_grad = True
        for p in model.entity_scene_attn.parameters():
            p.requires_grad = True
        for p in model.scene_pool_proj.parameters():
            p.requires_grad = True
        for p in model.scene_pool_norm.parameters():
            p.requires_grad = True
        for p in model.memory_norm.parameters():
            p.requires_grad = True

    def _set_stage2_grads():
        """Stage 2: LED encoder partially unfrozen (global attention layers).
        LoRA on entity Mamba. Everything else stays trainable."""
        _set_stage1_grads()  # base: everything trainable except encoder
        # Unfreeze LED encoder's global attention layers
        for name, p in model.led_encoder.named_parameters():
            if "global" in name.lower():
                p.requires_grad = True

    if start_epoch >= EPOCHS_STAGE1:
        apply_lora_entity_mamba(r=16, alpha=32)
        _set_stage2_grads()
    else:
        _set_stage1_grads()

    # Diagnostic
    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen_named    = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    trainable_M = sum(p.numel() for _, p in trainable_named) / 1e6
    frozen_M    = sum(p.numel() for _, p in frozen_named) / 1e6
    print(f"  Trainable: {len(trainable_named)} tensors / {trainable_M:.1f}M params")
    print(f"  Frozen:    {len(frozen_named)} tensors / {frozen_M:.1f}M params")
    print(f"  Trainable groups: {sorted(set(n.split('.')[0] for n, _ in trainable_named))}")

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
        lora_p = [p for n, p in model.named_parameters()
                  if p.requires_grad and "lora_" in n]
        # Pretrained LED decoder + head at lower LR
        decoder_p = [p for n, p in model.named_parameters()
                     if p.requires_grad
                     and ("led_decoder" in n or n.startswith("head."))
                     and "lora_" not in n]
        _dec_ids = {id(p) for p in decoder_p}
        other_p = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" not in n
                   and id(p) not in _dec_ids]
        groups = [
            {"params": other_p,    "lr": LR_NEW_LAYERS},
            {"params": decoder_p,  "lr": LR_DECODER},
        ]
        if lora_p:
            groups.append({"params": lora_p, "lr": LR_LORA})
        return AdamW(groups, weight_decay=0.01)

    optimizer = _make_optim()
    wandb.watch(model, criterion, log="all", log_freq=50)

    total_steps  = (len(train_dl) // ACCUMULATION_STEPS) * EPOCHS
    warmup_steps = int(0.05 * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── 8. Evaluation metrics ─────────────────────────────────────────────────
    rouge_metric     = hf_evaluate.load("rouge")
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

    for epoch in range(start_epoch, EPOCHS):

        # Curriculum: anneal entity penalty and contrastive weight
        if epoch >= EPOCHS_STAGE1:
            s2 = epoch - EPOCHS_STAGE1
            criterion.alpha          = 0.2 + 0.3 * (s2 / max(1, EPOCHS_STAGE2 - 1))
            criterion.entity_penalty = 3.0 + 4.0 * (s2 / max(1, EPOCHS_STAGE2 - 1))
        else:
            criterion.alpha          = 0.1
            criterion.entity_penalty = 3.0

        print(f"\n[Epoch {epoch+1}/{EPOCHS}]  "
              f"α={criterion.alpha:.3f}  entity_pen={criterion.entity_penalty:.2f}")

        # Stage transition
        if epoch == EPOCHS_STAGE1 and not lora_applied:
            print("→ Stage 2: applying LoRA to entity Mamba")
            torch.cuda.empty_cache()
            apply_lora_entity_mamba(r=16, alpha=32)
            _set_stage2_grads()
            optimizer = _make_optim()
            s2_total  = (len(train_dl) // ACCUMULATION_STEPS) * EPOCHS_STAGE2
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=int(0.05 * s2_total),
                num_training_steps=s2_total,
            )

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss_sum  = 0.0
        n_valid_batches = 0
        optimizer.zero_grad()

        bar = tqdm(train_dl, desc=f"E{epoch+1} Train", unit="batch")
        for bi, batch in enumerate(bar):
            inp   = batch["input_ids"].to(device)
            amsk  = batch["attention_mask"].to(device)
            gattn = batch["global_attention_mask"].to(device)
            sbnds = batch["scene_boundaries"].to(device)
            inc   = batch["incidence_matrix"].to(device)
            etid  = batch["edge_type_ids"].to(device)
            enid  = batch["entity_type_ids"].to(device)
            emk   = batch["entity_mask"].to(device)
            emot  = batch["emotion_matrix"].to(device)
            tgt   = batch["target_ids"].to(device)
            trip  = batch["triplets"]
            enames = batch.get("entity_names")

            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    log_pr, H_text, labels, _, H_hyp = model(
                        inp, amsk, sbnds, gattn,
                        inc, etid, enid, emk,
                        target_ids=tgt, entity_names=enames,
                        emotion_matrix=emot,
                    )
                    log_pr = log_pr.float()
                    loss   = criterion(
                        log_probs=log_pr.view(-1, log_pr.size(-1)),
                        targets=labels.view(-1),
                        triplets=trip[0] if trip else [],
                        hidden_states=H_text,
                        head_weight=model.head.weight,
                        incidence_matrix=inc,
                    )
                    loss = loss / ACCUMULATION_STEPS
            except torch.OutOfMemoryError:
                print(f"  ⚠ OOM at batch {bi}, skipping")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue

            val = loss.item() * ACCUMULATION_STEPS
            if math.isnan(val) or math.isinf(val):
                print(f"  ⚠ NaN/Inf at batch {bi}, skipping")
                nan_w = [(n, p) for n, p in model.named_parameters()
                         if not torch.isfinite(p.data).all()]
                if nan_w:
                    print(f"    → {len(nan_w)} weight tensor(s) NaN — recovering")
                    for _, p in nan_w:
                        p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)
                    for _, p in nan_w:
                        if p in optimizer.state:
                            del optimizer.state[p]
                optimizer.zero_grad()
                continue

            loss.backward()

            if (bi + 1) % ACCUMULATION_STEPS == 0 or (bi + 1) == len(train_dl):
                has_nan_grad = False
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    print(f"  ⚠ NaN gradient at batch {bi}, discarding step")
                    optimizer.zero_grad()
                else:
                    nn_utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    nan_params = [(n, p) for n, p in model.named_parameters()
                                  if p.requires_grad and not torch.isfinite(p.data).all()]
                    if nan_params:
                        names = [n for n, _ in nan_params[:3]]
                        print(f"  ☠ NaN in {len(nan_params)} params: {names}")
                        for _, p in nan_params:
                            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)
                        for _, p in nan_params:
                            if p in optimizer.state:
                                del optimizer.state[p]

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
        hg_entities_per_scene = []
        hg_scenes_per_entity  = []
        hg_entity_coverage    = []

        bar_e = tqdm(eval_dl, desc=f"E{epoch+1} Eval", unit="batch")
        with torch.no_grad():
            for bi, batch in enumerate(bar_e):
                inp   = batch["input_ids"].to(device)
                amsk  = batch["attention_mask"].to(device)
                gattn = batch["global_attention_mask"].to(device)
                sbnds = batch["scene_boundaries"].to(device)
                inc   = batch["incidence_matrix"].to(device)
                etid  = batch["edge_type_ids"].to(device)
                enid  = batch["entity_type_ids"].to(device)
                emk   = batch["entity_mask"].to(device)
                emot  = batch["emotion_matrix"].to(device)
                tgt   = batch["target_ids"].to(device)
                trip  = batch["triplets"]
                enames = batch.get("entity_names")

                try:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        log_pr, H_text, labels, _, H_hyp = model(
                            inp, amsk, sbnds, gattn,
                            inc, etid, enid, emk,
                            target_ids=tgt, entity_names=enames,
                            emotion_matrix=emot,
                        )
                        log_pr = log_pr.float()
                        eloss  = criterion(
                            log_probs=log_pr.view(-1, log_pr.size(-1)),
                            targets=labels.view(-1),
                            triplets=trip[0] if trip else [],
                            hidden_states=H_text,
                            head_weight=model.head.weight,
                            incidence_matrix=inc,
                        )
                except torch.OutOfMemoryError:
                    print(f"  ⚠ OOM at eval batch {bi}, skipping")
                    torch.cuda.empty_cache()
                    continue

                val = eloss.item()
                if math.isnan(val) or math.isinf(val):
                    continue
                eval_loss_sum += val
                bar_e.set_postfix(eval_loss=f"{val:.4f}")

                # Hypergraph quality stats
                inc_b = inc[0].cpu().float()
                active = (inc_b > 0).float()
                ents_per_scene = active.sum(dim=0)
                scenes_per_ent = active.sum(dim=1)
                hg_entities_per_scene.append(
                    ents_per_scene[ents_per_scene > 0].mean().item()
                    if ents_per_scene.sum() > 0 else 0.0)
                hg_scenes_per_entity.append(
                    scenes_per_ent[scenes_per_ent > 0].mean().item()
                    if scenes_per_ent.sum() > 0 else 0.0)

                if enames and enames[0]:
                    ref_text = tokenizer.decode(tgt[0].cpu().tolist(),
                                                skip_special_tokens=True).lower()
                    hg_names = [n.lower() for n in enames[0] if n]
                    if hg_names:
                        found = sum(1 for n in hg_names if n in ref_text)
                        hg_entity_coverage.append(found / len(hg_names))

                if hasattr(model, "_last_gate_mean") and model._last_gate_mean != 0.0:
                    wandb.log({"eval/fusion_gate_mean": model._last_gate_mean,
                               "epoch": epoch + 1})

                # Generation every 10th batch
                if bi % 10 == 0:
                    S_count = sbnds.size(1)
                    aligned_mem, _, __, dt_vals = model(
                        inp, amsk, sbnds, gattn,
                        inc, etid, enid, emk,
                        entity_names=enames, emotion_matrix=emot, return_dt=True,
                    )

                    mem_pad = torch.zeros(aligned_mem.size(0), aligned_mem.size(1),
                                          dtype=torch.bool, device=device)
                    for b_idx in range(aligned_mem.size(0)):
                        for s in range(S_count):
                            start, end = sbnds[b_idx, s].tolist()
                            if start >= end:
                                mem_pad[b_idx, s] = True
                    mem_pad[:, S_count:] = ~emk
                    all_masked = mem_pad.all(dim=1)
                    mem_pad[all_masked, 0] = False
                    enc_attn = (~mem_pad).long()

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = generate_summary(
                            model, aligned_mem, enc_attn, tokenizer,
                            device, max_new_tokens=200, beam_size=4,
                        )
                    ref = tokenizer.decode(tgt[0].cpu().tolist(),
                                           skip_special_tokens=True)
                    all_preds.append(pred)
                    all_refs.append(ref)
                    wandb.log({
                        "eval/sample": wandb.Html(
                            f"<b>Pred:</b> {pred}<br><b>Ref:</b> {ref}"),
                        "epoch": epoch + 1,
                    })

                    # dt heatmap (first batch of each epoch)
                    if bi == 0 and dt_vals is not None:
                        _mname = batch["movie_name"][0] if batch.get("movie_name") else ""
                        log_entity_dt_heatmap(dt_vals, enames, emk, _mname)

                    if bi % 50 == 0:
                        _mname = batch["movie_name"][0] if batch.get("movie_name") else ""
                        log_hyperedge_attention(model, H_hyp, inc, _mname)

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
        r1 = r2 = rL = met = ent_f1 = 0.0
        if all_preds:
            rs = rouge_metric.compute(predictions=all_preds, references=all_refs,
                                      use_stemmer=True)
            r1, r2, rL = rs["rouge1"], rs["rouge2"], rs["rougeL"]
            met = meteor_metric.compute(
                predictions=all_preds, references=all_refs)["meteor"]
            ent_f1 = _entity_f1(all_preds, all_refs)

        print(f"Epoch {epoch+1} | "
              f"Train {avg_train:.4f} | Eval {avg_eval:.4f} | "
              f"R1 {r1:.4f} | R2 {r2:.4f} | RL {rL:.4f} | "
              f"METEOR {met:.4f} | EntF1 {ent_f1:.4f}")

        avg_ents_per_scene = sum(hg_entities_per_scene) / max(len(hg_entities_per_scene), 1)
        avg_scenes_per_ent = sum(hg_scenes_per_entity) / max(len(hg_scenes_per_entity), 1)
        avg_ent_coverage   = sum(hg_entity_coverage) / max(len(hg_entity_coverage), 1)
        print(f"  Hypergraph: {avg_ents_per_scene:.1f} ents/scene, "
              f"{avg_scenes_per_ent:.1f} scenes/ent, "
              f"coverage={avg_ent_coverage:.3f}")

        wandb.log({
            "epoch/train_loss": avg_train, "epoch/eval_loss": avg_eval,
            "epoch/rouge1": r1, "epoch/rouge2": r2, "epoch/rougeL": rL,
            "epoch/meteor": met, "epoch/entity_f1": ent_f1,
            "epoch/hg_ents_per_scene": avg_ents_per_scene,
            "epoch/hg_scenes_per_entity": avg_scenes_per_ent,
            "epoch/hg_entity_coverage": avg_ent_coverage,
            "epoch": epoch + 1,
        })

        # ── Checkpoint ────────────────────────────────────────────────────────
        save_dir = "/tmp/uday/checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        ckpt_ep = f"{save_dir}/led_mamba_epoch_{epoch+1}.pt"
        torch.save({
            "epoch":            epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train, "eval_loss": avg_eval,
            "rouge1": r1, "rouge2": r2, "rougeL": rL,
        }, ckpt_ep)
        torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict()},
                   ckpt_latest)
        print(f"  Checkpoint saved → {ckpt_ep}")

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    train()

"""
train.py  —  GraM-Former v2 Training Pipeline
==============================================
Key changes over v1
-------------------
  - MovieGraphDatasetV2  builds TWO typed adjacency matrices per movie:
      causal_adj       (directed SVO causal chains)
      char_state_adj   (weighted by emotion-state-change magnitude)
  - IDF weights computed per-movie over entity co-occurrences
  - Separate learning-rate groups (encoder LoRA vs new layers)
  - Label-smoothing loss + graduated entity penalty (α_entity schedule)
  - Beam search evaluation loop (beam=4, no_repeat_ngram_size=3)
  - ROUGE-1/2/L computed every epoch via huggingface evaluate
  - Narrative coherence loss wired in from sum.py
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
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, BartForConditionalGeneration
import torch.nn.utils as nn_utils
import torch._dynamo

torch._dynamo.config.cache_size_limit = 512
os.environ["USE_TORCH"] = "1"
os.environ["TMPDIR"]    = "/tmp/uday"
sys.path.insert(0, "/tmp/uday/lib")

from peft import LoraConfig, get_peft_model
import sum as _sum_module
from sum import (
    GraMFormerV2,
    RelationalEventConsistencyLoss,
    RaftConsensusAttentionV2,
    log_character_attention_map,
    log_character_attention_map_labeled,
)

torch.set_float32_matmul_precision("high")

# =============================================================================
# Config
# =============================================================================
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument("--run_name",           type=str,   default="full_model")
_parser.add_argument("--bart_model", type=str,
                     default=("/tmp/uday/bart-large"
                              if __import__("os").path.isdir("/tmp/uday/bart-large")
                              else "facebook/bart-large"))
_parser.add_argument("--bart_tokenizer", type=str, default="",
                     help="Explicit tokenizer path/name (defaults to bart_model)")
_parser.add_argument("--d_model",            type=int,   default=1024)
_parser.add_argument("--no_causal_graph",    action="store_true")
_parser.add_argument("--no_char_state_graph",action="store_true")
_parser.add_argument("--no_coherence_loss",  action="store_true")
_parser.add_argument("--no_contrastive_loss",action="store_true")
_parser.add_argument("--no_pointer_head",    action="store_true")
_parser.add_argument("--use_raft_v1",        action="store_true")
_parser.add_argument("--single_binary_graph",action="store_true")
_parser.add_argument("--entity_penalty",     type=float, default=3.0)
_parser.add_argument("--dataset",            type=str,   default="mensa",
                     choices=["mensa", "moviesum", "both"])
_args, _ = _parser.parse_known_args()

# Ablation flags — each flag disables one component for ablation experiments.
# Run with e.g.:  python train.py --run_name ablation_no_causal --no_causal_graph
# C2: explicit BART tokenizer — falls back to bart_model path
_BART_TOKENIZER = _args.bart_tokenizer or _args.bart_model

ABLATION = {
    "run_name":            _args.run_name,
    "bart_model":          _args.bart_model,
    "bart_tokenizer":      _BART_TOKENIZER,
    "d_model":             _args.d_model,
    "use_causal_graph":    not _args.no_causal_graph,
    "use_char_state_graph":not _args.no_char_state_graph,

    "use_coherence_loss":  not _args.no_coherence_loss,
    "use_contrastive_loss":not _args.no_contrastive_loss,
    "use_pointer_head":    not _args.no_pointer_head,
    "use_raft_v1":         _args.use_raft_v1,
    "single_binary_graph": _args.single_binary_graph,
    "entity_penalty":      _args.entity_penalty,
    "dataset":             _args.dataset,
}

JSONL_PATH       = "/tmp/uday/mensa_train_data.jsonl.gz"
MOVIESUM_PATH    = "/tmp/uday/moviesum_data.jsonl.gz"
# Splits live on real disk (/tmp), NOT /dev/shm (RAM disk).
# Uncompressed splits at 512-dim adj matrices = 74GB+ which fills RAM.
# We use gzip compression — reads slightly slower but 10x smaller on disk.
TRAIN_SPLIT_PATH = f"/tmp/uday/train_{ABLATION['run_name']}.jsonl.gz"
EVAL_SPLIT_PATH  = f"/tmp/uday/eval_{ABLATION['run_name']}.jsonl.gz"
NUM_TRAIN_MOVIES = 700

BATCH_SIZE         = 1
ACCUMULATION_STEPS = 16
EPOCHS_STAGE1      = 2    # Encoder frozen
EPOCHS_STAGE2      = 20   # LoRA encoder unfrozen
LR_NEW_LAYERS      = 1e-4
LR_LORA            = 1e-5

MAX_SEQ_LEN        = 512   # must match extraction max_seq_len
MAX_SCENES         = 200

# =============================================================================
# MensaGraphDataset (low-level, scene-by-scene reader)
# =============================================================================

class MensaGraphDataset(Dataset):
    def __init__(self, jsonl_path, max_seq_len=256):
        self.max_seq_len = max_seq_len
        self.jsonl_path  = jsonl_path
        # C2: use BART tokenizer — same BPE vocab as RoBERTa, but explicit
        # decoder_start_token_id=2 semantics match the model's teacher forcing.
        self.tokenizer   = AutoTokenizer.from_pretrained(_BART_TOKENIZER)
        self.movie_ids   = []

        # Index-only pass: store byte offsets so __getitem__ can seek
        # directly to any record without loading everything into RAM.
        # This keeps memory usage at ~50MB regardless of dataset size.
        print(f"Indexing dataset from {jsonl_path}...")
        self._offsets = []   # list of byte offsets for each line
        open_fn = gzip.open if jsonl_path.endswith(".gz") else open

        # For gzip we can't seek, so we store all lines in a list of
        # (movie_id, json_string) — still much cheaper than parsed dicts.
        # For plain files we store byte offsets for true random access.
        if jsonl_path.endswith(".gz"):
            self._lines = []
            with gzip.open(jsonl_path, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    # Store raw JSON string — parse lazily in __getitem__
                    mid = json.loads(line)["movie_id"]
                    self._lines.append(line)
                    self.movie_ids.append(mid)
            print(f"Indexed {len(self._lines)} scenes (lazy loading).")
        else:
            self._lines = []
            with open(jsonl_path, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    mid = json.loads(line)["movie_id"]
                    self._lines.append(line)
                    self.movie_ids.append(mid)
            print(f"Indexed {len(self._lines)} scenes.")

    def __len__(self):
        return len(self._lines)

    def __getitem__(self, idx):
        item     = json.loads(self._lines[idx])
        seq_len  = self.max_seq_len

        def _pad_or_trunc(lst, length, pad=1):
            return (lst + [pad] * length)[:length]

        def _to_bool_tensor(lst):
            return torch.tensor(_pad_or_trunc(lst, seq_len, 0), dtype=torch.bool)

        # Adjacency matrix: stored at extraction seq_len (512), truncate to
        # current seq_len to handle any mismatch safely
        raw_adj = item["adjacency_matrix"]
        adj_t   = torch.tensor(raw_adj, dtype=torch.int8) if isinstance(raw_adj, list) else raw_adj
        adj_t   = adj_t[:seq_len, :seq_len]
        # Pad to seq_len x seq_len if smaller
        if adj_t.size(0) < seq_len:
            pad_adj = torch.zeros(seq_len, seq_len, dtype=torch.int8)
            pad_adj[:adj_t.size(0), :adj_t.size(1)] = adj_t
            adj_t = pad_adj

        return {
            "input_ids":        torch.tensor(_pad_or_trunc(item["input_ids"],  seq_len), dtype=torch.long),
            "target_ids":       torch.tensor(_pad_or_trunc(item["target_ids"], seq_len), dtype=torch.long),
            "adjacency_matrix": adj_t,
            "action_mask":      _to_bool_tensor(item["action_mask"]),
            "dialogue_mask":    _to_bool_tensor(item["dialogue_mask"]),
            "entity_mask":      _to_bool_tensor(item.get("entity_mask", [0] * seq_len)),
            "header_mask":      _to_bool_tensor(item.get("header_mask", [0] * seq_len)),
            "graph_triplets":   item.get("graph_triplets", []),
            "character_emotions": item.get("character_emotions", {}),
            "scene_meta":       item.get("scene_meta", {}),
            "movie_id":         item["movie_id"],
        }


# =============================================================================
# MovieGraphDatasetV2 — builds 3 typed movie-level graphs
# =============================================================================

class MovieGraphDatasetV2(Dataset):
    def __init__(self, scene_dataset, max_scenes=200):
        self.scene_dataset = scene_dataset
        self.max_scenes    = max_scenes
        self.movie_map     = defaultdict(list)
        self.adj_cache     = {}

        print("Building movie index...")
        for i in range(len(scene_dataset)):
            raw_id     = scene_dataset.movie_ids[i]
            movie_name = raw_id.split("_Scene_")[0] if "_Scene_" in raw_id else raw_id
            self.movie_map[movie_name].append(i)
        self.movie_names = list(self.movie_map.keys())
        print(f"Found {len(self.movie_names)} unique movies.")

    def __len__(self):
        return len(self.movie_names)

    def __getitem__(self, idx):
        movie_name   = self.movie_names[idx]
        scene_indices = self.movie_map[movie_name][:self.max_scenes]
        scenes        = [self.scene_dataset[i] for i in scene_indices]
        num_scenes    = len(scenes)
        S             = self.max_scenes

        if movie_name in self.adj_cache:
            causal_adj, char_state_adj, idf_weights = self.adj_cache[movie_name]
            return {
                "movie_name":     movie_name,
                "scenes":         scenes,
                "causal_adj":     causal_adj,
                "char_state_adj": char_state_adj,
                "idf_weights":    idf_weights,
            }

        # ── Graph 1: Causal Event Graph ─────────────────────────────────────
        # Directed edge i→j if obj(i) matches subj(j) (causal chain)
        causal_adj = torch.zeros(S, S)
        causal_adj.fill_diagonal_(1.0)   # self-loop

        for i in range(num_scenes):
            trips_i = scenes[i]["graph_triplets"]
            objs_i  = set()
            for t in trips_i:
                parts = t.split("_")
                if len(parts) >= 3:
                    objs_i.add(parts[2].strip().lower())
            objs_i.discard("")

            for j in range(i + 1, num_scenes):
                trips_j  = scenes[j]["graph_triplets"]
                subjs_j  = set()
                for t in trips_j:
                    parts = t.split("_")
                    if len(parts) >= 1:
                        subjs_j.add(parts[0].replace("NOT ", "").strip().lower())
                subjs_j.discard("")

                if objs_i & subjs_j:            # causal chain: obj(i)→subj(j)
                    causal_adj[i, j] = 1.0      # directed forward
                    causal_adj[j, i] = 0.3      # weak backward (context)

        # ── Graph 2: Character State Graph ──────────────────────────────────
        # Edge weight = magnitude of emotion-polarity change between appearances
        char_state_adj = torch.zeros(S, S)
        char_state_adj.fill_diagonal_(1.0)

        # Collect all emotion dicts
        emotions = [scenes[i]["character_emotions"] for i in range(num_scenes)]
        for i in range(num_scenes):
            for j in range(i + 1, num_scenes):
                shared_chars = set(emotions[i].keys()) & set(emotions[j].keys())
                if not shared_chars:
                    continue
                # Edge weight = mean absolute state change across shared characters
                changes = [
                    abs(emotions[i][c] - emotions[j][c])
                    for c in shared_chars
                ]
                weight = sum(changes) / len(changes)
                char_state_adj[i, j] = weight
                char_state_adj[j, i] = weight

        # ── IDF weights over entity co-occurrences ──────────────────────────
        # idf(entity) = log(N / df) where df = #scenes containing entity
        all_entities = defaultdict(int)
        scene_entity_sets = []
        for i in range(num_scenes):
            ents = set()
            for t in scenes[i]["graph_triplets"]:
                parts = t.split("_")
                if len(parts) >= 1:
                    ents.add(parts[0].replace("NOT ", "").strip().lower())
                if len(parts) >= 3:
                    ents.add(parts[2].strip().lower())
            ents.discard("")
            scene_entity_sets.append(ents)
            for e in ents:
                all_entities[e] += 1

        idf_weights = torch.zeros(S, S)
        N = max(num_scenes, 1)
        for i in range(num_scenes):
            for j in range(i + 1, num_scenes):
                shared = scene_entity_sets[i] & scene_entity_sets[j]
                if shared:
                    # Average IDF of shared entities (rare = high IDF = high weight)
                    avg_idf = sum(
                        math.log(N / max(all_entities[e], 1))
                        for e in shared
                    ) / len(shared)
                    idf_weights[i, j] = avg_idf
                    idf_weights[j, i] = avg_idf

        self.adj_cache[movie_name] = (causal_adj, char_state_adj, idf_weights)
        return {
            "movie_name":     movie_name,
            "scenes":         scenes,
            "causal_adj":     causal_adj,
            "char_state_adj": char_state_adj,
            "idf_weights":    idf_weights,
        }


# =============================================================================
# Collate function
# =============================================================================

def movie_collate_fn(batch):
    max_scenes = max(len(item["scenes"]) for item in batch)
    seq_len    = len(batch[0]["scenes"][0]["input_ids"])
    B          = len(batch)

    input_ids    = torch.full((B, max_scenes, seq_len), 1, dtype=torch.long)
    target_ids   = torch.full((B, max_scenes, seq_len), 1, dtype=torch.long)
    adj_matrix   = torch.zeros((B, max_scenes, seq_len, seq_len), dtype=torch.int8)
    action_mask  = torch.zeros((B, max_scenes, seq_len), dtype=torch.bool)
    dial_mask    = torch.zeros((B, max_scenes, seq_len), dtype=torch.bool)
    ent_mask     = torch.zeros((B, max_scenes, seq_len), dtype=torch.bool)
    head_mask    = torch.zeros((B, max_scenes, seq_len), dtype=torch.bool)

    causal_adj_b     = torch.zeros((B, max_scenes, max_scenes))
    char_state_adj_b = torch.zeros((B, max_scenes, max_scenes))
    idf_weights_b    = torch.zeros((B, max_scenes, max_scenes))

    all_triplets = []

    for b, item in enumerate(batch):
        ns = len(item["scenes"])
        for s in range(max_scenes):
            if s < ns:
                sc = item["scenes"][s]
                # Defensive truncation: handles seq_len mismatches between
                # extraction (512) and any future config changes
                SL = seq_len
                input_ids[b, s]   = sc["input_ids"][:SL]
                target_ids[b, s]  = sc["target_ids"][:SL]
                adj               = sc["adjacency_matrix"]
                adj_t             = torch.tensor(adj, dtype=torch.int8) if isinstance(adj, list) else adj
                adj_t             = adj_t[:SL, :SL]
                adj_matrix[b, s, :adj_t.size(0), :adj_t.size(1)] = adj_t
                action_mask[b, s] = torch.tensor(sc["action_mask"][:SL],   dtype=torch.bool)
                dial_mask[b, s]   = torch.tensor(sc["dialogue_mask"][:SL], dtype=torch.bool)
                ent_mask[b, s]    = torch.tensor(sc["entity_mask"][:SL],   dtype=torch.bool)
                head_mask[b, s]   = torch.tensor(sc["header_mask"][:SL],   dtype=torch.bool)
                all_triplets.append(sc["graph_triplets"])
            else:
                all_triplets.append([])

        causal_adj_b[b, :ns, :ns]     = item["causal_adj"][:ns, :ns]
        char_state_adj_b[b, :ns, :ns] = item["char_state_adj"][:ns, :ns]
        idf_weights_b[b, :ns, :ns]    = item["idf_weights"][:ns, :ns]

    return {
        "input_ids":      input_ids,
        "target_ids":     target_ids,
        "adj_matrix":     adj_matrix,
        "action_mask":    action_mask,
        "dial_mask":      dial_mask,
        "ent_mask":       ent_mask,
        "head_mask":      head_mask,
        "causal_adj":     causal_adj_b,
        "char_state_adj": char_state_adj_b,
        "idf_weights":    idf_weights_b,
        "triplets":       all_triplets,
    }


# =============================================================================
# Dataset splitting
# =============================================================================

def split_dataset_by_movie(input_path, train_path, eval_path, num_train=700):
    print(f"Splitting by movie (train: {num_train})...")
    
    # 1. Fail Fast: Make sure the file actually exists and isn't blocking
    import os
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Input dataset not found at {input_path}")
        
    movie_id_re  = re.compile(r'"movie_id"\s*:\s*"([^"]+)"')
    train_movies = set()
    eval_movies  = set()
    open_fn      = gzip.open if input_path.endswith(".gz") else open
    out_fn       = gzip.open if train_path.endswith(".gz") else open

    with open_fn(input_path, "rt", encoding="utf-8") as inf, \
         out_fn(train_path, "wt", encoding="utf-8") as tr_out, \
         out_fn(eval_path,  "wt", encoding="utf-8") as ev_out:

        # 2. Add tqdm to the file iterator so you can see the speed (lines/sec)
        from tqdm import tqdm
        for line in tqdm(inf, desc="Processing JSONL", unit=" lines"):
            if not line.strip():
                continue
            match      = movie_id_re.search(line)
            raw_id     = match.group(1) if match else "unknown"
            base_name  = raw_id.split("_Scene_")[0] if "_Scene_" in raw_id else raw_id

            if base_name in train_movies:
                tr_out.write(line)
            elif base_name in eval_movies:
                ev_out.write(line)
            else:
                if len(train_movies) < num_train:
                    train_movies.add(base_name)
                    tr_out.write(line)
                else:
                    eval_movies.add(base_name)
                    ev_out.write(line)

    print(f"Split complete: {len(train_movies)} train / {len(eval_movies)} eval movies")

# =============================================================================
# Beam-search generation (proper inference for ROUGE eval)
# =============================================================================

@torch.no_grad()
def generate_summary(model, aligned_memory, enc_attn_mask, tokenizer,
                     device, max_new_tokens=200, beam_size=4):
    """
    Beam search over the BART decoder using the pre-computed aligned_memory.
    Returns decoded string.
    """
    B = aligned_memory.size(0)
    assert B == 1, "generate_summary expects batch size 1"

    # Initialise beams: (score, token_ids)
    beams = [(0.0, [tokenizer.bos_token_id or 0])]
    completed = []

    eos_id = tokenizer.eos_token_id or 2
    pad_id = tokenizer.pad_token_id or 1

    for _ in range(max_new_tokens):
        new_beams = []
        for score, tokens in beams:
            if tokens[-1] == eos_id:
                completed.append((score, tokens))
                continue
            t_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            t_mask = (t_ids != pad_id).long()

            dec_out = model.bart_decoder(
                input_ids=t_ids,
                attention_mask=t_mask,
                encoder_hidden_states=aligned_memory,
                encoder_attention_mask=enc_attn_mask,
            )
            logits     = model.head(dec_out.last_hidden_state[:, -1, :]).float()
            log_probs  = F.log_softmax(logits, dim=-1).squeeze(0)

            # No-repeat trigram penalty
            if len(tokens) >= 3:
                ngrams = set(
                    tuple(tokens[k:k + 3])
                    for k in range(len(tokens) - 2)
                )
                for ng in ngrams:
                    if len(ng) == 3:
                        log_probs[ng[-1]] = -1e4  # F7: avoid -inf NaN in bfloat16

            top_vals, top_idx = log_probs.topk(beam_size)
            for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
                new_beams.append((score + v, tokens + [idx]))

        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
        if len(beams) == 0:
            break

    if completed:
        best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
    else:
        best = max(beams, key=lambda x: x[0])

    return tokenizer.decode(best[1], skip_special_tokens=True)


# =============================================================================
# Main training loop
# =============================================================================

import torch.nn.functional as F   # needed for generate_summary above

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    torch.cuda.empty_cache()

    # ── 1. Data splits ───────────────────────────────────────────────────────
    if not (os.path.exists(TRAIN_SPLIT_PATH) and os.path.exists(EVAL_SPLIT_PATH)):
        split_dataset_by_movie(JSONL_PATH, TRAIN_SPLIT_PATH, EVAL_SPLIT_PATH,
                               num_train=NUM_TRAIN_MOVIES)

    EPOCHS = EPOCHS_STAGE1 + EPOCHS_STAGE2

    wandb.init(
        project="GraM-Former-v2",
        name=ABLATION["run_name"],
        config={
            "lr_new":          LR_NEW_LAYERS,
            "lr_lora":         LR_LORA,
            "epochs":          EPOCHS,
            "batch_size":      BATCH_SIZE,
            "accum_steps":     ACCUMULATION_STEPS,
            "max_seq_len":     MAX_SEQ_LEN,
            "architecture":    "GraM-Former-v2 HGT",
            **ABLATION,
        },
    )

    # ── 2. Datasets ───────────────────────────────────────────────────────────
    base_train = MensaGraphDataset(TRAIN_SPLIT_PATH, max_seq_len=MAX_SEQ_LEN)
    base_eval  = MensaGraphDataset(EVAL_SPLIT_PATH,  max_seq_len=MAX_SEQ_LEN)
    train_ds   = MovieGraphDatasetV2(base_train, max_scenes=MAX_SCENES)
    eval_ds    = MovieGraphDatasetV2(base_eval,  max_scenes=MAX_SCENES)

    train_dl   = DataLoader(train_ds, batch_size=1, shuffle=True,
                            num_workers=4, pin_memory=True,
                            collate_fn=movie_collate_fn)
    eval_dl    = DataLoader(eval_ds,  batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True,
                            collate_fn=movie_collate_fn)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    tokenizer = base_train.tokenizer

    # Apply ablation backbone settings to sum.py module globals
    _sum_module.BART_MODEL = ABLATION["bart_model"]

    # Swap RAFT v1 for ablation if requested
    if ABLATION["use_raft_v1"]:
        from sum import RaftConsensusAttention as _RaftV1
        _sum_module.RaftConsensusAttentionV2 = _RaftV1
        print("ABLATION: using RAFT v1 (no cross-modal attention)")

    model = GraMFormerV2(
        vocab_size=len(tokenizer),
        d_model=ABLATION["d_model"],
        num_layers=4,
        tokenizer=tokenizer,
    ).to(device)
    print(f"Model backbone: {ABLATION['bart_model']}  d_model={ABLATION['d_model']}")

    print("Xavier init on new layers...")
    _skip_init = {"embedding", "head", "roberta", "scene_proj"}
    for name, param in model.named_parameters():
        if param.requires_grad and not any(s in name for s in _skip_init):
            if len(param.shape) > 1:
                torch.nn.init.xavier_uniform_(param)

    # Ablation: disable pointer head by making p_gen always 1.0 (pure generation)
    if not ABLATION["use_pointer_head"]:
        import types
        def _no_pointer(self, dec_states, scene_mem, trips, tok, emb_w, dev):
            B, T, D = dec_states.shape
            p_gen       = torch.ones(B, T, 1, device=dev)
            ptr_probs   = torch.zeros(B, T, self.vocab_size, device=dev)
            return p_gen, ptr_probs
        model.pointer_head.forward = types.MethodType(_no_pointer, model.pointer_head)
        print("ABLATION: pointer head disabled (p_gen=1 always)")

    # ── 4. Checkpoint resumption ──────────────────────────────────────────────
    ckpt_path   = "/tmp/uday/checkpoints/gramformer_v2_latest.pt"
    start_epoch = 0
    checkpoint  = None
    if os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}...")
        checkpoint  = torch.load(ckpt_path, map_location=device, weights_only=True)
        start_epoch = checkpoint.get("epoch", 0)

    # ── 5. Stage setup ────────────────────────────────────────────────────────
    lora_applied = False

    def apply_lora(r=16, alpha=32):
        nonlocal lora_applied
        lora_cfg = LoraConfig(
            r=r, lora_alpha=alpha,
            target_modules=["in_proj", "x_proj", "out_proj", "dt_proj"],
            lora_dropout=0.05, bias="none",
        )
        model.encoder = get_peft_model(model.encoder, lora_cfg)
        lora_applied  = True
        model.enable_gradient_checkpointing()

    def _freeze_roberta(model):
        """RoBERTa stays frozen throughout all training stages."""
        for name, param in model.named_parameters():
            if "roberta" in name:
                param.requires_grad = False

    if start_epoch >= EPOCHS_STAGE1:
        apply_lora(r=64, alpha=128)
        for name, param in model.named_parameters():
            if "roberta" in name:
                param.requires_grad = False          # A1: always frozen
            elif "bart_decoder.layers" in name and "encoder_attn" not in name:
                param.requires_grad = False
            elif name.startswith("head."):
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        # Stage 1: freeze Mamba encoder and RoBERTa, train everything else
        for name, param in model.named_parameters():
            param.requires_grad = (
                "encoder" not in name and "roberta" not in name
            )

    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # ── 6. Criterion ──────────────────────────────────────────────────────────
    criterion = RelationalEventConsistencyLoss(
        alpha=0.1 if ABLATION["use_contrastive_loss"] else 0.0,
        tokenizer=tokenizer,
        entity_penalty=ABLATION["entity_penalty"],
        label_smoothing=0.1,
        coherence_weight=0.05 if ABLATION["use_coherence_loss"] else 0.0,
    )
    if not ABLATION["use_contrastive_loss"]:
        print("ABLATION: contrastive loss disabled")
    if not ABLATION["use_coherence_loss"]:
        print("ABLATION: narrative coherence loss disabled")

    # ── 7. Optimiser — separate LR groups ────────────────────────────────────
    def _make_optimizer(model):
        lora_params  = [p for n, p in model.named_parameters()
                        if p.requires_grad and "lora_" in n]
        other_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and "lora_" not in n]
        return AdamW([
            {"params": other_params, "lr": LR_NEW_LAYERS},
            {"params": lora_params,  "lr": LR_LORA},
        ], weight_decay=0.01)

    optimizer = _make_optimizer(model)
    scaler    = torch.cuda.amp.GradScaler()
    wandb.watch(model, criterion, log="all", log_freq=50)

    total_steps       = (len(train_dl) // ACCUMULATION_STEPS) * EPOCHS
    warmup_steps      = int(0.05 * total_steps)
    scheduler         = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # N1: evaluation metrics — ROUGE, BERTScore, METEOR
    rouge      = hf_evaluate.load("rouge")
    bertscore  = hf_evaluate.load("bertscore")
    meteor     = hf_evaluate.load("meteor")

    # N2: entity F1 — use a minimal spaCy NER pipeline (en_core_web_sm)
    import spacy as _spacy
    _nlp_ner = _spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])

    def _entity_f1(preds, refs):
        """Micro-averaged entity F1 using spaCy NER on preds vs refs."""
        tp = fp = fn = 0
        for pred, ref in zip(preds, refs):
            p_ents = {e.text.lower() for e in _nlp_ner(pred).ents}
            r_ents = {e.text.lower() for e in _nlp_ner(ref).ents}
            tp += len(p_ents & r_ents)
            fp += len(p_ents - r_ents)
            fn += len(r_ents - p_ents)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # ── 8. Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS):

        # Curriculum: entity penalty and contrastive alpha
        if epoch >= EPOCHS_STAGE1:
            stage2_ep = epoch - EPOCHS_STAGE1
            criterion.alpha        = 0.2 + 0.3 * (stage2_ep / max(1, EPOCHS_STAGE2 - 1))
            criterion.entity_penalty = 3.0 + 4.0 * (stage2_ep / max(1, EPOCHS_STAGE2 - 1))
        else:
            criterion.alpha        = 0.1
            criterion.entity_penalty = 3.0

        print(f"\n[Epoch {epoch + 1}/{EPOCHS}] α={criterion.alpha:.3f}  "
              f"entity_pen={criterion.entity_penalty:.2f}")

        # Stage transition
        if epoch == EPOCHS_STAGE1 and not lora_applied:
            print("→ Transitioning to Stage 2: applying LoRA to Mamba encoder")
            apply_lora(r=16, alpha=32)
            for name, param in model.named_parameters():
                if "roberta" in name:
                    param.requires_grad = False      # A1: keep frozen
                elif "encoder" not in name:
                    param.requires_grad = True
            optimizer = _make_optimizer(model)
            scaler    = torch.cuda.amp.GradScaler(enabled=False)
            stage2_total  = (len(train_dl) // ACCUMULATION_STEPS) * EPOCHS_STAGE2
            warmup2       = int(0.05 * stage2_total)
            scheduler     = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup2,
                num_training_steps=stage2_total
            )

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        bar = tqdm(train_dl, desc=f"E{epoch + 1} Train", unit="batch")
        for batch_idx, batch in enumerate(bar):
            inp     = batch["input_ids"].to(device)
            adj_m   = batch["adj_matrix"].to(device)
            a_mask  = batch["action_mask"].to(device)
            d_mask  = batch["dial_mask"].to(device)
            e_mask  = batch["ent_mask"].to(device)
            h_mask  = batch["head_mask"].to(device)
            c_adj   = batch["causal_adj"].to(device)
            cs_adj  = batch["char_state_adj"].to(device)
            idf_w   = batch["idf_weights"].to(device)
            tgt     = batch["target_ids"].to(device)
            trips   = batch["triplets"]

            # Zero out graphs disabled by ablation flags
            if not ABLATION["use_causal_graph"]:
                c_adj  = torch.zeros_like(c_adj)
            if not ABLATION["use_char_state_graph"]:
                cs_adj = torch.zeros_like(cs_adj)
            if ABLATION["single_binary_graph"]:
                # Collapse both graphs to a single binary co-occurrence adj
                # (v1 baseline — tests whether typed graphs help over binary)
                binary = ((c_adj + cs_adj) > 0).float()
                c_adj = cs_adj = binary

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, enc_mem, labels, _, gated_causal = model(
                    inp, adj_m, a_mask, d_mask, e_mask, h_mask,
                    c_adj, cs_adj, tgt, trips, idf_w
                )
                logits = logits.float()
                loss   = criterion(
                    log_probs=logits.view(-1, logits.size(-1)),
                    targets=labels.view(-1),
                    triplets=trips,
                    hidden_states=enc_mem,
                    head_weight=model.head.weight,

                    causal_adj=gated_causal,
                )
                ortho = criterion.get_riemannian_orthogonality_loss(model)
                loss  = loss + 0.001 * ortho
                loss  = loss / ACCUMULATION_STEPS

            val = loss.item() * ACCUMULATION_STEPS
            if math.isnan(val) or math.isinf(val):
                print(f"  ⚠ Explosion at batch {batch_idx} — skipping")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or \
               (batch_idx + 1) == len(train_dl):
                scaler.unscale_(optimizer)
                nn_utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += val
            lr_now = optimizer.param_groups[0]["lr"]
            bar.set_postfix(loss=f"{val:.4f}", lr=f"{lr_now:.2e}")
            wandb.log({
                "train_batch_loss": val,
                "learning_rate":    lr_now,
                "alpha":            criterion.alpha,
                "entity_penalty":   criterion.entity_penalty,
                "epoch":            epoch + 1,
            })

        avg_train_loss = total_train_loss / max(len(train_dl), 1)

        # ── Eval ─────────────────────────────────────────────────────────────
        model.eval()
        total_eval_loss = 0.0
        all_preds, all_refs = [], []

        bar_e = tqdm(eval_dl, desc=f"E{epoch + 1} Eval", unit="batch")
        with torch.no_grad():
            for batch_idx, batch in enumerate(bar_e):
                inp    = batch["input_ids"].to(device)
                adj_m  = batch["adj_matrix"].to(device)
                a_mask = batch["action_mask"].to(device)
                d_mask = batch["dial_mask"].to(device)
                e_mask = batch["ent_mask"].to(device)
                h_mask = batch["head_mask"].to(device)
                c_adj  = batch["causal_adj"].to(device)
                cs_adj = batch["char_state_adj"].to(device)
                idf_w  = batch["idf_weights"].to(device)
                tgt    = batch["target_ids"].to(device)
                trips  = batch["triplets"]

                if not ABLATION["use_causal_graph"]:
                    c_adj  = torch.zeros_like(c_adj)
                if not ABLATION["use_char_state_graph"]:
                    cs_adj = torch.zeros_like(cs_adj)
                if ABLATION["single_binary_graph"]:
                    binary = ((c_adj + cs_adj) > 0).float()
                    c_adj = cs_adj = binary

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, enc_mem, labels, _, gated_causal = model(
                        inp, adj_m, a_mask, d_mask, e_mask, h_mask,
                        c_adj, cs_adj, tgt, trips, idf_w
                    )
                    logits  = logits.float()
                    e_loss  = criterion(
                        log_probs=logits.view(-1, logits.size(-1)),
                        targets=labels.view(-1),
                        triplets=trips,
                        hidden_states=enc_mem,
                        head_weight=model.head.weight,
    
                        causal_adj=gated_causal,
                    )

                total_eval_loss += e_loss.item()
                bar_e.set_postfix(eval_loss=f"{e_loss.item():.4f}")

                # ── Beam-search generation (every 10th batch) ────────────────
                if batch_idx % 10 == 0:
                    aligned_mem, _ = model(
                        inp, adj_m, a_mask, d_mask, e_mask, h_mask,
                        c_adj, cs_adj,
                        target_ids=None, triplets=None, idf_weights=idf_w,
                    )
                    mem_pad_mask  = (inp[:, :, 0] == 1)
                    enc_attn_mask = (~mem_pad_mask).long()

                    pred = generate_summary(
                        model, aligned_mem, enc_attn_mask, tokenizer, device,
                        max_new_tokens=200, beam_size=4,
                    )
                    ref  = tokenizer.decode(
                        tgt[0, 0].cpu().tolist(), skip_special_tokens=True
                    )
                    all_preds.append(pred)
                    all_refs.append(ref)

                    wandb.log({
                        "sample_pred": wandb.Html(
                            f"<b>Pred:</b> {pred}<br><b>Ref:</b> {ref}"
                        ),
                        "epoch": epoch + 1,
                    })

                    if batch_idx % 50 == 0:
                        log_character_attention_map(model, enc_mem, c_adj)
                        log_character_attention_map_labeled(
                            model, enc_mem, c_adj, trips
                        )

        avg_eval_loss = total_eval_loss / max(len(eval_dl), 1)

        # ── ROUGE ────────────────────────────────────────────────────────────
        rouge_scores = {}
        bs_f1 = met_score = ent_f1 = 0.0
        if all_preds:
            rouge_scores = rouge.compute(
                predictions=all_preds, references=all_refs, use_stemmer=True,
            )
            # N1: BERTScore (deberta-xlarge-mnli is slow; use roberta-large)
            bs_out   = bertscore.compute(
                predictions=all_preds, references=all_refs,
                lang="en", model_type="roberta-large",
            )
            bs_f1    = sum(bs_out["f1"]) / max(len(bs_out["f1"]), 1)
            # N1: METEOR
            met_score = meteor.compute(
                predictions=all_preds, references=all_refs,
            )["meteor"]
            # N2: entity F1
            ent_f1 = _entity_f1(all_preds, all_refs)

        r1 = rouge_scores.get("rouge1", 0.0)
        r2 = rouge_scores.get("rouge2", 0.0)
        rL = rouge_scores.get("rougeL", 0.0)

        print(f"Epoch {epoch + 1} | "
              f"Train {avg_train_loss:.4f} | Eval {avg_eval_loss:.4f} | "
              f"R1 {r1:.4f} | R2 {r2:.4f} | RL {rL:.4f} | "
              f"BS-F1 {bs_f1:.4f} | METEOR {met_score:.4f} | EntF1 {ent_f1:.4f}")

        wandb.log({
            "epoch_train_loss": avg_train_loss,
            "epoch_eval_loss":  avg_eval_loss,
            "rouge1":           r1,
            "rouge2":           r2,
            "rougeL":           rL,
            "bertscore_f1":     bs_f1,
            "meteor":           met_score,
            "entity_f1":        ent_f1,
            "epoch":            epoch + 1,
        })

        # ── Save checkpoint ──────────────────────────────────────────────────
        save_dir  = "/tmp/uday/checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        ckpt_save = f"{save_dir}/gramformer_v2_epoch_{epoch + 1}.pt"
        torch.save({
            "epoch":               epoch + 1,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss":          avg_train_loss,
            "eval_loss":           avg_eval_loss,
            "rouge1":              r1,
            "rouge2":              r2,
        }, ckpt_save)
        # Update "latest" pointer
        torch.save({
            "epoch":               epoch + 1,
            "model_state_dict":    model.state_dict(),
        }, ckpt_path)
        print(f"✅ Checkpoint saved → {ckpt_save}")

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    train()
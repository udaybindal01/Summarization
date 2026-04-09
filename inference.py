"""
inference.py — GraM-Former v2 Inference
========================================
Loads a checkpoint, reads a test movie from the JSONL data, builds the three
typed movie-level graphs (causal, character-state, character co-occurrence),
and generates a summary using beam search.

Graph construction mirrors MovieGraphDatasetV2 exactly so inference is
consistent with training.
"""

import torch
import torch.nn.functional as F
import json
import gzip
import os
import math
from collections import defaultdict
from transformers import AutoTokenizer

from sum import GraMFormerV2
from train import movie_collate_fn
from peft import LoraConfig, get_peft_model

# Must match the tokenizer used during extraction and training
_TOKENIZER_NAME = (
    "/tmp/uday/bart-large"
    if os.path.isdir("/tmp/uday/bart-large")
    else "facebook/bart-large"
)

_STOP_ENTITIES = frozenset({
    "man", "woman", "him", "her", "he", "she", "they", "them",
    "it", "one", "two", "other", "others", "that", "this",
})


# =============================================================================
# Graph construction (mirrors MovieGraphDatasetV2 exactly)
# =============================================================================

def _canonical_entities(triplet_list, field):
    """Return lowercase canonical entity set from SVO triplets."""
    ents = set()
    for t in triplet_list:
        parts = t.split("_")
        if field == "subj" and len(parts) >= 1:
            raw = parts[0].replace("NOT ", "").strip().lower()
        elif field == "obj" and len(parts) >= 3:
            raw = parts[2].strip().lower()
        else:
            continue
        if raw and raw.isalpha() and len(raw) > 1 and raw not in _STOP_ENTITIES:
            ents.add(raw)
    return ents


def _scene_chars(scene):
    """Canonical character set: speaker tags (if present) + SVO subjects."""
    chars = set()
    for c in scene.get("characters", []):
        key = c.strip().lower()
        if key and key.isalpha() and len(key) > 1:
            chars.add(key)
    chars |= _canonical_entities(scene.get("graph_triplets", []), "subj")
    return chars


def build_causal_graph(scenes, max_scenes):
    """
    Directed causal graph: edge i→j when a canonical object entity of scene i
    is a subject entity of scene j.  Self-loops on diagonal.
    """
    num_scenes = len(scenes)
    causal_adj = torch.zeros((max_scenes, max_scenes))
    causal_adj[:num_scenes, :num_scenes].fill_diagonal_(1.0)

    scene_obj_sets  = [_canonical_entities(s.get("graph_triplets", []), "obj")  for s in scenes]
    scene_subj_sets = [_canonical_entities(s.get("graph_triplets", []), "subj") for s in scenes]

    for i in range(num_scenes):
        if not scene_obj_sets[i]:
            continue
        for j in range(i + 1, num_scenes):
            if scene_obj_sets[i] & scene_subj_sets[j]:
                causal_adj[i, j] = 1.0
                causal_adj[j, i] = 0.3
    return causal_adj


def build_char_state_graph(scenes, max_scenes):
    """
    Character state graph: edge weight = mean absolute emotion-polarity
    change for shared characters between scenes.
    """
    num_scenes     = len(scenes)
    char_state_adj = torch.zeros((max_scenes, max_scenes))
    char_state_adj[:num_scenes, :num_scenes].fill_diagonal_(1.0)

    emotions = [s.get("character_emotions", {}) for s in scenes]
    for i in range(num_scenes):
        for j in range(i + 1, num_scenes):
            shared = set(emotions[i].keys()) & set(emotions[j].keys())
            if not shared:
                continue
            changes = [abs(emotions[i][c] - emotions[j][c]) for c in shared]
            w = sum(changes) / len(changes)
            char_state_adj[i, j] = w
            char_state_adj[j, i] = w
    return char_state_adj


def build_char_cooccur_graph(scenes, max_scenes):
    """
    Character co-occurrence graph: edge weight = Jaccard similarity of
    canonical character sets between scenes.
    """
    num_scenes       = len(scenes)
    char_cooccur_adj = torch.zeros((max_scenes, max_scenes))
    char_cooccur_adj[:num_scenes, :num_scenes].fill_diagonal_(1.0)

    scene_char_sets = [_scene_chars(s) for s in scenes]
    for i in range(num_scenes):
        if not scene_char_sets[i]:
            continue
        for j in range(i + 1, num_scenes):
            union = scene_char_sets[i] | scene_char_sets[j]
            if not union:
                continue
            jaccard = len(scene_char_sets[i] & scene_char_sets[j]) / len(union)
            if jaccard > 0:
                char_cooccur_adj[i, j] = jaccard
                char_cooccur_adj[j, i] = jaccard
    return char_cooccur_adj


def build_idf_weights(scenes, max_scenes):
    """IDF-weighted entity co-occurrence matrix (same as MovieGraphDatasetV2)."""
    num_scenes   = len(scenes)
    idf_weights  = torch.zeros((max_scenes, max_scenes))
    all_entities = defaultdict(int)
    scene_ent_sets = []
    for s in scenes:
        ents = (
            _canonical_entities(s.get("graph_triplets", []), "subj") |
            _canonical_entities(s.get("graph_triplets", []), "obj")
        )
        scene_ent_sets.append(ents)
        for e in ents:
            all_entities[e] += 1

    N = max(num_scenes, 1)
    for i in range(num_scenes):
        for j in range(i + 1, num_scenes):
            shared = scene_ent_sets[i] & scene_ent_sets[j]
            if shared:
                avg_idf = sum(
                    math.log(N / max(all_entities[e], 1)) for e in shared
                ) / len(shared)
                idf_weights[i, j] = avg_idf
                idf_weights[j, i] = avg_idf
    return idf_weights, scene_ent_sets, all_entities


_MAX_ENTITIES = 50   # must match train.py MAX_ENTITIES


def build_entity_graphs(scenes, max_scenes, scene_ent_sets, all_entities):
    """
    Build entity_presence [max_scenes, E] and entity_entity_adj [E, E]
    for the Dynamic Knowledge Graph, mirroring MovieGraphDatasetV2 exactly.
    """
    num_scenes = len(scenes)
    E = _MAX_ENTITIES

    # Global entity registry: top-E by frequency
    ent_freq = sorted(all_entities.items(), key=lambda x: -x[1])
    entity_registry = [e for e, _ in ent_freq[:E]]
    ent_to_idx = {e: i for i, e in enumerate(entity_registry)}

    entity_presence = torch.zeros((max_scenes, E))
    for s in range(num_scenes):
        for ent in scene_ent_sets[s]:
            if ent in ent_to_idx:
                entity_presence[s, ent_to_idx[ent]] = 1.0

    entity_entity_adj = torch.zeros((E, E))
    for s in range(num_scenes):
        present = [ent_to_idx[e] for e in scene_ent_sets[s] if e in ent_to_idx]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                entity_entity_adj[present[i], present[j]] += 1.0
                entity_entity_adj[present[j], present[i]] += 1.0

    return entity_presence, entity_entity_adj


# =============================================================================
# Beam-search generation
# =============================================================================

@torch.no_grad()
def generate_summary(model, tokenizer, batch, device,
                     max_new_tokens=200, beam_size=4):
    """Beam-search generation using the GraMFormerV2 encoder + BART decoder."""
    print(f"Generating summary (beam_size={beam_size})...\n")

    b_input_ids = batch["input_ids"].to(device)
    b_act_mask  = batch["action_mask"].to(device)
    b_dial_mask = batch["dial_mask"].to(device)
    b_ent_mask  = batch["ent_mask"].to(device)
    b_head_mask = batch["head_mask"].to(device)
    b_causal    = batch["causal_adj"].to(device)
    b_cs_adj    = batch["char_state_adj"].to(device)
    b_cc_adj    = batch["char_cooccur_adj"].to(device)
    b_idf_w     = batch["idf_weights"].to(device)
    b_ent_pres  = batch["entity_presence"].to(device)   # [1, S, E]
    b_ent_adj   = batch["entity_entity_adj"].to(device) # [1, E, E]

    # Encode once — returns aligned_memory [1, S, D]
    aligned_memory, _ = model(
        b_input_ids, b_act_mask, b_dial_mask, b_ent_mask, b_head_mask,
        b_causal, b_cs_adj, b_cc_adj,
        target_ids=None, triplets=None, idf_weights=b_idf_w,
        entity_presence=b_ent_pres, entity_entity_adj=b_ent_adj,
    )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    mem_pad_mask  = (b_input_ids[:, :, 0] == pad_id)
    enc_attn_mask = (~mem_pad_mask).long()

    beams     = [(0.0, [tokenizer.bos_token_id or 0])]
    completed = []

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
            logits   = model.head(dec_out.last_hidden_state[:, -1, :]).float()
            log_prob = F.log_softmax(logits, dim=-1).squeeze(0)

            # No-repeat trigram penalty (use -1e4, not -inf, for bfloat16 safety)
            if len(tokens) >= 3:
                ngrams = {tuple(tokens[k:k + 3]) for k in range(len(tokens) - 2)}
                for ng in ngrams:
                    if len(ng) == 3:
                        log_prob[ng[-1]] = -1e4

            top_vals, top_idx = log_prob.topk(beam_size)
            for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
                new_beams.append((score + v, tokens + [idx]))

        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
        if not beams:
            break

    best = (
        max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
        if completed
        else max(beams, key=lambda x: x[0])
    )
    return tokenizer.decode(best[1], skip_special_tokens=True)


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CHECKPOINT_PATH = "/tmp/uday/checkpoints/gramformer_v2_latest.pt"
    MAX_SCENES      = 64   # must match train.py MAX_SCENES

    print("Loading BART tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)

    print("Initializing model architecture...")
    model = GraMFormerV2(
        vocab_size=len(tokenizer),
        d_model=1024,   # must match checkpoint (1024=bart-large, 768=bart-base)
        num_layers=4,
        tokenizer=tokenizer,
    ).to(device)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        if any("lora" in k for k in state_dict.keys()):
            print("Auto-detected LoRA weights — applying PEFT wrapper...")
            lora_config = LoraConfig(
                r=16, lora_alpha=32,
                target_modules=["in_proj", "x_proj", "out_proj", "dt_proj"],
                lora_dropout=0.05, bias="none",
            )
            model.encoder = get_peft_model(model.encoder, lora_config)

        model.load_state_dict(state_dict, strict=False)
        print("Model loaded.")
    else:
        print(f"Checkpoint not found at {CHECKPOINT_PATH} — using random weights.")

    model.eval()

    # ── Load one test movie ───────────────────────────────────────────────────
    eval_data_path = "/tmp/uday/mensa_test_data.jsonl.gz"
    print(f"\nLoading test movie from {eval_data_path}...")

    movie_scenes       = []
    target_movie_name  = None

    with gzip.open(eval_data_path, "rt", encoding="utf-8") as f:
        for line in f:
            scene_data    = json.loads(line)
            current_movie = scene_data["movie_id"].split("_Scene_")[0]

            if target_movie_name is None:
                target_movie_name = current_movie
            if current_movie != target_movie_name:
                break

            # Tensors for fields used by movie_collate_fn
            for k in ("input_ids", "target_ids"):
                scene_data[k] = torch.tensor(scene_data[k], dtype=torch.long)
            for k in ("action_mask", "dialogue_mask", "entity_mask", "header_mask"):
                if k in scene_data:
                    scene_data[k] = torch.tensor(scene_data[k], dtype=torch.bool)
                else:
                    scene_data[k] = torch.zeros(len(scene_data["input_ids"]), dtype=torch.bool)
            scene_data.setdefault("graph_triplets",     [])
            scene_data.setdefault("character_emotions", {})
            scene_data.setdefault("characters",         [])
            scene_data.setdefault("scene_meta",         {})
            movie_scenes.append(scene_data)

    # Stride-sample if the movie exceeds MAX_SCENES (mirrors training behaviour)
    if len(movie_scenes) > MAX_SCENES:
        step = len(movie_scenes) / MAX_SCENES
        movie_scenes = [movie_scenes[int(i * step)] for i in range(MAX_SCENES)]

    print(f"Loaded '{target_movie_name}' ({len(movie_scenes)} scenes).")

    # ── Build movie-level graphs (same logic as MovieGraphDatasetV2) ──────────
    causal_adj       = build_causal_graph(movie_scenes,     MAX_SCENES)
    char_state_adj   = build_char_state_graph(movie_scenes, MAX_SCENES)
    char_cooccur_adj = build_char_cooccur_graph(movie_scenes, MAX_SCENES)
    idf_weights, scene_ent_sets, all_entities = build_idf_weights(movie_scenes, MAX_SCENES)
    entity_presence, entity_entity_adj = build_entity_graphs(
        movie_scenes, MAX_SCENES, scene_ent_sets, all_entities
    )

    mock_item = {
        "movie_name":        target_movie_name,
        "scenes":            movie_scenes,
        "causal_adj":        causal_adj,
        "char_state_adj":    char_state_adj,
        "char_cooccur_adj":  char_cooccur_adj,
        "idf_weights":       idf_weights,
        "entity_presence":   entity_presence,    # [MAX_SCENES, E]
        "entity_entity_adj": entity_entity_adj,  # [E, E]
    }

    batch = movie_collate_fn([mock_item])

    # ── Generate ──────────────────────────────────────────────────────────────
    summary = generate_summary(model, tokenizer, batch, device)

    print("\n" + "=" * 80)
    print("GENERATED SUMMARY:")
    print("=" * 80)
    print(summary)
    print("=" * 80)


if __name__ == "__main__":
    main()

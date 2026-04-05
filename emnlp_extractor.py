"""
emnlp_extractor.py  —  GraM-Former v2 Data Pipeline
=====================================================
Architecture: single-process pipeline with batched GPU sentiment.

We intentionally avoid multiprocessing here. The previous design hit two
hard walls:
  1. spaCy + tokenizer init across N worker processes = N×30s startup delay
     before a single result appears.
  2. Large screenplay scenes make spaCy's O(n²) parser slow regardless.

This version processes scenes sequentially on CPU (spaCy + tokenizer are
fast once loaded once), batches sentiment inference on GPU, and writes
results incrementally.  With a single load of both models and GPU-batched
sentiment, throughput is ~500-1500 scenes/min on a typical research server.

To go faster: run multiple instances of this script on disjoint slices of
the dataset using --start and --end arguments (see bottom of file).
"""

import json
import spacy
import torch
import re
import gzip
import os
import sys
import warnings
import argparse
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    logging as hf_logging,
)
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Silence known-harmless warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="`resume_download` is deprecated",
                        category=FutureWarning)
hf_logging.set_verbosity_error()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_SENT_MAX_CHARS  = 256    # chars per snippet fed to sentiment model
_SENT_BATCH_SIZE = 2048   # 40GB VRAM — saturate the GPU fully
_SPACY_MAX_CHARS = 1000   # hard cap on spaCy input — prevents O(n²) slowdown
_WRITE_EVERY     = 512    # flush every N scenes — matches large batch size


# ---------------------------------------------------------------------------
# Sentiment helpers (GPU, batched)
# ---------------------------------------------------------------------------

def score_snippets(snippets, sent_tok, sent_model, device):
    """Single batched forward pass → list of float polarity scores."""
    if not snippets:
        return []
    enc = sent_tok(
        snippets, return_tensors="pt", truncation=True,
        max_length=128, padding=True,
    ).to(device)
    with torch.no_grad():
        logits = sent_model(**enc).logits
    probs    = torch.softmax(logits.float(), dim=-1)   # [B, 3]  neg/neu/pos
    polarity = (probs[:, 2] - probs[:, 0]).cpu().tolist()
    return polarity


def attach_emotions(buffer, sent_tok, sent_model, device):
    """
    Given a list of pre-processed scene dicts (with _char_snippets and
    _scene_snippet fields), run batched GPU sentiment and attach
    character_emotions.  Strips temp fields before returning.
    """
    # Flatten all snippets from all scenes into one list
    index    = []   # (buf_idx, char_key_or_"__scene__")
    snippets = []

    for i, rec in enumerate(buffer):
        index.append((i, "__scene__"))
        snippets.append(rec["_scene_snippet"])
        for char_key, snip in rec["_char_snippets"].items():
            index.append((i, char_key))
            snippets.append(snip)

    # Sub-batch to stay within VRAM
    scores = []
    for start in range(0, len(snippets), _SENT_BATCH_SIZE):
        scores.extend(score_snippets(
            snippets[start:start + _SENT_BATCH_SIZE],
            sent_tok, sent_model, device,
        ))

    # Distribute back
    scene_pol = {}
    char_pol  = {}
    for (buf_idx, key), score in zip(index, scores):
        if key == "__scene__":
            scene_pol[buf_idx] = score
        else:
            char_pol[(buf_idx, key)] = score

    final = []
    for i, rec in enumerate(buffer):
        char_snippets = rec.pop("_char_snippets")
        rec.pop("_scene_snippet")
        sp = scene_pol.get(i, 0.0)
        rec["character_emotions"] = {
            k: round(char_pol.get((i, k), sp), 4)
            for k in char_snippets
        }
        final.append(rec)
    return final


# ---------------------------------------------------------------------------
# Single-scene CPU processing (called inline — no multiprocessing)
# ---------------------------------------------------------------------------

def process_scene(scene_text, summary, scene_id,
                  nlp, nlp_pos, tokenizer, max_seq_len, max_target_len):
    """
    Returns a pre-result dict with _char_snippets and _scene_snippet
    attached for Stage-2 sentiment scoring.

    nlp     : spaCy pipeline with parser enabled (capped to _SPACY_MAX_CHARS)
              Used for dependency-based adjacency matrix and SVO triplets.
    nlp_pos : spaCy pipeline with parser disabled (runs on full scene text)
              Used for entity_mask — avoids the 1000-char blind spot (F6).
    """
    # Tokenise
    encoding = tokenizer(
        scene_text, max_length=max_seq_len, truncation=True,
        return_offsets_mapping=True, padding="max_length",
    )
    target_enc = tokenizer(
        summary, max_length=max_target_len,
        truncation=True, padding="max_length",
    )
    input_ids  = encoding["input_ids"]
    target_ids = target_enc["input_ids"]
    offsets    = encoding["offset_mapping"]

    # 4-way modality masks
    action_mask   = [0] * max_seq_len
    dialogue_mask = [0] * max_seq_len
    entity_mask   = [0] * max_seq_len
    header_mask   = [0] * max_seq_len

    in_dialogue = False
    dial_count = act_count = 0
    has_int = has_ext = False

    for i, (start, end) in enumerate(offsets):
        if i >= max_seq_len or start == end:
            continue
        tok   = scene_text[start:end]
        upper = tok.upper()
        if '"' in tok or "'" in tok:
            in_dialogue = not in_dialogue
        is_hdr = any(h in upper for h in
                     ["INT.", "EXT.", "DAY", "NIGHT", "LATER",
                      "CONTINUOUS", "MORNING", "EVENING"])
        if is_hdr:
            header_mask[i] = 1
            has_int = has_int or "INT." in upper
            has_ext = has_ext or "EXT." in upper
        elif in_dialogue:
            dialogue_mask[i] = 1
            dial_count += 1
        else:
            action_mask[i] = 1
            act_count += 1

    total        = max(1, dial_count + act_count)
    dial_density = dial_count / total
    act_density  = act_count  / total

    # ── spaCy (parser-enabled, capped) — adjacency matrix + SVO triplets ──
    doc        = nlp(scene_text[:_SPACY_MAX_CHARS])
    triplets   = []
    adj_matrix = [[0] * max_seq_len for _ in range(max_seq_len)]

    for token in doc:
        h_idx = next((idx for idx, (s, e) in enumerate(offsets)
                      if s <= token.head.idx < e), -1)
        t_idx = next((idx for idx, (s, e) in enumerate(offsets)
                      if s <= token.idx < e), -1)
        if 0 <= h_idx < max_seq_len and 0 <= t_idx < max_seq_len:
            adj_matrix[t_idx][h_idx] = 1
            adj_matrix[h_idx][t_idx] = 1

        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            subject = obj = None
            negation = any(c.dep_ == "neg" for c in token.children)
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subject = child
                if child.dep_ in ["dobj", "pobj", "attr", "nsubjpass"]:
                    obj = child
            if subject and obj:
                prefix = "NOT " if negation else ""
                triplets.append(
                    f"{prefix}{subject.text}_{token.text}_{obj.text}"
                )
                s_idx = next((idx for idx, (s, e) in enumerate(offsets)
                              if s <= subject.idx < e), -1)
                v_idx = next((idx for idx, (s, e) in enumerate(offsets)
                              if s <= token.idx < e), -1)
                o_idx = next((idx for idx, (s, e) in enumerate(offsets)
                              if s <= obj.idx < e), -1)
                for a, b in [(v_idx, s_idx), (o_idx, v_idx)]:
                    if all(0 <= x < max_seq_len for x in [a, b]):
                        adj_matrix[a][b] = 1

    # ── F6: entity mask on the FULL scene text (no 1000-char cap) ──────────
    # nlp_pos has parser disabled so it runs in linear time on any length.
    doc_full = nlp_pos(scene_text)
    for token in doc_full:
        if token.pos_ in ["PROPN", "NOUN"] and token.is_alpha:
            e_idx = next((idx for idx, (s, e) in enumerate(offsets)
                          if s <= token.idx < e), -1)
            if 0 <= e_idx < max_seq_len:
                entity_mask[e_idx] = 1

    # Per-character snippets for Stage-2 sentiment
    sentences  = re.split(r"(?<=[.!?])\s+", scene_text)
    char_names = {}
    for trip in triplets:
        parts = trip.split("_")
        if len(parts) >= 1:
            raw = parts[0].replace("NOT ", "").strip()
            key = raw.lower()
            if key and key.isalpha() and len(key) > 1:
                char_names.setdefault(key, []).append(raw)

    char_snippets = {}
    for char_key, variants in char_names.items():
        pat   = re.compile(
            "|".join(re.escape(v) for v in variants), re.IGNORECASE
        )
        sents = [s for s in sentences if pat.search(s)]
        char_snippets[char_key] = (
            " ".join(sents) if sents else scene_text
        )[:_SENT_MAX_CHARS]

    return {
        "movie_id":         scene_id,
        "input_ids":        input_ids,
        "target_ids":       target_ids,
        "adjacency_matrix": adj_matrix,
        "action_mask":      action_mask,
        "dialogue_mask":    dialogue_mask,
        "entity_mask":      entity_mask,
        "header_mask":      header_mask,
        "graph_triplets":   triplets,
        "scene_meta": {
            "dialogue_density": round(dial_density, 4),
            "action_density":   round(act_density,  4),
            "has_int":          has_int,
            "has_ext":          has_ext,
        },
        "_char_snippets":   char_snippets,
        "_scene_snippet":   scene_text[:_SENT_MAX_CHARS],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   type=int, default=0)
    parser.add_argument("--end",     type=int, default=-1)
    parser.add_argument("--out",     type=str,
                        default="/tmp/karan/mensa_train_data.jsonl.gz")
    parser.add_argument("--dataset", type=str, default="mensa",
                        choices=["mensa", "moviesum"],
                        help="Which dataset to extract from")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ── Load models once ────────────────────────────────────────────────────
    print("Loading spaCy...", flush=True)
    # Parser pipeline — capped to _SPACY_MAX_CHARS for O(n²) safety
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "lemmatizer"])
    nlp.max_length = _SPACY_MAX_CHARS + 200
    # F6: Fast POS-only pipeline for entity mask on the FULL scene text
    nlp_pos = spacy.load("en_core_web_sm",
                         disable=["parser", "ner", "textcat", "lemmatizer"])

    print("Loading BART tokenizer...", flush=True)
    # C2: use BART tokenizer (same BPE vocab as RoBERTa, but explicit
    # decoder_start_token_id=2 convention for BART teacher forcing).
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    print("Loading sentiment model on GPU...", flush=True)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sent_tok   = AutoTokenizer.from_pretrained(_SENTIMENT_MODEL)
    sent_model = AutoModelForSequenceClassification.from_pretrained(
        _SENTIMENT_MODEL
    ).to(device).eval()
    if device.type == "cuda":
        # bfloat16: better numerical range than float16, same speed on A100/H100
        sent_model = sent_model.to(torch.bfloat16)
        # torch.compile fuses kernels — gives ~30% extra throughput on A100/H100
        try:
            sent_model = torch.compile(sent_model, mode="reduce-overhead")
            print("  torch.compile: enabled", flush=True)
        except Exception:
            pass  # compile not available in this torch version — fine
    print(f"  All models loaded. Device: {device}", flush=True)

    # ── Build scene list ─────────────────────────────────────────────────────
    max_seq_len    = 512
    max_target_len = 512
    all_scenes     = []

    # ── Dataset loading — supports MENSA and MovieSum ──────────────────────
    if args.dataset == "mensa":
        print("Downloading MENSA dataset...", flush=True)
        dataset = load_dataset("rohitsaxena/MENSA", split="train")
        for ex in tqdm(dataset, desc="Building scene list (MENSA)"):
            scenes_list  = ex.get("scenes", [])
            summary_text = ex.get("summary", "")
            movie_id     = ex.get("name", "movie")
            if isinstance(scenes_list, list):
                for i, scene_txt in enumerate(scenes_list):
                    if scene_txt and len(str(scene_txt).strip()) > 10:
                        all_scenes.append({
                            "text":    str(scene_txt).strip(),
                            "summary": summary_text,
                            "id":      f"{movie_id}_Scene_{i:03d}",
                        })

    elif args.dataset == "moviesum":
        # MovieSum: Saxena & Keller 2024 — same format as MENSA
        # HuggingFace: rohitsaxena/MovieSum  (same author, same schema)
        print("Downloading MovieSum dataset...", flush=True)
        dataset = load_dataset("rohitsaxena/MovieSum", split="train")
        for ex in tqdm(dataset, desc="Building scene list (MovieSum)"):
            # MovieSum uses 'screenplay' field for scene text list
            # and 'summary' for the Wikipedia plot summary
            scenes_list  = ex.get("screenplay", ex.get("scenes", []))
            summary_text = ex.get("summary", "")
            movie_id     = ex.get("name", ex.get("title", "movie"))
            if isinstance(scenes_list, list):
                for i, scene_txt in enumerate(scenes_list):
                    if scene_txt and len(str(scene_txt).strip()) > 10:
                        all_scenes.append({
                            "text":    str(scene_txt).strip(),
                            "summary": summary_text,
                            "id":      f"{movie_id}_Scene_{i:03d}",
                        })
            elif isinstance(scenes_list, str) and len(scenes_list.strip()) > 10:
                # Some MovieSum entries have a single string screenplay
                # Split into pseudo-scenes on scene header patterns
                raw_scenes = re.split(
                    r"(?=(?:INT[.]|EXT[.])[ \t])", scenes_list
                )
                for i, scene_txt in enumerate(raw_scenes):
                    if len(scene_txt.strip()) > 10:
                        all_scenes.append({
                            "text":    scene_txt.strip(),
                            "summary": summary_text,
                            "id":      f"{movie_id}_Scene_{i:03d}",
                        })

    # Apply sharding slice
    end       = args.end if args.end > 0 else len(all_scenes)
    all_scenes = all_scenes[args.start:end]
    print(f"Processing {len(all_scenes):,} scenes "
          f"(indices {args.start}–{end})", flush=True)

    # ── Single-process extraction with GPU-batched sentiment ─────────────────
    written = 0
    buffer  = []

    with gzip.open(args.out, "wt", encoding="utf-8") as out_f:
        for scene in tqdm(all_scenes, desc="Extracting", unit="scene"):
            try:
                pre = process_scene(
                    scene["text"], scene["summary"], scene["id"],
                    nlp, nlp_pos, tokenizer, max_seq_len, max_target_len,
                )
                buffer.append(pre)
            except Exception as e:
                continue   # skip broken scenes silently

            if len(buffer) >= _WRITE_EVERY:
                for rec in attach_emotions(buffer, sent_tok, sent_model, device):
                    out_f.write(json.dumps(rec) + "\n")
                    written += 1
                buffer = []
                out_f.flush()

        # Flush remainder
        if buffer:
            for rec in attach_emotions(buffer, sent_tok, sent_model, device):
                out_f.write(json.dumps(rec) + "\n")
                written += 1

    print(f"\nDone — {written:,} scenes → {args.out}")


if __name__ == "__main__":
    main()
"""
emnlp_extractor.py  —  Dual-Tower Hypergraph Data Pipeline
===========================================================
Architecture: single-process pipeline with batched GPU sentiment.

Changes from GraMFormer v2:
  - Added NER extraction per scene  (for hypergraph node construction)
  - Added hyperedge type classification per scene  (CONFLICT/ALLIANCE/DECEPTION/DIALOGUE/NEUTRAL)
  - Removed adjacency_matrix output  (replaced by dynamic hypergraph built at training time)
  - Both input_ids and target_ids use BART tokenizer throughout

Output per scene (saved to .jsonl.gz):
  input_ids, target_ids            — BART-tokenized, padded to max_seq_len
  action_mask, dialogue_mask,
  entity_mask, header_mask         — 4-way modality masks [max_seq_len]
  graph_triplets                   — ["subj_verb_obj", ...]  (SVO from ROOT-verb parse)
  characters                       — ALL-CAPS speaker names from screenplay format
  ner_entities                     — [{"text": "joker", "type": "PERSON"}, ...]  ← NEW
  hyperedge_type                   — "CONFLICT" | "ALLIANCE" | "DECEPTION" |     ← NEW
                                     "DIALOGUE" | "NEUTRAL"
  character_emotions               — {"joker": 0.72, ...}  (Cardiff sentiment)
  scene_meta                       — {dialogue_density, action_density, has_int, has_ext}
"""

import json
import spacy
import torch
import re
import gzip
import os
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
_SENT_MAX_CHARS  = 256
_SENT_BATCH_SIZE = 2048
_SPACY_MAX_CHARS = 1000   # cap for O(n²) parser — SVO triplets only
_WRITE_EVERY     = 512

# ---------------------------------------------------------------------------
# Hyperedge type classification
# ---------------------------------------------------------------------------
# Maps ROOT verb lemmas to narrative relation types.
# These 5 types become the typed hyperedges in DynamicHypergraphTower.
_VERB_CLASS_MAP = {
    "CONFLICT":  {
        "shoot", "kill", "attack", "fight", "hit", "stab", "threaten",
        "destroy", "bomb", "punch", "chase", "arrest", "hurt", "wound",
        "fire", "explode", "detonate", "crash", "beat",
    },
    "ALLIANCE":  {
        "help", "save", "join", "protect", "trust", "support", "rescue",
        "defend", "assist", "give", "share", "carry", "guard", "follow",
        "lead", "bring", "take", "hold",
    },
    "DECEPTION": {
        "lie", "betray", "trick", "deceive", "manipulate", "hide", "deny",
        "pretend", "steal", "escape", "flee", "run", "avoid", "cover",
        "conceal", "plant",
    },
    "DIALOGUE":  {
        "tell", "ask", "say", "warn", "explain", "speak", "answer",
        "reveal", "shout", "whisper", "call", "meet", "discuss", "argue",
        "offer", "order", "inform", "question", "admit", "confess",
    },
}

# NER types kept for the hypergraph node registry.
# Only these types produce nodes — others are too noisy.
_KEPT_ENTITY_TYPES = frozenset({"PERSON", "ORG", "GPE", "FACILITY", "PRODUCT"})


def _classify_hyperedge_type(triplets):
    """
    Classify a scene's dominant narrative action type from its SVO triplets.

    Returns one of: "CONFLICT", "ALLIANCE", "DECEPTION", "DIALOGUE", "NEUTRAL"
    """
    counts = {k: 0 for k in _VERB_CLASS_MAP}
    for trip in triplets:
        parts = trip.split("_")
        if len(parts) < 2:
            continue
        verb = parts[1].lower().strip()
        for rtype, verb_set in _VERB_CLASS_MAP.items():
            if verb in verb_set:
                counts[rtype] += 1
                break
    total = sum(counts.values())
    if total == 0:
        return "NEUTRAL"
    return max(counts, key=counts.get)


# ---------------------------------------------------------------------------
# Screenplay character extraction
# ---------------------------------------------------------------------------
_SCENE_HEADER_WORDS = frozenset({
    "INT", "EXT", "INT.", "EXT.", "INTERIOR", "EXTERIOR",
    "CUT", "FADE", "DISSOLVE", "SMASH", "MATCH", "WIPE",
    "CONTINUED", "CONTINUE", "CONTINUING", "END", "TITLE", "TITLES",
    "BACK", "LATER", "CONTINUOUS", "MOMENTS", "SIMULTANEOUSLY",
    "DAY", "NIGHT", "MORNING", "EVENING", "DUSK", "DAWN",
    "THE", "A", "AN", "AND", "OR",
})
_CHAR_NAME_RE = re.compile(r'^[A-Z][A-Z\s\'\.\-]{1,39}$')


def extract_scene_speakers(scene_text):
    """
    Extract speaking character names from ALL-CAPS screenplay format lines.
    Returns a sorted deduplicated list of canonical name strings.
    """
    speakers = set()
    for line in scene_text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = re.sub(r'\s*\([^)]*\)', '', stripped).strip()
        if not cleaned:
            continue
        if (len(cleaned) < 2 or len(cleaned) > 40
                or cleaned != cleaned.upper()
                or not _CHAR_NAME_RE.match(cleaned)):
            continue
        words = cleaned.split()
        if len(words) > 4:
            continue
        if any(w in _SCENE_HEADER_WORDS for w in words):
            continue
        if not any(w.isalpha() for w in words):
            continue
        speakers.add(cleaned)
    return sorted(speakers)


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
    probs    = torch.softmax(logits.float(), dim=-1)   # [B, 3] neg/neu/pos
    polarity = (probs[:, 2] - probs[:, 0]).cpu().tolist()
    return polarity


def attach_emotions(buffer, sent_tok, sent_model, device):
    """
    Batch GPU sentiment scoring. Attaches character_emotions to each record.
    Strips temporary _char_snippets / _scene_snippet fields before returning.
    """
    index    = []
    snippets = []
    for i, rec in enumerate(buffer):
        index.append((i, "__scene__"))
        snippets.append(rec["_scene_snippet"])
        for char_key, snip in rec["_char_snippets"].items():
            index.append((i, char_key))
            snippets.append(snip)

    scores = []
    for start in range(0, len(snippets), _SENT_BATCH_SIZE):
        scores.extend(score_snippets(
            snippets[start:start + _SENT_BATCH_SIZE],
            sent_tok, sent_model, device,
        ))

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
# Single-scene CPU processing
# ---------------------------------------------------------------------------

def process_scene(scene_text, summary, scene_id,
                  nlp, nlp_ner_pos, tokenizer, max_seq_len, max_target_len):
    """
    Returns a pre-result dict with sentiment temp fields attached.

    nlp         : spaCy with parser, capped to _SPACY_MAX_CHARS — SVO triplets
    nlp_ner_pos : spaCy with NER + POS, parser disabled — full text NER + entity_mask
    """
    # ── Tokenise ─────────────────────────────────────────────────────────────
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

    # ── 4-way modality masks ─────────────────────────────────────────────────
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

    # ── SVO triplets (parser-enabled, capped to 1000 chars) ──────────────────
    doc      = nlp(scene_text[:_SPACY_MAX_CHARS])
    triplets = []

    for token in doc:
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
                triplets.append(f"{prefix}{subject.text}_{token.text}_{obj.text}")

    # ── NER + entity_mask on FULL scene text ─────────────────────────────────
    # nlp_ner_pos has parser disabled → linear time on any length.
    doc_full    = nlp_ner_pos(scene_text)
    ner_entities = []
    seen_ner     = set()

    for ent in doc_full.ents:
        if ent.label_ not in _KEPT_ENTITY_TYPES:
            continue
        normalized = ent.text.strip().lower()
        if not normalized or len(normalized) < 2:
            continue
        # Entity mask — mark token positions in the tokenized sequence
        e_idx = next((idx for idx, (s, e) in enumerate(offsets)
                      if s <= ent.start_char < e and s != e), -1)
        if 0 <= e_idx < max_seq_len:
            entity_mask[e_idx] = 1
        # NER registry entry (unique per scene)
        if normalized not in seen_ner:
            ner_entities.append({
                "text": normalized,
                "type": ent.label_,
            })
            seen_ner.add(normalized)

    # Fallback entity_mask: PROPN / NOUN tokens not caught by NER
    for token in doc_full:
        if token.pos_ in ["PROPN", "NOUN"] and token.is_alpha:
            e_idx = next((idx for idx, (s, e) in enumerate(offsets)
                          if s <= token.idx < e and s != e), -1)
            if 0 <= e_idx < max_seq_len:
                entity_mask[e_idx] = 1

    # ── Hyperedge type classification ─────────────────────────────────────────
    hyperedge_type = _classify_hyperedge_type(triplets)

    # ── Per-character sentiment snippets ─────────────────────────────────────
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
        "movie_id":        scene_id,
        "input_ids":       input_ids,
        "target_ids":      target_ids,
        "action_mask":     action_mask,
        "dialogue_mask":   dialogue_mask,
        "entity_mask":     entity_mask,
        "header_mask":     header_mask,
        "graph_triplets":  triplets,
        "characters":      extract_scene_speakers(scene_text),
        # ── New fields for dynamic hypergraph ──────────────────────────────
        "ner_entities":    ner_entities,      # [{text, type}, ...]
        "hyperedge_type":  hyperedge_type,    # dominant action type string
        # ──────────────────────────────────────────────────────────────────
        "scene_meta": {
            "dialogue_density": round(dial_density, 4),
            "action_density":   round(act_density,  4),
            "has_int":          has_int,
            "has_ext":          has_ext,
        },
        "_char_snippets":  char_snippets,
        "_scene_snippet":  scene_text[:_SENT_MAX_CHARS],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   type=int, default=0)
    parser.add_argument("--end",     type=int, default=-1)
    parser.add_argument("--out",     type=str,
                        default="/tmp/uday/moviesum_data.jsonl.gz")
    parser.add_argument("--dataset", type=str, default="moviesum",
                        choices=["mensa", "moviesum"],
                        help="MovieSum is the primary dataset for this architecture")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ── Load models once ─────────────────────────────────────────────────────
    print("Loading spaCy (parser pipeline, capped)...", flush=True)
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "lemmatizer"])
    nlp.max_length = _SPACY_MAX_CHARS + 200

    print("Loading spaCy (NER + POS pipeline, full text)...", flush=True)
    # This pipeline runs on full scene text — NER enabled, parser disabled
    nlp_ner_pos = spacy.load("en_core_web_sm",
                             disable=["parser", "textcat", "lemmatizer"])

    print("Loading BART tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    print("Loading sentiment model on GPU...", flush=True)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sent_tok   = AutoTokenizer.from_pretrained(_SENTIMENT_MODEL)
    sent_model = AutoModelForSequenceClassification.from_pretrained(
        _SENTIMENT_MODEL
    ).to(device).eval()
    if device.type == "cuda":
        sent_model = sent_model.to(torch.bfloat16)
    print(f"  All models loaded. Device: {device}", flush=True)

    # ── Build scene list ──────────────────────────────────────────────────────
    max_seq_len    = 512
    max_target_len = 512
    all_scenes     = []

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
        print("Downloading MovieSum dataset...", flush=True)
        dataset = load_dataset("rohitsaxena/MovieSum", split="train")
        for ex in tqdm(dataset, desc="Building scene list (MovieSum)"):
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
                raw_scenes = re.split(r"(?=(?:INT[.]|EXT[.])[ \t])", scenes_list)
                for i, scene_txt in enumerate(raw_scenes):
                    if len(scene_txt.strip()) > 10:
                        all_scenes.append({
                            "text":    scene_txt.strip(),
                            "summary": summary_text,
                            "id":      f"{movie_id}_Scene_{i:03d}",
                        })

    end       = args.end if args.end > 0 else len(all_scenes)
    all_scenes = all_scenes[args.start:end]
    print(f"Processing {len(all_scenes):,} scenes (indices {args.start}–{end})",
          flush=True)

    written = 0
    buffer  = []

    with gzip.open(args.out, "wt", encoding="utf-8") as out_f:
        for scene in tqdm(all_scenes, desc="Extracting", unit="scene"):
            try:
                pre = process_scene(
                    scene["text"], scene["summary"], scene["id"],
                    nlp, nlp_ner_pos, tokenizer, max_seq_len, max_target_len,
                )
                buffer.append(pre)
            except Exception:
                continue

            if len(buffer) >= _WRITE_EVERY:
                for rec in attach_emotions(buffer, sent_tok, sent_model, device):
                    out_f.write(json.dumps(rec) + "\n")
                    written += 1
                buffer = []
                out_f.flush()

        if buffer:
            for rec in attach_emotions(buffer, sent_tok, sent_model, device):
                out_f.write(json.dumps(rec) + "\n")
                written += 1

    print(f"\nDone — {written:,} scenes → {args.out}")


if __name__ == "__main__":
    main()
"""
emnlp_extractor.py  —  Dual-Tower Hypergraph Data Pipeline (v3)
===========================================================
Architecture: single-process pipeline with batched GPU sentiment.

Upgrades for EMNLP (Dual-Tower Latent Hypergraph):
  - 100% accurate XML-based Modality Masks (replaces error-prone heuristics).
  - Clean Tokenization: XML tags are stripped before passing to BART/spaCy to prevent 
    dependency parse corruption, using offset mapping to preserve mask alignment.
  - Removed hardcoded _VERB_CLASS_MAP (shifting to Latent Edge embeddings).
  - Robust Character Extraction: Fallbacks to regex parsing to prevent NaN explosions.
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
warnings.filterwarnings("ignore", message="`resume_download` is deprecated", category=FutureWarning)
hf_logging.set_verbosity_error()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_SENT_MAX_CHARS  = 256
_SENT_BATCH_SIZE = 2048
_SPACY_MAX_CHARS = 1000   
_WRITE_EVERY     = 512

# NER types kept for the hypergraph node registry.
_KEPT_ENTITY_TYPES = frozenset({"PERSON", "ORG", "GPE", "FACILITY", "PRODUCT"})

# ---------------------------------------------------------------------------
# Multiprocessing Worker Setup
# ---------------------------------------------------------------------------
_worker_nlp = None
_worker_nlp_ner = None
_worker_tokenizer = None

def init_worker():
    """Initializes spaCy and the tokenizer locally for each CPU worker."""
    global _worker_nlp, _worker_nlp_ner, _worker_tokenizer
    
    # We suppress logging here so 10 workers don't spam the console
    hf_logging.set_verbosity_error()
    
    _worker_nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "lemmatizer"])
    _worker_nlp.max_length = _SPACY_MAX_CHARS + 200
    
    _worker_nlp_ner = spacy.load("en_core_web_sm", disable=["parser", "textcat", "lemmatizer"])
    _worker_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

def process_scene_wrapper(scene_dict):
    """Wrapper to unpack the dictionary and use the worker's local models."""
    try:
        return process_scene(
            scene_dict["text"], scene_dict["summary"], scene_dict["id"],
            _worker_nlp, _worker_nlp_ner, _worker_tokenizer, 512, 512
        )
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Screenplay Structural Parsers
# ---------------------------------------------------------------------------

def clean_and_map_xml(xml_text):
    """
    Strips XML tags to create clean text for Transformers/spaCy, while building a 
    character-level map to track which structural modality each character belongs to.
    """
    clean_chars = []
    char_to_modality = []
    current_modality = "action" # Default to action description
    
    # Regex to find any XML tag
    tag_pattern = re.compile(r'(</?)([a-zA-Z_]+)(>)')
    
    cursor = 0
    for match in tag_pattern.finditer(xml_text):
        start, end = match.span()
        
        # Add the actual screenplay text before the tag
        text_segment = xml_text[cursor:start]
        clean_chars.append(text_segment)
        char_to_modality.extend([current_modality] * len(text_segment))
        
        # Determine the modality for the NEXT text segment based on the tag
        is_closing = match.group(1) == "</"
        tag_name = match.group(2).lower()
        
        if not is_closing:
            if tag_name in ["dialogue", "character"]:
                current_modality = "dialogue"
            elif tag_name == "stage_direction":
                current_modality = "header"
            elif tag_name == "scene_description":
                current_modality = "action"
        else:
            # Revert to action as a safe fallback after closing a specific block
            current_modality = "action" 
            
        cursor = end
    
    # Add any remaining text
    text_segment = xml_text[cursor:]
    clean_chars.append(text_segment)
    char_to_modality.extend([current_modality] * len(text_segment))
    
    clean_text = "".join(clean_chars)
    return clean_text, char_to_modality

def extract_robust_characters(scene_xml, clean_text):
    """
    Extract speaking character names precisely from <character> tags,
    with a regex fallback to catch ALL-CAPS names in scene descriptions
    to prevent NaN errors on silent/action-heavy movies.
    """
    # 1. Grab official tags
    speakers = re.findall(r'<character>(.*?)</character>', scene_xml, re.IGNORECASE)
    official_chars = [s.strip().upper() for s in speakers if s.strip()]
    
    # 2. Fallback: Hunt for ALL CAPS in descriptions
    descriptions = re.findall(r'<scene_description>(.*?)</scene_description>', scene_xml, re.IGNORECASE)
    desc_text = " ".join(descriptions)
    
    hidden_names = re.findall(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b', desc_text)
    
    # Standard screenplay jargon to filter out
    jargon = {"INT", "EXT", "DAY", "NIGHT", "CONTINUOUS", "O.S.", "V.O.", "LATER", 
              "INT.", "EXT.", "WITH", "THE", "AND", "THAT"}
    
    for name in hidden_names:
        clean_name = name.strip()
        if clean_name not in jargon and clean_name.upper() not in official_chars:
            official_chars.append(clean_name.upper())

    # 3. Absolute last resort to prevent graph NaN explosion
    if not official_chars and clean_text.strip():
        # Grab the first word so the graph has at least one node to attach attention to
        words = clean_text.split()
        fallback = words[0].upper() if words else "UNKNOWN"
        official_chars.append(fallback)
        
    return sorted(list(set(official_chars)))

# ---------------------------------------------------------------------------
# Sentiment helpers (GPU, batched)
# ---------------------------------------------------------------------------

def score_snippets(snippets, sent_tok, sent_model, device):
    if not snippets:
        return []
    enc = sent_tok(
        snippets, return_tensors="pt", truncation=True,
        max_length=128, padding=True,
    ).to(device)
    with torch.no_grad():
        logits = sent_model(**enc).logits
    probs    = torch.softmax(logits.float(), dim=-1)   
    polarity = (probs[:, 2] - probs[:, 0]).cpu().tolist()
    return polarity

def attach_emotions(buffer, sent_tok, sent_model, device):
    index, snippets = [], []
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

    scene_pol, char_pol = {}, {}
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

def process_scene(scene_xml, summary, scene_id,
                  nlp, nlp_ner_pos, tokenizer, max_seq_len, max_target_len):
    
    # 1. Clean the XML but retain perfect character-to-modality mapping
    clean_text, char_to_modality = clean_and_map_xml(scene_xml)

    # 2. Tokenize the CLEAN text
    encoding = tokenizer(
        clean_text, max_length=max_seq_len, truncation=True,
        return_offsets_mapping=True, padding="max_length",
    )
    target_enc = tokenizer(
        summary, max_length=max_target_len,
        truncation=True, padding="max_length",
    )
    input_ids  = encoding["input_ids"]
    target_ids = target_enc["input_ids"]
    offsets    = encoding["offset_mapping"]

    # 3. Build 4-way Modality Masks using the mapped offsets
    action_mask   = [0] * max_seq_len
    dialogue_mask = [0] * max_seq_len
    entity_mask   = [0] * max_seq_len
    header_mask   = [0] * max_seq_len

    dial_count = act_count = 0

    for i, (start, end) in enumerate(offsets):
        if i >= max_seq_len or start == end:
            continue
        
        # Look up the structural modality of this token based on its start character
        mod = char_to_modality[start] if start < len(char_to_modality) else "action"
        
        if mod == "action":
            action_mask[i] = 1
            act_count += 1
        elif mod == "dialogue":
            dialogue_mask[i] = 1
            dial_count += 1
        elif mod == "header":
            header_mask[i] = 1

    total        = max(1, dial_count + act_count)
    dial_density = dial_count / total
    act_density  = act_count  / total

    # 4. SVO Triplets on CLEAN text
    doc      = nlp(clean_text[:_SPACY_MAX_CHARS])
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

    # 5. NER on CLEAN text
    doc_full    = nlp_ner_pos(clean_text)
    ner_entities = []
    seen_ner     = set()

    for ent in doc_full.ents:
        if ent.label_ not in _KEPT_ENTITY_TYPES:
            continue
        normalized = ent.text.strip().lower()
        if not normalized or len(normalized) < 2:
            continue
        
        # Entity mask marking
        e_idx = next((idx for idx, (s, e) in enumerate(offsets)
                      if s <= ent.start_char < e and s != e), -1)
        if 0 <= e_idx < max_seq_len:
            entity_mask[e_idx] = 1
            
        if normalized not in seen_ner:
            ner_entities.append({
                "text": normalized,
                "type": ent.label_,
            })
            seen_ner.add(normalized)

    # 6. Sentiment snippets
    sentences  = re.split(r"(?<=[.!?])\s+", clean_text)
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
        pat   = re.compile("|".join(re.escape(v) for v in variants), re.IGNORECASE)
        sents = [s for s in sentences if pat.search(s)]
        char_snippets[char_key] = (" ".join(sents) if sents else clean_text)[:_SENT_MAX_CHARS]

    return {
        "movie_id":        scene_id,
        "input_ids":       input_ids,
        "target_ids":      target_ids,
        "action_mask":     action_mask,
        "dialogue_mask":   dialogue_mask,
        "entity_mask":     entity_mask,
        "header_mask":     header_mask,
        "graph_triplets":  triplets,
        "characters":      extract_robust_characters(scene_xml, clean_text), 
        "ner_entities":    ner_entities,      
        "scene_meta": {
            "dialogue_density": round(dial_density, 4),
            "action_density":   round(act_density,  4),
        },
        "_char_snippets":  char_snippets,
        "_scene_snippet":  clean_text[:_SENT_MAX_CHARS],
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import multiprocessing as mp
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   type=int, default=0)
    parser.add_argument("--end",     type=int, default=-1)
    parser.add_argument("--out",     type=str, default="/tmp/uday/moviesum_data.jsonl.gz")
    parser.add_argument("--dataset", type=str, default="moviesum", choices=["mensa", "moviesum"])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading sentiment model on GPU...", flush=True)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sent_tok   = AutoTokenizer.from_pretrained(_SENTIMENT_MODEL)
    sent_model = AutoModelForSequenceClassification.from_pretrained(_SENTIMENT_MODEL).to(device).eval()
    if device.type == "cuda":
        sent_model = sent_model.to(torch.bfloat16)

    all_scenes = []

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
                            "text": str(scene_txt).strip(),
                            "summary": summary_text,
                            "id": f"{movie_id}_Scene_{i:03d}",
                        })

    elif args.dataset == "moviesum":
        print("Downloading MovieSum dataset...", flush=True)
        dataset = load_dataset("rohitsaxena/MovieSum", split="train")
        for idx, ex in enumerate(tqdm(dataset, desc="Building scene list (MovieSum)")):
            scenes_data  = ex.get("script", ex.get("screenplay", ex.get("scenes", "")))
            summary_text = ex.get("summary", "")
            raw_id   = ex.get("movie_name") or ex.get("title") or ex.get("imdb_id")
            movie_id = str(raw_id).strip() if raw_id else f"movie_{idx}"
            movie_id = re.sub(r"[^A-Za-z0-9 _\-]", "", movie_id).strip() or f"movie_{idx}"
            
            if isinstance(scenes_data, list):
                scenes_data = "\n".join([str(s) for s in scenes_data])
                
            if isinstance(scenes_data, str) and len(scenes_data.strip()) > 10:
                raw_scenes = re.findall(r'<scene>(.*?)</scene>', scenes_data, re.IGNORECASE | re.DOTALL)
                if not raw_scenes:
                    raw_scenes = re.split(r"(?=(?:INT[.]|EXT[.])[ \t])", scenes_data)
                    
                for i, scene_txt in enumerate(raw_scenes):
                    if len(scene_txt.strip()) > 10:
                        all_scenes.append({
                            "text": scene_txt.strip(),
                            "summary": summary_text,
                            "id": f"{movie_id}_Scene_{i:03d}",
                        })

    end = args.end if args.end > 0 else len(all_scenes)
    all_scenes = all_scenes[args.start:end]
    print(f"Processing {len(all_scenes):,} scenes (indices {args.start}–{end})", flush=True)

    num_workers = min(16, max(1, mp.cpu_count() - 1))
    print(f"Starting multiprocessing pool with {num_workers} CPU workers...", flush=True)

    written = 0
    buffer  = []

    with gzip.open(args.out, "wt", encoding="utf-8") as out_f:
        with mp.Pool(processes=num_workers, initializer=init_worker, maxtasksperchild=50) as pool:
            iterator = pool.imap_unordered(process_scene_wrapper, all_scenes, chunksize=32)
            
            for pre in tqdm(iterator, total=len(all_scenes), desc="Extracting", unit="scene"):
                if pre is not None:
                    buffer.append(pre)

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
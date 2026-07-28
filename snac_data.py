"""
snac_data.py — MovieSum loading + streaming-scene / entity-slot preparation for SNaC.

We deliberately avoid heavy NLP dependencies here: character entities are pulled
from screenplay ALL-CAPS cue frequency (robust and standard for scripts). spaCy
is an *optional* upgrade, not a requirement, so the pipeline can't break on it.

Exposes:
  SNaCConfig                     — shared hyperparameters (data + model shapes)
  load_movies(split, cfg)        — list of MovieRaw dicts from HF MovieSum/MENSA
  tokenize_movie(raw, tok, cfg)  — dict of CPU tensors ready for the model
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Any

import torch


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SNaCConfig:
    # backbone
    backbone: str = "facebook/bart-large-cnn"   # swap: google/flan-t5-large, allenai/led-large-16384
    # data shapes
    dataset: str = "moviesum"                    # 'moviesum' | 'mensa' | any HF id
    max_scenes: int = 48                         # stride-sample movies longer than this
    scene_tokens: int = 256                      # per-scene encoder budget
    max_entities: int = 24                       # entity-bound slots
    free_slots: int = 8                          # un-bound global/plot/theme slots
    max_target: int = 400                        # summary decoder budget
    min_entity_freq: int = 2                     # ALL-CAPS token must recur to count
    # model
    n_heads: int = 8
    k_retrieve: int = 6                          # scenes kept at full token resolution (local channel)
    lambda_probe: float = 0.3                    # weight on the entity-presence probe loss
    memory_mode: str = "two_source"              # 'two_source' | 'full' | 'state_only' (diagnostics)
    # misc
    seed: int = 0

    @property
    def n_slots(self) -> int:
        return self.max_entities + self.free_slots


MovieRaw = Dict[str, Any]

_HF_NAMES = {"moviesum": "rohitsaxena/MovieSum", "mensa": "rohitsaxena/MENSA"}

# tokens that look like character cues in ALL-CAPS but aren't people
_SCREEN_STOP = {
    "INT", "EXT", "CUT", "FADE", "DISSOLVE", "CONTINUED", "CONT", "OMITTED",
    "THE", "AND", "TO", "OF", "A", "IN", "ON", "DAY", "NIGHT", "LATER",
    "MORNING", "EVENING", "SCENE", "ANGLE", "CLOSE", "WIDE", "POV", "V", "O", "S",
    "INTERIOR", "EXTERIOR", "CAMERA", "SHOT", "BACK", "MONTAGE", "TITLE", "CARD",
    "SUPER", "INSERT", "END", "BEGIN", "FLASHBACK", "PRESENT", "MEANWHILE", "OS", "VO",
}
_CAPS_RE = re.compile(r"\b[A-Z][A-Z'\.\-]{1,}\b")


def split_scenes(script: str) -> List[str]:
    """Split a MovieSum `script` string into scene chunks."""
    if not isinstance(script, str):
        script = "\n".join(map(str, script)) if isinstance(script, list) else str(script)
    scenes = re.findall(r"<scene>(.*?)</scene>", script, re.IGNORECASE | re.DOTALL)
    if not scenes:
        # fall back: cut before each INT./EXT. slug line
        scenes = re.split(r"(?=(?:INT[.]|EXT[.])[ \t])", script)
    scenes = [s.strip() for s in scenes if len(s.strip()) > 15]
    return scenes


def _stride_sample(items: List[Any], k: int) -> List[Any]:
    if len(items) <= k:
        return items
    idx = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    seen, out = set(), []
    for j in idx:                       # keep order, drop dup indices
        if j not in seen:
            seen.add(j); out.append(items[j])
    return out


def extract_entities(script: str, cfg: SNaCConfig) -> List[str]:
    """Top character names by ALL-CAPS cue frequency."""
    counts: Dict[str, int] = {}
    for m in _CAPS_RE.findall(script):
        tok = m.strip(".'-")
        if len(tok) < 2 or tok in _SCREEN_STOP or not tok.isalpha():
            continue
        counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted((c for c in counts.items() if c[1] >= cfg.min_entity_freq),
                    key=lambda x: -x[1])
    return [name for name, _ in ranked[:cfg.max_entities]]


def _presence_row(scene: str, entities: List[str]) -> List[float]:
    """Per-scene role weight for each entity: 1.0 cue-speaker, 0.5 mention, 0.0 absent."""
    up = scene.upper()
    head = up[:120]                     # scene head where cues live
    row = []
    for e in entities:
        w = 0.0
        if re.search(rf"\b{re.escape(e)}\b", up):
            w = 0.5
            if re.search(rf"\b{re.escape(e)}\b", head):
                w = 1.0
        row.append(w)
    return row


def load_movies(split: str, cfg: SNaCConfig) -> List[MovieRaw]:
    """Load a HF split into per-movie raw structures (no tokenization yet)."""
    from datasets import load_dataset
    name = _HF_NAMES.get(cfg.dataset, cfg.dataset)
    ds = load_dataset(name, split=split)
    movies: List[MovieRaw] = []
    for ex in ds:
        script = ex.get("script") or ex.get("screenplay") or ex.get("scenes") or ""
        summary = (ex.get("summary") or "").strip()
        title = str(ex.get("movie_name") or ex.get("title") or ex.get("imdb_id") or "movie")
        scenes = split_scenes(script)
        if not scenes or not summary:
            continue
        entities = extract_entities(script if isinstance(script, str)
                                    else "\n".join(map(str, script)), cfg)
        scenes = _stride_sample(scenes, cfg.max_scenes)
        presence = [_presence_row(s, entities) for s in scenes]   # [S, E]
        movies.append({
            "title": title,
            "scenes": scenes,
            "entities": entities,
            "presence": presence,
            "summary": summary,
        })
    return movies


def tokenize_movie(raw: MovieRaw, tok, cfg: SNaCConfig) -> Dict[str, Any]:
    """Turn a MovieRaw into CPU tensors. Batch dim is implicit (one movie)."""
    scenes, entities = raw["scenes"], raw["entities"]
    E = cfg.max_entities

    enc = tok(scenes, truncation=True, max_length=cfg.scene_tokens,
              padding="max_length", return_tensors="pt")
    scene_ids = enc.input_ids                    # [S, L]
    scene_mask = enc.attention_mask              # [S, L]

    # entity name ids for grounded slot init (pad the entity list up to E)
    ent_names = entities + ["<pad>"] * (E - len(entities))
    ename = tok(ent_names, truncation=True, max_length=8,
                padding="max_length", return_tensors="pt")
    ename_ids, ename_mask = ename.input_ids, ename.attention_mask   # [E, Ln]

    # presence [S, E]  (pad entity dim to E)
    S = len(scenes)
    pres = torch.zeros(S, E)
    for s, row in enumerate(raw["presence"]):
        for e, w in enumerate(row[:E]):
            pres[s, e] = w
    entity_valid = torch.zeros(E)
    entity_valid[:len(entities)] = 1.0

    tgt = tok(raw["summary"], truncation=True, max_length=cfg.max_target,
              return_tensors="pt")
    labels = tgt.input_ids[0]                    # [T]

    return {
        "scene_ids": scene_ids, "scene_mask": scene_mask,
        "ename_ids": ename_ids, "ename_mask": ename_mask,
        "presence": pres, "entity_valid": entity_valid,
        "labels": labels,
        "n_entities": len(entities),
        "entities": entities, "scenes": scenes,
        "title": raw["title"], "summary": raw["summary"],
    }

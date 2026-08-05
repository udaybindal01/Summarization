"""
snac_data.py — MovieSum loading + whole-document LED preparation for SNaC.

We encode the WHOLE screenplay as one long sequence (LED, up to 16K tokens) with
`</s>` scene separators and *global attention* on those separators. This keeps the
decoder's cross-attention memory in-distribution and context-aware, instead of the
old per-scene contextless chunks that capped fluency and metrics.

Character entities are pulled from screenplay ALL-CAPS cue frequency (robust and
standard for scripts) — no spaCy/fastcoref dependency.

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
    backbone: str = "allenai/led-large-16384"    # long-context; whole-doc encoding
    # data shapes
    dataset: str = "moviesum"                    # 'moviesum' | 'mensa' | any HF id
    max_doc_tokens: int = 16384                  # LED encoder budget (whole script)
    max_scenes: int = 80                         # cap on #scenes in the state loop
    scene_tokens: int = 512                      # per-scene truncation cap when packing
    max_entities: int = 24                       # entity-bound slots
    free_slots: int = 8                          # un-bound global/plot/theme slots
    max_target: int = 400                        # summary decoder budget
    min_entity_freq: int = 2                     # ALL-CAPS token must recur to count
    # model
    n_heads: int = 8
    k_retrieve: int = 8                          # scenes kept verbatim (two_source local channel)
    lambda_probe: float = 0.3                    # weight on the entity-presence probe loss
    memory_mode: str = "full"                    # 'full' | 'two_source' | 'state_only'
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
        full_txt = script if isinstance(script, str) else "\n".join(map(str, script))
        entities = extract_entities(full_txt, cfg)
        movies.append({
            "title": title,
            "scenes": scenes,       # ALL scenes; packing/truncation happens at tokenize time
            "entities": entities,
            "summary": summary,
        })
    return movies


def tokenize_movie(raw: MovieRaw, tok, cfg: SNaCConfig) -> Dict[str, Any]:
    """
    Pack the whole screenplay into one LED sequence:
        [BOS] scene0 </s> scene1 </s> ... sceneN </s>
    up to `max_doc_tokens`, recording each scene's [start,end) token span so the
    streaming state and the two_source retrieval can slice per-scene encoder states.
    Global attention is placed on BOS + every </s> separator.
    """
    entities = raw["entities"]
    E = cfg.max_entities
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    sep = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id

    ids: List[int] = [bos]
    boundaries: List[List[int]] = []          # [S, 2] token spans (scene content, excl. sep)
    kept_scenes: List[str] = []
    global_pos: List[int] = [0]               # BOS gets global attention
    for scene in raw["scenes"]:
        if len(kept_scenes) >= cfg.max_scenes:
            break
        stoks = tok(scene, add_special_tokens=False, truncation=True,
                    max_length=cfg.scene_tokens).input_ids
        if not stoks:
            continue
        if len(ids) + len(stoks) + 1 >= cfg.max_doc_tokens:
            break
        a = len(ids)
        ids.extend(stoks)
        b = len(ids)
        ids.append(sep)                       # scene separator (</s>)
        global_pos.append(b)                  # separator gets global attention
        boundaries.append([a, b])
        kept_scenes.append(scene)

    if not boundaries:                        # degenerate movie: keep at least BOS+sep
        ids.append(sep); boundaries.append([1, 1]); kept_scenes.append("")

    n_tokens = len(ids)
    input_ids = torch.tensor(ids, dtype=torch.long)
    attn_mask = torch.ones(n_tokens, dtype=torch.long)
    global_mask = torch.zeros(n_tokens, dtype=torch.long)
    global_mask[torch.tensor(global_pos, dtype=torch.long)] = 1

    # presence [S, E] for the KEPT scenes (pad entity dim to E)
    S = len(kept_scenes)
    pres = torch.zeros(S, E)
    for s, scene in enumerate(kept_scenes):
        row = _presence_row(scene, entities)
        for e, w in enumerate(row[:E]):
            pres[s, e] = w
    entity_valid = torch.zeros(E)
    entity_valid[:len(entities)] = 1.0

    # entity name token ids for grounded slot init (pad the entity list up to E)
    ent_names = entities + ["<pad>"] * (E - len(entities))
    ename = tok(ent_names, add_special_tokens=False, truncation=True,
                max_length=8, padding="max_length", return_tensors="pt")
    ename_ids, ename_mask = ename.input_ids, ename.attention_mask   # [E, Ln]

    tgt = tok(raw["summary"], truncation=True, max_length=cfg.max_target,
              return_tensors="pt")
    labels = tgt.input_ids[0]                    # [T]

    return {
        "input_ids": input_ids, "attn_mask": attn_mask, "global_mask": global_mask,
        "n_tokens": n_tokens, "boundaries": boundaries,     # python list (indices)
        "ename_ids": ename_ids, "ename_mask": ename_mask,
        "presence": pres, "entity_valid": entity_valid,
        "labels": labels,
        "n_entities": len(entities),
        "entities": entities, "scenes": kept_scenes,
        "title": raw["title"], "summary": raw["summary"],
    }

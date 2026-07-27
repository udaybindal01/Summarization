"""
prototype_bottleneck.py  —  Decisive A/B test of the v5 failure hypothesis.

Hypothesis
----------
v5 produces junk (train loss ~7.5, ROUGE-2 ~0) NOT because of dead Mamba
gradients, but because the decoder only ever cross-attends to <=64 mean-pooled
"scene" vectors (sum.py:935, sum.py:986-1028). Mean-pooling ~250 tokens/scene
destroys the lexical information the decoder needs to generate faithful text.

This script holds EVERYTHING constant (same model, same data, same steps,
same batch) and flips ONE variable: the decoder's cross-attention memory.

  --mode full    : decoder attends to full token-level encoder states  (standard seq2seq)
  --mode pooled  : decoder attends to chunk-mean-pooled states          (reproduces v5)

Expected result if the hypothesis is correct:
  full   -> train loss falls toward ~1-2, generations are real English.
  pooled -> train loss plateaus high (~6-8), generations are junk.

We deliberately OVERFIT a small movie subset: if the pooled memory can't even
overfit 30 movies, it certainly can't learn the real task. Fast to run.

Run on the A100 server (where /tmp/uday data lives):
  python3 prototype_bottleneck.py --mode full   --n_movies 30 --steps 400
  python3 prototype_bottleneck.py --mode pooled --n_movies 30 --steps 400
"""
import argparse, gzip, json, re, sys, random
from collections import defaultdict

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration


def _movie_key(mid):
    return mid.split("_Scene_")[0] if "_Scene_" in mid else re.sub(r"_[Ss]cene_?\d+$", "", mid)


def _scene_idx(mid):
    m = re.search(r"_[Ss]cene_?(\d+)", mid)
    return int(m.group(1)) if m else 0


def load_movies(path, n_movies, tok, max_chars=200_000):
    """Group scene-level jsonl into (screenplay, summary) per movie.

    Mirrors train.py's summary logic: prefer scene-0 `summary_text`, else decode
    scene-0 `target_ids` with the tokenizer. Self-diagnoses on empty result.
    """
    import os
    if not os.path.exists(path):
        sys.exit(f"[error] --data path does not exist ON THIS NODE: {path}\n"
                 f"  hint: run  find /scratch /tmp /local -name '*.jsonl*' 2>/dev/null  "
                 f"ON the compute node (paths differ from the login node).")

    scenes = defaultdict(list)          # key -> [(scene_idx, clean_text), ...]
    summ_src = {}                       # key -> (scene_idx, summary_text, target_ids)
    n_lines = first_rec = 0
    first_keys = None
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            n_lines += 1
            try:
                r = json.loads(line)
            except Exception:
                continue
            if first_keys is None:
                first_keys = list(r.keys())
            mid = r.get("movie_id") or r.get("id") or ""
            key = _movie_key(mid)
            sidx = _scene_idx(mid)
            txt = (r.get("clean_text") or "").strip()
            if txt:
                scenes[key].append((sidx, txt))
            # keep the earliest scene's summary source per movie
            if key not in summ_src or sidx < summ_src[key][0]:
                summ_src[key] = (sidx, (r.get("summary_text") or "").strip(),
                                 r.get("target_ids"))

    movies, via_text, via_ids = [], 0, 0
    for key, chunks in scenes.items():
        sidx, stext, tids = summ_src.get(key, (0, "", None))
        summary = stext
        if summary:
            via_text += 1
        elif isinstance(tids, list) and tids:
            summary = tok.decode([t for t in tids if isinstance(t, int)],
                                 skip_special_tokens=True).strip()
            if summary:
                via_ids += 1
        if not summary:
            continue
        chunks.sort(key=lambda x: x[0])
        screenplay = ("\n</s>\n".join(t for _, t in chunks))[:max_chars]
        movies.append((screenplay, summary))

    print(f"[data] scanned {n_lines:,} scenes across {len(scenes):,} movies; "
          f"built {len(movies)} pairs (summary via_text={via_text} via_target_ids={via_ids})")
    if not movies:
        print(f"[data][diagnose] first record keys = {first_keys}")
        sys.exit("[error] 0 usable (screenplay, summary) pairs. The schema above "
                 "shows what's actually in the file — tell me these keys and I'll adapt the loader.")
    random.Random(0).shuffle(movies)
    return movies[:n_movies]


def encoder_memory(model, input_ids, attention_mask, mode, chunk=32):
    """Return (memory, memory_mask) for the decoder's cross-attention."""
    enc = model.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    h = enc.last_hidden_state                      # [B, T, D]
    if mode == "full":
        return h, attention_mask
    # --- pooled: reproduce the v5 bottleneck (mean-pool fixed-size chunks) ---
    B, T, D = h.shape
    pooled = []
    for b in range(B):
        valid = attention_mask[b].bool()
        hv = h[b][valid]                           # [t, D]
        t = hv.size(0)
        segs = [hv[i:i + chunk].mean(0) for i in range(0, t, chunk)]
        pooled.append(torch.stack(segs) if segs else h[b, :1])
    S = max(p.size(0) for p in pooled)
    mem = h.new_zeros(B, S, D)
    mmask = attention_mask.new_zeros(B, S)
    for b, p in enumerate(pooled):
        mem[b, :p.size(0)] = p
        mmask[b, :p.size(0)] = 1
    return mem, mmask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "pooled"], required=True)
    ap.add_argument("--data", default="/tmp/uday/moviesum_data.jsonl.gz")
    ap.add_argument("--model", default="facebook/bart-base")
    ap.add_argument("--n_movies", type=int, default=30)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_src", type=int, default=1024)
    ap.add_argument("--max_tgt", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--chunk", type=int, default=32, help="tokens per pooled scene")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[setup] mode={args.mode} device={dev} model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = BartForConditionalGeneration.from_pretrained(args.model).to(dev)
    model.train()

    movies = load_movies(args.data, args.n_movies, tok)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def batch(bm):
        src = tok([m[0] for m in bm], truncation=True, max_length=args.max_src,
                  padding=True, return_tensors="pt").to(dev)
        tgt = tok([m[1] for m in bm], truncation=True, max_length=args.max_tgt,
                  padding=True, return_tensors="pt").to(dev)
        return src, tgt

    step = 0
    while step < args.steps:
        random.shuffle(movies)
        for i in range(0, len(movies), args.bs):
            bm = movies[i:i + args.bs]
            if len(bm) < 1:
                continue
            src, tgt = batch(bm)
            mem, mmask = encoder_memory(model, src.input_ids, src.attention_mask,
                                        args.mode, chunk=args.chunk)
            labels = tgt.input_ids.clone()
            labels[labels == tok.pad_token_id] = -100
            out = model(
                encoder_outputs=(mem,),
                attention_mask=mmask,
                labels=labels,
            )
            loss = out.loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 25 == 0 or step == 1:
                print(f"[step {step:4d}] loss={loss.item():.3f}")
            if step >= args.steps:
                break

    # ---- qualitative check: generate on a held example ----
    model.eval()
    with torch.no_grad():
        src, tgt = batch(movies[:1])
        mem, mmask = encoder_memory(model, src.input_ids, src.attention_mask,
                                    args.mode, chunk=args.chunk)
        gen = model.generate(
            encoder_outputs=_wrap_enc(mem),
            attention_mask=mmask,
            num_beams=4,
            max_length=args.max_tgt,
            no_repeat_ngram_size=3,
        )
    print("\n========== SAMPLE GENERATION (mode=%s) ==========" % args.mode)
    print("GOLD :", tok.decode(tgt.input_ids[0], skip_special_tokens=True)[:400])
    print("PRED :", tok.decode(gen[0], skip_special_tokens=True)[:400])
    print("=================================================")


def _wrap_enc(mem):
    from transformers.modeling_outputs import BaseModelOutput
    return BaseModelOutput(last_hidden_state=mem)


if __name__ == "__main__":
    main()

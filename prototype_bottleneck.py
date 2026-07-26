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


def load_movies(path, n_movies, max_chars=200_000):
    """Group scene-level jsonl.gz into (screenplay_text, summary_text) per movie."""
    scenes = defaultdict(list)
    summaries = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            mid = r.get("movie_id", "")
            key = re.sub(r"_[Ss]cene_?\d+$", "", mid)
            txt = (r.get("clean_text") or "").strip()
            if txt:
                scenes[key].append(txt)
            summ = (r.get("summary_text") or "").strip()
            if summ and key not in summaries:
                summaries[key] = summ

    movies = []
    for key, chunks in scenes.items():
        if key in summaries and summaries[key]:
            screenplay = ("\n</s>\n".join(chunks))[:max_chars]
            movies.append((screenplay, summaries[key]))
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

    movies = load_movies(args.data, args.n_movies)
    if not movies:
        sys.exit(f"[error] no movies loaded from {args.data}")
    print(f"[data] {len(movies)} movies loaded")

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

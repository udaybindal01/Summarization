"""
snac_eval.py — evaluate a trained SNaC checkpoint on the MovieSum test split.

Produces (a) headline metrics: ROUGE-1/2/L + BERTScore over the FULL test set,
and (b) the paper's analysis artifacts:
    - turning points : per-scene ||S_t - S_{t-1}|| (should align with act structure)
    - causal probe   : regenerate with a key entity slot erased -> the summary
                       should change on exactly the facts that slot carried.

Example:
  python3 snac_eval.py --ckpt /scratch/karan/snac/snac_best.pt \
                       --out /scratch/karan/snac/test_report.json --intervene 10
"""
import argparse, json

import torch
from transformers import AutoTokenizer

from snac_data import SNaCConfig, load_movies, tokenize_movie
from snac_model import SNaC, move_batch


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = SNaCConfig(**ckpt["cfg"])
    model = SNaC(cfg, lora=ckpt.get("lora", False))
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[load] {len(missing)} missing keys (ok if frozen/tied), "
              f"{len(unexpected)} unexpected")
    return model.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="test_report.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gen_tokens", type=int, default=400)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = full test split")
    ap.add_argument("--intervene", type=int, default=10,
                    help="run the causal-intervention demo on this many movies")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)
    tok = AutoTokenizer.from_pretrained(cfg.backbone)

    movies = load_movies(args.split, cfg)
    if args.limit:
        movies = movies[:args.limit]
    print(f"[eval] {len(movies)} {args.split} movies · backbone={cfg.backbone}")

    preds, refs, records, turning = [], [], [], []
    for mi, raw in enumerate(movies):
        batch = move_batch(tokenize_movie(raw, tok, cfg), device)
        ids, deltas, ret_idx = model.generate(
            batch, num_beams=args.num_beams, max_new_tokens=args.gen_tokens)
        pred = tok.decode(ids[0], skip_special_tokens=True).strip()
        preds.append(pred); refs.append(raw["summary"])
        records.append({"title": raw["title"], "pred": pred,
                        "ref": raw["summary"][:600],
                        "retrieved_scenes": ret_idx.tolist()})
        # per-scene turning-point signal (mean slot change), for the first movies
        if mi < 40:
            turning.append({"title": raw["title"],
                            "entities": raw["entities"][:cfg.max_entities],
                            "delta_per_scene": deltas.mean(-1).tolist()})
        if (mi + 1) % 25 == 0:
            print(f"  generated {mi+1}/{len(movies)}")

    # ---- headline metrics ----
    report = {"n": len(preds), "backbone": cfg.backbone}
    try:
        import evaluate
        r = evaluate.load("rouge").compute(predictions=preds, references=refs)
        report["rouge"] = {k: round(float(v), 4) for k, v in r.items()}
        try:
            b = evaluate.load("bertscore").compute(
                predictions=preds, references=refs, lang="en")
            report["bertscore_f1"] = round(sum(b["f1"]) / len(b["f1"]), 4)
        except Exception as e:
            print(f"[metrics] bertscore failed: {e}")
    except Exception as e:
        print(f"[metrics] evaluate unavailable: {e}")
    print(f"[eval] metrics = {json.dumps(report.get('rouge', {}))} "
          f"bertF1={report.get('bertscore_f1')}")

    # ---- causal intervention demo ----
    interventions = []
    for raw in movies[:args.intervene]:
        if not raw["entities"]:
            continue
        batch = move_batch(tokenize_movie(raw, tok, cfg), device)
        base, _, _ = model.generate(batch, num_beams=args.num_beams,
                                    max_new_tokens=args.gen_tokens)
        edited, _, _ = model.generate(batch, perturb_slot=0,   # erase top entity
                                      num_beams=args.num_beams,
                                      max_new_tokens=args.gen_tokens)
        interventions.append({
            "title": raw["title"], "erased_entity": raw["entities"][0],
            "summary_base": tok.decode(base[0], skip_special_tokens=True).strip(),
            "summary_erased": tok.decode(edited[0], skip_special_tokens=True).strip(),
        })

    out = {"report": report, "turning_points": turning,
           "interventions": interventions, "predictions": records}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] wrote full report -> {args.out}")


if __name__ == "__main__":
    main()

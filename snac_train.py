"""
snac_train.py — train SNaC on the full MovieSum train split, validate each epoch.

Example (full train set, validate on the official validation split):
  python3 snac_train.py --epochs 8 --grad_accum 8 --out /scratch/karan/snac

Key flags: --backbone, --lora, --max_scenes, --lambda_probe, --lr_state, --lr_dec.
"""
import argparse, os, json, random, time

import torch
from transformers import AutoTokenizer

from snac_data import SNaCConfig, load_movies, tokenize_movie
from snac_model import SNaC, move_batch


def get_metrics():
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        try:
            bertscore = evaluate.load("bertscore")
        except Exception:
            bertscore = None
        return rouge, bertscore
    except Exception as e:
        print(f"[metrics] evaluate unavailable ({e}); ROUGE/BERTScore disabled.")
        return None, None


@torch.no_grad()
def run_eval(model, tok, movies, cfg, device, rouge, bertscore, max_eval, gen_tokens):
    model.eval()
    preds, refs = [], []
    for raw in movies[:max_eval]:
        batch = move_batch(tokenize_movie(raw, tok, cfg), device)
        ids, _, _ = model.generate(batch, num_beams=4, max_new_tokens=gen_tokens)
        preds.append(tok.decode(ids[0], skip_special_tokens=True).strip())
        refs.append(raw["summary"])
    out = {}
    if rouge is not None and preds:
        r = rouge.compute(predictions=preds, references=refs)
        out.update({k: round(float(v), 4) for k, v in r.items()})
    if bertscore is not None and preds:
        try:
            b = bertscore.compute(predictions=preds, references=refs, lang="en")
            out["bertscore_f1"] = round(sum(b["f1"]) / len(b["f1"]), 4)
        except Exception as e:
            print(f"[metrics] bertscore failed: {e}")
    out["_sample_pred"] = preds[0][:300] if preds else ""
    return out


def build_config(args) -> SNaCConfig:
    return SNaCConfig(
        backbone=args.backbone, dataset=args.dataset,
        max_scenes=args.max_scenes, scene_tokens=args.scene_tokens,
        max_entities=args.max_entities, free_slots=args.free_slots,
        max_target=args.max_target, k_retrieve=args.k_retrieve,
        lambda_probe=args.lambda_probe, seed=args.seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="facebook/bart-large-cnn")
    ap.add_argument("--dataset", default="moviesum")
    ap.add_argument("--out", default="/Summarization/summary")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr_state", type=float, default=1e-4)
    ap.add_argument("--lr_dec", type=float, default=2e-5)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--train_encoder", action="store_true",
                    help="unfreeze the encoder (most reliable; matches the working prototype)")
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="gradient checkpointing (saves memory when --train_encoder)")
    ap.add_argument("--max_scenes", type=int, default=48)
    ap.add_argument("--scene_tokens", type=int, default=256)
    ap.add_argument("--max_entities", type=int, default=24)
    ap.add_argument("--free_slots", type=int, default=8)
    ap.add_argument("--max_target", type=int, default=400)
    ap.add_argument("--k_retrieve", type=int, default=6)
    ap.add_argument("--lambda_probe", type=float, default=0.3)
    ap.add_argument("--gen_tokens", type=int, default=400)
    ap.add_argument("--max_eval", type=int, default=100)
    ap.add_argument("--limit_train", type=int, default=0, help="0 = full train set")
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = build_config(args)
    print(f"[setup] backbone={cfg.backbone} device={device} lora={args.lora} "
          f"slots={cfg.n_slots} (E={cfg.max_entities}+F={cfg.free_slots})")

    tok = AutoTokenizer.from_pretrained(cfg.backbone)
    model = SNaC(cfg, lora=args.lora, train_encoder=args.train_encoder).to(device)
    if args.grad_ckpt:
        model.backbone.config.use_cache = False
        model.backbone.gradient_checkpointing_enable()

    print("[data] loading MovieSum splits …")
    train = load_movies("train", cfg)
    val = load_movies("validation", cfg)
    if args.limit_train:
        train = train[:args.limit_train]
    print(f"[data] {len(train)} train / {len(val)} val movies")

    state_params = list(model.state.parameters())
    bb_params = [p for _, p in model.backbone.named_parameters() if p.requires_grad]
    n_state = sum(p.numel() for p in state_params)
    n_bb = sum(p.numel() for p in bb_params)
    print(f"[params] trainable: state={n_state/1e6:.1f}M  backbone={n_bb/1e6:.1f}M")
    opt = torch.optim.AdamW([
        {"params": state_params, "lr": args.lr_state},
        {"params": bb_params, "lr": args.lr_dec},
    ], weight_decay=0.01)

    total_updates = args.epochs * max(1, len(train) // args.grad_accum)
    sched = None
    try:
        from transformers import get_cosine_schedule_with_warmup
        sched = get_cosine_schedule_with_warmup(
            opt, int(args.warmup_frac * total_updates), total_updates)
        print(f"[sched] cosine warmup: {total_updates} updates, "
              f"{int(args.warmup_frac*total_updates)} warmup")
    except Exception as e:
        print(f"[sched] scheduler unavailable ({e}); constant LR")

    rouge, bertscore = get_metrics()
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train(); random.shuffle(train)
        opt.zero_grad()
        running = 0.0; seen = 0; t0 = time.time()
        for i, raw in enumerate(train, 1):
            batch = move_batch(tokenize_movie(raw, tok, cfg), device)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 \
                  else torch.autocast("cpu", enabled=False)
            with ctx:
                out = model(batch)
            loss = out["loss"] / args.grad_accum
            loss.backward()
            running += out["nll"].item(); seen += 1
            if i % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], 1.0)
                opt.step(); opt.zero_grad()
                if sched is not None:
                    sched.step()
            if i % args.log_every == 0 or i == 1:
                lr = opt.param_groups[1]["lr"]
                print(f"  e{epoch} {i}/{len(train)} "
                      f"nll={running/max(1,seen):.3f} probe={out['probe'].item():.3f} "
                      f"dΔ={out['delta_mean'].item():.3f} lr={lr:.2e} "
                      f"({(time.time()-t0)/60:.1f}m)", flush=True)
                running = 0.0; seen = 0
        opt.step(); opt.zero_grad()

        metrics = run_eval(model, tok, val, cfg, device, rouge, bertscore,
                           args.max_eval, args.gen_tokens)
        score = metrics.get("rougeL", metrics.get("rouge1", 0.0))
        print(f"[epoch {epoch}] val = {json.dumps(metrics)}")
        ckpt = {"model": model.state_dict(), "cfg": cfg.__dict__,
                "epoch": epoch, "metrics": metrics, "lora": args.lora,
                "train_encoder": args.train_encoder}
        torch.save(ckpt, os.path.join(args.out, "snac_last.pt"))
        if score > best:
            best = score
            torch.save(ckpt, os.path.join(args.out, "snac_best.pt"))
            print(f"[epoch {epoch}] new best (rougeL/1={score:.4f}) -> snac_best.pt")

    print(f"[done] best score = {best:.4f}. checkpoints in {args.out}")


if __name__ == "__main__":
    main()

"""
fix_checkpoint.py — one-shot patch for bad gate/entity_mem_scale values.

The previous nudge set gate_proj.bias=-2 (sigmoid=0.12 → 88% hypergraph)
and entity_mem_scale=0.5 (tanh=0.46), both far too aggressive for a model
that only completed 1 epoch of Stage 1 training. This script reverts them
to conservative values that still allow gradients to flow:

  gate_proj.bias   : -2.0 → -4.0   (sigmoid: 0.12 → 0.018)
  entity_mem_scale : 0.5  → 0.05   (tanh:    0.46 → 0.05)

Run once before resuming training:
    python fix_checkpoint.py
"""

import torch, os, shutil

CKPT = "/tmp/uday/checkpoints/led_mamba_latest.pt"
BAK  = CKPT + ".bak"

if not os.path.exists(CKPT):
    print(f"Checkpoint not found: {CKPT}")
    exit(1)

# Back up first
shutil.copy2(CKPT, BAK)
print(f"Backed up to {BAK}")

ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
sd   = ckpt["model_state_dict"]

# Fix gate_proj.bias
key_gate = "graph_text_fusion.gate_proj.bias"
if key_gate in sd:
    old = sd[key_gate].mean().item()
    sd[key_gate] = torch.full_like(sd[key_gate], -4.0)
    print(f"  {key_gate}: {old:.3f} → -4.0  "
          f"(sigmoid: {1/(1+torch.exp(torch.tensor(-old))):.4f} → 0.0180)")
else:
    print(f"  WARNING: {key_gate} not found in checkpoint")

# Fix entity_mem_scale
key_ems = "entity_mem_scale"
if key_ems in sd:
    old = sd[key_ems].item()
    sd[key_ems] = torch.tensor([0.05])
    print(f"  {key_ems}: {old:.4f} → 0.05  "
          f"(tanh: {torch.tanh(torch.tensor(old)):.4f} → 0.0500)")
else:
    print(f"  WARNING: {key_ems} not found in checkpoint")

# Clear optimizer state for these params (moments from bad values will cause
# large updates on first step — better to restart them fresh).
if "optimizer_state_dict" in ckpt:
    opt_sd = ckpt["optimizer_state_dict"]
    cleared = 0
    for group in opt_sd.get("param_groups", []):
        pass  # param_groups don't hold state; state is keyed by param index
    # optimizer state in AdamW is keyed by integer param index
    # We can't easily map param name → index without the model, so just
    # wipe the entire optimizer state — it will rebuild from scratch.
    ckpt["optimizer_state_dict"] = {}
    ckpt.pop("scheduler_state_dict", None)
    print("  Cleared optimizer state (will rebuild from scratch for clean restart)")

torch.save(ckpt, CKPT)
print(f"\nCheckpoint patched: {CKPT}")
print("Now run:  python train.py --run_name full_model_v5")

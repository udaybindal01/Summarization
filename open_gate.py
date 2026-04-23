"""
open_gate.py — gradually open the gate to allow hypergraph gradient flow.

Problem: gate_proj.bias=-4 (sigmoid=0.018) means only 1.8% of graph_info
reaches the fused representation. Backward pass gradients through the
hypergraph tower are 55x weaker than through H_text. The hypergraph is
barely being trained, causing the loss plateau at ~7.5.

Fix: gate_proj.bias -4 → -3 (sigmoid=0.047, 4.7% graph)  — 2.5x more
     entity_mem_scale   0 → 0.1 (tanh=0.100, 10% entity contribution)

This is conservative: 4.7% graph residual is still small enough not to
shock the decoder, but gives the hypergraph 2.5x more gradient signal.
Optimizer state cleared to avoid stale Adam moments.

Run once, then resume training normally.
"""

import torch, os, shutil, math

CKPT = "/tmp/uday/checkpoints/led_mamba_latest.pt"
BAK  = CKPT + ".bak_pre_open_gate"

if not os.path.exists(CKPT):
    print(f"Checkpoint not found: {CKPT}")
    exit(1)

shutil.copy2(CKPT, BAK)
print(f"Backed up to {BAK}")

ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
sd   = ckpt["model_state_dict"]

# Open gate
key_gate = "graph_text_fusion.gate_proj.bias"
if key_gate in sd:
    old = sd[key_gate].mean().item()
    new = -3.0
    sd[key_gate] = torch.full_like(sd[key_gate], new)
    def sig(x): return 1 / (1 + math.exp(-x))
    print(f"  {key_gate}: {old:.3f} → {new:.1f}  "
          f"(sigmoid: {sig(old):.4f} → {sig(new):.4f}  |  "
          f"graph weight: {sig(old)*100:.1f}% → {sig(new)*100:.1f}%)")
else:
    print(f"  WARNING: {key_gate} not found")

# Open entity_mem_scale
key_ems = "entity_mem_scale"
if key_ems in sd:
    old = sd[key_ems].item()
    sd[key_ems] = torch.tensor([0.1])
    print(f"  {key_ems}: {old:.4f} → 0.1  "
          f"(tanh: {math.tanh(old):.4f} → {math.tanh(0.1):.4f})")
else:
    print(f"  WARNING: {key_ems} not found")

# Clear optimizer state to avoid stale Adam moments for these params
if "optimizer_state_dict" in ckpt:
    ckpt["optimizer_state_dict"] = {}
    ckpt.pop("scheduler_state_dict", None)
    print("  Cleared optimizer state (fresh Adam moments)")

torch.save(ckpt, CKPT)
print(f"\nCheckpoint patched: {CKPT}")
print("Expect loss to temporarily rise to ~8-9 as the graph signal is "
      "absorbed, then continue decreasing below 7.5.")
print("Run: python train.py --run_name full_model_v5")

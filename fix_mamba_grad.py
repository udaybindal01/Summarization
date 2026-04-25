"""
fix_mamba_grad.py — unblock the two dead gradient paths to entity Mamba.

Problem: entity Mamba has zero gradients throughout all 11 epochs.
  Two paths carry gradient from loss → entity Mamba, both are blocked:

  Path 1: loss → aligned_memory entity positions → valid_nodes
            = entity_mask * H_nodes * tanh(entity_mem_scale)
          BLOCKED because decoder learned to ignore entity positions
          (they were zero for 7+ previous epochs). entity_mem_scale=0.1
          is too small — decoder attention weights on entity slots ≈ 0.

  Path 2: loss → fused_scenes → entity_scene_attn
            H_scenes += scene_ls * s_star  (s_star attends to H_nodes)
          BLOCKED because scene_ls was initialized to zeros(1024) and
          scene_ls≈0 → d_loss/d_H_nodes through this path ≈ 0.

Fix:
  entity_mem_scale : 0.1 → 0.6   (tanh: 0.100 → 0.537)
    Forces entity positions to 54% weight in aligned_memory.
    Decoder CANNOT ignore 100/164 positions all at 54% — must attend.
    → gradient flows through valid_nodes → H_nodes → Mamba.

  entity_scene_attn.scene_ls : 0 → 0.05 (per-dimension)
    Opens the scene←entity cross-attention gradient path.
    Small (0.05) so entity noise doesn't corrupt scene representations.
    → gradient also flows: fused_scenes → scene_ls * s_star → H_nodes.

  gate_proj.bias : keep at -3 (sigmoid=0.047, already patched)

Optimizer state cleared to avoid stale moments from dead-gradient era.
Run once, then: python train.py --run_name full_model_v5
"""

import torch, os, shutil, math

CKPT = "/tmp/uday/checkpoints/led_mamba_latest.pt"
BAK  = CKPT + ".bak_pre_mamba_fix"

if not os.path.exists(CKPT):
    print(f"Checkpoint not found: {CKPT}")
    exit(1)

shutil.copy2(CKPT, BAK)
print(f"Backed up to {BAK}")

ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
sd   = ckpt["model_state_dict"]

def tanh(x): return math.tanh(x)
def sig(x):  return 1 / (1 + math.exp(-x))

# ── 1. entity_mem_scale: 0.1 → 0.6 ─────────────────────────────────────────
key_ems = "entity_mem_scale"
if key_ems in sd:
    old = sd[key_ems].item()
    sd[key_ems] = torch.tensor([0.6])
    print(f"  {key_ems}: {old:.4f} → 0.6000  "
          f"(tanh: {tanh(old):.4f} → {tanh(0.6):.4f})")
else:
    print(f"  WARNING: {key_ems} not found")

# ── 2. scene_ls: 0 → 0.05 per dimension ─────────────────────────────────────
key_ls = "entity_scene_attn.scene_ls"
if key_ls in sd:
    old_mean = sd[key_ls].mean().item()
    d = sd[key_ls].shape[0]
    sd[key_ls] = torch.full((d,), 0.05)
    print(f"  {key_ls}: mean={old_mean:.5f} → 0.0500  (d={d})")
else:
    print(f"  WARNING: {key_ls} not found")

# ── 3. gate_proj.bias: keep at -3 if already there, else re-apply ────────────
key_gate = "graph_text_fusion.gate_proj.bias"
if key_gate in sd:
    cur = sd[key_gate].mean().item()
    if cur > -2.5:   # was accidentally opened too far — re-clamp
        sd[key_gate] = torch.full_like(sd[key_gate], -3.0)
        print(f"  {key_gate}: {cur:.3f} → -3.0 (re-clamped)")
    else:
        print(f"  {key_gate}: {cur:.3f} (unchanged, already ≤ -3)")

# ── 4. Clear optimizer state ─────────────────────────────────────────────────
if "optimizer_state_dict" in ckpt:
    ckpt["optimizer_state_dict"] = {}
    ckpt.pop("scheduler_state_dict", None)
    print("  Cleared optimizer + scheduler state")

torch.save(ckpt, CKPT)
print(f"\nCheckpoint patched: {CKPT}")
print("Expect loss spike to ~8-9 for 1-2 epochs as entity noise is absorbed,")
print("then continued decrease as Mamba learns to produce useful entity states.")
print("Watch entity_mamba gradient norms in W&B — should now be non-zero.")
print("Run: python train.py --run_name full_model_v5")

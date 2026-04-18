"""
inference.py — LED + Mamba-Hypergraph Inference (v5)
=====================================================
Loads a checkpoint, reads a test movie, and generates a summary
using the LED encoder + Mamba-hypergraph architecture.
"""

import torch
import gzip
import os
from transformers import AutoTokenizer

from sum import LEDMambaHypergraphSummariser, LED_MODEL, MAX_ENTITIES
from train import (
    MovieHypergraphDataset, SceneDataset, hypergraph_collate_fn,
    generate_summary, MAX_SCENES,
)
from peft import LoraConfig, get_peft_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CHECKPOINT_PATH = "/tmp/uday/checkpoints/led_mamba_latest.pt"

    print("Loading LED tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LED_MODEL)

    print("Initializing model architecture...")
    model = LEDMambaHypergraphSummariser(
        vocab_size=len(tokenizer),
        d_model=1024,
        max_entities=MAX_ENTITIES,
        max_scenes=MAX_SCENES,
        tokenizer=tokenizer,
    ).to(device)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        if any("lora" in k for k in state_dict.keys()):
            print("Auto-detected LoRA weights — applying PEFT wrapper...")
            lora_config = LoraConfig(
                r=16, lora_alpha=32,
                target_modules=["in_proj", "x_proj", "out_proj", "dt_proj"],
                lora_dropout=0.05, bias="none",
            )
            model.hypergraph_tower.entity_mamba = get_peft_model(
                model.hypergraph_tower.entity_mamba, lora_config)

        model.load_state_dict(state_dict, strict=False)
        print("Model loaded.")
    else:
        print(f"Checkpoint not found at {CHECKPOINT_PATH} — using random weights.")

    model.eval()

    # ── Load test data ────────────────────────────────────────────────────────
    eval_data_path = "/tmp/uday/mensa_test_data.jsonl.gz"
    eval_split     = "/tmp/uday/inference_test.jsonl"

    # Decompress to plain JSONL for SceneDataset
    if not os.path.exists(eval_split):
        print(f"Decompressing {eval_data_path} → {eval_split}...")
        with gzip.open(eval_data_path, "rt") as fin, open(eval_split, "wt") as fout:
            for line in fin:
                fout.write(line)

    print("Building dataset...")
    scene_ds = SceneDataset(eval_split, max_seq_len=256)
    movie_ds = MovieHypergraphDataset(scene_ds, tokenizer, max_scenes=MAX_SCENES)

    # Pick first movie
    item  = movie_ds[0]
    batch = hypergraph_collate_fn([item])

    print(f"\nGenerating summary for: {item['movie_name']} "
          f"({item['num_scenes']} scenes)...")

    with torch.no_grad():
        inp   = batch["input_ids"].to(device)
        amsk  = batch["attention_mask"].to(device)
        gattn = batch["global_attention_mask"].to(device)
        sbnds = batch["scene_boundaries"].to(device)
        inc   = batch["incidence_matrix"].to(device)
        etid  = batch["edge_type_ids"].to(device)
        enid  = batch["entity_type_ids"].to(device)
        emk   = batch["entity_mask"].to(device)
        enames = batch.get("entity_names")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            aligned_mem, _, __, dt_vals = model(
                inp, amsk, sbnds, gattn,
                inc, etid, enid, emk,
                entity_names=enames, return_dt=True,
            )

            S_count = sbnds.size(1)
            mem_pad = torch.zeros(1, aligned_mem.size(1),
                                  dtype=torch.bool, device=device)
            for s in range(S_count):
                start, end = sbnds[0, s].tolist()
                if start >= end:
                    mem_pad[0, s] = True
            mem_pad[:, S_count:] = ~emk
            all_masked = mem_pad.all(dim=1)
            mem_pad[all_masked, 0] = False
            enc_attn = (~mem_pad).long()

            summary = generate_summary(
                model, aligned_mem, enc_attn, tokenizer,
                device, max_new_tokens=300, beam_size=4,
            )

    print("\n" + "=" * 80)
    print("GENERATED SUMMARY:")
    print("=" * 80)
    print(summary)
    print("=" * 80)

    # Show reference
    ref = tokenizer.decode(batch["target_ids"][0].tolist(), skip_special_tokens=True)
    print("\nREFERENCE:")
    print("=" * 80)
    print(ref)
    print("=" * 80)


if __name__ == "__main__":
    main()

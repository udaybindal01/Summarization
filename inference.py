import torch
import torch.nn.functional as F
import json
import gzip
import os
from transformers import AutoTokenizer

from sum import GraMFormerV2
from train import movie_collate_fn
from peft import LoraConfig, get_peft_model


def build_causal_graph(scenes, max_scenes):
    """
    Builds the causal event adjacency matrix from SVO triplets.
    Edge i→j when an object entity of scene i matches a subject entity of scene j
    (causal chain), plus self-loops.
    """
    num_scenes = len(scenes)
    causal_adj = torch.zeros((max_scenes, max_scenes))
    causal_adj[:num_scenes, :num_scenes].fill_diagonal_(1.0)

    for i in range(num_scenes):
        objs_i = set()
        for t in scenes[i].get('graph_triplets', []):
            parts = t.split('_')
            if len(parts) >= 3:
                objs_i.add(parts[2].strip().lower())
        objs_i.discard("")

        for j in range(i + 1, num_scenes):
            subjs_j = set()
            for t in scenes[j].get('graph_triplets', []):
                parts = t.split('_')
                if len(parts) >= 1:
                    subjs_j.add(parts[0].replace("NOT ", "").strip().lower())
            subjs_j.discard("")

            if objs_i & subjs_j:
                causal_adj[i, j] = 1.0
                causal_adj[j, i] = 0.3

    return causal_adj


def build_char_state_graph(scenes, max_scenes):
    """
    Builds the character state adjacency matrix from emotion polarity data.
    Edge weight = mean absolute polarity change for shared characters.
    Falls back to zeros (with self-loops) when emotion data is absent.
    """
    num_scenes = len(scenes)
    char_state_adj = torch.zeros((max_scenes, max_scenes))
    char_state_adj[:num_scenes, :num_scenes].fill_diagonal_(1.0)

    emotions = [s.get('character_emotions', {}) for s in scenes]
    for i in range(num_scenes):
        for j in range(i + 1, num_scenes):
            shared = set(emotions[i].keys()) & set(emotions[j].keys())
            if not shared:
                continue
            changes = [abs(emotions[i][c] - emotions[j][c]) for c in shared]
            w = sum(changes) / len(changes)
            char_state_adj[i, j] = w
            char_state_adj[j, i] = w

    return char_state_adj


@torch.no_grad()
def generate_summary(model, tokenizer, batch, device,
                     max_new_tokens=200, beam_size=4):
    """Beam-search generation using the GraMFormerV2 interface."""
    print("Generating summary (beam search, beam={})...\n".format(beam_size))

    b_input_ids  = batch['input_ids'].to(device)
    b_adj        = batch['adj_matrix'].to(device)
    b_act_mask   = batch['action_mask'].to(device)
    b_dial_mask  = batch['dial_mask'].to(device)
    b_ent_mask   = batch['ent_mask'].to(device)
    b_head_mask  = batch['head_mask'].to(device)
    b_causal_adj = batch['causal_adj'].to(device)
    b_cs_adj     = batch['char_state_adj'].to(device)
    b_idf_w      = batch['idf_weights'].to(device)

    # Encode once
    aligned_memory, _ = model(
        b_input_ids, b_adj, b_act_mask, b_dial_mask,
        b_ent_mask, b_head_mask, b_causal_adj, b_cs_adj,
        target_ids=None, triplets=None, idf_weights=b_idf_w,
    )  # aligned_memory: [1, S, D]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    mem_pad_mask  = (b_input_ids[:, :, 0] == pad_id)
    enc_attn_mask = (~mem_pad_mask).long()

    # Beam search
    beams     = [(0.0, [tokenizer.bos_token_id or 0])]
    completed = []

    for _ in range(max_new_tokens):
        new_beams = []
        for score, tokens in beams:
            if tokens[-1] == eos_id:
                completed.append((score, tokens))
                continue

            t_ids  = torch.tensor([tokens], dtype=torch.long, device=device)
            t_mask = (t_ids != pad_id).long()

            dec_out = model.bart_decoder(
                input_ids=t_ids,
                attention_mask=t_mask,
                encoder_hidden_states=aligned_memory,
                encoder_attention_mask=enc_attn_mask,
            )
            logits   = model.head(dec_out.last_hidden_state[:, -1, :]).float()
            log_prob = F.log_softmax(logits, dim=-1).squeeze(0)

            # No-repeat trigram penalty
            if len(tokens) >= 3:
                ngrams = {tuple(tokens[k:k + 3]) for k in range(len(tokens) - 2)}
                for ng in ngrams:
                    if len(ng) == 3:
                        log_prob[ng[-1]] = -1e4  # F7: avoid -inf NaN in bfloat16

            top_vals, top_idx = log_prob.topk(beam_size)
            for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
                new_beams.append((score + v, tokens + [idx]))

        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
        if not beams:
            break

    if completed:
        best = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))
    else:
        best = max(beams, key=lambda x: x[0])

    return tokenizer.decode(best[1], skip_special_tokens=True)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CHECKPOINT_PATH = "/dev/shm/karan/checkpoints/gramformer_v2_latest.pt"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    print("Initializing model architecture...")
    model = GraMFormerV2(
        vocab_size=len(tokenizer),
        d_model=1024,   # must match checkpoint (1024=bart-large, 768=bart-base)
        num_layers=4,
        tokenizer=tokenizer,
    ).to(device)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        needs_lora = any('lora' in k for k in state_dict.keys())
        if needs_lora:
            print("Auto-detected LoRA weights — applying PEFT wrapper...")
            lora_config = LoraConfig(
                r=16, lora_alpha=32,
                target_modules=["in_proj", "x_proj", "out_proj", "dt_proj"],
                lora_dropout=0.05, bias="none",
            )
            model.encoder = get_peft_model(model.encoder, lora_config)

        model.load_state_dict(state_dict, strict=False)
        print("Model loaded.")
    else:
        print(f"Checkpoint {CHECKPOINT_PATH} not found — running with random weights.")

    model.eval()

    # ── Load one test movie ───────────────────────────────────────────────────
    eval_data_path = "/tmp/karan/mensa_test_data.jsonl.gz"
    print(f"\nLoading test movie from {eval_data_path}...")

    movie_scenes = []
    target_movie_name = None
    max_scenes = 200

    with gzip.open(eval_data_path, 'rt', encoding='utf-8') as f:
        for line in f:
            scene_data = json.loads(line)
            current_movie = scene_data["movie_id"].split("_Scene_")[0]

            if target_movie_name is None:
                target_movie_name = current_movie

            if current_movie == target_movie_name:
                for k in ['input_ids', 'target_ids']:
                    scene_data[k] = torch.tensor(scene_data[k], dtype=torch.long)
                scene_data['adjacency_matrix'] = torch.tensor(
                    scene_data['adjacency_matrix'], dtype=torch.int8
                )
                for k in ['action_mask', 'dialogue_mask', 'entity_mask', 'header_mask']:
                    if k in scene_data:
                        scene_data[k] = torch.tensor(scene_data[k], dtype=torch.bool)
                scene_data.setdefault('graph_triplets', [])
                scene_data.setdefault('character_emotions', {})
                scene_data.setdefault('scene_meta', {})
                movie_scenes.append(scene_data)
            else:
                break

    if len(movie_scenes) > max_scenes:
        movie_scenes = movie_scenes[:max_scenes]

    print(f"Loaded '{target_movie_name}' ({len(movie_scenes)} scenes).")

    # ── Build movie-level graphs ──────────────────────────────────────────────
    causal_adj    = build_causal_graph(movie_scenes, max_scenes)
    char_state_adj = build_char_state_graph(movie_scenes, max_scenes)

    # IDF weights — zeros is a valid fallback (DynamicGraphModulator handles None)
    idf_weights = torch.zeros(max_scenes, max_scenes)

    mock_item = {
        "movie_name":     target_movie_name,
        "scenes":         movie_scenes,
        "causal_adj":     causal_adj,
        "char_state_adj": char_state_adj,
        "idf_weights":    idf_weights,
    }

    batch = movie_collate_fn([mock_item])

    # ── Generate ──────────────────────────────────────────────────────────────
    summary = generate_summary(model, tokenizer, batch, device)

    print("\n" + "=" * 80)
    print("GENERATED SUMMARY:")
    print("=" * 80)
    print(summary)
    print("=" * 80)


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import json
import gzip
import os
from transformers import AutoTokenizer

# --- Import your exact classes from your files ---
from sum import GraMFormer
from train import movie_collate_fn
from peft import LoraConfig, get_peft_model

def build_movie_graph(scenes, max_scenes):
    """Replicates the graph builder from your MovieGraphDataset directly in RAM"""
    num_scenes = len(scenes)
    movie_adj = torch.zeros((max_scenes, max_scenes))
    
    # 1. Heartbeat (Self-loops)
    for i in range(num_scenes):
        movie_adj[i, i] = 1.0 
        
    # 2. Character Wormholes
    for i in range(num_scenes):
        for j in range(i + 1, num_scenes):
            trips_i = scenes[i].get('graph_triplets', [])
            trips_j = scenes[j].get('graph_triplets', [])
            if not trips_i or not trips_j: continue
            
            ents_i = set()
            for t in trips_i:
                parts = t.split('_')
                if len(parts) >= 1: 
                    ents_i.add(parts[0].replace("NOT ", "").replace("not ", "").strip().lower())
                if len(parts) >= 3: 
                    ents_i.add(parts[2].strip().lower())
            
            ents_j = set()
            for t in trips_j:
                parts = t.split('_')
                if len(parts) >= 1: 
                    ents_j.add(parts[0].replace("NOT ", "").replace("not ", "").strip().lower())
                if len(parts) >= 3: 
                    ents_j.add(parts[2].strip().lower())
            
            ents_i.discard("")
            ents_j.discard("")
            
            if ents_i.intersection(ents_j):
                movie_adj[i, j] = 1.0 
                movie_adj[j, i] = 1.0 
                
    return movie_adj

def generate_summary(model, tokenizer, batch, device, max_tokens=400, temperature=0.8, top_k=50, rep_penalty=1.2):
    print("🧠 Generating Summary Autoregressively...\n")
    
    input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
    generated_tokens = []
    
    with torch.no_grad():
        # 1. Extract Batch Data
        b_input_ids = batch['input_ids'].to(device)
        b_adj = batch['adj_matrix'].to(device)
        b_act_mask = batch['action_mask'].to(device)
        b_dial_mask = batch['dial_mask'].to(device)
        b_ent_mask = batch['ent_mask'].to(device)
        b_head_mask = batch['head_mask'].to(device)
        b_movie_adj = batch['movie_adj'].to(device)
        
        # 2. Map the Movie in native FP32
        # --- ARCHITECTURE FIX: It now returns the condensed 120-scene Concept Vectors! ---
        movie_level_memory, _ = model(
            b_input_ids, b_adj, b_act_mask, b_dial_mask, 
            b_ent_mask, b_head_mask, b_movie_adj, target_ids=None
        )
        
        # --- ARCHITECTURE FIX: Create a mask to hide the padded blank scenes ---
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        memory_key_padding_mask = (b_input_ids[:, :, 0] == pad_id)
            
        # 3. Autoregressive Loop in native FP32
        for step in range(max_tokens):
            
            # Embed current text
            dec_embs = model.embedding(input_ids)
            
            # --- ARCHITECTURE FIX: Add Positional Encodings to generation! ---
            seq_length = input_ids.size(1)
            positions = torch.arange(0, seq_length, dtype=torch.long, device=device)
            pos_embs = model.pos_embedding(positions).unsqueeze(0)
            dec_embs = dec_embs + pos_embs 
            # -----------------------------------------------------------------
            
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=device)
            
            # Decode using the intelligent pooled memory and padding mask
            decoder_out = model.seq2seq_decoder(
                tgt=dec_embs,
                memory=movie_level_memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            logits = model.head(decoder_out) 
            
            next_token_logits = logits[:, -1, :].squeeze().float() 
            
            if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                print("\n⚠️ Math Spike detected during generation. Stopping early.")
                break
                
            next_token_logits = next_token_logits / temperature
            
            # Repetition penalty
            for prev_token in set(generated_tokens[-20:]): 
                if next_token_logits[prev_token] < 0:
                    next_token_logits[prev_token] *= rep_penalty
                else:
                    next_token_logits[prev_token] /= rep_penalty
            
            v, _ = torch.topk(next_token_logits, top_k)
            next_token_logits[next_token_logits < v[-1]] = -float('Inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated_tokens.append(next_token)
            
            new_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(device)
            input_ids = torch.cat([input_ids, new_token_tensor], dim=-1)
            
            if next_token == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 1. Set the Checkpoint to Test ---
    # Change this to whatever the newest checkpoint is once your new Epoch 1 finishes!
    CHECKPOINT_PATH = "/dev/shm/karan/checkpoints/gramformer_epoch_1.pt" 
    
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    print("Initializing Model Architecture...")
    model = GraMFormer(vocab_size=len(tokenizer), d_model=768, num_layers=4).to(device)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading Weights from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # --- AUTO-LORA DETECTION ---
        needs_lora = any('lora' in k for k in state_dict.keys())
        if needs_lora:
            print("🔧 Auto-detected LoRA weights. Applying PEFT wrapper...")
            lora_config = LoraConfig(
                r=16, lora_alpha=32, target_modules=["in_proj", "x_proj", "out_proj", "dt_proj"], 
                lora_dropout=0.05, bias="none"
            )
            model.encoder = get_peft_model(model.encoder, lora_config)
        
        # Load safely
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model loaded successfully!")
    else:
        print(f"⚠️ Checkpoint {CHECKPOINT_PATH} not found. Running with untained weights for testing.")
        
    model.eval() 
    
    # --- 2. Stream One Movie from Test Set (Bypass HF Cache) ---
    print("\nLoading a single Test Movie directly from JSONL...")
    eval_data_path = "/tmp/karan/mensa_test_data.jsonl.gz" 
    
    movie_scenes = []
    target_movie_name = None
    
    with gzip.open(eval_data_path, 'rt', encoding='utf-8') as f:
        for line in f:
            scene_data = json.loads(line)
            current_movie = scene_data["movie_id"].split("_Scene_")[0]
            
            if target_movie_name is None:
                target_movie_name = current_movie
                
            if current_movie == target_movie_name:
                # Convert raw lists to PyTorch tensors so collate_fn doesn't crash
                for k in ['input_ids', 'target_ids']:
                    scene_data[k] = torch.tensor(scene_data[k], dtype=torch.long)
                scene_data['adjacency_matrix'] = torch.tensor(scene_data['adjacency_matrix'], dtype=torch.int8)
                for k in ['action_mask', 'dialogue_mask', 'entity_mask', 'header_mask']:
                    if k in scene_data:
                        scene_data[k] = torch.tensor(scene_data[k], dtype=torch.bool)
                
                # Make sure triplets exist for the builder
                if 'graph_triplets' not in scene_data:
                    scene_data['graph_triplets'] = []
                
                movie_scenes.append(scene_data)
            else:
                break # Stop reading! We have our movie.
                
    # Truncate to Max Scenes
    max_scenes = 200
    if len(movie_scenes) > max_scenes:
        movie_scenes = movie_scenes[:max_scenes]
        
    print(f"✅ Extracted '{target_movie_name}' ({len(movie_scenes)} scenes) without disk cache!")
    
    # Build Graph Matrix
    movie_adj = build_movie_graph(movie_scenes, max_scenes)
    
    # Package exactly how collate_fn expects it
    mock_dataset_item = {
        "movie_name": target_movie_name,
        "scenes": movie_scenes,
        "movie_adj": movie_adj
    }
    
    # Call your collate function directly
    sample_batch = movie_collate_fn([mock_dataset_item])
    
    # --- 3. Generate! ---
    summary = generate_summary(model, tokenizer, sample_batch, device)
    
    print("\n" + "="*80)
    print("🎬 FINAL GENERATED SUMMARY:")
    print("="*80)
    print(summary)
    print("="*80)

if __name__ == "__main__":
    main()
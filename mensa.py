import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import re

class MensaGraphDataset(Dataset):
    def __init__(self, jsonl_filepath, max_seq_len=512):
        self.jsonl_filepath = jsonl_filepath
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        self.max_seq_len = max_seq_len
        
        self.line_offsets = []
        self.movie_ids = [] # <-- NEW: Store IDs here instantly
        
        movie_id_pattern = re.compile(r'"movie_id"\s*:\s*"([^"]+)"')
        
        print(f"Indexing Phase 3 Hierarchical Data from {jsonl_filepath} (Lazy Load)...")
        with open(jsonl_filepath, 'rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.line_offsets.append(offset)
                    # Extract ID with regex without parsing JSON
                    match = movie_id_pattern.search(line.decode('utf-8', errors='ignore'))
                    self.movie_ids.append(match.group(1) if match else "unknown_movie")
                    
        print(f"Indexed {len(self.line_offsets)} scenes with 4-way modality alignment.")

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, idx):
        with open(self.jsonl_filepath, 'r', encoding='utf-8') as f:
            f.seek(self.line_offsets[idx])
            line = f.readline()
            item = json.loads(line)
        
        # 1. Load Pre-Aligned Tokens
        raw_input = item['input_ids'][:self.max_seq_len]
        raw_target = item['target_ids'][:self.max_seq_len]
        
        input_ids = torch.full((self.max_seq_len,), 1, dtype=torch.long)
        target_ids = torch.full((self.max_seq_len,), 1, dtype=torch.long)
        
        input_ids[:len(raw_input)] = torch.tensor(raw_input, dtype=torch.long)
        target_ids[:len(raw_target)] = torch.tensor(raw_target, dtype=torch.long)
        
        # 2. Adjacency Matrix
        adj_raw = torch.tensor(item['adjacency_matrix'], dtype=torch.int8)
        adjacency_matrix = torch.zeros((self.max_seq_len, self.max_seq_len), dtype=torch.int8)
        limit = min(self.max_seq_len, adj_raw.size(0))
        adjacency_matrix[:limit, :limit] = adj_raw[:limit, :limit]
        
        # 3. Load the 4 Hierarchical Modality Masks
        def get_padded_mask(key):
            mask_list = item.get(key, [0] * self.max_seq_len)
            mask_tensor = torch.tensor(mask_list[:self.max_seq_len], dtype=torch.bool)
            if mask_tensor.size(0) < self.max_seq_len:
                padding = torch.zeros(self.max_seq_len - mask_tensor.size(0), dtype=torch.bool)
                mask_tensor = torch.cat([mask_tensor, padding])
            return mask_tensor

        return {
            'movie_id': item.get('movie_id', 'unknown_movie'),
            'input_ids': input_ids,
            'target_ids': target_ids,
            'adjacency_matrix': adjacency_matrix,
            'action_mask': get_padded_mask('action_mask'),
            'dialogue_mask': get_padded_mask('dialogue_mask'),
            'entity_mask': get_padded_mask('entity_mask'),
            'header_mask': get_padded_mask('header_mask'),
            'graph_triplets': item.get('graph_triplets', []),
            'character_emotions': item.get('character_emotions', {}),
            'scene_meta': item.get('scene_meta', {}),
        }
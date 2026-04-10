# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Dual-Tower Dynamic Hypergraph Summariser (v3)** for screenplay/movie summarization. It processes movies scene-by-scene, builds a dynamic hypergraph over entity states, and generates summaries using a BART-based decoder fused with Mamba-style scene encoding.

Primary dataset: **MovieSum** (~2,200 movies). Secondary: **MENSA** (~500 movies). MovieSum-trained models are also evaluated zero-shot on MENSA.

## Pipeline Stages

### 1. Data Extraction (`emnlp_extractor.py`)
Converts raw MENSA/MovieSum HuggingFace datasets into compressed JSONL feature files.

```bash
python emnlp_extractor.py --dataset mensa --out /tmp/uday/mensa_train_data.jsonl.gz
python emnlp_extractor.py --dataset moviesum --out /tmp/uday/moviesum_data.jsonl.gz
python emnlp_extractor.py --start 0 --end 5000 --out /tmp/uday/shard_0.jsonl.gz  # sharded
```

### 2. Training (`train.py`)
```bash
# Full model training (MovieSum by default)
python train.py --run_name full_model

# Ablation flags
python train.py --run_name ablation_no_hypergraph   --no_hypergraph       # text-only baseline
python train.py --run_name ablation_static          --static_hypergraph   # DiscoGraMS-style baseline
python train.py --run_name ablation_no_gru          --no_gru              # EMA instead of GRU updates
python train.py --run_name ablation_no_raft         --no_raft             # no cross-modal fusion
python train.py --run_name ablation_no_pointer      --no_pointer_head
python train.py --run_name ablation_no_coherence    --no_coherence_loss
python train.py --run_name ablation_no_contrastive  --no_contrastive_loss
python train.py --run_name discograms_baseline      --static_hypergraph --no_gru  # paper baseline

# Key flags
--d_model 1024          # must match BART variant (1024=large, 768=base)
--num_layers 4          # number of MambaBlock layers
--entity_penalty 3.0    # weight for entity consistency loss term
--dataset moviesum|mensa|both
```

Checkpoints: `/tmp/uday/checkpoints/`. Split files: `/tmp/uday/train_{run_name}.jsonl` and `eval_{run_name}.jsonl`.

### 3. Inference (`inference.py`)
```bash
python inference.py
# Edit CHECKPOINT_PATH inside main() to point to the desired checkpoint
```

### 4. Evaluation (`eval.py`)
Standalone triplet extraction evaluation (precision/recall/F1):
```bash
python eval.py
```

### 5. Visualization (`visualize_graph.py`)
Utility for visualizing the movie hypergraph as a bipartite graph (entities ↔ scenes):
```python
from visualize_graph import plot_movie_hypergraph
plot_movie_hypergraph(incidence_matrix, entity_names, movie_name="...", save_path="out.png")
```

## Architecture (`sum.py`)

**`DualTowerHypergraphSummariser`** is the main model class with two towers fused via cross-attention.

### Tower 1 — Text Stream
`Frozen RoBERTa (768d) → scene_proj (768→d_model) → MambaBlock → RaftConsensusAttentionV2`

- **MambaBlock**: Stack of `num_layers` selective SSM layers with no token-level graph routing (removed in v2 due to O(seq²) OOM).
- **RaftConsensusAttentionV2**: Cross-modal attention across 4 modalities (action, dialogue, entity, header) with gated consensus fusion. Output: `H_text [B, S, L, d_model]`.
- LoRA (Stage 2) targets Mamba's `in_proj`, `x_proj`, `out_proj`, `dt_proj`. RoBERTa is always frozen.

### Tower 2 — Dynamic 3-Stream Hypergraph (DHEG)
`DynamicHypergraphTower` processes entities sequentially across scenes using an incidence matrix with **float role weights** (not binary):
- `1.0` = active speaker, `0.7` = SVO subject, `0.5` = SVO object, `0.3` = background mention

Three message streams per scene, fused via learnable softmax weights:
1. **Scene stream**: What is happening in this scene (from latent hyperedge).
2. **Arc stream**: Historical context from past scenes sharing these entities (temporal decay).
3. **Interaction stream**: Social context from co-occurring entities (role-weighted co-occurrence).

Entity states updated via `GRUCell` only for entities present in the current scene. `edge_type_ids` (hardcoded `NEUTRAL`) are kept for API compatibility but **ignored** — edges are fully latent.

### Fusion
Deep cross-attention over `[H_text_pooled, H_hyperedges, h_entities]` → BART decoder.

### Losses
- `RelationalEventConsistencyLoss`: label-smoothed entity NLL, weighted by `--entity_penalty`.
- `NarrativeCoherenceLoss`: scene-pair NT-Xent where positive pairs = scenes with Jaccard similarity > 0.25 in the incidence matrix (entities in common).

## Data Format

Each line in `.jsonl.gz` files is a single scene:
- `movie_id`: `"MovieName_Scene_NNN"`
- `input_ids`, `target_ids`: RoBERTa-tokenized, padded to `max_seq_len` (256 in v3, was 512)
- `action_mask`, `dialogue_mask`, `entity_mask`, `header_mask`: 4-way modality masks (bool)
- `graph_triplets`: list of `"Subject_Verb_Object"` strings (negation prefixed with `"NOT "`)
- `characters`: speaker names from screenplay ALL-CAPS tags
- `ner_entities`: list of `{text, type}` dicts (NER output)
- `hyperedge_type`: dominant scene type string (deprecated; model uses latent edges)
- `character_emotions`: dict of character → float polarity
- `scene_meta`: `{dialogue_density, action_density, has_int, has_ext}`

**`MovieHypergraphDataset`** groups scenes by movie and builds per-movie tensors:
- `incidence_matrix [MAX_ENTITIES=100, MAX_SCENES=64]` float — role-weighted
- `entity_type_ids [100]` long — from `ENTITY_TYPE_MAP` (PERSON/ORG/GPE/FACILITY/OTHER)
- `edge_type_ids [64]` long — legacy; always `NEUTRAL` (4)
- `entity_mask [100]` bool

Entity registry priority per scene: NER entities > speaker names > SVO subjects/objects. Movies with >64 scenes are stride-sampled.

**`SceneDataset`** requires **uncompressed** `.jsonl` — uses byte-offset seeking for O(1) random access. The source `.jsonl.gz` stays compressed; `split_dataset_by_movie()` inflates it into per-split plain `.jsonl` files. If you see a `ValueError` about `.gz` files, delete old `.gz` split files and re-run.

## Key Paths (hardcoded)

| Path | Purpose |
|------|---------|
| `/tmp/uday/` | Main data and checkpoints directory |
| `/tmp/uday/bart-large` | Local BART-large weights (checked before HF hub) |
| `/tmp/uday/moviesum_data.jsonl.gz` | Primary training data |
| `/tmp/uday/mensa_train_data.jsonl.gz` | Secondary training data |
| `/tmp/uday/mensa_test_data.jsonl.gz` | Test data |
| `/tmp/uday/checkpoints/` | Model checkpoints |

## Training Stages

- **Stage 1** (2 epochs): RoBERTa + Mamba frozen; train hypergraph tower + decoder. LR=1e-4.
- **Stage 2** (20 epochs): LoRA on Mamba; full model except frozen RoBERTa. LR=1e-5.
- Evaluation every epoch: ROUGE + BERTScore + METEOR + entity F1.

## Dependencies

Core: `torch`, `transformers`, `peft` (LoRA), `spacy` + `en_core_web_sm`, `fastcoref`, `datasets`, `wandb`, `evaluate`, `tqdm`, `matplotlib`, `seaborn`, `networkx`

Sentiment model: `cardiffnlp/twitter-roberta-base-sentiment-latest`

## Ablation System

All components are ablation-switchable via `ABLATION` dict in `train.py`. Each `--no_*` / `--static_*` flag disables one component. Use `--run_name` to distinguish W&B runs and output split files. Stream weight logs (`stream_weight/scene`, `stream_weight/arc`, `stream_weight/interaction`) in W&B show which hypergraph streams the model relies on.

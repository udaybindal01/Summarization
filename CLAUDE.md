# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **LED + Mamba-Hypergraph Narrative Summariser (v5)** for screenplay/movie summarization. It uses a LED encoder for long-context text processing (16K tokens), a dynamic hypergraph with **LED-grounded entity initialization** and **per-entity Mamba SSM trajectories** for tracking character state evolution, and a LED decoder for generation.

Primary dataset: **MovieSum** (~2,200 movies). Secondary: **MENSA** (~500 movies). MovieSum-trained models are also evaluated zero-shot on MENSA.

## Pipeline Stages

### 1. Data Extraction (`emnlp_extractor.py`)
Converts raw MENSA/MovieSum HuggingFace datasets into compressed JSONL feature files. Includes:
- `clean_text` and `summary_text` for LED re-tokenization
- **Movie-level coreference resolution** via fastcoref — groups mention chains (e.g., "the detective", "John", "he") into canonical entity names stored as `coref_entities` per scene.

```bash
python emnlp_extractor.py --dataset mensa --out /tmp/uday/mensa_train_data.jsonl.gz
python emnlp_extractor.py --dataset moviesum --out /tmp/uday/moviesum_data.jsonl.gz
```

### 2. Training (`train.py`)
```bash
# Full model training (MovieSum by default)
python train.py --run_name full_model_v5

# Key ablation flags
python train.py --run_name led_only          --no_hypergraph        # LED-only baseline
python train.py --run_name static_graph      --static_hypergraph    # static graph baseline
python train.py --run_name no_mamba_entity   --no_mamba_entity      # GRU instead of Mamba (ablation)
python train.py --run_name no_coherence      --no_coherence_loss
python train.py --run_name no_contrastive    --no_contrastive_loss
python train.py --run_name global_streams    --no_adaptive_streams
python train.py --run_name no_names          --no_entity_names
python train.py --run_name no_edgedrop       --edge_dropout 0.0

# Key flags
--d_model 1024          # must match LED variant (1024=large)
--mamba_layers 2        # number of EntityMamba layers for temporal dynamics
--entity_penalty 3.0    # weight for entity consistency loss term
--dataset moviesum|mensa|both
--edge_dropout 0.1      # incidence matrix edge dropout (0 = disabled)
```

Checkpoints: `/tmp/uday/checkpoints/led_mamba_*.pt`. Split files: `/tmp/uday/train_{run_name}.jsonl` and `eval_{run_name}.jsonl`.

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

**`LEDMambaHypergraphSummariser`** is the main model class.

### Tower 1 — Text Stream (LED encoder)
`Full screenplay (16K tokens) → LED encoder → scene boundary pooling → H_text [B, S, d_model]`

- LED encoder handles full screenplay as one sequence with `</s>` scene separators
- Global attention on separator tokens and BOS for cross-scene reasoning
- LED encoder frozen in Stage 1; global attention layers unfrozen in Stage 2
- Scene-level representations extracted via boundary pooling

### Tower 2 — Dynamic 3-Stream Hypergraph with LED-grounded init + Mamba Temporal Dynamics
`DynamicHypergraphTower` with **EntityMambaBlock** for temporal entity state tracking.

**LED-grounded entity initialization**: Entity states initialized by pooling LED encoder scene representations weighted by incidence roles (not random embeddings). Combined with type embeddings and optional name embeddings.

Incidence matrix uses **float role weights** (not binary):
- `1.0` = active speaker, `0.7` = SVO subject, `0.5` = SVO object, `0.3` = background mention

Three message streams per scene, fused via scene-conditioned adaptive gating:
1. **Scene stream**: Entity-aware bilinear attention — each entity extracts different information from the scene based on its identity.
2. **Arc stream**: Attention over past shared hyperedges with learnable keys/values and temporal decay bias.
3. **Interaction stream**: Social context from co-occurring entities.

**Key novelty**: 
- Entity states grounded in LED encoder output (not random init)
- Entity-aware scene messages via bilinear attention (not broadcast)
- Temporal dynamics via per-entity Mamba SSM trajectories
- Coreference-resolved entity nodes (from extractor)
- **dt values are interpretable** as state change magnitude → automatic narrative turning point detection

LoRA (Stage 2) targets entity Mamba's `in_proj`, `x_proj`, `out_proj`, `dt_proj`.

### Fusion & Decoder
- Gated fusion: `gate * H_text + (1-gate) * H_hyperedges`
- CrossAttentionAdapter: bidirectional cross-attention between scenes and entity nodes
- LED decoder: fully trainable, matched to LED encoder (no distribution mismatch)
- No pointer head — LED decoder handles entity name generation natively

### Losses
- `RelationalEventConsistencyLoss`: label-smoothed entity NLL, weighted by `--entity_penalty`.
- `NarrativeCoherenceLoss`: scene-pair NT-Xent where positive pairs = scenes with Jaccard similarity > 0.25.

## Data Format

Each line in `.jsonl.gz` files is a single scene:
- `movie_id`: `"MovieName_Scene_NNN"`
- `input_ids`, `target_ids`: BART-tokenized (legacy, used as fallback)
- `clean_text`: raw scene text (for LED re-tokenization)
- `summary_text`: raw summary text (for LED re-tokenization)
- `coref_entities`: dict mapping mention → canonical entity name (from fastcoref)
- `action_mask`, `dialogue_mask`, `entity_mask`, `header_mask`: 4-way modality masks
- `graph_triplets`: list of `"Subject_Verb_Object"` strings
- `characters`: speaker names from screenplay ALL-CAPS tags
- `ner_entities`: list of `{text, type}` dicts (NER output)
- `character_emotions`: dict of character → float polarity
- `scene_meta`: `{dialogue_density, action_density}`

**`MovieHypergraphDataset`** groups scenes by movie and:
1. Concatenates all scene texts with `</s>` separator for LED (up to 16384 tokens)
2. Records scene boundary positions for pooling LED output
3. Builds hypergraph incidence matrix from entity mentions (with coref resolution)
4. Collects per-scene triplets (used in loss, not model)

## Key Paths (hardcoded)

| Path | Purpose |
|------|---------|
| `/tmp/uday/` | Main data and checkpoints directory |
| `/tmp/uday/led-large-16384` | Local LED-large weights (checked before HF hub) |
| `/tmp/uday/moviesum_data.jsonl.gz` | Primary training data |
| `/tmp/uday/mensa_train_data.jsonl.gz` | Secondary training data |
| `/tmp/uday/mensa_test_data.jsonl.gz` | Test data |
| `/tmp/uday/checkpoints/` | Model checkpoints |

## Training Stages

- **Stage 1** (3 epochs): LED encoder frozen; train hypergraph tower + LED decoder + fusion. LR=1e-4 (new layers), 2e-5 (LED decoder).
- **Stage 2** (20 epochs): LoRA on entity Mamba; LED encoder global attention unfrozen. LR=1e-5 (LoRA).
- Evaluation every epoch: ROUGE + METEOR + entity F1 + dt heatmaps.

## Dependencies

Core: `torch`, `transformers`, `peft` (LoRA), `spacy` + `en_core_web_sm`, `fastcoref`, `datasets`, `wandb`, `evaluate`, `tqdm`, `matplotlib`, `seaborn`, `networkx`

Sentiment model: `cardiffnlp/twitter-roberta-base-sentiment-latest`

## Ablation System

All components are ablation-switchable via `ABLATION` dict in `train.py`. Each `--no_*` / `--static_*` flag disables one component. Use `--run_name` to distinguish W&B runs and output split files.

Key ablation comparisons for the paper:
1. LED-only (no hypergraph) — shows hypergraph contribution
2. LED + static graph — shows dynamic updates matter
3. LED + GRU updates — shows Mamba > GRU for temporal dynamics
4. LED + Mamba (full) — full model

Stream weight logs and **entity dt heatmaps** in W&B show which streams the model relies on and when entity states change significantly.

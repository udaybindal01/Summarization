# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **GraM-Former v2** — a graph-augmented transformer model for screenplay/movie summarization. It processes movies scene-by-scene, builds graph representations, and generates summaries using a BART-based encoder-decoder with custom Mamba-style scene encoding.

## Pipeline Stages

### 1. Data Extraction (`emnlp_extractor.py`)
Converts raw MENSA/MovieSum HuggingFace datasets into compressed JSONL feature files.

```bash
# Extract MENSA training data (supports sharding with --start/--end)
python emnlp_extractor.py --dataset mensa --out /tmp/karan/mensa_train_data.jsonl.gz

# Extract a slice (for parallel runs)
python emnlp_extractor.py --start 0 --end 5000 --out /tmp/karan/shard_0.jsonl.gz

# Extract MovieSum
python emnlp_extractor.py --dataset moviesum --out /tmp/karan/moviesum_data.jsonl.gz
```

### 2. Training (`train.py`)
```bash
# Full model training
python train.py --run_name full_model

# Ablation experiments (disable individual components)
python train.py --run_name ablation_no_causal --no_causal_graph
python train.py --run_name ablation_no_coherence --no_coherence_loss
python train.py --run_name ablation_single_graph --single_binary_graph

# Key flags
--d_model 1024          # must match BART variant (1024=large, 768=base)
--entity_penalty 3.0    # weight for entity consistency loss term
--dataset mensa|moviesum|both
```

Checkpoints are saved to `/dev/shm/karan/checkpoints/` (RAM disk for speed).

### 3. Inference (`inference.py`)
```bash
python inference.py
# Edit CHECKPOINT_PATH inside main() to point to the desired checkpoint
# Reads test data from /tmp/karan/mensa_test_data.jsonl.gz
```

### 4. Evaluation (`eval.py`)
Standalone triplet extraction evaluation (precision/recall/F1 for graph edges):
```bash
python eval.py
```

## Architecture (`sum.py`)

**GraMFormerV2** is the main model class. Key components:

- **RoBERTa (frozen) → `scene_proj` → GraMambaLayer**: Scene-level encoder pipeline. Frozen `roberta-base` (768d) provides pre-trained contextual token representations. A linear bridge projects to `d_model` (identity if 768, `nn.Linear(768, 1024)` for BART-large). Mamba SSM with graph-routing then operates on these projections. LoRA (Stage 2) targets the Mamba layers: `in_proj`, `x_proj`, `out_proj`, `dt_proj`. RoBERTa stays frozen in both stages.
- **HeterogeneousGraphTransformer (HGT)**: Fuses two typed movie-level adjacency matrices — causal event graph and character state graph — via type-aware message passing (2 heads).
- **RaftConsensusAttentionV2**: Cross-modal attention across 4 modalities (action, dialogue, entity, header), with gated fusion and Riemannian orthogonality penalty.
- **DynamicGraphModulator v2**: IDF-weighted edge pruning on entity co-occurrence graphs.
- **HierarchicalPointerHead v2**: Dual-level copy mechanism (scene attention + entity salience).
- **Losses**: `RelationalEventConsistencyLoss` (label-smoothed entity NLL) + `NarrativeCoherenceLoss` (scene-pair NT-Xent over encoder scene reps; positive pairs = scenes sharing a causal edge within the movie).

BART backbone: defaults to `facebook/bart-large` (d_model=1024). Can switch to `facebook/bart-base` (d_model=768) via `--bart_model` flag. Local path `/tmp/karan/bart-large` is checked first to avoid network access. Tokenizer defaults to `bart_model` but can be set independently via `--bart_tokenizer`. The extraction pipeline (`emnlp_extractor.py`) and dataset loader both use the BART tokenizer to ensure decoder start token semantics match (C1/C2).

## Data Format (`mensa.py` / JSONL)

Each line in the `.jsonl.gz` files is a single scene with:
- `movie_id`: `"MovieName_Scene_NNN"`
- `input_ids`, `target_ids`: RoBERTa-tokenized, padded to `max_seq_len=512`
- `adjacency_matrix`: token-level dependency graph `[512×512]` as int8
- `action_mask`, `dialogue_mask`, `entity_mask`, `header_mask`: 4-way modality masks (bool)
- `graph_triplets`: list of `"Subject_Verb_Object"` strings (negation prefixed with `"NOT "`)
- `scene_meta`: `{dialogue_density, action_density, has_int, has_ext}`
- `character_emotions`: dict of character → float polarity score

Movies are grouped in the collate function (`movie_collate_fn` in `train.py`) into batches where each batch is one movie (variable number of scenes).

## Key Paths (hardcoded)

| Path | Purpose |
|------|---------|
| `/tmp/karan/` | Main data directory |
| `/dev/shm/karan/` | RAM disk — checkpoints, fast I/O |
| `/tmp/karan/bart-large` | Local BART-large weights |
| `/tmp/karan/mensa_train_data.jsonl.gz` | Extracted train features |
| `/tmp/karan/mensa_test_data.jsonl.gz` | Extracted test features |
| `/dev/shm/karan/checkpoints/` | Model checkpoints |

## Dependencies

Core: `torch`, `transformers`, `peft` (LoRA), `spacy` + `en_core_web_sm`, `fastcoref`, `datasets`, `wandb`, `evaluate`, `tqdm`, `matplotlib`, `seaborn`

Sentiment model: `cardiffnlp/twitter-roberta-base-sentiment-latest`

## Ablation System

All components in `sum.py` are ablation-switchable via `ABLATION` dict in `train.py`. Each `--no_*` flag disables one architectural component. Use `--run_name` to distinguish W&B runs and output split files.

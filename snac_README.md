# SNaC — Streaming Narrative-state Compression

A research-ready reimplementation for long-form movie summarization. It replaces the
v5 LED+Mamba-hypergraph system (which was structurally unable to learn) with a clean,
novel mechanism:

- **Streaming, entity-bound story-state.** A fixed set of `M = max_entities + free_slots`
  slots is carried scene-by-scene. Entity slots are bound to screenplay characters and
  their writes are **addressable** — a slot only updates on scenes where its entity
  appears, so distant facts survive (effective path length = #character-scenes, not
  #scenes). This is what lets a betrayal planted in scene 20 reach the summary.
- **Two-source decoder.** The decoder cross-attends to `[ final state slots (global,
  compressed) ; retrieved top-k scenes at full token resolution (local, verbatim) ]`.
  Global carries the arc; local gives faithful surface wording.
- **Probe + turning-point signals.** An entity-presence probe forces the state to be
  decodable (interpretability); `‖S_t − S_{t−1}‖` per scene is logged for turning-point
  analysis. Erasing a slot at decode time changes exactly the facts it carried
  (causal-intervention result).

## Files
| file | role |
|------|------|
| `snac_data.py`  | MovieSum/MENSA loading, scene split, ALL-CAPS entity extraction, tokenization, `SNaCConfig` |
| `snac_model.py` | `StoryState` (read/write recurrence) + `SNaC` (backbone + two-source decoder + probe) |
| `snac_train.py` | full-train-set training, per-epoch ROUGE/BERTScore validation, checkpointing |
| `snac_eval.py`  | full test-set metrics + turning-point + causal-intervention artifacts (JSON) |

## Dependencies
```
torch  transformers  datasets  evaluate  rouge_score  bert_score  accelerate
peft            # only needed for --lora
```
(No spaCy/fastcoref required — entities come from screenplay ALL-CAPS cues.)

## Train on the full MovieSum train set
Run on a GPU node (needs internet or a warm HF cache for the dataset + backbone):
```bash
python3 snac_train.py \
  --backbone facebook/bart-large-cnn \
  --epochs 8 --grad_accum 8 \
  --out /scratch/karan/snac
```
- Validates on the official `validation` split every epoch; writes `snac_best.pt`
  (best ROUGE-L) and `snac_last.pt` to `--out`.
- Default: encoder frozen, decoder + state module trained. Add `--lora` to instead
  LoRA-tune the backbone attention (needs `peft`).
- Quick smoke test first: `--limit_train 50 --epochs 1 --max_eval 10`.

Backbone is swappable: `--backbone google/flan-t5-large` (stronger, modern) or
`allenai/led-large-16384` (long-context). BART-large-cnn is the default because it
carries a summarization prior and uses the exact decoder-memory API path already
verified in `prototype_bottleneck.py`.

## Evaluate on the test set
```bash
python3 snac_eval.py \
  --ckpt /scratch/karan/snac/snac_best.pt \
  --out  /scratch/karan/snac/test_report.json \
  --intervene 10
```
Writes `test_report.json` containing:
- `report.rouge` (R-1/2/L) and `report.bertscore_f1` over the **full** test split,
- `turning_points`: per-scene `‖ΔS_t‖` for the first 40 movies (act-structure figure),
- `interventions`: base vs. slot-erased summaries (controllability figure),
- `predictions`: every generated summary with its retrieved scene indices.

## Ablations (for the paper)
| flag | question |
|------|----------|
| `--free_slots 0`      | do free (non-entity) slots matter? |
| `--k_retrieve 0`*     | is the local verbatim channel necessary? (state-only decoding) |
| `--lambda_probe 0`    | does the interpretability probe help or cost metrics? |
| `--max_entities 0`*   | entity binding vs. unstructured memory |
| `--lora`              | LoRA backbone vs. frozen-encoder + trained decoder |

*`k_retrieve 0` / `max_entities 0` are analysis configs; keep at least the free slots
so the memory is non-empty.

## Notes
- One movie per forward step (variable scene count); effective batch = `--grad_accum`.
- bf16 autocast on GPU; frozen encoder runs under `no_grad` to save memory.
- This code has been syntax-checked but not run on hardware in this environment —
  expect to iterate on the first cluster run (dataset/version quirks).

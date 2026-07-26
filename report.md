---
title: "LED + Mamba-Hypergraph Narrative Summarizer (v5)"
subtitle: "A Research Report on Long-Form Movie Screenplay Summarization"
author: "Uday Bindal"
date: "May 2026"
geometry: margin=1in
fontsize: 11pt
linestretch: 1.25
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{array}
  - \usepackage{xcolor}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{\small LED + Mamba-Hypergraph Summarizer}
  - \fancyhead[R]{\small Uday Bindal · 2026}
  - \fancyfoot[C]{\thepage}
  - \definecolor{codebg}{HTML}{F5F5F5}
  - \usepackage{mdframed}
---

\newpage

# 1. Abstract

We present a dual-tower architecture for automatic narrative summarization of full-length movie screenplays. The system pairs a **Longformer Encoder-Decoder (LED)** for global 16,384-token text encoding with a **dynamic hypergraph tower** that tracks character state evolution over narrative time using **Mamba Selective State Spaces (SSMs)**. Character entities are initialized from LED encoder representations — grounded, not random — and updated through four specialized message streams per scene. The full model contains 516M parameters and is trained on MovieSum (~2,200 movies) with zero-shot transfer to MENSA (~500 movies).

After 23 epochs of training, the model achieves **ROUGE-1 = 0.187**, **ROUGE-2 = 0.005**, **ROUGE-L = 0.073**, **METEOR = 0.070**, and **Entity F1 = 0.098**. These results reflect the model running with a critical gradient blockage (dead Mamba paths) that was identified but not resolved before training completed; fixing this and retraining Stage 2 is the immediate next step and is expected to significantly improve all metrics.

---

# 2. Motivation

## 2.1 Why Screenplay Summarization is Hard

Movie screenplays are among the most structurally complex documents in natural language processing. A typical screenplay spans 90–120 pages, contains 50–150 named characters, interleaves action description with dialogue, uses stylized shorthand ("INT. OFFICE - DAY"), and encodes causal story logic across temporally separated scenes. Characters introduced in Act 1 return in Act 3; alliances and betrayals span dozens of scenes; emotional arcs are expressed through what characters *do not* say as much as what they do.

Standard summarization approaches fail here for two compounding reasons. First, **length**: a 16,000-token screenplay overwhelms standard transformer context windows (typically 512–1,024 tokens), causing important early scenes to be forgotten by the time late scenes are processed. Second, **narrative structure**: a summary of a movie is not an extraction of high-attention spans — it requires understanding *who did what to whom, and why*, tracking entity states across the full temporal arc of the story.

## 2.2 The Gap in Existing Work

Existing long-document summarizers (BigBird, LED, LongT5) handle the length problem through sparse or global attention, but treat documents as flat token sequences. They have no explicit representation of characters, no mechanism for tracking how a character's state changes from scene to scene, and no way to represent the narrative event graph that underlies the story.

Prior hypergraph-based NLP work models relational structure but typically operates on short documents with static graphs. No prior work combines long-context LED encoding with a *dynamic*, *temporally-evolving* hypergraph and Mamba SSM trajectories for entity state tracking at feature scale (d = 1,024).

## 2.3 The Core Research Question

> *Can a model that explicitly tracks named character state trajectories across the full narrative arc of a movie screenplay produce better summaries than one that treats the screenplay as a flat token sequence?*

This paper's answer is an architectural bet: yes, and the mechanism is a dynamic hypergraph whose entity nodes evolve through Selective State Space dynamics, conditioned on the global text representations produced by the LED encoder.

---

# 3. Architecture

The model consists of three components: a text tower, a hypergraph tower, and a fusion + decoder stage.

## 3.1 Tower 1 — LED Text Encoder

The full screenplay (up to 16,384 tokens) is tokenized as a single sequence with `</s>` separator tokens placed at scene boundaries. The LED encoder (400M parameters, d_model = 1,024, 16 layers) processes this sequence with **global attention** on separator tokens and the BOS token, enabling full cross-scene reasoning. Scene-level representations $H_{\text{text}} \in \mathbb{R}^{B \times S \times 1024}$ are extracted via mean pooling over each scene's token positions — a procedure called **boundary pooling**.

LED was chosen over alternatives (BigBird, LongT5) because its asymmetric sparse + global attention pattern is specifically designed for hierarchical documents where a few "anchor" tokens (scene headers, in our case) should attend to all positions while regular tokens attend locally.

## 3.2 Tower 2 — Dynamic 4-Stream Hypergraph with Mamba Temporal Dynamics

The hypergraph tower takes $H_{\text{text}}$ and processes it through a dynamic entity graph. Up to 100 named entities per movie are extracted through a pipeline combining:

- **NER** (spaCy `en_core_web_sm`) for typed entity extraction (PERSON, ORG, GPE, FACILITY)
- **fastcoref** for movie-level coreference resolution — "John", "the detective", and "he" map to a single canonical node
- **Screenplay artifact filtering** — `INT`, `EXT`, revision watermarks filtered via regex before entities enter the graph

The incidence matrix $I \in \mathbb{R}^{N \times S}$ uses **float role weights** encoding narrative role:

| Weight | Role |
|--------|------|
| 1.0 | Active speaker (ALL-CAPS character in scene header) |
| 0.7 | SVO subject of an extracted triplet |
| 0.5 | SVO object of a triplet |
| 0.3 | Background mention |

**Entity initialization** is grounded in the LED encoder. Initial entity states are computed as a weighted sum of scene representations (weighted by incidence roles), combined with type embeddings and optionally LED-encoded entity name strings:

$$h_{\text{init}}[n] = \text{proj}\!\left(\frac{\sum_s I[n,s] \cdot H_{\text{text}}[s]}{\sum_s I[n,s]}\right) + \text{type\_embed}(\text{type}[n]) + \text{name\_proj}(\text{LED}(\text{name}[n]))$$

This is a key architectural contribution: entity states are semantically meaningful from epoch 1, not random.

**Four message streams** are computed per scene per entity:

| Stream | Mechanism | Purpose |
|--------|-----------|---------|
| Scene | Entity-aware bilinear attention on scene hyperedge $e_s$ | What does this entity extract from this scene? |
| Arc | Temporal attention over past hyperedges, masked to shared co-occurrence, with linear decay bias | How does this entity's arc condition the current scene? |
| Social | Co-occurrence-weighted mean of co-entity states | Who else is here, and what do they look like? |
| Relational | Co-occurrence × cosine alignment of running entity biographies | Which co-entities have similar trajectories? |

Streams are fused via scene-conditioned adaptive gating. A learned 4-way softmax gate (MLP: $d \to d/4 \to 4$) outputs per-stream weights, with a floor of 0.15 and ceiling of 0.55 applied post-softmax to prevent winner-take-all collapse:

$$\text{stream\_attn} = 0.15 + 0.40 \times \text{softmax}(\text{gate}(e_s))$$

This formula sums exactly to 1.0 while keeping each stream in the range [0.15, 0.55].

**Mamba Temporal Dynamics**: After collecting per-entity per-scene message vectors into trajectories $[B \times N, S, D]$, the **EntityMambaBlock** (2 layers, d_state = 32, d_inner = 2,048) processes each entity's full scene-sequence as a temporal trajectory. Mamba's input-dependent gating via the $\Delta t$ parameter (state-change magnitude) naturally learns *when* an entity's state changes significantly — corresponding to narrative turning points — vs. when it persists. The $\Delta t$ values are logged as W&B heatmaps for interpretability.

**Emotion modulation**: Per-entity per-scene sentiment polarity (from CardiffNLP RoBERTa) biases $\Delta t$, so emotionally intense scenes produce larger state changes.

**Scene event typing**: Each scene hyperedge $e_s$ is augmented with an event type embedding (5 types: CONFLICT, ALLIANCE, DECEPTION, DIALOGUE, NEUTRAL) classified from the dominant SVO verb in extracted triplets.

**Two-phase HGNN**:

- *Phase 1 (entity→hyperedge)*: Each scene hyperedge $e_s$ = role-weighted mean entity state + scene representation + event type embedding
- *Phase 2 (hyperedge→entity)*: After Mamba, entities aggregate over all their scene hyperedges (weighted by role) for a final residual update

## 3.3 Fusion and Decoder

**GraphToTextFusion** injects hyperedge knowledge back into the text stream via a gated residual:

$$\text{fused} = H_{\text{text}} + \sigma(\text{gate\_proj}(H_{\text{text}})) \cdot \text{CrossAttn}(H_{\text{text}},\, H_{\text{edges}})$$

The gate bias is initialized to $-3$ ($\sigma \approx 0.047$), giving 4.7% graph contribution at the start. This is conservative by design — the LED decoder needs time to stabilize on text-only signal before graph noise can destabilize it.

**EntitySceneCrossAttention** provides bidirectional cross-attention between scene representations and entity node states, injecting the final entity biographical context back into scene slots.

**Aligned memory** $[\text{fused\_scenes};\, \text{entity\_nodes}]$ (up to 164 positions: 64 scenes + 100 entities) is passed to the **LED decoder** as cross-attention keys/values. The decoder (100% trainable) generates summaries auto-regressively with beam search (beam = 4, no-repeat 4-gram, max 256 tokens).

**Total: 516M parameters** (348.5M trainable in Stage 2, 167.9M frozen LED encoder body).

---

# 4. Data Pipeline

## 4.1 Extraction

Raw MovieSum and MENSA datasets (HuggingFace) are converted to compressed JSONL feature files via `emnlp_extractor.py`. Each line is one scene, containing:

- `clean_text` — raw scene text for LED re-tokenization
- `summary_text` — full gold summary (movie-level, stored with scene 0)
- `coref_entities` — mention→canonical map from fastcoref (movie-level resolution)
- `graph_triplets` — SVO strings extracted by dependency parsing
- `ner_entities`, `characters` — NER output and screenplay ALL-CAPS speaker tags
- `character_emotions` — per-character polarity from CardiffNLP RoBERTa
- `action_mask`, `dialogue_mask`, `entity_mask`, `header_mask` — 4-way modality masks

fastcoref runs **movie-level** coreference (all scenes concatenated), so pronoun chains resolve globally — critical because "he" in scene 40 must link to the character introduced in scene 3.

## 4.2 Training Split

1,500 movies for training, 298 for evaluation (from MovieSum's ~1,800 unique movies). Scene-level files are byte-offset indexed for O(1) random access across 194,302 scenes without loading everything into memory. Movies exceeding 64 scenes are stride-sampled to fit the `MAX_SCENES = 64` budget.

---

# 5. Training Protocol

Training uses a two-stage curriculum:

**Stage 1 (3 epochs)** — LED encoder frozen. Only the hypergraph tower, fusion layers, and LED decoder are trained. The decoder learns to generate coherent summaries from text representations; the hypergraph tower learns to produce meaningful entity states.

- Optimizer: AdamW
- LR: 1e-4 (new layers), 2e-5 (LED decoder, scene_pool_proj, global attention layers)
- Gradient accumulation: 16 steps (effective batch = 16 movies)
- Loss: cross-entropy with label smoothing 0.05 + entity relational consistency loss (weight 1.5)

**Stage 2 (20 epochs)** — LoRA (r = 16, $\alpha$ = 32) applied to entity Mamba's `in_proj`, `x_proj`, `out_proj`, `dt_proj`. LED encoder global attention layers unfrozen (LR = 2e-5). Cosine warmup schedule with 5% warmup steps.

A **contrastive coherence loss** (NT-Xent on scene pairs where positive pairs share Jaccard entity similarity > 0.25) was held at $\alpha = 0$ throughout — ROUGE-1 had not yet crossed 0.05 when Stage 2 began, and enabling it early would have destabilized the decoder.

---

# 6. Training Challenges and Engineering Decisions

## 6.1 Loss Plateau at 7.5

After Stage 1, training loss plateaued at ~7.5 with no further decrease. Root cause: `gate_proj.bias` was initialized to $-4$ ($\sigma = 0.018 = 1.8\%$ graph weight), making the hypergraph gradient path 55× weaker than the text path. Fix: bias re-initialized to $-3$ ($\sigma = 0.047$) via checkpoint surgery (`open_gate.py`), plus optimizer state cleared to avoid stale Adam moments.

## 6.2 Arc Stream Winner-Take-All

After the gate fix, W&B stream weight logs showed arc stream weights saturating at 0.85. The previous fusion formula (`0.05 + 0.80 × softmax`) allowed any single stream to hit 0.85, causing winner-take-all collapse. Fixed by tightening to `0.15 + 0.40 × softmax` (floor = 0.15, ceiling = 0.55), forcing all four streams to remain active and competitive.

## 6.3 Dead Mamba Gradients (Critical Issue)

The most severe problem: all entity Mamba parameters showed near-zero gradients for all 23 training epochs. Two blocked gradient paths were identified:

**Path 1 — Through aligned_memory**: `entity_mem_scale` (a learnable scalar controlling entity contribution to aligned memory) was initialized to 0. The decoder, having seen zero entity signal for 7+ early epochs, learned to ignore entity memory positions. With decoder attention weights on entity slots ≈ 0, $\partial \mathcal{L} / \partial \text{entity\_mem\_scale} \approx 0$, and therefore $\partial \mathcal{L} / \partial \text{Mamba} \approx 0$.

**Path 2 — Through entity_scene_attn**: `scene_ls` (a learnable scale vector controlling entity-to-scene feedback) was initialized to `zeros(1024)`, making $H_{\text{scenes}} \mathrel{+}= \text{scene\_ls} \times s^* \approx H_{\text{scenes}}$, zeroing the gradient through this path.

The fix (`fix_mamba_grad.py`): patch `entity_mem_scale: 0.1 → 0.6` (forcing decoder attention to entity positions at 54% weight, breaking the chicken-and-egg), and `scene_ls: zeros(1024) → full(1024, 0.05)`. This script was committed but failed to reach the training server before resumption, so the model ran all 23 epochs without Mamba gradients. **The evaluated model is effectively a LED + static hypergraph baseline.**

## 6.4 Scheduler State Corruption

Because `fix_mamba_grad.py` never ran (it would have cleared both optimizer and scheduler state), the stale scheduler state was restored from the checkpoint, producing an erratic LR pattern: LR fell correctly from 1.5e-5 → ~0 at epoch 15, then climbed back to 8.19e-5 by epoch 23 — 4× higher than intended for Stage 2. Epochs 18–23 trained at dangerously high LR. The epoch 21 checkpoint (best metrics) is safer than epoch 23.

## 6.5 ROUGE Evaluation Methodology Errors

Initial ROUGE evaluation was computed on only 5 movies (0.7% of eval set) with references truncated to 256 tokens by re-decoding from `target_ids`. Fixed to: 50 sampled movies, full gold `summary_text` from the raw data field (un-truncated). This significantly changed reported numbers.

## 6.6 scene_pool_proj Gradient Explosion

`scene_pool_proj` (initialized to identity for boundary pooling) was grouped with new-layer parameters at LR = 1e-4. This caused gradient spikes of ~10,000 and optimizer steps of ~1.0, destroying the identity initialization in the first batch. Fixed by assigning it to the LR_DECODER = 2e-5 group alongside other pretrained components.

---

# 7. Results

Best metrics across epochs 12–23 (epoch 21 checkpoint):

| Metric | Value |
|--------|-------|
| ROUGE-1 | 0.187 |
| ROUGE-2 | 0.005 |
| ROUGE-L | 0.073 |
| METEOR | 0.070 |
| Entity F1 | 0.098 |
| Train Loss | 7.478 |
| Eval Loss | 7.609 |
| Avg entities / scene | 9.0 |
| Avg scenes / entity | 3.3 |
| Hypergraph coverage | 14.8% |

ROUGE-2 at 0.005 indicates that while the model produces some correct unigrams, it rarely produces correct bigrams — the output still lacks syntactic fluency and coherent multi-word expressions. This is consistent with the model running without Mamba: the decoder generates semantically related but structurally inconsistent text. Entity F1 (0.098) shows a positive upward trend (0.072 → 0.098 across epochs), suggesting the model is progressively learning to include correct character names even without Mamba grounding.

The LR anomaly (climbing to 8.19e-5 in late epochs) partially explains the erratic ROUGE numbers across epochs: R1 dips from 0.187 at epoch 21 to 0.145 at epoch 22, then recovers to 0.166 at epoch 23, consistent with high-LR instability rather than monotone learning.

---

# 8. Areas for Improvement

## 8.1 Apply Mamba Fix and Retrain Stage 2

The single highest-impact action is to restart Stage 2 from the epoch 11 checkpoint with `fix_mamba_grad.py` properly applied. The 17+ hours of Stage 1 training are preserved; only Stage 2 (~50 hours on one A100) needs to be rerun. Expected outcome: non-zero Mamba gradients from epoch 12 batch 1, a loss spike to ~8.5 for 1–2 epochs as entity noise is absorbed, then continued decrease and improved ROUGE. The $\Delta t$ heatmaps should then show interpretable narrative turning points.

## 8.2 Hypergraph Coverage (14.8%)

Only 14.8% of entity-scene incidence slots are non-zero, meaning the graph is very sparse. Improvements:

- Lower the minimum scene frequency threshold for entity inclusion
- Add pronoun resolution at scene level to recover missed mentions
- Include implicit entity references from action lines (dependency-parsed subjects)

## 8.3 Re-enable Contrastive Coherence Loss

The NT-Xent narrative coherence loss was never enabled. Re-enabling it at ROUGE-1 > 0.10 (now crossed) would push scene representations toward a coherence structure where topically related scenes cluster together, improving summary fluency and entity consistency across sentences.

## 8.4 Full-Corpus ROUGE Evaluation

No full-corpus ROUGE evaluation currently exists (only the 50-sample approximation used during training). A dedicated inference pass on all ~700 eval movies with multi-sentence beam search (no 256-token truncation) would give definitive numbers for the paper submission.

## 8.5 LoRA Ablation

LoRA was applied with r = 16, $\alpha$ = 32 to entity Mamba's four projection matrices. Since Mamba never received gradients in Stage 2, all LoRA weights are essentially random at end of training. After the fix, an ablation comparing r = 8 vs r = 32 and including `conv1d` in LoRA targets would identify the optimal parameter budget.

## 8.6 Zero-Shot MENSA Transfer

No MENSA evaluation was run. MovieSum summaries are encyclopedic plot descriptions; MENSA summaries are shorter and more thematic. Zero-shot transfer would reveal whether entity tracking generalizes across narrative styles or overfits MovieSum's summary register.

## 8.7 Emotion Modulation

`emotion_scale` was initialized to 0 and is likely still near-zero given the dead Mamba gradient problem. Once Mamba is unblocked, verify whether emotion_scale learns a non-trivial value — if not, consider initializing it to a small positive constant (e.g., 0.1) to give emotionally intense scenes an active $\Delta t$ boost from the start.

---

# 9. Ablation Plan for the Paper

The architecture supports a clean ablation hierarchy across four levels of graph complexity:

| Run | Flag | Research Question |
|-----|------|-------------------|
| LED-only | `--no_hypergraph` | Is the hypergraph necessary at all? |
| LED + static graph | `--static_hypergraph` | Do dynamic entity updates matter? |
| LED + GRU (no Mamba) | `--no_mamba_entity` | Does Mamba outperform GRU for temporal dynamics? |
| LED + Mamba (full) | *(default)* | Full model |
| No entity names | `--no_entity_names` | Does name-grounded init help? |
| No edge dropout | `--edge_dropout 0.0` | Does structural regularization help? |
| Global stream weights | `--no_adaptive_streams` | Does per-scene gating help vs. fixed weights? |

The key comparison for the paper is (3) vs. (4): LED + GRU vs. LED + Mamba. If Mamba's input-dependent gating on entity trajectories produces better ROUGE and entity F1, that is the core empirical contribution. The $\Delta t$ heatmap visualizations provide qualitative evidence: large $\Delta t$ values should appear at scenes corresponding to plot climaxes and character revelations, demonstrating that the SSM has learned to parse narrative structure.

---

# 10. Conclusion

This project presents a principled approach to long-form narrative summarization that goes beyond treating a screenplay as a flat token stream. The dual-tower architecture — LED for global 16K-token text context and a 4-stream dynamic hypergraph for relational character tracking — addresses known weaknesses of prior summarization systems on long narrative documents. The Mamba SSM component provides an interpretable and computationally efficient mechanism for modeling entity state trajectories, with $\Delta t$ values directly encoding narrative turning point magnitude.

The primary limitation of the current results is that the Mamba component was gradient-blocked for all 23 training epochs due to a two-path dead-gradient problem in the aligned memory and entity-scene cross-attention modules, meaning the model was run as a LED + static hypergraph system throughout. Resolving this one issue — already diagnosed and patched — and rerunning Stage 2 is expected to substantially improve performance, particularly on Entity F1 and ROUGE-L.

The codebase is fully ablation-ready. The next milestone is a clean Stage 2 training run with working Mamba gradients, followed by full-corpus ROUGE evaluation on MovieSum and zero-shot MENSA transfer, and the GRU vs. Mamba ablation that constitutes the paper's central empirical claim.

---

\vspace{1em}
\noindent\rule{\textwidth}{0.4pt}
\small
\textbf{System:} LED-large-16384 (400M) + DynamicHypergraphTower (116M) \quad
\textbf{Dataset:} MovieSum (1,500 train / 298 eval) \quad
\textbf{Hardware:} 1× A100 \quad
\textbf{Training:} 23 epochs, ~100 hours \quad
\textbf{Best checkpoint:} Epoch 21

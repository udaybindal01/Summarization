"""
sum.py  —  GraM-Former v2 Model Architecture
=============================================
Key upgrades over v1
--------------------
  1. HeterogeneousGraphTransformer (HGT)
       Three typed movie-level graphs fused with type-aware message passing:
         - Causal Event Graph  (directed SVO causal edges)
         - Character State Graph (edge weight = emotion-state change magnitude)
         - Discourse Structure Graph (act/beat rhetorical relations)

  2. RAFT Consensus Attention v2
       Cross-modal attention: each modality attends over all 4, then a gated
       fusion collapses them.  Disentangled projections remain orthogonal via
       the Riemannian penalty.

  3. DynamicGraphModulator v2
       IDF-weighted edge pruning: rare entity co-occurrences get higher weight,
       common stop-entity matches are down-weighted.

  4. HierarchicalPointerHead v2
       Dual-level copy: scene-level attention + learned entity salience
       (dot-product decoder hidden state × embedded entity), not uniform.

  5. NarrativeCoherenceLoss  (new)
       Contrastive term that pulls together summary hidden states for
       causally-linked scene pairs and pushes apart unlinked ones.
       Directly targets the factuality weakness identified in EMNLP-2025.

  6. RelationalEventConsistencyLoss (improved)
       Label smoothing, graduated entity penalty (annealed via alpha_entity),
       NLL on pre-computed log-probabilities (not cross-entropy double-log).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import random
import math
import wandb
from transformers import BartForConditionalGeneration, RobertaModel
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Backbone config — swap to "facebook/bart-base" for ablation
# =============================================================================
# Load from local path — avoids network hang on restricted servers.
# Falls back to HuggingFace hub if local path not found.
import os as _os
_BART_LARGE_LOCAL = "/tmp/uday/bart-large"
_BART_BASE_LOCAL  = "/tmp/uday/bart-base"

BART_MODEL = (_BART_LARGE_LOCAL if _os.path.isdir(_BART_LARGE_LOCAL)
              else "facebook/bart-large")
# BART-large: 1024 hidden dim, 16 attention heads, 400M params
# BART-base:   768 hidden dim,  8 attention heads, 140M params
# d_model in GraMFormerV2.__init__ must match: 1024 for large, 768 for base


# =============================================================================
# 0. Utilities
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        x_f32 = x.float()
        out = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


# =============================================================================
# 1. Scene Encoder — Graph-Modulated Mamba (unchanged from v1, stable)
# =============================================================================

class GraMambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * 2

        self.in_proj             = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d              = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv - 1)
        self.x_proj              = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj             = nn.Linear(1, self.d_inner)
        self.graph_routing_matrix = nn.Linear(d_state, d_state)
        self.out_proj            = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x, adjacency_matrix):
        residual = x
        x = self.norm(x)
        batch, seq_len, _ = x.shape

        xz    = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_proj.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        x_act  = F.silu(x_conv)

        x_ssm = self.x_proj(x_act)
        delta, B, C = torch.split(x_ssm, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta).float())
        B, C  = B.float(), C.float()
        x_act_f32 = x_act.float()

        h = torch.zeros(batch, self.d_inner, self.d_state,
                        device=x.device, dtype=torch.float32)
        history_list  = []
        scan_outputs  = []
        adj_float = adjacency_matrix.float()  # cast once — avoids per-step dtype conversion

        for t in range(seq_len):
            delta_t = delta[:, t].unsqueeze(-1)
            B_t     = B[:, t].unsqueeze(1)
            x_t     = x_act_f32[:, t].unsqueeze(-1)
            h       = h * torch.exp(-delta_t) + (B_t * x_t)

            if t > 0:
                causal_weights = adj_float[:, t, :t]
                if causal_weights.any():
                    w     = causal_weights.view(batch, t, 1, 1)
                    hist  = torch.stack(history_list, dim=1)
                    deg   = w.sum(dim=1).clamp(min=1.0)
                    g_ctx = (w * hist).sum(dim=1) / deg
                    h     = h + 0.1 * F.silu(self.graph_routing_matrix(g_ctx))

            history_list.append(h)
            C_t = C[:, t].unsqueeze(1)
            scan_outputs.append((h * C_t).sum(dim=-1))

        y = torch.stack(scan_outputs, dim=1)
        return self.out_proj(y * F.silu(z)) + residual


class GraphModulatedMambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            GraMambaLayer(d_model, d_state, d_conv)
            for _ in range(num_layers)
        ])

    def forward(self, x, adjacency_matrix):
        for layer in self.layers:
            x = layer(x, adjacency_matrix)
        return x


# =============================================================================
# 2. RAFT Consensus Attention v2 — Cross-Modal Attention
# =============================================================================

class RaftConsensusAttentionV2(nn.Module):
    """
    Each modality attends over ALL four modalities before fusion.
    This lets tense dialogue reinforce violent action in the same scene.
    """
    def __init__(self, d_model=768, num_heads=4):
        super().__init__()
        self.num_heads  = num_heads
        # Independent projection per modality (kept for Riemannian penalty)
        self.action_proj = nn.Linear(d_model, d_model)
        self.dial_proj   = nn.Linear(d_model, d_model)
        self.ent_proj    = nn.Linear(d_model, d_model)
        self.head_proj   = nn.Linear(d_model, d_model)

        # Cross-modal attention: query = one modality, keys/values = all 4
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, dropout=0.1
        )
        # Gate fusing the cross-attended modality stack
        self.consensus_gate = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, features, action_mask, dial_mask, ent_mask, head_mask):
        masks = [action_mask, dial_mask, ent_mask, head_mask]
        projs = [self.action_proj, self.dial_proj,
                 self.ent_proj,   self.head_proj]

        # Project each modality and zero out irrelevant tokens
        modalities = [
            projs[i](features) * masks[i].unsqueeze(-1).float()
            for i in range(4)
        ]

        B, L, D = features.shape
        # Stack modalities along a "modality" dim: [B*L, 4, D]
        modal_stack = torch.stack(modalities, dim=2)   # [B, L, 4, D]
        modal_flat  = modal_stack.view(B * L, 4, D)

        # Each of the 4 modalities attends over all 4
        attended, _ = self.cross_attn(modal_flat, modal_flat, modal_flat)
        attended    = attended.view(B, L, 4, D)

        # Fuse via gating
        gate_in    = attended.reshape(B, L, 4 * D)
        consensus  = self.consensus_gate(gate_in)
        return self.norm(features + consensus)


# =============================================================================
# 3. Heterogeneous Graph Transformer (HGT)
# =============================================================================

class HeterogeneousGraphTransformer(nn.Module):
    """
    Fuses two typed movie-level graphs:
      0 = Causal Event Graph  (directed SVO causal chains)
      1 = Character State Graph (edge weight = emotion-state change magnitude)

    Type-aware Q/K/V projections give each edge type its own attention space.
    """
    def __init__(self, d_model=768, num_edge_types=2, num_heads=4):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.num_heads      = num_heads
        self.d_head         = d_model // num_heads

        self.q_proj  = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_edge_types)])
        self.k_proj  = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_edge_types)])
        self.v_proj  = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_edge_types)])
        self.out_proj = nn.Linear(d_model * num_edge_types, d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, scene_reps, adj_list):
        """
        scene_reps : [B, S, D]
        adj_list   : list of 3 tensors, each [B, S, S]  (may be weighted floats)
        """
        B, S, D = scene_reps.shape
        outputs = []

        for i, adj in enumerate(adj_list):
            q = self.q_proj[i](scene_reps)                        # [B, S, D]
            k = self.k_proj[i](scene_reps)
            v = self.v_proj[i](scene_reps)

            # Scaled dot-product with adjacency masking
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)  # [B, S, S]
            # Soft mask: multiply by adj so zero-edge pairs are strongly suppressed
            # adj can be float-weighted (Character State Graph uses magnitudes)
            adj_f  = adj.float().to(scores.device)
            # Where adj == 0, push score to -1e4 (not -inf to avoid NaN in bfloat16)
            scores = scores * (adj_f + 1e-6) + (adj_f == 0).float() * -1e4
            weights = F.softmax(scores, dim=-1)
            outputs.append(torch.matmul(weights, v))

        fused = self.out_proj(torch.cat(outputs, dim=-1))
        return self.norm(scene_reps + fused)


# =============================================================================
# 4. Dynamic Graph Modulator v2
# =============================================================================

class DynamicGraphModulatorV2(nn.Module):
    """
    IDF-weighted edge gating: entity pairs that co-occur rarely across the
    movie get a HIGHER gate value (they signal unusual, plot-relevant links).

    idf_weights: [B, S, S] tensor of pre-computed IDF scores stored in the
                  Character State Graph adjacency.  If absent, falls back to
                  the v1 sigmoid gating.
    """
    def __init__(self, d_model=768, reduction_factor=8):
        super().__init__()
        d_gate = d_model // reduction_factor
        self.query_proj = nn.Linear(d_model, d_gate)
        self.key_proj   = nn.Linear(d_model, d_gate)
        # Learnable IDF temperature
        self.idf_temp   = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, hidden_states, static_movie_adj, idf_weights=None):
        """
        hidden_states  : [B, S, L, D]
        static_movie_adj: [B, S, S]
        idf_weights    : [B, S, S] optional
        """
        scene_reps = hidden_states.mean(dim=2)   # [B, S, D]
        q = self.query_proj(scene_reps).float()
        k = self.key_proj(scene_reps).float()
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        dynamic_gate = torch.sigmoid(attn_scores + 2.0)

        if idf_weights is not None:
            idf = idf_weights.float().to(dynamic_gate.device)
            # Rare co-occurrences boost the gate
            idf_boost = torch.sigmoid(idf * self.idf_temp.float())
            dynamic_gate = dynamic_gate * idf_boost

        dynamic_gate = dynamic_gate.to(hidden_states.dtype)
        return static_movie_adj.to(hidden_states.dtype) * dynamic_gate


# =============================================================================
# 5. Graph Message Passing (used inside GlobalAggregator)
# =============================================================================

class GraphMessagePassing(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.message_proj = nn.Linear(d_model, d_model)
        self.norm         = nn.LayerNorm(d_model)

    def forward(self, scene_reps, dynamic_adj):
        degree   = dynamic_adj.sum(dim=-1, keepdim=True).clamp(min=1e-4)
        norm_adj = dynamic_adj / degree
        messages = torch.matmul(norm_adj, scene_reps)
        return self.norm(scene_reps + self.message_proj(messages))


# =============================================================================
# 6. Global Aggregator (bi-directional temporal Mamba + recovery gate)
# =============================================================================

class GlobalAggregator(nn.Module):
    def __init__(self, d_model=768, d_state=64):
        super().__init__()
        self.salience_pooler = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1),
        )
        self.hgt          = HeterogeneousGraphTransformer(d_model, num_edge_types=2)
        self.msg_pass     = GraphMessagePassing(d_model)
        self.temporal_mamba = GraphModulatedMambaBlock(
            d_model, d_state=d_state, d_conv=4, num_layers=1
        )
        self.recovery_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.norm = RMSNorm(d_model)

    def forward(self, scene_embeddings_batch, causal_adj, char_state_adj):
        """
        scene_embeddings_batch : [B, S, L, D]
        *_adj                  : [B, S, S]  (two typed graphs)
        """
        B, S, L, D = scene_embeddings_batch.shape

        # Salience pool each scene to a single vector
        flat_scenes = scene_embeddings_batch.view(B * S, L, D)
        weights     = self.salience_pooler(flat_scenes)           # [B*S, L, 1]
        pooled      = torch.bmm(weights.transpose(1, 2), flat_scenes).squeeze(1)  # [B*S, D]
        movie_seq   = pooled.view(B, S, D)                        # [B, S, D]

        # HGT over two typed graphs
        hgt_out = self.hgt(movie_seq, [causal_adj, char_state_adj])
        normed  = self.norm(hgt_out)

        # Bi-directional temporal Mamba
        # Use the averaged graph for the Mamba adjacency
        avg_adj      = (causal_adj + char_state_adj) / 2.0
        fwd          = self.temporal_mamba(normed, avg_adj)
        bwd_seq      = torch.flip(normed, dims=[1])
        bwd_adj      = torch.flip(avg_adj, dims=[1, 2])
        bwd          = torch.flip(self.temporal_mamba(bwd_seq, bwd_adj), dims=[1])
        global_narr  = fwd + bwd                                  # [B, S, D]

        # Broadcast global narrative back to token level and apply recovery gate
        global_bc  = global_narr.unsqueeze(2).expand(-1, -1, L, -1)
        combined   = torch.cat([scene_embeddings_batch, global_bc], dim=-1)
        gate       = self.recovery_gate(combined)
        enriched   = gate * scene_embeddings_batch + (1 - gate) * global_bc
        return enriched   # [B, S, L, D]


# =============================================================================
# 7. Hierarchical Pointer Head v2
# =============================================================================

class HierarchicalPointerHeadV2(nn.Module):
    """
    Dual-level copy mechanism:
      - Scene-level:  decoder attends over scene memory  →  scene_attn
      - Entity-level: salience = dot(decoder_state, entity_embedding)
                      not uniform 1/N — rare/salient entities score higher

    p_gen gate uses a hard sigmoid (×10 sharpness) to produce near-binary
    copy/generate decisions — prevents averaging-out of rare tokens.
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj   = nn.Linear(d_model, d_model)
        self.p_gen_linear = nn.Linear(d_model * 2, 1)
        nn.init.constant_(self.p_gen_linear.bias, 3.0)

    def forward(self, decoder_states, scene_memory,
                triplets, tokenizer, embedding_weight, device):
        """
        decoder_states  : [B, T, D]
        scene_memory    : [B, S, D]
        embedding_weight: [V, D]  (shared BART embedding)
        """
        B, T, D = decoder_states.shape
        _, S, _  = scene_memory.shape

        # ── Scene-level attention ──────────────────────────────────────────
        q      = self.query_proj(decoder_states)                # [B, T, D]
        k      = self.key_proj(scene_memory)                    # [B, S, D]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)
        scene_attn = F.softmax(scores, dim=-1)                  # [B, T, S]

        # ── Context vector + hard p_gen gate ──────────────────────────────
        context    = torch.matmul(scene_attn, scene_memory)     # [B, T, D]
        gate_in    = torch.cat([decoder_states, context], dim=-1)
        p_gen      = torch.sigmoid((self.p_gen_linear(gate_in) - 0.5) * 10.0)

        # ── Entity-level salience copy distribution ────────────────────────
        scene_to_token = torch.zeros(B, S, self.vocab_size, device=device)

        if triplets and tokenizer is not None:
            # Use last decoder step hidden state for entity salience
            dec_last = decoder_states[:, -1, :]                 # [B, D]

            for b in range(B):
                num_scenes = min(S, len(triplets[b]) if isinstance(triplets[b], list) else 1)
                for s in range(num_scenes):
                    if isinstance(triplets[b], list):
                        scene_trips = triplets[b][s] if s < len(triplets[b]) else []
                    else:
                        scene_trips = []
                    if not scene_trips:
                        continue

                    entities = set()
                    for trip in scene_trips:
                        parts = trip.split("_")
                        if len(parts) >= 1:
                            entities.add(parts[0].replace("NOT ", "").strip())
                        if len(parts) >= 3:
                            entities.add(parts[2].strip())
                    entities.discard("")

                    if not entities:
                        continue

                    encoded   = tokenizer(list(entities),
                                          add_special_tokens=False)["input_ids"]
                    token_ids = list({tid for sublist in encoded for tid in sublist
                                      if tid < self.vocab_size})
                    if not token_ids:
                        continue

                    # Learned entity salience: dot(dec_hidden, entity_emb)
                    ent_tensor  = torch.tensor(token_ids, device=device)
                    ent_embs    = F.embedding(ent_tensor, embedding_weight)  # [E, D]
                    salience    = torch.matmul(
                        dec_last[b].unsqueeze(0).float(),
                        ent_embs.float().T
                    ).squeeze(0)                                              # [E]
                    salience    = F.softmax(salience, dim=-1)
                    scene_to_token[b, s, token_ids] = salience.to(scene_to_token.dtype)

        # ── Map scene attention to vocabulary ──────────────────────────────
        pointer_probs = torch.matmul(scene_attn, scene_to_token)  # [B, T, V]
        return p_gen, pointer_probs


# =============================================================================
# 8. Narrative Coherence Loss  (NEW — targets EMNLP-2025 factuality gap)
# =============================================================================

class NarrativeCoherenceLoss(nn.Module):
    """
    Scene-pair contrastive loss within a single movie.

    Operates on encoder scene representations rather than decoder hidden states,
    which makes the geometry match what the causal_adj describes.  Works
    correctly at batch_size=1 — positives are scene pairs with a causal edge,
    negatives are all other scenes in the same movie.

    Temperature 0.1 (vs the original 0.07) avoids over-sharpening with a
    small number of scenes per movie.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, scene_hidden, causal_adj):
        """
        scene_hidden : [B, S, L, D]  — encoder hidden states per scene
        causal_adj   : [B, S, S]     — gated causal adjacency (float, [0,1])
        """
        B, S, L, D = scene_hidden.shape
        # Pool each scene: [B, S, D]
        scene_reps = scene_hidden.mean(dim=2)
        scene_reps = F.normalize(scene_reps.float(), p=2, dim=-1)

        total_loss = scene_reps.new_tensor(0.0)
        n_valid = 0

        for b in range(B):
            adj  = causal_adj[b].float()   # [S, S]
            reps = scene_reps[b]            # [S, D]

            # Symmetric positive mask: any directed edge in either direction
            pos_mask = ((adj + adj.T) > 0).float()
            pos_mask.fill_diagonal_(0)

            if pos_mask.sum() == 0:
                continue

            # Scaled dot-product similarity between all scene pairs
            sim = torch.matmul(reps, reps.T) / self.temperature  # [S, S]
            sim.fill_diagonal_(-1e4)  # exclude self-similarity

            # NT-Xent: for each anchor, positives = causally linked scenes
            log_probs  = F.log_softmax(sim, dim=-1)       # [S, S]
            pos_norm   = pos_mask / pos_mask.sum(dim=1, keepdim=True).clamp(min=1)
            loss_per_scene = -(pos_norm * log_probs).sum(dim=-1)

            # Only average over scenes that have at least one positive
            has_pos = pos_mask.sum(dim=1) > 0
            if not has_pos.any():
                continue

            total_loss = total_loss + loss_per_scene[has_pos].mean()
            n_valid += 1

        return total_loss / max(n_valid, 1)


# =============================================================================
# 9. Relational Event Consistency Loss (v2 — improved)
# =============================================================================

class RelationalEventConsistencyLoss(nn.Module):
    def __init__(self, alpha=0.1, tokenizer=None,
                 temperature=0.1, entity_penalty=3.0,
                 label_smoothing=0.1, coherence_weight=0.05):
        super().__init__()
        self.alpha             = alpha
        self.tokenizer         = tokenizer
        self.temperature       = temperature
        self.entity_penalty    = entity_penalty    # start low, anneal in train.py
        self.label_smoothing   = label_smoothing
        self.coherence_weight  = coherence_weight
        self.coherence_loss_fn = NarrativeCoherenceLoss(temperature=0.07)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _smooth_nll(self, log_probs, targets, weight_mask):
        """NLL with label smoothing."""
        V = log_probs.size(-1)
        # Smooth target distribution
        smooth_target = torch.zeros_like(log_probs).fill_(self.label_smoothing / V)
        smooth_target.scatter_(1, targets.unsqueeze(1).clamp(min=0), 1.0 - self.label_smoothing)
        nll = -(smooth_target * log_probs).sum(dim=-1)
        valid = (targets != 1).float()
        weighted = nll * weight_mask * valid
        return weighted.sum() / (weight_mask * valid).sum().clamp(min=1.0)

    def _get_triplet_embeddings(self, triplet_texts, head_weight, device, max_triplets=5):
        if not triplet_texts or not triplet_texts[0]:
            return None
        flat_texts = []
        for scene_trips in triplet_texts:
            sampled = (random.sample(scene_trips, max_triplets)
                       if len(scene_trips) > max_triplets else scene_trips)
            flat_texts.append(" ".join(sampled).replace("_", " "))
        encoded = self.tokenizer(flat_texts, padding=True, truncation=True,
                                 max_length=32, return_tensors="pt").to(device)
        token_embs   = F.embedding(encoded["input_ids"], head_weight)
        mask_exp     = encoded["attention_mask"].unsqueeze(-1).expand(token_embs.size()).float()
        sum_embs     = (token_embs * mask_exp).sum(1)
        sum_mask     = mask_exp.sum(1).clamp(min=1e-4)
        return sum_embs / sum_mask

    def _generate_negative_triplets(self, triplets):
        negatives = []
        for batch_trips in triplets:
            batch_negs = []
            for trip in batch_trips:
                parts = trip.split("_")
                if len(parts) == 3:
                    batch_negs.append(f"{parts[2]}_{parts[1]}_{parts[0]}")
                else:
                    batch_negs.append(trip)
            negatives.append(batch_negs)
        return negatives

    def get_riemannian_orthogonality_loss(self, model):
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        decoder = m.decoder
        weights = torch.stack([
            decoder.action_proj.weight.view(-1),
            decoder.dial_proj.weight.view(-1),
            decoder.ent_proj.weight.view(-1),
            decoder.head_proj.weight.view(-1),
        ])
        wn   = F.normalize(weights, p=2, dim=1, eps=1e-4)
        gram = torch.matmul(wn, wn.t())
        return F.mse_loss(gram, torch.eye(4, device=weights.device))

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, log_probs, targets, triplets,
                hidden_states=None, head_weight=None,
                causal_adj=None):
        device = log_probs.device

        # ── Entity weighting ──────────────────────────────────────────────
        weight_mask = torch.ones_like(targets, dtype=torch.float)
        if triplets and triplets[0] and self.tokenizer is not None:
            entity_token_ids = set()
            flat_ents = []
            for scene_trips in triplets:
                for trip in scene_trips:
                    parts = trip.split("_")
                    if len(parts) >= 1:
                        flat_ents.append(parts[0].replace("NOT ", "").strip())
                    if len(parts) >= 3:
                        flat_ents.append(parts[2].strip())
            if flat_ents:
                enc = self.tokenizer(flat_ents, add_special_tokens=False)["input_ids"]
                for sub in enc:
                    entity_token_ids.update(sub)
                if entity_token_ids:
                    ent_t = torch.tensor(list(entity_token_ids), device=device)
                    weight_mask[torch.isin(targets, ent_t)] = self.entity_penalty

        lm_loss = self._smooth_nll(log_probs, targets, weight_mask)

        # ── Contrastive triplet loss ───────────────────────────────────────
        if (not triplets or not triplets[0]
                or hidden_states is None or head_weight is None):
            return lm_loss

        B, S, L, D = hidden_states.shape
        h_flat   = hidden_states.view(B * S, L, D).mean(dim=1)
        valid_idx = [i for i, t in enumerate(triplets) if len(t) > 0]
        if not valid_idx:
            return lm_loss

        valid_trips = [triplets[i] for i in valid_idx]
        h_valid     = h_flat[valid_idx]
        neg_trips   = self._generate_negative_triplets(valid_trips)

        pos_embs = self._get_triplet_embeddings(valid_trips, head_weight, device)
        neg_embs = self._get_triplet_embeddings(neg_trips,   head_weight, device)
        if pos_embs is None or neg_embs is None:
            return lm_loss

        n = min(h_valid.size(0), pos_embs.size(0))
        h_n  = F.normalize(h_valid[:n].float(), p=2, dim=1, eps=1e-4)
        p_n  = F.normalize(pos_embs[:n].float(), p=2, dim=1, eps=1e-4)
        ng_n = F.normalize(neg_embs[:n].float(), p=2, dim=1, eps=1e-4)

        sim_pos  = (h_n * p_n).sum(-1)  / self.temperature
        sim_neg  = (h_n * ng_n).sum(-1) / self.temperature
        logits_c = torch.stack([sim_pos, sim_neg], dim=1).float()
        labels_c = torch.zeros(n, dtype=torch.long, device=device)
        contrastive_loss = F.cross_entropy(logits_c, labels_c)

        total = (1 - self.alpha) * lm_loss + self.alpha * contrastive_loss

        # ── Narrative coherence loss (scene-pair contrastive) ─────────────
        # Uses encoder scene representations, not decoder hidden states, so
        # the geometry aligns with what causal_adj encodes.
        if hidden_states is not None and causal_adj is not None:
            coh_loss = self.coherence_loss_fn(hidden_states, causal_adj)
            total    = total + self.coherence_weight * coh_loss

        return total


# =============================================================================
# 10. GraM-Former v2 (Final Assembly)
# =============================================================================

class GraMFormerV2(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=4, tokenizer=None):
        super().__init__()

        print(f"Loading BART backbone: {BART_MODEL}...")
        bart = BartForConditionalGeneration.from_pretrained(BART_MODEL)

        self.tokenizer = tokenizer
        self.d_model   = d_model

        # ── A1: Frozen RoBERTa scene encoder ─────────────────────────────
        # RoBERTa provides pre-trained contextual representations (768d).
        # It stays frozen throughout both training stages; LoRA is applied
        # only to the Mamba layers above it.
        print("Loading frozen RoBERTa scene encoder...")
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        for param in self.roberta.parameters():
            param.requires_grad = False

        # A3: Linear bridge 768 → d_model (identity when d_model == 768)
        self.scene_proj = (
            nn.Linear(768, d_model, bias=False)
            if d_model != 768
            else nn.Identity()
        )

        # ── Scene-level Mamba encoder (graph-modulated SSM) ───────────────
        self.encoder = GraphModulatedMambaBlock(
            d_model=d_model, d_state=64, d_conv=4, num_layers=num_layers
        )

        # ── RAFT v2 (cross-modal modality fusion) ─────────────────────────
        self.decoder = RaftConsensusAttentionV2(d_model=d_model, num_heads=4)

        # ── Movie-level graph modulation & aggregation ────────────────────
        self.graph_modulator  = DynamicGraphModulatorV2(d_model=d_model)
        self.global_aggregator = GlobalAggregator(d_model=d_model)

        # ── Latent bridge (geometry alignment) ───────────────────────────
        self.latent_bridge = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # ── BART decoder (iron vault) ─────────────────────────────────────
        self.bart_decoder = bart.model.decoder
        self.head         = bart.lm_head

        # ── Pointer head v2 ───────────────────────────────────────────────
        self.pointer_head = HierarchicalPointerHeadV2(d_model, vocab_size)

        # Freeze BART decoder (except cross-attention)
        for param in self.bart_decoder.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = False
        for name, param in self.bart_decoder.named_parameters():
            if "encoder_attn" in name:
                param.requires_grad = True

        self.memory_norm       = nn.LayerNorm(d_model)
        self.use_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.use_checkpointing = True

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, input_ids, adj, action_mask, dial_mask, ent_mask,
                head_mask, causal_adj, char_state_adj,
                target_ids=None, triplets=None, idf_weights=None):
        """
        input_ids       : [B, S, L]
        adj             : [B, S, L, L]   local token-level dependency graph
        *_mask          : [B, S, L]
        causal_adj      : [B, S, S]      directed SVO causal chains
        char_state_adj  : [B, S, S]      emotion-state-change weighted
        target_ids      : [B, S, L]  (training only)
        triplets        : list[list[str]]
        idf_weights     : [B, S, S] optional
        """
        B, S, L   = input_ids.shape
        d_model   = self.d_model
        pad_id    = 1   # shared pad_token_id for RoBERTa and BART
        chunk_sz  = 10
        is_frozen = not next(self.encoder.parameters()).requires_grad

        # ── Scene encoder: frozen RoBERTa → projection → Mamba ───────────
        # RoBERTa is always run without grad (frozen throughout).
        # Mamba encoder may be frozen (Stage 1) or LoRA-adapted (Stage 2).
        local_feats = []
        for i in range(0, S, chunk_sz):
            j     = min(i + chunk_sz, S)
            c_ids = input_ids[:, i:j].contiguous().view(-1, L)   # [B*chunk, L]
            c_adj = adj[:, i:j].contiguous().view(-1, L, L)       # [B*chunk, L, L]

            # A1: RoBERTa contextual encoding (frozen, always no_grad)
            c_attn = (c_ids != pad_id).long()
            with torch.no_grad():
                c_roberta = self.roberta(
                    c_ids, attention_mask=c_attn
                ).last_hidden_state                                # [B*chunk, L, 768]
            # A3: project 768 → d_model (identity if d_model == 768)
            c_embeds = self.scene_proj(c_roberta)                 # [B*chunk, L, d_model]

            if is_frozen:
                with torch.no_grad():
                    c_out = self.encoder(c_embeds, c_adj)
            elif self.use_checkpointing:
                c_out = checkpoint(
                    lambda x, a: self.encoder(x, a),
                    c_embeds, c_adj, use_reentrant=False
                )
            else:
                c_out = self.encoder(c_embeds, c_adj)
            local_feats.append(c_out)

        local_flat    = torch.cat(local_feats, dim=0)             # [B*S, L, D]
        local_features = local_flat.view(B, S, L, d_model)        # [B, S, L, D]

        # ── Dynamic graph modulation ──────────────────────────────────────
        # Build gated causal adj (character state adj is already weighted)
        gated_causal = self.graph_modulator(local_features, causal_adj, idf_weights)

        # ── Global aggregation with HGT over 2 typed graphs ──────────────
        enriched = self.global_aggregator(
            local_features, gated_causal, char_state_adj
        )   # [B, S, L, D]

        # ── RAFT v2 modality fusion ───────────────────────────────────────
        flat_enriched = enriched.view(B * S, L, d_model)
        encoded_mem   = self.decoder(
            flat_enriched,
            action_mask.view(B * S, L),
            dial_mask.view(B * S, L),
            ent_mask.view(B * S, L),
            head_mask.view(B * S, L),
        )   # [B*S, L, D]

        encoder_mem_4d  = encoded_mem.view(B, S, L, d_model)
        movie_level_mem = self.memory_norm(encoder_mem_4d.mean(dim=2))  # [B, S, D]
        aligned_memory  = self.latent_bridge(movie_level_mem)

        if target_ids is not None:
            single_target = target_ids[:, 0, :]              # [B, L]
            # C1: BART decoder must start from decoder_start_token_id=2.
            # Extraction stores targets as [BOS=0, t1..tn, EOS=2, PAD=1...].
            # Correct teacher-forcing:
            #   decoder input  = [2,  t1, t2, ..., t_{L-2}]  (len L-1)
            #   labels         = [t1, t2, ..., t_{L-1}]       (len L-1, strip BOS)
            labels        = single_target[:, 1:].contiguous()           # strip BOS
            dec_start     = torch.full((B, 1), 2, dtype=torch.long, device=input_ids.device)
            shifted_targets = torch.cat(
                [dec_start, single_target[:, 1:-1]], dim=1
            ).contiguous()                                               # [B, L-1]

            mem_pad_mask  = (input_ids[:, :, 0] == pad_id)
            all_masked    = mem_pad_mask.all(dim=1)
            mem_pad_mask[all_masked, 0] = False
            enc_attn_mask = (~mem_pad_mask).long()
            tgt_attn_mask = (shifted_targets != pad_id).long()

            decoder_out   = self.bart_decoder(
                input_ids=shifted_targets,
                attention_mask=tgt_attn_mask,
                encoder_hidden_states=aligned_memory,
                encoder_attention_mask=enc_attn_mask,
            )
            dec_hidden    = decoder_out.last_hidden_state    # [B, T, D]
            vocab_logits  = self.head(dec_hidden).float()

            if triplets is not None and self.tokenizer is not None:
                vocab_probs   = F.softmax(vocab_logits, dim=-1)
                # C6: use lm_head weight (same BART vocab space as decoder)
                p_gen, ptr_probs = self.pointer_head(
                    dec_hidden, movie_level_mem, triplets,
                    self.tokenizer, self.head.weight, input_ids.device
                )
                final_probs = p_gen * vocab_probs + (1 - p_gen) * ptr_probs
                final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                final_log_probs = torch.log(final_probs + 1e-8)
            else:
                final_log_probs = F.log_softmax(vocab_logits, dim=-1)

            return final_log_probs, encoder_mem_4d, labels, dec_hidden, gated_causal

        else:
            return aligned_memory, encoder_mem_4d


# =============================================================================
# 11. Logging helpers
# =============================================================================

def log_character_attention_map(model, batch_hidden_states, causal_adj):
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m.eval()
    with torch.no_grad():
        gate_matrix = m.graph_modulator(
            batch_hidden_states, causal_adj
        )[0].detach().cpu().float().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(gate_matrix, cmap="viridis", annot=False)
    plt.title("Dynamic Causal Graph Gate Scores")
    plt.xlabel("Scene Index")
    plt.ylabel("Scene Index")
    wandb.log({"causal_gate_map": wandb.Image(plt)})
    plt.close()


def log_character_attention_map_labeled(model, batch_hidden_states,
                                         causal_adj, triplets):
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m.eval()
    with torch.no_grad():
        gate_matrix = m.graph_modulator(
            batch_hidden_states, causal_adj
        )[0].detach().cpu().float().numpy()
    scene_labels = []
    for i, scene_trips in enumerate(triplets[0]):
        ents = set()
        for trip in scene_trips:
            parts = trip.split("_")
            if len(parts) == 3:
                ents.add(parts[0])
                ents.add(parts[2])
        label = ", ".join(list(ents)[:2])
        scene_labels.append(f"S{i}: {label}" if label else f"S{i}")
    plt.figure(figsize=(12, 10))
    sns.heatmap(gate_matrix, cmap="rocket",
                xticklabels=scene_labels, yticklabels=scene_labels)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.title("Character-Driven Narrative Graph")
    wandb.log({"labeled_causal_gate": wandb.Image(plt)})
    plt.close()
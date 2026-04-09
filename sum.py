"""
sum.py  —  Dual-Tower Dynamic Hypergraph Summariser
====================================================
Architecture
------------
Tower 1 — Text stream (Mamba SSM)
    Frozen RoBERTa → Linear(768→d_model) → MambaBlock → RAFT modality fusion
    Captures sequential narrative flow, dialogue rhythm, syntactic texture.
    Output: H_text [B, S, L, d_model]

Tower 2 — Dynamic Hypergraph (DHEG)
    Each scene is a typed hyperedge connecting all named entities present.
    Two-stage HGNN propagation per scene:
        Stage 1 (node→hyperedge):  e_j = W_e · mean({h_v : v∈e_j}) + type_embed_j
        Stage 2 (hyperedge→node):  h_v = GRU(h_v, message_from_e_j)
    Entity states update sequentially as scenes are processed.
    Captures factual entity continuity, character arcs, typed interactions.
    Output: H_hyperedges [B, S, d_model], H_nodes [B, MAX_ENTITIES, d_model]

Fusion
    Learned gate: memory = σ(W·[H_text, H_hyp]) ⊙ H_text + (1−σ) ⊙ H_hyp
    Decoder cross-attends to concat([fused_scenes, entity_nodes]) [B, S+N, D]

Decoder
    Frozen BART-large decoder with trainable cross-attention.
    HierarchicalPointerHeadV2: copy rare entity tokens via learned salience gate.

Positioning vs DiscoGraMS
    DiscoGraMS: static pairwise discourse graph + transformer encoder + decoder.
    This paper: dynamic hypergraph (scene = hyperedge, typed, GRU-updated)
                + dual tower (Mamba text + DHEG) + BART decoder.
    Two simultaneous upgrades: dynamic vs static, hypergraph vs pairwise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import random
import math
import os
import wandb
from transformers import BartForConditionalGeneration, RobertaModel
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Global config
# =============================================================================
import os as _os
_BART_LARGE_LOCAL = "/tmp/uday/bart-large"
BART_MODEL = (_BART_LARGE_LOCAL if _os.path.isdir(_BART_LARGE_LOCAL)
              else "facebook/bart-large")

# Hyperedge and entity type registries (must match emnlp_extractor.py)
HYPEREDGE_TYPE_MAP  = {"CONFLICT": 0, "ALLIANCE": 1, "DECEPTION": 2,
                        "DIALOGUE": 3, "NEUTRAL": 4}
NUM_HYPEREDGE_TYPES = 5

ENTITY_TYPE_MAP  = {"PERSON": 0, "ORG": 1, "GPE": 2, "FACILITY": 3, "OTHER": 4}
NUM_ENTITY_TYPES = 5

MAX_ENTITIES = 100   # per movie — padded; must match train.py


# =============================================================================
# 0. Utilities
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        x_f32 = x.float()
        out = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


# =============================================================================
# 1. Tower 1 — Text stream (Mamba SSM)
# =============================================================================

class MambaLayer(nn.Module):
    """
    Selective SSM (Mamba-style) — purely sequential token modelling.
    No graph adjacency routing: that belongs in Tower 2.
    O(seq_len) memory, no quadratic history accumulation.
    """
    def __init__(self, d_model, d_state, d_conv):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * 2

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv - 1)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        batch, seq_len, _ = x.shape

        xz       = self.in_proj(x)
        x_in, z  = xz.chunk(2, dim=-1)
        x_conv   = self.conv1d(x_in.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        x_act    = F.silu(x_conv)

        x_ssm              = self.x_proj(x_act)
        delta, B_mat, C    = torch.split(x_ssm, [1, self.d_state, self.d_state], dim=-1)
        delta              = F.softplus(self.dt_proj(delta).float())
        B_mat, C           = B_mat.float(), C.float()
        x_act_f32          = x_act.float()

        h = torch.zeros(batch, self.d_inner, self.d_state,
                        device=x.device, dtype=torch.float32)
        outs = []
        for t in range(seq_len):
            h = h * torch.exp(-delta[:, t].unsqueeze(-1)) + \
                (B_mat[:, t].unsqueeze(1) * x_act_f32[:, t].unsqueeze(-1))
            outs.append((h * C[:, t].unsqueeze(1)).sum(dim=-1))

        y = torch.stack(outs, dim=1)
        return self.out_proj(y * F.silu(z)) + residual


# Backward-compat alias
GraMambaLayer = MambaLayer


class MambaBlock(nn.Module):
    """Stack of MambaLayers for scene-level token encoding."""
    def __init__(self, d_model, d_state, d_conv, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaLayer(d_model, d_state, d_conv) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


GraphModulatedMambaBlock = MambaBlock   # backward compat


# =============================================================================
# 2. RAFT Consensus Attention v2 — cross-modal fusion within a scene
# =============================================================================

class RaftConsensusAttentionV2(nn.Module):
    """
    Fuses the 4 modality streams (action / dialogue / entity / header)
    via cross-modal multi-head attention inside each scene.
    Applied after MambaBlock, before mean-pooling to scene vectors.
    """
    def __init__(self, d_model=1024, num_heads=4):
        super().__init__()
        self.action_proj = nn.Linear(d_model, d_model)
        self.dial_proj   = nn.Linear(d_model, d_model)
        self.ent_proj    = nn.Linear(d_model, d_model)
        self.head_proj   = nn.Linear(d_model, d_model)
        self.cross_attn  = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, dropout=0.1
        )
        self.consensus_gate = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, features, action_mask, dial_mask, ent_mask, head_mask):
        masks = [action_mask, dial_mask, ent_mask, head_mask]
        projs = [self.action_proj, self.dial_proj, self.ent_proj, self.head_proj]
        modalities = [
            projs[i](features) * masks[i].unsqueeze(-1).float()
            for i in range(4)
        ]
        B, L, D    = features.shape
        modal_flat = torch.stack(modalities, dim=2).view(B * L, 4, D)
        attended, _= self.cross_attn(modal_flat, modal_flat, modal_flat)
        attended   = attended.view(B, L, 4, D)
        consensus  = self.consensus_gate(attended.reshape(B, L, 4 * D))
        return self.norm(features + consensus)


# =============================================================================
# 3. Tower 2 — Dynamic Hypergraph Encoder (DHEG)
# =============================================================================

class DynamicHypergraphTower(nn.Module):
    """
    Dynamic Heterogeneous Event Hypergraph (DHEG) Tower.

    Each scene s is a typed hyperedge e_s connecting all named entities
    present in that scene.  Entity states h_v evolve via a GRU update
    as scenes are processed sequentially.

    Two-stage HGNN propagation per scene:
        Stage 1 — node→hyperedge:
            e_s = LayerNorm(W_node · mean({h_v : v∈e_s}) + type_embed_s + text_s)
        Stage 2 — hyperedge→node (GRU update):
            For each entity v in scene s:
                h_v ← GRU(h_v,  W_msg · e_s)

    Parameters
    ----------
    d_model       : hidden dimension (must match text tower)
    max_entities  : padded entity count per movie (MAX_ENTITIES = 100)
    num_edge_types: 5 typed hyperedges (CONFLICT/ALLIANCE/DECEPTION/DIALOGUE/NEUTRAL)
    num_entity_types: 5 node types (PERSON/ORG/GPE/FACILITY/OTHER)
    """

    def __init__(self, d_model=1024, max_entities=100,
                 num_edge_types=5, num_entity_types=5):
        super().__init__()
        self.d_model      = d_model
        self.max_entities = max_entities

        # Type embeddings — initialise entity state + classify hyperedge
        self.edge_type_embed   = nn.Embedding(num_edge_types,   d_model)
        self.entity_type_embed = nn.Embedding(num_entity_types, d_model)

        # Stage 1: node → hyperedge
        self.node_to_edge  = nn.Linear(d_model, d_model)
        self.text_to_edge  = nn.Linear(d_model, d_model)  # inject text tower context
        self.edge_norm     = nn.LayerNorm(d_model)

        # Stage 2: hyperedge → node via GRU
        self.msg_proj   = nn.Linear(d_model, d_model)
        self.entity_gru = nn.GRUCell(d_model, d_model)

        # Output transform for hyperedge representations
        self.edge_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.residual_norm = nn.LayerNorm(d_model)

    def forward(self, scene_reps, incidence_matrix,
                edge_type_ids, entity_type_ids, entity_mask):
        """
        Parameters
        ----------
        scene_reps        : [B, S, D]  mean-pooled text tower output
        incidence_matrix  : [B, N, S]  float — B[n,s]=1 if entity n in scene s
        edge_type_ids     : [B, S]     long  — hyperedge type per scene
        entity_type_ids   : [B, N]     long  — entity type per node
        entity_mask       : [B, N]     bool  — True = valid (not padding)

        Returns
        -------
        H_hyperedges : [B, S, D]  — one representation per scene
        H_nodes      : [B, N, D]  — final entity state vectors
        """
        B, S, D = scene_reps.shape
        N = self.max_entities

        # Initialise entity states from type embeddings
        h_entities  = self.entity_type_embed(entity_type_ids)           # [B, N, D]
        ent_mask_f  = entity_mask.unsqueeze(-1).float()                  # [B, N, 1]
        h_entities  = h_entities * ent_mask_f

        edge_type_embs = self.edge_type_embed(edge_type_ids)             # [B, S, D]

        hyperedge_list = []

        for s in range(S):
            # Entities present in scene s: [B, N], float, masked
            in_scene = incidence_matrix[:, :, s] * entity_mask.float()  # [B, N]

            # ── Stage 1: Node → Hyperedge ────────────────────────────────
            entity_count = in_scene.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
            # Weighted mean of entity states
            mean_ent = torch.bmm(
                in_scene.unsqueeze(1),   # [B, 1, N]
                h_entities               # [B, N, D]
            ).squeeze(1) / entity_count  # [B, D]

            # Combine: node mean + type embedding + text tower scene rep
            e_s = self.edge_norm(
                self.node_to_edge(mean_ent)
                + edge_type_embs[:, s, :]
                + self.text_to_edge(scene_reps[:, s, :])
            )                                                            # [B, D]
            hyperedge_list.append(e_s)

            # ── Stage 2: Hyperedge → Node (GRU) ─────────────────────────
            msg = self.msg_proj(e_s)                                     # [B, D]
            msg_exp = msg.unsqueeze(1).expand(-1, N, -1)                 # [B, N, D]

            # Flatten B×N for GRUCell (each entity updated independently)
            h_flat   = h_entities.reshape(B * N, D)
            msg_flat = msg_exp.reshape(B * N, D)
            h_new    = self.entity_gru(msg_flat, h_flat).reshape(B, N, D)

            # Only update entities that (a) are in this scene AND (b) are valid
            upd = in_scene.unsqueeze(-1)                                 # [B, N, 1]
            h_entities = h_entities * (1 - upd) + h_new * upd

        H_hyperedges = torch.stack(hyperedge_list, dim=1)                # [B, S, D]
        H_hyperedges = self.residual_norm(H_hyperedges + scene_reps)     # residual
        H_hyperedges = self.edge_out(H_hyperedges)                       # [B, S, D]

        return H_hyperedges, h_entities                                  # [B,S,D], [B,N,D]


# =============================================================================
# 4. Hierarchical Pointer Head v2
# =============================================================================

class HierarchicalPointerHeadV2(nn.Module):
    """
    Dual-level copy mechanism.
    Scene-level: decoder attends over fused scene memory.
    Entity-level: salience = dot(decoder_state, entity_node_vector).
    Hard sigmoid p_gen for near-binary copy/generate decisions.
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.vocab_size    = vocab_size
        self.query_proj    = nn.Linear(d_model, d_model)
        self.key_proj      = nn.Linear(d_model, d_model)
        self.p_gen_linear  = nn.Linear(d_model * 2, 1)
        nn.init.constant_(self.p_gen_linear.bias, 3.0)

    def forward(self, decoder_states, scene_memory,
                triplets, tokenizer, embedding_weight, device):
        """
        decoder_states  : [B, T, D]
        scene_memory    : [B, S, D]   fused memory (scenes only, not nodes)
        embedding_weight: [V, D]
        """
        B, T, D = decoder_states.shape
        _, S, _ = scene_memory.shape

        q          = self.query_proj(decoder_states)
        k          = self.key_proj(scene_memory)
        scores     = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)
        scene_attn = F.softmax(scores, dim=-1)                          # [B, T, S]

        context   = torch.matmul(scene_attn, scene_memory)             # [B, T, D]
        p_gen     = torch.sigmoid(
            (self.p_gen_linear(torch.cat([decoder_states, context], dim=-1)) - 0.5) * 10.0
        )

        scene_to_token = torch.zeros(B, S, self.vocab_size, device=device)

        if triplets and tokenizer is not None:
            dec_last = decoder_states[:, -1, :]
            for b in range(B):
                ns = min(S, len(triplets[b]) if isinstance(triplets[b], list) else 1)
                for s in range(ns):
                    scene_trips = (triplets[b][s]
                                   if isinstance(triplets[b], list) and s < len(triplets[b])
                                   else [])
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
                    encoded  = tokenizer(list(entities), add_special_tokens=False)["input_ids"]
                    tok_ids  = list({tid for sub in encoded for tid in sub
                                     if tid < self.vocab_size})
                    if not tok_ids:
                        continue
                    ent_t    = torch.tensor(tok_ids, device=device)
                    ent_embs = F.embedding(ent_t, embedding_weight)
                    salience = F.softmax(
                        torch.matmul(dec_last[b].unsqueeze(0).float(),
                                     ent_embs.float().T).squeeze(0), dim=-1
                    )
                    scene_to_token[b, s, tok_ids] = salience.to(scene_to_token.dtype)

        pointer_probs = torch.matmul(scene_attn, scene_to_token)        # [B, T, V]
        return p_gen, pointer_probs


# =============================================================================
# 5. Narrative Coherence Loss (scene-pair contrastive via incidence overlap)
# =============================================================================

class NarrativeCoherenceLoss(nn.Module):
    """
    Scenes that share named entities (high Jaccard incidence overlap)
    should produce coherent, related summary segments.
    NT-Xent scene-pair contrastive loss using incidence matrix for positive pairs.

    Works correctly at batch_size=1 — positives are scene pairs sharing entities.
    Replaces the old causal_adj version which was broken at B=1.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, scene_hidden, incidence_matrix):
        """
        scene_hidden      : [B, S, L, D] or [B, S, D]
        incidence_matrix  : [B, N, S] float
        """
        if scene_hidden.dim() == 4:
            scene_reps = scene_hidden.mean(dim=2)   # [B, S, D]
        else:
            scene_reps = scene_hidden
        scene_reps = F.normalize(scene_reps.float(), p=2, dim=-1)

        total_loss = scene_reps.new_tensor(0.0)
        n_valid    = 0

        for b in range(scene_reps.size(0)):
            B_mat  = incidence_matrix[b].float()    # [N, S]
            inter  = torch.mm(B_mat.T, B_mat)       # [S, S]
            sums   = B_mat.sum(dim=0, keepdim=True) # [1, S]
            union  = sums + sums.T - inter
            jac    = inter / union.clamp(min=1e-8)  # [S, S]

            pos_mask = (jac > 0.25).float()
            pos_mask.fill_diagonal_(0)
            if pos_mask.sum() == 0:
                continue

            reps    = scene_reps[b]
            sim     = torch.matmul(reps, reps.T) / self.temperature
            sim.fill_diagonal_(-1e4)
            log_p   = F.log_softmax(sim, dim=-1)
            pos_n   = pos_mask / pos_mask.sum(dim=1, keepdim=True).clamp(min=1)
            loss_s  = -(pos_n * log_p).sum(dim=-1)

            has_pos = pos_mask.sum(dim=1) > 0
            if not has_pos.any():
                continue
            total_loss = total_loss + loss_s[has_pos].mean()
            n_valid   += 1

        return total_loss / max(n_valid, 1)


# =============================================================================
# 6. Relational Event Consistency Loss
# =============================================================================

class RelationalEventConsistencyLoss(nn.Module):
    """
    Combined loss:
        (1-α) * smooth NLL with entity token up-weighting
        + α   * contrastive triplet loss (SVO triplets vs reversed negatives)
        + coherence_weight * NarrativeCoherenceLoss (entity-overlap contrastive)

    Riemannian orthogonality penalty removed — unjustified overhead.
    causal_adj replaced by incidence_matrix for coherence loss.
    """
    def __init__(self, alpha=0.1, tokenizer=None,
                 temperature=0.1, entity_penalty=3.0,
                 label_smoothing=0.1, coherence_weight=0.05):
        super().__init__()
        self.alpha            = alpha
        self.tokenizer        = tokenizer
        self.temperature      = temperature
        self.entity_penalty   = entity_penalty
        self.label_smoothing  = label_smoothing
        self.coherence_weight = coherence_weight
        self.coherence_fn     = NarrativeCoherenceLoss(temperature=0.1)

    def _smooth_nll(self, log_probs, targets, weight_mask):
        V      = log_probs.size(-1)
        smooth = torch.zeros_like(log_probs).fill_(self.label_smoothing / V)
        smooth.scatter_(1, targets.unsqueeze(1).clamp(min=0), 1.0 - self.label_smoothing)
        nll    = -(smooth * log_probs).sum(dim=-1)
        valid  = (targets != 1).float()
        return (nll * weight_mask * valid).sum() / (weight_mask * valid).sum().clamp(min=1.0)

    def _triplet_embeds(self, trips, head_weight, device, max_t=5):
        if not trips or not trips[0]:
            return None
        texts = []
        for scene_trips in trips:
            sampled = (random.sample(scene_trips, max_t)
                       if len(scene_trips) > max_t else scene_trips)
            texts.append(" ".join(sampled).replace("_", " "))
        enc     = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=32, return_tensors="pt").to(device)
        embs    = F.embedding(enc["input_ids"], head_weight)
        mask_e  = enc["attention_mask"].unsqueeze(-1).expand(embs.size()).float()
        return (embs * mask_e).sum(1) / mask_e.sum(1).clamp(min=1e-4)

    def forward(self, log_probs, targets, triplets,
                hidden_states=None, head_weight=None,
                incidence_matrix=None):
        device = log_probs.device

        # Entity up-weighting
        weight_mask = torch.ones_like(targets, dtype=torch.float)
        if triplets and triplets[0] and self.tokenizer is not None:
            flat_ents = []
            for st in triplets:
                for t in st:
                    p = t.split("_")
                    if len(p) >= 1: flat_ents.append(p[0].replace("NOT ", "").strip())
                    if len(p) >= 3: flat_ents.append(p[2].strip())
            if flat_ents:
                enc = self.tokenizer(flat_ents, add_special_tokens=False)["input_ids"]
                ids = set(tid for sub in enc for tid in sub)
                if ids:
                    ent_t = torch.tensor(list(ids), device=device)
                    weight_mask[torch.isin(targets, ent_t)] = self.entity_penalty

        lm_loss = self._smooth_nll(log_probs, targets, weight_mask)

        if (not triplets or not triplets[0]
                or hidden_states is None or head_weight is None):
            return lm_loss

        # Contrastive triplet loss
        B, S, L, D  = hidden_states.shape
        h_flat       = hidden_states.view(B * S, L, D).mean(dim=1)
        valid_idx    = [i for i, t in enumerate(triplets) if len(t) > 0]
        if not valid_idx:
            return lm_loss

        valid_trips  = [triplets[i] for i in valid_idx]
        neg_trips    = [[f"{p.split('_')[2]}_{p.split('_')[1]}_{p.split('_')[0]}"
                         if len(p.split("_")) == 3 else p
                         for p in bt] for bt in valid_trips]
        pos_embs     = self._triplet_embeds(valid_trips, head_weight, device)
        neg_embs     = self._triplet_embeds(neg_trips,   head_weight, device)
        if pos_embs is None or neg_embs is None:
            return lm_loss

        h_v     = h_flat[valid_idx]
        n       = min(h_v.size(0), pos_embs.size(0))
        hn      = F.normalize(h_v[:n].float(),   p=2, dim=1, eps=1e-4)
        pn      = F.normalize(pos_embs[:n].float(), p=2, dim=1, eps=1e-4)
        nn_     = F.normalize(neg_embs[:n].float(), p=2, dim=1, eps=1e-4)
        logits_c = torch.stack([(hn * pn).sum(-1) / self.temperature,
                                 (hn * nn_).sum(-1) / self.temperature], dim=1)
        cont_loss = F.cross_entropy(logits_c, torch.zeros(n, dtype=torch.long, device=device))
        total     = (1 - self.alpha) * lm_loss + self.alpha * cont_loss

        # Narrative coherence loss (incidence-based scene-pair contrastive)
        if incidence_matrix is not None and self.coherence_weight > 0:
            coh = self.coherence_fn(hidden_states, incidence_matrix)
            total = total + self.coherence_weight * coh

        return total


# =============================================================================
# 7. Main model — DualTowerHypergraphSummariser
# =============================================================================

class DualTowerHypergraphSummariser(nn.Module):
    """
    Full dual-tower dynamic hypergraph summarisation model.

    forward() signature (training):
        Returns: (final_log_probs, H_text_4d, labels, dec_hidden, H_hyperedges)

    forward() signature (inference, target_ids=None):
        Returns: (aligned_memory [B, S+N, D],  H_text_4d [B, S, L, D])
    """

    def __init__(self, vocab_size, d_model=1024, num_layers=4,
                 max_entities=MAX_ENTITIES, tokenizer=None):
        super().__init__()
        self.d_model      = d_model
        self.max_entities = max_entities
        self.tokenizer    = tokenizer

        # Load BART backbone
        print(f"Loading BART backbone: {BART_MODEL}...")
        bart = BartForConditionalGeneration.from_pretrained(BART_MODEL)

        # ── Tower 1: Text (RoBERTa frozen → projection → Mamba → RAFT) ───
        print("Loading frozen RoBERTa scene encoder...")
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        for p in self.roberta.parameters():
            p.requires_grad = False

        self.scene_proj = (nn.Linear(768, d_model, bias=False)
                           if d_model != 768 else nn.Identity())

        self.mamba_tower = MambaBlock(d_model, d_state=64, d_conv=4,
                                      num_layers=num_layers)

        self.raft = RaftConsensusAttentionV2(d_model=d_model, num_heads=4)

        # ── Tower 2: Dynamic Hypergraph ───────────────────────────────────
        self.hypergraph_tower = DynamicHypergraphTower(
            d_model=d_model,
            max_entities=max_entities,
            num_edge_types=NUM_HYPEREDGE_TYPES,
            num_entity_types=NUM_ENTITY_TYPES,
        )

        # ── Gated fusion: text scenes + hyperedge scenes ──────────────────
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(d_model)

        # ── BART decoder (frozen except cross-attention) ──────────────────
        self.bart_decoder = bart.model.decoder
        self.head         = bart.lm_head

        for p in self.bart_decoder.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False
        for name, p in self.bart_decoder.named_parameters():
            if "encoder_attn" in name:
                p.requires_grad = True

        # ── Pointer head ──────────────────────────────────────────────────
        self.pointer_head = HierarchicalPointerHeadV2(d_model, vocab_size)

        self.memory_norm      = nn.LayerNorm(d_model)
        self.use_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.use_checkpointing = True

    # ── Internal: text tower ───────────────────────────────────────────────

    def _text_tower(self, input_ids, action_mask, dial_mask, ent_mask, head_mask):
        """
        Returns H_text [B, S, D] (pooled) and H_text_4d [B, S, L, D] (token-level).
        RoBERTa always runs under no_grad.
        """
        B, S, L = input_ids.shape
        pad_id   = 1
        chunk_sz = 32

        local_feats = []
        for i in range(0, S, chunk_sz):
            j     = min(i + chunk_sz, S)
            c_ids = input_ids[:, i:j].contiguous().view(-1, L)
            c_att = (c_ids != pad_id).long()

            with torch.no_grad():
                c_rob = self.roberta(c_ids, attention_mask=c_att).last_hidden_state
            c_emb = self.scene_proj(c_rob)

            if self.use_checkpointing:
                c_out = checkpoint(self.mamba_tower, c_emb, use_reentrant=False)
            else:
                c_out = self.mamba_tower(c_emb)
            local_feats.append(c_out)

        local_flat   = torch.cat(local_feats, dim=0)              # [B*S, L, D]
        H_text_4d    = local_flat.view(B, S, L, self.d_model)     # [B, S, L, D]

        # RAFT modality fusion across 4 mask types
        flat   = H_text_4d.view(B * S, L, self.d_model)
        fused  = self.raft(
            flat,
            action_mask.view(B * S, L),
            dial_mask.view(B * S, L),
            ent_mask.view(B * S, L),
            head_mask.view(B * S, L),
        )                                                          # [B*S, L, D]
        H_text_4d = fused.view(B, S, L, self.d_model)
        H_text    = H_text_4d.mean(dim=2)                         # [B, S, D]
        return H_text, H_text_4d

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, input_ids, action_mask, dial_mask, ent_mask, head_mask,
                incidence_matrix, edge_type_ids, entity_type_ids, entity_mask,
                target_ids=None, triplets=None):
        """
        Parameters
        ----------
        input_ids         : [B, S, L]
        *_mask            : [B, S, L]
        incidence_matrix  : [B, N, S]  float
        edge_type_ids     : [B, S]     long
        entity_type_ids   : [B, N]     long
        entity_mask       : [B, N]     bool

        Training (target_ids provided):
            Returns (log_probs, H_text_4d, labels, dec_hidden, H_hyperedges)

        Inference (target_ids=None):
            Returns (aligned_memory [B, S+N, D], H_text_4d [B, S, L, D])
        """
        B, S, L = input_ids.shape
        N       = self.max_entities
        pad_id  = 1

        # ── Tower 1 ────────────────────────────────────────────────────────
        H_text, H_text_4d = self._text_tower(
            input_ids, action_mask, dial_mask, ent_mask, head_mask
        )
        # H_text: [B, S, D],  H_text_4d: [B, S, L, D]

        # ── Tower 2 ────────────────────────────────────────────────────────
        H_hyperedges, H_nodes = self.hypergraph_tower(
            H_text, incidence_matrix, edge_type_ids, entity_type_ids, entity_mask
        )
        # H_hyperedges: [B, S, D],  H_nodes: [B, N, D]

        # ── Gated fusion ───────────────────────────────────────────────────
        gate         = self.fusion_gate(torch.cat([H_text, H_hyperedges], dim=-1))
        fused_scenes = self.fusion_norm(gate * H_text + (1 - gate) * H_hyperedges)
        # fused_scenes: [B, S, D]

        # ── Decoder memory: fused scenes + entity nodes ────────────────────
        # Valid entity nodes only (zero out padding)
        valid_nodes    = entity_mask.unsqueeze(-1).float() * H_nodes   # [B, N, D]
        aligned_memory = torch.cat([fused_scenes, valid_nodes], dim=1) # [B, S+N, D]
        aligned_memory = self.memory_norm(aligned_memory)

        if target_ids is not None:
            single_target   = target_ids[:, 0, :]                      # [B, L]
            labels          = single_target[:, 1:].contiguous()
            dec_start       = torch.full((B, 1), 2, dtype=torch.long,
                                         device=input_ids.device)
            shifted_targets = torch.cat(
                [dec_start, single_target[:, 1:-1]], dim=1
            ).contiguous()                                              # [B, L-1]

            # Encoder attention mask: scene positions + entity positions
            mem_pad = torch.zeros(B, S + N, dtype=torch.bool,
                                  device=input_ids.device)
            mem_pad[:, :S] = (input_ids[:, :, 0] == pad_id)
            mem_pad[:, S:] = ~entity_mask
            # Ensure at least one unmasked position
            all_masked = mem_pad.all(dim=1)
            mem_pad[all_masked, 0] = False
            enc_attn_mask = (~mem_pad).long()
            tgt_attn_mask = (shifted_targets != pad_id).long()

            decoder_out   = self.bart_decoder(
                input_ids=shifted_targets,
                attention_mask=tgt_attn_mask,
                encoder_hidden_states=aligned_memory,
                encoder_attention_mask=enc_attn_mask,
            )
            dec_hidden    = decoder_out.last_hidden_state               # [B, T, D]
            vocab_logits  = self.head(dec_hidden).float()

            if triplets is not None and self.tokenizer is not None:
                vocab_probs    = F.softmax(vocab_logits, dim=-1)
                p_gen, ptr_pr  = self.pointer_head(
                    dec_hidden, fused_scenes, triplets,
                    self.tokenizer, self.head.weight, input_ids.device
                )
                final_probs = p_gen * vocab_probs + (1 - p_gen) * ptr_pr
                final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                final_log_probs = torch.log(final_probs + 1e-8)
            else:
                final_log_probs = F.log_softmax(vocab_logits, dim=-1)

            return final_log_probs, H_text_4d, labels, dec_hidden, H_hyperedges

        else:
            return aligned_memory, H_text_4d


# Backward-compat alias so any external imports of GraMFormerV2 still resolve.
GraMFormerV2 = DualTowerHypergraphSummariser


# =============================================================================
# 8. Logging helpers
# =============================================================================

def log_hyperedge_attention(model, H_hyperedges, incidence_matrix, movie_name=""):
    """
    Log a heatmap showing hyperedge representation similarity (proxy for
    which scenes the model considers structurally related).
    """
    if not wandb.run:
        return
    with torch.no_grad():
        reps  = H_hyperedges[0].float().cpu()               # [S, D]
        reps  = F.normalize(reps, p=2, dim=-1)
        sim   = torch.matmul(reps, reps.T).numpy()           # [S, S]
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim, cmap="viridis", vmin=0, vmax=1, annot=False)
    plt.title(f"Hyperedge similarity — {movie_name}")
    plt.xlabel("Scene index")
    plt.ylabel("Scene index")
    wandb.log({"hyperedge_sim": wandb.Image(plt)})
    plt.close()


def log_entity_state_norms(H_nodes, entity_mask, step):
    """Log mean entity state vector norm — proxy for whether GRU is learning."""
    if not wandb.run:
        return
    with torch.no_grad():
        valid = H_nodes[0][entity_mask[0]]                   # [n_valid, D]
        if valid.size(0) > 0:
            mean_norm = valid.float().norm(dim=-1).mean().item()
            wandb.log({"entity_state_norm": mean_norm, "step": step})
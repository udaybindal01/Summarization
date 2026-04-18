"""
sum.py  —  LED + Mamba-Hypergraph Narrative Summariser (v5)
============================================================
Architecture
------------
Tower 1 — Text stream (LED encoder)
    Full screenplay → LED encoder (16K tokens) → scene-level pooling
    Output: H_text [B, S, d_model]

Tower 2 — Dynamic 3-Stream Hypergraph with LED-grounded entity init
    + Mamba Temporal Dynamics
    Entity states initialized from LED encoder scene representations
    (grounded, not random). Three message streams per scene:
        Stream 1 (Scene): Entity-aware bilinear attention on scene reps.
        Stream 2 (Arc): Attention over past shared hyperedges with decay.
        Stream 3 (Interaction): Social context from co-occurring entities.
    Per-entity trajectories processed by Mamba SSM for temporal dynamics.

Fusion
    Cross-Attention Adapter over [fused_scenes, entity_nodes] → LED decoder.

Key contributions:
1. LED-grounded entity initialization — entities begin with scene-derived
   representations, not random embeddings.
2. Entity-aware scene messages — bilinear attention where entity identity
   determines what information is extracted from each scene.
3. Selective State Spaces for entity temporal dynamics — Mamba's
   input-dependent gating learns *when* entity states change
   (narrative turning points) vs. persist, yielding interpretable dt values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import random
import math
import os
import wandb
from transformers import LEDForConditionalGeneration
import matplotlib.pyplot as plt
import seaborn as sns

import os as _os
_LED_LOCAL = "/tmp/uday/led-large-16384"
LED_MODEL = (_LED_LOCAL if _os.path.isdir(_LED_LOCAL)
             else "allenai/led-large-16384")

ENTITY_TYPE_MAP     = {"PERSON": 0, "ORG": 1, "GPE": 2, "FACILITY": 3, "OTHER": 4}
NUM_ENTITY_TYPES    = 5
HYPEREDGE_TYPE_MAP  = {"CONFLICT": 0, "ALLIANCE": 1, "DECEPTION": 2, "DIALOGUE": 3, "NEUTRAL": 4}
NUM_HYPEREDGE_TYPES = 5
MAX_ENTITIES = 100


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
# 1. Entity Mamba — Selective SSM for temporal entity state dynamics
# =============================================================================

class EntityMambaLayer(nn.Module):
    """
    Single Mamba layer operating on entity state trajectories.
    Input: [B*N, S, D] — each entity's state across S scenes.

    Supports emotion modulation: an optional emotion_bias [B*N, S, 1] adds
    a learned per-scene bias to the dt (state-change magnitude), making
    emotionally intense scenes produce larger state changes. emotion_scale
    is initialized to 0 so it doesn't affect early training.
    """
    def __init__(self, d_model, d_state=32, d_conv=4):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * 2
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                  padding=d_conv - 1, groups=self.d_inner)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        # Emotion bias scale — init at 0 so emotion has no effect at start
        self.emotion_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, emotion_bias=None, return_dt=False):
        """
        x:             [B*N, S, D]
        emotion_bias:  [B*N, S, 1] — per-scene emotional intensity per entity
        Returns: output [B*N, S, D], optionally dt_values [B*N, S, d_inner]
        """
        residual = x
        x = self.norm(x)
        batch, seq_len, _ = x.shape
        xz       = self.in_proj(x)
        x_in, z  = xz.chunk(2, dim=-1)
        x_conv   = self.conv1d(x_in.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        x_act    = F.silu(x_conv)
        x_ssm    = self.x_proj(x_act)
        delta, B_mat, C = torch.split(x_ssm, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta)).float()

        # Emotion modulation: emotionally intense scenes → larger state change
        if emotion_bias is not None:
            # emotion_bias: [B*N, S, 1] → broadcast over d_inner
            delta = delta + self.emotion_scale.float() * emotion_bias.float()
            delta = delta.clamp(min=1e-4)  # keep positive after bias

        B_mat    = B_mat.float()
        C        = C.float()
        x_act_f32 = x_act.float()

        h = torch.zeros(batch, self.d_inner, self.d_state,
                         device=x.device, dtype=torch.float32)
        y = torch.zeros(batch, seq_len, self.d_inner,
                         device=x.device, dtype=torch.float32)

        for t in range(seq_len):
            h = h * torch.exp(-delta[:, t].unsqueeze(-1)) + \
                (B_mat[:, t].unsqueeze(1) * x_act_f32[:, t].unsqueeze(-1))
            h = h.clamp(-100.0, 100.0)
            y[:, t, :] = (h * C[:, t].unsqueeze(1)).sum(dim=-1)

        output = self.out_proj(y.to(x.dtype) * F.silu(z)) + residual

        if return_dt:
            return output, delta  # delta = [B*N, S, d_inner] — interpretable
        return output


class EntityMambaBlock(nn.Module):
    """Stack of EntityMambaLayers for deep temporal modeling."""
    def __init__(self, d_model, d_state=32, d_conv=4, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            EntityMambaLayer(d_model, d_state, d_conv) for _ in range(num_layers)
        ])

    def forward(self, x, emotion_bias=None, return_dt=False):
        dt_values = None
        for i, layer in enumerate(self.layers):
            last = (i == len(self.layers) - 1)
            if return_dt and last:
                x, dt_values = layer(x, emotion_bias=emotion_bias, return_dt=True)
            else:
                x = layer(x, emotion_bias=emotion_bias)
        if return_dt:
            return x, dt_values
        return x


# =============================================================================
# 2. Tower 2 — 3-Stream Dynamic Hypergraph with Mamba Temporal Dynamics
# =============================================================================

class DynamicHypergraphTower(nn.Module):
    """
    3-Stream Dynamic Hypergraph Encoder with LED-grounded entity init
    and Mamba SSM temporal entity dynamics.

    Key improvements in v5:
    1. LED-grounded entity init — entity states initialized from LED encoder
       scene representations (weighted by incidence role), not random embeddings.
    2. Entity-aware scene messages — bilinear attention where each entity's
       identity determines what information it extracts from a scene.
    3. Attention-based arc stream — learnable temporal attention over past
       shared hyperedges (vs. fixed-weight sum).
    4. Mamba processes per-entity temporal trajectories [B*N, S, D].
    """
    def __init__(self, d_model=1024, max_entities=100,
                 num_entity_types=5, num_hyperedge_types=5,
                 use_adaptive_streams=True, use_entity_names=True,
                 edge_dropout=0.1, mamba_layers=2):
        super().__init__()
        self.d_model      = d_model
        self.max_entities = max_entities
        self.use_adaptive_streams = use_adaptive_streams
        self.use_entity_names     = use_entity_names
        self.edge_dropout_rate    = edge_dropout

        # ── Entity Init: LED-grounded + type + optional name ────────────
        self.entity_type_embed = nn.Embedding(num_entity_types, d_model)
        self.entity_init_proj  = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        if use_entity_names:
            self.entity_name_proj = nn.Sequential(
                nn.Linear(d_model, d_model, bias=False), nn.LayerNorm(d_model),
            )

        # ── Latent Hyperedge Generation (with event-type embedding) ─────
        self.edge_entity_proj = nn.Linear(d_model, d_model)
        self.edge_scene_proj  = nn.Linear(d_model, d_model)
        self.hyperedge_type_embed = nn.Embedding(num_hyperedge_types, d_model)
        self.edge_norm        = nn.LayerNorm(d_model)

        # ── Stream 1: Entity-aware scene messages (bilinear attention) ──
        self.scene_query = nn.Linear(d_model, d_model)
        self.scene_key   = nn.Linear(d_model, d_model)
        self.scene_value = nn.Linear(d_model, d_model)
        self.scene_msg_norm = nn.LayerNorm(d_model)

        # ── Stream 2: Narrative arc (temporal attention over past hyperedges)
        self.arc_query = nn.Linear(d_model, d_model)
        self.arc_key   = nn.Linear(d_model, d_model)
        self.arc_value = nn.Linear(d_model, d_model)
        self.arc_norm  = nn.LayerNorm(d_model)

        # ── Stream 3: Social interaction (static co-occurrence) ─────────
        self.interact_proj = nn.Linear(d_model, d_model)
        self.interact_norm = nn.LayerNorm(d_model)

        # ── Stream 4: Cross-scene relationship evolution ─────────────────
        # Tracks how entity-pair relationships evolve across narrative time.
        # Uses running-accumulated entity states (not static init embeddings)
        # so each entity's "biography so far" conditions the social weight.
        self.rel_proj        = nn.Linear(d_model, d_model)
        self.rel_norm        = nn.LayerNorm(d_model)
        # Learnable alignment weight: how much history alignment modulates
        # the relationship attention. Init at 0 = pure co-occurrence start.
        self.rel_align_scale = nn.Parameter(torch.zeros(1))

        # ── Stream Fusion (4 streams) ───────────────────────────────────
        num_streams = 4
        if use_adaptive_streams:
            self.stream_gate = nn.Sequential(
                nn.Linear(d_model, d_model // 4), nn.GELU(),
                nn.Linear(d_model // 4, num_streams),
            )
        else:
            self.stream_weights = nn.Parameter(torch.zeros(num_streams))

        self.role_proj = nn.Linear(1, d_model, bias=False)

        # ── HGNN Phase 2: global hyperedge→entity aggregation ───────────
        # After Mamba, each entity aggregates from ALL scene hyperedges it
        # belongs to (weighted by incidence role). This closes the HGNN loop:
        #   Phase 1 (Vertex→Hyperedge): entities → e_s  [done in stream loop]
        #   Phase 2 (Hyperedge→Vertex): e_s → h_entities [done here]
        self.he_ent_proj = nn.Linear(d_model, d_model)
        self.he_ent_norm = nn.LayerNorm(d_model)

        # ── Mamba temporal entity dynamics ──────────────────────────────
        self.entity_mamba = EntityMambaBlock(
            d_model, d_state=32, d_conv=4, num_layers=mamba_layers,
        )
        self.entity_update_norm = nn.LayerNorm(d_model)

        # ── Hyperedge output ────────────────────────────────────────────
        self.edge_out = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model),
        )
        self.residual_norm = nn.LayerNorm(d_model)

    def forward(self, scene_reps, incidence_matrix, edge_type_ids,
                entity_type_ids, entity_mask, entity_name_embs=None,
                emotion_matrix=None, return_dt=False):
        """
        Args:
            scene_reps:       [B, S, D] pooled LED encoder scene representations
            incidence_matrix: [B, N, S] float role-weighted
            edge_type_ids:    [B, S] long — event type per scene hyperedge
            entity_type_ids:  [B, N] long
            entity_mask:      [B, N] bool
            entity_name_embs: [B, N, D] optional LED-encoded entity names
            emotion_matrix:   [B, N, S] float — per-entity per-scene polarity
            return_dt:        if True, return Mamba dt values for interpretability
        """
        B, S, D = scene_reps.shape
        N = self.max_entities

        # ── LED-Grounded Entity Initialization ──────────────────────────
        led_grounded = torch.bmm(incidence_matrix, scene_reps)         # [B, N, D]
        presence_sum = incidence_matrix.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        led_grounded = self.entity_init_proj(led_grounded / presence_sum)

        h_entities_init = led_grounded + self.entity_type_embed(entity_type_ids)
        if entity_name_embs is not None and self.use_entity_names:
            h_entities_init = h_entities_init + self.entity_name_proj(entity_name_embs)

        ent_mask_f = entity_mask.unsqueeze(-1).float()
        h_entities_init = h_entities_init * ent_mask_f

        # ── Edge Dropout (training only) ────────────────────────────────
        inc = incidence_matrix
        if self.training and self.edge_dropout_rate > 0:
            drop_mask = torch.bernoulli(
                torch.full_like(inc, 1.0 - self.edge_dropout_rate))
            inc = inc * drop_mask

        # ── Precompute entity queries ───────────────────────────────────
        ent_q_scene = self.scene_query(h_entities_init)  # [B, N, D]
        ent_q_arc   = self.arc_query(h_entities_init)    # [B, N, D]

        # ── Stream 4 running accumulator ────────────────────────────────
        # h_accum tracks each entity's biographical context so far.
        # As scenes accumulate, h_accum[n] reflects all events entity n
        # has been through — used to compute relationship alignment.
        h_accum       = h_entities_init.clone()            # [B, N, D]
        accum_count   = torch.ones(B, N, 1,                # [B, N, 1]
                                   device=scene_reps.device,
                                   dtype=scene_reps.dtype)

        # ── Phase 1: Per-scene hyperedges and 4-stream messages ─────────
        hyperedge_list   = []
        msg_per_scene    = []
        last_stream_attn = None

        for s in range(S):
            roles    = inc[:, :, s] * entity_mask.float()     # [B, N]
            in_scene = (roles > 0).unsqueeze(-1).float()      # [B, N, 1]

            # Event-typed Latent Hyperedge
            entity_weight_sum = roles.sum(dim=1, keepdim=True).clamp(min=1.0)
            weighted_ents = torch.bmm(
                roles.unsqueeze(1), h_entities_init).squeeze(1)
            mean_ent = weighted_ents / entity_weight_sum
            e_s = self.edge_norm(
                self.edge_entity_proj(mean_ent)
                + self.edge_scene_proj(scene_reps[:, s, :])
                + self.hyperedge_type_embed(edge_type_ids[:, s])  # event type
            )
            hyperedge_list.append(e_s)

            # ── Stream 1: Entity-aware scene messages (HGNN Phase 2) ─────
            # Use e_s (entity-contextualized hyperedge) not raw scene_rep.
            # This closes the HGNN loop: entity→hyperedge (e_s formation)
            # then hyperedge→entity (stream 1 extraction from e_s).
            scene_k = self.scene_key(e_s)    # e_s already integrates entity context
            scene_v = self.scene_value(e_s)
            attn_score = (ent_q_scene * scene_k.unsqueeze(1)).sum(
                -1, keepdim=True) / math.sqrt(D)
            msg_scene = self.scene_msg_norm(
                torch.sigmoid(attn_score) * scene_v.unsqueeze(1))

            # ── Stream 2: Narrative arc ──────────────────────────────────
            msg_arc = torch.zeros(B, N, D, device=scene_reps.device,
                                  dtype=scene_reps.dtype)
            if s > 0:
                past_he  = torch.stack(hyperedge_list[:-1], dim=1).detach()
                past_k   = self.arc_key(past_he)
                past_v   = self.arc_value(past_he)
                past_roles = inc[:, :, :s]
                shared   = past_roles * roles.unsqueeze(-1)
                has_shared = (shared > 0).float()
                arc_scores = torch.einsum(
                    'bnd,bsd->bns', ent_q_arc, past_k) / math.sqrt(D)
                decay = torch.linspace(-1.0, 0.0, s, device=scene_reps.device)
                arc_scores = arc_scores + decay.unsqueeze(0).unsqueeze(0)
                arc_scores = arc_scores.masked_fill(has_shared == 0, -1e9)
                arc_attn   = F.softmax(arc_scores, dim=-1)
                arc_attn   = arc_attn * (has_shared.sum(-1, keepdim=True) > 0).float()
                msg_arc    = self.arc_norm(torch.bmm(arc_attn, past_v))

            # ── Stream 3: Static social co-occurrence ───────────────────
            co_occur = torch.bmm(roles.unsqueeze(-1), roles.unsqueeze(1))
            co_occur.diagonal(dim1=1, dim2=2).zero_()
            co_norm  = co_occur / co_occur.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            msg_interact = self.interact_norm(
                self.interact_proj(torch.bmm(co_norm, h_entities_init))
            )

            # ── Stream 4: Cross-scene relationship evolution ─────────────
            # Uses h_accum (running entity biographies) instead of static init.
            # Relationship weight = co-occurrence × history alignment:
            #   aligned entities (similar biographies) get higher weight
            #   divergent entities (different roles) get lower weight
            h_acc_norm = F.normalize(h_accum.detach(), p=2, dim=-1, eps=1e-8)
            rel_align  = torch.bmm(h_acc_norm,
                                   h_acc_norm.transpose(1, 2))  # [B, N, N] cosine
            rel_weight = co_occur * (1.0 + self.rel_align_scale * rel_align)
            rel_weight = rel_weight / rel_weight.sum(
                dim=-1, keepdim=True).clamp(min=1e-8)
            msg_rel    = self.rel_norm(self.rel_proj(
                torch.bmm(rel_weight, h_accum)))

            # ── Stream Fusion (4 streams) ────────────────────────────────
            if self.use_adaptive_streams:
                stream_attn = F.softmax(self.stream_gate(e_s), dim=-1)
                msg = (stream_attn[:, 0].view(B, 1, 1) * msg_scene +
                       stream_attn[:, 1].view(B, 1, 1) * msg_arc +
                       stream_attn[:, 2].view(B, 1, 1) * msg_interact +
                       stream_attn[:, 3].view(B, 1, 1) * msg_rel)
                last_stream_attn = stream_attn.detach()
            else:
                sw  = F.softmax(self.stream_weights, dim=0)
                msg = (sw[0] * msg_scene + sw[1] * msg_arc
                       + sw[2] * msg_interact + sw[3] * msg_rel)

            role_emb = self.role_proj(roles.unsqueeze(-1))
            msg = msg + role_emb * in_scene
            msg = msg * in_scene
            msg_per_scene.append(msg)

            # Update running biography: incremental mean of messages for
            # entities present in this scene
            h_accum     = (h_accum * accum_count + msg) / (accum_count + in_scene)
            accum_count = accum_count + in_scene

        # ── Phase 2: Per-entity Mamba trajectories ──────────────────────
        entity_msgs = torch.stack(msg_per_scene, dim=2)  # [B, N, S, D]
        entity_msgs = entity_msgs + h_entities_init.unsqueeze(2)
        entity_trajs = entity_msgs.view(B * N, S, D)

        # Emotion bias: [B, N, S] → [B*N, S, 1]
        emotion_bias = None
        if emotion_matrix is not None:
            emotion_bias = emotion_matrix.view(B * N, S, 1).abs()

        if return_dt:
            entity_out, dt_values = self.entity_mamba(
                entity_trajs, emotion_bias=emotion_bias, return_dt=True)
            dt_values = dt_values.view(B, N, S, -1)
        else:
            entity_out = self.entity_mamba(
                entity_trajs, emotion_bias=emotion_bias)
            dt_values = None

        entity_out = entity_out.view(B, N, S, D)

        presence     = (inc > 0).float()
        presence_sum_f = presence.sum(dim=-1, keepdim=True).clamp(min=1.0)
        h_entities   = (entity_out * presence.unsqueeze(-1)).sum(dim=2) / presence_sum_f
        h_entities   = self.entity_update_norm(h_entities)

        # ── HGNN Phase 2: global hyperedge→entity aggregation ───────────
        # Compute H_hyperedges once and reuse for both entity aggregation
        # and the return value.
        # Entity aggregation: h_entity += D_v^{-1} * B * hyperedges
        # This is the standard normalized HGNN second phase.
        H_hyperedges_raw = torch.stack(hyperedge_list, dim=1)   # [B, S, D]
        # Use original incidence_matrix (not edge-dropped inc) for global
        # readout — dropout is for training regularization in message passing,
        # not for the final aggregation step. Normalize by true entity degree.
        true_degree = incidence_matrix.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, N, 1]
        he_agg = torch.bmm(incidence_matrix, H_hyperedges_raw) / true_degree     # [B, N, D]
        h_entities = h_entities + self.he_ent_norm(self.he_ent_proj(he_agg))

        h_entities   = h_entities * ent_mask_f
        h_entities   = h_entities.clamp(-50.0, 50.0)

        # ── Logging ─────────────────────────────────────────────────────
        if wandb.run is not None and random.random() < 0.05:
            if self.use_adaptive_streams and last_stream_attn is not None:
                wandb.log({
                    "stream_weight/scene":        last_stream_attn[:, 0].mean().item(),
                    "stream_weight/arc":          last_stream_attn[:, 1].mean().item(),
                    "stream_weight/interaction":  last_stream_attn[:, 2].mean().item(),
                    "stream_weight/relationship": last_stream_attn[:, 3].mean().item(),
                })
            elif not self.use_adaptive_streams:
                w = F.softmax(self.stream_weights, dim=0).detach().cpu().numpy()
                wandb.log({
                    "stream_weight/scene": w[0], "stream_weight/arc": w[1],
                    "stream_weight/interaction": w[2], "stream_weight/relationship": w[3],
                })

        H_hyperedges = self.residual_norm(H_hyperedges_raw + scene_reps)
        H_hyperedges = self.edge_out(H_hyperedges)

        if return_dt:
            return H_hyperedges, h_entities, dt_values
        return H_hyperedges, h_entities


# =============================================================================
# 3. Losses
# =============================================================================

class NarrativeCoherenceLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, scene_reps, incidence_matrix):
        """scene_reps: [B, S, D], incidence_matrix: [B, N, S]"""
        scene_reps = F.normalize(scene_reps.float(), p=2, dim=-1, eps=1e-8)
        total_loss = scene_reps.new_tensor(0.0)
        n_valid    = 0
        for b in range(scene_reps.size(0)):
            B_mat  = incidence_matrix[b].float()
            inter  = torch.mm(B_mat.T, B_mat)
            sums   = B_mat.sum(dim=0, keepdim=True)
            union  = sums + sums.T - inter
            jac    = inter / union.clamp(min=1e-8)
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


class RelationalEventConsistencyLoss(nn.Module):
    def __init__(self, alpha=0.1, tokenizer=None, temperature=0.1,
                 entity_penalty=3.0, label_smoothing=0.1, coherence_weight=0.05):
        super().__init__()
        self.alpha            = alpha
        self.tokenizer        = tokenizer
        self.temperature      = temperature
        self.entity_penalty   = entity_penalty
        self.label_smoothing  = label_smoothing
        self.coherence_weight = coherence_weight
        self.coherence_fn     = NarrativeCoherenceLoss(temperature=0.1)

    def _smooth_nll(self, log_probs, targets, weight_mask):
        V = log_probs.size(-1)
        log_probs = log_probs.clamp(min=-100.0)
        smooth = torch.zeros_like(log_probs).fill_(self.label_smoothing / V)
        smooth.scatter_(1, targets.unsqueeze(1).clamp(min=0), 1.0 - self.label_smoothing)
        nll    = -(smooth * log_probs).sum(dim=-1)
        valid  = (targets != 1).float()
        return (nll * weight_mask * valid).sum() / (weight_mask * valid).sum().clamp(min=1.0)

    def _triplet_embeds(self, trips, head_weight, device, max_t=5):
        if not trips or not trips[0] or self.tokenizer is None:
            return None
        texts = []
        for scene_trips in trips:
            sampled = (random.sample(scene_trips, max_t)
                       if len(scene_trips) > max_t else scene_trips)
            texts.append(" ".join(sampled).replace("_", " "))
        enc  = self.tokenizer(texts, padding=True, truncation=True,
                              max_length=32, return_tensors="pt").to(device)
        embs = F.embedding(enc["input_ids"], head_weight)
        mask_e = enc["attention_mask"].unsqueeze(-1).expand(embs.size()).float()
        return (embs * mask_e).sum(1) / mask_e.sum(1).clamp(min=1e-4)

    def forward(self, log_probs, targets, triplets, hidden_states=None,
                head_weight=None, incidence_matrix=None):
        device = log_probs.device
        weight_mask = torch.ones_like(targets, dtype=torch.float)
        if triplets and triplets[0] and self.tokenizer is not None:
            flat_ents = []
            for st in triplets:
                for t in st:
                    p = t.split("_")
                    if len(p) >= 1:
                        flat_ents.append(p[0].replace("NOT ", "").strip())
                    if len(p) >= 3:
                        flat_ents.append(p[2].strip())
            if flat_ents:
                enc = self.tokenizer(flat_ents, add_special_tokens=False)["input_ids"]
                ids = set(tid for sub in enc for tid in sub)
                if ids:
                    ent_t = torch.tensor(list(ids), device=device)
                    weight_mask[torch.isin(targets, ent_t)] = self.entity_penalty

        lm_loss = self._smooth_nll(log_probs, targets, weight_mask)
        # Store for external logging without affecting the graph
        self._last_lm_loss   = lm_loss.item()
        self._last_cont_loss = 0.0
        self._last_coh_loss  = 0.0

        if (not triplets or not triplets[0] or hidden_states is None
                or head_weight is None):
            return lm_loss

        # hidden_states: [B, S, D] (scene-level, no token dim)
        B, S, D = hidden_states.shape
        h_flat   = hidden_states.view(B * S, D)
        valid_idx = [i for i, t in enumerate(triplets) if len(t) > 0]
        if not valid_idx:
            return lm_loss

        valid_trips = [triplets[i] for i in valid_idx]
        neg_trips = [
            [f"{p.split('_')[2]}_{p.split('_')[1]}_{p.split('_')[0]}"
             if len(p.split("_")) == 3 else p for p in bt]
            for bt in valid_trips
        ]
        pos_embs = self._triplet_embeds(valid_trips, head_weight, device)
        neg_embs = self._triplet_embeds(neg_trips, head_weight, device)
        if pos_embs is None or neg_embs is None:
            return lm_loss

        h_v  = h_flat[valid_idx]
        n    = min(h_v.size(0), pos_embs.size(0))
        hn   = F.normalize(h_v[:n].float(), p=2, dim=1, eps=1e-4)
        pn   = F.normalize(pos_embs[:n].float(), p=2, dim=1, eps=1e-4)
        nn_  = F.normalize(neg_embs[:n].float(), p=2, dim=1, eps=1e-4)
        logits_c = torch.stack([
            (hn * pn).sum(-1) / self.temperature,
            (hn * nn_).sum(-1) / self.temperature,
        ], dim=1)
        cont_loss = F.cross_entropy(
            logits_c, torch.zeros(n, dtype=torch.long, device=device))
        total = (1 - self.alpha) * lm_loss + self.alpha * cont_loss
        self._last_cont_loss = cont_loss.item()

        if incidence_matrix is not None and self.coherence_weight > 0:
            coh = self.coherence_fn(hidden_states, incidence_matrix)
            total = total + self.coherence_weight * coh
            self._last_coh_loss = coh.item()
        return total


# =============================================================================
# 4. Fusion modules
# =============================================================================

class GraphToTextFusion(nn.Module):
    """
    Anchored cross-attention fusion: H_text stays in LED encoder space.

    Problem with gated sum (gate * H_text + (1-gate) * H_hyperedges):
      The LED decoder's cross-attention is calibrated to the LED encoder's
      embedding space. Mixing in H_hyperedges (random-init space) pushes
      fused representations away from that space, degrading generation.

    Solution: keep H_text as the anchor. H_text queries attend to H_hyperedges
    (keys/values), pulling structural knowledge in as a residual update.
    The output stays geometrically close to H_text (LED space) while gaining
    graph-derived structure. A per-dimension sigmoid gate controls how much
    each feature dimension absorbs from the graph.
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        # Project H_hyperedges into a query-compatible space before fusion
        # (normalizes its statistics independently of LED's space)
        self.graph_align_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, dropout=dropout)
        # Normalize cross-attention output before gating — prevents scale
        # explosion from untrained hypergraph representations early in training
        self.graph_info_norm = nn.LayerNorm(d_model)
        # Per-dimension gate conditioned on H_text:
        #   "what does this text representation need from the graph?"
        # CRITICAL: bias init to -6 so gate ≈ 0.0025 at step 0.
        # The gate opens organically as the hypergraph learns to produce
        # useful signal — prevents corrupting LED space with early-training noise.
        self.gate_proj = nn.Linear(d_model, d_model)
        nn.init.constant_(self.gate_proj.bias, -6.0)
        self.norm_in   = nn.LayerNorm(d_model)
        self.norm_out  = nn.LayerNorm(d_model)
        # Small FFN to integrate after fusion.
        # Zero-init the output linear so the residual starts as identity —
        # prevents random FF noise from corrupting H_text at init.
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )
        nn.init.zeros_(self.ff[3].weight)
        nn.init.zeros_(self.ff[3].bias)

    def forward(self, H_text, H_hyperedges):
        """
        H_text:       [B, S, D] — anchor, LED encoder space
        H_hyperedges: [B, S, D] — hypergraph scene representations

        Returns fused [B, S, D] still in LED encoder space.
        """
        # Align H_hyperedges statistics before using as keys/values
        H_graph_aligned = self.graph_align_proj(H_hyperedges)

        # H_text queries pull graph knowledge in as residual
        norm_text = self.norm_in(H_text)
        graph_info, _ = self.cross_attn(
            query=norm_text,
            key=H_graph_aligned,
            value=H_graph_aligned,
        )
        # Normalize cross-attention output before gating
        graph_info = self.graph_info_norm(graph_info)
        # Per-dimension gate: conditioned on H_text, bias-initialized to -6
        # so the gate ≈ 0 at init and opens only when graph signal is useful
        gate = torch.sigmoid(self.gate_proj(norm_text))    # [B, S, D]
        fused = H_text + gate * graph_info                 # anchored residual
        fused = fused + self.ff(self.norm_out(fused))
        return fused


class EntitySceneCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between fused scenes and entity nodes.
    Scenes absorb final entity states; entities absorb narrative context.
    Applied after graph-text fusion, so scenes already carry graph knowledge.
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm_scene_1 = nn.LayerNorm(d_model)
        self.norm_node_1  = nn.LayerNorm(d_model)
        self.attn_scene_to_node = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, dropout=dropout)
        self.attn_node_to_scene = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, dropout=dropout)
        self.norm_scene_2 = nn.LayerNorm(d_model)
        self.norm_node_2  = nn.LayerNorm(d_model)
        self.ff_scene = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )
        nn.init.zeros_(self.ff_scene[3].weight)
        nn.init.zeros_(self.ff_scene[3].bias)
        self.ff_node = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )
        nn.init.zeros_(self.ff_node[3].weight)
        nn.init.zeros_(self.ff_node[3].bias)
        # LayerScale: scene residual from entity cross-attention starts at 0.
        # Entity states are random-init — without this, random H_nodes corrupt
        # the LED scene representations and inflate loss at the start of training.
        self.scene_ls = nn.Parameter(torch.zeros(d_model))

    def forward(self, H_scenes, H_nodes, entity_mask):
        norm_s = self.norm_scene_1(H_scenes)
        norm_n = self.norm_node_1(H_nodes)
        key_pad_nodes = ~entity_mask
        all_masked = key_pad_nodes.all(dim=-1)
        if all_masked.any():
            key_pad_nodes[all_masked, 0] = False

        s_star, _ = self.attn_scene_to_node(
            query=norm_s, key=norm_n, value=norm_n,
            key_padding_mask=key_pad_nodes,
        )
        # Scale starts at 0 — no entity influence at init, grows as entity
        # states become meaningful during training.
        H_scenes = H_scenes + self.scene_ls * s_star
        H_scenes = H_scenes + self.ff_scene(self.norm_scene_2(H_scenes))

        n_star, _ = self.attn_node_to_scene(
            query=norm_n, key=norm_s, value=norm_s,
        )
        H_nodes = H_nodes + n_star
        H_nodes = H_nodes + self.ff_node(self.norm_node_2(H_nodes))
        return H_scenes, H_nodes


# =============================================================================
# 5. Main model — LEDMambaHypergraphSummariser
# =============================================================================

class LEDMambaHypergraphSummariser(nn.Module):
    def __init__(self, vocab_size, d_model=1024, max_entities=MAX_ENTITIES,
                 max_scenes=64, tokenizer=None, use_adaptive_streams=True,
                 use_entity_names=True, edge_dropout=0.1, mamba_layers=2):
        super().__init__()
        self.d_model      = d_model
        self.max_entities = max_entities
        self.max_scenes   = max_scenes
        self.tokenizer    = tokenizer
        self.use_entity_names = use_entity_names

        # ── Tower 1: LED encoder + decoder ─────────────────────────────
        print(f"Loading LED backbone: {LED_MODEL}...")
        led = LEDForConditionalGeneration.from_pretrained(LED_MODEL)
        # Enable gradient checkpointing before extracting sub-modules.
        # Recomputes activations during backward instead of storing all 16
        # decoder layers — saves ~4-6GB at the cost of ~20% slower backward.
        led.gradient_checkpointing_enable()
        self.led_encoder = led.led.encoder
        self.led_decoder = led.led.decoder
        self.head        = led.lm_head

        # LED encoder: freeze everything except global attention layers
        # (global attention is what makes LED handle 16K — keep it trainable)
        for name, p in self.led_encoder.named_parameters():
            p.requires_grad = False
        # LED decoder + head: fully trainable (matched encoder, no mismatch)
        for p in self.led_decoder.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

        # Scene boundary pooling: map LED token output → scene-level reps.
        # Identity init: preserves pretrained LED representations at step 0.
        # A random-init linear here scrambles the encoder output and causes
        # the pretrained decoder to make confidently wrong predictions (loss >> 10).
        self.scene_pool_proj = nn.Linear(d_model, d_model)
        nn.init.eye_(self.scene_pool_proj.weight)
        nn.init.zeros_(self.scene_pool_proj.bias)
        self.scene_pool_norm = nn.LayerNorm(d_model)

        # ── Tower 2: Hypergraph with Mamba ─────────────────────────────
        self.hypergraph_tower = DynamicHypergraphTower(
            d_model=d_model, max_entities=max_entities,
            num_entity_types=NUM_ENTITY_TYPES,
            num_hyperedge_types=NUM_HYPEREDGE_TYPES,
            use_adaptive_streams=use_adaptive_streams,
            use_entity_names=use_entity_names,
            edge_dropout=edge_dropout,
            mamba_layers=mamba_layers,
        )

        # ── Fusion ─────────────────────────────────────────────────────
        # Stage 1: cross-attention pull of graph knowledge into LED space
        self.graph_text_fusion = GraphToTextFusion(d_model)
        # Stage 2: bidirectional scene↔entity cross-attention
        # entity_mem_scale: gates entity nodes in aligned_memory sent to decoder.
        # Init=0 so random entity states don't corrupt decoder at step 0.
        self.entity_mem_scale = nn.Parameter(torch.zeros(1))
        self.entity_scene_attn = EntitySceneCrossAttention(d_model)
        self.memory_norm  = nn.LayerNorm(d_model)
        self.use_checkpointing = True

    def enable_gradient_checkpointing(self):
        self.use_checkpointing = True

    @torch.no_grad()
    def _encode_entity_names(self, entity_names, device):
        """Encode entity names using LED word embeddings (768→d_model via proj)."""
        B = len(entity_names)
        N = self.max_entities
        embs = torch.zeros(B, N, self.d_model, device=device)
        for b in range(B):
            names = entity_names[b]
            valid = [(i, n) for i, n in enumerate(names) if n and i < N]
            if not valid:
                continue
            indices, texts = zip(*valid)
            enc = self.tokenizer(list(texts), padding=True, truncation=True,
                                 max_length=8, return_tensors="pt").to(device)
            word_embs = self.led_encoder.embed_tokens(enc["input_ids"])
            mask = enc["attention_mask"].unsqueeze(-1).float()
            avg = (word_embs * mask).sum(1) / mask.sum(1).clamp(min=1)
            for k, idx in enumerate(indices):
                embs[b, idx] = avg[k]
        return embs

    def _pool_scenes(self, encoder_output, scene_boundaries, attention_mask):
        """
        Pool LED encoder output into scene-level representations.

        Args:
            encoder_output: [B, T, D] from LED encoder
            scene_boundaries: [B, S, 2] — (start_tok, end_tok) for each scene
            attention_mask: [B, T]
        Returns:
            H_text: [B, S, D] scene-level representations
        """
        B, T, D = encoder_output.shape
        S = scene_boundaries.size(1)

        H_scenes = torch.zeros(B, S, D, device=encoder_output.device,
                                dtype=encoder_output.dtype)
        for b in range(B):
            for s in range(S):
                start, end = scene_boundaries[b, s].tolist()
                if start >= end or start >= T:
                    continue
                end = min(end, T)
                mask_slice = attention_mask[b, start:end].unsqueeze(-1).float()
                if mask_slice.sum() == 0:
                    continue
                pooled = (encoder_output[b, start:end] * mask_slice).sum(0) / mask_slice.sum().clamp(min=1)
                H_scenes[b, s] = pooled

        H_scenes = self.scene_pool_norm(self.scene_pool_proj(H_scenes))
        return H_scenes

    def forward(self, input_ids, attention_mask, scene_boundaries,
                global_attention_mask, incidence_matrix, edge_type_ids,
                entity_type_ids, entity_mask, target_ids=None,
                entity_names=None, emotion_matrix=None, return_dt=False):
        """
        Args:
            input_ids:      [B, T] — full screenplay tokenized with <scene> separators
            attention_mask: [B, T]
            scene_boundaries: [B, S, 2] — start/end token positions per scene
            global_attention_mask: [B, T] — 1 for <scene> tokens and first token
            incidence_matrix: [B, N, S]
            target_ids:     [B, T_tgt] — summary token IDs
        """
        B = input_ids.size(0)
        S = scene_boundaries.size(1)
        N = self.max_entities
        pad_id = self.tokenizer.pad_token_id or 1

        # ── Tower 1: LED encoder ───────────────────────────────────────
        encoder_frozen = not any(p.requires_grad for p in self.led_encoder.parameters())
        ctx = torch.no_grad() if encoder_frozen else torch.enable_grad()
        with ctx:
            encoder_out = self.led_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
        led_hidden = encoder_out.last_hidden_state  # [B, T, D]

        # Pool into scene-level representations
        H_text = self._pool_scenes(led_hidden, scene_boundaries, attention_mask)

        # ── Encode entity names ────────────────────────────────────────
        entity_name_embs = None
        if entity_names is not None and self.use_entity_names and self.tokenizer is not None:
            entity_name_embs = self._encode_entity_names(entity_names, input_ids.device)

        # ── Tower 2: Hypergraph with Mamba ─────────────────────────────
        if return_dt:
            H_hyperedges, H_nodes, dt_values = self.hypergraph_tower(
                H_text, incidence_matrix, edge_type_ids,
                entity_type_ids, entity_mask,
                entity_name_embs=entity_name_embs,
                emotion_matrix=emotion_matrix, return_dt=True,
            )
        else:
            H_hyperedges, H_nodes = self.hypergraph_tower(
                H_text, incidence_matrix, edge_type_ids,
                entity_type_ids, entity_mask,
                entity_name_embs=entity_name_embs,
                emotion_matrix=emotion_matrix,
            )
            dt_values = None

        # ── Fusion ─────────────────────────────────────────────────────
        # Stage 1: pull graph structure into LED space (H_text is anchor)
        # H_text queries attend to H_hyperedges → residual update → stays in LED space
        if self.use_checkpointing and target_ids is not None:
            fused_scenes = checkpoint(
                self.graph_text_fusion, H_text, H_hyperedges,
                use_reentrant=False,
            )
        else:
            fused_scenes = self.graph_text_fusion(H_text, H_hyperedges)

        # Stage 2: bidirectional scene ↔ entity cross-attention
        if self.use_checkpointing and target_ids is not None:
            fused_scenes, H_nodes = checkpoint(
                self.entity_scene_attn, fused_scenes, H_nodes, entity_mask,
                use_reentrant=False,
            )
        else:
            fused_scenes, H_nodes = self.entity_scene_attn(
                fused_scenes, H_nodes, entity_mask)

        # Log gate magnitude for W&B
        self._last_gate_mean = 0.0  # gate is internal to GraphToTextFusion

        # entity_mem_scale starts at 0: entity nodes contribute nothing to the
        # decoder's memory at init (they're random). Scale grows as training progresses.
        valid_nodes    = entity_mask.unsqueeze(-1).float() * H_nodes * torch.tanh(self.entity_mem_scale)
        aligned_memory = torch.cat([fused_scenes, valid_nodes], dim=1)
        aligned_memory = self.memory_norm(aligned_memory)
        aligned_memory = aligned_memory.clamp(-50.0, 50.0)

        # ── NaN trace ──────────────────────────────────────────────────
        def _nan_tag(name, t):
            if not torch.isfinite(t).all():
                print(f"    [NaN-trace] {name}: "
                      f"NaN={t.isnan().sum().item()} Inf={t.isinf().sum().item()} "
                      f"shape={list(t.shape)}")
                return True
            return False
        _nan_tag("H_text", H_text)
        _nan_tag("H_hyperedges", H_hyperedges)
        _nan_tag("H_nodes", H_nodes)
        _nan_tag("aligned_memory", aligned_memory)

        if target_ids is not None:
            labels          = target_ids[:, 1:].contiguous()
            dec_start       = torch.full((B, 1), 2, dtype=torch.long,
                                         device=input_ids.device)
            shifted_targets = torch.cat([dec_start, target_ids[:, 1:-1]],
                                         dim=1).contiguous()

            # Encoder attention mask for aligned_memory [S + N]
            mem_pad = torch.zeros(B, S + N, dtype=torch.bool,
                                  device=input_ids.device)
            # Scene slots: mark as padded if no tokens in that scene
            for b in range(B):
                for s in range(S):
                    start, end = scene_boundaries[b, s].tolist()
                    if start >= end:
                        mem_pad[b, s] = True
            mem_pad[:, S:] = ~entity_mask
            all_masked = mem_pad.all(dim=1)
            mem_pad[all_masked, 0] = False
            enc_attn_mask = (~mem_pad).long()
            tgt_attn_mask = (shifted_targets != pad_id).long()

            decoder_out = self.led_decoder(
                input_ids=shifted_targets,
                attention_mask=tgt_attn_mask,
                encoder_hidden_states=aligned_memory,
                encoder_attention_mask=enc_attn_mask,
            )
            dec_hidden   = decoder_out.last_hidden_state
            vocab_logits = self.head(dec_hidden)
            final_log_probs = F.log_softmax(vocab_logits.float(), dim=-1)

            return final_log_probs, H_text, labels, dec_hidden, H_hyperedges
        else:
            # Inference: return aligned_memory for generation
            return aligned_memory, H_text, scene_boundaries, dt_values


# Backwards compatibility alias
DualTowerHypergraphSummariser = LEDMambaHypergraphSummariser

# =============================================================================
# 6. Logging helpers
# =============================================================================

def log_hyperedge_attention(model, H_hyperedges, incidence_matrix, movie_name=""):
    if not wandb.run:
        return
    with torch.no_grad():
        reps = H_hyperedges[0].float().cpu()
        reps = F.normalize(reps, p=2, dim=-1)
        sim  = torch.matmul(reps, reps.T).numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim, cmap="viridis", vmin=0, vmax=1, annot=False)
    plt.title(f"Hyperedge similarity — {movie_name}")
    plt.xlabel("Scene index")
    plt.ylabel("Scene index")
    wandb.log({"hyperedge_sim": wandb.Image(plt)})
    plt.close()


def log_entity_state_norms(H_nodes, entity_mask, step):
    if not wandb.run:
        return
    with torch.no_grad():
        valid = H_nodes[0][entity_mask[0]]
        if valid.size(0) > 0:
            mean_norm = valid.float().norm(dim=-1).mean().item()
            wandb.log({"entity_state_norm": mean_norm, "step": step})


def log_entity_dt_heatmap(dt_values, entity_names, entity_mask, movie_name=""):
    """
    Visualize Mamba dt values as a heatmap (entity × scene).
    High dt = large state change = narrative turning point for that character.
    """
    if not wandb.run or dt_values is None:
        return
    with torch.no_grad():
        # dt_values: [B, N, S, d_inner] — take mean over d_inner
        dt_mean = dt_values[0].float().mean(dim=-1).cpu()  # [N, S]
        mask = entity_mask[0].cpu()
        valid_dt = dt_mean[mask]  # [n_valid, S]
        valid_names = [n for n, m in zip(entity_names[0], mask) if m]

        if valid_dt.size(0) == 0:
            return

        # Limit to top 20 entities for readability
        top_k = min(20, valid_dt.size(0))
        fig, ax = plt.subplots(figsize=(14, max(4, top_k * 0.4)))
        sns.heatmap(
            valid_dt[:top_k].numpy(), ax=ax,
            yticklabels=valid_names[:top_k],
            cmap="YlOrRd", cbar_kws={"label": "dt (state change magnitude)"},
        )
        ax.set_xlabel("Scene")
        ax.set_ylabel("Entity")
        ax.set_title(f"Entity State Change Magnitude — {movie_name}")
        wandb.log({"entity_dt_heatmap": wandb.Image(fig)})
        plt.close(fig)

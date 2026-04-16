"""
sum.py  —  Dual-Tower Dynamic Hypergraph Summariser (v3)
====================================================
Architecture
------------
Tower 1 — Text stream (Mamba SSM)
    Frozen RoBERTa → Linear(768→d_model) → MambaBlock → RAFT modality fusion
    Output: H_text [B, S, L, d_model]

Tower 2 — Dynamic 3-Stream Hypergraph (DHEG)
    Each scene is a Latent Hyperedge. Entity states update via a Sequential GRU 
    that aggregates three distinct learnable message streams:
        Stream 1 (Scene): Immediate event context.
        Stream 2 (Arc): Temporal context from past scenes sharing the same entities.
        Stream 3 (Interaction): Social context from co-occurring entities in the scene.
    
Fusion
    Deep Cross-Attention over [fused_scenes, entity_nodes].
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

import os as _os
_BART_LARGE_LOCAL = "/tmp/uday/bart-large"
BART_MODEL = (_BART_LARGE_LOCAL if _os.path.isdir(_BART_LARGE_LOCAL)
              else "facebook/bart-large")

# Note: Hardcoded types remain for backwards compatibility with train.py signatures, 
# but are effectively bypassed by the new Latent Edge architecture.
HYPEREDGE_TYPE_MAP  = {"CONFLICT": 0, "ALLIANCE": 1, "DECEPTION": 2, "DIALOGUE": 3, "NEUTRAL": 4}
NUM_HYPEREDGE_TYPES = 5
ENTITY_TYPE_MAP  = {"PERSON": 0, "ORG": 1, "GPE": 2, "FACILITY": 3, "OTHER": 4}
NUM_ENTITY_TYPES = 5
MAX_ENTITIES = 100   

# =============================================================================
# 0. Utilities & Tower 1 (Unchanged to maintain text pipeline integrity)
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

class MambaLayer(nn.Module):
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
        # Keep delta/B/C in native dtype (BF16 under autocast) — only h stays FP32
        # for SSM recurrence numerical stability over long sequences.
        delta    = F.softplus(self.dt_proj(delta)).float()   # [B, T, d_inner] FP32
        B_mat    = B_mat.float()                             # [B, T, d_state]
        C        = C.float()                                 # [B, T, d_state]
        x_act_f32 = x_act.float()                           # [B, T, d_inner]
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=torch.float32)
        # Pre-allocate output tensor — eliminates seq_len individual heap allocations
        # (the outs-list pattern created 256×[B,d_inner] tensors, fragmenting GPU memory).
        y = torch.zeros(batch, seq_len, self.d_inner, device=x.device, dtype=torch.float32)
        for t in range(seq_len):
            h = h * torch.exp(-delta[:, t].unsqueeze(-1)) + \
                (B_mat[:, t].unsqueeze(1) * x_act_f32[:, t].unsqueeze(-1))
            # Clamp SSM state to prevent unbounded accumulation → NaN after many training steps.
            # SSM equilibrium value is ~input_magnitude/delta; 100 is safe for all realistic inputs.
            h = h.clamp(-100.0, 100.0)
            y[:, t, :] = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
        return self.out_proj(y.to(x.dtype) * F.silu(z)) + residual

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([MambaLayer(d_model, d_state, d_conv) for _ in range(num_layers)])
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x

class RaftConsensusAttentionV2(nn.Module):
    def __init__(self, d_model=1024, num_heads=4):
        super().__init__()
        self.action_proj = nn.Linear(d_model, d_model)
        self.dial_proj   = nn.Linear(d_model, d_model)
        self.ent_proj    = nn.Linear(d_model, d_model)
        self.head_proj   = nn.Linear(d_model, d_model)
        self.cross_attn  = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.1)
        self.consensus_gate = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, features, action_mask, dial_mask, ent_mask, head_mask):
        masks = [action_mask, dial_mask, ent_mask, head_mask]
        projs = [self.action_proj, self.dial_proj, self.ent_proj, self.head_proj]
        modalities = [projs[i](features) * masks[i].unsqueeze(-1).float() for i in range(4)]
        B, L, D    = features.shape
        modal_flat = torch.stack(modalities, dim=2).view(B * L, 4, D)
        attended, _= self.cross_attn(modal_flat, modal_flat, modal_flat)
        attended   = attended.view(B, L, 4, D)
        consensus  = self.consensus_gate(attended.reshape(B, L, 4 * D))
        return self.norm(features + consensus)

# =============================================================================
# 3. Tower 2 — 3-Stream Dynamic Hypergraph Encoder (DHEG)
# =============================================================================

class DynamicHypergraphTower(nn.Module):
    """
    EMNLP v4: 3-Stream Dynamic Hypergraph Encoder with
    - Semantic entity name initialization (not just type embeddings)
    - Scene-conditioned adaptive stream gating (not global weights)
    - Role-modulated GRU messages
    - Entity state LayerNorm for long-sequence stability
    - Hyperedge incidence dropout for graph regularization
    """
    def __init__(self, d_model=1024, max_entities=100, num_edge_types=5, num_entity_types=5,
                 use_adaptive_streams=True, use_entity_names=True, edge_dropout=0.1):
        super().__init__()
        self.d_model      = d_model
        self.max_entities = max_entities
        self.use_adaptive_streams = use_adaptive_streams
        self.use_entity_names     = use_entity_names
        self.edge_dropout_rate    = edge_dropout

        # Entity Initialization: type embedding + optional name embedding
        self.entity_type_embed = nn.Embedding(num_entity_types, d_model)
        if use_entity_names:
            self.entity_name_proj = nn.Sequential(
                nn.Linear(768, d_model, bias=False), nn.LayerNorm(d_model),
            )

        # Stage 1: Latent Edge Generation
        self.node_to_edge  = nn.Linear(d_model, d_model)
        self.text_to_edge  = nn.Linear(d_model, d_model)
        self.edge_norm     = nn.LayerNorm(d_model)

        # Stage 2: 3-Stream Fusion Parameters
        if use_adaptive_streams:
            # Scene-conditioned gating: the latent edge representation determines
            # how much each stream contributes — action scenes can up-weight the
            # scene stream while character reunion scenes up-weight the arc stream.
            self.stream_gate = nn.Sequential(
                nn.Linear(d_model, d_model // 4), nn.GELU(),
                nn.Linear(d_model // 4, 3),
            )
        else:
            # Ablation: global learnable weights (content-agnostic)
            self.stream_weights = nn.Parameter(torch.zeros(3))

        self.msg_scene_proj    = nn.Linear(d_model, d_model)
        self.msg_arc_proj      = nn.Linear(d_model, d_model)
        self.msg_interact_proj = nn.Linear(d_model, d_model)

        # Role modulation: speakers update differently from background mentions
        self.role_proj = nn.Linear(1, d_model, bias=False)

        self.entity_gru = nn.GRUCell(d_model, d_model)
        # LayerNorm after GRU prevents entity state drift over 64 sequential updates
        self.entity_update_norm = nn.LayerNorm(d_model)

        self.edge_out = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model),
        )
        self.residual_norm = nn.LayerNorm(d_model)

    def forward(self, scene_reps, incidence_matrix, edge_type_ids, entity_type_ids, entity_mask,
                entity_name_embs=None):
        B, S, D = scene_reps.shape
        N = self.max_entities

        # ── Entity Initialization ───────────────────────────────────────
        h_entities = self.entity_type_embed(entity_type_ids)              # [B, N, D]
        if entity_name_embs is not None and self.use_entity_names:
            h_entities = h_entities + self.entity_name_proj(entity_name_embs)
        ent_mask_f = entity_mask.unsqueeze(-1).float()                    # [B, N, 1]
        h_entities = h_entities * ent_mask_f

        # ── Edge Dropout (training only) ────────────────────────────────
        if self.training and self.edge_dropout_rate > 0:
            drop_mask = torch.bernoulli(
                torch.full_like(incidence_matrix, 1.0 - self.edge_dropout_rate)
            )
            incidence_matrix = incidence_matrix * drop_mask

        hyperedge_list = []
        last_stream_attn = None   # for logging

        for s in range(S):
            roles = incidence_matrix[:, :, s] * entity_mask.float()       # [B, N]

            # ── Stage 1: Latent Edge Generation ─────────────────────────
            entity_weight_sum = roles.sum(dim=1, keepdim=True).clamp(min=1.0)
            mean_ent = torch.bmm(roles.unsqueeze(1), h_entities).squeeze(1) / entity_weight_sum

            e_s = self.edge_norm(
                self.node_to_edge(mean_ent) + self.text_to_edge(scene_reps[:, s, :])
            )
            hyperedge_list.append(e_s)

            # ── Stage 2: 3-Stream Message Passing ───────────────────────
            in_scene_mask = (roles > 0).unsqueeze(-1).float()             # [B, N, 1]

            # Stream 1 (Scene): immediate event context
            msg_scene = self.msg_scene_proj(e_s).unsqueeze(1).expand(-1, N, -1)

            # Stream 2 (Arc): temporal context from past shared scenes
            msg_arc = torch.zeros_like(msg_scene)
            if s > 0:
                past_hyperedges = torch.stack(hyperedge_list[:-1], dim=1).detach()
                past_roles = incidence_matrix[:, :, :s]
                arc_attention = past_roles * roles.unsqueeze(-1)
                decay = torch.exp(torch.linspace(-1, 0, s, device=scene_reps.device))
                arc_attention = arc_attention * decay.unsqueeze(0).unsqueeze(0)
                arc_attention = arc_attention / arc_attention.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                msg_arc = self.msg_arc_proj(torch.bmm(arc_attention, past_hyperedges))

            # Stream 3 (Interaction): social context from co-occurring entities
            co_occur = torch.bmm(roles.unsqueeze(-1), roles.unsqueeze(1))
            co_occur.diagonal(dim1=1, dim2=2).zero_()
            co_occur = co_occur / co_occur.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            msg_interact = self.msg_interact_proj(torch.bmm(co_occur, h_entities))

            # ── Adaptive Stream Fusion ──────────────────────────────────
            if self.use_adaptive_streams:
                # Scene-conditioned: the latent edge decides stream mix per scene
                stream_attn = F.softmax(self.stream_gate(e_s), dim=-1)    # [B, 3]
                msg_total = (stream_attn[:, 0].view(B, 1, 1) * msg_scene +
                             stream_attn[:, 1].view(B, 1, 1) * msg_arc +
                             stream_attn[:, 2].view(B, 1, 1) * msg_interact)
                last_stream_attn = stream_attn.detach()
            else:
                stream_attn = F.softmax(self.stream_weights, dim=0)
                msg_total = (stream_attn[0] * msg_scene +
                             stream_attn[1] * msg_arc +
                             stream_attn[2] * msg_interact)

            # ── Role Modulation ─────────────────────────────────────────
            # Speakers (1.0) receive stronger messages than background mentions (0.3)
            role_emb = self.role_proj(roles.unsqueeze(-1))                # [B, N, D]
            msg_total = msg_total + role_emb * in_scene_mask

            # ── GRU Update + Normalization ──────────────────────────────
            h_flat   = h_entities.reshape(B * N, D)
            msg_flat = msg_total.reshape(B * N, D)
            h_new    = self.entity_update_norm(
                self.entity_gru(msg_flat, h_flat).reshape(B, N, D)
            )
            # Clamp entity states: 64 sequential GRU updates can accumulate
            # magnitude despite LayerNorm, especially in BF16.
            h_new    = h_new.clamp(-50.0, 50.0)
            h_entities = h_entities * (1 - in_scene_mask) + h_new * in_scene_mask

        # ── Logging ─────────────────────────────────────────────────────
        if wandb.run is not None and random.random() < 0.05:
            if self.use_adaptive_streams and last_stream_attn is not None:
                wandb.log({
                    "stream_weight/scene": last_stream_attn[:, 0].mean().item(),
                    "stream_weight/arc": last_stream_attn[:, 1].mean().item(),
                    "stream_weight/interaction": last_stream_attn[:, 2].mean().item(),
                })
            elif not self.use_adaptive_streams:
                w = F.softmax(self.stream_weights, dim=0).detach().cpu().numpy()
                wandb.log({
                    "stream_weight/scene": w[0],
                    "stream_weight/arc": w[1],
                    "stream_weight/interaction": w[2],
                })

        H_hyperedges = torch.stack(hyperedge_list, dim=1)                 # [B, S, D]
        H_hyperedges = self.residual_norm(H_hyperedges + scene_reps)
        H_hyperedges = self.edge_out(H_hyperedges)

        return H_hyperedges, h_entities

# =============================================================================
# 4. Hierarchical Pointer Head & Losses (Unchanged logic, preserving stability)
# =============================================================================

class HierarchicalPointerHeadV2(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.vocab_size    = vocab_size
        self.query_proj    = nn.Linear(d_model, d_model)
        self.key_proj      = nn.Linear(d_model, d_model)
        self.p_gen_linear  = nn.Linear(d_model * 2, 1)
        nn.init.zeros_(self.p_gen_linear.weight)
        nn.init.zeros_(self.p_gen_linear.bias)
        # bias=0 → p_gen starts at sigmoid(0)=0.5 (50% vocab / 50% pointer)
        # Previously bias=3.0 with *(x-0.5)*10 scaling → sigmoid(25)≈1.0,
        # gradient≈0, pointer never learned anything.

    def forward(self, decoder_states, scene_memory, triplets, tokenizer, embedding_weight, device):
        B, T, D = decoder_states.shape
        _, S, _ = scene_memory.shape

        q          = self.query_proj(decoder_states)
        k          = self.key_proj(scene_memory)
        scores     = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)
        scene_attn = F.softmax(scores, dim=-1)                          

        context   = torch.matmul(scene_attn, scene_memory)             
        p_gen     = torch.sigmoid(self.p_gen_linear(torch.cat([decoder_states, context], dim=-1)))

        # Use the same dtype as decoder_states (BF16 under autocast) to avoid a [B,S,V] FP32 allocation
        scene_to_token = torch.zeros(B, S, self.vocab_size, device=device,
                                     dtype=decoder_states.dtype)

        if triplets and tokenizer is not None:
            dec_last = decoder_states[:, -1, :]
            for b in range(B):
                ns = min(S, len(triplets[b]) if isinstance(triplets[b], list) else 1)
                for s in range(ns):
                    scene_trips = (triplets[b][s] if isinstance(triplets[b], list) and s < len(triplets[b]) else [])
                    if not scene_trips: continue
                    entities = set()
                    for trip in scene_trips:
                        parts = trip.split("_")
                        if len(parts) >= 1: entities.add(parts[0].replace("NOT ", "").strip())
                        if len(parts) >= 3: entities.add(parts[2].strip())
                    entities.discard("")
                    if not entities: continue
                    
                    encoded  = tokenizer(list(entities), add_special_tokens=False)["input_ids"]
                    tok_ids  = list({tid for sub in encoded for tid in sub if tid < self.vocab_size})
                    if not tok_ids: continue
                    
                    ent_t    = torch.tensor(tok_ids, device=device)
                    ent_embs = F.embedding(ent_t, embedding_weight)
                    salience = F.softmax(
                        torch.matmul(dec_last[b].unsqueeze(0).float(), ent_embs.float().T).squeeze(0), dim=-1
                    )
                    scene_to_token[b, s, tok_ids] = salience.to(scene_to_token.dtype)

        pointer_probs = torch.matmul(scene_attn, scene_to_token)       
        return p_gen, pointer_probs

class NarrativeCoherenceLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, scene_hidden, incidence_matrix):
        if scene_hidden.dim() == 4:
            scene_reps = scene_hidden.mean(dim=2)   
        else:
            scene_reps = scene_hidden
        # scene_reps = F.normalize(scene_reps.float(), p=2, dim=-1)
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
            if pos_mask.sum() == 0: continue

            reps    = scene_reps[b]
            sim     = torch.matmul(reps, reps.T) / self.temperature
            sim.fill_diagonal_(-1e4)
            log_p   = F.log_softmax(sim, dim=-1)
            pos_n   = pos_mask / pos_mask.sum(dim=1, keepdim=True).clamp(min=1)
            loss_s  = -(pos_n * log_p).sum(dim=-1)

            has_pos = pos_mask.sum(dim=1) > 0
            if not has_pos.any(): continue
            total_loss = total_loss + loss_s[has_pos].mean()
            n_valid   += 1

        return total_loss / max(n_valid, 1)

class RelationalEventConsistencyLoss(nn.Module):
    def __init__(self, alpha=0.1, tokenizer=None, temperature=0.1, entity_penalty=3.0, label_smoothing=0.1, coherence_weight=0.05):
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
        # Clamp log_probs: avoids positive_smooth_weight * (-inf) = -inf in sum,
        # which propagates to loss=+inf → skipped batch → stalled training.
        # -100 is well below any meaningful log probability and safe for gradients.
        log_probs = log_probs.clamp(min=-100.0)
        smooth = torch.zeros_like(log_probs).fill_(self.label_smoothing / V)
        smooth.scatter_(1, targets.unsqueeze(1).clamp(min=0), 1.0 - self.label_smoothing)
        nll    = -(smooth * log_probs).sum(dim=-1)
        valid  = (targets != 1).float()
        return (nll * weight_mask * valid).sum() / (weight_mask * valid).sum().clamp(min=1.0)

    def _triplet_embeds(self, trips, head_weight, device, max_t=5):
        if not trips or not trips[0] or self.tokenizer is None: return None
        texts = []
        for scene_trips in trips:
            sampled = (random.sample(scene_trips, max_t) if len(scene_trips) > max_t else scene_trips)
            texts.append(" ".join(sampled).replace("_", " "))
        enc     = self.tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt").to(device)
        embs    = F.embedding(enc["input_ids"], head_weight)
        mask_e  = enc["attention_mask"].unsqueeze(-1).expand(embs.size()).float()
        return (embs * mask_e).sum(1) / mask_e.sum(1).clamp(min=1e-4)

    def forward(self, log_probs, targets, triplets, hidden_states=None, head_weight=None, incidence_matrix=None):
        device = log_probs.device
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

        if (not triplets or not triplets[0] or hidden_states is None or head_weight is None):
            return lm_loss

        B, S, L, D  = hidden_states.shape
        h_flat       = hidden_states.view(B * S, L, D).mean(dim=1)
        valid_idx    = [i for i, t in enumerate(triplets) if len(t) > 0]
        if not valid_idx: return lm_loss

        valid_trips  = [triplets[i] for i in valid_idx]
        neg_trips    = [[f"{p.split('_')[2]}_{p.split('_')[1]}_{p.split('_')[0]}" if len(p.split("_")) == 3 else p for p in bt] for bt in valid_trips]
        pos_embs     = self._triplet_embeds(valid_trips, head_weight, device)
        neg_embs     = self._triplet_embeds(neg_trips,   head_weight, device)
        if pos_embs is None or neg_embs is None: return lm_loss

        h_v     = h_flat[valid_idx]
        n       = min(h_v.size(0), pos_embs.size(0))
        hn      = F.normalize(h_v[:n].float(),   p=2, dim=1, eps=1e-4)
        pn      = F.normalize(pos_embs[:n].float(), p=2, dim=1, eps=1e-4)
        nn_     = F.normalize(neg_embs[:n].float(), p=2, dim=1, eps=1e-4)
        logits_c = torch.stack([(hn * pn).sum(-1) / self.temperature, (hn * nn_).sum(-1) / self.temperature], dim=1)
        cont_loss = F.cross_entropy(logits_c, torch.zeros(n, dtype=torch.long, device=device))
        total     = (1 - self.alpha) * lm_loss + self.alpha * cont_loss

        if incidence_matrix is not None and self.coherence_weight > 0:
            coh = self.coherence_fn(hidden_states, incidence_matrix)
            total = total + self.coherence_weight * coh
        return total

# =============================================================================
# 6b. Post-Scan Cross-Attention Adapter (Fix 3)
# =============================================================================

class PostScanCrossAttentionAdapter(nn.Module):
    """
    Fix 3: Post-scan cross-attention adapter.
    Runs parallel cross-attention between the completed scene sequence and the
    final entity node states so each side is mutually aware before decoder fusion.

    Cross 1: scenes query nodes  → each scene embedding absorbs final character states.
    Cross 2: nodes query scenes  → each entity node absorbs full narrative context.

    Both operations use Pre-LayerNorm + residual + FFN (standard transformer block).
    Frozen in Stage 1; trained at 5× LR from Stage 2 (random init must catch up).
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
        self.ff_node = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )

    def forward(self, H_scenes, H_nodes, entity_mask):
        norm_s = self.norm_scene_1(H_scenes)
        norm_n = self.norm_node_1(H_nodes)

        # True = ignore (PyTorch MHA convention)
        key_pad_nodes = ~entity_mask

        # --- NEW SAFETY VALVE ---
        # If any movie in the batch has 100% True (all masked), unmask the 0th index 
        # to prevent Softmax(-inf) generating NaNs.
        all_masked = key_pad_nodes.all(dim=-1)
        if all_masked.any():
            key_pad_nodes[all_masked, 0] = False
        # ------------------------

        # Cross 1: scenes attend to nodes
        s_star, _ = self.attn_scene_to_node(
            query=norm_s, key=norm_n, value=norm_n,
            key_padding_mask=key_pad_nodes,
        )
        H_scenes = H_scenes + s_star
        H_scenes = H_scenes + self.ff_scene(self.norm_scene_2(H_scenes))

        # Cross 2: nodes attend to scenes
        n_star, _ = self.attn_node_to_scene(
            query=norm_n, key=norm_s, value=norm_s,
        )
        H_nodes = H_nodes + n_star
        H_nodes = H_nodes + self.ff_node(self.norm_node_2(H_nodes))

        return H_scenes, H_nodes


# =============================================================================
# 7. Main model — DualTowerHypergraphSummariser
# =============================================================================

class DualTowerHypergraphSummariser(nn.Module):
    def __init__(self, vocab_size, d_model=1024, num_layers=4, max_entities=MAX_ENTITIES, tokenizer=None,
                 use_adaptive_streams=True, use_entity_names=True, edge_dropout=0.1):
        super().__init__()
        self.d_model      = d_model
        self.max_entities = max_entities
        self.tokenizer    = tokenizer
        self.use_entity_names = use_entity_names

        print(f"Loading BART backbone: {BART_MODEL}...")
        bart = BartForConditionalGeneration.from_pretrained(BART_MODEL)

        print("Loading frozen RoBERTa scene encoder...")
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        for p in self.roberta.parameters(): p.requires_grad = False

        self.scene_proj = (nn.Linear(768, d_model, bias=False) if d_model != 768 else nn.Identity())
        self.mamba_tower = MambaBlock(d_model, d_state=64, d_conv=4, num_layers=num_layers)

        # Fix 4: Scene-level positional embeddings (indexed by scene position in film)
        self.scene_pos_embed = nn.Embedding(512, d_model)
        nn.init.normal_(self.scene_pos_embed.weight, mean=0.0, std=0.02)
        self.scene_pos_drop  = nn.Dropout(0.1)

        self.raft = RaftConsensusAttentionV2(d_model=d_model, num_heads=4)

        # ── Tower 2 ───────────────────────────────────────────────────────
        self.hypergraph_tower = DynamicHypergraphTower(
            d_model=d_model, max_entities=max_entities,
            num_edge_types=NUM_HYPEREDGE_TYPES, num_entity_types=NUM_ENTITY_TYPES,
            use_adaptive_streams=use_adaptive_streams,
            use_entity_names=use_entity_names,
            edge_dropout=edge_dropout,
        )

        self.fusion_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.fusion_norm = nn.LayerNorm(d_model)

        self.bart_decoder = bart.model.decoder
        self.head         = bart.lm_head

        for p in self.bart_decoder.parameters(): p.requires_grad = False
        for p in self.head.parameters(): p.requires_grad = False

        # Reinitialize encoder_attn from scratch: pretrained BART cross-attention
        # expects BART encoder output which our custom encoder never produces.
        # Xavier init means cross-attn starts near-zero (no-op via residual connection),
        # so the decoder initially works as an LM and gradually learns to condition on
        # our encoder. No NaN because there's no pretrained-weight/novel-input mismatch.
        self._reinit_encoder_attn()

        # Fix 3: Post-scan cross-attention adapter (before pointer head)
        self.post_scan_adapter = PostScanCrossAttentionAdapter(d_model)
        self.pointer_head = HierarchicalPointerHeadV2(d_model, vocab_size)
        self.memory_norm  = nn.LayerNorm(d_model)
        self.use_checkpointing = True

    def _reinit_encoder_attn(self):
        """Reinitialize all encoder_attn + encoder_attn_layer_norm from scratch."""
        for name, p in self.bart_decoder.named_parameters():
            if "encoder_attn" not in name:
                continue
            p.requires_grad = True
            if "layer_norm" in name:
                if "weight" in name:
                    nn.init.ones_(p)
                else:
                    nn.init.zeros_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def enable_gradient_checkpointing(self):
        self.use_checkpointing = True

    def _text_tower(self, input_ids, action_mask, dial_mask, ent_mask, head_mask):
        B, S, L = input_ids.shape
        pad_id   = 1
        # Mamba chunk size: 16 scenes × [2048 d_inner × 64 d_state] FP32 = 32 MB/chunk — safe
        chunk_sz = 16

        # ── RoBERTa: ONE batched call over ALL scenes ────────────────────
        # Previously 8 serial calls of 8 scenes each. One batched call of 64 scenes
        # gives 3-4x GPU speedup due to parallelism.
        flat_ids = input_ids.view(B * S, L)
        flat_att = (flat_ids != pad_id).long()
        with torch.no_grad():
            all_rob = self.roberta(flat_ids, attention_mask=flat_att).last_hidden_state
        all_emb = self.scene_proj(all_rob)                         # [B*S, L, D]

        # Scene positional embeddings
        scene_ids = torch.arange(S, device=input_ids.device)
        pos_embs  = self.scene_pos_embed(scene_ids)                # [S, D]
        all_emb   = all_emb.view(B, S, L, self.d_model)
        all_emb   = all_emb + pos_embs.unsqueeze(0).unsqueeze(2)   # [B, S, L, D]
        all_emb   = self.scene_pos_drop(all_emb)

        # ── Mamba: chunked to control FP32 state memory ─────────────────
        local_feats = []
        for i in range(0, S, chunk_sz):
            j     = min(i + chunk_sz, S)
            c_emb = all_emb[:, i:j].contiguous().view(B * (j - i), L, self.d_model)
            if self.use_checkpointing:
                c_out = checkpoint(self.mamba_tower, c_emb, use_reentrant=False)
            else:
                c_out = self.mamba_tower(c_emb)
            local_feats.append(c_out)

        local_flat = torch.cat(local_feats, dim=0)
        # Mamba SSM can produce extreme activations when LoRA modifies dt_proj
        # (exp(-softplus(dt)*A) overflow). Clamp before downstream amplification.
        local_flat = local_flat.clamp(-100.0, 100.0)
        H_text_4d  = local_flat.view(B, S, L, self.d_model)
        flat  = H_text_4d.view(B * S, L, self.d_model)
        fused = self.raft(
            flat, action_mask.view(B * S, L), dial_mask.view(B * S, L),
            ent_mask.view(B * S, L), head_mask.view(B * S, L),
        )
        H_text_4d = fused.view(B, S, L, self.d_model)
        H_text    = H_text_4d.mean(dim=2)
        return H_text, H_text_4d

    @torch.no_grad()
    def _encode_entity_names(self, entity_names, device):
        """Encode entity names into 768d vectors using frozen RoBERTa word embeddings.
        Fast path: just embedding lookup + average, no full RoBERTa forward pass."""
        B = len(entity_names)
        N = self.max_entities
        embs = torch.zeros(B, N, 768, device=device)
        for b in range(B):
            names = entity_names[b]
            valid = [(i, n) for i, n in enumerate(names) if n and i < N]
            if not valid:
                continue
            indices, texts = zip(*valid)
            enc = self.tokenizer(list(texts), padding=True, truncation=True,
                                 max_length=8, return_tensors="pt").to(device)
            word_embs = self.roberta.embeddings.word_embeddings(enc["input_ids"])
            mask = enc["attention_mask"].unsqueeze(-1).float()
            avg = (word_embs * mask).sum(1) / mask.sum(1).clamp(min=1)
            for k, idx in enumerate(indices):
                embs[b, idx] = avg[k]
        return embs

    def forward(self, input_ids, action_mask, dial_mask, ent_mask, head_mask,
                incidence_matrix, edge_type_ids, entity_type_ids, entity_mask,
                target_ids=None, triplets=None, entity_names=None):
        B, S, L = input_ids.shape
        N       = self.max_entities
        pad_id  = 1

        H_text, H_text_4d = self._text_tower(input_ids, action_mask, dial_mask, ent_mask, head_mask)

        # ── Encode entity names (frozen RoBERTa word embeddings) ───────
        entity_name_embs = None
        if entity_names is not None and self.use_entity_names and self.tokenizer is not None:
            entity_name_embs = self._encode_entity_names(entity_names, input_ids.device)

        # ── Tower 2 ────────────────────────────────────────────────────────
        H_hyperedges, H_nodes = self.hypergraph_tower(
            H_text, incidence_matrix, edge_type_ids, entity_type_ids, entity_mask,
            entity_name_embs=entity_name_embs,
        )

        gate         = self.fusion_gate(torch.cat([H_text, H_hyperedges], dim=-1))
        # Store gate stats for diagnostics (readable via model._last_gate_mean)
        self._last_gate_mean = gate.detach().mean().item()
        fused_scenes = self.fusion_norm(gate * H_text + (1 - gate) * H_hyperedges)

        # Fix 3: bidirectional cross-attention adapter before decoder fusion.
        # Checkpoint it in Stage 2 so its activations are recomputed rather than stored.
        if self.use_checkpointing and target_ids is not None:
            fused_scenes, H_nodes = checkpoint(
                self.post_scan_adapter, fused_scenes, H_nodes, entity_mask,
                use_reentrant=False,
            )
        else:
            fused_scenes, H_nodes = self.post_scan_adapter(fused_scenes, H_nodes, entity_mask)

        valid_nodes    = entity_mask.unsqueeze(-1).float() * H_nodes
        aligned_memory = torch.cat([fused_scenes, valid_nodes], dim=1)
        aligned_memory = self.memory_norm(aligned_memory)
        aligned_memory = aligned_memory.clamp(-50.0, 50.0)

        # ── NaN trace: log first occurrence per forward pass ────────────
        def _nan_tag(name, t):
            if not torch.isfinite(t).all():
                print(f"    [NaN-trace] {name}: "
                      f"NaN={t.isnan().sum().item()} Inf={t.isinf().sum().item()} "
                      f"shape={list(t.shape)} range=[{t[t.isfinite()].min().item():.2f}, "
                      f"{t[t.isfinite()].max().item():.2f}]"
                      if t.isfinite().any() else
                      f"    [NaN-trace] {name}: ALL non-finite, shape={list(t.shape)}")
                return True
            return False
        _nan_tag("H_text", H_text)
        _nan_tag("H_hyperedges", H_hyperedges)
        _nan_tag("H_nodes", H_nodes)
        _nan_tag("fused_scenes", fused_scenes)
        _nan_tag("aligned_memory", aligned_memory)
        # ────────────────────────────────────────────────────────────────

        if target_ids is not None:
            single_target   = target_ids[:, 0, :]
            labels          = single_target[:, 1:].contiguous()
            dec_start       = torch.full((B, 1), 2, dtype=torch.long, device=input_ids.device)
            shifted_targets = torch.cat([dec_start, single_target[:, 1:-1]], dim=1).contiguous()

            mem_pad = torch.zeros(B, S + N, dtype=torch.bool, device=input_ids.device)
            mem_pad[:, :S] = (input_ids[:, :, 0] == pad_id)
            mem_pad[:, S:] = ~entity_mask
            all_masked = mem_pad.all(dim=1)
            mem_pad[all_masked, 0] = False
            enc_attn_mask = (~mem_pad).long()
            tgt_attn_mask = (shifted_targets != pad_id).long()

            decoder_out   = self.bart_decoder(
                input_ids=shifted_targets, attention_mask=tgt_attn_mask,
                encoder_hidden_states=aligned_memory, encoder_attention_mask=enc_attn_mask,
            )
            dec_hidden    = decoder_out.last_hidden_state
            vocab_logits  = self.head(dec_hidden)
            _nan_tag("dec_hidden", dec_hidden)
            _nan_tag("vocab_logits", vocab_logits)

            if triplets is not None and self.tokenizer is not None:
                # Compute softmax in float32 to avoid BF16 producing exact 0s → log(0) = NaN
                vocab_probs    = F.softmax(vocab_logits.float(), dim=-1).to(vocab_logits.dtype)
                p_gen, ptr_pr  = self.pointer_head(
                    dec_hidden, fused_scenes, triplets, self.tokenizer, self.head.weight, input_ids.device
                )
                _nan_tag("p_gen", p_gen)
                _nan_tag("ptr_pr", ptr_pr)
                final_probs = p_gen * vocab_probs + (1 - p_gen) * ptr_pr  # BF16
                final_probs = final_probs.float().clamp(min=0.0)
                final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                final_log_probs = torch.log(final_probs + 1e-8)
                _nan_tag("final_log_probs", final_log_probs)
            else:
                final_log_probs = F.log_softmax(vocab_logits.float(), dim=-1)

            return final_log_probs, H_text_4d, labels, dec_hidden, H_hyperedges
        else:
            return aligned_memory, H_text_4d

GraMFormerV2 = DualTowerHypergraphSummariser

# =============================================================================
# 8. Logging helpers
# =============================================================================

def log_hyperedge_attention(model, H_hyperedges, incidence_matrix, movie_name=""):
    if not wandb.run: return
    with torch.no_grad():
        reps  = H_hyperedges[0].float().cpu()               
        reps  = F.normalize(reps, p=2, dim=-1)
        sim   = torch.matmul(reps, reps.T).numpy()           
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim, cmap="viridis", vmin=0, vmax=1, annot=False)
    plt.title(f"Hyperedge similarity — {movie_name}")
    plt.xlabel("Scene index")
    plt.ylabel("Scene index")
    wandb.log({"hyperedge_sim": wandb.Image(plt)})
    plt.close()

def log_entity_state_norms(H_nodes, entity_mask, step):
    if not wandb.run: return
    with torch.no_grad():
        valid = H_nodes[0][entity_mask[0]]                   
        if valid.size(0) > 0:
            mean_norm = valid.float().norm(dim=-1).mean().item()
            wandb.log({"entity_state_norm": mean_norm, "step": step})
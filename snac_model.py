"""
snac_model.py — SNaC: Streaming Narrative-state Compression for narrative summarization.

Architecture (LED whole-document backbone)
-------------------------------------------
1. Document encoder   : LED encodes the WHOLE screenplay as one 16K sequence with
                        global attention on scene separators -> context-aware,
                        in-distribution encoder states H [T, D].
2. Story-state        : M slots carried scene->scene. A subset are ENTITY-BOUND;
                        writes are ADDRESSABLE — a slot only updates when its
                        entity is present, so distant facts survive (short
                        effective path length, not O(#scenes)). Each scene's read
                        is the LED encoder states sliced to that scene's span.
3. Memory / decoder   :
   - 'full'       : decoder cross-attends the FULL LED encoder states (strong,
                    in-distribution LED summariser; the headline-metric path).
   - 'two_source' : [ gated state slots (global, compressed)
                    ; retrieved top-k scene spans at full resolution (local) ].
                    The compression/interpretability novelty.
   - 'state_only' : slots only (diagnostic).
4. Probe head         : predicts per-scene entity presence from the slots ->
                        forces the state to be decodable (interpretability result).
5. Turning points     : ||S_t - S_{t-1}|| per scene is returned for analysis.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

from snac_data import SNaCConfig


class StoryState(nn.Module):
    """The streaming, entity-bound read/write state."""

    def __init__(self, cfg: SNaCConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.d = d_model
        M = cfg.n_slots
        # slot type embeddings (entity vs free) + free-slot content init
        self.slot_type = nn.Embedding(2, d_model)
        self.free_init = nn.Parameter(torch.randn(cfg.free_slots, d_model) * 0.02)
        self.name_proj = nn.Linear(d_model, d_model)

        self.read_attn = nn.MultiheadAttention(d_model, cfg.n_heads,
                                               batch_first=True, dropout=0.1)
        self.cand = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, d_model))
        self.gate = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        # entity-presence probe (shared across entity slots)
        self.probe = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                   nn.Linear(d_model // 2, 1))

        # --- in-distribution init: at step 0 the state must look like real encoder
        #     output so the pretrained decoder isn't thrown off. Entity slots start at
        #     the name embedding (name_proj = identity), type embeddings tiny, and the
        #     write gate starts near-closed so early scenes barely perturb the state.
        nn.init.eye_(self.name_proj.weight); nn.init.zeros_(self.name_proj.bias)
        nn.init.normal_(self.slot_type.weight, std=0.02)
        nn.init.constant_(self.gate.bias, -2.0)          # sigmoid(-2) ~ 0.12

        is_entity = torch.zeros(M, dtype=torch.long)
        is_entity[:cfg.max_entities] = 1
        self.register_buffer("is_entity", is_entity)

    def init_slots(self, name_embs: torch.Tensor) -> torch.Tensor:
        """name_embs: [E, D] pooled entity-name encodings -> initial [M, D] state."""
        E, F_ = self.cfg.max_entities, self.cfg.free_slots
        ent = self.name_proj(name_embs) + self.slot_type(self.is_entity.new_ones(E))
        free = self.free_init + self.slot_type(self.is_entity.new_zeros(F_))
        return torch.cat([ent, free], dim=0)                      # [M, D]

    def step(self, S_prev, scene_hidden, pres_vec):
        """One scene update. scene_hidden: [L, D] (all-valid LED states for the scene).
        Returns (S_new [M,D], delta [M], probe_logits [E])."""
        q = S_prev.unsqueeze(0)                                   # [1, M, D]
        kv = scene_hidden.unsqueeze(0)                            # [1, L, D]
        read, _ = self.read_attn(q, kv, kv)                       # no padding: all valid
        read = read.squeeze(0)                                    # [M, D]

        cand = self.norm(self.cand(torch.cat([S_prev, read], dim=-1)))   # [M, D]
        g = torch.sigmoid(self.gate(torch.cat([S_prev, read], dim=-1)))
        # ADDRESSABLE write: entity slots gate is scaled by presence; free slots = 1.
        eff = g * pres_vec.unsqueeze(-1)                          # [M, D]
        S_new = S_prev + eff * (cand - S_prev)                    # convex, bounded
        delta = (S_new - S_prev).norm(dim=-1)                     # [M]

        ent_slots = S_new[:self.cfg.max_entities]                 # [E, D]
        probe_logits = self.probe(ent_slots).squeeze(-1)          # [E]
        return S_new, delta, probe_logits

    def forward(self, hidden, boundaries, presence, init_state):
        """
        hidden    : [T, D] whole-document LED encoder states
        boundaries: list of [start, end) token spans, one per scene
        presence  : [S, E] entity role weights
        init_state: [M, D]
        Returns final_state [M,D], deltas [S,M], probe_logits [S,E], states [S,M,D]
        """
        M, E = self.cfg.n_slots, self.cfg.max_entities
        S = len(boundaries)
        pres_full = torch.ones(S, M, device=hidden.device)
        pres_full[:, :E] = presence                              # free slots always 1
        state = init_state
        deltas, probes, states = [], [], []
        for s, (a, b) in enumerate(boundaries):
            span = hidden[a:b] if b > a else hidden[a:a + 1]      # guard empty span
            state, delta, plog = self.step(state, span, pres_full[s])
            deltas.append(delta); probes.append(plog); states.append(state)
        return (state,
                torch.stack(deltas),      # [S, M]
                torch.stack(probes),      # [S, E]
                torch.stack(states))      # [S, M, D]


class SNaC(nn.Module):
    def __init__(self, cfg: SNaCConfig, lora: bool = False, train_encoder: bool = False):
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModelForSeq2SeqLM.from_pretrained(cfg.backbone)
        self.d = self.backbone.config.d_model
        # keep a direct handle on the encoder (survives LoRA in-place injection)
        self.encoder = self.backbone.get_encoder()

        if lora:
            from peft import LoraConfig, get_peft_model
            lc = LoraConfig(task_type="SEQ_2_SEQ_LM", r=16, lora_alpha=32,
                            lora_dropout=0.05,
                            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"])
            self.backbone = get_peft_model(self.backbone, lc)
            self.encoder_frozen = False
        elif train_encoder:
            self.encoder_frozen = False                  # full fine-tune (most reliable)
        else:
            for p in self.encoder.parameters():          # freeze encoder body
                p.requires_grad = False
            self.encoder_frozen = True

        self.state = StoryState(cfg, self.d)
        self.mem_norm = nn.LayerNorm(self.d)
        # Gate the state's contribution to the two_source memory, initialized to ZERO,
        # so at init the retrieved verbatim tokens dominate and an untrained state can
        # never disrupt the pretrained decoder; the state earns its way in if it helps.
        self.state_gate = nn.Parameter(torch.zeros(1))

    # --- helpers ------------------------------------------------------------
    def _enc_ctx(self):
        return torch.no_grad() if self.encoder_frozen else torch.enable_grad()

    def _encode_doc(self, batch) -> torch.Tensor:
        """Whole-document LED encoding -> [T, D] (padding appended at the tail)."""
        with self._enc_ctx():
            out = self.encoder(
                input_ids=batch["input_ids"].unsqueeze(0),
                attention_mask=batch["attn_mask"].unsqueeze(0),
                global_attention_mask=batch["global_mask"].unsqueeze(0))
        return out.last_hidden_state.squeeze(0)                   # [T, D]

    def _name_init(self, ename_ids, ename_mask) -> torch.Tensor:
        """Ground entity slots in the backbone input-embedding of their names. [E, D]."""
        emb = self.backbone.get_input_embeddings()
        with torch.no_grad():
            e = emb(ename_ids)                                    # [E, Ln, D]
        m = ename_mask.unsqueeze(-1).float()
        return (e * m).sum(1) / m.sum(1).clamp(min=1)            # [E, D]

    def _scene_pools(self, hidden, boundaries) -> torch.Tensor:
        pools = [hidden[a:b].mean(0) if b > a else hidden[a]
                 for (a, b) in boundaries]
        return torch.stack(pools)                                 # [S, D]

    def _build_memory(self, hidden, final_state, batch
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decoder cross-attention memory. Mode selects what goes in it."""
        boundaries = batch["boundaries"]
        n_tok = batch["n_tokens"]
        S = len(boundaries)
        mode = self.cfg.memory_mode
        dev = hidden.device

        if mode == "state_only":
            mem = self.mem_norm(final_state)
            msk = torch.ones(mem.size(0), device=dev)
            return mem.unsqueeze(0), msk.unsqueeze(0), torch.arange(0, device=dev)

        if mode == "full":
            # Full LED encoder states; mask out the tail padding LED appended internally.
            msk = torch.zeros(hidden.size(0), device=dev)
            msk[:n_tok] = 1.0
            return hidden.unsqueeze(0), msk.unsqueeze(0), torch.arange(S, device=dev)

        # --- two_source: [ gated state slots ; retrieved verbatim scene spans ] ------
        scene_pool = self._scene_pools(hidden, boundaries)        # [S, D]
        query = final_state.mean(0, keepdim=True)                 # [1, D]
        scores = (scene_pool @ query.t()).squeeze(-1)             # [S]
        k = min(self.cfg.k_retrieve, S)
        idx = torch.topk(scores, k).indices
        idx = idx.sort().values                                   # keep chronological order
        local = torch.cat([hidden[boundaries[i][0]:boundaries[i][1]]
                           if boundaries[i][1] > boundaries[i][0]
                           else hidden[boundaries[i][0]:boundaries[i][0] + 1]
                           for i in idx.tolist()], dim=0)         # [sum L_i, D] RAW
        # State slots normalized AND gated to ~0 at init -> starts from raw-token behavior.
        state = self.mem_norm(final_state) * torch.tanh(self.state_gate)   # [M, D]
        mem = torch.cat([state, local], dim=0)
        msk = torch.ones(mem.size(0), device=dev)
        return mem.unsqueeze(0), msk.unsqueeze(0), idx

    def run_state(self, batch):
        """Shared front half: encode the document + names, run the streaming state."""
        hidden = self._encode_doc(batch)
        name_embs = self._name_init(batch["ename_ids"], batch["ename_mask"])
        init = self.state.init_slots(name_embs)
        final, deltas, probes, states = self.state(
            hidden, batch["boundaries"], batch["presence"], init)
        return hidden, final, deltas, probes, states

    # --- training -----------------------------------------------------------
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        hidden, final, deltas, probes, _states = self.run_state(batch)
        mem, msk, ret_idx = self._build_memory(hidden, final, batch)

        labels = batch["labels"].unsqueeze(0)                     # [1, T]
        out = self.backbone(encoder_outputs=BaseModelOutput(last_hidden_state=mem),
                            attention_mask=msk, labels=labels)
        nll = out.loss

        # entity-presence probe loss (only over valid entities)
        ev = batch["entity_valid"].bool()                         # [E]
        if ev.any():
            tgt = (batch["presence"] > 0).float()[:, ev]          # [S, Ev]
            logit = probes[:, ev]                                 # [S, Ev]
            probe = F.binary_cross_entropy_with_logits(logit, tgt)
        else:
            probe = torch.zeros((), device=nll.device)

        loss = nll + self.cfg.lambda_probe * probe
        return {"loss": loss, "nll": nll, "probe": probe,
                "delta_mean": deltas.mean().detach(),
                "state_gate": torch.tanh(self.state_gate).detach().abs().mean(),
                "deltas": deltas.detach(), "ret_idx": ret_idx.detach()}

    # --- inference ----------------------------------------------------------
    @torch.no_grad()
    def generate(self, batch, perturb_slot: Optional[int] = None,
                 num_beams: int = 4, max_new_tokens: int = 400,
                 min_new_tokens: int = 60, **kw):
        hidden, final, deltas, _, _ = self.run_state(batch)
        if perturb_slot is not None:            # causal intervention: erase a slot
            final = final.clone(); final[perturb_slot] = 0.0
        mem, msk, ret_idx = self._build_memory(hidden, final, batch)
        ids = self.backbone.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=mem),
            attention_mask=msk, num_beams=num_beams,
            max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
            no_repeat_ngram_size=3, length_penalty=2.0, early_stopping=True, **kw)
        return ids, deltas, ret_idx


def move_batch(batch: Dict, device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out

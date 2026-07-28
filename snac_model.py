"""
snac_model.py — SNaC: Streaming Narrative-state Compression for narrative summarization.

Architecture (novelty)
-----------------------
1. Scene encoder      : reuse the backbone's pretrained encoder, run per scene.
2. Story-state        : M slots carried scene->scene. A subset are ENTITY-BOUND;
                        writes are ADDRESSABLE — a slot only updates when its
                        entity is present, so distant facts survive (short
                        effective path length, not O(#scenes)).
3. Two-source decoder : cross-attends to  [ final state slots (global, compressed)
                        ; retrieved top-k scenes at full token resolution (local) ].
                        Global carries the arc; local gives faithful surface form.
4. Probe head         : predicts per-scene entity presence from the slots ->
                        forces the state to be decodable (interpretability result).
5. Turning points     : ||S_t - S_{t-1}|| per scene is returned for analysis.

Everything but the (optionally LoRA'd / frozen) backbone is trained from scratch.
"""
from __future__ import annotations
from typing import Dict, Optional

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
        #     output so BART's pretrained decoder isn't thrown off (else it degenerates
        #     to "the the the"). Entity slots start EXACTLY at the encoder-pooled name
        #     embedding (name_proj = identity), type embeddings tiny, and the write gate
        #     starts near-closed so early scenes barely perturb the state.
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

    def step(self, S_prev, scene_tok, scene_mask, pres_vec):
        """One scene update. Returns (S_new [M,D], delta [M], probe_logits [E])."""
        q = S_prev.unsqueeze(0)                                   # [1, M, D]
        kv = scene_tok.unsqueeze(0)                               # [1, L, D]
        kpm = ~scene_mask.bool().unsqueeze(0)                     # True = pad
        read, _ = self.read_attn(q, kv, kv, key_padding_mask=kpm)
        read = read.squeeze(0)                                    # [M, D]

        cand = self.norm(self.cand(torch.cat([S_prev, read], dim=-1)))   # [M, D]
        g = torch.sigmoid(self.gate(torch.cat([S_prev, read], dim=-1)))
        # ADDRESSABLE write: entity slots gate is scaled by presence; free slots = 1.
        # Gated residual (GRU-like): with gate near 0 at init, S_new ~= S_prev, so the
        # state stays at its in-distribution name-embedding init.
        eff = g * pres_vec.unsqueeze(-1)                          # [M, D]
        S_new = S_prev + eff * (cand - S_prev)                    # convex, bounded
        delta = (S_new - S_prev).norm(dim=-1)                     # [M]

        ent_slots = S_new[:self.cfg.max_entities]                 # [E, D]
        probe_logits = self.probe(ent_slots).squeeze(-1)          # [E]
        return S_new, delta, probe_logits

    def forward(self, scene_tok, scene_mask, presence, init_state):
        """
        scene_tok : [S, L, D] encoder states per scene
        scene_mask: [S, L]
        presence  : [S, E] entity role weights
        init_state: [M, D]
        Returns final_state [M,D], deltas [S,M], probe_logits [S,E], states [S,M,D]
        """
        S = scene_tok.size(0)
        M, E = self.cfg.n_slots, self.cfg.max_entities
        pres_full = torch.ones(S, M, device=scene_tok.device)
        pres_full[:, :E] = presence                              # free slots always 1
        state = init_state
        deltas, probes, states = [], [], []
        for s in range(S):
            state, delta, plog = self.step(state, scene_tok[s], scene_mask[s], pres_full[s])
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
        # Gate the state's contribution to the decoder memory, initialized to ZERO.
        # At init the model behaves exactly like --memory_mode full (raw tokens only,
        # which provably learns); the state earns its way in only if it helps, so an
        # untrained state can never disrupt the pretrained decoder.
        self.state_gate = nn.Parameter(torch.zeros(1))

    # --- helpers ------------------------------------------------------------
    def _enc_ctx(self):
        return torch.no_grad() if self.encoder_frozen else torch.enable_grad()

    def _encode_scenes(self, scene_ids, scene_mask):
        """[S, L] ids -> [S, L, D] states (encoder may be frozen -> no grad)."""
        with self._enc_ctx():
            out = self.encoder(input_ids=scene_ids, attention_mask=scene_mask)
        return out.last_hidden_state

    def _encode_names(self, ename_ids, ename_mask):
        with self._enc_ctx():
            out = self.encoder(input_ids=ename_ids, attention_mask=ename_mask)
        h = out.last_hidden_state                                 # [E, Ln, D]
        m = ename_mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1)             # [E, D]

    def _build_memory(self, final_state, scene_tok, scene_mask):
        """Decoder cross-attention memory. Mode selects what goes in it."""
        S = scene_tok.size(0)
        mode = self.cfg.memory_mode

        # --- diagnostic: raw scene tokens only, NO state / retrieval / norm.
        #     Exactly the prototype's working recipe (per-scene-encoded). If this
        #     can't overfit, the memory->decoder->loss plumbing has a bug.
        if mode == "full":
            mem = scene_tok.reshape(-1, self.d)                    # [S*L, D] raw
            msk = scene_mask.reshape(-1).float()                   # [S*L]
            return mem.unsqueeze(0), msk.unsqueeze(0), torch.arange(S, device=mem.device)

        if mode == "state_only":
            mem = self.mem_norm(final_state)
            msk = torch.ones(final_state.size(0), device=mem.device)
            return mem.unsqueeze(0), msk.unsqueeze(0), torch.arange(0, device=mem.device)

        # --- two_source (default): [state slots ; retrieved verbatim scene tokens]
        scene_pool = (scene_tok * scene_mask.unsqueeze(-1)).sum(1) / \
                     scene_mask.sum(1, keepdim=True).clamp(min=1)   # [S, D]
        query = final_state.mean(0, keepdim=True)                  # [1, D]
        scores = (scene_pool @ query.t()).squeeze(-1)              # [S]
        k = min(self.cfg.k_retrieve, S)
        idx = torch.topk(scores, k).indices
        local_tok = scene_tok[idx].reshape(-1, self.d)             # [k*L, D] RAW
        local_msk = scene_mask[idx].reshape(-1)                    # [k*L]

        # Keep the retrieved tokens RAW (the distribution the decoder expects, proven
        # by --memory_mode full). The state slots are normalized AND gated to ~0 at
        # init, so training starts from the working raw-token behavior.
        state = self.mem_norm(final_state) * torch.tanh(self.state_gate)   # [M, D]
        mem = torch.cat([state, local_tok], dim=0)                 # [M + k*L, D]
        msk = torch.cat([torch.ones(state.size(0), device=mem.device),
                         local_msk.float()], dim=0)                # [M + k*L]
        return mem.unsqueeze(0), msk.unsqueeze(0), idx             # add batch dim

    def run_state(self, batch):
        """Shared front half: encode scenes + names, run the streaming state."""
        scene_tok = self._encode_scenes(batch["scene_ids"], batch["scene_mask"])
        name_embs = self._encode_names(batch["ename_ids"], batch["ename_mask"])
        init = self.state.init_slots(name_embs)
        final, deltas, probes, states = self.state(
            scene_tok, batch["scene_mask"], batch["presence"], init)
        return scene_tok, final, deltas, probes, states

    # --- training -----------------------------------------------------------
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        scene_tok, final, deltas, probes, _states = self.run_state(batch)
        mem, msk, ret_idx = self._build_memory(final, scene_tok, batch["scene_mask"])

        labels = batch["labels"].unsqueeze(0)                      # [1, T]
        out = self.backbone(encoder_outputs=BaseModelOutput(last_hidden_state=mem),
                            attention_mask=msk, labels=labels)
        nll = out.loss

        # entity-presence probe loss (only over valid entities)
        ev = batch["entity_valid"].bool()                          # [E]
        if ev.any():
            tgt = (batch["presence"] > 0).float()[:, ev]           # [S, Ev]
            logit = probes[:, ev]                                  # [S, Ev]
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
                 num_beams: int = 4, max_new_tokens: int = 400, **kw):
        scene_tok, final, deltas, _, _ = self.run_state(batch)
        if perturb_slot is not None:            # causal intervention: erase a slot
            final = final.clone(); final[perturb_slot] = 0.0
        mem, msk, ret_idx = self._build_memory(final, scene_tok, batch["scene_mask"])
        gen_model = getattr(self.backbone, "generate")
        ids = gen_model(encoder_outputs=BaseModelOutput(last_hidden_state=mem),
                        attention_mask=msk, num_beams=num_beams,
                        max_new_tokens=max_new_tokens, no_repeat_ngram_size=3,
                        length_penalty=1.0, **kw)
        return ids, deltas, ret_idx


def move_batch(batch: Dict, device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out

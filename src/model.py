"""Candidate-scoring Imitation Learning Policy Network for CABT Pokémon TCG.

Scores every legal option in `obs["select"]["option"]` conditioned on the board
state, visible hand/board card embeddings, and option entity attributes, plus a
learned stop-threshold logit `z_stop` for variable-count multi-select prompts.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features import (
    MAX_ATTACK_ID,
    MAX_CARD_ID,
    MAX_SELECT_CTX,
    MAX_SELECT_TYPE,
    OPT_DIM,
    STATE_DIM,
    extract_decision_features,
)


class PTCGPolicyNet(nn.Module):
    """Neural candidate-scoring policy with multi-select stop threshold."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        opt_dim: int = OPT_DIM,
        hidden_dim: int = 128,
        card_emb_dim: int = 24,
        atk_emb_dim: int = 16,
        ctx_emb_dim: int = 12,
        type_emb_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.opt_dim = opt_dim
        self.hidden_dim = hidden_dim

        self.card_emb = nn.Embedding(MAX_CARD_ID, card_emb_dim, padding_idx=0)
        self.atk_emb = nn.Embedding(MAX_ATTACK_ID, atk_emb_dim, padding_idx=0)
        self.ctx_emb = nn.Embedding(MAX_SELECT_CTX, ctx_emb_dim)
        self.type_emb = nn.Embedding(MAX_SELECT_TYPE, type_emb_dim)

        # State input: state_vec + hand_bag + my_active + my_bench_bag + opp_active + opp_bench_bag + ctx + type
        state_in_dim = state_dim + 5 * card_emb_dim + ctx_emb_dim + type_emb_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(state_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Option input: opt_feats + src_card_emb + tgt_card_emb + atk_emb
        opt_in_dim = opt_dim + 2 * card_emb_dim + atk_emb_dim
        self.opt_encoder = nn.Sequential(
            nn.Linear(opt_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Joint candidate scorer: [state_h, opt_h, state_h * opt_h] -> scalar logit
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Stop threshold head for variable-count selections (minCount < maxCount)
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode_state(
        self,
        state_vec: torch.Tensor,
        hand_cards: torch.Tensor,
        board_cards: torch.Tensor,
        select_ctx: torch.Tensor,
        select_type: torch.Tensor,
    ) -> torch.Tensor:
        # hand_cards: (B, MAX_HAND)
        hand_mask = (hand_cards > 0).unsqueeze(-1).float()
        hand_e = self.card_emb(hand_cards) * hand_mask
        hand_bag = hand_e.sum(dim=1) / hand_mask.sum(dim=1).clamp(min=1.0)

        # board_cards: (B, 12) -> [my_act, my_bench(5), opp_act, opp_bench(5)]
        my_act_e = self.card_emb(board_cards[:, 0])
        my_bench_ids = board_cards[:, 1:6]
        mb_mask = (my_bench_ids > 0).unsqueeze(-1).float()
        my_bench_bag = (self.card_emb(my_bench_ids) * mb_mask).sum(dim=1) / mb_mask.sum(dim=1).clamp(min=1.0)

        opp_act_e = self.card_emb(board_cards[:, 6])
        opp_bench_ids = board_cards[:, 7:12]
        ob_mask = (opp_bench_ids > 0).unsqueeze(-1).float()
        opp_bench_bag = (self.card_emb(opp_bench_ids) * ob_mask).sum(dim=1) / ob_mask.sum(dim=1).clamp(min=1.0)

        ctx_e = self.ctx_emb(select_ctx.clamp(0, MAX_SELECT_CTX - 1))
        type_e = self.type_emb(select_type.clamp(0, MAX_SELECT_TYPE - 1))

        x = torch.cat(
            [state_vec, hand_bag, my_act_e, my_bench_bag, opp_act_e, opp_bench_bag, ctx_e, type_e],
            dim=-1,
        )
        return self.state_encoder(x)

    def _encode_options(
        self,
        opt_feats: torch.Tensor,
        opt_card_ids: torch.Tensor,
        opt_target_card_ids: torch.Tensor,
        opt_attack_ids: torch.Tensor,
    ) -> torch.Tensor:
        c_e = self.card_emb(opt_card_ids.clamp(0, MAX_CARD_ID - 1))
        t_e = self.card_emb(opt_target_card_ids.clamp(0, MAX_CARD_ID - 1))
        a_e = self.atk_emb(opt_attack_ids.clamp(0, MAX_ATTACK_ID - 1))
        x = torch.cat([opt_feats, c_e, t_e, a_e], dim=-1)
        return self.opt_encoder(x)

    def forward_packed(
        self,
        state_vec: torch.Tensor,
        hand_cards: torch.Tensor,
        board_cards: torch.Tensor,
        select_ctx: torch.Tensor,
        select_type: torch.Tensor,
        num_options: torch.Tensor,
        opt_feats: torch.Tensor,
        opt_card_ids: torch.Tensor,
        opt_target_card_ids: torch.Tensor,
        opt_attack_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass on packed variable-length batch.

        Returns:
            opt_logits: (total_options,) scalar logit per option
            stop_logits: (batch_size,) scalar stop threshold logit per decision
        """
        state_h = self._encode_state(state_vec, hand_cards, board_cards, select_ctx, select_type)
        opt_h = self._encode_options(opt_feats, opt_card_ids, opt_target_card_ids, opt_attack_ids)

        state_rep = torch.repeat_interleave(state_h, num_options, dim=0)
        joint = torch.cat([state_rep, opt_h, state_rep * opt_h], dim=-1)
        opt_logits = self.scorer(joint).squeeze(-1)

        # Pool option embeddings per sample for stop threshold head
        batch_size = state_h.size(0)
        seg_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=state_h.device), num_options
        )
        opt_mean = torch.zeros_like(state_h)
        opt_mean.index_add_(0, seg_ids, opt_h)
        opt_mean = opt_mean / num_options.unsqueeze(-1).clamp(min=1).float()

        stop_logits = self.stop_head(torch.cat([state_h, opt_mean], dim=-1)).squeeze(-1)
        return opt_logits, stop_logits

    @staticmethod
    def decode_action_from_logits(
        logits: np.ndarray,
        stop_logit: float,
        min_count: int,
        max_count: int,
    ) -> list[int]:
        """Decode a legal `list[int]` action strictly respecting `[min_count, max_count]`."""
        n_opts = int(logits.shape[0])
        if n_opts == 0:
            return []
        min_c = max(0, min(int(min_count), n_opts))
        max_c = max(min_c, min(int(max_count), n_opts))
        if max_c == 0:
            return []
        if min_c == 1 and max_c == 1:
            return [int(np.argmax(logits))]

        # Rank legal option indices by descending score
        order = np.argsort(-logits)
        chosen: list[int] = []
        for rank_idx, opt_idx in enumerate(order):
            if len(chosen) >= max_c:
                break
            if rank_idx < min_c or float(logits[opt_idx]) > float(stop_logit):
                chosen.append(int(opt_idx))
            else:
                break

        # For multi-select in CABT, preserve sorted unique legal option indices
        if len(chosen) > 1:
            chosen.sort()
        return chosen

    @torch.no_grad()
    def select_action(
        self,
        obs: dict[str, Any],
        default_deck: list[int],
        device: torch.device | str = "cpu",
    ) -> list[int]:
        """Select a legal action `list[int]` for a live simulator `obs` dict."""
        if not isinstance(obs, dict):
            return list(default_deck)
        sel = obs.get("select")
        cur = obs.get("current")
        # Initial deck selection phase
        if sel is None or cur is None:
            return list(default_deck)

        opts = sel.get("option") or []
        n_opts = len(opts)
        min_c = int(sel.get("minCount") or 0)
        max_c = int(sel.get("maxCount") or 1)

        if n_opts == 0 or max_c <= 0:
            return []
        if n_opts == 1 and min_c == 1 and max_c >= 1:
            return [0]

        feats = extract_decision_features(obs)
        dev = torch.device(device)

        state_vec = torch.from_numpy(feats["state_vec"]).unsqueeze(0).to(dev)
        hand_cards = torch.from_numpy(feats["hand_cards"]).unsqueeze(0).to(dev)
        board_cards = torch.from_numpy(feats["board_cards"]).unsqueeze(0).to(dev)
        select_ctx = torch.tensor([feats["select_ctx"]], dtype=torch.long, device=dev)
        select_type = torch.tensor([feats["select_type"]], dtype=torch.long, device=dev)
        num_options = torch.tensor([n_opts], dtype=torch.long, device=dev)

        opt_feats = torch.from_numpy(feats["opt_feats"]).to(dev)
        opt_card_ids = torch.from_numpy(feats["opt_card_ids"]).to(dev)
        opt_target_card_ids = torch.from_numpy(feats["opt_target_card_ids"]).to(dev)
        opt_attack_ids = torch.from_numpy(feats["opt_attack_ids"]).to(dev)

        opt_logits, stop_logits = self.forward_packed(
            state_vec,
            hand_cards,
            board_cards,
            select_ctx,
            select_type,
            num_options,
            opt_feats,
            opt_card_ids,
            opt_target_card_ids,
            opt_attack_ids,
        )

        logits_np = opt_logits.cpu().numpy()
        stop_np = float(stop_logits[0].item())

        # Anti-loop guard: if turnActionCount >= 20 in MAIN menu, restrict to turn-completing legal options (ATTACK=13 or END=14)
        if int(cur.get("turnActionCount") or 0) >= 20 and int(sel.get("type") or -1) == 0:
            term_indices = [i for i, o in enumerate(opts) if int(o.get("type") or -1) in (13, 14)]
            if term_indices:
                mask = np.full_like(logits_np, -1e9)
                mask[term_indices] = logits_np[term_indices]
                logits_np = mask

        return self.decode_action_from_logits(logits_np, stop_np, min_c, max_c)

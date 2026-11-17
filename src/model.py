"""Candidate-scoring Imitation & RL Policy-Value Network for CABT Pokémon TCG.

Features:
- State encoder + Entity embeddings + Optional Set-Context Self-Attention across legal options.
- Marginal candidate scorer + Stop-threshold head for `minCount == 0` empty selections.
- Conditional Multi-Select Scorer (`cond_scorer`, `cond_stop_head`) for `2 <= maxCount <= 6`.
- `value_head` for RL / Self-Play Advantage & PPO/AWR fine-tuning (backward-compatible with Phase 2).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.features import (
    LEGACY_OPT_DIM,
    LEGACY_STATE_DIM,
    MAX_ATTACK_ID,
    MAX_CARD_ID,
    MAX_SELECT_CTX,
    MAX_SELECT_TYPE,
    OPT_DIM,
    STATE_DIM,
    extract_decision_features,
    get_attack_db,
    get_card_db,
    int_or,
)


@lru_cache(maxsize=16)
def _get_deck_pokemon_profiles(
    deck_sig: tuple[int, ...],
) -> tuple[set[int], dict[int, tuple[int, float, int, float]]]:
    """Compute achievable `(own_max_cost, own_max_dmg, line_max_cost, line_max_dmg)`
    for every Pokémon card given the energy types present in `deck_sig`.
    """
    card_db = get_card_db()
    atk_db = get_attack_db()
    deck_set = set(deck_sig)

    deck_etypes: set[int] = set()
    for cid in deck_set:
        cmeta = card_db.get(cid)
        if cmeta is not None and int(getattr(cmeta, "cardType", -1)) in (5, 6):
            et = int(getattr(cmeta, "energyType", 0) or 0)
            if et > 0:
                deck_etypes.add(et)
    if not deck_etypes:
        deck_etypes = {1, 2, 3, 4, 5, 6, 7, 8, 9}

    children_by_name: dict[str, list[int]] = {}
    for cid, cmeta in card_db.items():
        if int(getattr(cmeta, "cardType", -1)) != 0:
            continue
        ev_from = getattr(cmeta, "evolvesFrom", None)
        if ev_from and (not deck_set or cid in deck_set):
            children_by_name.setdefault(str(ev_from), []).append(cid)

    def _own_stats(cid: int) -> tuple[int, float]:
        cmeta = card_db.get(cid)
        if cmeta is None:
            return (2, 70.0)
        max_cost = 0
        max_dmg = 0.0
        for aid in getattr(cmeta, "attacks", []) or []:
            ameta = atk_db.get(int(aid))
            if ameta is None:
                continue
            reqs = list(getattr(ameta, "energies", []) or [])
            if any(r != 0 and r not in deck_etypes for r in reqs):
                continue
            cost = len(reqs)
            dmg = float(getattr(ameta, "damage", 0) or 0)
            if dmg <= 0.0 and cost >= 3:
                dmg = 80.0
            if dmg > 0.0 or cost > 0:
                if cost > max_cost:
                    max_cost = cost
                if dmg > max_dmg:
                    max_dmg = dmg
        return (max_cost, max_dmg)

    profiles: dict[int, tuple[int, float, int, float]] = {}
    for cid, cmeta in card_db.items():
        if int(getattr(cmeta, "cardType", -1)) != 0:
            continue
        own_c, own_d = _own_stats(cid)
        line_c, line_d = own_c, own_d
        cname = getattr(cmeta, "name", None)
        stack = list(children_by_name.get(str(cname), [])) if cname else []
        visited = {cid}
        while stack:
            ecid = stack.pop()
            if ecid in visited:
                continue
            visited.add(ecid)
            ec, ed = _own_stats(ecid)
            if ec > line_c:
                line_c = ec
            if ed > line_d:
                line_d = ed
            ename = getattr(card_db.get(ecid), "name", None)
            if ename:
                stack.extend(children_by_name.get(str(ename), []))
        profiles[cid] = (own_c, own_d, line_c, line_d)

    return deck_etypes, profiles


class PTCGPolicyNet(nn.Module):
    """Neural candidate-scoring policy-value network with conditional multi-select."""

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
        use_set_attn: bool = True,
        use_cond_multi: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.opt_dim = opt_dim
        self.hidden_dim = hidden_dim
        self.card_emb_dim = card_emb_dim
        self.atk_emb_dim = atk_emb_dim
        self.ctx_emb_dim = ctx_emb_dim
        self.type_emb_dim = type_emb_dim
        self.use_set_attn = use_set_attn
        self.use_cond_multi = use_cond_multi

        self.card_emb = nn.Embedding(MAX_CARD_ID, card_emb_dim, padding_idx=0)
        self.atk_emb = nn.Embedding(MAX_ATTACK_ID, atk_emb_dim, padding_idx=0)
        self.ctx_emb = nn.Embedding(MAX_SELECT_CTX, ctx_emb_dim)
        self.type_emb = nn.Embedding(MAX_SELECT_TYPE, type_emb_dim)
        self.emb_drop = nn.Dropout(0.15)

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

        if use_set_attn:
            self.opt_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            self.opt_attn_norm = nn.LayerNorm(hidden_dim)
            self.opt_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.opt_ffn_norm = nn.LayerNorm(hidden_dim)

        # Joint candidate scorer: [state_h, opt_ctx, state_h * opt_ctx] -> scalar logit
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

        # Conditional Multi-Select Scorer conditioned on already-picked subset pool
        if use_cond_multi:
            self.cond_scorer = nn.Sequential(
                nn.Linear(hidden_dim * 3 + 1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.cond_stop_head = nn.Sequential(
                nn.Linear(hidden_dim * 3 + 1, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        # Value head for RL / Self-Play fine-tuning (always initialized, safe on Phase 2 ckpts)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
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
        if state_vec.size(-1) != self.state_dim:
            state_vec = state_vec[..., : self.state_dim]
        hand_mask = (hand_cards > 0).unsqueeze(-1).float()
        hand_e = self.emb_drop(self.card_emb(hand_cards.clamp(0, MAX_CARD_ID - 1))) * hand_mask
        hand_bag = hand_e.sum(dim=1) / hand_mask.sum(dim=1).clamp(min=1.0)

        my_act_e = self.emb_drop(self.card_emb(board_cards[:, 0].clamp(0, MAX_CARD_ID - 1)))
        my_bench_ids = board_cards[:, 1:6].clamp(0, MAX_CARD_ID - 1)
        mb_mask = (my_bench_ids > 0).unsqueeze(-1).float()
        my_bench_bag = (self.emb_drop(self.card_emb(my_bench_ids)) * mb_mask).sum(dim=1) / mb_mask.sum(dim=1).clamp(min=1.0)

        opp_act_e = self.emb_drop(self.card_emb(board_cards[:, 6].clamp(0, MAX_CARD_ID - 1)))
        opp_bench_ids = board_cards[:, 7:12].clamp(0, MAX_CARD_ID - 1)
        ob_mask = (opp_bench_ids > 0).unsqueeze(-1).float()
        opp_bench_bag = (self.emb_drop(self.card_emb(opp_bench_ids)) * ob_mask).sum(dim=1) / ob_mask.sum(dim=1).clamp(min=1.0)

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
        if opt_feats.size(-1) != self.opt_dim:
            opt_feats = opt_feats[..., : self.opt_dim]
        c_e = self.emb_drop(self.card_emb(opt_card_ids.clamp(0, MAX_CARD_ID - 1)))
        t_e = self.emb_drop(self.card_emb(opt_target_card_ids.clamp(0, MAX_CARD_ID - 1)))
        a_e = self.emb_drop(self.atk_emb(opt_attack_ids.clamp(0, MAX_ATTACK_ID - 1)))
        x = torch.cat([opt_feats, c_e, t_e, a_e], dim=-1)
        return self.opt_encoder(x)

    def _contextualize_options(
        self,
        opt_h: torch.Tensor,
        num_options: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_set_attn:
            return opt_h
        batch_size = int(num_options.size(0))
        max_k = int(num_options.max().item())
        if max_k <= 1:
            return opt_h
        device = opt_h.device
        row_ids = torch.repeat_interleave(torch.arange(batch_size, device=device), num_options)
        cum_opts = torch.cumsum(num_options, dim=0) - num_options
        col_ids = torch.arange(opt_h.size(0), device=device) - torch.repeat_interleave(cum_opts, num_options)

        padded = torch.zeros((batch_size, max_k, self.hidden_dim), device=device, dtype=opt_h.dtype)
        pad_mask = torch.ones((batch_size, max_k), device=device, dtype=torch.bool)
        padded[row_ids, col_ids] = opt_h
        pad_mask[row_ids, col_ids] = False

        attn_out, _ = self.opt_attn(padded, padded, padded, key_padding_mask=pad_mask, need_weights=False)
        h = self.opt_attn_norm(padded + attn_out)
        h = self.opt_ffn_norm(h + self.opt_ffn(h))
        return h[row_ids, col_ids]

    def forward_packed_with_value(
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning `(opt_logits, stop_logits, values, state_h, opt_ctx)`."""
        state_h = self._encode_state(state_vec, hand_cards, board_cards, select_ctx, select_type)
        opt_raw = self._encode_options(opt_feats, opt_card_ids, opt_target_card_ids, opt_attack_ids)
        opt_ctx = self._contextualize_options(opt_raw, num_options)

        state_rep = torch.repeat_interleave(state_h, num_options, dim=0)
        joint = torch.cat([state_rep, opt_ctx, state_rep * opt_ctx], dim=-1)
        opt_logits = self.scorer(joint).squeeze(-1)

        batch_size = state_h.size(0)
        seg_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=state_h.device), num_options
        )
        opt_mean = torch.zeros_like(state_h)
        opt_mean.index_add_(0, seg_ids, opt_ctx)
        opt_mean = opt_mean / num_options.unsqueeze(-1).clamp(min=1).float()

        stop_logits = self.stop_head(torch.cat([state_h, opt_mean], dim=-1)).squeeze(-1)
        values = self.value_head(state_h).squeeze(-1)
        return opt_logits, stop_logits, values, state_h, opt_ctx

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
        """Forward pass on packed variable-length batch returning `(opt_logits, stop_logits)`."""
        opt_logits, stop_logits, _, _, _ = self.forward_packed_with_value(
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
        return opt_logits, stop_logits

    def forward_conditional_subset(
        self,
        state_h: torch.Tensor,
        opt_ctx: torch.Tensor,
        num_options: torch.Tensor,
        base_opt_logits: torch.Tensor,
        selected_flat_mask: torch.Tensor,
        selected_frac: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score remaining unselected options conditioned on an already-chosen subset `S`.

        Args:
            state_h: `(B, H)` encoded states.
            opt_ctx: `(sum_K, H)` contextualized option embeddings.
            num_options: `(B,)` option counts per decision.
            base_opt_logits: `(sum_K,)` marginal option logits.
            selected_flat_mask: `(sum_K,)` float 0/1 mask marking options already in `S`.
            selected_frac: `(B, 1)` normalized cardinality `|S| / max_count`.

        Returns:
            `(cond_opt_logits, cond_stop_logits)` where already-selected options are masked `-1e9`.
        """
        device = state_h.device
        batch_size = state_h.size(0)
        seg_ids = torch.repeat_interleave(torch.arange(batch_size, device=device), num_options)

        opt_mean = torch.zeros_like(state_h)
        opt_mean.index_add_(0, seg_ids, opt_ctx)
        opt_mean = opt_mean / num_options.unsqueeze(-1).clamp(min=1).float()

        sel_sum = torch.zeros_like(state_h)
        sel_sum.index_add_(0, seg_ids, opt_ctx * selected_flat_mask.unsqueeze(-1))
        sel_cnt = torch.zeros((batch_size, 1), device=device, dtype=state_h.dtype)
        sel_cnt.index_add_(0, seg_ids, selected_flat_mask.unsqueeze(-1))
        sel_pool = sel_sum / sel_cnt.clamp(min=1.0)

        if not self.use_cond_multi:
            cond_logits = base_opt_logits.masked_fill(selected_flat_mask > 0.5, -1e9)
            cond_stop = self.stop_head(torch.cat([state_h, opt_mean], dim=-1)).squeeze(-1)
            return cond_logits, cond_stop

        state_rep = torch.repeat_interleave(state_h, num_options, dim=0)
        pool_rep = torch.repeat_interleave(sel_pool, num_options, dim=0)
        frac_rep = torch.repeat_interleave(selected_frac, num_options, dim=0)

        delta = self.cond_scorer(
            torch.cat([state_rep, opt_ctx, pool_rep, frac_rep], dim=-1)
        ).squeeze(-1)
        cond_logits = (base_opt_logits + delta).masked_fill(selected_flat_mask > 0.5, -1e9)
        cond_stop = self.cond_stop_head(
            torch.cat([state_h, opt_mean, sel_pool, selected_frac], dim=-1)
        ).squeeze(-1)
        return cond_logits, cond_stop

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
        if max_c <= 0:
            return []
        if min_c == 1 and max_c == 1:
            return [int(np.argmax(logits))]

        order = np.argsort(-logits)
        chosen: list[int] = []
        for rank_idx, opt_idx in enumerate(order):
            if len(chosen) >= max_c:
                break
            if rank_idx < min_c or float(logits[opt_idx]) > float(stop_logit):
                chosen.append(int(opt_idx))
            else:
                break

        if len(chosen) > 1:
            chosen.sort()
        return chosen

    @staticmethod
    def _apply_tactical_guardrails(
        logits_np: np.ndarray,
        opts: list[dict[str, Any]],
        opt_feats: np.ndarray,
        stype: int,
        turn_action_count: int,
        state_vec: np.ndarray | None = None,
        min_c: int = 1,
        max_c: int = 1,
        opt_card_ids: np.ndarray | None = None,
        opt_target_card_ids: np.ndarray | None = None,
        hand_cards: np.ndarray | None = None,
        board_cards: np.ndarray | None = None,
        default_deck: list[int] | None = None,
        obs: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Apply domain-consistent tactical priors for unseen deck generalization
        and enforce strict anti-loop termination when `turn_action_count >= 20`.
        """
        out = logits_np.copy()
        deck_sig = tuple(sorted(set(default_deck))) if default_deck else ()
        _, profiles = _get_deck_pokemon_profiles(deck_sig)

        if stype != 0:
            is_turn0 = state_vec is not None and float(state_vec[1]) > 0.5
            # 1a. Turn-0 Active selection
            if is_turn0 and min_c == 1 and max_c == 1 and opt_card_ids is not None:
                for i in range(len(opts)):
                    cid = int(opt_card_ids[i])
                    own_c, own_d, line_c, line_d = profiles.get(cid, (1, 40.0, 2, 80.0))
                    if line_d <= 0.0:
                        out[i] -= 5.0
                    elif cid in (140, 184) and own_c >= 3:
                        out[i] -= 3.0
                    elif own_c <= 1 and own_d > 0.0:
                        out[i] += 2.0
                    elif line_d >= 150.0:
                        out[i] += 1.8
                return out

            # 1b. Turn-0 Bench Setup
            if is_turn0 and min_c == 0 and max_c > 0:
                for i in range(len(opts)):
                    cid = int(opt_card_ids[i]) if opt_card_ids is not None else 0
                    _, _, _, line_d = profiles.get(cid, (1, 40.0, 2, 80.0))
                    if line_d >= 70.0 and cid not in (140, 184):
                        out[i] += 4.0
                    else:
                        out[i] += 2.0
                return out

            # 2. Optional Deck/Looking search
            if min_c == 0 and max_c > 0 and len(opts) > 0:
                if all(int_or(o, "area", -1) in (1, 12) for o in opts):
                    for i in range(len(opts)):
                        f = opt_feats[i]
                        cid = int(opt_card_ids[i]) if opt_card_ids is not None else 0
                        _, _, _, line_d = profiles.get(cid, (1, 40.0, 2, 80.0))
                        out[i] += 3.5 + (1.5 if (f[31] > 0.5 and line_d >= 70.0) else 0.0)
                    return out

            # 3. Promoting or switching to a Bench Pokémon
            if min_c == 1 and max_c == 1 and len(opts) > 0:
                if all(opt_feats[i][21] > 0.5 and opt_feats[i][26] > 0.5 for i in range(len(opts))):
                    for i in range(len(opts)):
                        f = opt_feats[i]
                        cid = int(opt_card_ids[i]) if opt_card_ids is not None else 0
                        en_cnt = int(round(float(f[43]) * 6.0))
                        own_c, own_d, line_c, line_d = profiles.get(cid, (1, 40.0, 2, 80.0))
                        if line_d <= 0.0:
                            out[i] -= 4.0
                        elif own_d > 0.0 and en_cnt >= own_c:
                            out[i] += 4.0 + 0.02 * own_d
                        else:
                            out[i] += 2.0 * float(f[43]) + 0.008 * line_d + 0.8 * float(f[41]) - 0.5 * float(f[42])
                    return out
            return out

        # Strict Anti-Loop Guard when turnActionCount >= 20 in MAIN menu (stype == 0)
        if turn_action_count >= 20:
            term_indices = [i for i, o in enumerate(opts) if int_or(o, "type", -1) in (13, 14)]
            if term_indices:
                mask = np.full_like(out, -1e9)
                for idx in term_indices:
                    ot = int_or(opts[idx], "type", -1)
                    bonus = 5.0 if ot == 13 else 0.0
                    mask[idx] = out[idx] + bonus
                return mask

        atk_indices = [i for i, o in enumerate(opts) if int_or(o, "type", -1) == 13]
        has_attack = len(atk_indices) > 0
        max_eff_dmg = max((float(opt_feats[i][45]) for i in atk_indices), default=0.0)
        active_dmg = max_eff_dmg * 300.0
        any_lethal = any(opt_feats[i][47] > 0.5 for i in atk_indices)

        my_bench_norm = float(state_vec[37]) if state_vec is not None else 0.4
        my_deck_norm = float(state_vec[33]) if state_vec is not None else 0.5
        my_hand_norm = float(state_vec[31]) if state_vec is not None else 0.4
        active_en_cnt = int(round(float(state_vec[43]) * 6.0)) if state_vec is not None else 0
        hand_set = set(int(x) for x in hand_cards if int(x) > 0) if hand_cards is not None else set()
        board_set = set(int(x) for x in board_cards[:6] if int(x) > 0) if board_cards is not None else set()

        bench_max_ready_dmg = 0.0
        if obs is not None and isinstance(obs.get("current"), dict):
            cur_d = obs["current"]
            me_idx = int_or(cur_d, "yourIndex", 0)
            players = cur_d.get("players") or []
            if 0 <= me_idx < len(players):
                for bp in players[me_idx].get("bench") or []:
                    if isinstance(bp, dict):
                        bcid = int_or(bp, "id", 0)
                        ben = len(bp.get("energies") or [])
                        b_own_c, b_own_d, _, _ = profiles.get(bcid, (1, 40.0, 2, 80.0))
                        if b_own_d > 0.0 and ben >= b_own_c and b_own_d > bench_max_ready_dmg:
                            bench_max_ready_dmg = b_own_d
        should_pivot_to_bench = (not any_lethal) and (bench_max_ready_dmg >= max(70.0, active_dmg + 50.0))

        for i, o in enumerate(opts):
            ot = int_or(o, "type", -1)
            f = opt_feats[i]
            src_cid = int(opt_card_ids[i]) if opt_card_ids is not None else 0
            tgt_cid = int(opt_target_card_ids[i]) if opt_target_card_ids is not None else 0

            if ot == 13:  # ATTACK
                is_lethal = f[47] > 0.5
                eff_dmg = float(f[45])
                if is_lethal:
                    out[i] += 6.5 + 2.0 * eff_dmg
                elif eff_dmg >= max_eff_dmg - 1e-5:
                    out[i] += 2.0 + 2.5 * eff_dmg
                else:
                    out[i] -= 3.0
            elif ot == 9:  # EVOLVE
                to_active = f[24] > 0.5
                en_cnt = int(round(float(f[43]) * 6.0))
                own_c, own_d, _, line_d = profiles.get(src_cid, (1, 70.0, 2, 120.0))
                if to_active and own_d <= 0.0 and has_attack and 121 not in hand_set:
                    out[i] -= 2.5
                elif to_active and any_lethal and own_d < active_dmg:
                    out[i] -= 2.0
                else:
                    ev_bonus = 3.5 + 0.006 * line_d
                    if en_cnt > 0:
                        ev_bonus += 1.6 + 0.5 * float(en_cnt) + (0.4 if to_active else 0.0)
                    elif to_active and (own_d > 0.0 or src_cid == 1031):
                        ev_bonus += 0.5
                    out[i] += ev_bonus
            elif ot == 8:  # ATTACH energy/tool
                to_active = f[24] > 0.5
                en_cnt = int(round(float(f[43]) * 6.0))
                own_c, own_d, line_c, line_d = profiles.get(tgt_cid, (1, 40.0, 2, 80.0))
                if line_c <= 0 or line_d <= 0.0 or en_cnt >= line_c:
                    out[i] -= 4.5
                elif tgt_cid in (65, 66):
                    out[i] += 0.5
                else:
                    base_att = 2.6 + 0.006 * line_d
                    if en_cnt > 0:
                        base_att += 1.6 + 0.5 * float(en_cnt) + (0.4 if to_active else 0.0)
                    elif to_active and (own_d > 0.0 or tgt_cid == 1030):
                        base_att += 0.6
                    out[i] += base_att
            elif ot == 7:  # PLAY card from hand
                is_pokemon = f[31] > 0.5
                if is_pokemon:
                    _, _, _, line_d = profiles.get(src_cid, (1, 40.0, 2, 80.0))
                    if line_d <= 0.0 or src_cid in (140, 184):
                        if src_cid not in board_set and my_bench_norm < 0.6:
                            out[i] += 1.2
                        else:
                            out[i] -= 2.0
                    else:
                        out[i] += 3.2 if my_bench_norm < 0.8 else 1.5
                elif src_cid == 1123:  # Switch
                    if should_pivot_to_bench:
                        out[i] += 4.8
                    elif has_attack:
                        out[i] -= 5.0
                    else:
                        out[i] += 1.0
                elif src_cid == 1182:  # Boss's Orders
                    if any_lethal:
                        out[i] -= 5.0
                    else:
                        out[i] += 1.0
                elif src_cid == 1192:  # Carmine
                    if my_deck_norm <= 0.35 or my_hand_norm >= 0.5:
                        out[i] -= 4.5
                    else:
                        out[i] += 0.8
                elif src_cid == 1227:  # Lillie's Determination
                    if my_deck_norm <= 0.30:
                        out[i] -= 4.0
                    else:
                        out[i] += 1.4
                elif src_cid in (1079, 1086, 1097, 1198):
                    out[i] += 2.8 if src_cid == 1079 else 2.0
                else:
                    out[i] += 1.2
            elif ot == 10:  # ABILITY
                en_cnt = int(round(float(f[43]) * 6.0))
                if src_cid == 66:
                    if en_cnt > 0 or my_deck_norm <= 0.30 or f[24] > 0.5:
                        out[i] -= 5.0
                    elif turn_action_count < 5:
                        out[i] += 1.2
                    else:
                        out[i] -= 2.0
                elif my_deck_norm <= 0.18:
                    out[i] -= 3.5
                elif turn_action_count < 6:
                    out[i] += 1.0
                else:
                    out[i] -= 2.0
            elif ot == 12:  # RETREAT
                if should_pivot_to_bench and (184 in board_set or active_en_cnt == 0):
                    out[i] += 4.2
                elif has_attack:
                    out[i] -= 3.5
                else:
                    out[i] -= 0.5
            elif ot == 14:  # END turn
                if has_attack:
                    out[i] -= 5.0

        return out

    @torch.no_grad()
    def select_action(
        self,
        obs: dict[str, Any],
        default_deck: list[int],
        device: torch.device | str = "cpu",
        apply_guardrails: bool = True,
    ) -> list[int]:
        """Select a legal action `list[int]` for a live simulator `obs` dict."""
        if not isinstance(obs, dict):
            return list(default_deck)
        sel = obs.get("select")
        cur = obs.get("current")
        if sel is None or cur is None:
            return list(default_deck)

        opts = sel.get("option") or []
        n_opts = len(opts)
        min_c = int_or(sel, "minCount", 0)
        max_c = int_or(sel, "maxCount", 1)

        if n_opts == 0 or max_c <= 0:
            return []
        eff_min = max(0, min(min_c, n_opts))
        eff_max = max(eff_min, min(max_c, n_opts))
        if n_opts == 1 and eff_min == 1 and eff_max >= 1:
            return [0]

        feats = extract_decision_features(obs)
        dev = torch.device(device)

        state_vec = torch.from_numpy(feats["state_vec"][: self.state_dim]).unsqueeze(0).to(dev)
        hand_cards = torch.from_numpy(feats["hand_cards"]).unsqueeze(0).to(dev)
        board_cards = torch.from_numpy(feats["board_cards"]).unsqueeze(0).to(dev)
        select_ctx = torch.tensor([feats["select_ctx"]], dtype=torch.long, device=dev)
        select_type = torch.tensor([feats["select_type"]], dtype=torch.long, device=dev)
        num_options = torch.tensor([n_opts], dtype=torch.long, device=dev)

        opt_feats = torch.from_numpy(feats["opt_feats"][:, : self.opt_dim]).to(dev)
        opt_card_ids = torch.from_numpy(feats["opt_card_ids"]).to(dev)
        opt_target_card_ids = torch.from_numpy(feats["opt_target_card_ids"]).to(dev)
        opt_attack_ids = torch.from_numpy(feats["opt_attack_ids"]).to(dev)

        opt_logits, stop_logits, _, state_h, opt_ctx = self.forward_packed_with_value(
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

        stype = int_or(sel, "type", 0)
        turn_action_count = int_or(cur, "turnActionCount", 0)
        if apply_guardrails:
            logits_np = self._apply_tactical_guardrails(
                logits_np,
                opts,
                feats["opt_feats"],
                stype,
                turn_action_count,
                state_vec=feats["state_vec"],
                min_c=eff_min,
                max_c=eff_max,
                opt_card_ids=feats["opt_card_ids"],
                opt_target_card_ids=feats["opt_target_card_ids"],
                hand_cards=feats["hand_cards"],
                board_cards=feats["board_cards"],
                default_deck=default_deck,
                obs=obs,
            )

        # Conditional sequential decoding for Multi-Select (`eff_max > 1`)
        if self.use_cond_multi and eff_max > 1 and n_opts > 1:
            guard_delta = logits_np - opt_logits.cpu().numpy()
            chosen: list[int] = []
            sel_mask = torch.zeros(n_opts, device=dev, dtype=state_h.dtype)

            # First step uses base logits + base stop_logit
            first_idx = int(np.argmax(logits_np))
            if eff_min == 0 and float(logits_np[first_idx]) <= stop_np:
                return []
            chosen.append(first_idx)
            sel_mask[first_idx] = 1.0

            for step in range(1, eff_max):
                frac_t = torch.tensor([[float(len(chosen)) / max(eff_max, 1)]], device=dev, dtype=state_h.dtype)
                c_logits, c_stop = self.forward_conditional_subset(
                    state_h,
                    opt_ctx,
                    num_options,
                    opt_logits,
                    sel_mask,
                    frac_t,
                )
                c_logits_np = c_logits.cpu().numpy() + guard_delta
                for picked in chosen:
                    c_logits_np[picked] = -1e9
                next_idx = int(np.argmax(c_logits_np))
                c_stop_val = float(c_stop[0].item())
                if step >= eff_min and float(c_logits_np[next_idx]) <= c_stop_val:
                    break
                chosen.append(next_idx)
                sel_mask[next_idx] = 1.0

            chosen.sort()
            return chosen

        return self.decode_action_from_logits(logits_np, stop_np, eff_min, eff_max)

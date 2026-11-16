"""Feature extraction for CABT Pokémon TCG observations and legal options.

Strictly uses only information visible to the deciding player (`yourIndex`) at
the decision timestamp. Never accesses opponent hidden hand cards, future steps,
or internal simulator state (`search_begin_input`).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

STATE_DIM = 64
OPT_DIM = 48
MAX_HAND_CARDS = 16
NUM_BOARD_SLOTS = 12  # [my_active, my_bench_0..4, opp_active, opp_bench_0..4]
MAX_CARD_ID = 2048
MAX_ATTACK_ID = 2048
MAX_SELECT_CTX = 64
MAX_SELECT_TYPE = 16


def int_or(d: dict[str, Any], key: str, default: int) -> int:
    """Safely read an integer field from `d` without falsy-zero bugs (`0` is preserved)."""
    val = d.get(key)
    return default if val is None else int(val)


@lru_cache(maxsize=1)
def get_card_db() -> dict[int, Any]:
    """Load card metadata dictionary {cardId: CardData} from cg.api."""
    try:
        import cg.api
        return {int(c.cardId): c for c in cg.api.all_card_data()}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def get_attack_db() -> dict[int, Any]:
    """Load attack metadata dictionary {attackId: Attack} from cg.api."""
    try:
        import cg.api
        return {int(a.attackId): a for a in cg.api.all_attack()}
    except Exception:
        return {}


def _clamp_card_id(cid: int | None) -> int:
    if cid is None or not isinstance(cid, int) or cid <= 0:
        return 0
    return min(cid, MAX_CARD_ID - 1)


def _clamp_attack_id(aid: int | None) -> int:
    if aid is None or not isinstance(aid, int) or aid <= 0:
        return 0
    return min(aid, MAX_ATTACK_ID - 1)


def validate_no_leakage(obs: dict[str, Any], expected_seat: int | None = None) -> None:
    """Verify that the observation contains no hidden opponent information."""
    if not isinstance(obs, dict):
        raise ValueError("Observation must be a dict")
    cur = obs.get("current")
    if cur is None:
        return
    me = cur.get("yourIndex")
    if me not in (0, 1):
        raise ValueError(f"Invalid yourIndex in observation: {me}")
    if expected_seat is not None and me != expected_seat:
        raise ValueError(f"Seat mismatch: yourIndex={me} vs expected_seat={expected_seat}")
    players = cur.get("players") or []
    if len(players) != 2:
        raise ValueError(f"Expected 2 players in current state, got {len(players)}")
    opp = 1 - me
    if players[opp].get("hand") is not None:
        raise ValueError("Information leakage detected: opponent hand is not None!")
    if players[me].get("hand") is None:
        raise ValueError("Invalid self state: active player's own hand is None!")


def _pokemon_summary(p: dict[str, Any] | None, card_db: dict[int, Any]) -> tuple[float, float, float, float, float, float, float, int]:
    """Return (present, hp_norm, max_hp_norm, dmg_ratio, n_energy, n_tools, is_ex, card_id)."""
    if not p or not isinstance(p, dict):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    cid = _clamp_card_id(p.get("id"))
    hp = float(int_or(p, "hp", 0))
    max_hp = float(int_or(p, "maxHp", int(max(hp, 1.0))))
    dmg_ratio = max(0.0, (max_hp - hp) / max(max_hp, 1.0))
    energies = p.get("energies") or []
    tools = p.get("tools") or []
    cmeta = card_db.get(cid)
    is_ex = 1.0 if (cmeta and getattr(cmeta, "ex", False)) or max_hp >= 210.0 else 0.0
    return (
        1.0,
        min(hp / 340.0, 1.5),
        min(max_hp / 340.0, 1.5),
        min(dmg_ratio, 1.0),
        min(len(energies) / 6.0, 1.5),
        min(len(tools) / 2.0, 1.0),
        is_ex,
        cid,
    )


def _resolve_card_from_area(
    obs: dict[str, Any],
    player_idx: int,
    area: int | None,
    index: int | None,
) -> tuple[int, dict[str, Any] | None]:
    """Resolve (card_id, pokemon_dict_if_in_play) for a (player_idx, area, index) reference."""
    if area is None or index is None or index < 0:
        return 0, None
    cur = obs.get("current") or {}
    sel = obs.get("select") or {}
    players = cur.get("players") or []
    if not (0 <= player_idx < len(players)):
        return 0, None
    pstate = players[player_idx]

    # AreaType: 1=DECK, 2=HAND, 3=DISCARD, 4=ACTIVE, 5=BENCH, 6=PRIZE, 7=STADIUM, 12=LOOKING
    if area == 1:  # DECK
        dk = sel.get("deck")
        if isinstance(dk, list) and index < len(dk) and isinstance(dk[index], dict):
            return _clamp_card_id(dk[index].get("id")), None
    elif area == 2:  # HAND
        hand = pstate.get("hand")
        if isinstance(hand, list) and index < len(hand) and isinstance(hand[index], dict):
            return _clamp_card_id(hand[index].get("id")), None
    elif area == 3:  # DISCARD
        disc = pstate.get("discard")
        if isinstance(disc, list) and index < len(disc) and isinstance(disc[index], dict):
            return _clamp_card_id(disc[index].get("id")), None
    elif area == 4:  # ACTIVE
        act = pstate.get("active")
        if isinstance(act, list) and index < len(act) and isinstance(act[index], dict):
            return _clamp_card_id(act[index].get("id")), act[index]
    elif area == 5:  # BENCH
        bench = pstate.get("bench")
        if isinstance(bench, list) and index < len(bench) and isinstance(bench[index], dict):
            return _clamp_card_id(bench[index].get("id")), bench[index]
    elif area == 6:  # PRIZE
        prize = pstate.get("prize")
        if isinstance(prize, list) and index < len(prize) and isinstance(prize[index], dict):
            return _clamp_card_id(prize[index].get("id")), None
    elif area == 7:  # STADIUM
        stad = cur.get("stadium")
        if isinstance(stad, list) and index < len(stad) and isinstance(stad[index], dict):
            return _clamp_card_id(stad[index].get("id")), None
    elif area == 12:  # LOOKING
        looking = cur.get("looking")
        if isinstance(looking, list) and index < len(looking) and isinstance(looking[index], dict):
            return _clamp_card_id(looking[index].get("id")), None
    return 0, None


def extract_decision_features(obs: dict[str, Any]) -> dict[str, Any]:
    """Extract dense & entity features for state and all legal options in `obs`."""
    validate_no_leakage(obs)
    card_db = get_card_db()
    atk_db = get_attack_db()

    cur = obs["current"]
    sel = obs["select"]
    opts = sel.get("option") or []
    me = int_or(cur, "yourIndex", 0)
    opp = 1 - me
    p_me = cur["players"][me]
    p_opp = cur["players"][opp]

    state_vec = np.zeros(STATE_DIM, dtype=np.float32)
    turn = float(int_or(cur, "turn", 0))
    state_vec[0] = min(turn / 25.0, 2.0)
    state_vec[1] = 1.0 if turn == 0 else 0.0
    state_vec[2] = min(float(int_or(cur, "turnActionCount", 0)) / 15.0, 2.0)
    first_player = int_or(cur, "firstPlayer", -1)
    state_vec[3] = 1.0 if first_player == me else 0.0
    state_vec[4] = 1.0 if first_player == opp else 0.0
    state_vec[5] = 1.0 if cur.get("supporterPlayed") else 0.0
    state_vec[6] = 1.0 if cur.get("stadiumPlayed") else 0.0
    state_vec[7] = 1.0 if cur.get("energyAttached") else 0.0
    state_vec[8] = 1.0 if cur.get("retreated") else 0.0
    state_vec[9] = float(int_or(cur, "round", 1)) / 3.0

    stype = int_or(sel, "type", 0)
    sctx = int_or(sel, "context", 0)
    min_c = int_or(sel, "minCount", 0)
    max_c = int_or(sel, "maxCount", 1)
    if 0 <= stype < 11:
        state_vec[10 + stype] = 1.0
    state_vec[21] = min(min_c / 5.0, 1.5)
    state_vec[22] = min(max_c / 5.0, 1.5)
    state_vec[23] = 1.0 if max_c > 1 else 0.0
    state_vec[24] = 1.0 if min_c == 0 else 0.0
    state_vec[25] = min(float(int_or(sel, "remainDamageCounter", 0)) / 10.0, 2.0)
    state_vec[26] = min(float(int_or(sel, "remainEnergyCost", 0)) / 5.0, 2.0)
    state_vec[27] = min(len(opts) / 20.0, 2.0)

    # Player counts
    my_prizes = len(p_me.get("prize") or [])
    opp_prizes = len(p_opp.get("prize") or [])
    my_hand_cnt = int_or(p_me, "handCount", 0)
    opp_hand_cnt = int_or(p_opp, "handCount", 0)
    my_deck_cnt = int_or(p_me, "deckCount", 0)
    opp_deck_cnt = int_or(p_opp, "deckCount", 0)
    my_disc_cnt = len(p_me.get("discard") or [])
    opp_disc_cnt = len(p_opp.get("discard") or [])
    my_bench = p_me.get("bench") or []
    opp_bench = p_opp.get("bench") or []

    state_vec[28] = my_prizes / 6.0
    state_vec[29] = opp_prizes / 6.0
    state_vec[30] = (opp_prizes - my_prizes) / 6.0  # positive = I am ahead in prize race
    state_vec[31] = min(my_hand_cnt / 10.0, 2.0)
    state_vec[32] = min(opp_hand_cnt / 10.0, 2.0)
    state_vec[33] = my_deck_cnt / 60.0
    state_vec[34] = opp_deck_cnt / 60.0
    state_vec[35] = min(my_disc_cnt / 30.0, 2.0)
    state_vec[36] = min(opp_disc_cnt / 30.0, 2.0)
    state_vec[37] = len(my_bench) / 5.0
    state_vec[38] = len(opp_bench) / 5.0

    # Active Pokémon summaries
    my_act_list = p_me.get("active") or []
    opp_act_list = p_opp.get("active") or []
    my_act = my_act_list[0] if my_act_list else None
    opp_act = opp_act_list[0] if opp_act_list else None

    ma_pres, ma_hp, ma_mhp, ma_dmg, ma_en, ma_tl, ma_ex, ma_cid = _pokemon_summary(my_act, card_db)
    oa_pres, oa_hp, oa_mhp, oa_dmg, oa_en, oa_tl, oa_ex, oa_cid = _pokemon_summary(opp_act, card_db)

    state_vec[39:46] = [ma_pres, ma_hp, ma_mhp, ma_dmg, ma_en, ma_tl, ma_ex]
    state_vec[46:53] = [oa_pres, oa_hp, oa_mhp, oa_dmg, oa_en, oa_tl, oa_ex]

    # Status conditions & bench aggregates
    state_vec[53] = 1.0 if any(p_me.get(k) for k in ("poisoned", "burned", "asleep", "paralyzed", "confused")) else 0.0
    state_vec[54] = 1.0 if any(p_opp.get(k) for k in ("poisoned", "burned", "asleep", "paralyzed", "confused")) else 0.0
    state_vec[55] = min(sum(len(bp.get("energies") or []) for bp in my_bench if isinstance(bp, dict)) / 10.0, 2.0)
    state_vec[56] = min(sum(len(bp.get("energies") or []) for bp in opp_bench if isinstance(bp, dict)) / 10.0, 2.0)

    # Hand composition summary (only self hand is visible; includes CardType.STADIUM == 4)
    my_hand = p_me.get("hand") or []
    hand_cards = np.zeros(MAX_HAND_CARDS, dtype=np.int64)
    n_poke = n_item_or_stad = n_supp = n_energy = n_basic_poke = 0
    for idx_h, hcard in enumerate(my_hand):
        if not isinstance(hcard, dict):
            continue
        cid = _clamp_card_id(hcard.get("id"))
        if idx_h < MAX_HAND_CARDS:
            hand_cards[idx_h] = cid
        cmeta = card_db.get(cid)
        if cmeta is not None:
            ct = int(getattr(cmeta, "cardType", -1))
            if ct == 0:
                n_poke += 1
                if getattr(cmeta, "basic", False):
                    n_basic_poke += 1
            elif ct in (1, 2, 4):  # ITEM (1), TOOL (2), STADIUM (4)
                n_item_or_stad += 1
            elif ct == 3:  # SUPPORTER (3)
                n_supp += 1
            elif ct in (5, 6):  # BASIC_ENERGY (5), SPECIAL_ENERGY (6)
                n_energy += 1

    state_vec[57] = min(n_poke / 6.0, 1.5)
    state_vec[58] = min(n_basic_poke / 4.0, 1.5)
    state_vec[59] = min(n_item_or_stad / 6.0, 1.5)
    state_vec[60] = min(n_supp / 4.0, 1.5)
    state_vec[61] = min(n_energy / 6.0, 1.5)
    state_vec[62] = 1.0 if (cur.get("stadium") and len(cur["stadium"]) > 0) else 0.0
    state_vec[63] = float(int_or(p_me, "win", 0)) - float(int_or(p_opp, "win", 0))

    # Board slot card IDs: [my_active, my_bench_0..4, opp_active, opp_bench_0..4]
    board_cards = np.zeros(NUM_BOARD_SLOTS, dtype=np.int64)
    board_cards[0] = ma_cid
    for bi in range(min(5, len(my_bench))):
        if isinstance(my_bench[bi], dict):
            board_cards[1 + bi] = _clamp_card_id(my_bench[bi].get("id"))
    board_cards[6] = oa_cid
    for bi in range(min(5, len(opp_bench))):
        if isinstance(opp_bench[bi], dict):
            board_cards[7 + bi] = _clamp_card_id(opp_bench[bi].get("id"))

    # Per-option features
    n_opts = len(opts)
    opt_feats = np.zeros((max(n_opts, 1), OPT_DIM), dtype=np.float32)
    opt_card_ids = np.zeros(max(n_opts, 1), dtype=np.int64)
    opt_target_card_ids = np.zeros(max(n_opts, 1), dtype=np.int64)
    opt_attack_ids = np.zeros(max(n_opts, 1), dtype=np.int64)

    opp_act_hp = float(int_or(opp_act, "hp", 0)) if isinstance(opp_act, dict) else 0.0
    opp_act_weak = None
    if oa_cid in card_db:
        opp_act_weak = getattr(card_db[oa_cid], "weakness", None)
    my_act_etype = None
    if ma_cid in card_db:
        my_act_etype = getattr(card_db[ma_cid], "energyType", None)

    for i, o in enumerate(opts):
        v = opt_feats[i]
        otype = int_or(o, "type", 0)
        if 0 <= otype < 17:
            v[otype] = 1.0

        area = o.get("area")
        in_play_area = o.get("inPlayArea")
        o_index = o.get("index")
        in_play_index = o.get("inPlayIndex")
        p_idx = o.get("playerIndex")
        if p_idx is None:
            p_idx = me

        # Area encoding (17..23): includes STADIUM (7) mapped with LOOKING (12) in slot 23
        if area in (1, 2, 3, 4, 5, 6):
            area_slot = {1: 17, 2: 18, 3: 19, 4: 20, 5: 21, 6: 22}[area]
            v[area_slot] = 1.0
        elif area in (7, 12):  # STADIUM (7) or LOOKING (12)
            v[23] = 1.0

        if in_play_area == 4:
            v[24] = 1.0
        elif in_play_area == 5:
            v[25] = 1.0

        v[26] = 1.0 if p_idx == me else 0.0
        v[27] = 1.0 if p_idx == opp else 0.0
        v[28] = min(float(int_or(o, "number", 0)) / 10.0, 2.0)
        v[29] = min(float(int_or(o, "count", 0)) / 5.0, 2.0)
        v[30] = float(i) / max(n_opts - 1, 1)

        src_cid = 0
        tgt_cid = 0
        src_poke = None
        tgt_poke = None
        aid = _clamp_attack_id(o.get("attackId"))

        if otype == 7:  # PLAY from hand
            src_cid, _ = _resolve_card_from_area(obs, me, 2, o_index)
        elif otype in (8, 9):  # ATTACH or EVOLVE
            src_cid, _ = _resolve_card_from_area(obs, me, area, o_index)
            tgt_cid, tgt_poke = _resolve_card_from_area(obs, me, in_play_area, in_play_index)
        elif otype in (3, 10, 11):  # CARD, ABILITY, DISCARD
            src_cid, src_poke = _resolve_card_from_area(obs, int(p_idx), area, o_index)
            if src_poke is not None:
                tgt_poke = src_poke
                tgt_cid = src_cid
        elif otype in (4, 5, 6):  # TOOL_CARD, ENERGY_CARD, ENERGY
            tgt_cid, tgt_poke = _resolve_card_from_area(obs, int(p_idx), area, o_index)
            if isinstance(tgt_poke, dict):
                if otype == 4:
                    tools = tgt_poke.get("tools") or []
                    t_idx = o.get("toolIndex")
                    if isinstance(t_idx, int) and 0 <= t_idx < len(tools) and isinstance(tools[t_idx], dict):
                        src_cid = _clamp_card_id(tools[t_idx].get("id"))
                else:
                    ecards = tgt_poke.get("energyCards") or []
                    e_idx = o.get("energyIndex")
                    if isinstance(e_idx, int) and 0 <= e_idx < len(ecards) and isinstance(ecards[e_idx], dict):
                        src_cid = _clamp_card_id(ecards[e_idx].get("id"))
        elif otype == 12:  # RETREAT
            src_cid = ma_cid
            tgt_poke = my_act
        elif otype == 13:  # ATTACK
            src_cid = ma_cid
            tgt_cid = oa_cid
            tgt_poke = opp_act
        elif otype == 15:  # SKILL
            src_cid = _clamp_card_id(o.get("cardId"))

        opt_card_ids[i] = src_cid
        opt_target_card_ids[i] = tgt_cid
        opt_attack_ids[i] = aid

        # Source card metadata features (31..40)
        cmeta = card_db.get(src_cid)
        if cmeta is not None:
            ctype = int(getattr(cmeta, "cardType", -1))
            if 0 <= ctype < 7:
                v[31 + ctype] = 1.0
            v[38] = min(float(getattr(cmeta, "hp", 0) or 0) / 340.0, 1.5)
            v[39] = 1.0 if getattr(cmeta, "ex", False) else 0.0
            v[40] = 1.0 if (getattr(cmeta, "skills", None) and len(cmeta.skills) > 0) else 0.0
        elif isinstance(src_poke, dict):
            # Fallback metadata when cardId > 1267 is in play on the board
            v[31] = 1.0  # POKEMON
            hp_p = float(int_or(src_poke, "maxHp", 0))
            v[38] = min(hp_p / 340.0, 1.5)
            v[39] = 1.0 if hp_p >= 210.0 else 0.0
            v[40] = 1.0 if otype == 10 else 0.0

        # Target Pokémon state features (41..44)
        if isinstance(tgt_poke, dict):
            _, thp, tmhp, tdmg, ten, _, _, _ = _pokemon_summary(tgt_poke, card_db)
            v[41] = thp
            v[42] = tdmg
            v[43] = ten
            v[44] = 1.0 if tgt_poke.get("appearThisTurn") else 0.0

        # Attack features (45..47)
        if aid > 0 and aid in atk_db:
            ameta = atk_db[aid]
            dmg = float(getattr(ameta, "damage", 0) or 0)
            if dmg > 0 and opp_act_weak is not None and my_act_etype is not None and opp_act_weak == my_act_etype:
                eff_dmg = dmg * 2.0
            else:
                eff_dmg = dmg
            v[45] = min(eff_dmg / 300.0, 2.0)
            v[46] = min(len(getattr(ameta, "energies", []) or []) / 5.0, 1.5)
            v[47] = 1.0 if (opp_act_hp > 0 and eff_dmg >= opp_act_hp) else 0.0
        elif otype == 13 and aid > 0:
            # Fallback for newer attackId > 1556
            v[45] = 0.5
            v[46] = 0.4

    return {
        "state_vec": state_vec,
        "hand_cards": hand_cards,
        "board_cards": board_cards,
        "select_ctx": min(max(sctx, 0), MAX_SELECT_CTX - 1),
        "select_type": min(max(stype, 0), MAX_SELECT_TYPE - 1),
        "min_count": min_c,
        "max_count": max_c,
        "num_options": n_opts,
        "opt_feats": opt_feats[:n_opts] if n_opts > 0 else opt_feats[:1],
        "opt_card_ids": opt_card_ids[:n_opts] if n_opts > 0 else opt_card_ids[:1],
        "opt_target_card_ids": opt_target_card_ids[:n_opts] if n_opts > 0 else opt_target_card_ids[:1],
        "opt_attack_ids": opt_attack_ids[:n_opts] if n_opts > 0 else opt_attack_ids[:1],
    }

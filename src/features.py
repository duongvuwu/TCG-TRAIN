"""Feature extraction and static fallback metadata for CABT Pokémon TCG observations.

Strictly uses only information visible to the deciding player (`yourIndex`) at
the decision timestamp. Never accesses opponent hidden hand cards, future steps,
or internal simulator state (`search_begin_input`).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

STATE_DIM = 112
OPT_DIM = 64
LEGACY_STATE_DIM = 64
LEGACY_OPT_DIM = 48
MAX_HAND_CARDS = 16
NUM_BOARD_SLOTS = 12  # [my_active, my_bench_0..4, opp_active, opp_bench_0..4]
MAX_CARD_ID = 2048
MAX_ATTACK_ID = 2048
MAX_SELECT_CTX = 64
MAX_SELECT_TYPE = 16


@dataclass(frozen=True)
class FallbackCardMeta:
    """Lightweight card metadata mirroring `cg.api.CardData` for post-1267 cards."""

    cardId: int
    cardType: int = 0
    hp: int = 0
    ex: bool = False
    basic: bool = True
    stage: int = 0
    energyType: int = 0
    weakness: int | None = None
    attacks: tuple[int, ...] = ()
    skills: tuple[int, ...] = ()
    evolvesFrom: str | None = None
    name: str = ""


@dataclass(frozen=True)
class FallbackAttackMeta:
    """Lightweight attack metadata mirroring `cg.api.Attack` for post-1556 attacks."""

    attackId: int
    damage: int = 0
    energies: tuple[int, ...] = ()
    name: str = ""


# Static fallback metadata table inferred from 2026-11-13-v1 replays for IDs
# outside legacy `cg/cg.dll` (`cardId > 1267`, `attackId > 1556`).
FALLBACK_CARD_DB: dict[int, FallbackCardMeta] = {
    1331: FallbackCardMeta(
        cardId=1331,
        cardType=0,
        hp=180,
        ex=False,
        basic=False,
        stage=2,
        energyType=0,
        attacks=(1641, 1642),
        name="FallbackPokemon_1331",
    ),
    1387: FallbackCardMeta(
        cardId=1387,
        cardType=0,
        hp=70,
        ex=False,
        basic=True,
        stage=0,
        energyType=0,
        attacks=(1715, 1716),
        name="FallbackPokemon_1387",
    ),
    1406: FallbackCardMeta(
        cardId=1406,
        cardType=0,
        hp=340,
        ex=True,
        basic=False,
        stage=2,
        energyType=0,
        attacks=(1747, 1748),
        name="FallbackPokemonEx_1406",
    ),
    1413: FallbackCardMeta(
        cardId=1413,
        cardType=1,
        hp=0,
        ex=False,
        basic=False,
        stage=0,
        name="FallbackTrainer_1413",
    ),
}

FALLBACK_ATTACK_DB: dict[int, FallbackAttackMeta] = {
    1641: FallbackAttackMeta(attackId=1641, damage=60, energies=(0,), name="FallbackAttack_1641"),
    1642: FallbackAttackMeta(attackId=1642, damage=300, energies=(0, 0, 0, 0), name="FallbackAttack_1642"),
    1715: FallbackAttackMeta(attackId=1715, damage=0, energies=(0,), name="FallbackAttack_1715"),
    1716: FallbackAttackMeta(attackId=1716, damage=50, energies=(0, 0, 0), name="FallbackAttack_1716"),
    1747: FallbackAttackMeta(attackId=1747, damage=90, energies=(0, 0), name="FallbackAttack_1747"),
    1748: FallbackAttackMeta(attackId=1748, damage=200, energies=(0, 0, 0), name="FallbackAttack_1748"),
}

_RUNTIME_CARD_OVERRIDES: dict[int, FallbackCardMeta] = {}
_RUNTIME_ATTACK_OVERRIDES: dict[int, FallbackAttackMeta] = {}


def int_or(d: dict[str, Any], key: str, default: int) -> int:
    """Safely read an integer field from `d` without falsy-zero bugs (`0` is preserved)."""
    val = d.get(key)
    return default if val is None else int(val)


@lru_cache(maxsize=1)
def _load_cg_card_db() -> dict[int, Any]:
    try:
        import cg.api

        return {int(c.cardId): c for c in cg.api.all_card_data()}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_cg_attack_db() -> dict[int, Any]:
    try:
        import cg.api

        return {int(a.attackId): a for a in cg.api.all_attack()}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def get_card_db() -> dict[int, Any]:
    """Load card metadata dictionary `{cardId: CardData}` merged with fallback metadata."""
    merged: dict[int, Any] = dict(FALLBACK_CARD_DB)
    merged.update(_RUNTIME_CARD_OVERRIDES)
    merged.update(_load_cg_card_db())
    return merged


@lru_cache(maxsize=1)
def get_attack_db() -> dict[int, Any]:
    """Load attack metadata dictionary `{attackId: Attack}` merged with fallback metadata."""
    merged: dict[int, Any] = dict(FALLBACK_ATTACK_DB)
    merged.update(_RUNTIME_ATTACK_OVERRIDES)
    merged.update(_load_cg_attack_db())
    return merged


def register_fallback_metadata(
    cards: dict[int, FallbackCardMeta] | None = None,
    attacks: dict[int, FallbackAttackMeta] | None = None,
) -> None:
    """Register additional fallback metadata entries and invalidate cached lookups."""
    if cards:
        _RUNTIME_CARD_OVERRIDES.update(cards)
    if attacks:
        _RUNTIME_ATTACK_OVERRIDES.update(attacks)
    get_card_db.cache_clear()
    get_attack_db.cache_clear()


def infer_fallback_metadata_from_replays(
    replay_dir: str | Path,
    max_files: int = 300,
) -> tuple[dict[int, FallbackCardMeta], dict[int, FallbackAttackMeta]]:
    """Infer card and attack metadata for IDs absent from legacy `cg.api` using replay logs."""
    cg_cards = _load_cg_card_db()
    cg_atks = _load_cg_attack_db()

    files = sorted(
        [p for p in Path(replay_dir).glob("*.json") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )[:max_files]

    card_hps: dict[int, list[int]] = {}
    card_types: dict[int, Counter[int]] = {}
    card_pre_evo_max: dict[int, int] = {}
    card_etypes: dict[int, Counter[int]] = {}
    card_atks: dict[int, set[int]] = {}
    card_has_skill: set[int] = set()

    atk_dmgs: dict[int, list[int]] = {}
    atk_min_cost: dict[int, int] = {}

    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            continue

        for st in raw.get("steps") or []:
            for seat_obj in st:
                obs = seat_obj.get("observation") or {}
                cur = obs.get("current") or {}
                players = cur.get("players") or []
                for pl in players:
                    if not isinstance(pl, dict):
                        continue
                    for area_key in ("active", "bench"):
                        for p in pl.get(area_key) or []:
                            if not isinstance(p, dict):
                                continue
                            cid = p.get("id")
                            if not isinstance(cid, int) or cid <= 0 or cid in cg_cards:
                                continue
                            card_types.setdefault(cid, Counter())[0] += 1
                            mhp = int_or(p, "maxHp", 0)
                            if mhp > 0:
                                card_hps.setdefault(cid, []).append(mhp)
                            pre = p.get("preEvolution") or p.get("preEvolutions") or []
                            card_pre_evo_max[cid] = max(card_pre_evo_max.get(cid, 0), len(pre))

                sel = obs.get("select") or {}
                me = cur.get("yourIndex", 0)
                my_act = None
                if isinstance(me, int) and 0 <= me < len(players) and isinstance(players[me], dict):
                    act_l = players[me].get("active") or []
                    if act_l and isinstance(act_l[0], dict):
                        my_act = act_l[0]

                for o in sel.get("option") or []:
                    if not isinstance(o, dict):
                        continue
                    otype = int_or(o, "type", -1)
                    aid = o.get("attackId")
                    if isinstance(aid, int) and aid > 0 and aid not in cg_atks and my_act is not None:
                        src_cid = my_act.get("id")
                        if isinstance(src_cid, int) and src_cid > 0 and src_cid not in cg_cards:
                            card_atks.setdefault(src_cid, set()).add(aid)
                        en_cnt = len(my_act.get("energies") or [])
                        if en_cnt >= 0:
                            prev = atk_min_cost.get(aid)
                            atk_min_cost[aid] = en_cnt if prev is None else min(prev, en_cnt)
                    if otype == 10:  # ABILITY
                        area = o.get("area")
                        idx = o.get("index")
                        if isinstance(me, int) and 0 <= me < len(players) and isinstance(idx, int) and idx >= 0:
                            arr = players[me].get("active" if area == 4 else "bench" if area == 5 else "")
                            if isinstance(arr, list) and idx < len(arr) and isinstance(arr[idx], dict):
                                scid = arr[idx].get("id")
                                if isinstance(scid, int) and scid > 0 and scid not in cg_cards:
                                    card_has_skill.add(scid)
                    elif otype == 8:  # ATTACH (energy=5/6 or tool=2)
                        area = o.get("area")
                        idx = o.get("index")
                        if area == 2 and isinstance(me, int) and 0 <= me < len(players) and isinstance(idx, int):
                            hand = players[me].get("hand") or []
                            if 0 <= idx < len(hand) and isinstance(hand[idx], dict):
                                scid = hand[idx].get("id")
                                if isinstance(scid, int) and scid > 0 and scid not in cg_cards:
                                    card_types.setdefault(scid, Counter())[5] += 1
                    elif otype == 9:  # EVOLVE
                        area = o.get("area")
                        idx = o.get("index")
                        if area == 2 and isinstance(me, int) and 0 <= me < len(players) and isinstance(idx, int):
                            hand = players[me].get("hand") or []
                            if 0 <= idx < len(hand) and isinstance(hand[idx], dict):
                                scid = hand[idx].get("id")
                                if isinstance(scid, int) and scid > 0 and scid not in cg_cards:
                                    card_types.setdefault(scid, Counter())[0] += 1
                                    card_pre_evo_max[scid] = max(card_pre_evo_max.get(scid, 1), 1)
                    elif otype == 7:  # PLAY from hand (Trainer/Item/Supporter/Stadium/Basic)
                        idx = o.get("index")
                        if isinstance(me, int) and 0 <= me < len(players) and isinstance(idx, int):
                            hand = players[me].get("hand") or []
                            if 0 <= idx < len(hand) and isinstance(hand[idx], dict):
                                scid = hand[idx].get("id")
                                if isinstance(scid, int) and scid > 0 and scid not in cg_cards:
                                    # Only assign Trainer (1) if not seen on field as Pokemon
                                    card_types.setdefault(scid, Counter())[1] += 1

                logs = obs.get("logs") or []
                for i, lg in enumerate(logs):
                    if not isinstance(lg, dict):
                        continue
                    ltype = lg.get("type")
                    if ltype == 15:  # ATTACK
                        aid = lg.get("attackId")
                        scid = lg.get("cardId")
                        if isinstance(aid, int) and aid > 0 and aid not in cg_atks:
                            if isinstance(scid, int) and scid > 0 and scid not in cg_cards:
                                card_atks.setdefault(scid, set()).add(aid)
                            max_hit = 0
                            for j in range(i + 1, min(i + 6, len(logs))):
                                nxt = logs[j]
                                if nxt.get("type") in (15, 2, 3):
                                    break
                                if nxt.get("type") == 16 and not nxt.get("putDamageCounter"):
                                    val = int(nxt.get("value", 0) or 0)
                                    if val < 0:
                                        max_hit = max(max_hit, -val)
                            if max_hit > 0:
                                atk_dmgs.setdefault(aid, []).append(max_hit)
                    elif ltype == 11:  # ATTACH
                        scid = lg.get("cardId")
                        if isinstance(scid, int) and scid > 0 and scid not in cg_cards:
                            card_types.setdefault(scid, Counter())[5] += 1

    inferred_attacks: dict[int, FallbackAttackMeta] = {}
    all_aids = set(atk_dmgs.keys()) | set(atk_min_cost.keys())
    for aid in sorted(all_aids):
        dmgs = atk_dmgs.get(aid, [])
        dmg = int(max(dmgs)) if dmgs else 0
        # Weaknesses double damage; if median vs max is exact 2x, pick mode/median
        if len(dmgs) >= 3:
            cnt = Counter(dmgs)
            dmg = int(cnt.most_common(1)[0][0])
        cost = max(0, int(atk_min_cost.get(aid, 1)))
        inferred_attacks[aid] = FallbackAttackMeta(
            attackId=aid,
            damage=dmg,
            energies=tuple([0] * cost),
            name=f"FallbackAttack_{aid}",
        )

    inferred_cards: dict[int, FallbackCardMeta] = {}
    all_cids = set(card_types.keys()) | set(card_hps.keys()) | set(card_atks.keys())
    for cid in sorted(all_cids):
        tcounts = card_types.get(cid, Counter())
        if cid in card_hps or cid in card_atks or tcounts.get(0, 0) > 0:
            ctype = 0
        elif tcounts.get(5, 0) > 0:
            ctype = 5
        else:
            ctype = tcounts.most_common(1)[0][0] if tcounts else 1
        hps = card_hps.get(cid, [])
        hp_val = int(Counter(hps).most_common(1)[0][0]) if hps else 0
        stage_val = card_pre_evo_max.get(cid, 0)
        inferred_cards[cid] = FallbackCardMeta(
            cardId=cid,
            cardType=ctype,
            hp=hp_val,
            ex=(ctype == 0 and hp_val >= 210),
            basic=(ctype == 0 and stage_val == 0),
            stage=stage_val,
            energyType=0,
            attacks=tuple(sorted(card_atks.get(cid, set()))),
            skills=(1,) if cid in card_has_skill else (),
            name=f"FallbackCard_{cid}",
        )

    return inferred_cards, inferred_attacks


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


def _pokemon_stage_norm(p: dict[str, Any] | None, cmeta: Any) -> float:
    if isinstance(p, dict):
        pre = p.get("preEvolution") or p.get("preEvolutions") or []
        if isinstance(pre, list) and len(pre) > 0:
            return min(len(pre) * 0.5, 1.0)
    if cmeta is not None:
        st = getattr(cmeta, "stage", None)
        if isinstance(st, int) and st > 0:
            return min(st * 0.5, 1.0)
        if getattr(cmeta, "evolvesFrom", None):
            return 0.5
        if getattr(cmeta, "basic", True) is False:
            return 0.5
    return 0.0


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


def _discard_breakdown(discard_list: list[Any] | None, card_db: dict[int, Any]) -> tuple[int, int, int, Counter[int]]:
    """Count (pokemon, energy, trainer, card_id_counts) in a public discard pile."""
    n_poke = n_energy = n_trainer = 0
    cid_counts: Counter[int] = Counter()
    for dcard in discard_list or []:
        if not isinstance(dcard, dict):
            continue
        cid = _clamp_card_id(dcard.get("id"))
        if cid > 0:
            cid_counts[cid] += 1
        cmeta = card_db.get(cid)
        if cmeta is None:
            continue
        ct = int(getattr(cmeta, "cardType", -1))
        if ct == 0:
            n_poke += 1
        elif ct in (5, 6):
            n_energy += 1
        elif ct in (1, 2, 3, 4):
            n_trainer += 1
    return n_poke, n_energy, n_trainer, cid_counts


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
    hand_cid_counts: Counter[int] = Counter()
    n_poke = n_item_or_stad = n_supp = n_energy = n_basic_poke = 0
    for idx_h, hcard in enumerate(my_hand):
        if not isinstance(hcard, dict):
            continue
        cid = _clamp_card_id(hcard.get("id"))
        if cid > 0:
            hand_cid_counts[cid] += 1
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

    # --- Phase 3 Enriched State Features (64..111) ---
    # 64..83: My bench per-slot (5 x 4): [hp_norm, dmg_ratio, n_energy_norm, stage_norm]
    for bi in range(min(5, len(my_bench))):
        bp = my_bench[bi]
        if isinstance(bp, dict):
            _, b_hp, _, b_dmg, b_en, _, _, b_cid = _pokemon_summary(bp, card_db)
            b_stage = _pokemon_stage_norm(bp, card_db.get(b_cid))
            base = 64 + bi * 4
            state_vec[base] = b_hp
            state_vec[base + 1] = b_dmg
            state_vec[base + 2] = b_en
            state_vec[base + 3] = b_stage

    # 84..93: Opp bench per-slot (5 x 2): [hp_norm, n_energy_norm]
    for bi in range(min(5, len(opp_bench))):
        bp = opp_bench[bi]
        if isinstance(bp, dict):
            _, ob_hp, _, _, ob_en, _, _, _ = _pokemon_summary(bp, card_db)
            base = 84 + bi * 2
            state_vec[base] = ob_hp
            state_vec[base + 1] = ob_en

    # 94..95: Active evolution stage norms
    state_vec[94] = _pokemon_stage_norm(my_act, card_db.get(ma_cid))
    state_vec[95] = _pokemon_stage_norm(opp_act, card_db.get(oa_cid))

    # 96..101: Public discard pile breakdown (pokemon, energy, trainer)
    my_dp, my_de, my_dt, my_disc_counts = _discard_breakdown(p_me.get("discard"), card_db)
    opp_dp, opp_de, opp_dt, _ = _discard_breakdown(p_opp.get("discard"), card_db)
    state_vec[96] = min(my_dp / 15.0, 1.5)
    state_vec[97] = min(my_de / 10.0, 1.5)
    state_vec[98] = min(my_dt / 20.0, 1.5)
    state_vec[99] = min(opp_dp / 15.0, 1.5)
    state_vec[100] = min(opp_de / 10.0, 1.5)
    state_vec[101] = min(opp_dt / 20.0, 1.5)

    # 102..111: Causal short-term action history from `obs["logs"]` (strictly past/current logs)
    logs = obs.get("logs") or []
    state_vec[102] = min(len(logs) / 20.0, 2.0)
    n_draws_me = 0
    n_draws_opp = 0
    dmg_taken_me = 0.0
    dmg_dealt_opp = 0.0
    for lg in logs:
        if not isinstance(lg, dict):
            continue
        lt = lg.get("type")
        lp = lg.get("playerIndex")
        if lt == 10:  # PLAY
            state_vec[103] = 1.0
        elif lt == 11:  # ATTACH
            state_vec[104] = 1.0
        elif lt == 12:  # EVOLVE
            state_vec[105] = 1.0
        elif lt == 15:  # ATTACK
            state_vec[106] = 1.0
        elif lt == 8:  # SWITCH
            state_vec[107] = 1.0
        elif lt == 4 and lp == me:
            n_draws_me += 1
        elif lt in (4, 5) and lp == opp:
            n_draws_opp += 1
        elif lt == 16:  # HP_CHANGE
            val = float(lg.get("value", 0) or 0)
            if val < 0:
                if lp == me:
                    dmg_taken_me += -val
                elif lp == opp:
                    dmg_dealt_opp += -val
    state_vec[108] = min(n_draws_me / 7.0, 1.5)
    state_vec[109] = min(n_draws_opp / 7.0, 1.5)
    state_vec[110] = min(dmg_taken_me / 300.0, 1.5)
    state_vec[111] = min(dmg_dealt_opp / 300.0, 1.5)

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
    my_act_en_cnt = len(my_act.get("energies") or []) if isinstance(my_act, dict) else 0

    group_counts: Counter[tuple[int, int]] = Counter()
    resolved_opts: list[tuple[int, int, dict[str, Any] | None, dict[str, Any] | None, int]] = []

    for i, o in enumerate(opts):
        otype = int_or(o, "type", 0)
        area = o.get("area")
        in_play_area = o.get("inPlayArea")
        o_index = o.get("index")
        in_play_index = o.get("inPlayIndex")
        p_idx = o.get("playerIndex")
        if p_idx is None:
            p_idx = me

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

        group_counts[(otype, src_cid)] += 1
        resolved_opts.append((src_cid, tgt_cid, src_poke, tgt_poke, aid))

    for i, o in enumerate(opts):
        v = opt_feats[i]
        otype = int_or(o, "type", 0)
        if 0 <= otype < 17:
            v[otype] = 1.0

        area = o.get("area")
        in_play_area = o.get("inPlayArea")
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

        src_cid, tgt_cid, src_poke, tgt_poke, aid = resolved_opts[i]

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
            v[31] = 1.0  # POKEMON
            hp_p = float(int_or(src_poke, "maxHp", 0))
            v[38] = min(hp_p / 340.0, 1.5)
            v[39] = 1.0 if hp_p >= 210.0 else 0.0
            v[40] = 1.0 if otype == 10 else 0.0

        # Target Pokémon state features (41..44)
        tgt_en_cnt = 0
        if isinstance(tgt_poke, dict):
            _, thp, _, tdmg, ten, _, tex, _ = _pokemon_summary(tgt_poke, card_db)
            v[41] = thp
            v[42] = tdmg
            v[43] = ten
            v[44] = 1.0 if tgt_poke.get("appearThisTurn") else 0.0
            tgt_en_cnt = len(tgt_poke.get("energies") or [])
            v[63] = tex

        # Attack features (45..47)
        eff_dmg = 0.0
        atk_cost = 0
        if aid > 0 and aid in atk_db:
            ameta = atk_db[aid]
            dmg = float(getattr(ameta, "damage", 0) or 0)
            if dmg > 0 and opp_act_weak is not None and my_act_etype is not None and opp_act_weak == my_act_etype:
                eff_dmg = dmg * 2.0
            else:
                eff_dmg = dmg
            atk_cost = len(getattr(ameta, "energies", []) or [])
            v[45] = min(eff_dmg / 300.0, 2.0)
            v[46] = min(float(atk_cost) / 5.0, 1.5)
            v[47] = 1.0 if (opp_act_hp > 0 and eff_dmg >= opp_act_hp) else 0.0
        elif otype == 13 and aid > 0:
            eff_dmg = 150.0
            atk_cost = 2
            v[45] = 0.5
            v[46] = 0.4

        # --- Phase 3 Enriched Option Features (48..63) ---
        tmeta = card_db.get(tgt_cid)
        v[48] = _pokemon_stage_norm(src_poke, cmeta)
        v[49] = _pokemon_stage_norm(tgt_poke, tmeta)
        v[50] = 1.0 if (in_play_area == 4 or area == 4 or otype == 13) else 0.0
        v[51] = 1.0 if (in_play_area == 5 or area == 5) else 0.0

        src_is_energy = bool(
            (cmeta is not None and int(getattr(cmeta, "cardType", -1)) in (5, 6)) or otype in (5, 6)
        )
        v[52] = 1.0 if src_is_energy else 0.0

        if tmeta is not None and int(getattr(tmeta, "cardType", -1)) == 0:
            max_tgt_cost = 0
            needed_etypes: set[int] = set()
            for taid in getattr(tmeta, "attacks", ()) or ():
                tameta = atk_db.get(int(taid))
                if tameta is None:
                    continue
                reqs = list(getattr(tameta, "energies", ()) or ())
                max_tgt_cost = max(max_tgt_cost, len(reqs))
                for r in reqs:
                    needed_etypes.add(int(r))
            if src_is_energy and cmeta is not None:
                src_et = int(getattr(cmeta, "energyType", 0) or 0)
                if src_et in needed_etypes or 0 in needed_etypes or src_et == 10:
                    v[53] = 1.0
            if isinstance(tgt_poke, dict) and max_tgt_cost > 0:
                deficit = max(0, max_tgt_cost - tgt_en_cnt)
                v[54] = min(deficit / 4.0, 1.0)

        if otype == 13:
            v[55] = min(eff_dmg / max(opp_act_hp, 1.0), 2.0) if opp_act_hp > 0 else 0.0
            v[56] = min(float(my_act_en_cnt) / max(float(atk_cost), 1.0), 2.0)

        v[57] = 1.0 if max_c > 1 else 0.0
        v[58] = float(min_c) / max(n_opts, 1)
        v[59] = float(max_c) / max(n_opts, 1)
        v[60] = min(hand_cid_counts.get(src_cid, 0) / 4.0, 1.0) if src_cid > 0 else 0.0
        v[61] = min(my_disc_counts.get(src_cid, 0) / 4.0, 1.0) if src_cid > 0 else 0.0
        v[62] = float(group_counts.get((otype, src_cid), 1)) / max(n_opts, 1)

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

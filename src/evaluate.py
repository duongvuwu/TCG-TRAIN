"""Offline test evaluation and live CABT simulator battle evaluation."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import torch

from src import DATASET_REF, DATASET_VERSION
from src.features import get_attack_db, int_or
from src.train import (
    evaluate_packed_split,
    get_git_commit,
    load_default_deck,
    load_policy_from_checkpoint,
)


def make_random_baseline_agent(seed: int = 42, default_deck: list[int] | None = None) -> Callable[[dict[str, Any]], list[int]]:
    """Official CABT documentation random-choice baseline agent with seeded RNG."""
    rng = random.Random(seed)
    deck = list(default_deck) if default_deck is not None else load_default_deck("deck.csv")

    def _agent(obs_dict: dict[str, Any]) -> list[int]:
        sel = obs_dict.get("select") if isinstance(obs_dict, dict) else None
        if sel is None or obs_dict.get("current") is None:
            return list(deck)
        opts = sel.get("option") or []
        n = len(opts)
        raw_min = int_or(sel, "minCount", 0)
        raw_max = int_or(sel, "maxCount", 1)
        if n == 0 or raw_max <= 0:
            return []
        min_c = max(0, min(raw_min, n))
        max_c = max(min_c, min(raw_max, n))

        cur = obs_dict.get("current") or {}
        if int_or(cur, "turnActionCount", 0) >= 25 and int_or(sel, "type", -1) == 0 and max_c == 1:
            term = [i for i, o in enumerate(opts) if int_or(o, "type", -1) in (13, 14)]
            if term:
                return [rng.choice(term)]
        return sorted(rng.sample(range(n), max_c))

    return _agent


def make_heuristic_baseline_agent(seed: int = 42, default_deck: list[int] | None = None) -> Callable[[dict[str, Any]], list[int]]:
    """Simple rule-based baseline agent that prioritizes attacks, evolutions, and energy."""
    rng = random.Random(seed)
    deck = list(default_deck) if default_deck is not None else load_default_deck("deck.csv")
    atk_db = get_attack_db()

    def _agent(obs_dict: dict[str, Any]) -> list[int]:
        sel = obs_dict.get("select") if isinstance(obs_dict, dict) else None
        cur = obs_dict.get("current") if isinstance(obs_dict, dict) else None
        if sel is None or cur is None:
            return list(deck)
        opts = sel.get("option") or []
        n = len(opts)
        raw_min = int_or(sel, "minCount", 0)
        raw_max = int_or(sel, "maxCount", 1)
        if n == 0 or raw_max <= 0:
            return []
        min_c = max(0, min(raw_min, n))
        max_c = max(min_c, min(raw_max, n))
        if n == 1 and min_c == 1:
            return [0]

        stype = int_or(sel, "type", 0)
        if int_or(cur, "turnActionCount", 0) >= 20 and stype == 0 and max_c == 1:
            term = [i for i, o in enumerate(opts) if int_or(o, "type", -1) in (13, 14)]
            if term:
                return [term[0]]

        if stype == 0 and max_c == 1:
            best_idx = 0
            best_score = -1e9
            for i, o in enumerate(opts):
                ot = int_or(o, "type", 0)
                if ot == 13:  # ATTACK
                    aid = int_or(o, "attackId", 0)
                    dmg = float(getattr(atk_db.get(aid), "damage", 0) or 0) if aid in atk_db else 10.0
                    score = 1000.0 + dmg
                elif ot == 9:  # EVOLVE
                    score = 500.0
                elif ot == 8:  # ATTACH
                    score = 400.0 + (50.0 if o.get("inPlayArea") == 4 else 0.0)
                elif ot == 7:  # PLAY
                    score = 300.0
                elif ot == 10:  # ABILITY
                    score = 250.0
                elif ot == 14:  # END
                    score = 0.0
                elif ot == 12:  # RETREAT
                    score = -100.0
                else:
                    score = 10.0
                if score > best_score:
                    best_score = score
                    best_idx = i
            return [best_idx]

        return sorted(rng.sample(range(n), max_c))

    return _agent


def play_single_match(
    agent0: Callable[[dict[str, Any]], list[int]],
    agent1: Callable[[dict[str, Any]], list[int]],
    deck0: list[int] | None = None,
    deck1: list[int] | None = None,
    max_steps: int = 3000,
) -> dict[str, Any]:
    """Run one complete match in the CABT simulator (`cg.game`) and record stats.

    Fixes BUG-1: When `deck0` and `deck1` are explicitly provided, they are used
    directly for `battle_start(deck0, deck1)` (never overwritten by `agents[s](init_obs)`),
    while `agents[s](init_obs)` is still verified for valid 60-card deck format.
    """
    import cg.game

    agents = [agent0, agent1]
    explicit_decks = [
        list(deck0) if deck0 is not None else None,
        list(deck1) if deck1 is not None else None,
    ]
    decks: list[list[int]] = [[], []]
    invalid_actions = [0, 0]
    runtime_errors = [0, 0]

    for s in (0, 1):
        init_obs = {"select": None, "current": None, "logs": []}
        try:
            dk = agents[s](init_obs)
            valid_init_dk = isinstance(dk, list) and len(dk) == 60 and all(isinstance(c, int) for c in dk)
        except Exception:
            runtime_errors[s] += 1
            dk = []
            valid_init_dk = False

        if not valid_init_dk:
            invalid_actions[s] += 1

        if explicit_decks[s] is not None:
            decks[s] = explicit_decks[s]  # Preserve requested mirror-match deck
        elif valid_init_dk:
            decks[s] = list(dk)
        else:
            decks[s] = load_default_deck("deck.csv")

    t0 = time.perf_counter()
    obs, start_data = cg.game.battle_start(decks[0], decks[1])
    if obs is None:
        return {
            "winner": -1,
            "steps": 0,
            "turns": 0,
            "decks_equal": (decks[0] == decks[1]),
            "invalid_actions": [
                invalid_actions[0] + (1 if start_data.errorPlayer == 0 else 0),
                invalid_actions[1] + (1 if start_data.errorPlayer == 1 else 0),
            ],
            "runtime_errors": runtime_errors,
            "error": f"battle_start failed: player={start_data.errorPlayer}, type={start_data.errorType}",
        }

    steps = 0
    decision_counts = [0, 0]
    decision_time_ms = [0.0, 0.0]

    try:
        while int_or(obs["current"], "result", -1) == -1 and steps < max_steps:
            who = int_or(obs["current"], "yourIndex", 0)
            sel = obs["select"]
            opts = sel.get("option") or []
            min_c = int_or(sel, "minCount", 0)
            max_c = int_or(sel, "maxCount", 1)

            ts = time.perf_counter()
            try:
                act = agents[who](obs)
            except Exception:
                runtime_errors[who] += 1
                act = list(range(min(max(0, min_c), len(opts))))
            dt_ms = (time.perf_counter() - ts) * 1000.0
            decision_counts[who] += 1
            decision_time_ms[who] += dt_ms

            is_valid = (
                isinstance(act, list)
                and all(isinstance(x, int) and 0 <= x < len(opts) for x in act)
                and (min_c <= len(act) <= min(max(0, max_c), len(opts)))
                and len(set(act)) == len(act)
            )
            if not is_valid:
                invalid_actions[who] += 1
                act = list(range(min(max(0, min_c), len(opts))))

            try:
                obs = cg.game.battle_select(act)
            except Exception:
                invalid_actions[who] += 1
                fallback = list(range(min(max(0, min_c), len(opts))))
                obs = cg.game.battle_select(fallback)

            steps += 1

        winner = int_or(obs["current"], "result", -1)
        turns = int_or(obs["current"], "turn", 0)
    finally:
        cg.game.battle_finish()

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "winner": winner,
        "steps": steps,
        "turns": turns,
        "decks_equal": (decks[0] == decks[1]),
        "elapsed_ms": round(elapsed_ms, 2),
        "invalid_actions": invalid_actions,
        "runtime_errors": runtime_errors,
        "decision_counts": decision_counts,
        "avg_latency_ms": [
            round(decision_time_ms[0] / max(decision_counts[0], 1), 4),
            round(decision_time_ms[1] / max(decision_counts[1], 1), 4),
        ],
    }


def evaluate_in_simulator(
    checkpoint_path: str | Path = "checkpoints/best_model.pt",
    num_games_per_opponent: int = 40,
    seed: int = 42,
    out_file: str | Path = "checkpoints/eval_results.json",
) -> dict[str, Any]:
    """Evaluate offline test accuracy and online simulator win-rate vs baselines on true mirror matches."""
    model, default_deck, ckpt = load_policy_from_checkpoint(checkpoint_path, device="cpu")

    test_path = Path("data/processed/test.pt")
    val_path = Path("data/processed/val.pt")
    offline_test = (
        evaluate_packed_split(model, torch.load(test_path, weights_only=False), torch.device("cpu"))
        if test_path.exists()
        else {}
    )
    offline_val = (
        evaluate_packed_split(model, torch.load(val_path, weights_only=False), torch.device("cpu"))
        if val_path.exists()
        else {}
    )

    deck_pool: list[tuple[str, list[int]]] = [("default_deck", default_deck)]
    for dpath in [Path("decks/dragapult_deck.csv"), Path("decks/starmie_deck.csv")]:
        if dpath.exists():
            dk = load_default_deck(dpath)
            if len(dk) == 60:
                deck_pool.append((dpath.stem, dk))

    opponents = {
        "cabt_random_baseline": lambda s, dk: make_random_baseline_agent(seed=s, default_deck=dk),
        "heuristic_baseline": lambda s, dk: make_heuristic_baseline_agent(seed=s, default_deck=dk),
    }

    match_summaries: dict[str, Any] = {}
    for opp_name, opp_factory in opponents.items():
        wins = draws = losses = timeouts = 0
        p0_wdl = [0, 0, 0]
        p1_wdl = [0, 0, 0]
        by_deck: dict[str, dict[str, int]] = {
            dname: {"games": 0, "wins": 0, "draws": 0, "losses": 0, "timeouts": 0}
            for dname, _ in deck_pool
        }
        model_invalid = 0
        model_errors = 0
        opp_invalid = 0
        total_turns = 0
        total_steps = 0
        latencies: list[float] = []

        for g in range(num_games_per_opponent):
            deck_name, dk = deck_pool[(g // 2) % len(deck_pool)]
            opp_agent = opp_factory(seed + 1000 + g, dk)

            # Fix BUG-1: policy_agent uses the exact mirror deck `dk` for this game
            def policy_agent(obs_dict: dict[str, Any], _dk: list[int] = dk) -> list[int]:
                return model.select_action(obs_dict, default_deck=_dk, device="cpu")

            model_seat = g % 2  # Alternate Seat 0 and Seat 1
            if model_seat == 0:
                res = play_single_match(policy_agent, opp_agent, dk, dk)
            else:
                res = play_single_match(opp_agent, policy_agent, dk, dk)

            assert res["decks_equal"], f"Mirror match deck mismatch on {deck_name}!"
            by_deck[deck_name]["games"] += 1

            w = res["winner"]
            if w == model_seat:
                wins += 1
                by_deck[deck_name]["wins"] += 1
                (p0_wdl if model_seat == 0 else p1_wdl)[0] += 1
            elif w == (1 - model_seat):
                losses += 1
                by_deck[deck_name]["losses"] += 1
                (p0_wdl if model_seat == 0 else p1_wdl)[2] += 1
            elif w == 2:
                draws += 1
                by_deck[deck_name]["draws"] += 1
                (p0_wdl if model_seat == 0 else p1_wdl)[1] += 1
            else:
                timeouts += 1
                by_deck[deck_name]["timeouts"] += 1
                (p0_wdl if model_seat == 0 else p1_wdl)[1] += 1

            model_invalid += res["invalid_actions"][model_seat]
            model_errors += res["runtime_errors"][model_seat]
            opp_invalid += res["invalid_actions"][1 - model_seat]
            total_turns += res["turns"]
            total_steps += res["steps"]
            latencies.append(res["avg_latency_ms"][model_seat])

        win_rate = wins / max(num_games_per_opponent, 1)
        score_rate = (wins + 0.5 * draws) / max(num_games_per_opponent, 1)
        match_summaries[opp_name] = {
            "games": num_games_per_opponent,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "timeouts": timeouts,
            "win_rate": round(win_rate, 4),
            "score_rate": round(score_rate, 4),
            "as_seat0_wdl": p0_wdl,
            "as_seat1_wdl": p1_wdl,
            "by_deck": by_deck,
            "model_invalid_actions": model_invalid,
            "model_runtime_errors": model_errors,
            "opponent_invalid_actions": opp_invalid,
            "avg_turns": round(total_turns / max(num_games_per_opponent, 1), 2),
            "avg_steps": round(total_steps / max(num_games_per_opponent, 1), 2),
            "avg_decision_latency_ms": round(sum(latencies) / max(len(latencies), 1), 4),
        }

    results = {
        "dataset_ref": DATASET_REF,
        "dataset_version": DATASET_VERSION,
        "git_commit": get_git_commit(),
        "seed": seed,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": ckpt.get("epoch"),
        "offline_val_metrics": offline_val,
        "offline_test_metrics": offline_test,
        "simulator_matches": match_summaries,
    }

    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results

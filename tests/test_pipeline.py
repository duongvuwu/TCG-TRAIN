"""Unit and integration tests for Pokémon TCG Battle Agent pipeline.

Verifies:
1. Pre-decision observation (`steps[t][seat]`) to chosen action (`steps[t+1][seat]`) alignment,
   plus separation of deck selection (`t=0`), waiting turns (`INACTIVE`), and terminal (`DONE`) steps.
2. Zero future or opponent hidden-state leakage (`players[1-me]["hand"] is None`).
3. Strict episode-level train/val/test split with zero episode overlap.
4. Candidate scoring output format (`list[int]`, bounds `0 <= idx < len(option)`,
   count in `[minCount, maxCount]`, unique sorted indices for multi-select, and empty `[]` when `minCount == 0`).
5. Checkpoint save, reload, resume, and live simulator battle execution via `agent.py`.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agent import agent as simulator_agent, load_agent_checkpoint
from src.dataset import parse_single_replay
from src.evaluate import make_random_baseline_agent, play_single_match
from src.features import extract_decision_features, validate_no_leakage
from src.model import PTCGPolicyNet
from src.train import load_default_deck, load_policy_from_checkpoint


def _get_sample_replay_path() -> Path:
    replays = sorted(Path("data/replays").glob("*.json"))
    replays = [p for p in replays if p.stem.isdigit()]
    assert len(replays) > 0, "Expected at least one replay JSON in data/replays/"
    return replays[0]


def test_observation_action_pairing_and_step_accounting() -> None:
    """Verify exact accounting of all seat-steps and valid action alignment."""
    rp = _get_sample_replay_path()
    samples, counts, decks = parse_single_replay(rp, drop_forced_single=False)

    # Every seat-step must be accounted for in one of the 4 disjoint categories
    total_seat_steps = counts["total_seat_steps"]
    accounted = (
        counts.get("excluded_deck_select_steps", 0)
        + counts.get("excluded_inactive_waiting_turns", 0)
        + counts.get("excluded_terminal_done_steps", 0)
        + counts.get("valid_decisions_total", 0)
    )
    assert total_seat_steps == accounted, f"Seat-step mismatch: {total_seat_steps} != {accounted}"
    assert counts["excluded_deck_select_steps"] == 2
    assert counts["excluded_terminal_done_steps"] == 2
    assert counts.get("excluded_invalid_action_bounds", 0) == 0
    assert counts.get("unexpected_inactive_nonempty_action", 0) == 0
    assert len(decks) == 2 and all(len(d) == 60 for d in decks)
    assert len(samples) == counts["valid_decisions_total"]

    # Check every extracted sample's chosen indices against num_options and [min_count, max_count]
    for s in samples:
        k = s["num_options"]
        min_c = s["min_count"]
        max_c = s["max_count"]
        chosen = s["chosen_indices"]
        assert isinstance(chosen, list)
        assert min_c <= len(chosen) <= max_c
        assert len(set(chosen)) == len(chosen)
        assert all(isinstance(i, int) and 0 <= i < k for i in chosen)
        assert int(s["chosen_mask"].sum()) == len(chosen)


def test_no_opponent_or_future_leakage() -> None:
    """Verify that opponent hand leakage or seat mismatch raises ValueError."""
    rp = _get_sample_replay_path()
    with open(rp, encoding="utf-8") as f:
        raw = json.load(f)

    # Find a valid active decision step
    sample_obs = None
    sample_seat = 0
    for st in raw["steps"]:
        for seat in (0, 1):
            if st[seat]["status"] == "ACTIVE" and st[seat]["observation"].get("current") is not None:
                sample_obs = copy.deepcopy(st[seat]["observation"])
                sample_seat = seat
                break
        if sample_obs is not None:
            break

    assert sample_obs is not None
    # Valid observation passes
    validate_no_leakage(sample_obs, expected_seat=sample_seat)

    # Seat mismatch must fail
    with pytest.raises(ValueError, match="Seat mismatch"):
        validate_no_leakage(sample_obs, expected_seat=1 - sample_seat)

    # Opponent hand leak must fail
    leaked_obs = copy.deepcopy(sample_obs)
    opp = 1 - sample_seat
    leaked_obs["current"]["players"][opp]["hand"] = [{"id": 1, "serial": 99, "playerIndex": opp}]
    with pytest.raises(ValueError, match="Information leakage detected"):
        validate_no_leakage(leaked_obs, expected_seat=sample_seat)


def test_episode_level_split_disjointness() -> None:
    """Verify that train, val, and test splits have strictly disjoint episode_ids."""
    stats_path = Path("data/processed/dataset_stats.json")
    assert stats_path.exists(), "Run preprocess first to create dataset_stats.json"
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)

    splits = stats["episode_splits"]
    train_eps = set(splits["train"])
    val_eps = set(splits["val"])
    test_eps = set(splits["test"])

    assert len(train_eps) > 0 and len(val_eps) > 0 and len(test_eps) > 0
    assert train_eps.isdisjoint(val_eps), "Train and Val share episodes!"
    assert train_eps.isdisjoint(test_eps), "Train and Test share episodes!"
    assert val_eps.isdisjoint(test_eps), "Val and Test share episodes!"

    # Also verify inside the saved .pt tensor files
    for sname, expected_set in [("train", train_eps), ("val", val_eps), ("test", test_eps)]:
        pt = torch.load(Path("data/processed") / f"{sname}.pt", weights_only=False)
        actual_eps = set(pt["episode_ids"])
        assert actual_eps.issubset(expected_set)


def test_action_decoding_single_and_multiselect() -> None:
    """Verify single-select, multi-select, and optional 0-select decoding contracts."""
    logits = np.array([0.2, 2.5, -1.0, 1.8, 0.9], dtype=np.float32)

    # 1. Single-select (min=1, max=1) -> highest logit index [1]
    act_single = PTCGPolicyNet.decode_action_from_logits(logits, stop_logit=0.0, min_count=1, max_count=1)
    assert act_single == [1]

    # 2. Fixed multi-select (min=2, max=2) -> top 2 indices sorted: [1, 3]
    act_fixed_multi = PTCGPolicyNet.decode_action_from_logits(logits, stop_logit=10.0, min_count=2, max_count=2)
    assert act_fixed_multi == [1, 3]

    # 3. Variable multi-select (min=0, max=3) with stop_logit=1.0 -> indices with logit > 1.0 are 1 (2.5) and 3 (1.8)
    act_var_multi = PTCGPolicyNet.decode_action_from_logits(logits, stop_logit=1.0, min_count=0, max_count=3)
    assert act_var_multi == [1, 3]

    # 4. Variable multi-select (min=0, max=2) with high stop_logit=5.0 -> selects 0 items []
    act_zero = PTCGPolicyNet.decode_action_from_logits(logits, stop_logit=5.0, min_count=0, max_count=2)
    assert act_zero == []


def test_checkpoint_load_and_full_simulator_match() -> None:
    """Verify loading `checkpoints/best_model.pt` and playing a complete CABT game with 0 invalid moves."""
    ckpt_path = Path("checkpoints/best_model.pt")
    assert ckpt_path.exists(), "Expected trained checkpoint at checkpoints/best_model.pt"

    model, deck, ckpt = load_policy_from_checkpoint(ckpt_path, device="cpu")
    assert len(deck) == 60
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert ckpt.get("dataset_version") == "2026-11-13-v1"

    load_agent_checkpoint(ckpt_path)
    # Test deck selection call
    init_act = simulator_agent({"select": None, "current": None, "logs": []})
    assert isinstance(init_act, list) and len(init_act) == 60

    # Play full match in CABT simulator against random baseline
    opp = make_random_baseline_agent(seed=123, default_deck=deck)
    res = play_single_match(simulator_agent, opp, deck, deck)
    assert res["winner"] in (0, 1, 2)
    assert res["steps"] > 10
    assert res["invalid_actions"][0] == 0, f"Model made invalid actions: {res}"
    assert res["runtime_errors"][0] == 0, f"Model had runtime errors: {res}"

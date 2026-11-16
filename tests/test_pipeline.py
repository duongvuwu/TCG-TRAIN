"""Unit and integration tests for Pokémon TCG Battle Agent pipeline.

Verifies:
1. Pre-decision observation (`steps[t][seat]`) to chosen action (`steps[t+1][seat]`) alignment,
   plus separation of deck selection (`t=0`), waiting turns (`INACTIVE`), and terminal (`DONE`) steps.
2. Zero future or opponent hidden-state leakage (`players[1-me]["hand"] is None`).
3. Strict episode-level train/val/test split with zero episode overlap.
4. Candidate scoring output format (`list[int]`, bounds `0 <= idx < len(option)`,
   count in `[minCount, maxCount]`, unique sorted indices for multi-select, and empty `[]` when `minCount == 0`).
5. Regression tests for BUG-1 through BUG-7 (mirror-match deck preservation, falsy-zero `maxCount == 0`
   & `SelectType.MAIN == 0` anti-loop guard, high card/attack IDs `1331..1748`, `minCount == 0` empty-choice
   loss gradient, monotonic LR on `--resume`, and `AreaType.STADIUM (7)` encoding).
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
from src.features import (
    MAX_ATTACK_ID,
    MAX_CARD_ID,
    _clamp_attack_id,
    _clamp_card_id,
    extract_decision_features,
    validate_no_leakage,
)
from src.model import PTCGPolicyNet
from src.train import compute_policy_loss, load_default_deck, load_policy_from_checkpoint


def _get_sample_replay_path() -> Path:
    replays = sorted(Path("data/replays").glob("*.json"))
    replays = [p for p in replays if p.stem.isdigit()]
    assert len(replays) > 0, "Expected at least one replay JSON in data/replays/"
    return replays[0]


def _get_sample_active_obs() -> tuple[dict, int]:
    rp = _get_sample_replay_path()
    with open(rp, encoding="utf-8") as f:
        raw = json.load(f)
    for st in raw["steps"]:
        for seat in (0, 1):
            if st[seat]["status"] == "ACTIVE" and st[seat]["observation"].get("current") is not None:
                return copy.deepcopy(st[seat]["observation"]), seat
    raise AssertionError("No active observation found")


def test_observation_action_pairing_and_step_accounting() -> None:
    """Verify exact accounting of all seat-steps and valid action alignment."""
    rp = _get_sample_replay_path()
    samples, counts, decks = parse_single_replay(rp, drop_forced_single=False)

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
    sample_obs, sample_seat = _get_sample_active_obs()
    validate_no_leakage(sample_obs, expected_seat=sample_seat)

    with pytest.raises(ValueError, match="Seat mismatch"):
        validate_no_leakage(sample_obs, expected_seat=1 - sample_seat)

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
    assert train_eps.isdisjoint(val_eps)
    assert train_eps.isdisjoint(test_eps)
    assert val_eps.isdisjoint(test_eps)

    for sname, expected_set in [("train", train_eps), ("val", val_eps), ("test", test_eps)]:
        pt = torch.load(Path("data/processed") / f"{sname}.pt", weights_only=False)
        actual_eps = set(pt["episode_ids"])
        assert actual_eps.issubset(expected_set)


def test_action_decoding_single_and_multiselect() -> None:
    """Verify single-select, multi-select, and optional 0-select decoding contracts."""
    logits = np.array([0.2, 2.5, -1.0, 1.8, 0.9], dtype=np.float32)

    assert PTCGPolicyNet.decode_action_from_logits(logits, stop_logit=0.0, min_count=1, max_count=1) == [1]
    assert PTCGPolicyNet.decode_action_from_logits(logits, stop_logit=10.0, min_count=2, max_count=2) == [1, 3]
    assert PTCGPolicyNet.decode_action_from_logits(logits, stop_logit=1.0, min_count=0, max_count=3) == [1, 3]
    assert PTCGPolicyNet.decode_action_from_logits(logits, stop_logit=5.0, min_count=0, max_count=2) == []
    assert PTCGPolicyNet.decode_action_from_logits(logits, stop_logit=-5.0, min_count=0, max_count=0) == []


def test_bug1_mirror_match_deck_preservation() -> None:
    """Verify BUG-1 fix: play_single_match never overwrites explicit mirror decks."""
    starmie_deck = load_default_deck("decks/starmie_deck.csv")
    default_deck = load_default_deck("deck.csv")
    assert starmie_deck != default_deck

    # Even if agent0 returns default_deck on init_obs, explicit deck0=starmie_deck must be used
    a0 = make_random_baseline_agent(seed=1, default_deck=default_deck)
    a1 = make_random_baseline_agent(seed=2, default_deck=starmie_deck)
    res = play_single_match(a0, a1, starmie_deck, starmie_deck, max_steps=100)
    assert res["decks_equal"] is True
    assert res["invalid_actions"] == [0, 0]


def test_bug2_falsy_zero_maxcount_zero_and_antiloop_guard() -> None:
    """Verify BUG-2 fix: maxCount==0 returns [] and turnActionCount>=20 restricts to ATTACK/END."""
    sample_obs, _ = _get_sample_active_obs()
    load_agent_checkpoint("checkpoints/best_model.pt")
    model, deck, _ = load_policy_from_checkpoint("checkpoints/best_model.pt", device="cpu")

    # 1. maxCount == 0 with non-empty options must return []
    obs_zero = copy.deepcopy(sample_obs)
    obs_zero["select"]["minCount"] = 0
    obs_zero["select"]["maxCount"] = 0
    obs_zero["select"]["option"] = [{"type": 3, "area": 2, "index": 0}, {"type": 3, "area": 2, "index": 1}]
    assert simulator_agent(obs_zero) == []
    assert model.select_action(obs_zero, default_deck=deck) == []

    # 2. Anti-loop guard when turnActionCount >= 20 and select.type == 0 (SelectType.MAIN)
    obs_loop = copy.deepcopy(sample_obs)
    obs_loop["current"]["turn"] = 5
    obs_loop["current"]["turnActionCount"] = 25
    obs_loop["select"]["type"] = 0  # SelectType.MAIN (0 is falsy!)
    obs_loop["select"]["context"] = 0
    obs_loop["select"]["minCount"] = 1
    obs_loop["select"]["maxCount"] = 1
    obs_loop["select"]["option"] = [
        {"type": 10, "area": 5, "index": 0},  # ABILITY (index 0)
        {"type": 10, "area": 5, "index": 1},  # ABILITY (index 1)
        {"type": 14},                         # END (index 2)
    ]
    chosen = model.select_action(obs_loop, default_deck=deck)
    assert chosen == [2], f"Expected anti-loop guard to pick END [2], got {chosen}"


def test_bug3_high_card_and_attack_ids_not_clamped() -> None:
    """Verify BUG-3 fix: newer card IDs (1331..1413) and attack IDs (1641..1748) do not collide."""
    assert MAX_CARD_ID >= 2048
    assert MAX_ATTACK_ID >= 2048

    high_cards = [1331, 1387, 1406, 1413]
    clamped_cards = [_clamp_card_id(cid) for cid in high_cards]
    assert clamped_cards == high_cards
    assert len(set(clamped_cards)) == len(high_cards)

    high_atks = [1641, 1642, 1715, 1716, 1747, 1748]
    clamped_atks = [_clamp_attack_id(aid) for aid in high_atks]
    assert clamped_atks == high_atks
    assert len(set(clamped_atks)) == len(high_atks)


def test_bug4_and_bug5_zero_select_loss_and_resume_lr_monotonic() -> None:
    """Verify BUG-4 (empty choice [] trains stop_logits upward) and BUG-5 (resume LR does not bounce)."""
    opt_logits = torch.tensor([0.5, 1.0, -0.2], requires_grad=True)
    stop_logits = torch.tensor([0.0], requires_grad=True)
    num_options = torch.tensor([3], dtype=torch.long)
    chosen_mask = torch.tensor([0.0, 0.0, 0.0])  # Player chose [] when minCount == 0
    min_count = torch.tensor([0], dtype=torch.long)
    max_count = torch.tensor([2], dtype=torch.long)

    loss = compute_policy_loss(opt_logits, stop_logits, num_options, chosen_mask, min_count, max_count)
    loss.backward()
    # Gradient wrt stop_logits must be negative (so gradient descent increases stop_logits above opt_logits)
    assert stop_logits.grad is not None and float(stop_logits.grad.item()) < 0.0
    assert opt_logits.grad is not None and all(float(g) > 0.0 for g in opt_logits.grad)

    # Check train_summary.json learning rate curve is non-increasing across resume epochs
    summary_path = Path("checkpoints/train_summary.json")
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            hist = json.load(f)["history"]
        lrs = [row["lr"] for row in hist]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-9, f"LR bounced upward at step {i}: {lrs[i-1]} -> {lrs[i]}"


def test_bug6_stadium_area_and_card_type_encoding() -> None:
    """Verify BUG-6 fix: AreaType.STADIUM (7) and CardType.STADIUM (4) are encoded."""
    sample_obs, _ = _get_sample_active_obs()
    obs_stad = copy.deepcopy(sample_obs)
    obs_stad["current"]["stadium"] = [{"id": 1256, "serial": 77, "playerIndex": 0}]
    obs_stad["select"]["option"] = [{"type": 10, "area": 7, "index": 0}]
    feats = extract_decision_features(obs_stad)
    assert feats["opt_feats"][0, 23] == 1.0, "AreaType.STADIUM (7) must set slot 23"
    assert feats["opt_card_ids"][0] == 1256


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
    init_act = simulator_agent({"select": None, "current": None, "logs": []})
    assert isinstance(init_act, list) and len(init_act) == 60

    opp = make_random_baseline_agent(seed=123, default_deck=deck)
    res = play_single_match(simulator_agent, opp, deck, deck)
    assert res["winner"] in (0, 1, 2)
    assert res["steps"] > 10
    assert res["decks_equal"] is True
    assert res["invalid_actions"][0] == 0, f"Model made invalid actions: {res}"
    assert res["runtime_errors"][0] == 0, f"Model had runtime errors: {res}"

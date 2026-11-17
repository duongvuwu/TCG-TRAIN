"""Unit and integration tests for Pokémon TCG Battle Agent Phase 1-3 pipeline.

Verifies:
1. Pre-decision observation (`steps[t][seat]`) to chosen action (`steps[t+1][seat]`) alignment,
   plus separation of deck selection (`t=0`), waiting turns (`INACTIVE`), and terminal (`DONE`) steps.
2. Zero future or opponent hidden-state leakage (`players[1-me]["hand"] is None`).
3. Sharded tensor storage & strict episode-level train/val/test split with zero episode overlap.
4. Static fallback metadata (`cardId >= 1268`, `attackId >= 1557`) and replay metadata inference.
5. Value head, conditional multi-select decoding, and Phase 2 checkpoint backward compatibility.
6. PPO + KL-to-BC RL fine-tuning step & resume in CABT simulator.
7. Cross-deck matchup matrix, CPU latency (`mean`, `p99`), and `agent.py` Kaggle entrypoint.
8. Regression tests for BUG-1 through BUG-6.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from agent import agent as simulator_agent, load_agent_checkpoint
from src.dataset import (
    get_split_num_samples,
    get_split_shard_paths,
    iter_split_shards,
    parse_single_replay,
    preprocess_replays,
    select_manifest_indices,
)
from src.evaluate import evaluate_cross_deck_matrix, make_random_baseline_agent, play_single_match
from src.features import (
    LEGACY_OPT_DIM,
    LEGACY_STATE_DIM,
    MAX_ATTACK_ID,
    MAX_CARD_ID,
    OPT_DIM,
    STATE_DIM,
    _clamp_attack_id,
    _clamp_card_id,
    extract_decision_features,
    get_attack_db,
    get_card_db,
    infer_fallback_metadata_from_replays,
    validate_no_leakage,
)
from src.model import PTCGPolicyNet
from src.train import (
    compute_policy_loss,
    load_default_deck,
    load_policy_from_checkpoint,
    train_rl_policy,
)


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
    """Verify that train, val, and test shards have strictly disjoint episode_ids."""
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
        shard_paths = get_split_shard_paths("data/processed", sname)
        assert len(shard_paths) >= 1
        seen_eps: set[str] = set()
        for sp in shard_paths:
            pt = torch.load(sp, weights_only=False)
            ep_set = set(pt["episode_ids"])
            assert ep_set.issubset(expected_set)
            seen_eps.update(ep_set)
        assert len(seen_eps) > 0


def test_sharded_preprocess_and_streaming_loader(tmp_path: Path) -> None:
    """Verify bounded sharded preprocessing, multi-shard creation, and streaming iteration."""
    replays = sorted([p for p in Path("data/replays").glob("*.json") if p.stem.isdigit()])[:5]
    assert len(replays) >= 4
    rdir = tmp_path / "replays"
    pdir = tmp_path / "processed"
    rdir.mkdir()
    for rp in replays:
        shutil.copy2(rp, rdir / rp.name)

    summary = preprocess_replays(
        replay_dir=rdir,
        out_dir=pdir,
        seed=42,
        shard_size_samples=80,
        infer_metadata=False,
    )
    assert summary["shard_summaries"]["train"]["num_shards"] >= 2
    train_shards = get_split_shard_paths(pdir, "train")
    assert len(train_shards) >= 2
    assert all("train_shard_" in p.name for p in train_shards)

    streamed_total = 0
    streamed_eps: set[str] = set()
    for _, shard in iter_split_shards(pdir, "train", shuffle=True, rng=np.random.default_rng(7)):
        streamed_total += int(shard["num_samples"])
        streamed_eps.update(shard["episode_ids"])
        assert shard["state_vec"].shape[1] == STATE_DIM
        assert shard["opt_feats"].shape[1] == OPT_DIM

    assert streamed_total == get_split_num_samples(pdir, "train")
    val_eps = set(summary["episode_splits"]["val"])
    test_eps = set(summary["episode_splits"]["test"])
    assert streamed_eps.isdisjoint(val_eps)
    assert streamed_eps.isdisjoint(test_eps)

    # Also verify manifest index selector diversification
    idx = select_manifest_indices(1644, num_top=10, num_strided=10, max_episodes=15)
    assert len(idx) == 15 and idx[:10] == list(range(10)) and idx[-1] <= 1643


def test_fallback_metadata_table_and_inference() -> None:
    """Verify static fallback metadata and replay inference for post-1267 cards & post-1556 attacks."""
    cdb = get_card_db()
    adb = get_attack_db()
    for cid in (1331, 1387, 1406, 1413):
        assert cid in cdb, f"Expected fallback metadata for cardId={cid}"
    for aid in (1641, 1642, 1715, 1716, 1747, 1748):
        assert aid in adb, f"Expected fallback metadata for attackId={aid}"
    assert adb[1748].damage == 200

    inf_c, inf_a = infer_fallback_metadata_from_replays("data/replays", max_files=10)
    assert isinstance(inf_c, dict) and isinstance(inf_a, dict)


def test_action_decoding_single_and_multiselect() -> None:
    """Verify single-select, multi-select, and optional 0-select decoding contracts."""
    logits = np.array([0.2, 2.5, -1.0, 1.8, 0.9], dtype=np.float32)

    assert PTCGPolicyNet.decode_action_from_logits(logits, stop_logit=0.0, min_count=1, max_count=1) == [1]
    assert PTCGPolicyNet.decode_action_from_logits(logits, stop_logit=10.0, min_count=2, max_count=2) == [1, 3]
    assert PTCGPolicyNet.decode_action_from_logits(logits, stop_logit=1.0, min_count=0, max_count=3) == [1, 3]
    assert PTCGPolicyNet.decode_action_from_logits(logits, stop_logit=5.0, min_count=0, max_count=2) == []
    assert PTCGPolicyNet.decode_action_from_logits(logits, stop_logit=-5.0, min_count=0, max_count=0) == []


def test_value_head_and_phase2_checkpoint_compat(tmp_path: Path) -> None:
    """Verify value_head output and strict backward compatibility with Phase 2 checkpoints."""
    sample_obs, _ = _get_sample_active_obs()
    feats = extract_decision_features(sample_obs)

    # 1. Phase 3 model forward_packed_with_value
    m3 = PTCGPolicyNet()
    m3.eval()
    opt_l, stop_l, vals, state_h, opt_ctx = m3.forward_packed_with_value(
        torch.from_numpy(feats["state_vec"]).unsqueeze(0),
        torch.from_numpy(feats["hand_cards"]).unsqueeze(0),
        torch.from_numpy(feats["board_cards"]).unsqueeze(0),
        torch.tensor([feats["select_ctx"]], dtype=torch.long),
        torch.tensor([feats["select_type"]], dtype=torch.long),
        torch.tensor([feats["num_options"]], dtype=torch.long),
        torch.from_numpy(feats["opt_feats"]),
        torch.from_numpy(feats["opt_card_ids"]),
        torch.from_numpy(feats["opt_target_card_ids"]),
        torch.from_numpy(feats["opt_attack_ids"]),
    )
    assert opt_l.shape == (feats["num_options"],)
    assert stop_l.shape == (1,)
    assert vals.shape == (1,) and torch.isfinite(vals).all()
    assert state_h.shape == (1, m3.hidden_dim)
    assert opt_ctx.shape == (feats["num_options"], m3.hidden_dim)

    # 2. Legacy Phase 2 checkpoint (64/48 dims, no set_attn/cond_multi, no value_head keys)
    m2 = PTCGPolicyNet(
        state_dim=LEGACY_STATE_DIM,
        opt_dim=LEGACY_OPT_DIM,
        hidden_dim=128,
        use_set_attn=False,
        use_cond_multi=False,
    )
    legacy_sd = {k: v for k, v in m2.state_dict().items() if not k.startswith("value_head.")}
    legacy_ckpt = tmp_path / "phase2_legacy.pt"
    torch.save(
        {
            "epoch": 21,
            "model_state_dict": legacy_sd,
            "config": {"hidden_dim": 128, "dropout": 0.1},
            "default_deck": load_default_deck("deck.csv"),
        },
        legacy_ckpt,
    )

    loaded, dk, _ = load_policy_from_checkpoint(legacy_ckpt, device="cpu")
    assert loaded.state_dim == LEGACY_STATE_DIM
    assert loaded.opt_dim == LEGACY_OPT_DIM
    assert loaded.use_set_attn is False
    assert loaded.use_cond_multi is False
    act = loaded.select_action(sample_obs, default_deck=dk, device="cpu")
    assert isinstance(act, list)


def test_rl_finetune_step_and_resume(tmp_path: Path) -> None:
    """Verify PPO + KL-to-BC RL fine-tune update and `--resume` in CABT simulator."""
    ckpt_path = Path("checkpoints/best_model.pt")
    assert ckpt_path.exists()
    rl_dir = tmp_path / "rl_ckpts"
    s1 = train_rl_policy(
        bc_checkpoint=ckpt_path,
        checkpoint_dir=rl_dir,
        updates=1,
        episodes_per_update=2,
        ppo_epochs=1,
        mini_batch_size=32,
        eval_games=2,
        seed=42,
        device_str="cpu",
        promote_to_best=False,
    )
    assert Path(s1["last_rl_checkpoint"]).exists()
    assert len(s1["history"]) == 1

    s2 = train_rl_policy(
        bc_checkpoint=ckpt_path,
        checkpoint_dir=rl_dir,
        updates=2,
        episodes_per_update=2,
        ppo_epochs=1,
        mini_batch_size=32,
        eval_games=2,
        seed=42,
        device_str="cpu",
        resume_path=s1["last_rl_checkpoint"],
        promote_to_best=False,
    )
    assert len(s2["history"]) == 2
    assert s2["history"][-1]["rl_update"] == 2
    assert s2["history"][-1]["eval_invalid_actions"] == 0


def test_cross_deck_evaluation_and_latency() -> None:
    """Verify cross-deck matchup matrix and CPU inference latency metrics."""
    model, default_deck, _ = load_policy_from_checkpoint("checkpoints/best_model.pt", device="cpu")
    starmie_deck = load_default_deck("decks/starmie_deck.csv")
    pool = [("default_deck", default_deck), ("starmie_deck", starmie_deck)]

    res = evaluate_cross_deck_matrix(model, pool, games_per_pair=1, seed=42)
    assert res["total_games"] == 4
    assert res["model_invalid_actions"] == 0
    assert res["model_runtime_errors"] == 0
    assert res["timeouts"] == 0
    assert res["cpu_latency_ms"]["total_decisions"] > 0
    assert res["cpu_latency_ms"]["p99"] >= res["cpu_latency_ms"]["mean"] > 0.0
    assert res["matrix"]["default_deck"]["starmie_deck"]["is_mirror"] is False


def test_bug1_mirror_match_deck_preservation() -> None:
    """Verify BUG-1 fix: play_single_match never overwrites explicit mirror decks."""
    starmie_deck = load_default_deck("decks/starmie_deck.csv")
    default_deck = load_default_deck("deck.csv")
    assert starmie_deck != default_deck

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

    obs_zero = copy.deepcopy(sample_obs)
    obs_zero["select"]["minCount"] = 0
    obs_zero["select"]["maxCount"] = 0
    obs_zero["select"]["option"] = [{"type": 3, "area": 2, "index": 0}, {"type": 3, "area": 2, "index": 1}]
    assert simulator_agent(obs_zero) == []
    assert model.select_action(obs_zero, default_deck=deck) == []

    obs_loop = copy.deepcopy(sample_obs)
    obs_loop["current"]["turn"] = 5
    obs_loop["current"]["turnActionCount"] = 25
    obs_loop["select"]["type"] = 0
    obs_loop["select"]["context"] = 0
    obs_loop["select"]["minCount"] = 1
    obs_loop["select"]["maxCount"] = 1
    obs_loop["select"]["option"] = [
        {"type": 10, "area": 5, "index": 0},
        {"type": 10, "area": 5, "index": 1},
        {"type": 14},
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
    chosen_mask = torch.tensor([0.0, 0.0, 0.0])
    min_count = torch.tensor([0], dtype=torch.long)
    max_count = torch.tensor([2], dtype=torch.long)

    loss = compute_policy_loss(opt_logits, stop_logits, num_options, chosen_mask, min_count, max_count)
    loss.backward()
    assert stop_logits.grad is not None and float(stop_logits.grad.item()) < 0.0
    assert opt_logits.grad is not None and all(float(g) > 0.0 for g in opt_logits.grad)

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
    assert feats["opt_feats"][0, 23] == 1.0
    assert feats["opt_card_ids"][0] == 1256


def test_checkpoint_load_and_agent_entrypoint_full_match() -> None:
    """Verify loading `checkpoints/best_model.pt`, Kaggle wrapper compatibility, and full CABT game."""
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

    # Verify wrapped Kaggle observation payload compatibility
    sample_obs, _ = _get_sample_active_obs()
    wrapped_act = simulator_agent({"observation": sample_obs})
    assert isinstance(wrapped_act, list)

    opp = make_random_baseline_agent(seed=123, default_deck=deck)
    res = play_single_match(simulator_agent, opp, deck, deck)
    assert res["winner"] in (0, 1, 2)
    assert res["steps"] > 10
    assert res["decks_equal"] is True
    assert res["invalid_actions"][0] == 0, f"Model made invalid actions: {res}"
    assert res["runtime_errors"][0] == 0, f"Model had runtime errors: {res}"

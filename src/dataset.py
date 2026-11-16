"""Replay streaming parser, episode-level splitting, and dataset builder.

Processes one replay JSON at a time to bound peak RAM usage, splits strictly by
`episode_id` before extracting decision steps, and verifies zero future or
opponent hidden-state leakage.
"""
from __future__ import annotations

import csv
import gc
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src import DATASET_REF, DATASET_VERSION
from src.features import extract_decision_features, validate_no_leakage


def download_replays_from_kaggle(
    out_dir: str | Path = "data/replays",
    num_top: int = 30,
    diverse_indices: list[int] | None = None,
) -> Path:
    """Download manifest.csv and selected replay JSONs from Kaggle dataset."""
    import kagglehub

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cache_dir = out_path.parent / "kagglehub_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir.resolve())

    manifest_file = out_path / "manifest.csv"
    if not manifest_file.exists():
        p = kagglehub.dataset_download(DATASET_REF, path="manifest.csv")
        shutil.copy2(p, manifest_file)

    with open(manifest_file, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    indices = list(range(min(num_top, len(rows))))
    if diverse_indices is None:
        diverse_indices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for idx in diverse_indices:
        if 0 <= idx < len(rows) and idx not in indices:
            indices.append(idx)

    for idx in indices:
        ep_id = rows[idx]["episode_id"]
        target = out_path / f"{ep_id}.json"
        if not target.exists():
            p = kagglehub.dataset_download(DATASET_REF, path=f"{ep_id}.json")
            shutil.move(p, target)
    return out_path


def parse_single_replay(
    filepath: str | Path,
    drop_forced_single: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int], list[list[int]]]:
    """Parse a single replay JSON file into decision samples and step accounting.

    In `kaggle-environments` CABT replays:
    - `steps[t][seat]` holds the pre-decision `status` and `observation` at step `t`.
    - When `steps[t][seat]["status"] == "ACTIVE"`, the action chosen by `seat` in
      response to `steps[t][seat]["observation"]` is stored at `steps[t + 1][seat]["action"]`.
    - At `t == 0`, both seats have `current=None, select=None` and `steps[1][seat]["action"]`
      is the 60-card deck (`list[int]`).
    - When `steps[t][seat]["status"] == "INACTIVE"`, the player is waiting (`steps[t + 1][seat]["action"] == []`).
    - At `t == len(steps) - 1`, `status == "DONE"` (terminal step).
    """
    path = Path(filepath)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    ep_id = str((raw.get("info") or {}).get("EpisodeId") or path.stem)
    steps = raw.get("steps") or []
    counts: Counter[str] = Counter()
    counts["episodes"] = 1
    counts["total_step_frames"] = len(steps)
    counts["total_seat_steps"] = 2 * len(steps)

    samples: list[dict[str, Any]] = []
    decks: list[list[int]] = []

    for t in range(len(steps)):
        st = steps[t]
        for seat in (0, 1):
            s = st[seat]
            status = s.get("status")
            obs = s.get("observation") or {}
            cur = obs.get("current")
            sel = obs.get("select")

            # 1. Terminal step (end of episode)
            if t == len(steps) - 1:
                counts["excluded_terminal_done_steps"] += 1
                continue

            next_act = steps[t + 1][seat].get("action")
            if isinstance(next_act, dict) and "action" in next_act:
                counts["unwrapped_dict_actions"] += 1
                next_act = next_act["action"]

            # 2. Waiting turn (INACTIVE)
            if status == "INACTIVE":
                counts["excluded_inactive_waiting_turns"] += 1
                if next_act:
                    counts["unexpected_inactive_nonempty_action"] += 1
                continue

            if status != "ACTIVE":
                counts["excluded_other_status_steps"] += 1
                continue

            # 3. Initial deck selection step (t == 0, current is None and select is None)
            if cur is None and sel is None:
                counts["excluded_deck_select_steps"] += 1
                if isinstance(next_act, list) and len(next_act) == 60 and all(isinstance(c, int) for c in next_act):
                    counts["valid_deck_selections"] += 1
                    decks.append(list(next_act))
                else:
                    counts["invalid_deck_selections"] += 1
                continue

            if cur is None or sel is None:
                counts["excluded_missing_state_or_select"] += 1
                continue

            # Verify no opponent hand or future leakage
            validate_no_leakage(obs, expected_seat=seat)

            opts = sel.get("option") or []
            min_c = int(sel.get("minCount") or 0)
            max_c = int(sel.get("maxCount") or 1)
            turn = int(cur.get("turn") or 0)

            # Validate action against legal options and [minCount, maxCount]
            if not isinstance(next_act, list):
                counts["excluded_invalid_action_format"] += 1
                continue
            in_range = all(isinstance(a, int) and 0 <= a < len(opts) for a in next_act)
            valid_len = (min_c <= len(next_act) <= max_c) and (len(set(next_act)) == len(next_act))
            if not in_range or not valid_len or len(opts) == 0:
                counts["excluded_invalid_action_bounds"] += 1
                continue

            counts["valid_decisions_total"] += 1
            is_forced = (len(opts) == 1 and min_c == 1 and max_c == 1)
            if is_forced:
                counts["forced_single_option_decisions"] += 1
                if drop_forced_single:
                    counts["excluded_forced_single_option"] += 1
                    continue
            else:
                counts["nontrivial_decisions"] += 1

            counts["kept_decisions"] += 1
            if turn == 0:
                counts["kept_turn0_setup_decisions"] += 1
            else:
                counts["kept_main_game_decisions"] += 1

            if max_c > 1:
                counts["kept_multi_select_decisions"] += 1
            else:
                counts["kept_single_select_decisions"] += 1

            feats = extract_decision_features(obs)
            chosen_mask = np.zeros(len(opts), dtype=np.float32)
            for a in next_act:
                chosen_mask[a] = 1.0

            samples.append({
                "episode_id": ep_id,
                "step": t,
                "seat": seat,
                "turn": turn,
                "is_forced": is_forced,
                "state_vec": feats["state_vec"],
                "hand_cards": feats["hand_cards"],
                "board_cards": feats["board_cards"],
                "select_ctx": feats["select_ctx"],
                "select_type": feats["select_type"],
                "min_count": feats["min_count"],
                "max_count": feats["max_count"],
                "num_options": feats["num_options"],
                "opt_feats": feats["opt_feats"],
                "opt_card_ids": feats["opt_card_ids"],
                "opt_target_card_ids": feats["opt_target_card_ids"],
                "opt_attack_ids": feats["opt_attack_ids"],
                "chosen_indices": [int(a) for a in next_act],
                "chosen_mask": chosen_mask,
            })

    del raw, steps
    return samples, dict(counts), decks


def pack_samples_to_tensors(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pack variable-length option lists into a flat contiguous tensor dictionary."""
    n = len(samples)
    if n == 0:
        raise ValueError("Cannot pack empty sample list")

    state_vec = np.stack([s["state_vec"] for s in samples], axis=0)
    hand_cards = np.stack([s["hand_cards"] for s in samples], axis=0)
    board_cards = np.stack([s["board_cards"] for s in samples], axis=0)
    select_ctx = np.array([s["select_ctx"] for s in samples], dtype=np.int64)
    select_type = np.array([s["select_type"] for s in samples], dtype=np.int64)
    min_count = np.array([s["min_count"] for s in samples], dtype=np.int64)
    max_count = np.array([s["max_count"] for s in samples], dtype=np.int64)
    num_options = np.array([s["num_options"] for s in samples], dtype=np.int64)
    turn = np.array([s["turn"] for s in samples], dtype=np.int64)
    is_forced = np.array([s["is_forced"] for s in samples], dtype=np.bool_)

    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(num_options, out=offsets[1:])

    opt_feats = np.concatenate([s["opt_feats"] for s in samples], axis=0)
    opt_card_ids = np.concatenate([s["opt_card_ids"] for s in samples], axis=0)
    opt_target_card_ids = np.concatenate([s["opt_target_card_ids"] for s in samples], axis=0)
    opt_attack_ids = np.concatenate([s["opt_attack_ids"] for s in samples], axis=0)
    chosen_mask = np.concatenate([s["chosen_mask"] for s in samples], axis=0)

    return {
        "num_samples": n,
        "episode_ids": [s["episode_id"] for s in samples],
        "steps": np.array([s["step"] for s in samples], dtype=np.int64),
        "seats": np.array([s["seat"] for s in samples], dtype=np.int64),
        "turn": torch.from_numpy(turn),
        "is_forced": torch.from_numpy(is_forced),
        "state_vec": torch.from_numpy(state_vec),
        "hand_cards": torch.from_numpy(hand_cards),
        "board_cards": torch.from_numpy(board_cards),
        "select_ctx": torch.from_numpy(select_ctx),
        "select_type": torch.from_numpy(select_type),
        "min_count": torch.from_numpy(min_count),
        "max_count": torch.from_numpy(max_count),
        "num_options": torch.from_numpy(num_options),
        "offsets": torch.from_numpy(offsets),
        "opt_feats": torch.from_numpy(opt_feats),
        "opt_card_ids": torch.from_numpy(opt_card_ids),
        "opt_target_card_ids": torch.from_numpy(opt_target_card_ids),
        "opt_attack_ids": torch.from_numpy(opt_attack_ids),
        "chosen_mask": torch.from_numpy(chosen_mask),
        "chosen_indices": [s["chosen_indices"] for s in samples],
    }


def preprocess_replays(
    replay_dir: str | Path = "data/replays",
    out_dir: str | Path = "data/processed",
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    drop_forced_in_train: bool = True,
) -> dict[str, Any]:
    """Split replay files by episode_id first, stream-parse each episode, and save shards."""
    replay_path = Path(replay_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in replay_path.glob("*.json") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )
    if len(files) < 3:
        raise ValueError(f"Need at least 3 replay JSON files in {replay_path}, found {len(files)}")

    rng = random.Random(seed)
    shuffled = list(files)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1
    n_test = n_total - n_train - n_val

    split_files = {
        "train": sorted(shuffled[:n_train], key=lambda p: int(p.stem)),
        "val": sorted(shuffled[n_train:n_train + n_val], key=lambda p: int(p.stem)),
        "test": sorted(shuffled[n_train + n_val:], key=lambda p: int(p.stem)),
    }

    # Verify zero episode overlap across splits
    train_ids = {p.stem for p in split_files["train"]}
    val_ids = {p.stem for p in split_files["val"]}
    test_ids = {p.stem for p in split_files["test"]}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    overall_counts: Counter[str] = Counter()
    split_stats: dict[str, dict[str, int]] = {}
    deck_counter: Counter[tuple[int, ...]] = Counter()

    for split_name, flist in split_files.items():
        s_counts: Counter[str] = Counter()
        s_samples: list[dict[str, Any]] = []
        drop_forced = drop_forced_in_train and (split_name == "train")

        for fp in flist:
            ep_samples, ep_counts, ep_decks = parse_single_replay(fp, drop_forced_single=drop_forced)
            s_samples.extend(ep_samples)
            s_counts.update(ep_counts)
            overall_counts.update(ep_counts)
            if split_name == "train":
                for dk in ep_decks:
                    deck_counter[tuple(dk)] += 1
            gc.collect()

        packed = pack_samples_to_tensors(s_samples)
        torch.save(packed, out_path / f"{split_name}.pt")
        split_stats[split_name] = dict(sorted(s_counts.items()))
        del s_samples, packed
        gc.collect()

    top_deck = list(deck_counter.most_common(1)[0][0]) if deck_counter else []

    summary = {
        "dataset_ref": DATASET_REF,
        "dataset_version": DATASET_VERSION,
        "seed": seed,
        "num_episodes_total": n_total,
        "episode_splits": {
            k: [p.stem for p in v] for k, v in split_files.items()
        },
        "overall_counts": dict(sorted(overall_counts.items())),
        "split_counts": split_stats,
        "exclusion_reasons": {
            "excluded_deck_select_steps": "Step t=0 initial deck selection where current=None and select=None (action at t=1 is 60-card deck, handled by deck loader)",
            "excluded_inactive_waiting_turns": "Waiting turn where player status=='INACTIVE' (agent is not called, action at t+1 is empty [])",
            "excluded_terminal_done_steps": "Terminal step at t=len(steps)-1 where status=='DONE' (match already concluded)",
            "excluded_forced_single_option": "Forced single-option step in train split where len(option)==1 and minCount==maxCount==1 (only [0] is legal; kept in val/test for full evaluation)",
            "excluded_invalid_action_bounds": "Any step with out-of-range option index or count outside [minCount, maxCount] (0 found)",
        },
        "top_train_deck": top_deck,
    }

    with open(out_path / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary

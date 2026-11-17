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
from src.features import (
    extract_decision_features,
    infer_fallback_metadata_from_replays,
    int_or,
    register_fallback_metadata,
    validate_no_leakage,
)


def select_manifest_indices(
    num_rows: int,
    num_top: int | None = None,
    num_strided: int | None = None,
    max_episodes: int | None = None,
    diverse_indices: list[int] | None = None,
) -> list[int]:
    """Select ordered, deduplicated row indices from manifest.csv for diverse deck coverage."""
    if num_rows <= 0:
        return []

    if max_episodes is not None and num_top is None and num_strided is None and diverse_indices is None:
        cap = min(max(0, max_episodes), num_rows)
        if cap >= num_rows:
            return list(range(num_rows))
        if cap == 0:
            return []
        top_k = cap // 2
        stride_k = cap - top_k
        indices = list(range(top_k))
        if stride_k > 0 and top_k < num_rows:
            grid = np.linspace(top_k, num_rows - 1, num=stride_k, dtype=int).tolist()
            seen = set(indices)
            for idx in grid:
                if idx not in seen:
                    indices.append(idx)
                    seen.add(idx)
        return indices[:cap]

    eff_top = 30 if num_top is None else max(0, min(num_top, num_rows))
    indices = list(range(eff_top))
    seen = set(indices)

    if diverse_indices is not None:
        for idx in diverse_indices:
            if 0 <= idx < num_rows and idx not in seen:
                indices.append(idx)
                seen.add(idx)

    if num_strided is not None and num_strided > 0 and eff_top < num_rows:
        grid = np.linspace(eff_top, num_rows - 1, num=min(num_strided, num_rows - eff_top), dtype=int).tolist()
        for idx in grid:
            if 0 <= idx < num_rows and idx not in seen:
                indices.append(idx)
                seen.add(idx)
    elif num_strided is None and diverse_indices is None:
        for idx in (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000):
            if 0 <= idx < num_rows and idx not in seen:
                indices.append(idx)
                seen.add(idx)

    if max_episodes is not None and max_episodes >= 0:
        indices = indices[:max_episodes]
    return indices


def download_replays_from_kaggle(
    out_dir: str | Path = "data/replays",
    num_top: int | None = None,
    num_strided: int | None = None,
    max_episodes: int | None = None,
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

    indices = select_manifest_indices(
        num_rows=len(rows),
        num_top=num_top,
        num_strided=num_strided,
        max_episodes=max_episodes,
        diverse_indices=diverse_indices,
    )

    downloaded = 0
    skipped_existing = 0
    for pos, idx in enumerate(indices, start=1):
        ep_id = rows[idx]["episode_id"]
        target = out_path / f"{ep_id}.json"
        if target.exists() and target.stat().st_size > 0:
            skipped_existing += 1
            continue
        p = kagglehub.dataset_download(DATASET_REF, path=f"{ep_id}.json")
        shutil.move(p, target)
        downloaded += 1
        if downloaded % 20 == 0 or pos == len(indices):
            print(f"[download-data] {pos}/{len(indices)} ready (downloaded={downloaded}, cached={skipped_existing})")

    return out_path


def parse_single_replay(
    filepath: str | Path,
    drop_forced_single: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int], list[list[int]]]:
    """Parse a single replay JSON file into decision samples and step accounting."""
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
            min_c = int_or(sel, "minCount", 0)
            max_c = int_or(sel, "maxCount", 1)
            turn = int_or(cur, "turn", 0)

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


def _save_shard(
    samples: list[dict[str, Any]],
    out_path: Path,
    split_name: str,
    shard_idx: int,
) -> dict[str, Any]:
    """Pack a bounded sample buffer into a shard file and return its manifest entry."""
    packed = pack_samples_to_tensors(samples)
    fname = f"{split_name}_shard_{shard_idx:04d}.pt"
    fpath = out_path / fname
    torch.save(packed, fpath)
    ep_ids = sorted(set(packed["episode_ids"]))
    n_samples = int(packed["num_samples"])
    del packed
    gc.collect()
    return {
        "file": fname,
        "num_samples": n_samples,
        "num_episodes": len(ep_ids),
        "episode_ids": ep_ids,
    }


def load_split_shard_manifest(data_dir: str | Path, split_name: str) -> dict[str, Any]:
    """Load the shard manifest JSON for `split_name` (`train`/`val`/`test`)."""
    data_path = Path(data_dir)
    manifest_p = data_path / f"{split_name}_shards.json"
    if manifest_p.exists():
        with open(manifest_p, encoding="utf-8") as f:
            return json.load(f)

    shard_files = sorted(data_path.glob(f"{split_name}_shard_*.pt"))
    if shard_files:
        return {
            "split": split_name,
            "num_shards": len(shard_files),
            "num_samples": None,
            "shards": [{"file": p.name} for p in shard_files],
        }

    legacy_p = data_path / f"{split_name}.pt"
    if legacy_p.exists():
        return {
            "split": split_name,
            "num_shards": 1,
            "num_samples": None,
            "shards": [{"file": legacy_p.name}],
        }
    raise FileNotFoundError(f"No shards or legacy .pt found for split '{split_name}' in {data_path}")


def get_split_shard_paths(data_dir: str | Path, split_name: str) -> list[Path]:
    """Return ordered list of shard file paths for `split_name`."""
    data_path = Path(data_dir)
    manifest = load_split_shard_manifest(data_path, split_name)
    return [data_path / entry["file"] for entry in manifest["shards"]]


def get_split_num_samples(data_dir: str | Path, split_name: str) -> int:
    """Return total decision samples in `split_name` without keeping all shards in RAM."""
    data_path = Path(data_dir)
    manifest = load_split_shard_manifest(data_path, split_name)
    if manifest.get("num_samples") is not None:
        return int(manifest["num_samples"])
    total = 0
    for p in get_split_shard_paths(data_path, split_name):
        pt = torch.load(p, weights_only=False)
        total += int(pt["num_samples"])
        del pt
    return total


def iter_split_shards(
    data_dir: str | Path,
    split_name: str,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
):
    """Stream-load one shard at a time so Peak CPU RAM remains O(1 shard)."""
    paths = get_split_shard_paths(data_dir, split_name)
    order = list(range(len(paths)))
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        order = rng.permutation(len(paths)).tolist()
    for idx in order:
        p = paths[idx]
        shard = torch.load(p, weights_only=False)
        yield p, shard
        del shard
        gc.collect()


def preprocess_replays(
    replay_dir: str | Path = "data/replays",
    out_dir: str | Path = "data/processed",
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    drop_forced_in_train: bool = True,
    shard_size_samples: int = 15000,
    infer_metadata: bool = True,
) -> dict[str, Any]:
    """Split replay files by episode_id first, stream-parse each episode, and save bounded shards."""
    replay_path = Path(replay_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Remove stale shard files from previous runs
    for old_f in list(out_path.glob("*_shard_*.pt")) + list(out_path.glob("*_shards.json")):
        old_f.unlink()
    for legacy_name in ("train.pt", "val.pt", "test.pt"):
        lp = out_path / legacy_name
        if lp.exists():
            lp.unlink()

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

    # Infer fallback metadata strictly from train split files (never val/test)
    # and never overwrite curated static FALLBACK_CARD_DB / FALLBACK_ATTACK_DB entries.
    if infer_metadata:
        inf_cards, inf_atks = infer_fallback_metadata_from_replays(
            replay_files=split_files["train"],
            max_files=len(split_files["train"]),
        )
        register_fallback_metadata(inf_cards, inf_atks, overwrite_curated=False)
        from src.features import FALLBACK_ATTACK_DB, FALLBACK_CARD_DB
        unseen_cards = {k: v for k, v in inf_cards.items() if k not in FALLBACK_CARD_DB}
        unseen_atks = {k: v for k, v in inf_atks.items() if k not in FALLBACK_ATTACK_DB}
        meta_dump = {
            "source_split": "train",
            "num_train_episodes_scanned": len(split_files["train"]),
            "cards": {
                str(k): {
                    "cardId": v.cardId,
                    "cardType": v.cardType,
                    "hp": v.hp,
                    "ex": v.ex,
                    "basic": v.basic,
                    "stage": v.stage,
                    "attacks": list(v.attacks),
                    "skills": list(v.skills),
                }
                for k, v in unseen_cards.items()
            },
            "attacks": {
                str(k): {
                    "attackId": v.attackId,
                    "damage": v.damage,
                    "energies": list(v.energies),
                }
                for k, v in unseen_atks.items()
            },
        }
        with open(out_path / "inferred_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_dump, f, indent=2)

    overall_counts: Counter[str] = Counter()
    split_stats: dict[str, dict[str, int]] = {}
    shard_summaries: dict[str, dict[str, Any]] = {}
    deck_counter: Counter[tuple[int, ...]] = Counter()
    eff_shard_cap = max(1, int(shard_size_samples))

    for split_name, flist in split_files.items():
        s_counts: Counter[str] = Counter()
        shard_buf: list[dict[str, Any]] = []
        shard_entries: list[dict[str, Any]] = []
        shard_idx = 0
        split_total_samples = 0
        split_ep_ids: list[str] = []
        drop_forced = drop_forced_in_train and (split_name == "train")

        for fp_idx, fp in enumerate(flist, start=1):
            ep_samples, ep_counts, ep_decks = parse_single_replay(fp, drop_forced_single=drop_forced)
            shard_buf.extend(ep_samples)
            s_counts.update(ep_counts)
            overall_counts.update(ep_counts)
            split_ep_ids.append(fp.stem)
            if split_name == "train":
                for dk in ep_decks:
                    deck_counter[tuple(dk)] += 1
            del ep_samples

            while len(shard_buf) >= eff_shard_cap:
                chunk = shard_buf[:eff_shard_cap]
                shard_buf = shard_buf[eff_shard_cap:]
                entry = _save_shard(chunk, out_path, split_name, shard_idx)
                shard_entries.append(entry)
                split_total_samples += entry["num_samples"]
                shard_idx += 1
                del chunk
                gc.collect()

            if fp_idx % 25 == 0 or fp_idx == len(flist):
                print(
                    f"[preprocess:{split_name}] {fp_idx}/{len(flist)} episodes, "
                    f"shards={len(shard_entries)}, buffered={len(shard_buf)}"
                )

        if shard_buf:
            entry = _save_shard(shard_buf, out_path, split_name, shard_idx)
            shard_entries.append(entry)
            split_total_samples += entry["num_samples"]
            shard_buf.clear()
            gc.collect()

        manifest = {
            "split": split_name,
            "num_shards": len(shard_entries),
            "num_samples": split_total_samples,
            "episode_ids": split_ep_ids,
            "shards": shard_entries,
        }
        with open(out_path / f"{split_name}_shards.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        shard_summaries[split_name] = {
            "num_shards": len(shard_entries),
            "num_samples": split_total_samples,
            "shard_files": [e["file"] for e in shard_entries],
        }
        split_stats[split_name] = dict(sorted(s_counts.items()))

    top_deck = list(deck_counter.most_common(1)[0][0]) if deck_counter else []

    summary = {
        "dataset_ref": DATASET_REF,
        "dataset_version": DATASET_VERSION,
        "seed": seed,
        "shard_size_samples": eff_shard_cap,
        "num_episodes_total": n_total,
        "episode_splits": {
            k: [p.stem for p in v] for k, v in split_files.items()
        },
        "shard_summaries": shard_summaries,
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

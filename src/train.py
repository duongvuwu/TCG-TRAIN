"""Training, sharded streaming, checkpoint resume, and RL self-play fine-tuning pipeline."""
from __future__ import annotations

import copy
import gc
import json
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src import DATASET_REF, DATASET_VERSION
from src.dataset import get_split_num_samples, iter_split_shards
from src.features import (
    LEGACY_OPT_DIM,
    LEGACY_STATE_DIM,
    OPT_DIM,
    STATE_DIM,
    extract_decision_features,
    int_or,
)
from src.model import PTCGPolicyNet


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return res.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_default_deck(deck_path: str | Path = "deck.csv") -> list[int]:
    p = Path(deck_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            cards = [int(x.strip()) for x in f if x.strip()]
        if len(cards) == 60:
            return cards
    return [1] * 60


def slice_packed_batch(
    data: dict[str, Any],
    indices: np.ndarray | list[int],
    device: torch.device,
) -> dict[str, Any]:
    """Extract a mini-batch of samples and their variable-length options."""
    idx_t = torch.as_tensor(indices, dtype=torch.long)
    state_vec = data["state_vec"][idx_t].to(device)
    hand_cards = data["hand_cards"][idx_t].to(device)
    board_cards = data["board_cards"][idx_t].to(device)
    select_ctx = data["select_ctx"][idx_t].to(device)
    select_type = data["select_type"][idx_t].to(device)
    min_count = data["min_count"][idx_t].to(device)
    max_count = data["max_count"][idx_t].to(device)
    num_options = data["num_options"][idx_t].to(device)

    offsets = data["offsets"]
    opt_slices = []
    for i in indices:
        s = int(offsets[i].item())
        e = int(offsets[i + 1].item())
        opt_slices.append(torch.arange(s, e, dtype=torch.long))
    flat_idx = torch.cat(opt_slices, dim=0)

    opt_feats = data["opt_feats"][flat_idx].to(device)
    opt_card_ids = data["opt_card_ids"][flat_idx].to(device)
    opt_target_card_ids = data["opt_target_card_ids"][flat_idx].to(device)
    opt_attack_ids = data["opt_attack_ids"][flat_idx].to(device)
    chosen_mask = data["chosen_mask"][flat_idx].to(device)
    chosen_indices = [data["chosen_indices"][int(i)] for i in indices]

    return {
        "state_vec": state_vec,
        "hand_cards": hand_cards,
        "board_cards": board_cards,
        "select_ctx": select_ctx,
        "select_type": select_type,
        "min_count": min_count,
        "max_count": max_count,
        "num_options": num_options,
        "opt_feats": opt_feats,
        "opt_card_ids": opt_card_ids,
        "opt_target_card_ids": opt_target_card_ids,
        "opt_attack_ids": opt_attack_ids,
        "chosen_mask": chosen_mask,
        "chosen_indices": chosen_indices,
    }


def compute_policy_loss(
    opt_logits: torch.Tensor,
    stop_logits: torch.Tensor,
    num_options: torch.Tensor,
    chosen_mask: torch.Tensor,
    min_count: torch.Tensor | None = None,
    max_count: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized listwise cross-entropy (with explicit stop token for minCount==0)
    plus unbiased relative stop-threshold BCE loss.
    """
    device = opt_logits.device
    batch_size = int(num_options.size(0))
    max_opts = int(num_options.max().item())

    stop_rep = torch.repeat_interleave(stop_logits, num_options, dim=0)
    rel_logits = opt_logits - stop_rep
    raw_bce = F.binary_cross_entropy_with_logits(rel_logits, chosen_mask, reduction="none")

    if min_count is not None and max_count is not None:
        is_var = (min_count.to(device) < max_count.to(device)).float()
        sample_weights = 0.25 + 1.75 * is_var
        opt_weights = torch.repeat_interleave(sample_weights, num_options, dim=0)
        bce_loss = (raw_bce * opt_weights).sum() / opt_weights.sum().clamp(min=1.0)
    else:
        bce_loss = raw_bce.mean()

    padded_logits = torch.full((batch_size, max_opts + 1), -1e9, device=device, dtype=opt_logits.dtype)
    padded_targets = torch.zeros((batch_size, max_opts + 1), device=device, dtype=opt_logits.dtype)

    row_ids = torch.repeat_interleave(torch.arange(batch_size, device=device), num_options)
    cum_opts = torch.cumsum(num_options, dim=0) - num_options
    col_ids = torch.arange(opt_logits.size(0), device=device) - torch.repeat_interleave(cum_opts, num_options)

    padded_logits[row_ids, col_ids] = opt_logits
    padded_targets[row_ids, col_ids] = chosen_mask

    target_sums = padded_targets[:, :max_opts].sum(dim=1)
    is_empty_choice = (target_sums == 0)
    if min_count is not None:
        allow_zero = (min_count.to(device) == 0) | is_empty_choice
    else:
        allow_zero = is_empty_choice

    padded_logits[allow_zero, max_opts] = stop_logits[allow_zero]
    padded_targets[is_empty_choice, max_opts] = 1.0

    full_sums = padded_targets.sum(dim=1, keepdim=True).clamp(min=1.0)
    norm_targets = padded_targets / full_sums
    log_probs = F.log_softmax(padded_logits, dim=-1)
    ce_loss = -(norm_targets * log_probs).sum(dim=-1).mean()

    return ce_loss + 0.5 * bce_loss


def compute_conditional_multi_loss(
    model: PTCGPolicyNet,
    state_h: torch.Tensor,
    opt_ctx: torch.Tensor,
    opt_logits: torch.Tensor,
    batch: dict[str, Any],
    rng: np.random.Generator,
) -> torch.Tensor:
    """Train conditional multi-select heads (`cond_scorer`, `cond_stop_head`) via
    random-prefix subset conditioning on `max_count > 1` decisions.
    """
    if not model.use_cond_multi:
        return opt_logits.new_zeros(())

    device = opt_logits.device
    num_opts_cpu = batch["num_options"].cpu().numpy()
    min_c_cpu = batch["min_count"].cpu().numpy()
    max_c_cpu = batch["max_count"].cpu().numpy()
    chosen_list = batch["chosen_indices"]

    offsets = np.zeros(len(num_opts_cpu) + 1, dtype=np.int64)
    np.cumsum(num_opts_cpu, out=offsets[1:])

    sub_rows: list[int] = []
    sel_masks_per_row: list[torch.Tensor] = []
    rem_targets_per_row: list[torch.Tensor] = []
    fracs: list[float] = []
    rem_mins: list[int] = []
    rem_maxs: list[int] = []

    for bi in range(len(num_opts_cpu)):
        k = int(num_opts_cpu[bi])
        min_c = int(min_c_cpu[bi])
        max_c = min(int(max_c_cpu[bi]), k)
        gold = [g for g in chosen_list[bi] if 0 <= g < k]
        if max_c <= 1 or k <= 1 or len(gold) == 0:
            continue

        # Build subset-to-remainder transition(s)
        if len(gold) >= 2:
            # Sample random subset of size 1..len(gold)-1 to score remaining gold items
            sub_k = int(rng.integers(1, len(gold)))
            perm_g = rng.permutation(len(gold))
            prefix = [gold[j] for j in perm_g[:sub_k]]
            rem = [gold[j] for j in perm_g[sub_k:]]

            sm = torch.zeros(k, device=device, dtype=opt_logits.dtype)
            rt = torch.zeros(k, device=device, dtype=opt_logits.dtype)
            for idx in prefix:
                sm[idx] = 1.0
            for idx in rem:
                rt[idx] = 1.0
            sub_rows.append(bi)
            sel_masks_per_row.append(sm)
            rem_targets_per_row.append(rt)
            fracs.append(float(len(prefix)) / max(max_c, 1))
            rem_mins.append(max(0, min_c - len(prefix)))
            rem_maxs.append(max(0, max_c - len(prefix)))

        if len(gold) < max_c and min_c < max_c:
            # Full-gold stop supervision: after picking all of `gold`, teach cond_stop_head to STOP
            sm = torch.zeros(k, device=device, dtype=opt_logits.dtype)
            rt = torch.zeros(k, device=device, dtype=opt_logits.dtype)
            for idx in gold:
                sm[idx] = 1.0
            sub_rows.append(bi)
            sel_masks_per_row.append(sm)
            rem_targets_per_row.append(rt)
            fracs.append(float(len(gold)) / max(max_c, 1))
            rem_mins.append(0)
            rem_maxs.append(max(0, max_c - len(gold)))

    if not sub_rows:
        return opt_logits.new_zeros(())

    row_t = torch.tensor(sub_rows, dtype=torch.long, device=device)
    sub_state_h = state_h[row_t]
    sub_num_opts = batch["num_options"][row_t]
    sub_opt_ctx = torch.cat([opt_ctx[offsets[bi]:offsets[bi + 1]] for bi in sub_rows], dim=0)
    sub_base_logits = torch.cat([opt_logits[offsets[bi]:offsets[bi + 1]] for bi in sub_rows], dim=0)
    sub_sel_mask = torch.cat(sel_masks_per_row, dim=0)
    sub_rem_targets = torch.cat(rem_targets_per_row, dim=0)
    sub_frac = torch.tensor(fracs, dtype=opt_logits.dtype, device=device).unsqueeze(-1)
    sub_min = torch.tensor(rem_mins, dtype=torch.long, device=device)
    sub_max = torch.tensor(rem_maxs, dtype=torch.long, device=device)

    cond_logits, cond_stop = model.forward_conditional_subset(
        sub_state_h,
        sub_opt_ctx,
        sub_num_opts,
        sub_base_logits,
        sub_sel_mask,
        sub_frac,
    )
    return compute_policy_loss(
        cond_logits,
        cond_stop,
        sub_num_opts,
        sub_rem_targets,
        sub_min,
        sub_max,
    )


@torch.no_grad()
def _decode_batch_predictions(
    model: PTCGPolicyNet,
    state_h: torch.Tensor,
    opt_ctx: torch.Tensor,
    opt_logits: torch.Tensor,
    stop_logits: torch.Tensor,
    num_options: torch.Tensor,
    min_count: torch.Tensor,
    max_count: torch.Tensor,
) -> list[list[int]]:
    """Decode legal actions for an entire batch (conditional decoding for multi-select)."""
    logits_cpu = opt_logits.cpu().numpy()
    stop_cpu = stop_logits.cpu().numpy()
    num_opts_cpu = num_options.cpu().numpy()
    min_c_cpu = min_count.cpu().numpy()
    max_c_cpu = max_count.cpu().numpy()
    device = opt_logits.device

    preds: list[list[int]] = []
    cursor = 0
    for bi in range(len(num_opts_cpu)):
        k = int(num_opts_cpu[bi])
        s_logits = logits_cpu[cursor:cursor + k]
        s_start = cursor
        cursor += k
        min_c = max(0, min(int(min_c_cpu[bi]), k))
        max_c = max(min_c, min(int(max_c_cpu[bi]), k))

        if k == 0 or max_c <= 0:
            preds.append([])
            continue
        if not model.use_cond_multi or max_c <= 1 or k <= 1:
            preds.append(PTCGPolicyNet.decode_action_from_logits(s_logits, float(stop_cpu[bi]), min_c, max_c))
            continue

        # Conditional sequential decoding matching `model.select_action`
        chosen: list[int] = []
        first_idx = int(np.argmax(s_logits))
        if min_c == 0 and float(s_logits[first_idx]) <= float(stop_cpu[bi]):
            preds.append([])
            continue
        chosen.append(first_idx)

        sel_mask = torch.zeros(k, device=device, dtype=state_h.dtype)
        sel_mask[first_idx] = 1.0
        sh_i = state_h[bi:bi + 1]
        oc_i = opt_ctx[s_start:s_start + k]
        ol_i = opt_logits[s_start:s_start + k]
        no_i = num_options[bi:bi + 1]

        for step in range(1, max_c):
            if len(chosen) >= k:
                break
            frac_t = torch.tensor([[float(len(chosen)) / max(max_c, 1)]], device=device, dtype=state_h.dtype)
            c_logits, c_stop = model.forward_conditional_subset(sh_i, oc_i, no_i, ol_i, sel_mask, frac_t)
            c_np = c_logits.cpu().numpy()
            for p in chosen:
                c_np[p] = -1e9
            next_idx = int(np.argmax(c_np))
            if c_np[next_idx] <= -1e8:
                break
            if step >= min_c and float(c_np[next_idx]) <= float(c_stop[0].item()):
                break
            chosen.append(next_idx)
            sel_mask[next_idx] = 1.0

        chosen.sort()
        preds.append(chosen)

    return preds


@torch.no_grad()
def evaluate_packed_split(
    model: PTCGPolicyNet,
    data_or_dir: dict[str, Any] | str | Path,
    device: torch.device,
    batch_size: int = 256,
    split_name: str = "val",
) -> dict[str, Any]:
    """Evaluate exact-match accuracy, single/multi-select metrics, and loss across shards or a packed dict."""
    model.eval()
    if isinstance(data_or_dir, dict):
        shard_iter = [(None, data_or_dir)]
    else:
        shard_iter = iter_split_shards(data_or_dir, split_name, shuffle=False)

    total_loss = 0.0
    n_batches = 0
    n_total = 0

    exact_all = 0
    exact_nontrivial = 0
    n_nontrivial = 0

    single_correct = 0
    single_total = 0
    multi_exact = 0
    multi_jaccard_sum = 0.0
    multi_total = 0

    zero_select_correct = 0
    zero_select_total = 0

    turn0_correct = 0
    turn0_total = 0
    main_correct = 0
    main_total = 0

    invalid_predictions = 0

    for _, data in shard_iter:
        n_shard = int(data["num_samples"])
        n_total += n_shard
        all_indices = np.arange(n_shard)
        turns_np = data["turn"].numpy()
        forced_np = data["is_forced"].numpy()

        for start in range(0, n_shard, batch_size):
            batch_idx = all_indices[start:start + batch_size]
            batch = slice_packed_batch(data, batch_idx, device)
            opt_logits, stop_logits, _, state_h, opt_ctx = model.forward_packed_with_value(
                batch["state_vec"],
                batch["hand_cards"],
                batch["board_cards"],
                batch["select_ctx"],
                batch["select_type"],
                batch["num_options"],
                batch["opt_feats"],
                batch["opt_card_ids"],
                batch["opt_target_card_ids"],
                batch["opt_attack_ids"],
            )
            loss = compute_policy_loss(
                opt_logits,
                stop_logits,
                batch["num_options"],
                batch["chosen_mask"],
                batch["min_count"],
                batch["max_count"],
            )
            total_loss += float(loss.item())
            n_batches += 1

            preds = _decode_batch_predictions(
                model,
                state_h,
                opt_ctx,
                opt_logits,
                stop_logits,
                batch["num_options"],
                batch["min_count"],
                batch["max_count"],
            )
            num_opts_cpu = batch["num_options"].cpu().numpy()
            min_c_cpu = batch["min_count"].cpu().numpy()
            max_c_cpu = batch["max_count"].cpu().numpy()

            for bi, global_i in enumerate(batch_idx):
                k = int(num_opts_cpu[bi])
                min_c = int(min_c_cpu[bi])
                max_c = int(max_c_cpu[bi])
                pred = preds[bi]
                gold = batch["chosen_indices"][bi]

                if (
                    not isinstance(pred, list)
                    or not (min_c <= len(pred) <= min(max_c, k))
                    or len(set(pred)) != len(pred)
                    or any(not isinstance(a, int) or a < 0 or a >= k for a in pred)
                ):
                    invalid_predictions += 1

                is_exact = (sorted(pred) == sorted(gold))
                if is_exact:
                    exact_all += 1

                if len(gold) == 0:
                    zero_select_total += 1
                    if len(pred) == 0:
                        zero_select_correct += 1

                if not forced_np[global_i]:
                    n_nontrivial += 1
                    if is_exact:
                        exact_nontrivial += 1

                if max_c > 1:
                    multi_total += 1
                    if is_exact:
                        multi_exact += 1
                    pset, gset = set(pred), set(gold)
                    union = len(pset | gset)
                    jacc = 1.0 if union == 0 else len(pset & gset) / union
                    multi_jaccard_sum += jacc
                else:
                    single_total += 1
                    if is_exact:
                        single_correct += 1

                if turns_np[global_i] == 0:
                    turn0_total += 1
                    if is_exact:
                        turn0_correct += 1
                else:
                    main_total += 1
                    if is_exact:
                        main_correct += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "num_samples": n_total,
        "num_nontrivial": n_nontrivial,
        "exact_match_all": exact_all / max(n_total, 1),
        "exact_match_all_count": exact_all,
        "exact_match_nontrivial": exact_nontrivial / max(n_nontrivial, 1),
        "exact_match_nontrivial_count": exact_nontrivial,
        "single_select_acc": single_correct / max(single_total, 1),
        "single_select_correct": single_correct,
        "single_select_total": single_total,
        "multi_select_exact_acc": multi_exact / max(multi_total, 1),
        "multi_select_exact_count": multi_exact,
        "multi_select_jaccard": multi_jaccard_sum / max(multi_total, 1),
        "multi_select_total": multi_total,
        "zero_select_acc": zero_select_correct / max(zero_select_total, 1),
        "zero_select_correct": zero_select_correct,
        "zero_select_total": zero_select_total,
        "turn0_setup_acc": turn0_correct / max(turn0_total, 1),
        "turn0_setup_correct": turn0_correct,
        "turn0_setup_total": turn0_total,
        "main_game_acc": main_correct / max(main_total, 1),
        "main_game_correct": main_correct,
        "main_game_total": main_total,
        "invalid_actions": invalid_predictions,
    }


def train_policy(
    data_dir: str | Path = "data/processed",
    checkpoint_dir: str | Path = "checkpoints",
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    seed: int = 42,
    device_str: str = "auto",
    resume_path: str | Path | None = None,
    use_set_attn: bool = True,
    use_cond_multi: bool = True,
) -> dict[str, Any]:
    """Train the imitation learning policy over sharded storage with bounded CPU RAM."""
    set_all_seeds(seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )

    data_path = Path(data_dir)
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    n_train = get_split_num_samples(data_path, "train")

    model = PTCGPolicyNet(
        state_dim=STATE_DIM,
        opt_dim=OPT_DIM,
        hidden_dim=hidden_dim,
        dropout=dropout,
        use_set_attn=use_set_attn,
        use_cond_multi=use_cond_multi,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.05)

    start_epoch = 1
    best_val_acc = -1.0
    history: list[dict[str, Any]] = []
    default_deck = load_default_deck("deck.csv")

    if resume_path is not None and Path(resume_path).exists():
        ckpt = torch.load(Path(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            scheduler.T_max = max(epochs, start_epoch)
        scheduler.T_max = max(epochs, start_epoch)
        best_val_acc = float(ckpt.get("best_val_acc", -1.0))
        history = list(ckpt.get("history", []))
        if "default_deck" in ckpt and len(ckpt["default_deck"]) == 60:
            default_deck = list(ckpt["default_deck"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rng = np.random.default_rng(seed + start_epoch)
    git_commit = get_git_commit()
    t_start_all = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t_ep = time.perf_counter()
        ep_loss = 0.0
        n_steps = 0

        for _, train_shard in iter_split_shards(data_path, "train", shuffle=True, rng=rng):
            n_shard = int(train_shard["num_samples"])
            perm = rng.permutation(n_shard)

            for start in range(0, n_shard, batch_size):
                batch_idx = perm[start:start + batch_size]
                batch = slice_packed_batch(train_shard, batch_idx, device)

                optimizer.zero_grad(set_to_none=True)
                opt_logits, stop_logits, _, state_h, opt_ctx = model.forward_packed_with_value(
                    batch["state_vec"],
                    batch["hand_cards"],
                    batch["board_cards"],
                    batch["select_ctx"],
                    batch["select_type"],
                    batch["num_options"],
                    batch["opt_feats"],
                    batch["opt_card_ids"],
                    batch["opt_target_card_ids"],
                    batch["opt_attack_ids"],
                )
                base_loss = compute_policy_loss(
                    opt_logits,
                    stop_logits,
                    batch["num_options"],
                    batch["chosen_mask"],
                    batch["min_count"],
                    batch["max_count"],
                )
                cond_loss = compute_conditional_multi_loss(
                    model, state_h, opt_ctx, opt_logits, batch, rng
                )
                loss = base_loss + 0.75 * cond_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                ep_loss += float(loss.item())
                n_steps += 1

        if scheduler.last_epoch < scheduler.T_max:
            scheduler.step()

        train_time = time.perf_counter() - t_ep
        samples_per_sec = n_train / max(train_time, 1e-6)
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )

        val_metrics = evaluate_packed_split(model, data_path, device, batch_size=batch_size * 2, split_name="val")
        val_acc = float(val_metrics["exact_match_nontrivial"])

        row = {
            "epoch": epoch,
            "train_loss": ep_loss / max(n_steps, 1),
            "val_loss": val_metrics["loss"],
            "val_exact_all": val_metrics["exact_match_all"],
            "val_exact_nontrivial": val_acc,
            "val_single_acc": val_metrics["single_select_acc"],
            "val_multi_exact_acc": val_metrics["multi_select_exact_acc"],
            "val_multi_jaccard": val_metrics["multi_select_jaccard"],
            "val_zero_select_acc": val_metrics["zero_select_acc"],
            "val_invalid_actions": val_metrics["invalid_actions"],
            "epoch_time_sec": round(train_time, 4),
            "samples_per_sec": round(samples_per_sec, 1),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"[Epoch {epoch:02d}/{epochs:02d}] "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_acc(nontrivial)={val_acc*100:.2f}% val_acc(all)={row['val_exact_all']*100:.2f}% "
            f"multi_acc={row['val_multi_exact_acc']*100:.2f}% zero_acc={row['val_zero_select_acc']*100:.1f}% "
            f"speed={samples_per_sec:.0f} samp/s vram={peak_vram_mb:.1f}MB lr={row['lr']:.6f}"
        )

        ckpt_payload = {
            "epoch": epoch,
            "best_val_acc": max(best_val_acc, val_acc),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {
                "state_dim": STATE_DIM,
                "opt_dim": OPT_DIM,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "use_set_attn": use_set_attn,
                "use_cond_multi": use_cond_multi,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "epochs": epochs,
                "seed": seed,
            },
            "dataset_ref": DATASET_REF,
            "dataset_version": DATASET_VERSION,
            "git_commit": git_commit,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "default_deck": default_deck,
            "val_metrics": val_metrics,
            "history": history,
        }
        torch.save(ckpt_payload, ckpt_path / "last_model.pt")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt_payload["best_val_acc"] = best_val_acc
            torch.save(ckpt_payload, ckpt_path / "best_bc_model.pt")
            torch.save(ckpt_payload, ckpt_path / "best_model.pt")

    total_time = time.perf_counter() - t_start_all
    summary = {
        "best_val_acc_nontrivial": best_val_acc,
        "num_train_samples": n_train,
        "total_train_time_sec": round(total_time, 3),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "peak_vram_mb": round(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0,
            2,
        ),
        "best_checkpoint": str((ckpt_path / "best_model.pt").resolve()),
        "last_checkpoint": str((ckpt_path / "last_model.pt").resolve()),
        "history": history,
    }
    with open(ckpt_path / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


# ==============================================================================
# Phase 3: PPO + KL-to-BC Self-Play / Opponent-Pool RL Fine-Tuning in CABT
# ==============================================================================

def _compute_log_prob_and_entropy_single_sample(
    model: PTCGPolicyNet,
    step_rec: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute differentiable `(log_prob, entropy, value, opt_logits, stop_logits)` for 1 step."""
    state_vec = step_rec["state_vec"].unsqueeze(0).to(device)
    hand_cards = step_rec["hand_cards"].unsqueeze(0).to(device)
    board_cards = step_rec["board_cards"].unsqueeze(0).to(device)
    select_ctx = torch.tensor([step_rec["select_ctx"]], dtype=torch.long, device=device)
    select_type = torch.tensor([step_rec["select_type"]], dtype=torch.long, device=device)
    num_options = torch.tensor([step_rec["num_options"]], dtype=torch.long, device=device)
    opt_feats = step_rec["opt_feats"].to(device)
    opt_card_ids = step_rec["opt_card_ids"].to(device)
    opt_target_card_ids = step_rec["opt_target_card_ids"].to(device)
    opt_attack_ids = step_rec["opt_attack_ids"].to(device)

    opt_logits, stop_logits, values, state_h, opt_ctx = model.forward_packed_with_value(
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
    guard_delta = step_rec["guard_delta"].to(device)
    eff_logits = opt_logits + guard_delta

    min_c = step_rec["min_c"]
    max_c = step_rec["max_c"]
    chosen: list[int] = step_rec["chosen"]
    n_opts = step_rec["num_options"]

    if min_c == 1 and max_c == 1:
        log_probs = F.log_softmax(eff_logits, dim=0)
        probs = log_probs.exp()
        valid_mask = eff_logits > -1e8
        entropy = -(probs[valid_mask] * log_probs[valid_mask]).sum()
        lp = log_probs[chosen[0]]
        return lp, entropy, values[0], eff_logits, stop_logits[0]

    # Sequential Plackett-Luce + stop token log-prob for multi-select
    total_lp = eff_logits.new_zeros(())
    total_ent = eff_logits.new_zeros(())
    sel_mask = torch.zeros(n_opts, device=device, dtype=eff_logits.dtype)

    if len(chosen) == 0:
        # Chose stop immediately at step 0
        comb = torch.cat([eff_logits, stop_logits], dim=0)
        lps = F.log_softmax(comb, dim=0)
        ps = lps.exp()
        total_lp = lps[-1]
        total_ent = -(ps * lps).sum()
        return total_lp, total_ent, values[0], eff_logits, stop_logits[0]

    for s_idx, picked in enumerate(chosen):
        if s_idx == 0 or not model.use_cond_multi:
            cur_logits = eff_logits.masked_fill(sel_mask > 0.5, -1e9)
            cur_stop = stop_logits
        else:
            frac_t = torch.tensor([[float(s_idx) / max(max_c, 1)]], device=device, dtype=state_h.dtype)
            c_logits, c_stop = model.forward_conditional_subset(
                state_h, opt_ctx, num_options, opt_logits, sel_mask, frac_t
            )
            cur_logits = (c_logits + guard_delta).masked_fill(sel_mask > 0.5, -1e9)
            cur_stop = c_stop

        if s_idx >= min_c:
            comb = torch.cat([cur_logits, cur_stop], dim=0)
            lps = F.log_softmax(comb, dim=0)
            ps = lps.exp()
            valid = comb > -1e8
            total_lp = total_lp + lps[picked]
            total_ent = total_ent - (ps[valid] * lps[valid]).sum()
        else:
            lps = F.log_softmax(cur_logits, dim=0)
            ps = lps.exp()
            valid = cur_logits > -1e8
            total_lp = total_lp + lps[picked]
            total_ent = total_ent - (ps[valid] * lps[valid]).sum()
        sel_mask = sel_mask.clone()
        sel_mask[picked] = 1.0

    # If stopped before max_c, add stop log-prob at final step
    if len(chosen) < max_c and len(chosen) >= min_c:
        frac_t = torch.tensor([[float(len(chosen)) / max(max_c, 1)]], device=device, dtype=state_h.dtype)
        c_logits, c_stop = model.forward_conditional_subset(
            state_h, opt_ctx, num_options, opt_logits, sel_mask, frac_t
        )
        cur_logits = (c_logits + guard_delta).masked_fill(sel_mask > 0.5, -1e9)
        comb = torch.cat([cur_logits, c_stop], dim=0)
        lps = F.log_softmax(comb, dim=0)
        ps = lps.exp()
        valid = comb > -1e8
        total_lp = total_lp + lps[-1]
        total_ent = total_ent - (ps[valid] * lps[valid]).sum()

    return total_lp, total_ent, values[0], eff_logits, stop_logits[0]


@torch.no_grad()
def _sample_rollout_action(
    model: PTCGPolicyNet,
    obs: dict[str, Any],
    default_deck: list[int],
    rng: np.random.Generator,
    device: torch.device,
    temperature: float = 1.0,
) -> tuple[list[int], dict[str, Any] | None]:
    """Sample a legal action during live RL rollout and return optional transition record."""
    sel = obs.get("select")
    cur = obs.get("current")
    if sel is None or cur is None:
        return list(default_deck), None

    opts = sel.get("option") or []
    n_opts = len(opts)
    min_c = int_or(sel, "minCount", 0)
    max_c = int_or(sel, "maxCount", 1)
    if n_opts == 0 or max_c <= 0:
        return [], None
    eff_min = max(0, min(min_c, n_opts))
    eff_max = max(eff_min, min(max_c, n_opts))
    if n_opts == 1 and eff_min == 1 and eff_max >= 1:
        return [0], None

    feats = extract_decision_features(obs)
    state_vec = torch.from_numpy(feats["state_vec"][: model.state_dim]).unsqueeze(0).to(device)
    hand_cards = torch.from_numpy(feats["hand_cards"]).unsqueeze(0).to(device)
    board_cards = torch.from_numpy(feats["board_cards"]).unsqueeze(0).to(device)
    select_ctx = torch.tensor([feats["select_ctx"]], dtype=torch.long, device=device)
    select_type = torch.tensor([feats["select_type"]], dtype=torch.long, device=device)
    num_options = torch.tensor([n_opts], dtype=torch.long, device=device)
    opt_feats = torch.from_numpy(feats["opt_feats"][:, : model.opt_dim]).to(device)
    opt_card_ids = torch.from_numpy(feats["opt_card_ids"]).to(device)
    opt_target_card_ids = torch.from_numpy(feats["opt_target_card_ids"]).to(device)
    opt_attack_ids = torch.from_numpy(feats["opt_attack_ids"]).to(device)

    opt_logits, stop_logits, values, state_h, opt_ctx = model.forward_packed_with_value(
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

    raw_np = opt_logits.cpu().numpy()
    stype = int_or(sel, "type", 0)
    tac = int_or(cur, "turnActionCount", 0)
    guarded_np = PTCGPolicyNet._apply_tactical_guardrails(
        raw_np,
        opts,
        feats["opt_feats"],
        stype,
        tac,
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
    guard_delta_np = np.where(guarded_np < -1e8, -1e9, guarded_np - raw_np).astype(np.float32)
    guard_delta_t = torch.from_numpy(guard_delta_np).to(device)
    eff_logits = opt_logits + guard_delta_t

    chosen: list[int] = []
    if eff_min == 1 and eff_max == 1:
        scaled = eff_logits / max(temperature, 1e-3)
        probs = F.softmax(scaled, dim=0).cpu().numpy()
        probs = probs / max(probs.sum(), 1e-12)
        pick = int(rng.choice(n_opts, p=probs))
        chosen = [pick]
    else:
        sel_mask = torch.zeros(n_opts, device=device, dtype=eff_logits.dtype)
        for s_idx in range(eff_max):
            if s_idx == 0 or not model.use_cond_multi:
                cur_logits = eff_logits.masked_fill(sel_mask > 0.5, -1e9)
                cur_stop = stop_logits
            else:
                frac_t = torch.tensor([[float(s_idx) / max(eff_max, 1)]], device=device, dtype=state_h.dtype)
                c_logits, c_stop = model.forward_conditional_subset(
                    state_h, opt_ctx, num_options, opt_logits, sel_mask, frac_t
                )
                cur_logits = (c_logits + guard_delta_t).masked_fill(sel_mask > 0.5, -1e9)
                cur_stop = c_stop

            if s_idx >= eff_min:
                comb = torch.cat([cur_logits, cur_stop], dim=0)
                probs = F.softmax(comb / max(temperature, 1e-3), dim=0).cpu().numpy()
                probs = probs / max(probs.sum(), 1e-12)
                pick = int(rng.choice(n_opts + 1, p=probs))
                if pick == n_opts:
                    break
            else:
                probs = F.softmax(cur_logits / max(temperature, 1e-3), dim=0).cpu().numpy()
                probs = probs / max(probs.sum(), 1e-12)
                pick = int(rng.choice(n_opts, p=probs))
            chosen.append(pick)
            sel_mask[pick] = 1.0
        chosen.sort()

    rec = {
        "state_vec": state_vec.squeeze(0).cpu(),
        "hand_cards": hand_cards.squeeze(0).cpu(),
        "board_cards": board_cards.squeeze(0).cpu(),
        "select_ctx": int(feats["select_ctx"]),
        "select_type": int(feats["select_type"]),
        "num_options": n_opts,
        "opt_feats": opt_feats.cpu(),
        "opt_card_ids": opt_card_ids.cpu(),
        "opt_target_card_ids": opt_target_card_ids.cpu(),
        "opt_attack_ids": opt_attack_ids.cpu(),
        "guard_delta": torch.from_numpy(guard_delta_np),
        "min_c": eff_min,
        "max_c": eff_max,
        "chosen": chosen,
        "value": float(values[0].item()),
    }
    lp, _, _, _, _ = _compute_log_prob_and_entropy_single_sample(model, rec, device)
    rec["old_log_prob"] = float(lp.item())
    return chosen, rec


def _collect_rl_episode(
    model: PTCGPolicyNet,
    opp_fn: Any,
    model_deck: list[int],
    opp_deck: list[int],
    model_seat: int,
    rng: np.random.Generator,
    device: torch.device,
    temperature: float = 0.85,
    max_steps: int = 2500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one live match in `cg.game` and collect `(obs, action, reward, value)` trajectory."""
    import cg.game

    decks = [model_deck, opp_deck] if model_seat == 0 else [opp_deck, model_deck]
    obs, _ = cg.game.battle_start(decks[0], decks[1])
    if obs is None:
        return [], {"winner": -1, "steps": 0, "invalid": 1}

    traj: list[dict[str, Any]] = []
    steps = 0
    invalid = 0
    prev_my_prizes = 6
    prev_opp_prizes = 6

    try:
        while int_or(obs["current"], "result", -1) == -1 and steps < max_steps:
            who = int_or(obs["current"], "yourIndex", 0)
            sel = obs["select"]
            opts = sel.get("option") or []
            min_c = int_or(sel, "minCount", 0)
            max_c = int_or(sel, "maxCount", 1)

            if who == model_seat:
                players = obs["current"]["players"]
                cur_my_prizes = len(players[model_seat].get("prize") or [])
                cur_opp_prizes = len(players[1 - model_seat].get("prize") or [])
                prize_shaping = 0.15 * float(prev_opp_prizes - cur_opp_prizes) - 0.15 * float(prev_my_prizes - cur_my_prizes)
                if traj:
                    traj[-1]["reward"] += prize_shaping
                prev_my_prizes = cur_my_prizes
                prev_opp_prizes = cur_opp_prizes

                act, rec = _sample_rollout_action(
                    model, obs, model_deck, rng, device, temperature=temperature
                )
                if rec is not None:
                    rec["reward"] = 0.0
                    traj.append(rec)
            else:
                act = opp_fn(obs)

            is_valid = (
                isinstance(act, list)
                and all(isinstance(x, int) and 0 <= x < len(opts) for x in act)
                and (min_c <= len(act) <= min(max(0, max_c), len(opts)))
                and len(set(act)) == len(act)
            )
            if not is_valid:
                if who == model_seat:
                    invalid += 1
                act = list(range(min(max(0, min_c), len(opts))))

            obs = cg.game.battle_select(act)
            steps += 1

        winner = int_or(obs["current"], "result", -1)
    finally:
        cg.game.battle_finish()

    if winner == model_seat:
        term_r = 1.0
    elif winner == (1 - model_seat):
        term_r = -1.0
    elif winner == 2:
        term_r = 0.0
    else:
        term_r = -0.5

    if traj:
        traj[-1]["reward"] += term_r

    return traj, {"winner": winner, "steps": steps, "invalid": invalid, "term_reward": term_r}


def train_rl_policy(
    bc_checkpoint: str | Path = "checkpoints/best_bc_model.pt",
    checkpoint_dir: str | Path = "checkpoints",
    updates: int = 12,
    episodes_per_update: int = 18,
    ppo_epochs: int = 3,
    mini_batch_size: int = 64,
    lr: float = 3e-4,
    kl_coef: float = 0.05,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    clip_eps: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    eval_games: int = 18,
    seed: int = 42,
    device_str: str = "auto",
    resume_path: str | Path | None = None,
    promote_to_best: bool = True,
) -> dict[str, Any]:
    """Fine-tune `PTCGPolicyNet` via PPO + KL-to-BC Anchor against heuristic & self-play snapshots."""
    from src.evaluate import make_heuristic_baseline_agent, make_random_baseline_agent, play_single_match

    set_all_seeds(seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    init_ckpt_path = Path(bc_checkpoint)
    if not init_ckpt_path.exists():
        init_ckpt_path = ckpt_path / "best_model.pt"

    ref_model, default_deck, bc_ckpt = load_policy_from_checkpoint(init_ckpt_path, device=device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    model, _, _ = load_policy_from_checkpoint(init_ckpt_path, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    start_update = 1
    best_win_rate = -1.0
    history: list[dict[str, Any]] = []
    snapshot_pool: list[dict[str, Any]] = [
        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ]

    if resume_path is not None and Path(resume_path).exists():
        rckpt = torch.load(Path(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(rckpt["model_state_dict"])
        if "optimizer_state_dict" in rckpt:
            optimizer.load_state_dict(rckpt["optimizer_state_dict"])
        start_update = int(rckpt.get("rl_update", 0)) + 1
        best_win_rate = float(rckpt.get("best_win_rate", -1.0))
        history = list(rckpt.get("history", []))

    deck_pool: list[tuple[str, list[int]]] = [("default_deck", default_deck)]
    for dpath in (Path("decks/dragapult_deck.csv"), Path("decks/starmie_deck.csv")):
        if dpath.exists():
            dk = load_default_deck(dpath)
            if len(dk) == 60:
                deck_pool.append((dpath.stem, dk))

    # Measure initial BC win rate across the 3 decks vs heuristic_baseline
    def _eval_vs_heuristic(eval_model: PTCGPolicyNet, n_games: int, eval_seed: int) -> tuple[float, int]:
        eval_model.eval()
        wins = 0
        invalids = 0
        for g in range(n_games):
            _, dk = deck_pool[(g // 2) % len(deck_pool)]
            opp_fn = make_heuristic_baseline_agent(seed=eval_seed + g, default_deck=dk)

            def pol_fn(obs_d: dict[str, Any], _dk: list[int] = dk) -> list[int]:
                return eval_model.select_action(obs_d, default_deck=_dk, device=device)

            m_seat = g % 2
            res = play_single_match(pol_fn, opp_fn, dk, dk) if m_seat == 0 else play_single_match(opp_fn, pol_fn, dk, dk)
            if res["winner"] == m_seat:
                wins += 1
            invalids += res["invalid_actions"][m_seat]
        return wins / max(n_games, 1), invalids

    bc_eval_wr, _ = _eval_vs_heuristic(ref_model, eval_games, seed + 5000)
    if best_win_rate < 0.0:
        best_win_rate = bc_eval_wr
    print(f"[RL Init] BC baseline win rate vs heuristic ({eval_games} games, 3 decks): {bc_eval_wr*100:.2f}%")

    rng = np.random.default_rng(seed + start_update * 17)
    git_commit = get_git_commit()
    t0_all = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for upd in range(start_update, updates + 1):
        t_upd = time.perf_counter()
        model.eval()
        buffer: list[dict[str, Any]] = []
        ep_returns: list[float] = []
        ep_wins = 0

        for ep in range(episodes_per_update):
            _, dk_model = deck_pool[ep % len(deck_pool)]
            # Mix mirror & cross-deck matchups
            if ep % 3 == 2 and len(deck_pool) > 1:
                _, dk_opp = deck_pool[(ep + 1) % len(deck_pool)]
            else:
                dk_opp = dk_model
            model_seat = ep % 2

            roll = float(rng.random())
            if roll < 0.50:
                opp_fn = make_heuristic_baseline_agent(seed=seed + upd * 100 + ep, default_deck=dk_opp)
            elif roll < 0.85:
                snap_model = copy.deepcopy(ref_model)
                snap_sd = snapshot_pool[int(rng.integers(0, len(snapshot_pool)))]
                snap_model.load_state_dict({k: v.to(device) for k, v in snap_sd.items()})
                snap_model.eval()

                def opp_fn(obs_d: dict[str, Any], _m: PTCGPolicyNet = snap_model, _dk: list[int] = dk_opp) -> list[int]:
                    return _m.select_action(obs_d, default_deck=_dk, device=device)
            else:
                opp_fn = make_random_baseline_agent(seed=seed + upd * 100 + ep, default_deck=dk_opp)

            traj, info = _collect_rl_episode(
                model,
                opp_fn,
                dk_model,
                dk_opp,
                model_seat=model_seat,
                rng=rng,
                device=device,
                temperature=0.85,
            )
            if info["winner"] == model_seat:
                ep_wins += 1
            ep_returns.append(sum(r["reward"] for r in traj))

            # Compute GAE returns & advantages
            gae = 0.0
            next_val = 0.0
            for t in reversed(range(len(traj))):
                delta = traj[t]["reward"] + gamma * next_val - traj[t]["value"]
                gae = delta + gamma * gae_lambda * gae
                traj[t]["advantage"] = gae
                traj[t]["return"] = gae + traj[t]["value"]
                next_val = traj[t]["value"]
            buffer.extend(traj)

        if not buffer:
            continue

        advs = np.array([b["advantage"] for b in buffer], dtype=np.float32)
        adv_mean = float(advs.mean())
        adv_std = float(advs.std()) + 1e-8
        for b in buffer:
            b["norm_adv"] = (b["advantage"] - adv_mean) / adv_std

        model.train()
        tot_pi = tot_vf = tot_kl = tot_ent = 0.0
        n_mb = 0
        for _ in range(ppo_epochs):
            perm = rng.permutation(len(buffer))
            for start in range(0, len(buffer), mini_batch_size):
                mb_idx = perm[start:start + mini_batch_size]
                if len(mb_idx) == 0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                pi_losses = []
                vf_losses = []
                kl_losses = []
                ents = []

                for idx in mb_idx:
                    rec = buffer[int(idx)]
                    new_lp, ent, val, eff_logits, _ = _compute_log_prob_and_entropy_single_sample(
                        model, rec, device
                    )
                    with torch.no_grad():
                        _, _, _, ref_eff_logits, _ = _compute_log_prob_and_entropy_single_sample(
                            ref_model, rec, device
                        )
                    valid = eff_logits > -1e8
                    if valid.sum() > 0:
                        ref_p = F.softmax(ref_eff_logits[valid], dim=0)
                        new_lp_all = F.log_softmax(eff_logits[valid], dim=0)
                        ref_lp_all = F.log_softmax(ref_eff_logits[valid], dim=0)
                        kl = (ref_p * (ref_lp_all - new_lp_all)).sum()
                    else:
                        kl = eff_logits.new_zeros(())

                    ratio = torch.exp((new_lp - rec["old_log_prob"]).clamp(-10.0, 10.0))
                    adv_t = torch.tensor(rec["norm_adv"], dtype=eff_logits.dtype, device=device)
                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
                    pi_losses.append(-torch.min(surr1, surr2))

                    ret_t = torch.tensor(rec["return"], dtype=eff_logits.dtype, device=device)
                    vf_losses.append(F.smooth_l1_loss(val, ret_t))
                    kl_losses.append(kl)
                    ents.append(ent)

                loss_pi = torch.stack(pi_losses).mean()
                loss_vf = torch.stack(vf_losses).mean()
                loss_kl = torch.stack(kl_losses).mean()
                mean_ent = torch.stack(ents).mean()
                loss = loss_pi + vf_coef * loss_vf + kl_coef * loss_kl - ent_coef * mean_ent

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                tot_pi += float(loss_pi.item())
                tot_vf += float(loss_vf.item())
                tot_kl += float(loss_kl.item())
                tot_ent += float(mean_ent.item())
                n_mb += 1

        if upd % 3 == 0:
            snapshot_pool.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if len(snapshot_pool) > 5:
                snapshot_pool.pop(0)

        eval_wr, eval_inv = _eval_vs_heuristic(model, eval_games, seed + 6000 + upd * 50)
        upd_time = time.perf_counter() - t_upd
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )

        row = {
            "rl_update": upd,
            "rollout_steps": len(buffer),
            "rollout_win_rate": round(ep_wins / max(episodes_per_update, 1), 4),
            "mean_episode_return": round(float(np.mean(ep_returns)), 4),
            "eval_vs_heuristic_wr": round(eval_wr, 4),
            "eval_invalid_actions": eval_inv,
            "policy_loss": round(tot_pi / max(n_mb, 1), 4),
            "value_loss": round(tot_vf / max(n_mb, 1), 4),
            "kl_to_bc": round(tot_kl / max(n_mb, 1), 4),
            "entropy": round(tot_ent / max(n_mb, 1), 4),
            "update_time_sec": round(upd_time, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
        }
        history.append(row)
        print(
            f"[RL {upd:02d}/{updates:02d}] steps={len(buffer)} roll_wr={row['rollout_win_rate']*100:.1f}% "
            f"eval_heur_wr={eval_wr*100:.1f}% (bc={bc_eval_wr*100:.1f}%) "
            f"kl={row['kl_to_bc']:.4f} vf={row['value_loss']:.4f} time={upd_time:.1f}s"
        )

        rl_payload = {
            "epoch": bc_ckpt.get("epoch", 0),
            "rl_update": upd,
            "best_win_rate": max(best_win_rate, eval_wr),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "state_dim": model.state_dim,
                "opt_dim": model.opt_dim,
                "hidden_dim": model.hidden_dim,
                "use_set_attn": model.use_set_attn,
                "use_cond_multi": model.use_cond_multi,
                "rl_lr": lr,
                "kl_coef": kl_coef,
                "seed": seed,
            },
            "dataset_ref": DATASET_REF,
            "dataset_version": DATASET_VERSION,
            "git_commit": git_commit,
            "default_deck": default_deck,
            "history": history,
        }
        torch.save(rl_payload, ckpt_path / "last_rl_model.pt")
        if eval_inv == 0 and eval_wr >= best_win_rate:
            best_win_rate = eval_wr
            rl_payload["best_win_rate"] = best_win_rate
            torch.save(rl_payload, ckpt_path / "best_rl_model.pt")
            if promote_to_best and eval_wr >= bc_eval_wr:
                shutil.copy2(ckpt_path / "best_rl_model.pt", ckpt_path / "best_model.pt")

    total_time = time.perf_counter() - t0_all
    rl_summary = {
        "bc_checkpoint": str(init_ckpt_path.resolve()),
        "bc_eval_win_rate": round(bc_eval_wr, 4),
        "best_rl_eval_win_rate": round(best_win_rate, 4),
        "total_rl_time_sec": round(total_time, 3),
        "device": str(device),
        "peak_vram_mb": round(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0,
            2,
        ),
        "best_rl_checkpoint": str((ckpt_path / "best_rl_model.pt").resolve()),
        "last_rl_checkpoint": str((ckpt_path / "last_rl_model.pt").resolve()),
        "history": history,
    }
    with open(ckpt_path / "rl_summary.json", "w", encoding="utf-8") as f:
        json.dump(rl_summary, f, indent=2, ensure_ascii=False)
    return rl_summary


def load_policy_from_checkpoint(
    checkpoint_path: str | Path = "checkpoints/best_model.pt",
    device: str | torch.device = "cpu",
) -> tuple[PTCGPolicyNet, list[int], dict[str, Any]]:
    """Load trained `PTCGPolicyNet` (Phase 2 or Phase 3) and its 60-card deck from checkpoint."""
    dev = torch.device(device)
    ckpt = torch.load(Path(checkpoint_path), map_location=dev, weights_only=False)
    cfg = ckpt.get("config") or {}
    sd = ckpt["model_state_dict"]

    hidden_dim = int(cfg.get("hidden_dim", 128))
    card_emb_dim = int(cfg.get("card_emb_dim", 24))
    ctx_emb_dim = int(cfg.get("ctx_emb_dim", 12))
    type_emb_dim = int(cfg.get("type_emb_dim", 8))

    # Auto-detect Phase 2 vs Phase 3 dimensions for strict backward compatibility
    in_state_w = sd["state_encoder.0.weight"].shape[1]
    inferred_state_dim = in_state_w - (5 * card_emb_dim + ctx_emb_dim + type_emb_dim)
    state_dim = int(cfg.get("state_dim", inferred_state_dim))

    in_opt_w = sd["opt_encoder.0.weight"].shape[1]
    atk_emb_dim = int(cfg.get("atk_emb_dim", 16))
    inferred_opt_dim = in_opt_w - (2 * card_emb_dim + atk_emb_dim)
    opt_dim = int(cfg.get("opt_dim", inferred_opt_dim))

    use_set_attn = bool(cfg.get("use_set_attn", "opt_attn.in_proj_weight" in sd))
    use_cond_multi = bool(cfg.get("use_cond_multi", "cond_scorer.0.weight" in sd))

    model = PTCGPolicyNet(
        state_dim=state_dim,
        opt_dim=opt_dim,
        hidden_dim=hidden_dim,
        dropout=float(cfg.get("dropout", 0.1)),
        use_set_attn=use_set_attn,
        use_cond_multi=use_cond_multi,
    ).to(dev)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    allowed_missing = {k for k in missing if k.startswith("value_head.")}
    disallowed_missing = set(missing) - allowed_missing
    if disallowed_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint key mismatch! disallowed_missing={sorted(disallowed_missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    model.eval()
    deck = ckpt.get("default_deck") or load_default_deck("deck.csv")
    return model, list(deck), ckpt

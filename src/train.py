"""Training & checkpoint resume pipeline for CABT Pokémon TCG Policy Network."""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src import DATASET_REF, DATASET_VERSION
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
    min_count = data["min_count"][idx_t]
    max_count = data["max_count"][idx_t]
    num_options = data["num_options"][idx_t].to(device)

    offsets = data["offsets"]
    opt_ConstantSlices = []
    for i in indices:
        s = int(offsets[i].item())
        e = int(offsets[i + 1].item())
        opt_ConstantSlices.append(torch.arange(s, e, dtype=torch.long))
    flat_idx = torch.cat(opt_ConstantSlices, dim=0)

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
) -> torch.Tensor:
    """Vectorized listwise cross-entropy + relative stop-threshold BCE loss."""
    device = opt_logits.device
    batch_size = int(num_options.size(0))
    max_opts = int(num_options.max().item())

    # 1. Relative stop-threshold BCE loss: chosen options should exceed stop_logit
    stop_rep = torch.repeat_interleave(stop_logits, num_options, dim=0)
    rel_logits = opt_logits - stop_rep
    pos_weight = ((1.0 - chosen_mask).sum() / chosen_mask.sum().clamp(min=1.0)).clamp(1.0, 5.0)
    bce_loss = F.binary_cross_entropy_with_logits(rel_logits, chosen_mask, pos_weight=pos_weight)

    # 2. Segmented listwise cross-entropy over legal options
    padded_logits = torch.full((batch_size, max_opts), -1e9, device=device, dtype=opt_logits.dtype)
    padded_targets = torch.zeros((batch_size, max_opts), device=device, dtype=opt_logits.dtype)

    row_ids = torch.repeat_interleave(torch.arange(batch_size, device=device), num_options)
    cum_opts = torch.cumsum(num_options, dim=0) - num_options
    col_ids = torch.arange(opt_logits.size(0), device=device) - torch.repeat_interleave(cum_opts, num_options)

    padded_logits[row_ids, col_ids] = opt_logits
    padded_targets[row_ids, col_ids] = chosen_mask

    target_sums = padded_targets.sum(dim=1, keepdim=True)
    has_pos = (target_sums.squeeze(-1) > 0)
    if has_pos.any():
        norm_targets = padded_targets[has_pos] / target_sums[has_pos].clamp(min=1.0)
        log_probs = F.log_softmax(padded_logits[has_pos], dim=-1)
        ce_loss = -(norm_targets * log_probs).sum(dim=-1).mean()
    else:
        ce_loss = torch.tensor(0.0, device=device)

    return ce_loss + 0.5 * bce_loss


@torch.no_grad()
def evaluate_packed_split(
    model: PTCGPolicyNet,
    data: dict[str, Any],
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Evaluate exact-match accuracy, single/multi-select metrics, and loss."""
    model.eval()
    n_total = int(data["num_samples"])
    all_indices = np.arange(n_total)

    total_loss = 0.0
    n_batches = 0

    exact_all = 0
    exact_nontrivial = 0
    n_nontrivial = 0

    single_correct = 0
    single_total = 0
    multi_exact = 0
    multi_jaccard_sum = 0.0
    multi_total = 0

    turn0_correct = 0
    turn0_total = 0
    main_correct = 0
    main_total = 0

    invalid_predictions = 0

    turns_np = data["turn"].numpy()
    forced_np = data["is_forced"].numpy()

    for start in range(0, n_total, batch_size):
        batch_idx = all_indices[start:start + batch_size]
        batch = slice_packed_batch(data, batch_idx, device)
        opt_logits, stop_logits = model.forward_packed(
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
            opt_logits, stop_logits, batch["num_options"], batch["chosen_mask"]
        )
        total_loss += float(loss.item())
        n_batches += 1

        logits_cpu = opt_logits.cpu().numpy()
        stop_cpu = stop_logits.cpu().numpy()
        num_opts_cpu = batch["num_options"].cpu().numpy()
        min_c_cpu = batch["min_count"].numpy()
        max_c_cpu = batch["max_count"].numpy()

        cursor = 0
        for bi, global_i in enumerate(batch_idx):
            k = int(num_opts_cpu[bi])
            s_logits = logits_cpu[cursor:cursor + k]
            cursor += k
            min_c = int(min_c_cpu[bi])
            max_c = int(max_c_cpu[bi])

            pred = PTCGPolicyNet.decode_action_from_logits(
                s_logits, float(stop_cpu[bi]), min_c, max_c
            )
            gold = batch["chosen_indices"][bi]

            # Check action legality
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
        "exact_match_nontrivial": exact_nontrivial / max(n_nontrivial, 1),
        "single_select_acc": single_correct / max(single_total, 1),
        "single_select_total": single_total,
        "multi_select_exact_acc": multi_exact / max(multi_total, 1),
        "multi_select_jaccard": multi_jaccard_sum / max(multi_total, 1),
        "multi_select_total": multi_total,
        "turn0_setup_acc": turn0_correct / max(turn0_total, 1),
        "turn0_setup_total": turn0_total,
        "main_game_acc": main_correct / max(main_total, 1),
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
) -> dict[str, Any]:
    """Train the imitation learning policy and save checkpoints with full resume state."""
    set_all_seeds(seed)
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    data_path = Path(data_dir)
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    train_data = torch.load(data_path / "train.pt", weights_only=False)
    val_data = torch.load(data_path / "val.pt", weights_only=False)

    model = PTCGPolicyNet(hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.05)

    start_epoch = 1
    best_val_acc = -1.0
    history: list[dict[str, Any]] = []
    default_deck = load_default_deck("deck.csv")

    if resume_path is not None and Path(resume_path).exists():
        ckpt = torch.load(Path(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_acc = float(ckpt.get("best_val_acc", -1.0))
        history = list(ckpt.get("history", []))
        if "default_deck" in ckpt and len(ckpt["default_deck"]) == 60:
            default_deck = list(ckpt["default_deck"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    n_train = int(train_data["num_samples"])
    rng = np.random.default_rng(seed + start_epoch)
    git_commit = get_git_commit()
    t_start_all = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t_ep = time.perf_counter()
        perm = rng.permutation(n_train)
        ep_loss = 0.0
        n_steps = 0

        for start in range(0, n_train, batch_size):
            batch_idx = perm[start:start + batch_size]
            batch = slice_packed_batch(train_data, batch_idx, device)

            optimizer.zero_grad(set_to_none=True)
            opt_logits, stop_logits = model.forward_packed(
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
                opt_logits, stop_logits, batch["num_options"], batch["chosen_mask"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += float(loss.item())
            n_steps += 1

        scheduler.step()
        train_time = time.perf_counter() - t_ep
        samples_per_sec = n_train / max(train_time, 1e-6)
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )

        val_metrics = evaluate_packed_split(model, val_data, device, batch_size=batch_size * 2)
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
            "val_invalid_actions": val_metrics["invalid_actions"],
            "epoch_time_sec": round(train_time, 4),
            "samples_per_sec": round(samples_per_sec, 1),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        print(
            f"[Epoch {epoch:02d}/{epochs:02d}] "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_acc(nontrivial)={val_acc*100:.2f}% val_acc(all)={row['val_exact_all']*100:.2f}% "
            f"multi_acc={row['val_multi_exact_acc']*100:.2f}% "
            f"speed={samples_per_sec:.0f} samp/s vram={peak_vram_mb:.1f}MB"
        )

        ckpt_payload = {
            "epoch": epoch,
            "best_val_acc": max(best_val_acc, val_acc),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {
                "hidden_dim": hidden_dim,
                "dropout": dropout,
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
            torch.save(ckpt_payload, ckpt_path / "best_model.pt")

    total_time = time.perf_counter() - t_start_all
    summary = {
        "best_val_acc_nontrivial": best_val_acc,
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


def load_policy_from_checkpoint(
    checkpoint_path: str | Path = "checkpoints/best_model.pt",
    device: str | torch.device = "cpu",
) -> tuple[PTCGPolicyNet, list[int], dict[str, Any]]:
    """Load trained `PTCGPolicyNet` and its default 60-card deck from checkpoint."""
    dev = torch.device(device)
    ckpt = torch.load(Path(checkpoint_path), map_location=dev, weights_only=False)
    cfg = ckpt.get("config") or {}
    model = PTCGPolicyNet(
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    deck = ckpt.get("default_deck") or load_default_deck("deck.csv")
    return model, list(deck), ckpt

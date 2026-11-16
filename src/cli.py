"""Command-line interface for Pokémon TCG Battle Agent pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dataset import download_replays_from_kaggle, preprocess_replays
from src.evaluate import (
    evaluate_in_simulator,
    make_random_baseline_agent,
    play_single_match,
)
from src.train import load_default_deck, train_policy


def cmd_download(args: argparse.Namespace) -> None:
    p = download_replays_from_kaggle(out_dir=args.out_dir, num_top=args.num_top)
    print(f"Downloaded replays to: {p}")


def cmd_preprocess(args: argparse.Namespace) -> None:
    summary = preprocess_replays(
        replay_dir=args.replay_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        drop_forced_in_train=not args.keep_forced_in_train,
    )
    print(json.dumps(summary["overall_counts"], indent=2))
    print("Split counts:")
    for k, v in summary["split_counts"].items():
        print(f"  {k}: episodes={v.get('episodes')}, kept_decisions={v.get('kept_decisions')}")


def cmd_smoke_test(args: argparse.Namespace) -> None:
    from agent import agent as policy_entrypoint, load_agent_checkpoint

    deck = load_default_deck(args.deck)
    rand0 = make_random_baseline_agent(seed=args.seed, default_deck=deck)
    rand1 = make_random_baseline_agent(seed=args.seed + 1, default_deck=deck)
    r_base = play_single_match(rand0, rand1, deck, deck)
    print("Smoke test (random vs random):", json.dumps(r_base))

    if Path(args.checkpoint).exists():
        load_agent_checkpoint(args.checkpoint)
        r_model = play_single_match(policy_entrypoint, rand1, deck, deck)
        print("Smoke test (checkpoint agent vs random):", json.dumps(r_model))


def cmd_train(args: argparse.Namespace) -> None:
    summary = train_policy(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        seed=args.seed,
        device_str=args.device,
        resume_path=args.resume,
    )
    print(
        f"Training finished! Best val nontrivial acc: {summary['best_val_acc_nontrivial']*100:.2f}%, "
        f"Peak VRAM: {summary['peak_vram_mb']} MB, Checkpoint: {summary['best_checkpoint']}"
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    res = evaluate_in_simulator(
        checkpoint_path=args.checkpoint,
        num_games_per_opponent=args.games,
        seed=args.seed,
        out_file=args.out_file,
    )
    print("Offline Test Metrics:", json.dumps(res["offline_test_metrics"], indent=2))
    print("Simulator Match Results:", json.dumps(res["simulator_matches"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pokémon TCG Battle Agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download-data", help="Download manifest.csv and replay JSONs from Kaggle")
    p_dl.add_argument("--out-dir", default="data/replays")
    p_dl.add_argument("--num-top", type=int, default=30)
    p_dl.set_defaults(func=cmd_download)

    p_pre = sub.add_parser("preprocess", help="Stream-parse replays and split by episode_id")
    p_pre.add_argument("--replay-dir", default="data/replays")
    p_pre.add_argument("--out-dir", default="data/processed")
    p_pre.add_argument("--seed", type=int, default=42)
    p_pre.add_argument("--train-ratio", type=float, default=0.70)
    p_pre.add_argument("--val-ratio", type=float, default=0.15)
    p_pre.add_argument("--keep-forced-in-train", action="store_true")
    p_pre.set_defaults(func=cmd_preprocess)

    p_smk = sub.add_parser("smoke-test", help="Run CPU simulator smoke test matches")
    p_smk.add_argument("--deck", default="deck.csv")
    p_smk.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p_smk.add_argument("--seed", type=int, default=42)
    p_smk.set_defaults(func=cmd_smoke_test)

    p_trn = sub.add_parser("train", help="Train candidate-scoring policy network")
    p_trn.add_argument("--data-dir", default="data/processed")
    p_trn.add_argument("--checkpoint-dir", default="checkpoints")
    p_trn.add_argument("--epochs", type=int, default=15)
    p_trn.add_argument("--batch-size", type=int, default=128)
    p_trn.add_argument("--lr", type=float, default=2e-3)
    p_trn.add_argument("--weight-decay", type=float, default=1e-4)
    p_trn.add_argument("--hidden-dim", type=int, default=128)
    p_trn.add_argument("--dropout", type=float, default=0.1)
    p_trn.add_argument("--seed", type=int, default=42)
    p_trn.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p_trn.add_argument("--resume", default=None)
    p_trn.set_defaults(func=cmd_train)

    p_evl = sub.add_parser("evaluate", help="Evaluate checkpoint on held-out test split and in CABT simulator")
    p_evl.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p_evl.add_argument("--games", type=int, default=40)
    p_evl.add_argument("--seed", type=int, default=42)
    p_evl.add_argument("--out-file", default="checkpoints/eval_results.json")
    p_evl.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

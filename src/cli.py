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
from src.train import load_default_deck, train_policy, train_rl_policy


def cmd_download(args: argparse.Namespace) -> None:
    p = download_replays_from_kaggle(
        out_dir=args.out_dir,
        num_top=args.num_top,
        num_strided=args.num_strided,
        max_episodes=args.max_episodes,
    )
    print(f"Downloaded replays to: {p}")


def cmd_preprocess(args: argparse.Namespace) -> None:
    summary = preprocess_replays(
        replay_dir=args.replay_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        drop_forced_in_train=not args.keep_forced_in_train,
        shard_size_samples=args.shard_size,
    )
    print(json.dumps(summary["overall_counts"], indent=2))
    print("Split counts:")
    for k, v in summary["split_counts"].items():
        sh = summary["shard_summaries"][k]
        print(
            f"  {k}: episodes={v.get('episodes')}, kept_decisions={v.get('kept_decisions')}, "
            f"shards={sh['num_shards']}"
        )


def cmd_smoke_test(args: argparse.Namespace) -> None:
    from agent import agent as policy_entrypoint, load_agent_checkpoint

    deck = load_default_deck(args.deck)
    rand0 = make_random_baseline_agent(seed=args.seed, default_deck=deck)
    rand1 = make_random_baseline_agent(seed=args.seed + 1, default_deck=deck)
    r_base = play_single_match(rand0, rand1, deck, deck)
    r_base_slim = {k: v for k, v in r_base.items() if k != "decision_latencies_ms"}
    print("Smoke test (random vs random):", json.dumps(r_base_slim))

    if Path(args.checkpoint).exists():
        load_agent_checkpoint(args.checkpoint)
        r_model = play_single_match(policy_entrypoint, rand1, deck, deck)
        r_model_slim = {k: v for k, v in r_model.items() if k != "decision_latencies_ms"}
        print("Smoke test (checkpoint agent vs random):", json.dumps(r_model_slim))


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
    if args.rl_finetune:
        rl_summary = train_rl_policy(
            bc_checkpoint=summary["best_checkpoint"],
            checkpoint_dir=args.checkpoint_dir,
            updates=args.rl_updates,
            episodes_per_update=args.rl_episodes,
            lr=args.rl_lr,
            kl_coef=args.rl_kl_coef,
            eval_games=args.rl_eval_games,
            seed=args.seed,
            device_str=args.device,
        )
        print(
            f"RL fine-tuning finished! BC wr={rl_summary['bc_eval_win_rate']*100:.1f}% -> "
            f"Best RL wr={rl_summary['best_rl_eval_win_rate']*100:.1f}%"
        )


def cmd_train_rl(args: argparse.Namespace) -> None:
    summary = train_rl_policy(
        bc_checkpoint=args.bc_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        lr=args.lr,
        kl_coef=args.kl_coef,
        eval_games=args.eval_games,
        seed=args.seed,
        device_str=args.device,
        resume_path=args.resume,
        promote_to_best=not args.no_promote,
    )
    print(
        f"RL fine-tuning finished! BC wr={summary['bc_eval_win_rate']*100:.1f}% -> "
        f"Best RL wr={summary['best_rl_eval_win_rate']*100:.1f}%, "
        f"Checkpoint: {summary['best_rl_checkpoint']}"
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    res = evaluate_in_simulator(
        checkpoint_path=args.checkpoint,
        num_games_per_opponent=args.games,
        cross_deck_games_per_pair=args.cross_games,
        seed=args.seed,
        out_file=args.out_file,
        data_dir=args.data_dir,
    )
    print("Offline Test Metrics:", json.dumps(res["offline_test_metrics"], indent=2))
    print("Simulator Match Results:", json.dumps(res["simulator_matches"], indent=2))
    print("Cross-Deck Matrix:", json.dumps(res["cross_deck_matrix"], indent=2))
    print("CPU Latency (ms):", json.dumps(res["cpu_latency_ms"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pokémon TCG Battle Agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download-data", help="Download manifest.csv and replay JSONs from Kaggle")
    p_dl.add_argument("--out-dir", default="data/replays")
    p_dl.add_argument("--num-top", type=int, default=None)
    p_dl.add_argument("--num-strided", type=int, default=None)
    p_dl.add_argument("--max-episodes", type=int, default=None)
    p_dl.set_defaults(func=cmd_download)

    p_pre = sub.add_parser("preprocess", help="Stream-parse replays and split by episode_id into shards")
    p_pre.add_argument("--replay-dir", default="data/replays")
    p_pre.add_argument("--out-dir", default="data/processed")
    p_pre.add_argument("--seed", type=int, default=42)
    p_pre.add_argument("--train-ratio", type=float, default=0.70)
    p_pre.add_argument("--val-ratio", type=float, default=0.15)
    p_pre.add_argument("--shard-size", type=int, default=15000)
    p_pre.add_argument("--keep-forced-in-train", action="store_true")
    p_pre.set_defaults(func=cmd_preprocess)

    p_smk = sub.add_parser("smoke-test", help="Run CPU simulator smoke test matches")
    p_smk.add_argument("--deck", default="deck.csv")
    p_smk.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p_smk.add_argument("--seed", type=int, default=42)
    p_smk.set_defaults(func=cmd_smoke_test)

    p_trn = sub.add_parser("train", help="Train candidate-scoring policy network (BC + optional RL)")
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
    p_trn.add_argument("--rl-finetune", action="store_true")
    p_trn.add_argument("--rl-updates", type=int, default=10)
    p_trn.add_argument("--rl-episodes", type=int, default=18)
    p_trn.add_argument("--rl-lr", type=float, default=3e-4)
    p_trn.add_argument("--rl-kl-coef", type=float, default=0.05)
    p_trn.add_argument("--rl-eval-games", type=int, default=18)
    p_trn.set_defaults(func=cmd_train)

    p_rl = sub.add_parser("train-rl", help="PPO + KL-to-BC self-play/opponent-pool RL fine-tuning in CABT")
    p_rl.add_argument("--bc-checkpoint", default="checkpoints/best_bc_model.pt")
    p_rl.add_argument("--checkpoint-dir", default="checkpoints")
    p_rl.add_argument("--updates", type=int, default=12)
    p_rl.add_argument("--episodes-per-update", type=int, default=18)
    p_rl.add_argument("--ppo-epochs", type=int, default=3)
    p_rl.add_argument("--mini-batch-size", type=int, default=64)
    p_rl.add_argument("--lr", type=float, default=3e-4)
    p_rl.add_argument("--kl-coef", type=float, default=0.05)
    p_rl.add_argument("--eval-games", type=int, default=18)
    p_rl.add_argument("--seed", type=int, default=42)
    p_rl.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p_rl.add_argument("--resume", default=None)
    p_rl.add_argument("--no-promote", action="store_true")
    p_rl.set_defaults(func=cmd_train_rl)

    p_evl = sub.add_parser("evaluate", help="Evaluate checkpoint on held-out test split, mirror & cross-deck simulator matches")
    p_evl.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p_evl.add_argument("--data-dir", default="data/processed")
    p_evl.add_argument("--games", type=int, default=42)
    p_evl.add_argument("--cross-games", type=int, default=6)
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

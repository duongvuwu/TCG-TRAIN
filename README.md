# Pokémon TCG Battle Agent (CABT Simulator — Kaggle Playground)

A legal-action battle agent for the **Pokémon TCG (`cabt` / `cg`)** simulator in the Kaggle Playground competition (*The Pokémon Company PTCG AI Battle Challenge Playground*), trained from official replay data (`2026-11-13`, Version 1) with a sharded, zero-leakage imitation learning + PPO self-play RL pipeline and live mirror/cross-deck evaluation in the simulator.

---

## 1. Architecture & Pipeline Overview

The project provides a self-contained CLI pipeline (`src/cli.py`) designed for memory-bounded CPU preprocessing, GPU training (local or Colab Pro), RL fine-tuning in `cg.game`, and automated offline/simulator evaluation:

1. **Scalable Data Download (`download-data`)**: Downloads `manifest.csv` (`1,644` episodes, `2026-11-13-v1`) and replay JSON files from Kaggle with `--num-top`, `--num-strided`, and `--max-episodes` (up to all `1,644` episodes) for diverse meta deck coverage.
2. **Sharded Streaming Preprocessing & Episode-Level Splitting (`preprocess`)**:
   - Infers static fallback metadata (`infer_fallback_metadata_from_replays`) for post-`cg.dll` card/attack IDs (`cardId > 1267`, `attackId > 1556`).
   - Streams one replay JSON at a time, aligns pre-decision observations `steps[t][seat]["observation"]` with chosen actions `steps[t + 1][seat]["action"]`, and writes bounded shards (`train_shard_*.pt`, `val_shard_*.pt`, `test_shard_*.pt` + `*_shards.json`) so Peak CPU RAM stays `O(1 shard)`.
   - Enforces **Zero Leakage** (`validate_no_leakage`): verifies no future-step access and confirms opponent hidden state remains masked (`players[1 - seat]["hand"] is None`).
   - Splits `train / val / test` **strictly by `episode_id`** before unpacking decision steps.
3. **Imitation Learning Policy (`PTCGPolicyNet` — `src/model.py`)**:
   - **Enriched State & Option Encoders**: `112`-dimensional dense state vector (turn flags, `SelectType`, `minCount`/`maxCount`, active & per-slot bench HP/damage/energy/stage, public discard counts, and causal turn action history from `obs["logs"]`) and `64`-dimensional option features (stage, energy-type matching, energy deficit, damage/HP ratio, duplicate counts, `AreaType.STADIUM = 7`, `2x` Weakness, Lethal KO) with Entity Embeddings (`MAX_CARD_ID = 2048`, `MAX_ATTACK_ID = 2048`).
   - **Set-Context Self-Attention & Conditional Multi-Select**: `opt_attn` (4-head self-attention across candidate options) and `cond_scorer` / `cond_stop_head` for context-aware sequential multi-select decoding (`2 <= maxCount <= 6`).
   - **Value Head & Backward Compatibility**: Built-in `value_head` for RL advantage estimation, with automatic shape/key detection in `load_policy_from_checkpoint` for strict backward compatibility with Phase 2 (`64/48`) checkpoints.
4. **RL / Self-Play Fine-Tuning (`train-rl` / `--rl-finetune`)**:
   - Fine-tunes the BC policy via **PPO + GAE + KL-to-BC Anchor** directly inside `cg.game` across all 3 decks (`default_deck`, `dragapult_deck`, `starmie_deck`) against `heuristic_baseline`, self-play checkpoint snapshots, and `cabt_random_baseline`, with full `--resume` support.
5. **Mirror, Cross-Deck Matrix & Latency Evaluation (`evaluate`, `smoke-test`)**:
   - Evaluates offline `val` and `test` shards, runs **fair mirror matches** (`decks_equal == True`) against `cabt_random_baseline` and `heuristic_baseline`, evaluates the full **3×3 Cross-Deck Matchup Matrix** vs `heuristic_baseline`, and reports CPU inference latency (`mean`, `p50`, `p95`, `p99`) while enforcing `invalid_actions == 0`, `runtime_errors == 0`, and `timeouts == 0`.

---

## 2. Repository Structure

| Path | Description |
|---|---|
| [`agent.py`](agent.py) | Standard entrypoint `agent(obs_dict: dict) -> list[int]` loading `checkpoints/best_model.pt` and guaranteeing legal CABT / Kaggle actions. |
| [`cg/`](cg/) | Python bindings (`api.py`, `game.py`, `sim.py`, `utils.py`) interfacing with the CABT C++ engine (native binaries `cg/*.dll`, `cg/*.so` are gitignored). |
| [`src/cli.py`](src/cli.py) | Unified CLI entrypoint: `download-data`, `preprocess`, `smoke-test`, `train`, `train-rl`, `evaluate`. |
| [`src/dataset.py`](src/dataset.py) | Diverse Kaggle downloader, streaming replay parser, sharded tensor storage (`*_shard_*.pt`), and episode-level `train/val/test` splitter. |
| [`src/features.py`](src/features.py) | `112`-dim state & `64`-dim option extraction, `FALLBACK_CARD_DB`/`FALLBACK_ATTACK_DB`, `infer_fallback_metadata_from_replays`, `int_or`, and `validate_no_leakage`. |
| [`src/model.py`](src/model.py) | `PTCGPolicyNet` (`opt_attn`, `cond_scorer`, `cond_stop_head`, `value_head`), `_get_deck_pokemon_profiles`, `_apply_tactical_guardrails`, and conditional decoding. |
| [`src/train.py`](src/train.py) | Sharded BC training, `compute_conditional_multi_loss`, PPO + KL-to-BC `train_rl_policy`, LR/RL `--resume`, and backward-compatible checkpoint loading. |
| [`src/evaluate.py`](src/evaluate.py) | Baseline agents, `play_single_match`, `evaluate_cross_deck_matrix`, `evaluate_in_simulator`, and CPU latency percentiles. |
| [`tests/test_pipeline.py`](tests/test_pipeline.py) | `pytest` suite (`15/15` unit & integration tests covering sharding, fallback metadata, `value_head`, Phase 2 compat, RL step/resume, cross-deck matrix, `agent.py`, and `BUG-1`..`BUG-6`). |
| [`deck.csv`](deck.csv), [`decks/`](decks/) | Valid 60-card test decks (`deck.csv`, `decks/dragapult_deck.csv`, `decks/starmie_deck.csv`). |
| [`notebooks/colab_train_eval.ipynb`](notebooks/colab_train_eval.ipynb) | Thin Colab Pro notebook supporting up to `1,644` episodes, sharded BC + RL fine-tuning, and Google Drive checkpoint sync. |
| [`AGENTS.md`](AGENTS.md) | Pipeline guidelines, agent/model role assignments, and Colab Pro execution rules. |
| [`REPORT_PHASE1_2.md`](REPORT_PHASE1_2.md) | Detailed Phase 1 & Phase 2 data, architecture, and benchmark report. |
| [`REPORT_PHASE3.md`](REPORT_PHASE3.md) | Detailed Phase 3 data scaling, conditional multi-select, RL self-play, and Phase 2 vs Phase 3 benchmark report. |
| [`REVIEW_REPORT.md`](REVIEW_REPORT.md) | Adversarial bug review (`BUG-1`..`BUG-7`) and regression verification report. |
| [`HANDOFF.md`](HANDOFF.md) | Project handoff summary covering environment, git state, and Phase 1–3 results. |

> **Git Note**: Downloaded replays (`data/`), native simulator binaries (`cg/*.dll`, `cg/*.so`), model checkpoints (`checkpoints/`), and credentials are excluded from Git via [`.gitignore`](.gitignore).

---

## 3. Benchmark Results Summary (Phase 2 vs Phase 3, `seed = 42`)

- **Phase 3 Dataset (`209` diverse episodes, `seed = 42`)**: `93,829` frames (`187,658` seat-steps) -> **`90,609` kept decisions** (`61,890` train across `5` shards / `13,970` val / `14,749` test; `5,707` multi-select decisions), `0` invalid action bounds, `100%` episode-disjoint splits.
- **Training & GPU Efficiency (`RTX 4050`, PyTorch `2.6.0+cu124`)**:
  - **BC (`20` epochs + `2` resume epochs)**: `3,126 – 5,595 samples/sec`, Peak VRAM **`88.00 MB`**, `best_bc_model.pt` at Epoch 22 (`val_exact_nontrivial = 60.55%`, `val_exact_all = 62.43%`, `val_multi_exact_acc = 62.39%`).
  - **RL Fine-Tuning (`train-rl`, `10` updates + `2` resume updates)**: Peak VRAM **`42.41 MB`**, improving fast-eval win rate vs Heuristic from `94.44%` (BC) to **`100.00%`** (`best_rl_model.pt` promoted to `checkpoints/best_model.pt`, Epoch 22 + RL Update 8).

| Metric | Phase 2 (`40` eps, Ep 21) | Phase 3 Final (`209` eps, Ep 22 + RL Upd 8) | Delta |
|---|---:|---:|---:|
| **Offline Test `exact_match_all`** | `55.06%` (`1,350/2,452`) | **`61.12%`** (`9,014/14,749`) | **`+6.06%`** |
| **Offline Test `exact_match_nontrivial`** | `52.40%` (`1,213/2,315`) | **`59.27%`** (`8,345/14,080`) | **`+6.87%`** |
| **Test `single_select_acc`** | `55.47%` (`1,283/2,313`) | **`61.05%`** (`8,407/13,770`) | **`+5.58%`** |
| **Test `multi_select_exact_acc`** | `48.20%` (`67/139`) | **`62.00%`** (`607/979`) | **`+13.80%`** |
| **Test `multi_select_jaccard`** | `60.38%` | **`73.89%`** | **`+13.51%`** |
| **Mirror vs `cabt_random_baseline` (`42` games)** | `83.33%` (`35-0-7`) | **`97.62%`** (`41-0-1`) | **`+14.29%`** |
| **Mirror vs `heuristic_baseline` (`42` games)** | `78.57%` (`33-0-9`) | **`92.86%`** (`39-0-3`) | **`+14.29%`** |
| — `default_deck` vs Heuristic (`14` games) | `92.86%` (`13-0-1`) | **`100.00%`** (`14-0-0`) | **`+7.14%`** |
| — `dragapult_deck` vs Heuristic (`14` games) | `85.71%` (`12-0-2`) | **`100.00%`** (`14-0-0`) | **`+14.29%`** |
| — `starmie_deck` vs Heuristic (`14` games) | `57.14%` (`8-0-6`) | **`78.57%`** (`11-0-3`) | **`+21.43%`** |
| **Cross-Deck Matrix vs Heuristic (`54` games)** | *(N/A)* | **`90.74%`** (`49-0-5`) | **New** |
| **CPU Inference Latency (`4,908` mirror decisions)** | `1.22–1.34 ms` mean | **`4.39 ms` mean / `14.48 ms` P99** | Within limit |
| **Invalid Actions / Runtime Errors / Timeouts** | `0 / 0 / 0` | **`0 / 0 / 0`** | **Zero faults** |
| **Automated Test Suite (`pytest`)** | `10/10 PASSED` | **`15/15 PASSED` (`27.05s`)** | All pass |

---

## 4. Quick Start (CLI & Colab Pro)

### Local CLI

```bash
# 1. Download manifest.csv and diverse replay JSONs from Kaggle (up to --max-episodes 1644)
python -m src.cli download-data --out-dir data/replays --num-top 100 --num-strided 100

# 2. Stream-preprocess replays on CPU into bounded shards and split by episode_id
python -m src.cli preprocess --replay-dir data/replays --out-dir data/processed --seed 42 --shard-size 15000

# 3. Train BC policy on GPU (and verify monotonic LR checkpoint resume)
python -m src.cli train --data-dir data/processed --checkpoint-dir checkpoints --epochs 20 --batch-size 128 --lr 0.002 --seed 42 --device cuda
python -m src.cli train --data-dir data/processed --checkpoint-dir checkpoints --epochs 22 --batch-size 128 --lr 0.002 --seed 42 --device cuda --resume checkpoints/last_model.pt

# 4. Run PPO + KL-to-BC RL fine-tuning across all 3 decks (and verify RL resume)
python -m src.cli train-rl --bc-checkpoint checkpoints/best_bc_model.pt --checkpoint-dir checkpoints --updates 10 --episodes-per-update 18 --lr 0.0003 --kl-coef 0.05 --eval-games 18 --seed 42 --device cuda
python -m src.cli train-rl --bc-checkpoint checkpoints/best_bc_model.pt --checkpoint-dir checkpoints --updates 12 --episodes-per-update 18 --lr 0.0003 --kl-coef 0.05 --eval-games 18 --seed 42 --device cuda --resume checkpoints/last_rl_model.pt

# 5. Run CPU simulator smoke test, full mirror + cross-deck evaluation, and pytest
python -m src.cli smoke-test --deck deck.csv --checkpoint checkpoints/best_model.pt --seed 42
python -m src.cli evaluate --checkpoint checkpoints/best_model.pt --data-dir data/processed --games 42 --cross-games 6 --seed 42 --out-file checkpoints/eval_results.json
python -m pytest -p no:cacheprovider -v tests/test_pipeline.py
```

### Colab Pro
Open [`notebooks/colab_train_eval.ipynb`](notebooks/colab_train_eval.ipynb), select **Runtime -> Change runtime type -> T4 GPU** (or `L4` / High-RAM for full `1,644` episodes), and click **Runtime -> Run all** to execute the full sharded BC + RL pipeline and persist checkpoints to Google Drive.

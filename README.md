# Pokémon TCG Battle Agent (CABT Simulator — Kaggle Playground)

A legal-action battle agent for the **Pokémon TCG (`cabt` / `cg`)** simulator in the Kaggle Playground competition (*The Pokémon Company PTCG AI Battle Challenge Playground*), trained from official replay data (`2026-11-13`, Version 1) with an end-to-end, zero-leakage pipeline and live mirror-match evaluation in the simulator.

---

## 1. Architecture & Pipeline Overview

The project provides a self-contained CLI pipeline (`src/cli.py`) designed for memory-bounded CPU preprocessing, GPU training (local or Colab Pro), and automated offline/simulator evaluation:

1. **Data Download (`download-data`)**: Downloads `manifest.csv` (`1,644` episodes, `2026-11-13-v1`) and replay JSON files from Kaggle.
2. **Streaming Preprocessing & Episode-Level Splitting (`preprocess`)**:
   - Streams one replay JSON at a time to bound peak RAM usage.
   - Aligns pre-decision observations `steps[t][seat]["observation"]` with the resulting actions `steps[t + 1][seat]["action"]` (unwrapping both `list[int]` and `dict` action payloads).
   - Separates deck-selection steps (`t = 0`), turn-0 setup prompts (`turn = 0`), waiting turns (`INACTIVE`), and terminal steps (`DONE`).
   - Enforces **Zero Leakage** (`validate_no_leakage`): verifies no future-step access and confirms opponent hidden state remains masked (`players[1 - seat]["hand"] is None`).
   - Splits `train / val / test` **strictly by `episode_id`** before unpacking individual decision steps.
3. **Imitation Learning Policy (`PTCGPolicyNet` — `src/model.py`)**:
   - **State Encoder**: `64`-dimensional dense state vector (turn number, turn action count, per-turn action flags, `SelectType`, `minCount`, `maxCount`, Prize/Hand/Deck/Discard/Bench counts, active Pokémon attributes for both players, and hand card-type breakdown including `CardType.STADIUM = 4`) combined with Entity Embeddings (`MAX_CARD_ID = 2048`, `MAX_ATTACK_ID = 2048`).
   - **Option Encoder & Candidate Scorer**: `48`-dimensional option feature vector for each legal candidate `0 .. K-1` (covering `OptionType`, `AreaType` including `AreaType.STADIUM = 7`, card attributes, effective attack damage with `2x` Weakness, and a **Lethal KO** flag).
   - **Action Legality & Multi-Select Handling**: Scores only over the legal candidate list `0 .. K-1`, learns a stop threshold `stop_head` (`z_stop`) plus a virtual stop logit when `minCount == 0` (to support empty `[]` selections), safely handles `maxCount == 0`, and applies `_apply_tactical_guardrails` (using `cg.api` Pokémon/energy profiles) alongside an anti-loop guard (`turnActionCount >= 20` at `SelectType.MAIN == 0`).
4. **Offline & Simulator Evaluation (`evaluate`, `smoke-test`)**:
   - Measures Exact Match (overall, nontrivial, single-select, multi-select, zero-select, turn-0 setup, main-game) and multi-select Jaccard similarity on held-out `val`/`test` splits.
   - Runs **fair mirror matches** (`decks_equal == True`) across multiple decks (`default_deck`, `dragapult_deck`, `starmie_deck`), alternating `Seat 0` and `Seat 1`, against `cabt_random_baseline` and `heuristic_baseline`, tracking `wins / draws / losses / timeouts / invalid_actions / runtime_errors`.

---

## 2. Repository Structure

| Path | Description |
|---|---|
| [`agent.py`](agent.py) | Standard entrypoint `agent(obs_dict: dict) -> list[int]` loading `checkpoints/best_model.pt` and guaranteeing legal CABT / Kaggle actions. |
| [`cg/`](cg/) | Python bindings (`api.py`, `game.py`, `sim.py`, `utils.py`) interfacing with the CABT C++ engine (native binaries `cg/*.dll`, `cg/*.so` are gitignored). |
| [`src/cli.py`](src/cli.py) | Unified CLI entrypoint: `download-data`, `preprocess`, `smoke-test`, `train`, `evaluate`. |
| [`src/dataset.py`](src/dataset.py) | Kaggle replay downloader, streaming replay parser, and episode-level `train/val/test` splitter. |
| [`src/features.py`](src/features.py) | State/option feature extraction, `int_or` falsy-zero guard, fallback metadata, and `validate_no_leakage`. |
| [`src/model.py`](src/model.py) | `PTCGPolicyNet`, `_get_deck_pokemon_profiles`, `_apply_tactical_guardrails`, and `decode_action_from_logits`. |
| [`src/train.py`](src/train.py) | Combined CE (with virtual stop) + relative BCE loss, GPU training, VRAM/throughput logging, and LR-consistent `--resume`. |
| [`src/evaluate.py`](src/evaluate.py) | Baseline agents (`random`, `heuristic`), `play_single_match`, and `evaluate_in_simulator`. |
| [`tests/test_pipeline.py`](tests/test_pipeline.py) | `pytest` suite (`10/10` unit & integration tests covering `BUG-1`..`BUG-6` and live CABT simulator checks). |
| [`deck.csv`](deck.csv), [`decks/`](decks/) | Valid 60-card test decks (`deck.csv`, `decks/dragapult_deck.csv`, `decks/starmie_deck.csv`). |
| [`notebooks/colab_train_eval.ipynb`](notebooks/colab_train_eval.ipynb) | Thin Colab Pro notebook invoking the repository CLI and syncing checkpoints to Google Drive. |
| [`AGENTS.md`](AGENTS.md) | Pipeline guidelines, agent/model role assignments, and Colab Pro execution rules. |
| [`REPORT_PHASE1_2.md`](REPORT_PHASE1_2.md) | Detailed Phase 1 & Phase 2 data, architecture, and benchmark report. |
| [`REVIEW_REPORT.md`](REVIEW_REPORT.md) | Adversarial bug review (`BUG-1`..`BUG-7`) and regression verification report (`PASS 7/7`). |
| [`HANDOFF.md`](HANDOFF.md) | Project handoff summary covering environment, git state, and Phase 1–2 results. |

> **Git Note**: Downloaded replays (`data/`), native simulator binaries (`cg/*.dll`, `cg/*.so`), model checkpoints (`checkpoints/`), and credentials are excluded from Git via [`.gitignore`](.gitignore).

---

## 3. Phase 1 & Phase 2 Summary (Completed & Verified)

### Phase 1 — Data Inspection & Preprocessing Pipeline
- **Baseline Replay Sample**: `40` episodes from `2026-11-13-v1` (top `30` from `manifest.csv` + `10` episodes sampled at ranks `100, 200, ..., 1000`).
- **Processing Breakdown**:
  - `17,490` frames (`34,980` seat-steps): excluded `80` deck-selection steps (`t=0`), `17,410` waiting turns (`INACTIVE`), `80` terminal steps (`DONE`), and `0` out-of-bounds actions.
  - Out of `17,410` valid decisions, `558` forced single-option steps were filtered from `train`, retaining **`16,852` decisions** (`11,399` train / `3,001` val / `2,452` test across `28 / 6 / 6` disjoint episodes).

### Phase 2 — Imitation Learning, Bug Fixes & Evaluation
- **Training & Resume**: Trained on GPU (`torch 2.6.0+cu124`, `batch_size=128`, `lr=2e-3`, `hidden_dim=128`, Peak VRAM **`31.74 MB`**), selecting `best_model.pt` at **Epoch 21** and verifying `--resume` through **Epoch 22** with monotonic `CosineAnnealingLR`.
- **Verification**: Resolved all `7/7` reviewer bugs (`BUG-1`..`BUG-7` in commit `02035ac`, verified in [`REVIEW_REPORT.md`](REVIEW_REPORT.md)); `pytest` passes **`10/10`**.
- **Official Benchmark Results (Epoch 21 Checkpoint, `seed = 42`)**:
  - **Offline Held-Out Test (`2,452` samples / `2,315` nontrivial)**:
    - `exact_match_all`: **`55.06%`** (`1,350 / 2,452`)
    - `exact_match_nontrivial`: **`52.40%`** (`1,213 / 2,315`)
    - `single_select_acc`: **`55.47%`** | `multi_select_exact_acc`: **`48.20%`** (Jaccard **`60.38%`**)
    - `zero_select_acc`: **`64.29%`** | `turn0_setup_acc`: **`88.33%`** | `main_game_acc`: **`54.22%`**
    - `invalid_actions`: **`0`**
  - **Simulator Mirror Matches (`84` games = `42` per opponent, `decks_equal == True`)**:
    - Vs **`cabt_random_baseline`**: **`35W - 0D - 7L` (`83.33%`)** — `default_deck`: `13-0-1` (`92.86%`), `dragapult_deck`: `12-0-2` (`85.71%`), `starmie_deck`: `10-0-4` (`71.43%`).
    - Vs **`heuristic_baseline`**: **`33W - 0D - 9L` (`78.57%`)** — `default_deck`: `13-0-1` (`92.86%`), `dragapult_deck`: `12-0-2` (`85.71%`), `starmie_deck`: `8-0-6` (`57.14%`).
    - **`0` invalid actions, `0` runtime errors, `0` timeouts** across all `84` matches.

---

## 4. Phase 3 Roadmap (Planned Next Stage)

> **Status**: Upcoming phase (not yet implemented in the current codebase), building directly on the validated Phase 1 & Phase 2 foundation.

Phase 3 focuses on **scaling data volume, improving tactical generalization beyond pure behavioral cloning, and preparing the Kaggle competition submission**:

1. **Full-Dataset Scaling on Colab Pro (`1,644` Episodes)**:
   - Scale `download-data` and `preprocess` from the `40`-episode baseline sample to all (or high-MMR subsets of) `1,644` episodes in `manifest.csv` (`2026-11-13-v1`).
   - Add sharded tensor storage so streaming preprocessing remains memory-safe on CPU at hundreds of thousands of decision steps.
   - Run full training on **Colab Pro** (starting on `NVIDIA T4` to benchmark `nvidia-smi` VRAM and throughput, scaling to `L4` / `A100` if needed) with persistent Google Drive checkpointing and resume verification.

2. **Richer State Representation & Multi-Deck Generalization**:
   - Supply complete static metadata/embeddings for newer card and attack IDs (`cardId` up to `1413`, `attackId` up to `1748`) that fall outside the legacy `cg.dll` lookup tables (`1,267` cards / `1,556` attacks).
   - Encode deeper board and resource context (bench Pokémon damage/attached energies/tools/special conditions, short-term turn action history, revealed discard-pile counts) to boost win rates on underrepresented decks (such as `starmie_deck`, currently `57.14%` vs Heuristic).
   - Upgrade **Multi-Select** (`2 <= maxCount <= 6`) decoding with conditional/autoregressive set scoring to push `multi_select_exact_acc` beyond the `48.20%` baseline.

3. **Reinforcement Learning & Self-Play in CABT**:
   - Add a **Value Head** to `PTCGPolicyNet` and initialize from the Phase 2 imitation-learning checkpoint (BC warm-start).
   - Train via **Self-Play and Opponent-Pool RL** (e.g., PPO / advantage-weighted updates against historical checkpoints and `heuristic_baseline`) directly inside `cg.game`.
   - Optimize long-horizon game objectives (Prize-card racing, evolution line setup, Supporter/Gust timing, resource conservation) that supervised imitation learning alone cannot fully capture.

4. **Cross-Deck Evaluation & Kaggle Submission Packaging**:
   - Expand simulator evaluation from mirror matches to a **cross-deck matchup matrix** across diverse meta archetypes.
   - Benchmark CPU per-step inference latency against Kaggle Environments turn-time limits while enforcing `invalid_actions == 0`, `runtime_errors == 0`, and `timeouts == 0` in `agent.py`.
   - Package the final `agent.py`, checkpoint weights, and reproducibility logs (`commit`, `seed`, `dataset_version`, configs, metrics) for Kaggle submission.

---

## 5. Quick Start (CLI & Colab Pro)

### Local CLI

```bash
# 1. Download manifest.csv and replay JSONs from Kaggle (Version 1, 2026-11-13)
python -m src.cli download-data --out-dir data/replays --num-top 30

# 2. Stream-preprocess replays on CPU and split by episode_id
python -m src.cli preprocess --replay-dir data/replays --out-dir data/processed --seed 42

# 3. Run CPU simulator smoke test
python -m src.cli smoke-test --deck deck.csv --checkpoint checkpoints/best_model.pt --seed 42

# 4. Train on GPU (and verify checkpoint resume)
python -m src.cli train --data-dir data/processed --checkpoint-dir checkpoints --epochs 20 --batch-size 128 --lr 0.002 --seed 42 --device cuda
python -m src.cli train --data-dir data/processed --checkpoint-dir checkpoints --epochs 22 --batch-size 128 --lr 0.002 --seed 42 --device cuda --resume checkpoints/last_model.pt

# 5. Evaluate on held-out test split and 42 simulator mirror matches per opponent
python -m src.cli evaluate --checkpoint checkpoints/best_model.pt --games 42 --seed 42 --out-file checkpoints/eval_results.json

# 6. Run automated test suite
python -m pytest -p no:cacheprovider -v tests/test_pipeline.py
```

### Colab Pro
Open [`notebooks/colab_train_eval.ipynb`](notebooks/colab_train_eval.ipynb), select **Runtime -> Change runtime type -> T4 GPU** (or an available GPU per `nvidia-smi`), and click **Runtime -> Run all** to execute the full pipeline and persist checkpoints to Google Drive.

# Báo Cáo Phase 3 — Scaling Dữ Liệu, Biểu Diễn Giàu Thông Tin, Conditional Multi-Select & RL Self-Play (`cabt` / `cg`)

## 1. Tóm Tắt Cấu Hình & Khả Năng Tái Lập

- **Dataset Kaggle**: [`kaggle/the-pokemon-company-ptcg-ai-battle-challenge-playground-episodes-2026-11-13`](https://www.kaggle.com/datasets/kaggle/the-pokemon-company-ptcg-ai-battle-challenge-playground-episodes-2026-11-13) — **Version 1** (`2026-11-13-v1`), `manifest.csv` (`1,644` episodes).
- **Seed**: `42`
- **Nhánh Git & Commit đánh giá**: `phase3` — Commit mã nguồn được ghi nhận trong `checkpoints/eval_results.json`: `c410ec83ece379f99f8fa5c3d718c779538bf3b2` (`c410ec8`, branched from `main` `b993aa5`).
- **Phần cứng kiểm chứng local**:
  - **CPU**: Windows 11 (`Python 3.12.9`, `pytest 9.1.1`)
  - **GPU**: `NVIDIA GeForce RTX 4050 Laptop GPU` (6 GB VRAM, Driver `591.86`, CUDA `13.1`, PyTorch `2.6.0+cu124`)
- **Colab Pro**: Đã cập nhật [`notebooks/colab_train_eval.ipynb`](notebooks/colab_train_eval.ipynb) hỗ trợ đầy đủ `1,644` episodes với lưu trữ shard giới hạn RAM, `train`, `--resume`, `train-rl`, `evaluate` (mirror + cross-deck) và `pytest`.

---

## 2. Deliverable 1 — Data Scaling & Sharded Tensor Storage (`src/dataset.py`, `src/cli.py`)

### 2.1. Mở rộng `download-data` & Quy mô Dữ liệu Local
- Đã nâng cấp `download_replays_from_kaggle` và `select_manifest_indices` với các tham số `--num-top`, `--num-strided`, `--max-episodes` (hỗ trợ tải toàn bộ `1,644` episodes từ `manifest.csv` hoặc lấy mẫu phân tầng MMR).
- Tập replay mở rộng thực nghiệm trên máy local: **`209` episodes** đa dạng (`100` top MMR + `100` phân bố đều trên toàn bộ bảng xếp hạng + `9` replay seed cache), chia rời nhau hoàn toàn theo `episode_id` trước khi tách bước quyết định (`seed = 42`, tỷ lệ `70% / 15% / 15%`).

### 2.2. Sharded Tensor Storage (`train_shard_*.pt`) & Zero Leakage
- `preprocess_replays` chia `train / val / test` theo `episode_id` trước, suy diễn metadata bổ sung **chỉ từ các file thuộc tập `train`** (`source_split == "train"` trong `inferred_metadata.json`, không ghi đè bảng metadata tĩnh đã hiệu chuẩn), rồi xử lý tuần tự từng replay và xuất theo shard giới hạn dung lượng (`shard_size_samples = 15,000`), kèm manifest `*_shards.json` và `dataset_stats.json`.
- `train_policy` và `evaluate_packed_split` nạp từng shard thông qua `iter_split_shards` (`del shard; gc.collect()` sau mỗi shard), giữ Peak CPU RAM ở mức `O(1 shard)`.
- Kiểm tra `validate_no_leakage` (bài trên tay đối thủ luôn `None`, không rò rỉ bước tương lai) và tính rời nhau 100% của `episode_id` giữa `train / val / test` trên toàn bộ các shard.

| Hạng mục thống kê (`data/processed/dataset_stats.json`) | Phase 2 (`40` eps) | Phase 3 (`209` eps) | Ghi chú |
|---|---:|---:|---|
| Tổng số episode (`num_episodes_total`) | `40` | **`209`** | `146` train / `31` val / `32` test (disjoint 100%) |
| Tổng số frame (`total_step_frames`) | `17,490` | **`93,829`** | Trung bình `448.94` frames / episode |
| Tổng lượt seat-step (`total_seat_steps`) | `34,980` | **`187,658`** | Khớp `418 + 93,411 + 418 + 93,411 = 187,658` |
| Bước chọn deck (`excluded_deck_select_steps`) | `80` | **`418`** | `t = 0`, 60 lá hợp lệ `418/418` |
| Lượt chờ (`excluded_inactive_waiting_turns`) | `17,410` | **`93,411`** | `status == "INACTIVE"` |
| Bước kết thúc (`excluded_terminal_done_steps`) | `80` | **`418`** | `t = len(steps) - 1`, `status == "DONE"` |
| Lỗi ngoài miền (`excluded_invalid_action_bounds`) | `0` | **`0`** | Không có bước lỗi |
| Quyết định hợp lệ (`valid_decisions_total`) | `17,410` | **`93,411`** | Đã unwrap `833` dict actions |
| Loại khỏi train (`excluded_forced_single_option`) | `558` | **`2,802`** | `len(option) == 1` & `min == max == 1` |
| **Quyết định giữ lại (`kept_decisions`)** | **`16,852`** | **`90,609`** | **`61,890` train (`5` shards) / `13,970` val (`1` shard) / `14,749` test (`1` shard)** |
| — Trong đó: Turn-0 setup (`turn == 0`) | `380` | **`1,931`** | `1,250` train / `326` val / `355` test |
| — Trong đó: Multi-Select (`maxCount > 1`) | `952` | **`5,707`** | `3,949` train / `779` val / `979` test |

---

## 3. Deliverable 2 — Fallback Metadata, Richer Features & Conditional Multi-Select (`src/features.py`, `src/model.py`)

- **Bảng Metadata Dự Phòng (`FALLBACK_CARD_DB`, `FALLBACK_ATTACK_DB`) & `infer_fallback_metadata_from_replays`**:
  - Bổ sung metadata tĩnh cho các thẻ ngoài `cg/cg.dll` (`1,267` cards / `1,556` attacks), bao gồm `cardId` `1331, 1387, 1406, 1413` và `attackId` `1641, 1642, 1715, 1716, 1747, 1748`.
  - Khi chạy `preprocess`, `infer_fallback_metadata_from_replays` chỉ quét `split_files["train"]`, không ghi đè các mục tĩnh đã hiệu chuẩn (`overwrite_curated=False`), và lưu các ID mới phát sinh (nếu có) vào `inferred_metadata.json` để quá trình đánh giá đọc nhất quán.
- **Mở rộng Biểu diễn Trạng thái & Lựa chọn (`STATE_DIM: 64 -> 112`, `OPT_DIM: 48 -> 64`)**:
  - `state_vec[64..111]`: Thông số chi tiết từng vị trí Bench của cả hai bên (`hp_norm`, `dmg_ratio`, `n_energy`, `stage_norm`), Stage của Active Pokémon, phân rã bài trong Discard pile công khai (`Pokémon / Energy / Trainer`), và lịch sử hành động ngắn hạn từ `obs["logs"]` (chỉ các sự kiện quá khứ/hiện tại).
  - `opt_feats[48..63]`: Tương tác option–board (`stage`, `tgt_is_active/bench`, energy type matching với chiêu của Pokémon đích, energy deficit, tỷ lệ sát thương/HP mục tiêu, số bản sao cùng lá bài trên tay/discard, tỷ lệ chọn `min/max` trên tổng lựa chọn).
- **Set-Context Self-Attention & Conditional Multi-Select (`2 <= maxCount <= 6`)**:
  - Bổ sung `opt_attn` (Multi-Head Self-Attention 4 heads trên tập các lựa chọn hợp lệ trong cùng một quyết định) để từng phương án nhận biết các phương án đồng hành/trùng lặp/thay thế.
  - Bổ sung `cond_scorer` và `cond_stop_head` (huấn luyện bằng `compute_conditional_multi_loss` với random-subset conditioning và giải mã tuần tự theo tập đã chọn `S_t`), nâng **`multi_select_exact_acc` trên tập Test từ `48.20%` (Phase 2) lên `62.31%` (Phase 3 BC) và `59.86%` (Phase 3 RL)**, với Jaccard tăng từ `60.38%` lên **`72.87%` (`74.45%` ở BC)**.

---

## 4. Deliverable 3 — Value Head & PPO + KL-to-BC Self-Play Fine-Tuning (`src/model.py`, `src/train.py`, `src/cli.py`)

- **`value_head` & Tương thích ngược Phase 2**:
  - `PTCGPolicyNet` tích hợp `value_head` dự đoán `V(s)` từ `state_h`.
  - `load_policy_from_checkpoint` tự động nhận diện kích thước `state_dim/opt_dim` (`64/48` cho Phase 2 vs `112/64` cho Phase 3) và nạp an toàn các checkpoint Phase 2 không có `value_head.*` / `opt_attn.*` / `cond_*`.
- **Huấn luyện BC trên GPU (`RTX 4050`)**:
  - `epochs = 20` + `--resume` đến `epochs = 22` (`batch_size = 128`, `lr = 2e-3`, `CosineAnnealingLR` giữ LR ổn định tại `0.000100`).
  - Tốc độ (`train_summary.json`): Epochs `1–20` đạt `2,655.3 – 5,800.7 samples/sec` (`10.67 – 23.31s / epoch`); Epochs `21–22` (resume) đạt `4,966.8 – 5,083.0 samples/sec` (`12.46s` và `12.18s`); **Peak VRAM = `88.00 MB`**.
  - Checkpoint BC tốt nhất (`best_bc_model.pt` tại Epoch 22): `val_exact_nontrivial = 61.01%`, `val_exact_all = 62.86%`, `val_multi_exact_acc = 60.98%`.
- **Huấn luyện RL / Self-Play (`train-rl`)**:
  - Chạy **10 updates** + kiểm chứng **`--resume` đến 12 updates** bằng **PPO + GAE + KL-to-BC Anchor (`kl_coef = 0.05`, `lr = 3e-4`)** trực tiếp trong `cg.game` trên cả 3 bộ bài (`default_deck`, `dragapult_deck`, `starmie_deck`), đấu với `heuristic_baseline` (`50%`), pool snapshot tự đấu (`35%`), và `cabt_random_baseline` (`15%`).
  - Peak VRAM trong `train-rl`: **`42.70 MB`**.
  - Tỷ lệ thắng trên tập eval nhanh (`18` trận, 3 bộ bài vs `heuristic_baseline`) tăng từ `88.89%` (khởi tạo BC Epoch 22 trong đợt 10 updates) / `94.44%` (khởi tạo eval resume) lên **`100.00%`** (`best_rl_model.pt` tại **RL Update 5** -> promoted to `checkpoints/best_model.pt`, `checkpoint_epoch = 22`, `checkpoint_rl_update = 5`).

---

## 5. Deliverable 4 — So Sánh Phase 2 vs Phase 3, Cross-Deck Matrix & Độ Trễ CPU (`checkpoints/eval_results.json`)

### 5.1. Bảng So Sánh Tổng Hợp Phase 2 vs Phase 3 (`seed = 42`)

| Chỉ số đánh giá | Phase 2 (`Epoch 21`, `40` eps) | Phase 3 BC (`Epoch 22`, `209` eps) | Phase 3 Final RL (`Ep 22 + RL Upd 5`, `209` eps) | Chênh lệch (P3 Final vs P2) |
|---|---:|---:|---:|---:|
| **Offline Test Samples** | `2,452` (`2,315` nontrivial) | `14,749` (`14,080` nontrivial) | **`14,749`** (`14,080` nontrivial) | `6.0x` |
| **Test `exact_match_all`** | `55.06%` (`1,350/2,452`) | `62.87%` (`9,273/14,749`) | **`61.53%`** (`9,075/14,749`) | **`+6.47%`** |
| **Test `exact_match_nontrivial`** | `52.40%` (`1,213/2,315`) | `61.11%` (`8,604/14,080`) | **`59.70%`** (`8,406/14,080`) | **`+7.30%`** |
| **Test `single_select_acc`** | `55.47%` (`1,283/2,313`) | `62.91%` (`8,663/13,770`) | **`61.65%`** (`8,489/13,770`) | **`+6.18%`** |
| **Test `multi_select_exact_acc`** | `48.20%` (`67/139`) | `62.31%` (`610/979`) | **`59.86%`** (`586/979`) | **`+11.66%`** |
| **Test `multi_select_jaccard`** | `60.38%` | `74.45%` | **`72.87%`** | **`+12.49%`** |
| **Test `zero_select_acc`** | `64.29%` (`9/14`) | `58.72%` (`64/109`) | **`68.81%`** (`75/109`) | **`+4.52%`** |
| **Test `turn0_setup_acc`** | `88.33%` (`53/60`) | `90.99%` (`323/355`) | **`90.99%`** (`323/355`) | **`+2.66%`** |
| **Test `invalid_actions`** | `0` | `0` | **`0`** | `0` |
| **Mirror vs Random (`42` trận)** | `83.33%` (`35-0-7`) | `97.62%` (`41-0-1`) | **`92.86%`** (`39-0-3`) | **`+9.53%`** |
| **Mirror vs Heuristic (`42` trận)** | `78.57%` (`33-0-9`) | `85.71%` (`36-0-6`) | **`88.10%`** (`37-0-5`) | **`+9.53%`** |
| — `default_deck` vs Heuristic (`14` trận) | `92.86%` (`13-0-1`) | `100.00%` (`14-0-0`) | **`92.86%`** (`13-0-1`) | `0.00%` |
| — `dragapult_deck` vs Heuristic (`14` trận) | `85.71%` (`12-0-2`) | `85.71%` (`12-0-2`) | **`85.71%`** (`12-0-2`) | `0.00%` |
| — `starmie_deck` vs Heuristic (`14` trận) | `57.14%` (`8-0-6`) | `71.43%` (`10-0-4`) | **`85.71%`** (`12-0-2`) | **`+28.57%`** |
| **Cross-Deck Matrix vs Heuristic (`54` trận)** | *(Chưa có)* | `83.33%` (`45-0-9`) | **`92.59%`** (`50-0-4`) | **Mới (`+9.26%` vs BC)** |
| **Model Invalid / Runtime Errors / Timeouts** | `0 / 0 / 0` | `0 / 0 / 0` | **`0 / 0 / 0`** | An toàn 100% |

> **Đóng góp rõ nét của RL Fine-Tuning**: So với Phase 3 BC thuần (`eval_bc_results.json`, Epoch 22), PPO + KL-to-BC nâng tỷ lệ thắng trên bộ bài khó `starmie_deck` từ `71.43%` (`10-0-4`) lên **`85.71%` (`12-0-2`)** (`+28.57%` so với Phase 2 `57.14%`) và nâng **Cross-Deck Matrix** từ `83.33%` (`45-0-9`) lên **`92.59%` (`50-0-4`)**!

### 5.2. Ma trận Đối kháng Cross-Deck (`54` trận vs `heuristic_baseline`, `6` trận/cặp)

| Model Deck \ Opponent Deck | `default_deck` | `dragapult_deck` | `starmie_deck` |
|---|---:|---:|---:|
| **`default_deck`** | `6-0-0` (**100.00%**) | `6-0-0` (**100.00%**) | `5-0-1` (**83.33%**) |
| **`dragapult_deck`** | `6-0-0` (**100.00%**) | `6-0-0` (**100.00%**) | `5-0-1` (**83.33%**) |
| **`starmie_deck`** | `6-0-0` (**100.00%**) | `5-0-1` (**83.33%**) | `5-0-1` (**83.33%**) |

### 5.3. Độ Trễ Suy Luận CPU & Kiểm Thử Tự Động (`pytest`)

- **Độ trễ CPU trên `5,007` quyết định Mirror Matches**: `mean = 1.9230 ms/decision`, `p50 = 1.9432 ms`, `p95 = 2.6736 ms`, **`p99 = 3.1413 ms`** (max `26.6510 ms`).
- **Độ trễ CPU trên `3,015` quyết định Cross-Deck Matrix**: `mean = 1.8672 ms/decision`, `p50 = 1.8908 ms`, `p95 = 2.5647 ms`, **`p99 = 3.1299 ms`** (max `6.1156 ms`).
- **Smoke test CPU (`python -m src.cli smoke-test`)**: `Checkpoint Agent vs Random` thắng tại step `100` (`15` turns, `160.27 ms` tổng, `2.2373 ms/decision`), `invalid_actions = [0, 0]`, `runtime_errors = [0, 0]`.
- **Kiểm thử tự động (`python -m pytest -p no:cacheprovider -v tests/test_pipeline.py`)**: **`15/15 PASSED` (`19.34s`)**.

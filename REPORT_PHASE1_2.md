# Báo Cáo Triển Khai End-to-End Pokémon TCG Battle Agent (Phase 1 & Phase 2)

## 1. Tổng Quan & Đối Chiếu Dữ Liệu / Simulator

- **Dataset Kaggle**: [`kaggle/the-pokemon-company-ptcg-ai-battle-challenge-playground-episodes-2026-11-13`](https://www.kaggle.com/datasets/kaggle/the-pokemon-company-ptcg-ai-battle-challenge-playground-episodes-2026-11-13) — **Version 1** (`2026-11-13-v1`), gồm `manifest.csv` (1.644 episodes) và các file JSON replay định dạng `kaggle-environments`.
- **Simulator Engine**: **CABT Engine** (`cg` package: `cg/api.py`, `cg/game.py`, `cg/sim.py`, `cg/utils.py`, cùng thư viện biên dịch `cg/cg.dll` cho Windows và `cg/libcg.so` cho Linux/Colab).
- **Xác minh cơ chế ghép `observation` – `action` trên replay thật**:
  - Mỗi episode là một trận đấu BO3 (`configuration.bo = 3`, `version = 2`), gồm danh sách `steps[0 .. T-1]`, mỗi bước có 2 phần tử cho `seat 0` và `seat 1`.
  - Tại bước `t` (`0 <= t < T - 1`), `steps[t][seat]` chứa trạng thái `status` và `observation` **trước quyết định**.
  - Khi `steps[t][seat]["status"] == "ACTIVE"`, agent tại `seat` nhận `obs = steps[t][seat]["observation"]` và trả về hành động được ghi nhận tại **`steps[t + 1][seat]["action"]`** (có thể ở dạng `list[int]` hoặc dict `{"action": list[int], "action_scores": ..., "value": ...}`).
  - **Tách riêng các bước đặc biệt**:
    1. **Bước chọn deck khởi tạo (`t == 0`)**: Cả 2 seat có `status == "ACTIVE"`, `obs["current"] is None`, `obs["select"] is None`, và `steps[1][seat]["action"]` là danh sách 60 ID lá bài (`len == 60`).
    2. **Bước setup đầu ván (`turn == 0`, `select is not None`)**: Gồm chọn người đi trước (`SelectContext.IS_FIRST = 41`), mulligan (`42`), chọn Active Pokémon (`SETUP_ACTIVE_POKEMON = 1`) và chọn Bench Pokémon (`SETUP_BENCH_POKEMON = 2`, `minCount = 0`).
    3. **Lượt chờ (`status == "INACTIVE"`)**: Agent không được gọi; `steps[t + 1][seat]["action"] == []`.
    4. **Bước kết thúc (`t == T - 1`, `status == "DONE"`)**: Trận đấu đã kết thúc, không còn hành động tiếp theo.
- **Đảm bảo không rò rỉ thông tin (Zero Leakage)**:
  - Hàm `validate_no_leakage(obs, expected_seat)` kiểm tra chặt chẽ `obs["current"]["yourIndex"] == seat`, `obs["current"]["players"][1 - seat]["hand"] is None` (bài trên tay đối thủ luôn ẩn), `obs["current"]["players"][seat]["hand"] is not None`, và bỏ qua hoàn toàn chuỗi nội bộ `search_begin_input` cũng như mọi bước tương lai `steps[t+1:]` (ngoại trừ nhãn hành động `steps[t+1][seat]["action"]`).

---

## 2. Thống Kê Tiền Xử Lý & Chia Tập Theo Episode

Đã tải `manifest.csv` và **40 episodes** (30 episodes điểm cao nhất đầu bảng + 10 episodes phân bố đều tại các vị trí `100, 200, ..., 1000` trên bảng xếp hạng để đa dạng hóa đội chơi và chiến thuật), xử lý tuần tự từng file replay để giới hạn RAM và chia `train / val / test` **theo `episode_id` trước khi tách thành các quyết định** (`seed = 42`, tỷ lệ `70% / 15% / 15%` -> `28 train / 6 val / 6 test` episodes, kiểm chứng hoàn toàn rời nhau):

| Hạng mục thống kê | Số lượng | Ghi chú / Lý do giữ lại hoặc loại bỏ |
|---|---:|---|
| Tổng số episode xử lý | **40** | `28` train, `6` val, `6` test (chia theo `episode_id`, `seed=42`) |
| Tổng số frame (`steps`) | **17.490** | Trung bình `437,25` steps / episode |
| Tổng số lượt seat-step (`2 * steps`) | **34.980** | `80 + 17.410 + 80 + 17.410 = 34.980` (khớp 100%) |
| Loại: Bước chọn deck (`excluded_deck_select_steps`) | **80** | `t = 0`, `current=None, select=None`, trả về 60 lá bài hợp lệ (`80/80`) |
| Loại: Lượt chờ (`excluded_inactive_waiting_turns`) | **17.410** | `status == "INACTIVE"`, agent không ra quyết định (`action == []`) |
| Loại: Bước kết thúc (`excluded_terminal_done_steps`) | **80** | `t = len(steps) - 1`, `status == "DONE"` |
| Loại: Bước lỗi / ngoài miền (`excluded_invalid_action_bounds`) | **0** | Không có bước nào vi phạm miền hợp lệ `0 <= idx < len(option)` |
| Quyết định hợp lệ (`valid_decisions_total`) | **17.410** | Trong đó có `169` bước dạng dict `{"action": [...]}` đã unwrap chuẩn |
| Loại khỏi `train`: Quyết định 1 lựa chọn bắt buộc (`excluded_forced_single_option`) | **558** | `len(option) == 1` và `minCount == maxCount == 1` (chỉ loại ở tập `train` để tập trung học; giữ nguyên ở `val`/`test`) |
| **Tổng mẫu quyết định giữ lại (`kept_decisions`)** | **16.852** | **`11.399` train**, **`3.001` val**, **`2.452` test** |
| — Trong đó: Quyết định setup (`turn == 0`) | **380** | `IS_FIRST`, `MULLIGAN`, `SETUP_ACTIVE_POKEMON`, `SETUP_BENCH_POKEMON` |
| — Trong đó: Quyết định trong trận (`turn >= 1`) | **16.472** | Main phase, chọn bài, đính năng lượng, tiến hóa, dùng chiêu, rút lui |
| — Trong đó: Quyết định chọn 1 mục (`maxCount == 1`) | **15.900** | Single-select (`minCount <= 1, maxCount == 1`) |
| — Trong đó: Quyết định chọn nhiều mục (`maxCount > 1`) | **952** | Multi-select (`maxCount in {2, 3, 4}`, gồm cả `minCount == 0` và `minCount == maxCount`) |

---

## 3. Kiến Trúc Policy Imitation Learning & Trạng Thái Colab Pro / GPU

### 3.1. Kiến trúc `PTCGPolicyNet` (`src/model.py`)
- **Biểu diễn trạng thái (`State Encoder`)**: Kết hợp vector đặc trưng dày 64 chiều (`turn`, `turnActionCount`, `firstPlayer`, flags `supporterPlayed`/`stadiumPlayed`/`energyAttached`/`retreated`, one-hot `SelectType`, `minCount`, `maxCount`, `remainDamageCounter`, `remainEnergyCost`, số Prize/Hand/Deck/Discard/Bench hai bên, thông số Active Pokémon hai bên, cơ cấu bài trên tay của bản thân) cùng **Entity Embeddings** (`card_emb` 24 chiều cho 1.267 lá bài, `ctx_emb` 12 chiều cho 49 ngữ cảnh `SelectContext`, `type_emb` 8 chiều cho `SelectType`).
- **Biểu diễn lựa chọn (`Option Encoder` & `Candidate Scorer`)**: Mỗi lựa chọn hợp lệ `i in 0..K-1` trong `obs["select"]["option"]` được trích xuất vector đặc trưng 48 chiều (one-hot `OptionType`, `AreaType`, `inPlayArea`, thông số `CardData` của lá bài nguồn, trạng thái Pokémon đích, sát thương hiệu dụng có tính Weakness `2x` và cờ **Lethal KO** nếu sát thương `>= HP` của Active Pokémon đối thủ) kết hợp embedding của `src_card_id`, `target_card_id`, và `attack_id`.
- **Giới hạn hợp lệ & Multi-Select (`Stop-Threshold Head`)**:
  - Mạng chỉ chấm điểm trên đúng danh sách `K` lựa chọn hợp lệ hiện tại (`0 .. K-1`).
  - Nhánh `stop_head` dự đoán ngưỡng dừng `z_stop` cho các quyết định có số lượng chọn linh hoạt (`minCount < maxCount`): các lựa chọn được xếp hạng giảm dần theo `z_i`, lấy tối thiểu `minCount` mục và tiếp tục chọn tối đa `min(maxCount, K)` mục nếu `z_i > z_stop` (trả về danh sách chỉ số tăng dần duy nhất `list[int]`).
  - Bảo vệ chống lặp nội bộ trong lượt (`turnActionCount >= 20` tại menu `MAIN` sẽ ưu tiên hành động kết thúc lượt `ATTACK` hoặc `END`).

### 3.2. Kiểm tra Colab MCP & Huấn luyện GPU thực tế
- **Trạng thái Colab MCP**: Trong phiên làm việc hiện tại, các công cụ Colab MCP (`get_subscription`, `list_runtimes`, `list_runtime_specs`, `execute_code`) **không được gắn vào danh sách tool của phiên** (bị chặn/chưa kết nối ở tầng MCP).
- **Giải pháp đã thực hiện**:
  1. Đã cài đặt **PyTorch 2.6.0+cu124** và chạy huấn luyện trực tiếp trên **GPU NVIDIA GeForce RTX 4050 Laptop GPU** có sẵn trên máy (đo tốc độ thực tế và VRAM đỉnh, kèm kiểm증 resume từ checkpoint từ epoch 20 lên epoch 22).
  2. Đã tạo sẵn notebook Colab Pro mỏng tại [`notebooks/colab_train_eval.ipynb`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/notebooks/colab_train_eval.ipynb) gọi cùng bộ lệnh CLI (`nvidia-smi`, mount Google Drive lưu checkpoint bền vững, tải dữ liệu, tiền xử lý trên CPU, train/resume trên GPU T4 và đánh giá). Khi mở trên Colab Pro, người dùng chỉ cần chọn **Runtime -> Change runtime type -> T4 GPU** và bấm **Run all**.

### 3.3. Thông số Huấn luyện & Thiết bị
- **Thiết bị GPU**: `NVIDIA GeForce RTX 4050 Laptop GPU` (Driver `591.86`, CUDA `13.1`, PyTorch `2.6.0+cu124`)
- **Cấu hình**: `epochs = 20` (+ `2` epochs resume = `22`), `batch_size = 128`, `lr = 2e-3`, `weight_decay = 1e-4`, `hidden_dim = 128`, `dropout = 0.1`, `seed = 42`
- **Tốc độ huấn luyện**: `~10.200 – 11.166 samples/sec` (`~1,1s / epoch`)
- **VRAM đỉnh (Peak VRAM)**: **`31,22 MB`**
- **Checkpoint đã lưu**:
  - Best model: [`checkpoints/best_model.pt`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/checkpoints/best_model.pt) (Epoch 13, `val_acc(nontrivial) = 52,99%`, `val_acc(all) = 55,48%`)
  - Last model (sau khi resume đến Epoch 22): [`checkpoints/last_model.pt`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/checkpoints/last_model.pt)
  - Log huấn luyện: [`checkpoints/train_summary.json`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/checkpoints/train_summary.json)

---

## 4. Kết Quả Đánh Giá (Offline Test & Đấu Trận Trong Simulator CABT)

File kết quả chi tiết: [`checkpoints/eval_results.json`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/checkpoints/eval_results.json)

### 4.1. Độ khớp hành động trên tập Validation & Test (chia theo Episode, `seed = 42`)

| Chỉ số đánh giá | Validation (`6` episodes, `3.001` mẫu) | Test (`6` episodes, `2.452` mẫu) |
|---|---:|---:|
| **Exact Match (Tất cả quyết định)** | **55,48%** (`1.665 / 3.001`) | **53,59%** (`1.314 / 2.452`) |
| **Exact Match (Quyết định không tầm thường `nontrivial`)** | **52,99%** (`1.506 / 2.842`) | **50,84%** (`1.177 / 2.315`) |
| **Single-Select Exact Match (`maxCount == 1`)** | **55,72%** (`2.796 / 2.859` mẫu) | **53,83%** (`1.245 / 2.313`) |
| **Multi-Select Exact Match (`maxCount > 1`)** | **50,70%** (`72 / 142`) | **49,64%** (`69 / 139`) |
| **Multi-Select Jaccard Similarity** | **62,38%** | **61,89%** |
| **Turn-0 Setup Accuracy (`turn == 0`)** | **83,33%** (`45 / 54`) | **90,00%** (`54 / 60`) |
| **Main-Game Accuracy (`turn >= 1`)** | **54,97%** (`1.620 / 2.947`) | **52,68%** (`1.260 / 2.392`) |
| **Số hành động không hợp lệ (`invalid_actions`)** | **0** | **0** |

### 4.2. Kết quả Smoke Test & Đấu Trận Trực Tiếp Trong Simulator CABT (`seed = 42`)

- **Smoke Test trên CPU** (`python -m src.cli smoke-test --deck deck.csv --checkpoint checkpoints/best_model.pt --seed 42`):
  - `Random vs Random`: hoàn tất sau `34,47 ms` (`213` steps, `53` turns, `winner = 1`, `invalid_actions = [0, 0]`, `runtime_errors = [0, 0]`).
  - `Checkpoint Agent vs Random`: hoàn tất sau `237,64 ms` (`199` steps, `27` turns, `winner = 1`, `invalid_actions = [0, 0]`, `runtime_errors = [0, 0]`, độ trễ trung bình `1,67 ms/quyết định`).
- **Đánh giá 80 trận ghép cặp đổi chỗ (Seat-Swapped Paired Matches, `40` trận / đối thủ, luân phiên Seat 0 và Seat 1 trên 3 bộ bài hợp lệ `deck.csv`, `dragapult_deck.csv`, `starmie_deck.csv`)**:

| Đối thủ Baseline | Số trận | Thắng - Hòa - Thua | Tỷ lệ thắng (`Win Rate`) | Kết quả khi đi Seat 0 (W-D-L) | Kết quả khi đi Seat 1 (W-D-L) | Lỗi hành động (`invalid_actions`) | Lỗi chạy trận (`runtime_errors`) | Độ trễ TB / nước đi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **CABT Random Baseline** | **40** | **30 - 0 - 10** | **75,00%** | `13 - 0 - 7` | `17 - 0 - 3` | **0** | **0** | `1,17 ms` |
| **Rule-Based Heuristic Baseline** | **40** | **26 - 0 - 14** | **65,00%** | `14 - 0 - 6` | `12 - 0 - 8` | **0** | **0** | `1,17 ms` |

---

## 5. Danh Sách File Đã Tạo & Các Lệnh CLI Tái Hiện End-to-End

### 5.1. Danh sách file trong repository
- [`.gitignore`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/.gitignore): Giữ dữ liệu replay tải xuống (`data/`), binary simulator (`cg/*.dll`, `cg/*.so`), và checkpoint (`checkpoints/`) ngoài Git.
- [`agent.py`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/agent.py): Entrypoint chuẩn `agent(obs_dict: dict) -> list[int]` nạp checkpoint và chơi trận trong CABT simulator.
- [`deck.csv`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/deck.csv), [`decks/dragapult_deck.csv`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/decks/dragapult_deck.csv), [`decks/starmie_deck.csv`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/decks/starmie_deck.csv): Các bộ bài 60 lá hợp lệ dùng cho đấu trận.
- [`cg/api.py`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/cg/api.py), [`cg/game.py`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/cg/game.py), [`cg/sim.py`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/cg/sim.py), [`cg/utils.py`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/cg/utils.py): Gói Python giao tiếp với CABT C++ simulator.
- [`src/features.py`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/src/features.py): Trích xuất đặc trưng trạng thái, bài trên tay/bàn và từng lựa chọn hợp lệ kèm kiểm tra chống rò rỉ thông tin.
- [`src/dataset.py`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/src/dataset.py): Tải dữ liệu Kaggle, đọc tuần tự từng replay giới hạn RAM, chia `train/val/test` theo `episode_id`.
- [`src/model.py`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/src/model.py): Mạng `PTCGPolicyNet` chấm điểm lựa chọn hợp lệ và ngưỡng dừng multi-select.
- [`src/train.py`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/src/train.py): Huấn luyện, đo tốc độ/VRAM, lưu và nạp/resume checkpoint.
- [`src/evaluate.py`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/src/evaluate.py): Đánh giá độ khớp trên tập test giữ lại và chạy trận đấu đối kháng trong CABT simulator.
- [`src/cli.py`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/src/cli.py): Giao diện dòng lệnh hợp nhất cho toàn bộ pipeline.
- [`tests/test_pipeline.py`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/tests/test_pipeline.py): Bộ unit tests (`pytest`) kiểm tra ghép `observation-action`, chống rò rỉ tương lai/bài ẩn đối thủ, tính rời nhau của tập chia theo episode, định dạng đầu ra single/multi-select, và nạp checkpoint đấu trận.
- [`notebooks/colab_train_eval.ipynb`](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-5/notebooks/colab_train_eval.ipynb): Notebook Colab Pro mỏng gọi trực tiếp các lệnh CLI của repo.

### 5.2. Các lệnh CLI chạy lại toàn bộ pipeline End-to-End

```bash
# 1. Tải manifest.csv và replay JSON từ Kaggle dataset (Version 1, 2026-11-13)
python -m src.cli download-data --out-dir data/replays --num-top 30

# 2. Tiền xử lý từng replay trên CPU và chia train/val/test theo episode_id (seed=42)
python -m src.cli preprocess --replay-dir data/replays --out-dir data/processed --seed 42

# 3. Huấn luyện model imitation learning trên GPU (lưu best_model.pt và last_model.pt)
python -m src.cli train --data-dir data/processed --checkpoint-dir checkpoints --epochs 20 --batch-size 128 --lr 0.002 --seed 42 --device cuda

# 4. Kiểm tra resume từ checkpoint đã lưu
python -m src.cli train --data-dir data/processed --checkpoint-dir checkpoints --epochs 22 --batch-size 128 --lr 0.002 --seed 42 --device cuda --resume checkpoints/last_model.pt

# 5. Chạy smoke test trên CPU trong simulator CABT
python -m src.cli smoke-test --deck deck.csv --checkpoint checkpoints/best_model.pt --seed 42

# 6. Đánh giá trên tập test giữ lại và đấu 80 trận với các agent baseline trong simulator
python -m src.cli evaluate --checkpoint checkpoints/best_model.pt --games 40 --seed 42 --out-file checkpoints/eval_results.json

# 7. Chạy bộ kiểm thử tự động pytest
python -m pytest -p no:cacheprovider -v tests/test_pipeline.py
```

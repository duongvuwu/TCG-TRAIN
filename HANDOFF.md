# HANDOFF — Pokémon TCG Battle Agent (CABT Imitation Learning Baseline)

**Cập nhật lần cuối**: `2026-11-16T22:25+07:00`  
**Worktree / CWD**: `Z:\Word-by-Word Video Supercut\.ao-local\data\worktrees\tcg-train\tcg-train-5`  
**Git Branch**: `ao/quy-t-c/root` (local-only; **KHÔNG** `git push` hoặc mở PR theo yêu cầu)

---

## 1. Mục Tiêu & Quy Tắc Dự Án (`./AGENTS.md`)

1. **Bài toán**: Huấn luyện và đánh giá agent chọn hành động hợp lệ cho simulator **Pokémon TCG (`cabt` / `cg`)** của Kaggle Playground, sử dụng dataset replay ngày `2026-11-13` (Version 1):
   - Dataset: `kaggle/the-pokemon-company-ptcg-ai-battle-challenge-playground-episodes-2026-11-13` (Version 1, `2026-11-13-v1`)
   - Tài liệu simulator CABT: `https://matsuoinstitute.github.io/cabt/` (đã lưu bản text tại `cabt_api_docs.txt`, gitignored)
2. **Phân công vai trò (theo `./AGENTS.md`)**:
   - **Worker coding**: `Gemini 4 Argon (Medium)` (`--model gemini-4-argon-medium`)
   - **Reviewer bug**: `Gemini 4 Hydrogen (High)` (`--model gemini-4-hydrogen-high`)
3. **Quy tắc cứng**:
   - Mọi thao tác đọc/ghi/chạy lệnh chỉ thực hiện trong CWD hiện tại (tránh bật xác nhận TUI ngoài CWD).
   - Giữ token, dữ liệu tải xuống (`data/`), thư viện nhị phân simulator (`cg/*.dll`, `cg/*.so`), và checkpoint (`checkpoints/`) ngoài Git (đã cấu hình trong `.gitignore`).
   - Không rò rỉ thông tin tương lai hoặc trạng thái ẩn của đối thủ (`players[1 - yourIndex]["hand"] is None`).
   - Lưu ý Windows Sandbox: Một số lệnh ghi file nặng (`torch.save`, `kagglehub`, `git commit`) có thể gặp `PermissionError` / `WinError 5` nếu chạy trong sandbox hạn chế; cần chạy với `BypassSandbox: true` khi cần ghi vào `data/`, `checkpoints/`, hoặc `.git` worktree lockfile.

---

## 2. Trạng Thái Git & Tiến Độ Hiện Tại (Đã Hoàn Tất Toàn Bộ Bugfix & Đánh Giá)

### 2.1. Lịch sử Commit
- `a28d144` — `initial commit`
- `f96eb62` — `Implement end-to-end Pokemon TCG CABT imitation learning baseline and evaluation`
- Bản cập nhật hiện tại trên nhánh `ao/quy-t-c/root` — Đã xử lý triệt để toàn bộ **7 bugs (`BUG-1` .. `BUG-7`)** do Reviewer (`gemini-4-hydrogen-high`) phát hiện, đồng bộ `notebooks/colab_train_eval.ipynb`, chạy lại `smoke-test`, `evaluate` (`42` trận/đối thủ, `seed=42`), và `pytest` (`10/10 PASSED`).

### 2.2. Bảng Trạng Thái Các Hạng Mục

| Thành phần | Trạng thái hiện tại | Ghi chú |
|---|---|---|
| Code sửa `BUG-1`..`BUG-6` (`agent.py`, `src/features.py`, `src/dataset.py`, `src/model.py`, `src/train.py`, `src/evaluate.py`) | **ĐÃ HOÀN TẤT & KIỂM CHỨNG** | Bao gồm cả nâng cấp `_apply_tactical_guardrails` với `_get_deck_pokemon_profiles` từ `cg.api` |
| Bộ test hồi quy (`tests/test_pipeline.py`) | **10/10 PASSED (`5.25s`)** | Đã kiểm chứng `python -m pytest -p no:cacheprovider -v tests/test_pipeline.py` |
| Tiền xử lý dữ liệu (`data/processed/train.pt`, `val.pt`, `test.pt`, `dataset_stats.json`) | **ĐÃ HOÀN TẤT** | `MAX_CARD_ID=2048`, `MAX_ATTACK_ID=2048`, `AreaType.STADIUM (7)` và `CardType.STADIUM (4)` |
| Huấn luyện & Resume (`checkpoints/best_model.pt`, `checkpoints/last_model.pt`, `checkpoints/train_summary.json`) | **ĐÃ HOÀN TẤT** | `best_model.pt` tại **Epoch 21** (`val_exact_nontrivial = 54.54%`, `val_exact_all = 56.95%`, `multi_acc = 57.75%`, `zero_select_acc = 57.14%`, `turn0_setup_acc = 91.30%`); `last_model.pt` tại **Epoch 22** (`lr = 0.000100`) |
| Đánh giá simulator & offline test (`checkpoints/eval_results.json`) | **ĐÃ HOÀN TẤT (`checkpoint_epoch = 21`)** | Offline Test `exact_match_all = 55.06%` (`1,350/2,452`), `turn0_setup_acc = 88.33%` (`53/60`); Simulator `42` trận/đối thủ: thắng **83.33% (`35-0-7`)** trước Random Baseline và **78.57% (`33-0-9`)** trước Heuristic Baseline (`0` lỗi hợp lệ, `0` lỗi runtime, `0` timeout) |
| Báo cáo `REPORT_PHASE1_2.md` & `HANDOFF.md` (`BUG-7`) | **ĐÃ ĐỒNG BỘ 100%** | Khớp tuyệt đối với `dataset_stats.json`, `train_summary.json` và `eval_results.json` |
| Notebook Colab Pro (`notebooks/colab_train_eval.ipynb`) | **ĐÃ ĐỒNG BỘ 100%** | Khớp hoàn toàn với `src/cli.py` (`--epochs 20`, resume `--epochs 22`, `--games 42`, `pytest`) |

---

## 3. Tóm Tắt 7 Bug Từ Reviewer & Cách Đã Xử Lý

1. **`BUG-1 [CRITICAL]` Asymmetric Deck Mismatch (`src/evaluate.py`)**:
   - *Lỗi cũ*: `play_single_match` gọi `agents[s](init_obs)` và ghi đè `decks[s] = dk` nên khi `policy_agent` giữ `default_deck=deck.csv`, model cầm `deck.csv` đấu với `dragapult_deck` / `starmie_deck` thay vì mirror match thật.
   - *Đã sửa*: `play_single_match` giữ nguyên `deck0`, `deck1` truyền vào tường minh (chỉ kiểm tra tính hợp lệ của `agents[s](init_obs)`), đồng thời `evaluate_in_simulator` truyền đúng `_dk=dk` vào `policy_agent` và assert `res["decks_equal"]`, tách thống kê `by_deck`.
2. **`BUG-2 [HIGH]` Falsy-Zero (`0 or -1`, `0 or 1`) (`agent.py`, `src/features.py`, `src/dataset.py`, `src/model.py`, `src/evaluate.py`)**:
   - *Lỗi cũ*: `SelectType.MAIN == 0` và `maxCount == 0` bị biểu thức `val or -1` / `val or 1` biến sai thành `-1` / `1`, làm vô hiệu hóa anti-loop guard và trả về `[idx]` thay vì `[]` khi `maxCount == 0`.
   - *Đã sửa*: Bổ sung helper `int_or(d, key, default)` (`default if d.get(key) is None else int(d.get(key))`) và thay thế ở tất cả các file.
3. **`BUG-3 [HIGH]` Trần `MAX_CARD_ID=1300` & `MAX_ATTACK_ID=1600` (`src/features.py`)**:
   - *Lỗi cũ*: 4 card ID (`1331..1413`) và 6 attack ID (`1641..1748`) trong replay mới bị kẹp vào một vector embedding duy nhất.
   - *Đã sửa*: Nâng `MAX_CARD_ID = 2048`, `MAX_ATTACK_ID = 2048` + fallback metadata cho thẻ/chiêu mới ngoài `cg/cg.dll` (`1267` cards / `1556` attacks).
4. **`BUG-4 [MEDIUM-HIGH]` Bất nhất Loss với `minCount == 0` (`[]`) & `stop_head` (`src/train.py`)**:
   - *Lỗi cũ*: `ce_loss` bỏ qua mẫu `gold == []`, còn `bce_loss` kẹp `pos_weight = 5.0` làm lệch ngưỡng `stop_logit`.
   - *Đã sửa*: Thêm cột virtual stop (`max_opts`) vào `padded_logits` khi `min_count == 0`, đặt `pos_weight = 1.0` và tăng trọng số mẫu `min_count < max_count`.
5. **`BUG-5 [MEDIUM]` `CosineAnnealingLR` bật ngược LR khi `--resume` (`src/train.py`)**:
   - *Lỗi cũ*: `scheduler.load_state_dict(...)` ghi đè `T_max` về `20`, khiến epoch `21..22` tăng ngược LR.
   - *Đã sửa*: Cập nhật `scheduler.T_max = max(epochs, start_epoch)` và chặn `scheduler.step()` khi `last_epoch >= T_max`. Đã xác minh thực tế LR epoch 20->21->22 giữ ổn định `0.000100`.
6. **`BUG-6 [MEDIUM]` Thiếu `AreaType.STADIUM (7)`, `CardType.STADIUM (4)` & tách `timeouts` (`src/features.py`, `src/evaluate.py`)**:
   - *Đã sửa*: Map `area == 7` vào slot `23`, tính `CardType.STADIUM (4)` trong `hand` vào `state_vec[59]`, và tách `timeouts` (`winner == -1`) khỏi `draws` (`winner == 2`) trong `evaluate_in_simulator`.
7. **`BUG-7 [LOW/DOC]` Số liệu `REPORT_PHASE1_2.md` & `eval_results.json`**:
   - *Đã sửa*: Đã chạy lại `evaluate` (`checkpoint_epoch = 21`) và `pytest` (`10/10 PASSED`), cập nhật toàn bộ bảng số liệu offline/simulator (`overall` & `by_deck`) và mã `git_commit` trong `REPORT_PHASE1_2.md`, `HANDOFF.md`, và `checkpoints/eval_results.json` khớp 100% với `data/processed/dataset_stats.json` và `checkpoints/train_summary.json`.

---

## 4. Môi Trường, Dữ Liệu & Kiến Trúc Repo

### 4.1. Môi trường thực thi đã cài đặt
- **OS**: Windows (`powershell`)
- **Python**: `3.12.9` (`C:\Program Files\Python312\python.exe`)
- **GPU Local**: `NVIDIA GeForce RTX 4050 Laptop GPU` (6 GB VRAM, Driver `591.86`, CUDA `13.1`)
- **Các gói Python chính**: `torch 2.6.0+cu124` (CUDA active), `numpy 2.4.2`, `pandas 3.0.1`, `pytest 9.1.1`, `kagglehub 1.0.3`, `kaggle 2.2.5`, `kaggle-environments 1.33.3`.
- **CABT Simulator (`cg/`)**: Đã tải từ `rahulsiiitm/ptcg-rl-agent` (chứa `cg/api.py`, `cg/game.py`, `cg/sim.py`, `cg/utils.py`, `cg/cg.dll` cho Windows, `cg/libcg.so` cho Linux). `cg.api.all_card_data()` có `1.267` thẻ; `all_attack()` có `1.556` chiêu thức.
- **Colab MCP**: Không có sẵn / chưa kết nối trong session MCP; đã chuẩn bị notebook mỏng `notebooks/colab_train_eval.ipynb` gọi trực tiếp CLI của repo.

### 4.2. Dữ liệu Replay & Chia Tập (`data/replays/` & `data/processed/`)
- `data/replays/manifest.csv`: `1.644` episodes (`2026-11-13-v1`).
- `data/replays/*.json`: **40 replays** (top 30 từ `manifest.csv` + 10 replay rải đều ở index `100, 200, ..., 1000`).
- Chia tập nghiêm ngặt theo `episode_id` (`seed = 42`, `28 train / 6 val / 6 test`):
  - Tổng `34.980` seat-steps (`17.490` frames): `80` bước chọn deck (`t=0`), `17.410` lượt chờ (`INACTIVE`), `80` bước kết thúc (`DONE`), `17.410` quyết định hợp lệ (`169` dict actions đã unwrap).
  - Loại `558` bước 1 lựa chọn bắt buộc khỏi `train`; **giữ lại `16.852` mẫu** (`11.399` train, `3.001` val, `2.452` test; `380` turn-0 setup, `16.472` main-game; `15.900` single-select, `952` multi-select với `2 <= maxCount <= 6`).

### 4.3. Bản đồ File trong Repo
- `agent.py` — Entrypoint chuẩn `agent(obs_dict: dict) -> list[int]` nạp checkpoint `checkpoints/best_model.pt`.
- `deck.csv` — Bộ bài 60 lá chuẩn (copy từ `top_train_deck` trong replay: Dragapult/Drakloak/Budew/Munkidori).
- `decks/dragapult_deck.csv`, `decks/starmie_deck.csv` — 2 bộ bài kiểm thử bổ sung cho đánh giá mirror match.
- `src/features.py` — Trích xuất vector trạng thái (64 chiều), hand/board card IDs, option features (48 chiều) + `validate_no_leakage` + `int_or`.
- `src/dataset.py` — Tải Kaggle (`download_replays_from_kaggle`), stream từng replay giới hạn RAM (`parse_single_replay`), chia split theo `episode_id` (`preprocess_replays`).
- `src/model.py` — `PTCGPolicyNet` (Entity Embeddings + State/Option Encoders + Joint Scorer + `stop_head` + `_get_deck_pokemon_profiles` + `_apply_tactical_guardrails` + `decode_action_from_logits`).
- `src/train.py` — `compute_policy_loss`, `evaluate_packed_split`, `train_policy` (hỗ trợ `--resume`), `load_policy_from_checkpoint`.
- `src/evaluate.py` — `make_random_baseline_agent`, `make_heuristic_baseline_agent`, `play_single_match`, `evaluate_in_simulator`.
- `src/cli.py` — CLI thống nhất (`download-data`, `preprocess`, `smoke-test`, `train`, `evaluate`).
- `tests/test_pipeline.py` — Bộ kiểm thử `pytest` (`10/10` unit & integration tests cho BUG-1..BUG-6 và chạy trận trong simulator).
- `notebooks/colab_train_eval.ipynb` — Notebook Colab Pro mỏng.
- `REPORT_PHASE1_2.md` — Báo cáo tổng hợp Phase 1 & 2 (đã đồng bộ 100% số liệu mới).

---

## 5. Tóm Tắt Kết Quả Đánh Giá Cuối Cùng (`checkpoints/eval_results.json`, Epoch 21)

- **Offline Validation (`3.001` mẫu, `2.842` nontrivial)**:
  - `loss = 1.6249`, `exact_match_all = 56.95% (1,709/3,001)`, `exact_match_nontrivial = 54.54% (1,550/2,842)`
  - `single_select_acc = 56.91% (1,627/2,859)`, `multi_select_exact_acc = 57.75% (82/142)`, `multi_select_jaccard = 67.37%`, `zero_select_acc = 57.14% (20/35)`
  - `turn0_setup_acc = 91.30% (63/69)`, `main_game_acc = 56.14% (1,646/2,932)`, `invalid_actions = 0`
- **Offline Test (`2.452` mẫu, `2.315` nontrivial)**:
  - `loss = 1.6691`, `exact_match_all = 55.06% (1,350/2,452)`, `exact_match_nontrivial = 52.40% (1,213/2,315)`
  - `single_select_acc = 55.47% (1,283/2,313)`, `multi_select_exact_acc = 48.20% (67/139)`, `multi_select_jaccard = 60.38%`, `zero_select_acc = 64.29% (9/14)`
  - `turn0_setup_acc = 88.33% (53/60)`, `main_game_acc = 54.22% (1,297/2,392)`, `invalid_actions = 0`
- **Simulator Mirror Matches (`42` trận / đối thủ, `14` trận / bộ bài, `decks_equal == True`, `model_invalid_actions = 0`, `model_runtime_errors = 0`, `timeouts = 0`)**:
  - **vs `cabt_random_baseline`**: **35W – 0D – 7L (`83.33%`)** (`Seat 0`: `16-0-5`, `Seat 1`: `19-0-2`)
    - `default_deck`: **13W – 0D – 1L (`92.86%`)**
    - `dragapult_deck`: **12W – 0D – 2L (`85.71%`)**
    - `starmie_deck`: **10W – 0D – 4L (`71.43%`)**
  - **vs `heuristic_baseline`**: **33W – 0D – 9L (`78.57%`)** (`Seat 0`: `15-0-6`, `Seat 1`: `18-0-3`)
    - `default_deck`: **13W – 0D – 1L (`92.86%`)**
    - `dragapult_deck`: **12W – 0D – 2L (`85.71%`)**
    - `starmie_deck`: **8W – 0D – 6L (`57.14%`)**

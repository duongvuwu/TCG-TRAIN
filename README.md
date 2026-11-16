# Pokémon TCG Battle Agent (CABT Simulator — Kaggle Playground)

Agent chọn hành động hợp lệ cho simulator **Pokémon TCG (`cabt` / `cg`)** của cuộc thi Kaggle Playground (*The Pokémon Company PTCG AI Battle Challenge Playground*), được huấn luyện từ dữ liệu replay chính thức (`2026-11-13`, Version 1) theo quy trình End-to-End có kiểm chứng chống rò rỉ thông tin và đánh giá đối kháng trực tiếp trong simulator.

---

## 1. Tổng Quan Kiến Trúc & Pipeline

Hệ thống được thiết kế thành một pipeline CLI khép kín (`src/cli.py`), chạy tiền xử lý tiết kiệm RAM trên CPU, huấn luyện trên GPU (local hoặc Colab Pro) và đánh giá tự động trên tập giữ lại lẫn CABT simulator:

1. **Tải dữ liệu (`download-data`)**: Tải `manifest.csv` (`1.644` episodes, `2026-11-13-v1`) và các file replay JSON từ Kaggle.
2. **Tiền xử lý Streaming & Chia tập (`preprocess`)**:
   - Đọc tuần tự từng replay để giới hạn RAM.
   - Ghép cặp chuẩn xác trạng thái trước quyết định `steps[t][seat]["observation"]` với hành động `steps[t + 1][seat]["action"]` (hỗ trợ unwrap cả định dạng `list[int]` và `dict`).
   - Tách riêng các bước khởi tạo chọn deck (`t = 0`), setup đầu ván (`turn = 0`), lượt chờ (`INACTIVE`) và bước kết thúc (`DONE`).
   - Kiểm tra **Zero Leakage** (`validate_no_leakage`): xác nhận không đọc bước tương lai hay thông tin ẩn của đối thủ (`players[1 - seat]["hand"] is None`).
   - Chia `train / val / test` **theo `episode_id`** trước khi tách thành từng bước quyết định.
3. **Mô hình Policy Imitation Learning (`PTCGPolicyNet` — `src/model.py`)**:
   - **State Encoder**: Vector trạng thái `64` chiều (lượt, cờ hành động trong lượt, `SelectType`, `minCount`, `maxCount`, Prize/Hand/Deck/Discard/Bench, Active Pokémon hai bên, cơ cấu bài trên tay gồm cả `CardType.STADIUM = 4`) + Entity Embeddings (`MAX_CARD_ID = 2048`, `MAX_ATTACK_ID = 2048`).
   - **Option Encoder & Candidate Scorer**: Vector đặc trưng `48` chiều cho từng lựa chọn hợp lệ `0 .. K-1` (gồm `OptionType`, `AreaType` có cả `AreaType.STADIUM = 7`, thuộc tính thẻ, sát thương hiệu dụng có tính Weakness `2x` và cờ **Lethal KO**).
   - **Ràng buộc hợp lệ & Multi-Select**: Chỉ chấm điểm trong danh sách lựa chọn hợp lệ `0 .. K-1`, học ngưỡng dừng `stop_head` (`z_stop`) và cột virtual stop khi `minCount == 0` (trả về `[]`), xử lý an toàn `maxCount == 0`, kèm `_apply_tactical_guardrails` (dựa trên hồ sơ năng lượng/chiêu thức từ `cg.api`) và anti-loop guard (`turnActionCount >= 20` tại `SelectType.MAIN == 0`).
4. **Đánh giá Offline & Simulator (`evaluate`, `smoke-test`)**:
   - Đo Exact Match (overall, nontrivial, single-select, multi-select, zero-select, turn-0 setup, main-game) và Jaccard similarity trên tập `val`/`test`.
   - Đấu trận **Mirror Match công bằng** (`decks_equal == True`) trên nhiều bộ bài (`default_deck`, `dragapult_deck`, `starmie_deck`), luân phiên `Seat 0` và `Seat 1`, đối đầu `cabt_random_baseline` và `heuristic_baseline`, tách bạch thống kê `wins / draws / losses / timeouts / invalid_actions / runtime_errors`.

---

## 2. Cấu Trúc Repository

| Đường dẫn | Vai trò |
|---|---|
| [`agent.py`](agent.py) | Entrypoint chuẩn `agent(obs_dict: dict) -> list[int]` nạp `checkpoints/best_model.pt` và bảo đảm đầu ra hợp lệ cho CABT / Kaggle. |
| [`cg/`](cg/) | Gói Python (`api.py`, `game.py`, `sim.py`, `utils.py`) giao tiếp với CABT C++ engine (thư viện nhị phân `cg/*.dll`, `cg/*.so` được gitignore). |
| [`src/cli.py`](src/cli.py) | Giao diện CLI thống nhất: `download-data`, `preprocess`, `smoke-test`, `train`, `evaluate`. |
| [`src/dataset.py`](src/dataset.py) | Tải dữ liệu Kaggle, stream parse từng replay, chia `train/val/test` theo `episode_id`. |
| [`src/features.py`](src/features.py) | Trích xuất state/option features, `int_or` chống lỗi falsy-zero, fallback metadata và `validate_no_leakage`. |
| [`src/model.py`](src/model.py) | Mạng `PTCGPolicyNet`, hồ sơ bài `_get_deck_pokemon_profiles`, `_apply_tactical_guardrails` và `decode_action_from_logits`. |
| [`src/train.py`](src/train.py) | Hàm loss kết hợp CE (có virtual stop) + relative BCE, huấn luyện GPU, đo VRAM/throughput và `--resume` giữ nguyên đường cong LR. |
| [`src/evaluate.py`](src/evaluate.py) | Baseline agents (`random`, `heuristic`), chạy trận `play_single_match` và tổng hợp `evaluate_in_simulator`. |
| [`tests/test_pipeline.py`](tests/test_pipeline.py) | Bộ kiểm thử `pytest` (`10/10` unit & integration tests bao phủ `BUG-1`..`BUG-6` và CABT simulator). |
| [`deck.csv`](deck.csv), [`decks/`](decks/) | Bộ bài 60 lá chuẩn (`deck.csv`, `decks/dragapult_deck.csv`, `decks/starmie_deck.csv`). |
| [`notebooks/colab_train_eval.ipynb`](notebooks/colab_train_eval.ipynb) | Notebook Colab Pro mỏng gọi trực tiếp CLI của repo và đồng bộ checkpoint về Google Drive. |
| [`AGENTS.md`](AGENTS.md) | Quy chuẩn pipeline, phân công model/agent và quy tắc chạy trên Colab Pro. |
| [`REPORT_PHASE1_2.md`](REPORT_PHASE1_2.md) | Báo cáo chi tiết dữ liệu, mô hình và kết quả đánh giá Phase 1 & Phase 2. |
| [`REVIEW_REPORT.md`](REVIEW_REPORT.md) | Báo cáo rà soát lỗi (`BUG-1`..`BUG-7`) và xác minh hồi quy (`PASS 7/7`). |
| [`HANDOFF.md`](HANDOFF.md) | Tài liệu bàn giao trạng thái dự án, môi trường và kết quả mốc Phase 1 & 2. |

> **Lưu ý Git**: Dữ liệu tải xuống (`data/`), binary simulator (`cg/*.dll`, `cg/*.so`), checkpoint (`checkpoints/`) và thông tin xác thực được giữ ngoài Git theo [`.gitignore`](.gitignore).

---

## 3. Tổng Kết Phase 1 & Phase 2 (Đã Hoàn Thành & Kiểm Chứng)

### Phase 1 — Khảo Sát Dữ Liệu & Pipeline Tiền Xử Lý
- **Mẫu dữ liệu baseline**: `40` episodes từ `2026-11-13-v1` (top `30` theo `manifest.csv` + `10` episodes tại các vị trí `100, 200, ..., 1000`).
- **Thống kê xử lý**:
  - `17.490` frames (`34.980` seat-steps): loại `80` bước chọn deck (`t=0`), `17.410` lượt chờ (`INACTIVE`), `80` bước kết thúc (`DONE`), `0` bước vượt biên.
  - Từ `17.410` quyết định hợp lệ, loại `558` bước 1 lựa chọn bắt buộc khỏi `train`, giữ lại **`16.852` mẫu** (`11.399` train / `3.001` val / `2.452` test; `28 / 6 / 6` episodes tách rời).

### Phase 2 — Huấn Luyện Imitation Learning, Vá Lỗi & Đánh Giá
- **Huấn luyện & Resume**: Chạy trên GPU (`torch 2.6.0+cu124`, `batch_size=128`, `lr=2e-3`, `hidden_dim=128`, Peak VRAM **`31,74 MB`**), đạt `best_model.pt` tại **Epoch 21** và kiểm chứng `--resume` lên **Epoch 22** với `CosineAnnealingLR` ổn định.
- **Kiểm chứng chất lượng**: Đã khắc phục dứt điểm `7/7` bugs từ Reviewer (`BUG-1`..`BUG-7` trong commit `02035ac`, xác nhận tại [`REVIEW_REPORT.md`](REVIEW_REPORT.md)), bộ test `pytest` đạt **`10/10 PASSED`**.
- **Kết quả đánh giá chính thức (Checkpoint Epoch 21, `seed = 42`)**:
  - **Offline Test (`2.452` mẫu / `2.315` nontrivial)**:
    - `exact_match_all`: **`55,06%`** (`1.350 / 2.452`)
    - `exact_match_nontrivial`: **`52,40%`** (`1.213 / 2.315`)
    - `single_select_acc`: **`55,47%`** | `multi_select_exact_acc`: **`48,20%`** (Jaccard **`60,38%`**)
    - `zero_select_acc`: **`64,29%`** | `turn0_setup_acc`: **`88,33%`** | `main_game_acc`: **`54,22%`**
    - `invalid_actions`: **`0`**
  - **Simulator Mirror Matches (`84` trận = `42` trận/đối thủ, `decks_equal == True`)**:
    - Vs **`cabt_random_baseline`**: **`35W - 0D - 7L` (`83,33%`)** — `default_deck`: `13-0-1` (`92,86%`), `dragapult_deck`: `12-0-2` (`85,71%`), `starmie_deck`: `10-0-4` (`71,43%`).
    - Vs **`heuristic_baseline`**: **`33W - 0D - 9L` (`78,57%`)** — `default_deck`: `13-0-1` (`92,86%`), `dragapult_deck`: `12-0-2` (`85,71%`), `starmie_deck`: `8-0-6` (`57,14%`).
    - **`0` lỗi hợp lệ, `0` lỗi runtime, `0` timeouts** trên toàn bộ `84` trận.

---

## 4. Nội Dung & Lộ Trình Phase 3 (Giai Đoạn Tiếp Theo)

> **Trạng thái**: Kế hoạch triển khai tiếp theo (chưa code trong codebase hiện tại), kế thừa toàn bộ nền tảng Phase 1 & Phase 2.

Phase 3 tập trung vào việc **mở rộng quy mô dữ liệu, nâng cấp năng lực chiến thuật vượt ngưỡng Imitation Learning và chuẩn bị nộp bài Kaggle**:

1. **Mở rộng quy mô dữ liệu & Huấn luyện trên Colab Pro (`Full 1.644 Episodes`)**:
   - Mở rộng pipeline `download-data` và `preprocess` từ `40` episodes mẫu lên toàn bộ hoặc phần lớn `1.644` episodes trong `manifest.csv` (`2026-11-13-v1`), ưu tiên lọc các replay chất lượng cao/đa dạng archetype theo thứ hạng bảng xếp hạng.
   - Tối ưu định dạng tensor/shard khi số mẫu tăng lên hàng trăm nghìn quyết định để vẫn giữ RAM an toàn trên CPU.
   - Chạy huấn luyện chính thức trên **Colab Pro** (ưu tiên `NVIDIA T4` cho baseline mở rộng, đo `nvidia-smi`, VRAM đỉnh và tốc độ thực tế; nâng cấp lên `L4` / `A100` nếu thiếu VRAM hoặc thời gian train quá dài), lưu checkpoint định kỳ vào Google Drive để resume tin cậy.

2. **Nâng cấp biểu diễn trạng thái & Tổng quát hóa đa bộ bài (Multi-Deck Generalization)**:
   - Bổ sung bảng metadata/embedding đầy đủ cho các lá bài và chiêu thức mới xuất hiện trong replay (`cardId` tới `1413`, `attackId` tới `1748`) vốn chưa có trong `cg.dll` cũ (`1.267` thẻ / `1.556` chiêu).
   - Mã hóa sâu hơn cấu trúc **bàn đấu & tài nguyên** (Pokémon trên Bench kèm sát thương/năng lượng đính/tool/trạng thái đặc biệt, lịch sử hành động ngắn hạn trong lượt, lá bài đã lộ trong Discard Pile của hai bên) để cải thiện quyết định ở các bộ bài ít xuất hiện trong tập train (như `starmie_deck`, hiện đạt `57,14%` trước Heuristic).
   - Cải thiện kiến trúc xử lý **Multi-Select** (`2 <= maxCount <= 6`) theo hướng autoregressive hoặc set-scoring có điều kiện để tăng `multi_select_exact_acc` vượt mức `48,20%` hiện tại.

3. **Reinforcement Learning / Self-Play trên Simulator CABT**:
   - Bổ sung **Value Head** cho `PTCGPolicyNet` và khởi tạo trọng số từ checkpoint Imitation Learning tốt nhất ở Phase 2 (BC warm-start) để tránh khám phá ngẫu nhiên kém hiệu quả.
   - Tổ chức vòng lặp **Self-Play & Pool Evaluation** (PPO / Advantage-Weighted Policy Gradient / Best-Response against League) trực tiếp trong `cg.game` với đối thủ là các checkpoint trước đó và `heuristic_baseline`.
   - Tối ưu mục tiêu dài hạn của ván đấu (quản lý Prize race, giữ/lập chuỗi tiến hóa, chọn thời điểm dùng Supporter/Gust, tránh kẹt bàn hoặc hết bài) mà Imitation Learning thuần túy khó học trọn vẹn.

4. **Đánh giá toàn diện & Đóng gói Submission Kaggle**:
   - Mở rộng bộ đánh giá từ mirror-match sang **cross-deck matchup matrix** giữa nhiều archetype meta khác nhau.
   - Kiểm tra nghiêm ngặt ngân sách thời gian suy luận trên CPU (`latency / step`), `invalid_actions == 0`, `runtime_errors == 0`, `timeouts == 0` cho `agent.py`.
   - Đóng gói `agent.py`, trọng số cuối cùng và tài liệu tái lập kết quả (`commit`, `seed`, `dataset_version`, cấu hình, metrics) phục vụ nộp bài Kaggle Playground.

---

## 5. Hướng Dẫn Chạy Nhanh (CLI & Colab Pro)

### Chạy cục bộ qua CLI

```bash
# 1. Tải manifest.csv và replay JSON từ Kaggle (Version 1, 2026-11-13)
python -m src.cli download-data --out-dir data/replays --num-top 30

# 2. Tiền xử lý từng replay trên CPU và chia train/val/test theo episode_id
python -m src.cli preprocess --replay-dir data/replays --out-dir data/processed --seed 42

# 3. Smoke test simulator trên CPU
python -m src.cli smoke-test --deck deck.csv --checkpoint checkpoints/best_model.pt --seed 42

# 4. Huấn luyện trên GPU (và resume từ checkpoint)
python -m src.cli train --data-dir data/processed --checkpoint-dir checkpoints --epochs 20 --batch-size 128 --lr 0.002 --seed 42 --device cuda
python -m src.cli train --data-dir data/processed --checkpoint-dir checkpoints --epochs 22 --batch-size 128 --lr 0.002 --seed 42 --device cuda --resume checkpoints/last_model.pt

# 5. Đánh giá trên tập test giữ lại và đấu 42 trận/đối thủ trong CABT simulator
python -m src.cli evaluate --checkpoint checkpoints/best_model.pt --games 42 --seed 42 --out-file checkpoints/eval_results.json

# 6. Chạy bộ kiểm thử tự động
python -m pytest -p no:cacheprovider -v tests/test_pipeline.py
```

### Chạy trên Colab Pro
Mở [`notebooks/colab_train_eval.ipynb`](notebooks/colab_train_eval.ipynb), chọn **Runtime -> Change runtime type -> T4 GPU** (hoặc GPU khả dụng theo `nvidia-smi`), rồi chọn **Runtime -> Run all** để chạy toàn bộ pipeline và đồng bộ checkpoint về Google Drive.

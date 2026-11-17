# Adversarial Bug Review Report — Pokémon TCG Battle Agent

- **Commit reviewed**: `f96eb6201876b27c7f9e80a5ed52781bb913d820` (`ao/quy-t-c/root` merged fast-forward into working tree)
- **Reviewer role**: Read-only Bug Reviewer (no edits made to Worker source files)
- **Scope checked**: [agent.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/agent.py), [src/features.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/features.py), [src/dataset.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/dataset.py), [src/model.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/model.py), [src/train.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/train.py), [src/evaluate.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/evaluate.py), [src/cli.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/cli.py), [tests/test_pipeline.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/tests/test_pipeline.py), [REPORT_PHASE1_2.md](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/REPORT_PHASE1_2.md), plus empirical verification against all 40 replays in `data/replays/`, packed tensors in `data/processed/`, `checkpoints/best_model.pt`, and live `cg.game` / `cg.api`.

---

## Tóm tắt kết quả rà soát

- **Step alignment (ghép `obs_t` – `action_{t+1}` & chuyển vòng BO3)**: **ĐẠT** trên toàn bộ 40 replay (`34,980` seat-steps, `108` lần chuyển round BO3, `169` dict actions, `0` lỗi lệch bước hay vượt biên).
- **Zero future / hidden information leakage**: **ĐẠT** (`0` rò rỉ bài tay đối thủ, `0` prize ẩn đối thủ bị lộ, `top_train_deck` chỉ tính trên tập `train`, `train/val/test` tách rời hoàn toàn theo `episode_id`).
- **Lỗi nghiêm trọng & trung bình phát hiện được**: **7 bugs** (1 Critical, 2 High, 1 Medium-High, 2 Medium, 1 Low/Doc). Đặc biệt nghiêm trọng là **lỗi lệch bộ bài trong `evaluate_in_simulator`** làm sai lệch hoàn toàn tỷ lệ thắng 65%–75% báo cáo trên 3 bộ bài, **2 lỗi `0 or -1` / `0 or 1`** làm vô hiệu hóa anti-loop guard và trả hành động sai khi `maxCount == 0`, và **lỗi kẹp `MAX_CARD_ID`/`MAX_ATTACK_ID`** làm đụng độ embedding cho 13.324 lượt xuất hiện card mới & 362 lượt attack mới.

---

## Chi tiết các Bug phát hiện

### [BUG-1 — CRITICAL] Đánh giá đối kháng trong simulator bị lệch bộ bài (Asymmetric Deck Mismatch)
- **Vị trí**: [src/evaluate.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/evaluate.py#L122-L129) và [src/evaluate.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/evaluate.py#L236-L265)
- **Nguyên nhân**:
  1. Trong `evaluate_in_simulator`, vòng lặp xoay `dk` qua 3 bộ bài (`default_deck`, `dragapult_deck`, `starmie_deck`) và gọi `play_single_match(..., dk, dk)` với ý định đấu mirror match.
  2. Tuy nhiên `policy_agent` luôn gọi `model.select_action(obs_dict, default_deck=default_deck)` (cố định là `deck.csv`), còn `opp_agent` trả về `dk`.
  3. Bên trong `play_single_match`, đoạn code gọi `agents[s](init_obs)` và **ghi đè** `decks[s] = dk` trước khi gọi `cg.game.battle_start(decks[0], decks[1])`.
  4. Hậu quả: Ở 2/3 số trận (`dragapult_deck` và `starmie_deck`), model **không hề chơi mirror match**, mà cầm bộ bài meta `deck.csv` (`top_train_deck`) đi đánh với `dragapult_deck` / `starmie_deck` yếu hơn.
- **Bằng chứng thực nghiệm (12 trận/bộ bài, luân phiên Seat 0/1)**:
  - **Đấu với Heuristic Baseline**:
    - `default_deck`: Mirror thật **8/12** (`66.7%`) vs Buggy **8/12** (`66.7%`).
    - `dragapult_deck`: Mirror thật **3/12** (`25.0%`) vs Buggy (dùng `deck.csv`) **11/12** (`91.7%`).
    - `starmie_deck`: Mirror thật **3/12** (`25.0%`) vs Buggy (dùng `deck.csv`) **10/12** (`83.3%`).
    - Tổng: Mirror thật chỉ đạt **14/36 (38.9%)**, trong khi code lỗi đạt **29/36 (80.6%)**.
  - **Đấu với Random Baseline**:
    - `default_deck`: Mirror thật **10/12** (`83.3%`) vs Buggy **6/12** (`50.0%`).
    - `dragapult_deck`: Mirror thật **1/12** (`8.3%`) vs Buggy **12/12** (`100.0%`).
    - `starmie_deck`: Mirror thật **3/12** (`25.0%`) vs Buggy **8/12** (`66.7%`).
    - Tổng: Mirror thật chỉ đạt **14/36 (38.9%)**, trong khi code lỗi đạt **26/36 (72.2%)**.
- **Đề xuất sửa**: Truyền `dk` vào `policy_agent` cho pha chọn deck (hoặc bỏ bước ghi đè `decks[s]` khi `play_single_match` đã nhận `deck0, deck1` tường minh, chỉ test `init_obs` riêng).

---

### [BUG-2 — HIGH] Lỗi Falsy-Zero (`0 or -1`, `0 or 1`): Vô hiệu hóa Anti-Loop Guard & Sai hợp lệ khi `maxCount == 0`
- **Vị trí**:
  - [src/model.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/model.py#L267) & [src/evaluate.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/evaluate.py#L38): `int(sel.get("type") or -1) == 0`
  - [agent.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/agent.py#L66), [src/model.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/model.py#L228), [src/features.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/features.py#L184), [src/dataset.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/dataset.py#L142), [src/evaluate.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/evaluate.py#L36): `int(sel.get("maxCount") or 1)`
- **Nguyên nhân**:
  1. Trong `cg.api`, `SelectType.MAIN == 0`. Biểu thức `sel.get("type") or -1` khi `type == 0` sẽ trả về **`-1`** (vì `0` là falsy trong Python). Do đó `int(-1) == 0` **luôn luôn `False`**, khiến toàn bộ anti-loop guard ở `src/model.py:267` và `src/evaluate.py:38` trở thành **dead code**.
  2. Tương tự, `sel.get("maxCount") or 1` khi `maxCount == 0` sẽ biến thành `1` thay vì `0`, khiến `agent.py` và `model.select_action` chọn 1 phần tử (`[idx]`) thay vì `[]` (vi phạm ràng buộc `len(act) <= maxCount`).
- **Bằng chứng thực nghiệm**:
  - Gọi `model.select_action` với `turnActionCount = 25` và `select.type = 0` trên observation thật từ `126640082.json`: hàm trả về `idx=[6]` có `OptionType = 10 (ABILITY)` thay vì chỉ cho phép `13 (ATTACK)` / `14 (END)`.
  - Gọi `agent(obs)` với `minCount=0, maxCount=0` và `option` khác rỗng: hàm trả về `[6]` (không hợp lệ!) thay vì `[]`.
- **Đề xuất sửa**: Thay tất cả phép `or` trên field số nguyên có thể bằng `0` bằng kiểm tra `if val is None` (ví dụ `0 if sel.get("type") is None else int(sel["type"])`).

---

### [BUG-3 — HIGH] Trần `MAX_CARD_ID = 1300` và `MAX_ATTACK_ID = 1600` gây đụng độ Embedding nghiêm trọng
- **Vị trí**: [src/features.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/features.py#L18-L19) và [src/features.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/features.py#L44-L53)
- **Nguyên nhân**:
  - `_clamp_card_id` kẹp mọi `cardId >= 1300` về `1299`; `_clamp_attack_id` kẹp mọi `attackId >= 1600` về `1599`.
  - Dataset replay Kaggle `2026-11-13-v1` chứa các lá bài và chiêu thức mới hơn thư viện `cg/cg.dll` cục bộ (vốn chỉ có `1,267` lá bài và `1,556` chiêu thức).
- **Bằng chứng thực nghiệm trên 40 replay thật**:
  - Có **4 card ID >= 1300** (`1331, 1387, 1406, 1413`, max `1413`) xuất hiện tổng cộng **13.324 lần** trong deck/tay/bàn/option — tất cả bị gộp chung thành vector embedding `1299`.
  - Có **6 attack ID >= 1600** (`1641, 1642, 1715, 1716, 1747, 1748`, max `1748`) xuất hiện **362 lần** — tất cả bị gộp chung thành embedding `1599`.
  - (Lưu ý thêm: vì `cg/cg.dll` chỉ có metadata tới 1267/1556 nên `card_db`/`atk_db` trả về vector metadata `0` cho các ID mới này).
- **Đề xuất sửa**: Nâng `MAX_CARD_ID` (ví dụ `2048` hoặc `4096`) và `MAX_ATTACK_ID` (ví dụ `2048` hoặc `4096`) trước khi train lại, hoặc ánh xạ ID động qua từ điển kèm token `<UNK>`.

---

### [BUG-4 — MEDIUM-HIGH] Bất nhất giữa Loss huấn luyện và Giải mã khi `minCount == 0` (`[]`) & `stop_head`
- **Vị trí**: [src/train.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/train.py#L101-L138) và [src/model.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/model.py#L176-L207)
- **Nguyên nhân**:
  1. Trong `compute_policy_loss`, nhánh `ce_loss` lọc `has_pos = (target_sums.squeeze(-1) > 0)`. Mọi mẫu mà người chơi chọn rỗng `[]` khi `minCount == 0` (`84` mẫu ở `train`, `35` ở `val`, `14` ở `test`) bị **bỏ qua hoàn toàn** khỏi `ce_loss`.
  2. Nhánh `bce_loss` áp dụng `pos_weight = clamp((1 - chosen_mask).sum() / chosen_mask.sum(), 1.0, 5.0)` trên toàn bộ batch. Vì tỷ lệ âm/dương trung bình trong `train.pt` là **`5.92`**, `pos_weight` luôn bị kẹp tại **`5.00`**.
  3. Hơn nữa `bce_loss` tính trên **cả các mẫu `min_c == max_c`** (~90% dataset, nơi `stop_logit` không bao giờ được dùng lúc suy luận). Việc dùng `pos_weight = 5.0` làm lệch ngưỡng tối ưu của `opt_logit > stop_logit` xuống mức xác suất `P(chosen) > 1/6 ≈ 16.7%`, gây mất cân chỉnh ngưỡng dừng khi suy luận `minCount == 0` hoặc `minCount < maxCount`.
- **Bằng chứng thực nghiệm trên `test.pt` (`checkpoints/best_model.pt`)**:
  - Trong `225` mẫu `min_count == 0` ở `test.pt`: có `14` mẫu `gold == []`, model chỉ dự đoán đúng `5/14` (**35.7%** recall), đồng thời tạo ra `23` trường hợp dự đoán nhầm `[]` khi `gold != []`.
  - Độ chính xác `(min=0, max=1)` là `95/188 (50.53%)` và `(min=0, max=4)` là `0/2 (0.00%)`.
- **Đề xuất sửa**: Thêm một "null/stop option" tường minh vào `ce_loss` cho các bước có `minCount == 0` (để `[]` được học bằng softmax cross-entropy trực tiếp với các lựa chọn), hoặc chỉ huấn luyện `stop_head` trên các mẫu `minCount < maxCount` với `pos_weight = 1.0` (unbiased calibration).

---

### [BUG-5 — MEDIUM] Lỗi `CosineAnnealingLR` bật ngược Learning Rate khi `--resume`
- **Vị trí**: [src/train.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/train.py#L304) và [src/train.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/train.py#L316-L317)
- **Nguyên nhân**: `train_policy` khởi tạo `CosineAnnealingLR(optimizer, T_max=max(epochs, 1), ...)`, nhưng sau đó gọi `scheduler.load_state_dict(ckpt["scheduler_state_dict"])`. Lời gọi này **ghi đè `T_max` về giá trị cũ** (`20`) trong checkpoint, nên khi resume lên `epochs=22`, bước 21–22 vượt quá `T_max=20` làm đường cong cosine **tăng ngược LR trở lại**.
- **Bằng chứng thực nghiệm trong `checkpoints/train_summary.json`**:
  - Epoch 19: `lr = 0.000112`
  - Epoch 20: `lr = 0.000100`
  - Epoch 21 (resume): `lr = 0.000112` (tăng lên)
  - Epoch 22 (resume): `lr = 0.000146` (tăng tiếp)
- **Đề xuất sửa**: Sau `scheduler.load_state_dict(...)`, đặt lại `scheduler.T_max = max(epochs, 1)` (hoặc tái tạo scheduler với `last_epoch=start_epoch - 1` phù hợp với tổng `epochs` mới).

---

### [BUG-6 — MEDIUM] Thiếu mã hóa `AreaType.STADIUM (7)`, `CardType.STADIUM (4)` trong hand, và xử lý `max_steps` timeout thành Hòa
- **Vị trí**:
  - [src/features.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/features.py#L310-L312): One-hot `area` chỉ nhận `{1, 2, 3, 4, 5, 6, 12}`, bỏ sót **`area == 7` (`STADIUM`)** dù xuất hiện **72 lần** trong 40 replay với `(OptionType.ABILITY=10, AreaType.STADIUM=7)`.
  - [src/features.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/features.py#L249-L259): Tổng hợp bài trên tay (`state_vec[57..61]`) bỏ qua `CardType.STADIUM == 4`.
  - [src/evaluate.py](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/src/evaluate.py#L266-L275): Khi trận đấu chạm `max_steps=3000` (`winner == -1`), nhánh `else: draws += 1` âm thầm tính thành trận hòa thay vì tách riêng `timeouts`.

---

### [BUG-7 — LOW / DOCUMENTATION] Số liệu & metadata trong `REPORT_PHASE1_2.md` và `eval_results.json` không khớp
- **Vị trí**: [REPORT_PHASE1_2.md](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/REPORT_PHASE1_2.md#L40) & [REPORT_PHASE1_2.md](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/REPORT_PHASE1_2.md#L82-L86) vs [eval_results.json](file:///Z:/Word-by-Word%20Video%20Supercut/.ao-local/data/worktrees/tcg-train/tcg-train-9/checkpoints/eval_results.json)
- **Sai lệch quan sát được**:
  - `REPORT_PHASE1_2.md:82` ghi Validation `Single-Select Exact Match`: `55,72% (2.796 / 2.859 mẫu)` — thực tế `2.796 / 2.859 = 97,80%`; số đúng trong `eval_results.json` là `0.557188` (`1.593 / 2.859`).
  - `REPORT_PHASE1_2.md:84-86` ghi Validation `Multi-Select Jaccard = 62,38%`, `Turn-0 Setup = 83,33% (45/54)`, `Main-Game = 54,97% (1.620/2.947)` — trong khi `eval_results.json` ghi `61,97%`, `86,96% (60/69)`, và `54,74% (1.605/2.932)`.
  - `REPORT_PHASE1_2.md:40` ghi `maxCount in {2, 3, 4}` — thực tế tập `train` có `maxCount` tới `6` (`(1, 5)` xuất hiện 3 lần, `(6, 6)` xuất hiện 2 lần).
  - `checkpoints/eval_results.json` ghi `"git_commit": "a28d1449e836510ca3ed3e1d3b484dfc635c8c77"` (commit khởi tạo trước khi commit `f96eb6201876b27c7f9e80a5ed52781bb913d820`).

---

## Kiểm chứng các hạng mục đã đạt (Không phát hiện bug)

1. **Step Alignment (`src/dataset.py`)**:
   - Đã quét toàn bộ 40 replay (`17,490` frames, `34,980` seat-steps):
     - `80` deck select (`t=0`, `ACTIVE`, `current=None, select=None`)
     - `17,410` waiting turns (`INACTIVE`, `0` trường hợp có action khác `[]`)
     - `80` terminal (`t=len(steps)-1`, `DONE`)
     - `17,410` valid decisions (`ACTIVE`, khớp `0 <= idx < len(opts)`, `minCount <= len(act) <= maxCount`, không trùng lặp chỉ số, `0` lỗi ngoài biên)
     - Không có frame nào (`t > 0`) có cả 2 seat cùng `ACTIVE`.
     - Toàn bộ `169` dict actions (`{"action": [...]}`) đều unwrap chính xác.
     - Toàn bộ `108` lần chuyển vòng BO3 (`round` `1->2`, `2->3`) đều giữ nguyên cặp `(steps[t][seat]["observation"], steps[t+1][seat]["action"])` chuẩn xác.
2. **Zero Leakage (`src/features.py` & `src/dataset.py`)**:
   - Không đọc `steps[t+1:]` (ngoài nhãn `action` ở `t+1`), không đọc `rewards`, `visualize`, hoặc `search_begin_input`.
   - Kiểm tra 40 replay: `opp_prize_nonnone = 0`, `win_anomalies = 0` (`p_me["win"] + p_opp["win"] <= round - 1` ở mọi bước).
   - `top_train_deck` chỉ thống kê trên `split_name == "train"`; 3 tập `train/val/test` (`28/6/6` episodes) rời nhau 100%.
3. **Pytest (`tests/test_pipeline.py`)**:
   - Đã chạy lại `python -m pytest -p no:cacheprovider -v tests/test_pipeline.py`: **5/5 tests PASSED** (lưu ý: bộ test hiện tại chưa test `maxCount == 0`, `turnActionCount >= 20`, resume LR, hay mirror-match khác `default_deck`).

---

## Regression Review (Bước 2 — Commit `02035ac00469b00c53c52ff0045ee2d18548fab9`)

- **Kết luận tổng quát**: **PASS (7/7 bugs đã được khắc phục triệt để, 0 hồi quy mới)**
- **Worktree kiểm chứng**: `Z:\Word-by-Word Video Supercut\.ao-local\data\worktrees\tcg-train\tcg-train-5` (branch `ao/quy-t-c/root`, commit `02035ac00469b00c53c52ff0045ee2d18548fab9`, working tree sạch).
- **Pytest độc lập (`python -m pytest -p no:cacheprovider -v tests/test_pipeline.py`)**: **10/10 PASSED** (`6.62s`).
- **Xác nhận an toàn & công bằng**:
  - `invalid_actions == 0`, `runtime_errors == 0`, `timeouts == 0` trên cả đánh giá offline (`val`/`test`) và online simulator (`84` trận trong `eval_results.json` + `36` trận spot-check độc lập).
  - `0` rò rỉ dữ liệu tương lai hay bài ẩn đối thủ (`validate_no_leakage` đạt trên toàn bộ 40 replay; `train/val/test` tách rời theo `episode_id`: `28 / 6 / 6`).
  - Mirror-match công bằng (`decks_equal == True` trên cả 3 bộ bài `default_deck`, `dragapult_deck`, `starmie_deck`).

### Bằng chứng kiểm chứng từng Bug (`BUG-1` .. `BUG-7`)

1. **`BUG-1` (Asymmetric Deck Mismatch) — PASS**:
   - `play_single_match` (`src/evaluate.py:111-154`) giữ nguyên `deck0`/`deck1` truyền vào tường minh thay vì ghi đè từ `agents[s](init_obs)` (vẫn kiểm tra hợp lệ 60 lá từ `init_obs`), đồng thời `evaluate_in_simulator` (`src/evaluate.py:284-299`) truyền đúng `_dk=dk` vào `policy_agent` và assert `res["decks_equal"]`.
   - Kiểm chứng `test_bug1_mirror_match_deck_preservation` PASS và kiểm tra thực nghiệm trực tiếp 36 trận mirror-match (`6` trận/bộ bài x `3` bộ bài x `2` đối thủ, luân phiên Seat 0/1): `decks_equal_all=True`, `inv=0`, `err=0`, `timeouts=0` (vs Random: `default_deck` `6-0-0`, `dragapult_deck` `5-0-1`, `starmie_deck` `5-0-1`; vs Heuristic: `default_deck` `5-0-1`, `dragapult_deck` `5-0-1`, `starmie_deck` `3-0-3`).
2. **`BUG-2` (Falsy-Zero `0 or -1`, `0 or 1`) — PASS**:
   - Đã thêm `int_or(d, key, default)` (`src/features.py:24-27`) và thay thế toàn bộ các biểu thức `.get(...) or <int>` trong `agent.py`, `src/features.py`, `src/dataset.py`, `src/model.py`, `src/evaluate.py` (quét regex toàn bộ file `.py` cho `0` kết quả sót).
   - Kiểm chứng `test_bug2_falsy_zero_maxcount_zero_and_antiloop_guard` PASS; kiểm tra trực tiếp trên observation thật từ `126640082.json` với `turnActionCount=25`, `select.type=0`: `model.select_action` trả về `[8]` có `OptionType = 14 (END)`.
3. **`BUG-3` (Trần `MAX_CARD_ID` & `MAX_ATTACK_ID`) — PASS**:
   - Đã nâng `MAX_CARD_ID = 2048` và `MAX_ATTACK_ID = 2048` (`src/features.py:18-19`), bổ sung fallback metadata cho thẻ/chiêu ngoài `cg.api` (`src/features.py:386-392, 413-416`).
   - Kiểm chứng `test_bug3_high_card_and_attack_ids_not_clamped` PASS và đọc trực tiếp tensor đã đóng gói: `train.pt` có `max_card_id = 1413`, `max_attack_id = 1748` (`has_1413=True`, `has_1748=True`); `test.pt` có `max_card_id = 1413`.
4. **`BUG-4` (`minCount == 0` empty choice `[]` & `stop_head`) — PASS**:
   - `compute_policy_loss` (`src/train.py:100-156`) dùng `pos_weight = 1.0` cho relative BCE (`0.25 + 1.75 * is_var`) và thêm cột virtual stop (`max_opts`) vào `ce_loss` khi `min_count == 0` hoặc `target_sums == 0`.
   - Kiểm chứng `test_bug4_and_bug5_zero_select_loss_and_resume_lr_monotonic` PASS (gradient theo `stop_logits` âm khi `gold == []`); tái đánh giá độc lập `best_model.pt` đạt `zero_select_acc` **57.14% (`20/35`)** trên `val.pt` và **64.29% (`9/14`)** trên `test.pt` (tăng từ `35.7%` trước bản vá).
5. **`BUG-5` (`CosineAnnealingLR` bật ngược LR khi `--resume`) — PASS**:
   - `train_policy` (`src/train.py:355-360, 414-415`) cập nhật `scheduler.T_max = max(epochs, start_epoch)` sau `load_state_dict` và chặn `scheduler.step()` khi `last_epoch >= T_max`.
   - Kiểm tra toàn bộ 22 epochs trong `checkpoints/train_summary.json`: `monotonic_non_increasing=True` (Epoch 19: `0.000112`, Epoch 20: `0.000100`, Epoch 21: `0.000100`, Epoch 22: `0.000100`).
6. **`BUG-6` (`AreaType.STADIUM (7)`, `CardType.STADIUM (4)`, tách `timeouts`) — PASS**:
   - `AreaType.STADIUM (7)` được mã hóa vào `opt_feats` slot `23` (`src/features.py:316-322`), `CardType.STADIUM (4)` được đếm vào `state_vec[59]` (`src/features.py:261-270`), và `timeouts` (`winner == -1`) được tách khỏi `draws` (`winner == 2`) (`src/evaluate.py:310-318`).
   - Quét toàn bộ 40 replay xác nhận `stadium_area7_opts = 72`, `slot23_ok = 72/72`, và `test_bug6_stadium_area_and_card_type_encoding` PASS.
7. **`BUG-7` (Đồng bộ số liệu `REPORT_PHASE1_2.md`, `eval_results.json`, `train_summary.json`, `dataset_stats.json`) — PASS**:
   - `checkpoints/eval_results.json` ghi nhận đúng `git_commit = 02035ac00469b00c53c52ff0045ee2d18548fab9`, `checkpoint_epoch = 21`, `dataset_version = 2026-11-13-v1`.
   - Tái chạy `evaluate_packed_split` độc lập trên `val.pt` (`3,001` mẫu) và `test.pt` (`2,452` mẫu) khớp **100%** (`diffs_vs_json = {}`) với `eval_results.json` và toàn bộ bảng số liệu trong `REPORT_PHASE1_2.md` (`Val`: `exact_all=56.95% (1,709/3,001)`, `nontrivial=54.54% (1,550/2,842)`, `single=56.91% (1,627/2,859)`, `multi=57.75% (82/142)`, `jaccard=67.37%`, `zero=57.14% (20/35)`, `turn0=91.30% (63/69)`, `main=56.14% (1,646/2,932)`; `Test`: `exact_all=55.06% (1,350/2,452)`, `nontrivial=52.40% (1,213/2,315)`, `single=55.47% (1,283/2,313)`, `multi=48.20% (67/139)`, `jaccard=60.38%`, `zero=64.29% (9/14)`, `turn0=88.33% (53/60)`, `main=54.22% (1,297/2,392)`).
   - Tổng `by_deck` và `as_seat0_wdl + as_seat1_wdl` khớp tuyệt đối với `wins-draws-losses` trong `eval_results.json` và `REPORT_PHASE1_2.md` (`35-0-7` = `83.33%` vs `cabt_random_baseline`; `33-0-9` = `78.57%` vs `heuristic_baseline`; `0` invalid/errors/timeouts).


---

## Phase 3 Verification & Regression Review (Commits `c410ec8` / `7b52548`)

- **Kết luận tổng quát**: **100% PASS (0 bugs / 0 hồi quy sau vòng sửa lỗi)**
- **Worktree kiểm chứng**: `C:\Users\Admin\.gemini\antigravity-cli\worktrees\5a478e0f-eda6-4c5b-9082-f62139c71f53\swarm-worker-a85e1e68`
- **Branch & Commit**: `phase3` — code commit `c410ec83ece379f99f8fa5c3d718c779538bf3b2`, doc-sync HEAD `7b52548`
- **Vòng phát hiện & khắc phục trước đó (`bcb45e4` -> `c410ec8` / `7b52548`)**:
  1. Phát hiện `preprocess_replays` quét toàn bộ thư mục replay trước khi chia `train/val/test` và ghi đè `FALLBACK_CARD_DB`/`FALLBACK_ATTACK_DB` trong RAM (`1406: stage=1`, `1748: damage=330`) mà tiến trình `evaluate`/`agent.py` không nạp lại (`1406: stage=2`, `1748: damage=200`). Đã sửa triệt để ở `c410ec8`: tách `split_files` trước, chỉ suy diễn trên `split_files["train"]`, đặt `overwrite_curated=False`, và nạp `inferred_metadata.json` khi đánh giá.
  2. Phát hiện `eval_results.json` ghi `git_commit` của `main` (`b993aa5`) và `REPORT_PHASE3.md` ghi nhầm `Epoch 19` cho `eval_bc_results.json` cùng khoảng thời gian/tốc độ epoch. Đã chạy lại toàn bộ pipeline (`preprocess -> BC 20+2 -> train-rl 10+2 -> evaluate`) trên `c410ec8` và đồng bộ 100% tài liệu ở `7b52548`.

### Bằng chứng kiểm chứng độc lập (trên `7b52548` / `c410ec8`)

1. **Zero Leakage & Sharded Storage (`data/processed/`)**:
   - Đã kiểm tra toàn bộ `209` episodes (`93,829` frames, `90,609` kept decisions): `opp_hand_leak = 0`, `opp_prize_nonempty_ids = 0`, `win_anomalies = 0`.
   - `episode_id` tách rời **100%** giữa `train` (`146` eps, `61,890` mẫu / `5` shards), `val` (`31` eps, `13,970` mẫu / `1` shard), và `test` (`32` eps, `14,749` mẫu / `1` shard); `inferred_metadata.json` ghi nhận `source_split == "train"`, `num_train_episodes_scanned == 146`.
2. **Pytest & Smoke Test**:
   - `python -m pytest -p no:cacheprovider -v tests/test_pipeline.py`: **`15/15 PASSED` (`18.97s`)**.
   - `python -m src.cli smoke-test --deck deck.csv --checkpoint checkpoints/best_model.pt --seed 42`: Random vs Random (`166` steps, `0/0` faults); Checkpoint vs Random (`111` steps, `13` turns, `winner=0`, `decks_equal=true`, `invalid_actions=[0, 0]`, `runtime_errors=[0, 0]`, `1.8277 ms/decision`).
3. **Đánh giá độc lập `checkpoints/best_model.pt` (BC Ep 22 + RL Upd 5)**:
   - **Offline Test (`14,749` mẫu)** khớp 100% với `checkpoints/eval_results.json`: `loss = 1.2271`, `exact_match_all = 61.53% (9,075/14,749)`, `exact_match_nontrivial = 59.70% (8,406/14,080)`, `single_select_acc = 61.65% (8,489/13,770)`, `multi_select_exact_acc = 59.86% (586/979)`, `multi_select_jaccard = 72.87%`, `zero_select_acc = 68.81% (75/109)`, `turn0_setup_acc = 90.99% (323/355)`, `invalid_actions = 0`.
   - **Simulator Live Check độc lập** (`invalid_actions = 0`, `runtime_errors = 0`, `timeouts = 0`):
     - Mirror vs `cabt_random_baseline` (`42` trận): **`41-0-1 (97.62%)`** (báo cáo trong `eval_results.json`: `39-0-3 / 92.86%`).
     - Mirror vs `heuristic_baseline` (`42` trận): **`37-0-5 (88.10%)`** (khớp `37-0-5 / 88.10%` trong `eval_results.json`).
     - Cross-Deck Matrix vs `heuristic_baseline` (`54` trận): **`43-0-11 (79.63%)`** (`eval_results.json`: `50-0-4 / 92.59%`; sai khác do RNG xáo bài nội bộ C++ của `cg.game`).
     - CPU Latency thực đo: mirror `mean = 2.0233 ms`, `p99 = 3.5264 ms`; cross-deck `mean = 2.1908 ms`, `p99 = 4.0867 ms` (tương đồng với `1.9230 ms` / `3.1413 ms` trong `eval_results.json`).
4. **Cấu hình & Git Hygiene**:
   - `AGENTS.md` cập nhật đúng quy tắc model cho `@worker` (`gemini-4-hydrogen-high`), `@reviewer` (`gemini-4-hydrogen-high`), và `@monitor` (`gemini-4-argon-high`).
   - `.gitignore` giữ `data/`, `checkpoints/`, `cg/*.dll`, `cg/*.so` ở trạng thái untracked/ignored (`git status --ignored -s` xác nhận).
   - Toàn bộ số liệu trong `REPORT_PHASE3.md`, `README.md`, và `HANDOFF.md` khớp 100% với `dataset_stats.json`, `train_summary.json`, `rl_summary.json`, `eval_bc_results.json`, và `eval_results.json`.

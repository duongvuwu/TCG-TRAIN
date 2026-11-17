# HANDOFF — Pokémon TCG Battle Agent (Phase 3: Scaling, Conditional Multi-Select & RL Self-Play)

**Cập nhật lần cuối**: `2026-11-17T23:28+07:00`  
**Worktree / CWD**: `C:\Users\Admin\.gemini\antigravity-cli\worktrees\5a478e0f-eda6-4c5b-9082-f62139c71f53\swarm-worker-a85e1e68`  
**Git Branch**: `phase3` (branched from `main` `b993aa5`; code commit `c410ec8`; local-only, **KHÔNG** `git push`)

---

## 1. Mục Tiêu & Quy Tắc Dự Án (`./AGENTS.md`)

1. **Bài toán**: Huấn luyện và đánh giá agent chọn hành động hợp lệ cho simulator **Pokémon TCG (`cabt` / `cg`)** của Kaggle Playground, sử dụng dataset replay ngày `2026-11-13` (Version 1):
   - Dataset: `kaggle/the-pokemon-company-ptcg-ai-battle-challenge-playground-episodes-2026-11-13` (Version 1, `2026-11-13-v1`, `1,644` episodes trong `manifest.csv`)
   - Tài liệu simulator CABT: `https://matsuoinstitute.github.io/cabt/`
2. **Phân công vai trò (đã cập nhật trong `./AGENTS.md`)**:
   - **Worker coding (`@worker`)**: **Gemini 4 Hydrogen (High)** (`--model gemini-4-hydrogen-high`)
   - **Reviewer bug (`@reviewer`)**: **Gemini 4 Hydrogen (High)** (`--model gemini-4-hydrogen-high`)
   - **Swarm Monitor (`@monitor`)**: **Gemini 4 Argon (High)** (`--model gemini-4-argon-high`)
3. **Quy tắc cứng**:
   - Giữ token, dữ liệu tải xuống (`data/`), thư viện nhị phân simulator (`cg/*.dll`, `cg/*.so`), và checkpoint (`checkpoints/`) ngoài Git (`.gitignore`).
   - Không rò rỉ thông tin tương lai hoặc trạng thái ẩn của đối thủ (`players[1 - yourIndex]["hand"] is None`).

---

## 2. Trạng Thái Các Hạng Mục Phase 3

| Thành phần | Trạng thái hiện tại | Ghi chú |
|---|---|---|
| Cập nhật `AGENTS.md` | **ĐÃ HOÀN TẤT** | Cập nhật model rules cho `@worker`, `@reviewer`, `@monitor` |
| Data Scaling & Sharded Storage (`src/dataset.py`, `src/cli.py`, `notebooks/colab_train_eval.ipynb`) | **ĐÃ HOÀN TẤT** | Hỗ trợ `--num-top`, `--num-strided`, `--max-episodes 1644`; lưu shard `*_shard_*.pt` + `*_shards.json` giới hạn RAM `O(1 shard)` |
| Tiền xử lý mở rộng (`data/replays/` `209` episodes -> `data/processed/`) | **ĐÃ HOÀN TẤT** | `93,829` frames (`187,658` seat-steps), **`90,609` kept decisions** (`61,890` train across `5` shards / `13,970` val / `14,749` test), `0` lỗi ngoài miền, `100%` episode-disjoint, `inferred_metadata.json` (`source_split == "train"`) |
| Fallback Metadata, Richer Features & Conditional Multi-Select (`src/features.py`, `src/model.py`) | **ĐÃ HOÀN TẤT** | `FALLBACK_CARD_DB` (`1331..1413`), `FALLBACK_ATTACK_DB` (`1641..1748`), `infer_fallback_metadata_from_replays` (chỉ quét train), `STATE_DIM=112`, `OPT_DIM=64`, `opt_attn`, `cond_scorer`, `cond_stop_head` |
| Value Head & PPO + KL-to-BC RL Self-Play (`src/model.py`, `src/train.py`, `src/cli.py`) | **ĐÃ HOÀN TẤT** | `value_head` tương thích ngược 100% với Phase 2 (`64/48`), `train-rl` + `--resume` chạy trên cả 3 bộ bài trong `cg.game` |
| Huấn luyện BC & Resume (`checkpoints/best_bc_model.pt`, `last_model.pt`, `train_summary.json`) | **ĐÃ HOÀN TẤT** | BC Epoch 22 (`val_exact_nontrivial = 61.01%`, `val_exact_all = 62.86%`, `val_multi_exact_acc = 60.98%`, `lr = 0.000100`, Peak VRAM `88.00 MB`, speed `2,655.3–5,800.7 samp/s`) |
| Huấn luyện RL & Resume (`checkpoints/best_rl_model.pt` -> `best_model.pt`, `rl_summary.json`) | **ĐÃ HOÀN TẤT** | 10 updates + resume 2 updates (`12` total), Peak VRAM `42.70 MB`, eval nhanh 18 trận vs Heuristic đạt **`100.00%`** (`best_rl_model.pt` tại RL Update 5 -> `best_model.pt`) |
| Đánh giá Mirror + Cross-Deck Matrix + Latency (`checkpoints/eval_results.json`, `git_commit = c410ec8`) | **ĐÃ HOÀN TẤT** | Mirror vs Random `39-0-3 (92.86%)`, Mirror vs Heuristic `37-0-5 (88.10%)`, Cross-Deck Matrix `50-0-4 (92.59%)`, CPU latency `mean = 1.9230 ms`, `p99 = 3.1413 ms`, `0` lỗi |
| Bộ kiểm thử hồi quy (`tests/test_pipeline.py`) | **15/15 PASSED (`19.34s`)** | Đã kiểm chứng `python -m pytest -p no:cacheprovider -v tests/test_pipeline.py` |
| Tài liệu (`REPORT_PHASE3.md`, `README.md`, `HANDOFF.md`) | **ĐÃ HOÀN TẤT** | Đồng bộ 100% với `dataset_stats.json`, `train_summary.json`, `rl_summary.json`, `eval_bc_results.json`, `eval_results.json` |

---

## 3. Tóm Tắt Kết Quả Đánh Giá Phase 3 (`checkpoints/eval_results.json`, BC Ep 22 + RL Upd 5, `c410ec8`)

- **Offline Validation (`13,970` mẫu, `13,305` nontrivial)**:
  - `loss = 1.2187`, `exact_match_all = 61.04% (8,527/13,970)`, `exact_match_nontrivial = 59.09% (7,862/13,305)`
  - `single_select_acc = 61.06% (8,054/13,191)`, `multi_select_exact_acc = 60.72% (473/779)`, `multi_select_jaccard = 72.66%`, `zero_select_acc = 76.47% (91/119)`, `turn0_setup_acc = 88.65% (289/326)`, `main_game_acc = 60.38% (8,238/13,644)`, `invalid_actions = 0`
- **Offline Test (`14,749` mẫu, `14,080` nontrivial)**:
  - `loss = 1.2271`, `exact_match_all = 61.53% (9,075/14,749)`, `exact_match_nontrivial = 59.70% (8,406/14,080)`
  - `single_select_acc = 61.65% (8,489/13,770)`, `multi_select_exact_acc = 59.86% (586/979)`, `multi_select_jaccard = 72.87%`, `zero_select_acc = 68.81% (75/109)`
  - `turn0_setup_acc = 90.99% (323/355)`, `main_game_acc = 60.80% (8,752/14,394)`, `invalid_actions = 0`
  - *(Phase 3 BC trước RL tại `checkpoints/eval_bc_results.json`, Epoch 22: `exact_match_all = 62.87% (9,273/14,749)`, `exact_match_nontrivial = 61.11% (8,604/14,080)`, `multi_select_exact_acc = 62.31% (610/979)`)*
- **Simulator Mirror Matches (`42` trận / đối thủ, `14` trận / bộ bài, `decks_equal == True`, `model_invalid_actions = 0`, `model_runtime_errors = 0`, `timeouts = 0`)**:
  - **vs `cabt_random_baseline`**: **39W – 0D – 3L (`92.86%`)** (`Seat 0: 20-0-1`, `Seat 1: 19-0-2`; `default_deck: 12-0-2`, `dragapult_deck: 13-0-1`, `starmie_deck: 14-0-0`)
  - **vs `heuristic_baseline`**: **37W – 0D – 5L (`88.10%`)** (`Seat 0: 19-0-2`, `Seat 1: 18-0-3`; `default_deck: 13-0-1 (92.86%)`, `dragapult_deck: 12-0-2 (85.71%)`, `starmie_deck: 12-0-2 (85.71%)`)
- **Cross-Deck Matchup Matrix (`54` trận = `6` trận x `9` cặp vs `heuristic_baseline`, `0` lỗi)**:
  - **Tổng cộng**: **50W – 0D – 4L (`92.59%`)** (tăng từ `83.33%` / `45-0-9` ở BC Epoch 22)
  - `default_deck`: `6-0-0` vs `default`, `6-0-0` vs `dragapult`, `5-0-1` vs `starmie`
  - `dragapult_deck`: `6-0-0` vs `default`, `6-0-0` vs `dragapult`, `5-0-1` vs `starmie`
  - `starmie_deck`: `6-0-0` vs `default`, `5-0-1` vs `dragapult`, `5-0-1` vs `starmie`
- **CPU Inference Latency**:
  - Mirror matches (`5,007` decisions): `mean = 1.9230 ms`, `p50 = 1.9432 ms`, `p95 = 2.6736 ms`, `p99 = 3.1413 ms`, `max = 26.6510 ms`
  - Cross-deck matrix (`3,015` decisions): `mean = 1.8672 ms`, `p50 = 1.8908 ms`, `p95 = 2.5647 ms`, `p99 = 3.1299 ms`, `max = 6.1156 ms`

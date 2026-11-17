# Pokémon TCG Battle Agent

## Mục tiêu

Huấn luyện và đánh giá một agent chọn hành động hợp lệ trong simulator Pokémon TCG của Kaggle Playground. Dùng [dataset replay ngày 2026-11-13](https://www.kaggle.com/datasets/kaggle/the-pokemon-company-ptcg-ai-battle-challenge-playground-episodes-2026-11-13), phiên bản 1, và đối chiếu với [tài liệu simulator](https://matsuoinstitute.github.io/cabt/) cùng [mô tả cuộc thi](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/overview/description).

## Giao việc cho agent

- **Worker coding (`@worker`)** dùng **Gemini 4 Hydrogen (High)** cho task triển khai hoặc sửa code. Với Antigravity CLI, đặt `--model gemini-4-hydrogen-high`.
- **Reviewer bug (`@reviewer`)** dùng **Gemini 4 Hydrogen (High)** cho task rà lỗi và hồi quy. Với Antigravity CLI, đặt `--model gemini-4-hydrogen-high`.
- **Swarm Monitor (`@monitor`)** dùng **Gemini 4 Argon (High)** cho task giám sát tiến độ và gỡ kẹt swarm. Với Antigravity CLI, đặt `--model gemini-4-argon-high`.
- Khi tạo task mới cho các vai trò trên, chỉ định model và mức suy luận ngay lúc giao việc. Khi điều phối từ Codex, dùng Antigravity cho các task cần các model Gemini này. Nếu model đã chọn không khả dụng, báo rõ model và vai trò bị ảnh hưởng.

## Pipeline

1. Đọc `manifest.csv` và một nhóm replay nhỏ trên CPU. Xác minh cách ghép quan sát trước quyết định, danh sách lựa chọn hợp lệ và hành động đã chọn. Nhận diện riêng các bước setup/chọn deck, lượt chờ và kết thúc. Ghi số mẫu giữ lại hoặc loại và lý do.
2. Xử lý từng replay để giới hạn RAM. Chia train/validation/test theo episode trước khi tách thành các quyết định. Chỉ dùng thông tin mà agent thực sự nhìn thấy tại thời điểm chọn nước đi.
3. Nếu replay cung cấp đủ cặp quan sát–hành động, bắt đầu bằng một policy imitation learning nhỏ. Giới hạn đầu ra theo lựa chọn hợp lệ và giữ đúng dạng chỉ số mà simulator yêu cầu, kể cả khi phải chọn nhiều mục. Chạy thử pipeline và nạp lại checkpoint trên một tập nhỏ trước khi chạy lớn.
4. Đánh giá trên episode giữ lại và chạy trận với đối thủ baseline trong simulator. Báo cáo độ chính xác bắt chước, tỷ lệ thắng và lỗi hành động hoặc lỗi chạy trận. Ghi phiên bản dữ liệu, commit, seed, cấu hình và kết quả của từng lần chạy.

## Chạy trên Colab Pro

- Đặt logic tải dữ liệu, tiền xử lý, train, resume và eval vào script Python chạy từ CLI.
- Dùng **CPU** cho kiểm tra replay mẫu, tiền xử lý ban đầu và smoke test simulator. Khi các bước này thành công, ưu tiên **một NVIDIA T4** cho lần train baseline nếu Colab cấp được.
- Kiểm tra GPU và VRAM thực tế bằng `nvidia-smi`. Chạy một đoạn train ngắn để đo tốc độ và VRAM đỉnh, rồi ước tính thời gian chạy đầy đủ. Nếu T4 không khả dụng, chọn GPU phù hợp từ các runtime tài khoản có thể dùng. Nếu thiếu VRAM hoặc train quá chậm, cân nhắc GPU khác dựa trên VRAM cần thiết, thời gian dự kiến và mức tiêu thụ compute units nếu có số liệu.
- Ưu tiên [Colab MCP chính thức](https://developers.google.com/colab/api/reference/mcp) khi đã cấu hình và xác thực bằng tài khoản có Colab Pro. Kiểm tra `get_subscription`, `list_runtimes` và `list_runtime_specs` trước khi chọn runtime. Tái sử dụng runtime phù hợp, chuyển mã theo một commit xác định hoặc tải mã vào runtime, rồi gọi CLI bằng `execute_code` trong session đó.
- Với lần chạy dài qua MCP, dùng `session` và `executionId`. Khi gặp `DEADLINE_EXCEEDED` hoặc `UNAVAILABLE`, gửi lại đúng lời gọi với cùng ID để nối lại. Khi runtime đã mất, tiếp tục từ checkpoint đã lưu. Lưu định kỳ checkpoint có trạng thái đủ để resume, cấu hình và metrics vào Google Drive hoặc nơi lưu trữ bền vững đã được cấu hình. Kiểm tra đọc lại được checkpoint trên phiên mới.
- Khi MCP chưa dùng được, tạo một notebook Colab mỏng để nạp mã, cấu hình lưu trữ, kiểm tra thiết bị và gọi cùng các lệnh CLI. Người dùng mở notebook, chọn runtime và bấm **Run all**. Giữ logic train trong repo.
- Dùng môi trường hoặc kho bí mật phù hợp cho thông tin xác thực. Giữ token, replay tải xuống và checkpoint dung lượng lớn ngoài Git.

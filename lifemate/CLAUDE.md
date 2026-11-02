# LifeMate (Zalo AI Challenge 2026)

Khi bắt đầu phiên làm việc trong thư mục này:

1. Chạy `bash scripts/setup_env.sh`. Script kiểm tra `.venv` + kaggle CLI, cài nếu thiếu, rồi kiểm tra đăng nhập. Nếu báo chưa đăng nhập (exit 2): dừng và nhờ người dùng chạy `! .venv/bin/kaggle auth login` (tài khoản jasonhoang666). Không tự làm bước OAuth, không đọc/in token.
2. Đọc `memory/MEMORY.md` và các file trong `memory/` (quy tắc làm việc, ràng buộc cuộc thi, cách chạy Kaggle).
3. Đọc `PROGRESS.md`: tiến độ, điểm, job đang chạy, bước tiếp theo. Cập nhật sau mỗi lượt chạy.

Khi học được điều mới cần nhớ lâu dài (quy tắc người dùng đặt, quyết định dự án), ghi vào `memory/` theo cùng định dạng và thêm 1 dòng vào `memory/MEMORY.md`, để máy khác cũng có.

`data/`, `models/`, `output/`, `.venv/` không nằm trong git (xem `.gitignore`).

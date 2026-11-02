# Tiến độ: lifemate

| Bước | Trạng thái | Ghi chú |
|---|---|---|
| Rule | xong | `00-rules.md`, bản gốc trong `rules-raw/`. Tab Data cần đăng nhập, chưa đọc |
| P0 | xong | Hướng: tool-calling LLM nhỏ + ASR tiếng Việt. Venue lọc mềm. Đầu ra tiếng Việt |
| P1 | xong | 14 paper, đủ PDF, không có paper phải tải tay |
| P2 | xong | `01-papers.html` |
| DỪNG 1 | xong | Chọn: chen-2025-hammer, zhuo-2025-vietasr, prabhakar-2025-apigenmt. Ưu tiên paper về kiến trúc/phương pháp trước, paper về dataset làm sau (hiện chỉ dùng dataset public) |
| P3 | đang làm | Xong: chen-2025-hammer (16 node), zhuo-2025-vietasr (18 node). Chưa làm: prabhakar-2025-apigenmt (paper về dataset, để sau) |

Điểm xuất phát: người dùng đã biết Transformer/LLM decoder, SFT/LoRA, function calling. Chưa trả lời câu hỏi về nền speech, nên roadmap VietASR bắt đầu từ audio → Fbank.

Phát hiện phụ cho cuộc thi: checkpoint `zzasdf/viet_iter3_pseudo_label` (Apache-2.0, tạo 2025-04-20) dùng được theo rule; ChunkFormer-CTC-Large-Vie dùng CC BY-NC 4.0, không hợp rule.

2026-10-29: người dùng chuyển sang dựng pipeline thay vì đọc khớp từng paper. Notebook: `notebooks/lifemate_pipeline.ipynb` (nguồn `notebooks/lifemate_pipeline.py`). Khối ADAPTER còn phải sửa theo dev kit thật.

Bước kế (nếu quay lại research): người dùng học 2 roadmap → P4 deep dive Hammer, VietASR → sau đó P3/P4 cho APIGen-MT.

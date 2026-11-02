# Phạm vi
- Chủ đề: tool-calling agent dùng LLM nhỏ + ASR tiếng Việt, phục vụ Zalo AI Challenge 2026 - LifeMate (rule: `00-rules.md`).
- Câu hỏi:
  1. Làm sao để LLM < 5B (context ≤ 16K, greedy, ≤ 30s/sample trên A30) gọi tool multi-step đúng tên + đúng tham số, biết hỏi thông tin thiếu và biết từ chối khi ngoài phạm vi?
  2. ASR tiếng Việt nào (đa vùng miền, lẫn tiếng Anh, tên riêng/số tiền/địa chỉ) vừa đủ chính xác vừa vừa ngân sách tham số còn lại (tổng < 6B)?
- Năm: 2024–2026 (+ paper nền tảng)
- Mục đích: thi đấu (chọn model, cách fine-tune, sinh dữ liệu), có dàn ý/report nội bộ.
- Ngôn ngữ đầu ra (dàn ý, report, chữ trong hình): tiếng Việt.
- Ràng buộc từ rule ảnh hưởng tới chọn model: license OSI-approved không cấm thương mại, phát hành trước 26/10/2026.

## Venue (lọc mềm, người dùng đã chọn)
| Venue | Loại | Hạng | Nguồn tra hạng | Ghi chú |
|---|---|---|---|---|
| ACL, EMNLP, NAACL | conference | CORE A* / A* / A | portal.core.edu.au (CORE2023) | venue chính NLP |
| ICLR, NeurIPS, ICML | conference | CORE A* | portal.core.edu.au (CORE2023) | |
| ACL/EMNLP/NAACL Findings | conference track | không xếp hạng riêng | | ghi rõ Findings |
| Interspeech | conference | CORE A | portal.core.edu.au (CORE2023) | venue chính ngành speech |
| ICASSP | conference | CORE B | portal.core.edu.au (CORE2023) | giữ lại vì là venue chính ngành speech |
| arXiv | preprint | - | | gắn `[arXiv]`, kiểm đã được nhận chưa |

Hạng CORE cần tra lại ở P1 trước khi ghi vào `01-candidates.md`.

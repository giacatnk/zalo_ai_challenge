# Zalo AI Challenge 2026 - LifeMate: tóm tắt rule

Nguồn: tab Overview của portal (API `api-challenge.zalo.ai/fe/api/config-detail?title=1`), lấy ngày 2026-10-29. Bản gốc lưu ở `rules-raw/`. Tab Data cần đăng nhập + đăng ký đội, chưa đọc được.

## Bài toán
Agent nhận **audio tiếng Việt** (có thể lẫn từ tiếng Anh, giọng Bắc/Trung/Nam, cả giọng tổng hợp lẫn giọng thật) và sinh ra **danh sách tool call có thứ tự** để thực hiện yêu cầu.

Input:
- Audio: yêu cầu mới nhất của người dùng.
- User context: thời gian hiện tại, timezone, vị trí, user profile (địa chỉ nhà, nơi làm...).
- Available apps: smart_home, banking, ride, food delivery, email..., mỗi app có tool mô tả bằng JSON schema.
- Conversation history: các lượt trước (nếu có).
- Môi trường giả lập (`zaic_tools`, `ToolEnv`) giữ dữ liệu app. Chỉ đọc/ghi qua tool call.

Output: mỗi dòng trong `submission.jsonl` = `{"id", "tool_calls": [{"name", "arguments"}...], "state_hash"}`. Ba kiểu hành vi:
1. Thực thi: gọi tool theo chuỗi (kết quả bước trước dùng cho bước sau, ví dụ `smart_home.get_status` để tìm `device_id`).
2. Thiếu thông tin: gọi `sys.ask_missing_info` (chỉ rõ tool dự định + tham số thiếu), có thể sau khi đã gọi vài tool khác.
3. Ngoài phạm vi: trả về `[]` nếu không app nào xử lý được.
`state_hash` lấy từ `env.get_state_hash()` sau khi chạy xong.

## Metric
Với mỗi sample:
- Gold rỗng: điểm 1 nếu submit cũng rỗng, ngược lại 0.
- Gold không rỗng: điểm 0 nếu `hash_sys != hash_gold` hoặc M = 0; ngược lại điểm = F1 = 2PR/(P+R), với P = M/(M+U), R = M/|G|.
  - M: số call khớp gold (cùng tên tool, mọi tham số non-fuzzy giống hệt, tham số fuzzy có token-F1 > 0.6 sau lowercase + Unicode NFC). Mỗi call khớp tối đa 1 gold call, **không xét thứ tự**.
  - U: số call submit không khớp (kể cả call bị lỗi schema).
  - Trường free-text (fuzzy) không ảnh hưởng hash.
- Final score = 100/N · Σ score_i.

Hệ quả:
- **State sai thì mất trắng điểm sample**, nên đúng tham số của các tool ghi (write) là quan trọng nhất.
- Call đọc thừa (get_status không có trong gold) làm giảm P. Cần biết gold có chứa các call đọc hay không (ví dụ mẫu có `smart_home.get_status` trong gold) để không gọi thừa hoặc gọi thiếu.
- Mọi call trên ToolEnv phải có trong submission, nên không được "thử rồi bỏ".

## Ràng buộc kỹ thuật
- Tổng toàn hệ thống (ASR + LLM + thành phần khác) **< 6B tham số**. LLM **< 5B**, context **≤ 16,384 token**.
- Mỗi sample **≤ 30 giây**. Phần cứng chấm: 1× NVIDIA A30 (24 GB), CPU Xeon Gold 6442Y 16 core, 64 GB RAM.
- Chạy **offline hoàn toàn**, không gọi API ngoài lúc inference.
- Chỉ dùng dữ liệu public và model open-source **phát hành trước 26/10/2026**, license cho phép. Quy chế mục 4.5: thư viện/source code open-source phải có **license OSI-approved, không cấm dùng thương mại** (cần kiểm license từng model: Apache-2.0/MIT ổn; Gemma, Llama, Qwen-Research license cần xem kỹ).
- Được dùng LLM/TTS bên ngoài để **sinh dữ liệu tổng hợp** lúc phát triển, không dùng lúc inference.
- Cấm gán nhãn public_test/private_test (tay hoặc bằng model) để train/tune.
- Cấm đọc `state.json` hay bất kỳ file nào dưới `<split>_environment/`. Đúng một ToolEnv cho mỗi sample.
- Tái lập được: cố định seed, greedy / temperature = 0. Nếu chạy lại khác kết quả, BTC chạy Docker 5 lần lấy trung bình.
- Tie-break: chấm lại F1 tool_calls bỏ qua hash, rồi đến thời gian xử lý trung bình thấp hơn.

## Vòng thi và nộp bài
| Mốc (2026) | Việc |
|---|---|
| 26/10 | Mở: dev kit, train + public test, phần mềm chấm |
| 26/10 – 16/11 | Public test, nộp jsonl, tối đa 5 lần/ngày |
| 16/11 23:59 | Nộp solution: Docker image (train code, inference code, data), link Google Drive + MD5, tối đa 2 lần, chấm lần cuối |
| 20/11 09:00 | Phát private test + environment |
| 23/11 23:59 | Chạy solution đã freeze, nộp jsonl private |
| 02/12 | Công bố top 2 mỗi track |
| 12/12 | Công bố kết quả cuối |

Docker: BTC mount private test vào `/data`, chạy script inference, lấy `result/submission.jsonl`.

## Điều kiện chung (quy chế)
- Quốc tịch Việt Nam. Đội tối đa 5 người, mỗi người 1 đội.
- Giải: Nhất 3,500 USD, Nhì 2,500 USD mỗi track, top 5 nhận quà.
- Người thắng phải nộp source code + tài liệu, cấp quyền sử dụng không độc quyền cho BTC.

## Public leaderboard (29/10/2026, ngày thứ 4)
Top 1: 97.57, top 10 ≈ 94.9, top 20 ≈ 90.9. Điểm public đã rất cao, nên khác biệt nằm ở các ca khó (ASR tên riêng/số tiền, thiếu thông tin, out-of-scope) và ở private test.

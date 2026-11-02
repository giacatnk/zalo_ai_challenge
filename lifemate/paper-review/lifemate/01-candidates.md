# Danh sách paper (n = 14)

| # | id | Title | Venue + năm | Hạng | Cờ | Hướng | PDF |
|---|---|---|---|---|---|---|---|
| 1 | prabhakar-2025-apigenmt | APIGen-MT | NeurIPS D&B 2025 | A* | | sinh dữ liệu multi-turn có kiểm chứng | pdf/prabhakar-2025-apigenmt.pdf |
| 2 | chen-2025-hammer | Hammer | ICLR 2025 | A* | | FC cho model nhỏ, function masking | pdf/chen-2025-hammer.pdf |
| 3 | ross-2025-when2call | When2Call | NAACL 2025 | A | | gọi tool / hỏi lại / từ chối | pdf/ross-2025-when2call.pdf |
| 4 | qian-2025-toolrl | ToolRL | arXiv 2025 | - | [arXiv] | GRPO + reward chi tiết | pdf/qian-2025-toolrl.pdf |
| 5 | peng-2026-spokenfc | Spoken Function Calling | ACM MM 2026 | A* | | FC từ giọng nói | pdf/peng-2026-spokenfc.pdf |
| 6 | zhuo-2025-vietasr | VietASR | Interspeech 2025 | A | | ASR Việt nhỏ, SSL | pdf/zhuo-2025-vietasr.pdf |
| 7 | yao-2025-taubench | τ-bench | ICLR 2025 | A* | [nền tảng] | chấm theo trạng thái DB | pdf/yao-2025-taubench.pdf |
| 8 | patil-2025-bfcl | BFCL | ICML 2025 | A* | [nền tảng] | benchmark FC | pdf/patil-2025-bfcl.pdf |
| 9 | liu-2025-toolace | ToolACE | ICLR 2025 | A* | | sinh dữ liệu FC | pdf/liu-2025-toolace.pdf |
| 10 | yin-2025-magnet | Magnet | ACL 2025 | A* | | đồ thị function → multi-turn | pdf/yin-2025-magnet.pdf |
| 11 | zhang-2025-xlam | xLAM | NAACL 2025 | A | | model agent mở | pdf/zhang-2025-xlam.pdf |
| 12 | le-2024-phowhisper | PhoWhisper | ICLR 2024 Tiny Papers | - | [nền tảng] | Whisper fine-tune Việt | pdf/le-2024-phowhisper.pdf |
| 13 | le-2025-chunkformer | ChunkFormer | ICASSP 2025 | B | [hạng B] | ASR long-form | pdf/le-2025-chunkformer.pdf |
| 14 | mishra-2026-mtvabench | MTVA-Bench | arXiv 2026 | - | [arXiv] | LLM trong voice cascade | pdf/mishra-2026-mtvabench.pdf |

Venue đã kiểm bằng web: ToolACE, Hammer (header ICLR 2025 trong PDF/proceedings), BFCL (PMLR v267), APIGen-MT (NeurIPS 2025 D&B), When2Call, xLAM (ACL Anthology NAACL 2025), Magnet (ACL Anthology ACL 2025), VietASR (ISCA Archive), ChunkFormer (IEEE Xplore ICASSP 2025), Spoken FC (ACM reference trong PDF: MM '26). τ-bench ICLR 2025 chỉ có nguồn thứ cấp xác nhận. Hạng CORE ghi theo CORE2023, chưa tra lại trên portal.

## Đề xuất 6 paper (sau P2)
1. prabhakar-2025-apigenmt: dữ liệu multi-turn có hành động ground-truth + môi trường thực thi, giống setup zaic_tools; có model 1B/3B.
2. chen-2025-hammer: FC cho model 1.5B/4B, có dữ liệu irrelevance (ca trả `[]`).
3. ross-2025-when2call: ca hỏi lại / từ chối, ứng với `sys.ask_missing_info` và `[]`.
4. qian-2025-toolrl: reward theo tên + tham số chuyển thẳng từ metric cuộc thi; thí nghiệm 1.5B/3B.
5. peng-2026-spokenfc: đo mức rơi điểm do lỗi ASR trong FC, lỗi chính ở tên riêng và số.
6. zhuo-2025-vietasr: ASR Việt 68M tham số, WER tốt hơn Whisper large-v3.

## Paper bỏ ra và lý do
- Các paper TTS, speech LLM end-to-end: người dùng không chọn hướng này ở P0.
- VietPrism (arXiv 2609.30005), ViMedCSS (arXiv 2602.12911): là dataset, không phải phương pháp; có thể dùng khi tìm dữ liệu train ASR.
- MFCL Audio (ICML 2026): chưa tìm thấy bản PDF công khai.

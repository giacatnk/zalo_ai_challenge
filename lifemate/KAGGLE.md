# Chạy LifeMate pipeline trên Kaggle

Model (Qwen3-4B-Instruct-2507, VietASR) được notebook tự tải từ Hugging Face trên Kaggle. Máy bạn chỉ giữ dataset.

```
zalo-ai/
  data/lifemate-data.zip              file upload lên Kaggle (đã bỏ __MACOSX, .DS_Store)
  data/lifemate/                      bản giải nén để xem trên máy
  notebooks/lifemate_pipeline.ipynb   notebook import vào Kaggle
  notebooks/lifemate_pipeline.py      bản text; sửa file này rồi chạy: python3 scripts/py_to_ipynb.py notebooks/lifemate_pipeline.py
```

Dùng `lifemate-data.zip`, **không** dùng `train+public_test.zip` gốc: file gốc có thư mục `__MACOSX/` chứa file rác tên `._train.jsonl`, `._public_test.jsonl`.

---

## Bước 1. Upload dataset lên Kaggle (làm một lần)

1. Đăng nhập https://www.kaggle.com.
2. Thanh bên trái → **Datasets** → nút **+ New Dataset** (góc trên phải).
3. Kéo thả file `zalo-ai/data/lifemate-data.zip` vào khung upload (hoặc bấm **Browse Files** và chọn file). Chờ thanh tiến trình chạy hết (khoảng 830 MB).
4. Ô **Dataset title**: gõ `lifemate-data`. Kaggle tự đặt URL `kaggle.com/datasets/<username>/lifemate-data`.
5. Giữ **Private** (mặc định). README của BTC và quy chế mục 4.3 cấm chia sẻ dữ liệu ra ngoài đội, nên **không bật Public**.
6. Bấm **Create**. Kaggle tự giải nén zip; chờ trạng thái chuyển từ "Processing" sang xong (vài phút).
7. Kiểm tra: trang dataset → tab **Data** → thấy cây `lifemate/train/train.jsonl`, `lifemate/public_test/public_test.jsonl`, `audio/`...
8. (Tuỳ chọn) Chia cho đồng đội: trang dataset → **Settings → Sharing** → thêm username của họ.

## Bước 2. Tạo notebook (làm một lần)

1. Thanh bên trái → **Code** → **+ New Notebook**.
2. Menu **File → Import Notebook** → chọn `zalo-ai/notebooks/lifemate_pipeline.ipynb` → **Import**.
3. Panel bên phải (bấm biểu tượng ⚙ / **Session options** nếu bị ẩn):
   - **Accelerator** → **GPU T4 x2**.
   - **Internet** → **On**. Nếu bị khoá, vào https://www.kaggle.com/settings xác minh số điện thoại trước.
   - **Input** → **+ Add Input** → tab **Your Work / Datasets** → chọn `lifemate-data` → dấu **+**.
4. Đổi tên notebook (góc trên trái), vd `lifemate-pipeline`.

Không cần sửa đường dẫn: notebook tự tìm thư mục chứa `<split>/<split>.jsonl` trong `/kaggle/input`.

## Bước 3. Chạy thử trên train (khoảng 30 sample)

1. Ô **1. Config** để mặc định: `SPLIT = "train"`, `LIMIT = 30`.
2. **Run All** (hoặc Shift+Enter từng ô).
3. Đọc output:
   - ô 3: `Đã xong 0/30, còn 30 sample`.
   - ô 5: transcript ASR vs `audio_text` gốc của 5 sample đầu; cảnh báo nếu hotword bị tắt.
   - ô 10: tiến độ từng chunk.
   - ô 12: **điểm cục bộ** theo act / ask / oos và các sample sai (ASR, text gốc, call dự đoán, call đúng).
4. Muốn tách lỗi LLM khỏi lỗi ASR: đặt `TRANSCRIPT_SRC = "gold"` (dùng text gốc thay ASR), đổi `RUN_TAG` (vd `"v1-text"`), chạy lại. Muốn dùng transcript của notebook `zalo-ai-asr`: gắn output của nó làm Input và đặt `TRANSCRIPT_SRC = "asr_<model>.json"`.

## Bước 4. Chạy public_test và lấy file nộp

1. Ô Config: `SPLIT = "public_test"`, `LIMIT = None`.
2. Góc trên phải → **Save Version** → chọn **Save & Run All (Commit)** → **Save**. Notebook chạy nền trên máy Kaggle, có thể đóng tab.
3. Theo dõi: trang notebook → **Versions** (hoặc biểu tượng đồng hồ) → bấm version đang chạy → xem log.
4. Khi xong, mở version → tab **Output**:
   - Có `result/submission.jsonl` → tải về → nộp tại **challenge.zalo.ai → Portal → LifeMate → Submission** (tối đa 5 lần/ngày).
   - Log có dòng `CHƯA ĐỦ: còn N sample` → làm Bước 5.

## Bước 5. Chạy tiếp từ checkpoint (khi một phiên 12 giờ không đủ)

Notebook ghi kết quả vào `/kaggle/working/checkpoint/` sau mỗi chunk 64 sample, và sau 11 giờ (`TIME_BUDGET_H`) thì không nhận chunk mới để kịp lưu output trước giới hạn 12 giờ. Checkpoint chỉ chứa sample đã chạy xong; sample đang dở khi dừng sẽ chạy lại từ đầu ở lần sau.

1. Mở lại notebook (**Edit**).
2. Panel phải → **Input** → **+ Add Input** → tab **Your Work** → lọc **Notebooks** → chọn **chính notebook này** (output của version vừa chạy) → **+**.
   Cách khác: tab Output của version cũ → **New Dataset** để tạo dataset từ output, rồi Add Input dataset đó.
3. Giữ nguyên Config (cùng `SPLIT`, cùng `RUN_TAG`).
4. **Save Version → Save & Run All** lại. Ô 3 sẽ in `checkpoint /kaggle/input/...: +N sample` và `Đã xong N/634`; notebook chỉ chạy phần còn thiếu.
5. Output của lần mới chứa toàn bộ kết quả cộng dồn. Nếu vẫn chưa đủ, lặp lại bước 2–4 với output của version mới nhất (bỏ Input cũ để khỏi nhầm, dù gắn cả hai cũng không sao: kết quả được gộp theo `id`).

Đổi prompt, model hoặc cách chạy thì đổi `RUN_TAG` (vd `"v2"`): checkpoint mang tag cũ bị bỏ qua, mọi sample chạy lại.

## Lỗi hay gặp

| Lỗi | Cách xử lý |
|---|---|
| `Không thấy public_test/public_test.jsonl dưới /kaggle/input` | Chưa Add Input dataset, hoặc dataset còn đang Processing |
| `PyTorch has CUDA version 13.0 whereas TorchAudio has CUDA version 12.8` | Notebook cũ cài vllm mới nhất. Import lại notebook bản mới (ghim `vllm==0.10.2`, `transformers==4.56.2`), **Restart session** rồi Run All |
| `torch cũ vẫn đang nạp` ở ô cài đặt | Run → **Restart session**, chạy lại từ ô cài đặt |
| Ô 9 lỗi ở `LLM(...)` hoặc smoke test, log có `attention` / `triton` | Thêm ô ngay trước mục 9: `import os; os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"`, Restart session, Run All |
| `No operator found for memory_efficient_attention_forward` ... `capability (7, 5) (too old)` | Notebook cũ chọn xformers, không chạy được trên T4. Import notebook bản mới (Triton) |
| `pip install vllm` lỗi phiên bản torch/CUDA khác | Restart session rồi chạy lại ô pip |
| CUDA out of memory khi tạo `LLM(...)` (vd `warming up sampler with 256 dummy requests`) | Notebook đã đặt `max_num_seqs = CHUNK_SIZE` (64) và `gpu_memory_utilization = 0.85`. Nếu vẫn OOM: giảm `CHUNK_SIZE` xuống 32 |
| `Cannot re-initialize CUDA in forked subprocess` | Restart session, chạy lại từ đầu |
| Validator ở ô 11 báo lỗi schema | Xem `checkpoint/<split>_trace.jsonl` của sample đó |
| Hết quota GPU (30 giờ/tuần) | Chạy thử với `LIMIT` nhỏ; chỉ chạy full khi cần nộp |

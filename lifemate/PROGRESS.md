# LifeMate: tiến độ và thông tin quan trọng

Cập nhật lần cuối: 2026-11-01 ~00:45 (GMT+7). File này là nguồn chính để nối lại công việc sau khi compact.
Tài liệu liên quan: `KAGGLE.md` (cách upload/chạy tay trên Kaggle), `paper-review/lifemate/00-rules.md` (luật thi tóm tắt).

## 1. Mục tiêu và luật thi

- Zalo AI Challenge 2026, track LifeMate: audio tiếng Việt + lịch sử hội thoại -> chuỗi tool call theo thứ tự + `state_hash` (qua `zaic_tools`).
- Chấm: hash sai = 0 điểm sample; hash đúng = F1 trên các call. Tham số fuzzy so token-F1 > 0.6. Thừa tham số = call không khớp.
- **Mục tiêu của người dùng: > 80 điểm.**
- Ràng buộc: tổng < 6B tham số, LLM < 5B, context <= 16K, <= 30 s/sample trên 1×A30, offline hoàn toàn, model OSI license, phát hành trước 2026-10-26.
- Deadline (GMT+7): nộp Docker **16/11/2026 23:59** (tự đặt mốc đóng gói 13/11); private test 20/11 09:00; kết quả private 23/11 23:59.
- Dữ liệu v3.2.0: train 1545 / public_test 634. Public test có 34/122 tool không xuất hiện trong train (nên classifier nhãn cố định không dùng được).

## 2. Quy tắc làm việc với người dùng (bắt buộc)

- Luôn trả lời **tiếng Việt**.
- Không tải model/trọng số về Mac; chỉ dataset ở local. Mọi thứ chạy trên Kaggle (T4×2). Fetch output luôn loại `*.safetensors`.
- Dataset Kaggle `jasonhoang666/lifemate-data`, `jasonhoang666/lifemate-sft` phải **Private**.
- Không đọc/in token Kaggle. Khi CLI báo `Permission 'kernels.get' was denied` = OAuth hết hạn: nhờ người dùng chạy `! .venv/bin/kaggle auth login` (không tự làm bước đồng ý OAuth).
- Quota GPU 30 h/tuần: nói rõ chạy gì + ước tính thời gian trước khi push. Người dùng (2026-10-31) đã cho phép: "làm cho ra kết quả, hết thì cứ báo".
- Được chạy 2 kernel song song; không poll liên tục (dùng background task, chờ thông báo).
- "Research kĩ trước khi fix để đảm bảo chạy đúng" - smoke test local trước khi push nếu có thể.
- Sửa file bằng Edit/Write, không sed -i/heredoc. Comment code: không em-dash, chỉ comment WHY.

## 3. Hạ tầng Kaggle

| `--nb` | Kernel | Thư mục push | Nguồn |
|---|---|---|---|
| pipeline | `jasonhoang666/zalo-ai` | `kaggle/` | `notebooks/lifemate_pipeline.py` |
| pipeline2 | `jasonhoang666/zalo-ai-b` | `kaggle_b/` | `notebooks/lifemate_pipeline.py` |
| asr | `jasonhoang666/zalo-ai-asr` | `kaggle_asr/` | `notebooks/asr_bench.py` |
| lora | `jasonhoang666/zalo-ai-asr-lora` | `kaggle_asr_lora/` | `notebooks/asr_lora.py` |
| sft | `jasonhoang666/zalo-ai-sft` | `kaggle_sft/` | `notebooks/llm_sft.py` |

- Lệnh: `.venv/bin/python -u scripts/kaggle_run.py [--nb X] [--source owner/kernel ...] --set KEY=VALUE ... 'RUN_TAG="tag"'` -> push, chờ, tải về `output/<tag>/`. Chạy bằng `run_in_background`.
- Kaggle cho tối đa **2 phiên GPU cùng lúc**. Không chạy 2 version cùng một kernel (status/output bám version mới nhất); muốn fetch version cũ: `kaggle_run.fetch(kernel, tag, "<version>")`.
- Fetch tay: `.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); import kaggle_run as k; k.fetch('jasonhoang666/zalo-ai','tag','14')"`.
- Mac ngủ/mất mạng làm tiến trình theo dõi chết; kernel vẫn chạy tiếp: kiểm `kaggle kernels status` rồi fetch tay.
- Môi trường local: `bash scripts/setup_env.sh` tạo `.venv` (Python 3.12, kaggle + six + numpy) và kiểm tra đăng nhập. `--fetch-only` nhận tag trần (`--fetch-only pub-x`); viết `'RUN_TAG="pub-x"'` sẽ tạo thư mục tên lạ.

## 4. Kiến trúc hiện tại

- **ASR:** PhoWhisper-large + LoRA r32 (kernel `zalo-ai-asr-lora`, file `asr_phowhisper-lora.json`), WER dev 0.101, 0.50 s/lượt.
- **LLM:** Qwen3-4B-Instruct-2507 trên vLLM 0.10.2, TP=2, fp16, `TRITON_ATTN_VLLM_V1` (T4). Ghim torch 2.8 cu128, transformers 4.56.2, huggingface_hub<1.0.
- Tổng ≈ 1.55B + 4.02B = 5.57B (< 6B, sát).
- Agent loop trong `lifemate_pipeline.py` mục 10 (`agent_step`): parser `<tool_call>`, check_call, `_unseen_id` (chặn ID bịa), ask guard (search 1 kết quả mà vẫn hỏi), search-hint khi không tìm thấy, prose nudge, date_hints, `READ_BARRIER` (mới).
- CFG chính: `SPLIT` (train|public_test|private_test), `SUBSET` ("dev" = 200 sample cố định), `LIMIT`, `TRANSCRIPT_SRC` ("gold" | "asr" | "asr_<model>.json"), `LLM`, `FEWSHOT_K` (3; model SFT dùng 0), `MAX_STEPS` 10, `READ_BARRIER` True, `JEV_PROBE` False.
- **Dev set cố định:** 200 sample train (act 158 / ask 28 / oos 14) chọn theo md5(id) trong từng kind, 730 lượt audio; loại khỏi few-shot, thống kê tham số và dữ liệu SFT.
- Model SFT: `--source jasonhoang666/zalo-ai-sft --set 'LLM="input:qwen3-4b-sft"' FEWSHOT_K=0`. ASR LoRA: thêm source `jasonhoang666/zalo-ai-asr-lora` + `'TRANSCRIPT_SRC="asr_phowhisper-lora.json"'`.

## 5. Lịch sử điểm (dev 200 trừ khi ghi khác)

| Run | Text | Điểm | Ghi chú |
|---|---|---|---|
| v2 / v3 / v4 | ASR, 30-60 sample | 40.8 / 47.8 / 46.6 | prompt tweaks, plateau |
| v5-text | gold, 60 sample | 67.1 | |
| v6-gold | gold | 49.7 | 83 sai hash |
| **v7-gold** | gold | **56.41** | hash_ok 133; act 53.8 / ask 60.1 / oos 78.6. Baseline prompt tốt nhất |
| **v7-lora** | ASR LoRA | **44.00** | hash_ok 114; act 39.7 / ask 54.5 / oos 71.4 |
| q35-gold40 | gold, 40 sample | 50.05 | Qwen3.5-4B, kém hơn Qwen3-4B (73.87 trên cùng 40 id) -> bỏ |
| sft1-gold | gold | 49.86 | SFT không barrier; act 47.9 / ask 46.5 / oos 78.6 |
| sft1-lora | ASR LoRA | 41.04 | hash_ok 103 |
| **sft1-gold-rb** | gold | **57.90** | barrier chặn cả write+ask; act **61.9** / ask 25.0 / oos 78.6; 145 phút |
| sft1-gold-rb2 | gold | 54.02 | barrier chỉ chặn write; act 54.0 / ask 41.7 / oos 78.6; hash_ok 125; 1.24 h agent. Cho ask qua làm act -8 (27 sample act hỏi ngay sau list/get) |
| sft1-lora-rb2 | ASR LoRA | 42.18 | act 40.1 / ask 35.7 / oos 78.6; hash_ok 104 |
| **sft1-lora-rb3** | ASR LoRA | **47.88** | act 47.5 / ask 34.5 / oos 78.6; hash_ok 118. TỐT NHẤT với ASR |
| **sft1-gold-rb4** | gold | **60.38** | rb3 + bỏ ask sau >=2 tool đọc + ask giờ cho reminders.create; act 60.6 / ask 50.0 / oos 78.6; hash_ok 138; 3 lên / 0 xuống. Model từ dataset `qwen3-4b-sft1` |
| sft1-lora-rb4 | ASR LoRA | 47.88 | act 46.9 / ask 38.1; 1 lên (0360) / 1 xuống (1479, nhiễu sinh) |
| (rb5, chưa chạy) | | ~61.4 / ~48.9 | + `_apply_pending_ask`: ask bị hoãn do nhiều tool đọc được nhớ, lượt sau model gọi đúng tool đó thì thay bằng ask. Replay rb4: chỉ đổi 0207, 0404 (0 -> khớp gold) ở cả gold và lora. Gộp vào run sau, không chạy riêng để tiết kiệm quota |
| sft2-gold | gold | 54.29 | xem mục sft2 |
| sft2-lora | ASR LoRA | 40.06 | act 37.4 / ask 32.1 / oos 85.7; hash_ok 104; 16 lên / 33 xuống so với sft1-lora-rb3. Lỗi chính: trả DONE ngay bước đầu ở 24 sample có gold (sft1: 5). Dữ liệu v2 có cùng phân bố lượt đầu với v1 -> nghi do train ít hơn (83 bước, 1 epoch so với 120 bước) |
| **sft1-gold-rb3** | gold | **59.32** | ask sau tool đọc bị hoãn, chỉ giữ nếu tool đọc cuối là `.search` trả >1 kết quả (`_keep_deferred_ask`); + bỏ qua read trùng. act 60.5 / ask 42.9 / oos 78.6; hash_ok 136; 1.42 h agent. TỐT NHẤT hiện tại. Ask sai còn lại: bịa giá trị thay vì hỏi (reminders.create thiếu giờ), thấy list nhiều mục vẫn tự chọn, bỏ sót call ghi trước ask |

- Oracle max(sft1-gold-rb, v7-gold) theo từng sample = 72.3 -> hai hệ mạnh ở chỗ khác nhau (47 up / 37 down).
- ASR làm mất khoảng 9-12 điểm so với gold.

## 6. Phát hiện quan trọng

### SFT
- Dữ liệu: `scripts/make_sft.py data/lifemate <out.jsonl>` tạo 1345 hội thoại từ train trừ dev, dùng chính hàm prompt của notebook (FEWSHOT_K 0), DONE ở cuối, ask kết thúc lượt. Upload: dataset private `jasonhoang666/lifemate-sft` (file `data/sft/sft_train.jsonl` = bản CŨ).
- Train (`notebooks/llm_sft.py`): torchrun DDP 2×T4, base fp16 + LoRA fp32 r32/alpha 64 mọi lớp linear, lr 1e-4, grad_accum 8, MAX_LEN 10240, loss chỉ trên token assistant, merge ra `/kaggle/working/qwen3-4b-sft`. `TRAIN_BUDGET_H` 8.0.
- sft1: dừng vì hết budget 8 h ở bước 120/166 (~1.45 epoch), loss 1.09 -> ~0.05 (gần như phẳng từ epoch 2). ~4 phút/bước.
- **Lỗi dữ liệu đã tìm ra:** `group_steps` cũ gộp tool đọc + tool ghi vào cùng một lượt assistant (682/4572 lượt), nên model học gọi ghi trước khi thấy kết quả đọc (vd `gold_price.get_price` + `reminders.create` khi điều kiện giá chưa thoả). Trên dev: 68/200 sample bị, 42 sai hash.
  - Đã sửa `make_sft.py`: tách lượt khi call ghi/ask đứng sau tool đọc (dựa trường `kind` read/write/meta trong catalog). Bản mới (từng ở scratchpad, nay đã thành `data/sft/sft_train.jsonl` v2, xem mục sft2) (1345 hội thoại, tb 2.60 lượt). **Chưa upload, chưa train.** Có thể tạo lại bất cứ lúc nào bằng lệnh trên.
  - Đã thêm `READ_BARRIER` vào pipeline (`_read_barrier` trước `agent_step`): bỏ call ghi đi sau tool đọc trong cùng lượt. Bản đầu chặn cả ask -> ask tụt 46.5 -> 25 vì dữ liệu cũ để ask cùng lượt với search; model thấy kết quả rồi tự chọn một mục. Bản hiện tại chỉ chặn `kind == "write"`.
- Catalog: train 84 write / 62 read / 1 meta; public_test 69 / 52 / 1. ASK = `sys.ask_missing_info` (meta).

### Lỗi còn lại của v7-gold (67 sai hash)
mixed 20, gold hỏi/pred làm 9, pred hỏi/gold làm 8, sai tham số 8, thừa call 8, dừng sớm 6. Prompt patch đã chững -> SFT là hướng chính.

### ASR
- asr1: VietASR WER .190, PhoWhisper-large .267, medium .290, whisper-v3-turbo .359. asr2: VietASR pad0.5 + beam4 = .179.
- lora1: PhoWhisper-large + LoRA r32, 2 epoch trên 4306 lượt non-dev -> **WER .101**, CER .061 (thắng rõ).
- asr3 (zero-shot): Qwen3-ASR-1.7B WER .284 (viết số bằng chữ, 0.39 s/lượt); Gemma 4 E2B fp16 .456 / fp32 .406 (nghe sai động từ, 1.6 s/lượt). -> **Bỏ hướng audio end-to-end**, giữ PhoWhisper-LoRA.
- WER_nd (chỉ lượt không có chữ số, tính local): gần bằng WER cho mọi model (LoRA .103, VietASR-beam4 .168, Qwen3-ASR .276) -> cách viết số không phải lý do Qwen3-ASR kém.
- **asr4 kết quả (dev, WER / WER_nd / s mỗi lượt):** chunkformer-rnnt .206 / .192 / 0.087 (arg_recall .806); chunkformer-ctc .219 / .206 / 0.044; phoasr-whisper-small .278 / .265. Đều kém xa PhoWhisper-LoRA .101 -> giữ LoRA; ChunkFormer là dự phòng nếu cần nhanh (phải fine-tune). Output `output/asr4/`.
- Parakeet lỗi: `from_pretrained` tìm `model_config.yaml` không có trong repo HF. Sửa nếu thử lại: `hf_hub_download` file `.nemo` rồi `ASRModel.restore_from(path)`. Chưa chạy lại (chờ người dùng quyết; khả năng thắng LoRA thấp).
- asr4 danh sách model: chunkformer-rnnt-large-vie (113M, CC BY 4.0, 10/2025), chunkformer-ctc-large-vie (110M, CC BY-NC), PhoASR-whisper-small (Qualcomm, 0.2B, BSD-3-Clear + Qualcomm RAI), nvidia/parakeet-ctc-0.6b-Vietnamese (600M, NVIDIA Open Model License, nemo cài sau cùng). Bỏ VieGem-ASR (CC BY-NC-SA + phải chấp nhận Gemma terms để tải encoder). Người dùng: "làm tốt nhất, license tính sau" -> chọn theo chất lượng, ghi license để hỏi BTC sau.
- ~10/41 sample mất điểm do ASR có text gần như giống hệt (chỉ khác hoa/thường, dấu câu) = LLM không ổn định; còn lại là nghe sai thật (tên, từ hiếm, giọng địa phương).

### Jev (ý tưởng người dùng, đã thử)
- Jev thật (TypeSafe) là API đóng, không dùng offline được. FastJev/clone = chấm lựa chọn bằng logit của causal LM.
- Probe (cell 9b, `JEV_PROBE=True`, teacher-forced trên gold, base Qwen3-4B, 642 bước): top1 .841, top3 .935; bước đầu .790; DONE .919; ask chỉ .607; đúng mọi bước 125/200 (v7 116). Độ tin cậy vô dụng (sai vẫn p≈0.999). 4.3 s/bước nhưng ~14 s/sample, nhanh hơn generate ~2-3 lần ở khâu chọn tool; chi phí chính là prefill prompt dài, không phải decode.
- Kết luận: chưa dùng. Có thể probe lại với model SFT (cần AUROC > 0.85 mới đáng). Nếu làm thật: 1 request constrained decoding + logprobs, tái dùng prefix cache.

### Model khác đã loại
Qwen3.5-4B (kém hơn), Gemma 4 E4B (~8B, vượt), Voxtral/Phi-4-mm (không có tiếng Việt), Qwen2.5-Omni (không OSI), thinking models (quá chậm cho 30 s).

## 7. Đang chạy (2026-11-01 ~00:40)

- `sft1-gold-rb2` trên `zalo-ai` v14 (task nền bhisoslyn) -> `output/sft1-gold-rb2/`. Kỳ vọng ~61 (act ~62 + ask ~46).
- `sft1-lora-rb2` trên `zalo-ai-b` v7 (task nền b100bamjk) -> `output/sft1-lora-rb2/`. So với 41.0 / 44.0.
- Hai kernel nằm hàng đợi GPU ~7.5 h, bắt đầu chạy thật 08:18 VN (lastRunTime 01:18 UTC; xem bằng `kaggle kernels list --mine --search zalo`). Kaggle không cho tải log khi kernel đang chạy.
- Script kiểm tra local (stub vLLM + ToolEnv thật, phát lại llm_outputs đã ghi): `scripts/dev/barrier_smoke.py <ids>` (sim_ask.py ở scratchpad cũ đã mất). Chạy bằng `.venv/bin/python`. `zaic_tools` có sẵn local trong `data/lifemate/train/`.
- Lưu ý đăng nhập Kaggle: phải là tài khoản **jasonhoang666** (không phải jasonhoang6201).
- Sau khi push rb2: `_read_barrier(calls, done_keys)` bỏ qua tool đọc đã chạy ở lượt trước (trong sft1-gold-rb có 2/200 sample bị mất call ghi vì lỗi này). Bản sửa chưa chạy trên Kaggle.
- Nếu task nền chết: `kaggle kernels status jasonhoang666/zalo-ai` rồi fetch tay version 14 (zalo-ai) / 7 (zalo-ai-b).

Script phân tích nhanh (so 2 run theo sample):
```python
import json
def L(p): return {r['id']: r for r in map(json.loads, open(p))}
a = L('output/<run>/result/train_eval.jsonl')   # fields: id, kind, score, hash_ok, pred, gold
```
Trace LLM: `output/<run>/checkpoint/train_trace.jsonl` (id, llm_outputs).

### sft2 (2026-11-01)
- Dữ liệu v2 (tách lượt sau tool đọc) đã ghi đè `data/sft/sft_train.jsonl` và upload thành version mới của `lifemate-sft` (private). Bản v1 lưu ở `output/sft-data-v1/`.
- Lệnh push (tự chạy khi slot trống, task nền bto58twgw, `--no-wait`): `kaggle_run.py --nb sft --no-wait --set EPOCHS=1 TRAIN_BUDGET_H=8.0 'RUN_TAG="sft2"'`. Ước tính ~5.5-6 h. Output ghi đè model `qwen3-4b-sft` trong kernel zalo-ai-sft (pipeline `--source jasonhoang666/zalo-ai-sft` sẽ lấy sft2). Sau đó đánh giá với cấu hình rb3 trên gold + lora.
- Quota: reset thứ Bảy; 2026-11-01 11:25 còn 23h57.
- **sft2 xong** (zalo-ai-sft v4): 83 bước, 1 epoch, 340 phút train; loss cuối ~0.07-0.11. Output `output/sft2/` (không có trọng số).
- **sft2-gold 54.29** (act 53.8 / ask 41.1 / oos 85.7; hash_ok 124; DONE ngay bước đầu sai 11 so với 2 của sft1). sft2 KÉM sft1 -> giữ sft1. Oracle max(sft1, sft2) gold = 65.68.
- **Model sft1 tách riêng:** output mới nhất của zalo-ai-sft giờ là sft2, nên `--source jasonhoang666/zalo-ai-sft` KHÔNG còn là sft1. Đã tải sft1 (zalo-ai-sft v3, 7.5 GB, người dùng cho phép) về `models/qwen3-4b-sft1/` và upload dataset private `jasonhoang666/qwen3-4b-sft1`. Dùng: `--dataset jasonhoang666/qwen3-4b-sft1 --set 'LLM="input:qwen3-4b-sft1"'` (bỏ `--source jasonhoang666/zalo-ai-sft`). `kaggle_run.py` có thêm `--dataset`.
- **rb4 (pipeline, 2026-11-01 ~22:40):** (1) `_keep_deferred_ask(turn_calls, res)`: lượt có >= 2 tool đọc rồi ask thì bỏ ask (rb3: 6/6 sample gold không hỏi ngay); (2) `_ask_reminder_time`: `reminders.create` khi câu không có từ chỉ thời gian (`_TIME_CUE`) -> ask time (train 6/6, 0 báo nhầm). Replay rb3: đổi 5 (gold) / 4 (lora) sample, tất cả đang 0 điểm. Push tự động sau khi dataset sft1 sẵn sàng (task nền buj14yb3j, log `output/push_rb4_*.log`): `sft1-gold-rb4` (zalo-ai), `sft1-lora-rb4` (zalo-ai-b). Lệnh upload dataset báo 403 ở bước GetDatasetStatus nhưng phần tải lên đã xong; Kaggle cần vài phút xử lý.
- Đánh giá 20:20 VN: `sft2-gold` (zalo-ai, task nền bld6ob2tf), `sft2-lora` (zalo-ai-b `--nb pipeline2`, source zalo-ai-sft + zalo-ai-asr-lora, task nền b9popeytg). ETA ~22:00-22:30 nếu không phải xếp hàng GPU. So với 59.32 / 47.88.

## 7b. Nghiên cứu 2026-11-02 (người dùng: dừng train, phân tích trước)

- **Leaderboard public chấm jsonl, không chấm Docker** -> đội khác có thể dùng model lớn/API ở vòng public; 90-97 điểm public chưa chắc là hệ thống < 6B offline.
- **Train vs public_test:** test có 11 app / 34 tool KHÔNG có trong train (ecommerce, grocery, stocks, parking, public_transit, train_bus_tickets, event_tickets, recipes, pantry, habit_tracker, translator); 410/634 (65%) sample test có cài app mới; history test có 306 call vào app mới. Tool dùng chung: schema giống hệt. Phân bố history giống nhau. -> Dev hiện tại (lấy từ train) KHÔNG đo khả năng tổng quát sang tool mới; SFT trên tool train có thể overfit. Ý tưởng: dev giữ lại theo app (leave-app-out) để mô phỏng test; private test có thể lại có app khác nữa.
- **Lỗi rb4-gold (mất 39.6 điểm):** đúng tool sai tham số làm sai hash 9.0 (điều kiện nhiều vế "trừ phòng đang bật điều hoà", tính phút 14:30-15:15, "sớm hơn 1 tiếng", chọn rẻ nhất); gold hỏi mà pred làm 6.0; thiếu call 5.7; thừa call 5.1; pred hỏi mà gold làm 4.0; số lần lặp call sai 3.9; tham số tuỳ chọn/fuzzy 3.5 (gold có `title`, `near`... mà pred bỏ: xung đột với luật 4 của prompt "chỉ điền tham số bắt buộc").
- **Audio:** train và test cùng phân bố (16 kHz, ~4 s, SNR-proxy trung vị ~39 dB, không clip). WER LoRA theo SNR: phần tư nhiễu nhất .124 vs .087-.093 còn lại -> enhance giọng lợi tối đa ~.008 WER; ưu tiên thấp.
- **ASR gây mất 15.0 điểm (41 sample), 7 sample ngược lại +2.5:** gần như toàn bộ là nghe sai thật: tên riêng (Lan->Lâm, Hồ Gươm->Mòn Gốm, Sơn Tùng), từ tiếng Anh (SMS->"ít đêm ít", logistics), cách viết số ("50 nghìn" vs "năm mươi ngàn"). Hướng: biasing tên từ kết quả tool (2-pass, chỉ dùng dữ liệu agent đã đọc qua tool), N-best, train LoRA thêm (cả dev khi nộp cuối), TTS augmentation cho tên/tiếng Anh.
- **Web:** không có lời giải LifeMate công khai. Papers tổng quát hoá tool mới: GenTool (ACL Findings 2025), Trajectory2Task (arXiv 2601.20144), GenesisFunc (ACL 2026), ToolVerifier: đều dùng dữ liệu tổng hợp với nhiều bộ tool đa dạng.
- **Chạy full public_test (yêu cầu người dùng 2026-11-02):** bước 1 `--nb lora_infer --resume` (notebook mới `notebooks/asr_lora_infer.py`, kernel zalo-ai-asr-lora v2): chép public_test bằng adapter LoRA, gộp vào `asr_phowhisper-lora.json` (giữ dev). Bước 2: pipeline `SPLIT="public_test"`, sft1 dataset, rb5, RUN_TAG `pub-sft1-rb5` -> `output/pub-sft1-rb5/result/submission.jsonl`. Task nền bcees1hxu, log `output/push_asr_pub.log`, `output/push_pub.log`. `kaggle_run.py` giờ đặt `code_file` theo notebook nguồn.
  - **XONG 13:31 (zalo-ai v18, 4.04 h):** `output/pub-sft1-rb5/result/submission.jsonl`, 634 dòng, 0 lỗi schema, act 512 / oos 67 / ask 55. Người dùng tự nộp portal để lấy điểm public gốc. Bản này CHƯA có xref-prereq và luật thứ tự bên dưới.
- **Thứ tự call và hash:** `state_hash` băm STATE cuối (canon trong `zaic_tools/utils.py`), không băm danh sách call; F1 call không xét thứ tự. Nhưng list-of-dict giữ thứ tự -> nhiều call ghi vào cùng list mà đổi thứ tự thì sai hash. Dev chỉ 2/200 sai vì thứ tự (0388 theo thứ tự người dùng nói, 1049 rẻ nhất trước) ~1 điểm. History public: đọc trước ghi trong cùng app (`parking.find`>`reserve`, `grocery.search`>`add_to_cart`), xen kẽ search A>add A>search B>add B.
- **Đã sửa, chưa chạy:** (1) `_xref_prereq(TOOL_SPEC)` suy quy trình đọc->ghi từ `x-ref` trong catalog (khớp gold train 97%, phủ 43 tool ghi public kể cả app mới), gộp vào `PREREQ` khi `PREREQ_HINT`; (2) luật 3 SYSTEM_RULES thêm câu thứ tự call ghi (theo lời người dùng / theo tiêu chí). Cần chạy dev (gold + lora) trước khi chạy public lần 2.
- **rb6 = rb5 + (1) + (2) + luật 4 sửa + luật 4b** (từ 28 sample sai tham số: tham số tuỳ chọn người dùng đã nói như seat_class/delay_minutes phải điền; tính giờ "chín giờ tối"=21:00, "sớm hơn N tiếng", số phút = kết thúc - bắt đầu, "xong thì nhắc" = giờ kết thúc). Replay 200 sample không lỗi. **Đang chạy:** `sft1-gold-rb6`, zalo-ai v19, push 13:55, ETA ~16:20 GMT+7, task nền b3rm6k062, log `output/push_rb6_gold.log`. So với rb4-gold 60.38.
- **Quota (14:10 2026-11-02): còn ~5 h.** Lỗi Kaggle: session bắt đầu khi còn quota thì chạy được đủ 12 h. Kế hoạch: rb6 dev (~2.4 h) + asr-prompt (~0.6 h) là test cuối; sau đó MỘT run public cuối (rb6 hoặc rb5 tuỳ dev, transcript prompt nếu WER tốt hơn), push khi còn quota.
- **Verifier (ý người dùng 2026-11-02: handoff, mỗi agent một việc, cùng model nên không vượt 6B):** `verify()` chạy sau vòng agent mỗi chunk (`CFG["VERIFY"]`). Prompt `VERIFY_PROMPT`: yêu cầu + lịch sử (kể cả call/kết quả tool trong history, 14 dòng cuối) + tool đọc lượt này + call ghi đánh số; trả JSON `{"ly_do","bo":[...]}`, CHỈ được bỏ call ghi (loại trừ "trừ X", điều kiện sai, đã huỷ, trùng/không yêu cầu, đã đúng trạng thái). Bỏ thì replay call còn lại trên ToolEnv mới để lấy hash. Row lưu `base_calls/base_hash`; ghi `submission.jsonl` (đã verify) + `submission_base.jsonl`; chấm dev in cả hai điểm và số sample lên/xuống. Smoke `scripts/dev/verify_smoke.py` (REPLY=... giả lời verifier): replay đúng 151/156, không lỗi.
- **rb6-gold = 60.64** (rb4 60.38; act 59.4, ask 58.9, oos 78.6; 13 lên / 16 xuống). Xuống do câu "PHẢI điền tham số tuỳ chọn" ở luật 4 (tool đọc tự thêm period/type/currency, update_event thêm end, thêm get_quotes) và xref gợi ý cả tool train mà thống kê đã lọc. Sửa (rb7): bỏ câu đó (giữ 4b tính giờ, cứu 1244/1432), xref chỉ cho tool train chưa từng gọi (trên train gợi ý giống rb4).
- **XONG (run cuối, tải về 2026-11-02 20:16):** public `pub-sft1-rb7v` = rb7 + từ địa phương + verifier, zalo-ai v20. File nộp: `output/pub-sft1-rb7v/result/submission_base.jsonl` (634 dòng, schema id/tool_calls/state_hash OK). `submission.jsonl` là bản verifier (đổi 293 sample), KHÔNG nộp. Base trùng hash với rb5 (44.88) 555/634. Điểm public: chờ người dùng nộp. Dev `sft1-lora-rb6v` (zalo-ai-b v11) chạy prompt rb6 (bản lỗi) + verifier, không huỷ được bằng CLI; vẫn dùng để so verify vs base trong cùng run -> chọn file nộp.
- **KẾT QUẢ `sft1-lora-rb6v` (dev ASR, 18:37):** base (prompt rb6, ASR thật) **49.47** (rb4-lora 47.88, +1.6). Có verifier **30.27**: đổi 100/200 sample, lên 7 / xuống 50. Model SFT gần như luôn trả `"bo": [1]` bất kể câu, lý do bịa ("đèn đang tắt nên không cần bật") -> verifier bằng sft1 HỎNG, tắt `VERIFY` cho run sau. **Nộp `submission_base.jsonl` của run public cuối.** Bài học: model SFT kém ở vai mới; vai phụ (verify, ASR_FIX) nên dùng model gốc Qwen3-4B qua vLLM LoRA (base cho vai phụ, adapter cho agent, cùng ~4B nên vẫn < 6B) - cần adapter sft1 chưa merge. Adapter trong output zalo-ai-sft hiện là của sft2 (sft1 chỉ còn bản merge, dataset `qwen3-4b-sft1`). Cách lấy lại: delta = W_sft1 - W_base từng lớp linear có hạng <= 32 (LoRA r32), SVD rank 32 -> adapter PEFT gần như chính xác; rồi vLLM `enable_lora` với base Qwen3-4B-Instruct-2507.
- **v2 cho cả hai vai (ý người dùng: model nhỏ cần khuôn chặt, ca mẫu cụ thể):** model chỉ điền khuôn, CODE quyết định. Verifier v2: mỗi call ghi điền `nguoi_dung_noi / loai_tru / dieu_kien / dieu_kien_dung` (trích nguyên văn) + 5 ca mẫu; `_parse_drop` chỉ bỏ khi cụm trích có nguyên văn trong câu người dùng và chứa từ loại trừ (`_EXCLUDE_CUE`: trừ/khỏi/thôi/đừng/không cần/bỏ qua) hoặc điều kiện (`_COND_CUE`: nếu/hễ/lỡ/trường hợp) bị đánh giá false. Offline trên rb6v: tầng câu đã chặn hẳn 31/50 sample bị hại (còn tối đa 8.5 điểm rủi ro, trước 21.7), giữ 3/7 sample được lợi. ASR_FIX v2: model trả `{"viec","sua":[[cũ,mới]]}` theo B1-B3 + 6 ca mẫu (giặt->dịch, cất giữ nguyên, Lâm->Lan, vay->tàu, ít đêm ít->SMS); `apply_fixes` chỉ thay khi từ cũ có trong câu, mọi chữ từ mới nằm trong `_fix_vocab` (tên đã biết + tên/mô tả app/tool đã cài), không đụng số (`_NUMWORDS`), giữ số âm tiết (ngoại lệ: từ viết tắt ASCII hoa khi từ cũ >= 2 âm tiết), tối đa 3 cặp. Cả hai vẫn tắt mặc định; nên chạy bằng model gốc (xem trên).
- **ASR_FIX (ý người dùng 2026-11-02, CHƯA chạy GPU, cờ mặc định False):** vai sửa lời nghe trước agent chính, cùng model. Prompt `ASR_FIX_PROMPT`: app đã cài (name_vi + mô tả), tên đã biết (`_known_names`: chuỗi ngắn viết hoa trong user_profile + tool_calls/tool_results của history), 3 câu trợ lý gần nhất; chỉ sửa khi âm gần giống VÀ câu gốc vô lý / gần một tên đã biết ("giặt câu qua tiếng Anh" -> "dịch"; "cất sách" giữ). Trả `{"cau","sua"}`; sửa > 30% số từ thì bỏ. Agent nhận câu đã sửa + câu ASR gốc; row lưu `asr_raw`, trace lưu `asr_fix_out`. Ước lượng trên rb4-lora: 41 sample mất 15.05 điểm vì ASR, 24 sample (8.2 điểm) có từ sai xuất hiện trong context/history, 8 (2.9) trong mô tả app/tool, 8 (3.6) không có nguồn. Test cần: dev lora `ASR_FIX=True` (so rb7 lora) khi có quota.
- **Run cuối (kế hoạch cũ):** `sft1-lora-rb6v` dev ASR + verifier (zalo-ai-b v11, push 15:52, task be9rzm0km, log `output/push_rb6v_lora.log`, ETA ~18:30) so với rb4-lora 47.88; khi rb6 gold xong (~16:20) push public `pub-sft1-rb6v` trên zalo-ai -> 2 file submission, dev quyết định nộp bản nào.
- **ĐIỂM PUBLIC rb5 = 44.88** (người dùng nộp 2026-11-02; dev-lora rb4 47.88). Mục tiêu tiếp: > 50.
- **ASR public kém hơn dev:** OOV so với từ vựng train 3.6% (public) vs 2.5% (dev); lỗi rơi vào từ khoá app mới ("Dịch" -> "Giặt", "tàu" -> "vay", "dời lịch" -> "vội lịch"), LoRA chỉ học vocab train. 41 sample public có app mới bị trả OOS, nhiều câu là huỷ đúng nhưng một số do ASR (0331 vé tàu, 0455 xe buýt, 0536 công thức).
- **Thử Whisper initial prompt = tên (+ mô tả) app đã cài của sample** (`notebooks/asr_prompt.py`, `--nb asr_prompt --resume`, kernel zalo-ai-asr-lora v3, push 13:58, task bmepldodu, log `output/push_asr_prompt.log`): đo WER dev 2 kiểu prompt vs LoRA cũ; kiểu tốt nhất nếu hơn thì chép lại toàn bộ public và ghi đè `asr_phowhisper-lora.json` (dev giữ bản cũ).
  - Kết quả v3 (18 phút): WER dev lora .1012, names .1047, names_desc .1030 -> giữ transcript cũ. Phần lớn chênh do prompt toàn chữ làm Whisper viết số bằng chữ; từ vựng có chỗ tốt hơn ("quyền"->"còn", "quá thức"->"báo thức").
  - **Vùng miền (người dùng nêu):** WER dev theo vùng (từ đặc trưng trong text gốc): Bắc/khác .096, Trung .140, Nam .134. Tỉ lệ câu Trung/Nam: train 7.8/5.6%, dev 8.2/5.2%, public (ASR) 7.2/4.8% -> không phải nguyên nhân public thấp hơn dev, nhưng là điểm yếu thật. Text gốc train đã chuẩn hoá ("sáng ni rứa" ghi "sáng nay") nên LLM không học từ địa phương -> thêm bảng từ địa phương vào luật 7 (chưa test, đi vào run cuối). Hướng sau (cần quota): LoRA ASR oversample câu vùng miền.
  - v4 (push 14:18, task b93iz6a1k, `output/asr-prompt2/`): names_desc + câu mẫu có chữ số cuối prompt, chấp nhận nếu WER dev <= lora + .003 (dev không đo được app mới) rồi chép lại public.
  - Kết quả v4: WER dev .114 (tệ hơn .101) -> BỎ hướng Whisper prompt; run cuối dùng transcript LoRA cũ (`zalo-ai-asr-lora` output vẫn là bản cũ 2663 khoá).

## 8. Việc tiếp theo (hướng tới > 80)

1. Đọc kết quả rb2 (gold + lora). Nếu tốt: giữ barrier.
2. **sft2**: upload `sft_v2.jsonl` (tách lượt sau đọc) lên `lifemate-sft` (private), cân nhắc:
   - thêm bản sao hội thoại dùng transcript ASR LoRA (nhiễu ASR thật) thay cho audio_text gốc - cần chạy ASR LoRA trên train non-dev (hiện chỉ có transcript dev);
   - cân bằng ask/act (ask đang yếu);
   - tăng `TRAIN_BUDGET_H` hoặc giảm epoch để train hết (sft1 chỉ 1.45 epoch trong 8 h); 1 epoch có lẽ đủ.
3. Probe Jev lại với model SFT (tuỳ chọn).
4. Đo độ trễ từng sample (không batch) so với 30 s/sample trên A30; rb chạy 145 phút/200 sample batch trên T4.
5. ~~Chạy full public_test~~ (xong, rb5); chạy dev rb6 (xref-prereq + luật thứ tự), nếu tốt hơn thì chạy public lần 2; đóng Docker trước 13/11.
6. Ý tưởng treo: sửa tên contact ASR nghe sai bằng state env - phải hỏi người dùng/luật trước (thiết kế hiện tại agent không đọc state.json).

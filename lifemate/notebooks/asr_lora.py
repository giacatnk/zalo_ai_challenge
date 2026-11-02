# %% [markdown]
# # LifeMate: LoRA cho PhoWhisper-large
# Train LoRA trên các lượt user của train (bỏ 200 sample dev), nhãn là `audio_text` nguyên dạng viết ("50 nghìn", "SJC", "19:45") để ASR ra đúng kiểu chữ mà tham số tool cần. Chấm trên dev bằng cùng cách chuẩn hoá với notebook `zalo-ai-asr`, xuất `asr_phowhisper-lora.json` cho notebook `zalo-ai` dùng qua `TRANSCRIPT_SRC`.
#
# Mốc so sánh (asr1, dev 200, 730 lượt): vietasr WER .190, PhoWhisper-large chưa train .267.

# %%
!pip install -q "transformers==4.56.2" "huggingface_hub<1.0" "peft==0.17.1" accelerate soundfile librosa jiwer

# %%
import os, json, re, time, math, random, hashlib, unicodedata
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, torch, soundfile as sf, librosa

CFG = {
    "INPUT_ROOT": "/kaggle/input",
    "WORK_DIR": "/kaggle/working",
    "MODEL": "vinai/PhoWhisper-large",
    "DEV_N": 200,                                       # phải trùng dev_ids() trong notebook zalo-ai
    "EPOCHS": 2,
    "BATCH": 4,
    "GRAD_ACCUM": 4,
    "LR": 5e-4,
    "WARMUP": 50,
    "LORA_R": 32,
    "LORA_ALPHA": 64,
    "TRAIN_BUDGET_H": 3.0,                              # dừng train sau ngần này giờ dù chưa hết epoch
    "TRAIN_LIMIT": None,                                # số lượt train; None = tất cả (chạy thử nhanh: 200)
    "DEV_LIMIT": None,
    "EVAL_BATCH": 16,
    "SEED": 42,
}
WORK = Path(CFG["WORK_DIR"])
ASK_TOOL = "sys.ask_missing_info"
random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"]); torch.manual_seed(CFG["SEED"])
print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


def find_data_root(root, max_depth=5):
    base = root.rstrip("/").count("/")
    for cur, dirs, _ in os.walk(root):
        if os.path.isfile(os.path.join(cur, "train", "train.jsonl")):
            return Path(cur)
        if cur.count("/") - base >= max_depth:
            dirs[:] = []
    raise FileNotFoundError("Không thấy train/train.jsonl; kiểm tra đã Add Input dataset chưa")


TRAIN_DIR = find_data_root(CFG["INPUT_ROOT"]) / "train"
TRAIN = [json.loads(l) for l in open(TRAIN_DIR / "train.jsonl", encoding="utf-8") if l.strip()]


def kind_of(s):
    calls = s["label"]["tool_calls"]
    return "oos" if not calls else ("ask" if any(c["name"] == ASK_TOOL for c in calls) else "act")


def dev_ids(train_rows, n=200):
    by_kind = defaultdict(list)
    for t in train_rows:
        by_kind[kind_of(t)].append(t["id"])
    out = []
    for ids in by_kind.values():
        ids.sort(key=lambda i: hashlib.md5(i.encode()).hexdigest())
        out += ids[: round(n * len(ids) / len(train_rows))]
    return set(out)


def utterances(samples):
    out = []
    for s in samples:
        out.append({"key": s["id"], "audio": str(TRAIN_DIR / s["audio"]), "text": s["audio_text"]})
        for k, t in enumerate(s["history"]):
            if t.get("role") == "user" and t.get("audio") and t.get("audio_text"):
                out.append({"key": f"{s['id']}#{k}", "audio": str(TRAIN_DIR / t["audio"]), "text": t["audio_text"]})
    return out


DEV_IDS = dev_ids(TRAIN, CFG["DEV_N"])
TR_UTT = utterances([s for s in TRAIN if s["id"] not in DEV_IDS])
DEV_UTT = utterances([s for s in TRAIN if s["id"] in DEV_IDS])
random.shuffle(TR_UTT)
TR_UTT = TR_UTT[: CFG["TRAIN_LIMIT"]] if CFG["TRAIN_LIMIT"] else TR_UTT
DEV_UTT = DEV_UTT[: CFG["DEV_LIMIT"]] if CFG["DEV_LIMIT"] else DEV_UTT
print(f"train {len(TR_UTT)} lượt | dev {len(DEV_UTT)} lượt")

# %% [markdown]
# ## Model + LoRA
# T4 không có bf16: giữ trọng số gốc fp32 (khoảng 6 GB), chạy autocast fp16 với GradScaler, gradient checkpointing để vừa VRAM.

# %%
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model

proc = WhisperProcessor.from_pretrained(CFG["MODEL"], language="vietnamese", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(CFG["MODEL"], torch_dtype=torch.float32).cuda()
model.config.forced_decoder_ids = None
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False
# use_reentrant=False: input của encoder là mel features không cần grad; kiểu reentrant sẽ bỏ mất grad của LoRA trong encoder
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = get_peft_model(model, LoraConfig(r=CFG["LORA_R"], lora_alpha=CFG["LORA_ALPHA"], lora_dropout=0.05,
                                         target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
model.print_trainable_parameters()
START_ID = model.config.decoder_start_token_id


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    return librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav


def collate(batch):
    feats = proc.feature_extractor([load_audio(b["audio"]) for b in batch], sampling_rate=16000, return_tensors="pt").input_features
    lab = proc.tokenizer([b["text"] for b in batch], padding=True, return_tensors="pt")
    labels = lab.input_ids.masked_fill(lab.attention_mask == 0, -100)
    # tokenizer đã chèn <|startoftranscript|>; model tự thêm decoder_start khi shift labels nên bỏ token đầu
    if (labels[:, 0] == START_ID).all():
        labels = labels[:, 1:]
    return feats, labels

# %% [markdown]
# ## Train

# %%
loader = torch.utils.data.DataLoader(TR_UTT, batch_size=CFG["BATCH"], shuffle=True, collate_fn=collate, num_workers=2, drop_last=True)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=CFG["LR"], weight_decay=0.0)
total = CFG["EPOCHS"] * len(loader) // CFG["GRAD_ACCUM"]
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / CFG["WARMUP"]) * max(0.0, 1 - s / max(total, 1)))
scaler = torch.amp.GradScaler("cuda")
model.train()
t0, step, stop = time.time(), 0, False
for ep in range(CFG["EPOCHS"]):
    run = 0.0
    for i, (feats, labels) in enumerate(loader):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_features=feats.cuda(), labels=labels.cuda()).loss / CFG["GRAD_ACCUM"]
        scaler.scale(loss).backward()
        run += loss.item()
        if (i + 1) % CFG["GRAD_ACCUM"] == 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
            if step % 25 == 0:
                print(f"ep {ep} step {step}/{total} loss {run / 25:.4f} | {(time.time() - t0) / 60:.1f} phút", flush=True)
                run = 0.0
        if time.time() - t0 > CFG["TRAIN_BUDGET_H"] * 3600:
            print("hết TRAIN_BUDGET_H, dừng train"); stop = True; break
    if stop:
        break
print(f"train xong {step} bước, {(time.time() - t0) / 60:.1f} phút")
model.save_pretrained(WORK / "phowhisper_lora_adapter")

# %% [markdown]
# ## Chấm trên dev

# %%
import jiwer

model = model.merge_and_unload().half().eval()
model.config.use_cache = True
HYP = {}
t0 = time.time()
for i in range(0, len(DEV_UTT), CFG["EVAL_BATCH"]):
    batch = DEV_UTT[i: i + CFG["EVAL_BATCH"]]
    feats = proc.feature_extractor([load_audio(b["audio"]) for b in batch], sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        out = model.generate(input_features=feats.cuda().half(), language="vi", task="transcribe", max_new_tokens=160, num_beams=1)
    for b, t in zip(batch, proc.batch_decode(out, skip_special_tokens=True)):
        HYP[b["key"]] = t.strip()
sec_per_utt = (time.time() - t0) / len(DEV_UTT)
(WORK / "asr_phowhisper-lora.json").write_text(json.dumps(HYP, ensure_ascii=False, indent=1), encoding="utf-8")


def norm(t):
    # "hoà"/"hòa" là cùng một từ, chỉ khác vị trí dấu
    t = unicodedata.normalize("NFC", str(t).lower())
    for a, b in (("òa", "oà"), ("óa", "oá"), ("ỏa", "oả"), ("õa", "oã"), ("ọa", "oạ"), ("ùy", "uỳ"), ("úy", "uý"), ("ủy", "uỷ"), ("ũy", "uỹ"), ("ụy", "uỵ")):
        t = t.replace(a, b)
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


keys = [u["key"] for u in DEV_UTT if norm(u["text"])]
REF = {u["key"]: u["text"] for u in DEV_UTT}
refs, hyps = [norm(REF[k]) for k in keys], [norm(HYP.get(k, "")) for k in keys]
edge = [(r.split()[0] != h.split()[0], r.split()[-1] != h.split()[-1]) for r, h in zip(refs, hyps) if r and h]
score = {"model": "phowhisper-lora", "WER": jiwer.wer(refs, hyps), "CER": jiwer.cer(refs, hyps),
         "first_err": sum(e[0] for e in edge) / len(edge), "last_err": sum(e[1] for e in edge) / len(edge), "s/utt": sec_per_utt}
print(json.dumps(score, indent=1))
(WORK / "asr_lora_score.json").write_text(json.dumps(score, indent=1), encoding="utf-8")
for u in DEV_UTT[:10]:
    print(f"\n{u['key']}\n  REF: {u['text']}\n  HYP: {HYP[u['key']]}")

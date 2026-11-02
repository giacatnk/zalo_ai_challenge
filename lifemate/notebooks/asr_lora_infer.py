# %% [markdown]
# # LifeMate: chép lời bằng PhoWhisper-large + LoRA đã train
# Không train lại: nạp adapter từ output trước của chính kernel `zalo-ai-asr-lora` (gắn làm Input), chép mọi lượt user
# (lượt cuối + lượt trong history) của `SPLIT`, rồi gộp vào `asr_phowhisper-lora.json` cùng transcript dev cũ, để notebook
# `zalo-ai` dùng một file cho cả dev lẫn test. Adapter được chép lại vào output vì output mới thay output cũ của kernel.

# %%
!pip install -q "transformers==4.56.2" "huggingface_hub<1.0" "peft==0.17.1" accelerate soundfile librosa

# %%
import os, json, time, shutil
from pathlib import Path
import torch, soundfile as sf, librosa

CFG = {
    "INPUT_ROOT": "/kaggle/input",
    "WORK_DIR": "/kaggle/working",
    "MODEL": "vinai/PhoWhisper-large",
    "SPLIT": "public_test",
    "BATCH": 16,
}
WORK = Path(CFG["WORK_DIR"])


def find_one(name, is_dir=False):
    for cur, dirs, files in os.walk(CFG["INPUT_ROOT"]):
        if (name in dirs) if is_dir else (name in files):
            return Path(cur) / name
    raise FileNotFoundError(f"Không thấy {name} dưới {CFG['INPUT_ROOT']}")


SPLIT = CFG["SPLIT"]
SPLIT_JSONL = find_one(f"{SPLIT}.jsonl")
SPLIT_DIR = SPLIT_JSONL.parent
ADAPTER = find_one("phowhisper_lora_adapter", is_dir=True)
OLD = json.loads(find_one("asr_phowhisper-lora.json").read_text(encoding="utf-8"))
print(SPLIT_DIR, ADAPTER, len(OLD), "transcript cũ")

items = []
for line in open(SPLIT_JSONL, encoding="utf-8"):
    if not line.strip():
        continue
    s = json.loads(line)
    # khoá giống manifest của notebook zalo-ai: id cho lượt cuối, id#k với k là vị trí trong history
    items.append({"key": s["id"], "audio": str(SPLIT_DIR / s["audio"])})
    items += [{"key": f"{s['id']}#{k}", "audio": str(SPLIT_DIR / t["audio"])}
              for k, t in enumerate(s["history"]) if t.get("role") == "user" and t.get("audio")]
print(len(items), "lượt audio")

# %%
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

proc = WhisperProcessor.from_pretrained(CFG["MODEL"], language="vietnamese", task="transcribe")
base = WhisperForConditionalGeneration.from_pretrained(CFG["MODEL"], torch_dtype=torch.float32)
base.config.forced_decoder_ids = None
base.generation_config.forced_decoder_ids = None
# merge ở fp32 rồi mới half, giống notebook train (dev WER .101 đo theo cách này)
model = PeftModel.from_pretrained(base, str(ADAPTER)).merge_and_unload().half().cuda().eval()
model.config.use_cache = True


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    return librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav


HYP, t0 = {}, time.time()
for i in range(0, len(items), CFG["BATCH"]):
    batch = items[i: i + CFG["BATCH"]]
    feats = proc.feature_extractor([load_audio(b["audio"]) for b in batch], sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        out = model.generate(input_features=feats.cuda().half(), language="vi", task="transcribe", max_new_tokens=160, num_beams=1)
    for b, t in zip(batch, proc.batch_decode(out, skip_special_tokens=True)):
        HYP[b["key"]] = t.strip()
    if i % (CFG["BATCH"] * 20) == 0:
        print(i, batch[0]["key"], HYP[batch[0]["key"]], f"{time.time() - t0:.0f}s", flush=True)
print(f"{len(HYP)} lượt, {(time.time() - t0) / max(1, len(HYP)):.3f} s/lượt")

# %%
merged = {**OLD, **HYP}
(WORK / "asr_phowhisper-lora.json").write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
shutil.copytree(ADAPTER, WORK / "phowhisper_lora_adapter", dirs_exist_ok=True)
for name in ("asr_lora_score.json",):
    try:
        shutil.copy(find_one(name), WORK / name)
    except FileNotFoundError:
        pass
print(len(merged), "transcript ->", WORK / "asr_phowhisper-lora.json")

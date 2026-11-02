# %% [markdown]
# # LifeMate: PhoWhisper-large + LoRA, thêm initial prompt theo app đã cài
# Public test có 11 app không có trong train; LoRA chỉ học từ vựng train nên nghe sai từ khoá app mới ("dịch" -> "giặt",
# "tàu" -> "vay"). Thử đưa tên + mô tả các app đã cài của chính sample (lấy từ context, không dùng nhãn) làm prompt cho Whisper.
# Đo WER dev cho từng kiểu prompt, so với transcript LoRA cũ. Kiểu tốt nhất (nếu hơn bản cũ) được dùng để chép lại public_test;
# output `asr_phowhisper-lora.json` gộp như notebook lora_infer để notebook `zalo-ai` đọc được ngay.

# %%
!pip install -q "transformers==4.56.2" "huggingface_hub<1.0" "peft==0.17.1" accelerate soundfile librosa jiwer

# %%
import os, json, time, shutil, re, hashlib, unicodedata
from collections import defaultdict
from pathlib import Path
import torch, soundfile as sf, librosa, jiwer

CFG = {
    "INPUT_ROOT": "/kaggle/input",
    "WORK_DIR": "/kaggle/working",
    "MODEL": "vinai/PhoWhisper-large",
    "DEV_N": 200,
    "MODES": ["names_desc"],
    "PUBLIC": True,
    "MAX_PROMPT_TOK": 180,
    # dev chỉ có app của train nên không đo được lợi ích trên app mới; chấp nhận prompt khi WER dev chỉ ngang bản cũ
    "WER_TOL": 0.003,
}
WORK = Path(CFG["WORK_DIR"])
ASK_TOOL = "sys.ask_missing_info"


def find_one(name, is_dir=False):
    for cur, dirs, files in os.walk(CFG["INPUT_ROOT"]):
        if (name in dirs) if is_dir else (name in files):
            return Path(cur) / name
    raise FileNotFoundError(f"Không thấy {name} dưới {CFG['INPUT_ROOT']}")


def load_split(split):
    p = find_one(f"{split}.jsonl")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    cat = json.loads((p.parent / f"{split}_tools_catalog.json").read_text(encoding="utf-8"))
    return p.parent, rows, {a["id"]: a for a in cat["apps"]}


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


def utterances(s, base):
    out = [{"key": s["id"], "audio": str(base / s["audio"]), "text": s.get("audio_text")}]
    out += [{"key": f"{s['id']}#{k}", "audio": str(base / t["audio"]), "text": t.get("audio_text")}
            for k, t in enumerate(s["history"]) if t.get("role") == "user" and t.get("audio")]
    return out


def prompt_text(s, apps, mode):
    ids = list(s["context"]["installed_apps"])
    if mode == "names":
        return ", ".join(apps[a]["name_vi"] for a in ids if a in apps) + "."
    # câu mẫu có chữ số đặt cuối (sát phần cần chép): prompt toàn chữ làm Whisper viết số bằng chữ ("9 giờ" -> "chín giờ")
    return ". ".join(f"{apps[a]['name_vi']}: {apps[a]['description']}" for a in ids if a in apps) + \
        ". Nhắc tôi lúc 9 giờ sáng mai, chuyển 50 nghìn, bật 30 phần trăm, 2 vé."


TRAIN_DIR, TRAIN, TRAIN_APPS = load_split("train")
DEV = [s for s in TRAIN if s["id"] in dev_ids(TRAIN, CFG["DEV_N"])]
ADAPTER = find_one("phowhisper_lora_adapter", is_dir=True)
OLD = json.loads(find_one("asr_phowhisper-lora.json").read_text(encoding="utf-8"))
print(len(DEV), "dev sample |", len(OLD), "transcript cũ")
print(prompt_text(DEV[0], TRAIN_APPS, "names_desc")[:300])

# %%
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

proc = WhisperProcessor.from_pretrained(CFG["MODEL"], language="vietnamese", task="transcribe")
base = WhisperForConditionalGeneration.from_pretrained(CFG["MODEL"], torch_dtype=torch.float32)
base.config.forced_decoder_ids = None
base.generation_config.forced_decoder_ids = None
model = PeftModel.from_pretrained(base, str(ADAPTER)).merge_and_unload().half().cuda().eval()
model.config.use_cache = True


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    return librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav


def transcribe(samples, base_dir, apps, mode):
    hyp, t0, n = {}, time.time(), 0
    for s in samples:
        utt = utterances(s, base_dir)
        feats = proc.feature_extractor([load_audio(u["audio"]) for u in utt], sampling_rate=16000, return_tensors="pt").input_features
        ptxt = prompt_text(s, apps, mode)
        ids = proc.get_prompt_ids(ptxt, return_tensors="pt")
        # giữ token <|startofprev|> đầu và phần cuối prompt: Whisper chỉ nhận nửa context cho prompt
        if len(ids) > CFG["MAX_PROMPT_TOK"]:
            ids = torch.cat([ids[:1], ids[-(CFG["MAX_PROMPT_TOK"] - 1):]])
        with torch.no_grad():
            out = model.generate(input_features=feats.cuda().half(), prompt_ids=ids.cuda(), language="vi", task="transcribe",
                                 max_new_tokens=160, num_beams=1)
        for u, t in zip(utt, proc.batch_decode(out, skip_special_tokens=True)):
            t = t.strip()
            # tuỳ phiên bản transformers, chuỗi giải mã có thể còn dính prompt ở đầu
            hyp[u["key"]] = t[len(ptxt):].strip() if t.startswith(ptxt) else t
        n += len(utt)
    print(f"{mode}: {n} lượt, {(time.time() - t0) / max(1, n):.3f} s/lượt", flush=True)
    return hyp


def norm(t):
    t = unicodedata.normalize("NFC", str(t).lower())
    for a, b in (("òa", "oà"), ("óa", "oá"), ("ỏa", "oả"), ("õa", "oã"), ("ọa", "oạ"), ("ùy", "uỳ"), ("úy", "uý"), ("ủy", "uỷ"), ("ũy", "uỹ"), ("ụy", "uỵ")):
        t = t.replace(a, b)
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


REF = {u["key"]: u["text"] for s in DEV for u in utterances(s, TRAIN_DIR) if u["text"] and norm(u["text"])}


def wer(hyp):
    keys = sorted(REF)
    return jiwer.wer([norm(REF[k]) for k in keys], [norm(hyp.get(k, "")) for k in keys])


# %%
SCORES = {"lora": wer(OLD)}
DEV_HYP = {}
for mode in CFG["MODES"]:
    DEV_HYP[mode] = transcribe(DEV, TRAIN_DIR, TRAIN_APPS, mode)
    SCORES[mode] = wer(DEV_HYP[mode])
    print(mode, SCORES[mode], flush=True)
print(json.dumps(SCORES, indent=1))
(WORK / "asr_prompt_score.json").write_text(json.dumps(SCORES, indent=1), encoding="utf-8")
(WORK / "asr_prompt_dev.json").write_text(json.dumps(DEV_HYP, ensure_ascii=False, indent=1), encoding="utf-8")
diff = [k for k in sorted(REF) if norm(OLD.get(k, "")) != norm(DEV_HYP[CFG["MODES"][0]].get(k, ""))]
for k in diff[:15]:
    print(f"\n{k}\n  REF: {REF[k]}\n  OLD: {OLD.get(k)}\n  NEW: {DEV_HYP[CFG['MODES'][0]].get(k)}")

# %%
best = min(CFG["MODES"], key=SCORES.get)
merged = dict(OLD)
if CFG["PUBLIC"] and SCORES[best] <= SCORES["lora"] + CFG["WER_TOL"]:
    PUB_DIR, PUB, PUB_APPS = load_split("public_test")
    pub_hyp = transcribe(PUB, PUB_DIR, PUB_APPS, best)
    changed = [k for k in pub_hyp if norm(pub_hyp[k]) != norm(OLD.get(k, ""))]
    print(best, len(changed), "lượt public thay đổi")
    for k in changed[:30]:
        print(f"{k}\n  OLD: {OLD.get(k)}\n  NEW: {pub_hyp[k]}")
    merged.update(pub_hyp)
else:
    print("prompt không hơn LoRA cũ, giữ transcript cũ")
(WORK / "asr_phowhisper-lora.json").write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
shutil.copytree(ADAPTER, WORK / "phowhisper_lora_adapter", dirs_exist_ok=True)
print(len(merged), "transcript")

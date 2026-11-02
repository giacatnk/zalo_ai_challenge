# %% [markdown]
# # LifeMate: so sánh ASR
# Chép mọi lượt user (lượt cuối và history) của dev set bằng nhiều ASR, chấm với `audio_text` của train, xuất transcript để notebook `zalo-ai` dùng lại qua `TRANSCRIPT_SRC`.
#
# | Chỉ số | Ý nghĩa |
# |---|---|
# | WER, CER | sai số từ / ký tự sau khi bỏ dấu câu, lowercase, NFC |
# | arg_recall | trong các từ của tham số string ở gold tool call (tên, nội dung tin nhắn, địa điểm...) mà `audio_text` có nói, tỉ lệ transcript cũng có; chỉ tính lượt cuối |
# | first/last | tỉ lệ lượt sai từ đầu / từ cuối (dấu hiệu ASR cắt mép audio) |
# | s/utt | giây mỗi lượt trên T4 (giới hạn đề: 30 s/sample gồm cả LLM, trên A30) |

# %%
# transformers không ghim ở đây: mỗi model ghi gói cần cài trong CFG["MODELS"] (qwen-asr và Gemma 4 cần bản mới, khác nhau)
!pip install -q sherpa-onnx sentencepiece soundfile librosa jiwer

# %%
import os, sys, json, re, time, hashlib, unicodedata, subprocess
from pathlib import Path
from collections import Counter, defaultdict

CFG = {
    "INPUT_ROOT": "/kaggle/input",
    "WORK_DIR": "/kaggle/working",
    "DEV_N": 200,                                       # phải trùng dev_ids() trong notebook zalo-ai
    "LIMIT": None,                                      # số sample dev; None = cả dev
    # tên -> (backend, model). Tổng hệ thống < 6B và LLM đã ~4B, nên ASR tối đa ~2B
    # Run asr1 (dev 200): vietasr WER .190, phowhisper-large .267, phowhisper-medium .290, whisper-v3-turbo .359.
    # vietasr sai từ đầu/cuối ở ~23% lượt nên thử đệm khoảng lặng hai đầu và beam search.
    # backend "vietasr:pad=<giây>:beam=<n>"; asr2: vietasr-pad05-beam4 .179. PhoWhisper-large + LoRA (zalo-ai-asr-lora) .101.
    # tên -> (backend, model, gói pip cài trước khi chạy model đó). gemma4 "limit=<n>": chỉ chạy n lượt đầu.
    # T4 không có bf16, nên so fp16 với fp32 (2 GPU) xem Gemma có tràn số ở fp16 không.
    # asr3: qwen3-asr-1.7b .284, gemma4-e2b fp16 .456 / fp32 .406.
    # nemo cài sau cùng vì nó kéo theo torch/transformers bản khác, làm hỏng môi trường cho model chạy sau.
    "MODELS": {
        "chunkformer-rnnt": ("chunkformer", "khanhld/chunkformer-rnnt-large-vie", "chunkformer"),
        "chunkformer-ctc": ("chunkformer", "khanhld/chunkformer-ctc-large-vie", "chunkformer"),
        "phoasr-whisper-small": ("whisper", "Qualcomm-AI-Research/PhoASR-whisper-small", ""),
        "parakeet-ctc-0.6b-vi": ("nemo", "nvidia/parakeet-ctc-0.6b-Vietnamese", "nemo_toolkit[asr]"),
    },
    "WHISPER_BEAMS": 1,
}
ASK_TOOL = "sys.ask_missing_info"
WORK = Path(CFG["WORK_DIR"])


def find_data_root(root, max_depth=5):
    base = root.rstrip("/").count("/")
    for cur, dirs, _ in os.walk(root):
        if os.path.isfile(os.path.join(cur, "train", "train.jsonl")):
            return Path(cur)
        if cur.count("/") - base >= max_depth:
            dirs[:] = []
    raise FileNotFoundError("Không thấy train/train.jsonl; kiểm tra đã Add Input dataset chưa")


ROOT = find_data_root(CFG["INPUT_ROOT"])
TRAIN_DIR = ROOT / "train"
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


DEV = [s for s in TRAIN if s["id"] in dev_ids(TRAIN, CFG["DEV_N"])]
if CFG["LIMIT"]:
    DEV = DEV[: CFG["LIMIT"]]

ITEMS, REF, LAST = [], {}, set()
for s in DEV:
    ITEMS.append({"key": s["id"], "audio": str(TRAIN_DIR / s["audio"])})
    REF[s["id"]] = s["audio_text"]; LAST.add(s["id"])
    for k, t in enumerate(s["history"]):
        if t.get("role") == "user" and t.get("audio"):
            ITEMS.append({"key": f"{s['id']}#{k}", "audio": str(TRAIN_DIR / t["audio"])})
            REF[f"{s['id']}#{k}"] = t.get("audio_text", "")
(WORK / "asr_items.json").write_text(json.dumps(ITEMS, ensure_ascii=False), encoding="utf-8")
print(f"{len(DEV)} sample dev, {len(ITEMS)} lượt audio", Counter(kind_of(s) for s in DEV))

# %% [markdown]
# ## ASR runner
# Mỗi model chạy trong process riêng để giải phóng VRAM giữa các model.

# %%
%%writefile /kaggle/working/asr_one.py
import json, sys, time
from pathlib import Path
import soundfile as sf, librosa

items_path, out_path, backend, model, beams = sys.argv[1:6]
items = json.loads(Path(items_path).read_text(encoding="utf-8"))


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    return librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav


res = {}
if backend.startswith("vietasr"):
    import numpy as np, sherpa_onnx
    from huggingface_hub import snapshot_download
    opts = dict(o.split("=") for o in backend.split(":")[1:])
    pad = np.zeros(int(16000 * float(opts.get("pad", 0))), dtype="float32")
    beam = int(opts.get("beam", 0))
    d = Path(snapshot_download(model, allow_patterns=["exp/*-epoch-12-avg-8.onnx", "data/Vietnam_bpe_2000_new/tokens.txt"]))
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(d / "exp/encoder-epoch-12-avg-8.onnx"), decoder=str(d / "exp/decoder-epoch-12-avg-8.onnx"),
        joiner=str(d / "exp/joiner-epoch-12-avg-8.onnx"), tokens=str(d / "data/Vietnam_bpe_2000_new/tokens.txt"),
        num_threads=4, sample_rate=16000, feature_dim=80,
        **({"decoding_method": "modified_beam_search", "max_active_paths": beam} if beam else {"decoding_method": "greedy_search"}))
    t0 = time.time()
    for it in items:
        st = rec.create_stream(); st.accept_waveform(16000, np.concatenate([pad, load_audio(it["audio"]), pad])); rec.decode_stream(st)
        res[it["key"]] = st.result.text.strip()
elif backend == "qwen3asr":
    import torch
    from qwen_asr import Qwen3ASRModel
    asr = Qwen3ASRModel.from_pretrained(model, dtype=torch.float16, device_map="cuda:0", max_inference_batch_size=16, max_new_tokens=256)
    t0 = time.time()
    for i in range(0, len(items), 16):
        part = items[i: i + 16]
        outs = asr.transcribe(audio=[(load_audio(it["audio"]), 16000) for it in part], language=["Vietnamese"] * len(part))
        res.update({it["key"]: o.text.strip() for it, o in zip(part, outs)})
elif backend.startswith("gemma4"):
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    opts = dict(o.split("=") for o in backend.split(":")[1:])
    items = items[: int(opts["limit"])] if "limit" in opts else items
    proc = AutoProcessor.from_pretrained(model)
    lm = AutoModelForMultimodalLM.from_pretrained(model, dtype=getattr(torch, opts.get("dtype", "float16")), device_map="auto")
    # câu lệnh theo khuyến nghị của model card; audio_text của đề viết số bằng chữ số
    prompt = ("Transcribe the following speech segment in Vietnamese into Vietnamese text. Only output the transcription, with no newlines. "
              "When transcribing numbers, write the digits, i.e. write 50 instead of năm mươi.")
    t0 = time.time()
    for it in items:
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "audio", "audio": it["audio"]}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True).to(lm.device, dtype=lm.dtype)
        with torch.inference_mode():
            out = lm.generate(**inp, max_new_tokens=192, do_sample=False)
        res[it["key"]] = proc.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
elif backend in ("chunkformer", "nemo"):
    # Cả hai nhận đường dẫn file, card không nói sample rate: ghi lại wav mono 16 kHz
    wav_dir = Path(out_path).parent / "wav16"
    wav_dir.mkdir(exist_ok=True)
    paths = []
    for it in items:
        p = wav_dir / (it["key"].replace("#", "_") + ".wav")
        if not p.exists():
            sf.write(p, load_audio(it["audio"]), 16000)
        paths.append(str(p))
    t0 = time.time()
    if backend == "chunkformer":
        from chunkformer import ChunkFormerModel
        asr = ChunkFormerModel.from_pretrained(model)
        asr = asr.to("cuda") if hasattr(asr, "to") else asr
        outs = []
        for i in range(0, len(paths), 64):
            outs += asr.batch_decode(audio_paths=paths[i: i + 64], chunk_size=64, left_context_size=128, right_context_size=128, total_batch_duration=1800)
    else:
        import nemo.collections.asr as nemo_asr
        asr = nemo_asr.models.ASRModel.from_pretrained(model_name=model).cuda().eval()
        outs = [o.text if hasattr(o, "text") else o for o in asr.transcribe(paths, batch_size=16)]
    res = {it["key"]: str(o).strip() for it, o in zip(items, outs)}
else:
    import torch
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=model, torch_dtype=torch.float16, device=0)
    t0 = time.time()
    for it in items:
        out = asr(load_audio(it["audio"]), generate_kwargs={"language": "vi", "task": "transcribe", "num_beams": int(beams)})
        res[it["key"]] = out["text"].strip()
dt = time.time() - t0
Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"n": len(res), "sec": dt}))

# %%
TIMING = {}
for name, (backend, model, pkgs) in CFG["MODELS"].items():
    out = WORK / f"asr_{name}.json"
    if out.exists():
        print(name, "đã có"); continue
    if pkgs:
        subprocess.run(f"{sys.executable} -m pip install -q {pkgs}", shell=True, check=False)
    r = subprocess.run([sys.executable, str(WORK / "asr_one.py"), str(WORK / "asr_items.json"), str(out), backend, model, str(CFG["WHISPER_BEAMS"])],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[lỗi] {name}:\n{r.stderr[-3000:]}"); continue
    info = json.loads(r.stdout.strip().splitlines()[-1])
    TIMING[name] = info["sec"] / info["n"]
    print(f"{name}: {TIMING[name]:.2f} s/lượt")

# %% [markdown]
# ## Chấm

# %%
import jiwer

def norm(t):
    # "hoà"/"hòa" là cùng một từ, chỉ khác vị trí dấu
    t = unicodedata.normalize("NFC", str(t).lower()).replace("òa", "oà").replace("óa", "oá").replace("ỏa", "oả").replace("õa", "oã").replace("ọa", "oạ").replace("ùy", "uỳ").replace("úy", "uý").replace("ủy", "uỷ").replace("ũy", "uỹ").replace("ụy", "uỵ")
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


def gold_strings(sample):
    return [v for c in sample["label"]["tool_calls"] if c["name"] != ASK_TOOL for v in (c["arguments"] or {}).values()
            if isinstance(v, str) and not re.fullmatch(r"[a-z]+_\d+|\d{4}-\d\d-\d\d.*|[a-z_]+", v)]


rows = []
for name in CFG["MODELS"]:
    p = WORK / f"asr_{name}.json"
    if not p.exists():
        continue
    hyp = json.loads(p.read_text(encoding="utf-8"))
    # model chạy với limit chỉ chấm trên các lượt đã chép
    keys = [k for k in REF if norm(REF[k]) and k in hyp]
    refs, hyps = [norm(REF[k]) for k in keys], [norm(hyp.get(k, "")) for k in keys]
    hit_w = tot_w = 0
    for s in (s for s in DEV if s["id"] in hyp):
        hw = set(norm(hyp.get(s["id"], "")).split())
        # Chỉ tính phần có trong audio_text: giá trị lấy từ kết quả tool (số tiền hoá đơn, ID) ASR không thể nghe ra
        rw = set(norm(s["audio_text"]).split())
        for v in gold_strings(s):
            ws = [w for w in norm(v).split() if w in rw]; tot_w += len(ws); hit_w += sum(w in hw for w in ws)
    edge = [(r.split()[0] != h.split()[0], r.split()[-1] != h.split()[-1]) for r, h in zip(refs, hyps) if r and h]
    # Nhiều model viết số bằng chữ trong khi audio_text dùng chữ số: WER_nd chỉ chấm các lượt không có chữ số để so công bằng
    nd = [(r, h) for r, h in zip(refs, hyps) if not re.search(r"\d", r)]
    rows.append({"model": name, "WER": jiwer.wer(refs, hyps), "WER_nd": jiwer.wer(*map(list, zip(*nd))), "CER": jiwer.cer(refs, hyps), "arg_recall": hit_w / max(tot_w, 1),
                 "first_err": sum(e[0] for e in edge) / len(edge), "last_err": sum(e[1] for e in edge) / len(edge), "s/utt": TIMING.get(name)})

print(f"{'model':22s} {'WER':>6s} {'WER_nd':>6s} {'CER':>6s} {'arg_rec':>8s} {'first':>6s} {'last':>6s} {'s/utt':>6s}")
for r in sorted(rows, key=lambda r: r["WER"]):
    print(f"{r['model']:22s} {r['WER']:6.3f} {r['WER_nd']:6.3f} {r['CER']:6.3f} {r['arg_recall']:8.3f} {r['first_err']:6.3f} {r['last_err']:6.3f} {r['s/utt'] or float('nan'):6.2f}")
(WORK / "asr_scores.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

# %%
for s in DEV[:8]:
    print(f"\n{s['id']} REF: {s['audio_text']}")
    for name in CFG["MODELS"]:
        p = WORK / f"asr_{name}.json"
        if p.exists():
            print(f"  {name:18s}: {json.loads(p.read_text(encoding='utf-8')).get(s['id'])}")

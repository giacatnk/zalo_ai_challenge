# %% [markdown]
# # LifeMate pipeline: audio → ASR → agent gọi tool → `submission.jsonl`
#
# Chạy trên Kaggle (GPU T4 x2, bật Internet). Kết quả: `/kaggle/working/result/submission.jsonl`.
#
# | Bước | Làm gì | Ý lấy từ paper |
# |---|---|---|
# | 1. ASR | VietASR (Zipformer 68M, ONNX, CPU) chép audio của lượt cuối và các lượt user trong history; hotword lấy từ user profile + enum trong tool schema | VietASR (Interspeech 2025); Spoken FC (ACM MM 2026): lỗi chính nằm ở tên riêng và số |
# | 2. Chuẩn hoá số | "năm trăm nghìn" → 500000; LLM nhận cả bản gốc lẫn bản chuẩn hoá | Spoken FC |
# | 3. Few-shot | Lấy k mẫu train có câu lệnh gần nhất (TF-IDF ký tự) làm ví dụ chuỗi tool call đúng, để model học quy ước của gold (vd đọc trước khi ghi) | APIGen-MT: dữ liệu có hành động ground-truth |
# | 4. Router | LLM phân loại ACT / ASK / OUT_OF_SCOPE; OUT_OF_SCOPE trả `[]`, không gọi tool nào | When2Call (NAACL 2025), Hammer irrelevance (ICLR 2025) |
# | 5. Agent loop | Qwen3-4B-Instruct-2507 gọi tool nhiều bước trên `ToolEnv`; batch theo bước trên vLLM | APIGen-MT / τ-bench: multi-step, chấm theo state |
# | 6. Kiểm call trước khi chạy | Ép kiểu nhẹ + `env.validate` (không tính vào bài nộp); call sai không chạy mà trả lỗi để LLM sửa | ToolACE: rule checker; call bị từ chối vẫn tính vào U nếu đã `env.call` |
# | 7. Ghi kết quả | `env.get_tool_calls()` + `env.get_state_hash()`; kiểm bằng validator của BTC | README dữ liệu |
#
# **Model** (Apache-2.0, phát hành trước 26/10/2026): `Qwen/Qwen3-4B-Instruct-2507` (~4B) + `zzasdf/viet_iter3_pseudo_label` (68M). Tổng < 6B, LLM < 5B, context 16,384.
#
# **Chạy thử:** để `SPLIT = "train"`, `LIMIT = 30` → xem điểm cục bộ ở mục 12. **Nộp bài:** `SPLIT = "public_test"`, `LIMIT = None`.
#
# **Checkpoint (phiên Kaggle tối đa 12 giờ):** kết quả ghi vào `/kaggle/working/checkpoint/` sau mỗi chunk; sau `TIME_BUDGET_H` giờ notebook dừng nhận chunk mới để kịp lưu output. Lần sau, gắn output của lần trước làm Input: notebook tự tìm thư mục `checkpoint/`, bỏ qua sample đã xong và chạy tiếp. `submission.jsonl` chỉ được ghi khi đủ mọi sample. Đổi `RUN_TAG` khi đổi prompt/model để không dùng lại kết quả cũ.

# %%
# vllm không ghim sẽ kéo torch bản CUDA 13, lệch với torchaudio CUDA 12.8 có sẵn trên Kaggle.
# vllm 0.10.2 ghim torch/torchaudio/torchvision 2.8 cu128 cùng bộ, có wheel cho Python 3.13.
# transformers 5 (Kaggle cài sẵn) ra sau vllm 0.10.2 nên ghim bản cùng thời 4.56.2; bản này cần huggingface_hub < 1.0.
# Qwen3.5 (Gated DeltaNet) cần vllm >= 0.17; vllm 0.31 ghim torch 2.13 + transformers 5, nên là một bộ cài riêng.
STACK = {
    "VLLM": "0.10.2",                                   # "0.10.2": Qwen3 | "0.31.0": Qwen3.5
}
if STACK["VLLM"] == "0.10.2":
    !pip install -q "vllm==0.10.2" "transformers==4.56.2" "huggingface_hub<1.0" sherpa-onnx sentencepiece soundfile librosa scikit-learn
else:
    !pip install -q "vllm=={STACK['VLLM']}" sherpa-onnx sentencepiece soundfile librosa scikit-learn
    # transformers tự import torchaudio nếu có; bản CUDA 12.8 của Kaggle lệch torch CUDA 13 nên gỡ đi
    !pip uninstall -y -q torchaudio
# Không import torchaudio: pipeline không dùng, và với bộ vllm 0.31 (torch CUDA 13) bản torchaudio CUDA 12.8 của Kaggle báo lỗi khi import.
import torch, transformers, vllm
print("vllm", vllm.__version__, "| torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__)
print("GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert vllm.__version__ == STACK["VLLM"], "torch/vllm cũ vẫn đang nạp: Restart session rồi chạy lại từ ô này"
assert torch.cuda.is_available(), "torch không thấy GPU: kiểm Accelerator = GPU T4 x2"

# %% [markdown]
# ## 1. Config

# %%
import os, sys, json, re, random, unicodedata, time, subprocess, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta

CFG = {
    "SPLIT": "train",                                   # train | public_test | private_test
    "SUBSET": "dev",                                    # train: "dev" = 200 sample cố định (loại khỏi few-shot), "all" = cả train
    "LIMIT": None,                                      # số sample chạy thử; None = tất cả
    "INPUT_ROOT": "/kaggle/input",
    "DATA_ROOT": None,                                  # None = tự tìm thư mục chứa <split>/<split>.jsonl
    "WORK_DIR": "/kaggle/working",
    # "asr": chạy ASR ở mục 5 | "gold": audio_text của train (đo riêng phần LLM)
    # | tên file transcript do notebook zalo-ai-asr xuất, vd "asr_phowhisper-large.json" (gắn output đó làm Input)
    "TRANSCRIPT_SRC": "asr",
    "ASR_BACKEND": "vietasr",                           # vietasr | whisper
    "ASR_HOTWORDS": True,
    "HOTWORDS_SCORE": 1.5,
    "WHISPER_MODEL": "openai/whisper-large-v3-turbo",   # MIT, 809M
    "LLM": "Qwen/Qwen3-4B-Instruct-2507",              # "Qwen/Qwen3.5-4B" cần STACK VLLM 0.31.0 ở ô cài đặt
    "LLM_TEXT_ONLY": False,                             # True cho model đa phương thức (Qwen3.5)
    "MAX_MODEL_LEN": 16384,
    "MAX_NEW_TOKENS": 768,
    "MAX_STEPS": 10,
    "MAX_INVALID_RETRIES": 2,
    "READ_BARRIER": True,                                # một lượt có tool đọc thì bỏ các call ghi đi sau nó, để model quyết định sau khi thấy kết quả
    "TOOL_RESULT_MAX_CHARS": 3000,
    "FEWSHOT_K": 3,
    "USE_ROUTER": False,                                # run v1: router gắn nhầm 2/29 sample trong phạm vi thành OUT_OF_SCOPE
    "ARG_PRUNE": True,                                  # bỏ tham số mà gold train gần như không bao giờ điền (mục 4)
    "PREREQ_HINT": True,                                # nhắc tool đọc mà gold luôn gọi trước tool ghi (mục 4)
    "VERIFY": False,                                    # vai kiểm tra chỉ được bỏ call ghi sai; dev lora rb6v: model SFT gần như luôn bỏ call 1, 49.47 -> 30.27
    "VERIFY_MAX_TOKENS": 600,
    "VERIFY_RESULT_CHARS": 1500,
    "ASR_FIX": False,                                   # vai sửa lời nghe trước agent chính (tên riêng, động từ vô lý); chưa test trên GPU
    "ASR_FIX_MAX_TOKENS": 200,
    "ASR_FIX_MAX_PAIRS": 3,
    "SEED": 42,
    "RUN_TAG": "v7",                                  # đổi khi đổi prompt/model; checkpoint khác tag bị bỏ qua
    "CHUNK_SIZE": 64,                                   # số sample mỗi chunk; ghi checkpoint sau mỗi chunk
    "TIME_BUDGET_H": 11.0,                              # dừng nhận chunk mới sau ngần này giờ (phiên Kaggle tối đa 12h)
    "JEV_PROBE": False,                                 # True: chỉ chấm xác suất bước kế tiếp theo chuỗi call gold (mục 9b), không chạy agent
}
ASK_TOOL = "sys.ask_missing_info"
T_START = time.time()


def find_data_root(split, root, max_depth=5):
    """Vị trí mount của dataset phụ thuộc tên dataset trên Kaggle, nên tìm theo file <split>/<split>.jsonl."""
    base = root.rstrip("/").count("/")
    for cur, dirs, files in os.walk(root):
        if os.path.isfile(os.path.join(cur, split, f"{split}.jsonl")):
            return Path(cur)
        if cur.count("/") - base >= max_depth:
            dirs[:] = []
    raise FileNotFoundError(f"Không thấy {split}/{split}.jsonl dưới {root}; kiểm tra đã Add Input dataset chưa")


ROOT = Path(CFG["DATA_ROOT"]) if CFG["DATA_ROOT"] else find_data_root(CFG["SPLIT"], CFG["INPUT_ROOT"])
SPLIT = CFG["SPLIT"]
SPLIT_DIR = ROOT / SPLIT
WORK = Path(CFG["WORK_DIR"])
OUT_DIR = WORK / "result"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("DATA ROOT:", ROOT)
print((ROOT / "VERSION").read_text() if (ROOT / "VERSION").exists() else "")

random.seed(CFG["SEED"])
os.environ["PYTHONHASHSEED"] = str(CFG["SEED"])
import numpy as np, torch
np.random.seed(CFG["SEED"]); torch.manual_seed(CFG["SEED"])

# %% [markdown]
# ## 2. Dữ liệu, tool catalog và `zaic_tools`
# Format theo `README.md` của gói dữ liệu. Agent không đọc `state.json`; mọi dữ liệu app lấy qua `env.call`.

# %%
sys.dont_write_bytecode = True                          # /kaggle/input chỉ đọc
sys.path.insert(0, str(SPLIT_DIR))
from zaic_tools import ToolEnv


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def kind_of(s):
    calls = s["label"]["tool_calls"]
    return "oos" if not calls else ("ask" if any(c["name"] == ASK_TOOL for c in calls) else "act")


def dev_ids(train_rows, n=200):
    """Dev set cố định, chia theo tỉ lệ act/ask/oos; thứ tự theo md5(id) nên notebook ASR chọn ra đúng cùng tập."""
    by_kind = defaultdict(list)
    for t in train_rows:
        by_kind[kind_of(t)].append(t["id"])
    out = []
    for ids in by_kind.values():
        ids.sort(key=lambda i: hashlib.md5(i.encode()).hexdigest())
        out += ids[: round(n * len(ids) / len(train_rows))]
    return set(out)


TRAIN_ROWS = read_jsonl(ROOT / "train" / "train.jsonl")
DEV_IDS = dev_ids(TRAIN_ROWS) if SPLIT == "train" and CFG["SUBSET"] == "dev" else set()
# Sample dev không được làm few-shot hay nguồn thống kê, để điểm dev gần với điểm trên test
TRAIN_POOL = [t for t in TRAIN_ROWS if t["id"] not in DEV_IDS]

SAMPLES = read_jsonl(SPLIT_DIR / f"{SPLIT}.jsonl")
if DEV_IDS:
    SAMPLES = [s for s in SAMPLES if s["id"] in DEV_IDS]
if CFG["LIMIT"]:
    SAMPLES = SAMPLES[: CFG["LIMIT"]]

CATALOG = json.loads((SPLIT_DIR / f"{SPLIT}_tools_catalog.json").read_text(encoding="utf-8"))
TOOL_SPEC = {t["name"]: t for a in CATALOG["apps"] for t in a["tools"]}
APP_INFO = {a["id"]: a for a in CATALOG["apps"]}
FUZZY = {(t["name"], k) for t in TOOL_SPEC.values() for k, v in (t["parameters"].get("properties") or {}).items() if v.get("x-match") == "fuzzy"}

print(f"{len(SAMPLES)} sample | {len(TOOL_SPEC)} tool | {len(FUZZY)} tham số fuzzy")
print(Counter(len(s["history"]) for s in SAMPLES))
if "label" in SAMPLES[0]:
    print(Counter(kind_of(s) for s in SAMPLES))

# %% [markdown]
# ## 3. Checkpoint: nạp kết quả của các lần chạy trước
# Tìm mọi thư mục `checkpoint/` trong `/kaggle/input` (output notebook lần trước đã gắn làm Input) và trong `/kaggle/working`. Chỉ nhận dòng cùng `SPLIT` và `RUN_TAG`. Kết quả gộp được ghi lại vào `/kaggle/working/checkpoint/`, nên output lần này chứa toàn bộ kết quả cộng dồn.
#
# Checkpoint chỉ lưu sample đã chạy xong. Sample đang chạy dở khi phiên dừng không được lưu, nên lần sau chạy lại từ đầu với một `ToolEnv` mới; không call nào của lần dở dang lọt vào bài nộp.

# %%
CKPT_DIR = WORK / "checkpoint"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RES_FILE = CKPT_DIR / f"{SPLIT}_results.jsonl"
ASR_FILE = CKPT_DIR / f"{SPLIT}_transcripts.json"
TRACE_FILE = CKPT_DIR / f"{SPLIT}_trace.jsonl"


def find_checkpoint_dirs(roots, max_depth=5):
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        base = root.rstrip("/").count("/")
        for cur, dirs, _ in os.walk(root):
            if os.path.basename(cur) == "checkpoint":
                found.append(Path(cur)); dirs[:] = []; continue
            if cur.count("/") - base >= max_depth:
                dirs[:] = []
    return found


DONE, TRANSCRIPTS, TRACES = {}, {}, {}
for d in find_checkpoint_dirs([CFG["INPUT_ROOT"], str(WORK)]):
    n_before = len(DONE)
    if (d / RES_FILE.name).exists():
        for row in read_jsonl(d / RES_FILE.name):
            if row.get("run_tag") == CFG["RUN_TAG"]:
                DONE[row["id"]] = row
    if (d / ASR_FILE.name).exists():
        TRANSCRIPTS.update(json.loads((d / ASR_FILE.name).read_text(encoding="utf-8")))
    if (d / TRACE_FILE.name).exists():
        TRACES.update({r["id"]: r for r in read_jsonl(d / TRACE_FILE.name) if r["id"] in DONE})
    print(f"checkpoint {d}: +{len(DONE) - n_before} sample")

with open(RES_FILE, "w", encoding="utf-8") as f:
    for row in DONE.values():
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
with open(TRACE_FILE, "w", encoding="utf-8") as f:
    for row in TRACES.values():
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
ASR_FILE.write_text(json.dumps(TRANSCRIPTS, ensure_ascii=False), encoding="utf-8")

TODO = [s for s in SAMPLES if s["id"] not in DONE]
print(f"Đã xong {len(SAMPLES) - len(TODO)}/{len(SAMPLES)}, còn {len(TODO)} sample (RUN_TAG={CFG['RUN_TAG']})")

# %% [markdown]
# ## 4. Tool schema và kiểm call
# Ép kiểu nhẹ ("26" → 26) rồi hỏi `env.validate` (validator chính thức, không tính vào bài nộp). Chỉ call qua được mới được `env.call`, vì call bị từ chối vẫn nằm trong `get_tool_calls()` và bị tính vào U.
#
# **Cắt tham số theo thống kê gold train:** call chỉ khớp khi tham số khớp, nên tham số thừa làm mất điểm (vd `alarm.create` kèm `label`, `calendar.list_events` kèm khoảng ngày). Từ nhãn train: tool đọc được gọi với `{}` ở ≥ 80% lần thì luôn gọi `{}`; tham số tuỳ chọn của tool ghi xuất hiện < 5% số lần thì bỏ.

# %%
def _arg_stats(train_rows, min_calls=5):
    n_calls, n_empty, used = Counter(), Counter(), Counter()
    for t in train_rows:
        for c in t["label"]["tool_calls"] or []:
            n_calls[c["name"]] += 1
            n_empty[c["name"]] += not c["arguments"]
            used.update((c["name"], k) for k in c["arguments"])
    read_empty, rare_opt = set(), set()
    for name, n in n_calls.items():
        spec = TOOL_SPEC.get(name)
        if spec is None or n < min_calls:
            continue
        if spec.get("kind") == "read" and n_empty[name] / n >= 0.8:
            read_empty.add(name)
        if spec.get("kind") == "write":
            # tham số thuộc x-one-of là "bắt buộc một trong", bỏ đi có thể làm call thiếu nhóm bắt buộc
            required = set(spec["parameters"].get("required") or []) | {k for g in spec["parameters"].get("x-one-of", []) for k in g}
            rare_opt |= {(name, k) for k in (spec["parameters"].get("properties") or {}) if k not in required and used[(name, k)] / n < 0.05}
    return read_empty, rare_opt


def _prereq_stats(train_rows, min_calls=20, min_share=0.7):
    """Tool ghi -> các tool đọc mà gold gần như luôn gọi trước nó (vd banking.transfer cần get_balance + list_beneficiaries)."""
    n_calls, before = Counter(), defaultdict(Counter)
    for t in train_rows:
        names = [c["name"] for c in t["label"]["tool_calls"] or []]
        for i, n in enumerate(names):
            if TOOL_SPEC.get(n, {}).get("kind") != "write":
                continue
            n_calls[n] += 1
            before[n].update({p for p in names[:i] if TOOL_SPEC.get(p, {}).get("kind") == "read"})
    return {n: [p for p, c in before[n].most_common() if c / k >= min_share] for n, k in n_calls.items() if k >= min_calls}


def _xref_prereq(tool_spec):
    """Tool ghi -> tool đọc cùng app trả về tập dữ liệu mà tham số bắt buộc của nó x-ref tới (vd parking.reserve.lot_id x-ref
    parking.lots, parking.find trả lots[] -> gọi parking.find trước). Thống kê train không có gì cho app chỉ có trong test
    (public_test: 11 app mới, 65% sample); trên train luật này khớp gold 1119/1159 lần, chỗ lệch chủ yếu là tham số x-one-of
    (phone.call với số đọc thẳng) nên bỏ nhóm đó."""
    returns = defaultdict(list)
    for n, t in tool_spec.items():
        if t.get("kind") == "read":
            for r in str(t.get("returns") or "").replace(",", " ").split():
                returns[(n.split(".")[0], r.rstrip("[]"))].append(n)
    out = {}
    for n, t in tool_spec.items():
        if t.get("kind") != "write":
            continue
        props = t["parameters"].get("properties") or {}
        refs = [props[k]["x-ref"] for k in t["parameters"].get("required") or [] if (props.get(k) or {}).get("x-ref")]
        ps = sorted({p for ref in refs for p in returns.get(tuple(ref.split(".", 1)), [])})
        if ps:
            out[n] = ps
    return out


READ_EMPTY, RARE_OPT = _arg_stats(TRAIN_POOL) if CFG["ARG_PRUNE"] else (set(), set())
PREREQ = {k: v for k, v in _prereq_stats(TRAIN_POOL).items() if v} if CFG["PREREQ_HINT"] else {}
if CFG["PREREQ_HINT"]:
    # rb6-gold: thêm xref cho cả tool có trong train làm gợi ý cả những tool đọc mà gold thường bỏ qua (ID đã có trong lịch sử);
    # thống kê train đã lọc các tool đó, nên xref chỉ dùng cho tool train chưa từng gọi (app mới của test)
    _TRAIN_USED = {c["name"] for t in TRAIN_POOL for c in t["label"]["tool_calls"]} | \
                  {c["name"] for t in TRAIN_POOL for h in t["history"] for c in (h.get("tool_calls") or [])}
    PREREQ = {**{k: v for k, v in _xref_prereq(TOOL_SPEC).items() if k not in _TRAIN_USED}, **PREREQ}
print(f"{len(READ_EMPTY)} tool đọc luôn gọi {{}} | {len(RARE_OPT)} tham số tuỳ chọn bị bỏ, vd {sorted(RARE_OPT)[:6]}")
print("tool đọc thường gọi trước tool ghi:", PREREQ)


def prune_args(name, args):
    if name in READ_EMPTY:
        return {}
    return {k: v for k, v in args.items() if (name, k) not in RARE_OPT}


def tools_for_sample(sample):
    names = [n for app in sample["context"]["installed_apps"].values() for n in app["tools"]] + [ASK_TOOL]
    tools = []
    for n in names:
        spec = TOOL_SPEC.get(n)
        if spec is None:
            continue
        params = {k: v for k, v in spec["parameters"].items() if not k.startswith("x-")}
        desc = spec.get("description", "")
        if "x-one-of" in spec["parameters"]:
            desc += f" (bắt buộc một trong các nhóm: {spec['parameters']['x-one-of']})"
        tools.append({"type": "function", "function": {"name": n, "description": desc, "parameters": params}})
    return tools


def _coerce(value, typ):
    if typ == "integer" and isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value.strip())
    if typ == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if typ == "number" and isinstance(value, str) and re.fullmatch(r"-?\d+(\.\d+)?", value.strip()):
        v = float(value.strip())
        return int(v) if v.is_integer() else v
    if typ == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    if typ == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return value


QUERY_LOWER = {w.lower() for t in TRAIN_POOL for c in t["label"]["tool_calls"] if c["name"] == "contacts.search"
               for w in str(c["arguments"].get("query", "")).split()[1:] if w[:1].islower()}
# Run v5 (train_0005): dù đã có rule, model vẫn điền pickup "công ty" thay vì work_address
PLACE_ALIAS = {"nhà": "home_address", "nhà tôi": "home_address", "nhà mình": "home_address",
               "công ty": "work_address", "cơ quan": "work_address", "chỗ làm": "work_address"}
PLACE_ARGS = {"pickup", "dropoff", "origin", "destination", "address", "from", "to"}


def check_call(call, env, allowed, profile=None):
    """Trả về (call đã ép kiểu, lỗi hoặc None)."""
    profile = profile or {}
    name = call.get("name")
    if name not in allowed:
        return call, f"Tool '{name}' không có trong các app đã cài của người dùng."
    args = call.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return call, "arguments không phải JSON hợp lệ."
    args = prune_args(name, dict(args or {}))
    # query không chấm fuzzy; gold viết hoa mọi chữ trừ vài chức danh ("Bác Ba", "Cô giáo Mai"); run v3/v5: "bố", "Bác ba" mất điểm
    if name == "contacts.search" and isinstance(args.get("query"), str):
        ws = args["query"].split()
        args["query"] = " ".join(w if (i and w.lower() in QUERY_LOWER) else w[:1].upper() + w[1:] for i, w in enumerate(ws))
    # maps.get_route giữ nguyên "nhà" ở 2 gold train, nên chỉ thay cho app đặt xe / lấy hàng
    for k in (PLACE_ARGS & set(args)) if name.split(".")[0] in ("ride_hailing", "package_tracking") else ():
        if isinstance(args[k], str) and args[k].strip().lower() in PLACE_ALIAS and profile.get(PLACE_ALIAS[args[k].strip().lower()]):
            args[k] = profile[PLACE_ALIAS[args[k].strip().lower()]]
    props = TOOL_SPEC[name]["parameters"].get("properties") or {}
    for k, v in list(args.items()):
        typ = props.get(k, {}).get("type")
        if isinstance(typ, str):
            args[k] = _coerce(v, typ)
    res = env.validate(name, args)
    if not res.get("ok"):
        err = res.get("error", "lỗi schema")
        if name != ASK_TOOL and "thiếu" in err:
            err += f". Nếu người dùng không cung cấp và không suy ra được từ context/kết quả tool, gọi {ASK_TOOL}."
        return {"name": name, "arguments": args}, err
    return {"name": name, "arguments": args}, None

# %% [markdown]
# ## 5. ASR
# Chép lượt cuối và các lượt user trong history của các sample chưa xong; transcript đã có trong checkpoint thì dùng lại. Chạy trong process riêng để process chính không giữ CUDA trước khi vLLM khởi động. Hotword lấy từ user profile và enum trong schema; không gọi tool trên env để lấy hotword, vì mọi `env.call` đều vào bài nộp.

# %%
def build_hotwords(sample, limit=60):
    words = [v for v in sample["context"].get("user_profile", {}).values() if isinstance(v, str)]
    for t in tools_for_sample(sample):
        for spec in (t["function"]["parameters"].get("properties") or {}).values():
            words += [e.replace("_", " ") for e in spec.get("enum", []) or [] if isinstance(e, str)]
    seen, out = set(), []
    for w in words:
        if 2 <= len(w) <= 40 and w.lower() not in seen:
            seen.add(w.lower()); out.append(w)
    return out[:limit]


manifest = []
for s in TODO:
    hw = build_hotwords(s) if CFG["ASR_HOTWORDS"] else []
    items = [(s["id"], s["audio"])] + [(f"{s['id']}#{k}", t["audio"]) for k, t in enumerate(s["history"]) if t.get("role") == "user" and t.get("audio")]
    manifest += [{"key": key, "audio": str(SPLIT_DIR / a), "hotwords": hw} for key, a in items if key not in TRANSCRIPTS]
(WORK / "asr_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
print(len(manifest), "file audio cần chép")

# %%
%%writefile /kaggle/working/asr_run.py
import json, sys
from pathlib import Path
import soundfile as sf, librosa

manifest_path, out_path, backend, hotwords_on, hw_score, whisper_model = sys.argv[1:7]
hotwords_on = hotwords_on == "1"
items = json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav


results = {}
if backend == "vietasr":
    import sherpa_onnx, sentencepiece as spm
    from huggingface_hub import snapshot_download
    d = Path(snapshot_download("zzasdf/viet_iter3_pseudo_label", allow_patterns=[
        "exp/*-epoch-12-avg-8.onnx", "data/Vietnam_bpe_2000_new/*.model", "data/Vietnam_bpe_2000_new/tokens.txt"]))
    lang = d / "data/Vietnam_bpe_2000_new"
    kw = dict(encoder=str(d / "exp/encoder-epoch-12-avg-8.onnx"), decoder=str(d / "exp/decoder-epoch-12-avg-8.onnx"),
              joiner=str(d / "exp/joiner-epoch-12-avg-8.onnx"), tokens=str(lang / "tokens.txt"),
              num_threads=4, sample_rate=16000, feature_dim=80)
    tokens = [l.split()[0] for l in (lang / "tokens.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    lowercase = not any(ch.isupper() for t in tokens for ch in t)
    use_hw = False
    if hotwords_on:
        # sherpa-onnx cần file vocab "piece<TAB>score"; chỉ bật hotword khi bpe.model khớp tokens.txt
        sp = spm.SentencePieceProcessor(model_file=str(lang / "bpe.model"))
        pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        if set(pieces) <= set(tokens):
            vocab = Path("/kaggle/working/bpe.vocab")
            vocab.write_text("\n".join(f"{p}\t{sp.get_score(i)}" for i, p in enumerate(pieces)), encoding="utf-8")
            try:
                rec = sherpa_onnx.OfflineRecognizer.from_transducer(
                    **kw, decoding_method="modified_beam_search", max_active_paths=4,
                    hotwords_file="", hotwords_score=float(hw_score), modeling_unit="bpe", bpe_vocab=str(vocab))
                use_hw = True
            except Exception as e:
                print("[warn] tắt hotword:", e)
        else:
            print("[warn] bpe.model không khớp tokens.txt, tắt hotword")
    if not use_hw:
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(**kw, decoding_method="greedy_search")
    for i, it in enumerate(items):
        hw = "/".join(w.lower() if lowercase else w for w in it["hotwords"]) if use_hw else ""
        try:
            st = rec.create_stream(hotwords=hw) if hw else rec.create_stream()
        except Exception:
            st = rec.create_stream()
        st.accept_waveform(16000, load_audio(it["audio"]))
        rec.decode_stream(st)
        results[it["key"]] = st.result.text.strip()
        if i % 200 == 0:
            print(i, it["key"], results[it["key"]], flush=True)
else:
    import torch
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=whisper_model, torch_dtype=torch.float16, device=0)
    for it in items:
        out = asr(load_audio(it["audio"]), generate_kwargs={"language": "vi", "task": "transcribe", "num_beams": 1})
        results[it["key"]] = out["text"].strip()

Path(out_path).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
print("done", len(results))

# %%
t0 = time.time()
SRC = CFG["TRANSCRIPT_SRC"]
if SRC == "gold":
    assert SPLIT == "train", "chỉ train có audio_text"
    for s in TODO:
        TRANSCRIPTS[s["id"]] = s["audio_text"]
        for k, turn in enumerate(s["history"]):
            if turn.get("role") == "user":
                TRANSCRIPTS[f"{s['id']}#{k}"] = turn.get("audio_text", "")
elif SRC != "asr":
    hits = [Path(c) / SRC for c, _, files in os.walk(CFG["INPUT_ROOT"]) if SRC in files]
    assert hits, f"Không thấy {SRC} dưới {CFG['INPUT_ROOT']}: gắn output notebook zalo-ai-asr làm Input"
    cached = json.loads(hits[0].read_text(encoding="utf-8"))
    missing = [m["key"] for m in manifest if m["key"] not in cached]
    assert not missing, f"{len(missing)} audio không có trong {SRC}, vd {missing[:3]}"
    TRANSCRIPTS.update({m["key"]: cached[m["key"]] for m in manifest})
elif manifest:
    subprocess.run([sys.executable, str(WORK / "asr_run.py"), str(WORK / "asr_manifest.json"), str(WORK / "transcripts_new.json"),
                    CFG["ASR_BACKEND"], "1" if CFG["ASR_HOTWORDS"] else "0", str(CFG["HOTWORDS_SCORE"]), CFG["WHISPER_MODEL"]], check=True)
    TRANSCRIPTS.update(json.loads((WORK / "transcripts_new.json").read_text(encoding="utf-8")))
if SRC == "asr":
    ASR_FILE.write_text(json.dumps(TRANSCRIPTS, ensure_ascii=False), encoding="utf-8")
print(f"{len(TRANSCRIPTS)} transcript, {time.time() - t0:.0f}s")
for s in TODO[:5]:
    print(s["id"], "|", TRANSCRIPTS.get(s["id"]), "|", s.get("audio_text", ""))

# %% [markdown]
# ## 6. Chuẩn hoá số tiếng Việt
# Chỉ đổi những cụm chắc chắn là số (có "mươi/trăm/nghìn/triệu/tỷ...", chuỗi ≥ 3 chữ số đọc liền, hoặc một chữ số trước đơn vị như "độ", "giờ").

# %%
_DIGITS = {"không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5, "lăm": 5,
           "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9}
_SCALES = {"nghìn": 1000, "ngàn": 1000, "triệu": 10**6, "tỷ": 10**9, "tỉ": 10**9}
_MULT = {"mười", "mươi", "trăm", "linh", "lẻ", "rưỡi"} | set(_SCALES)
_NUMWORDS = set(_DIGITS) | _MULT
_UNITS_AFTER = {"độ", "giờ", "phút", "giây", "người", "phần", "ngày", "tuần", "k", "đồng", "lít", "suất", "phòng", "lần", "chiếc", "cái"}


def _parse_run(toks):
    if all(t in _DIGITS for t in toks):
        return "".join(str(_DIGITS[t]) for t in toks)
    total, small, pending, prev = 0, 0, None, None
    for t in toks:
        if t in _DIGITS:
            if pending is not None:
                return None
            pending = _DIGITS[t]
        elif t == "mười":
            small += 10
        elif t == "mươi":
            small += (pending if pending is not None else 1) * 10; pending = None
        elif t == "trăm":
            small += (pending if pending is not None else 1) * 100; pending = None
        elif t in _SCALES:
            total += (small + (pending or 0) or 1) * _SCALES[t]; small, pending = 0, None
        elif t == "rưỡi":
            if prev == "trăm":
                small += 50
            elif prev in _SCALES:
                total += _SCALES[prev] // 2
            else:
                return None
        prev = t
    # nói tắt: "một triệu hai" = 1,200,000; "hai trăm năm" = 250
    if pending is not None and len(toks) >= 2 and toks[-2] in _SCALES and small == 0:
        return str(total + pending * _SCALES[toks[-2]] // 10)
    if pending is not None and len(toks) >= 2 and toks[-2] == "trăm":
        return str(total + small + pending * 10)
    return str(total + small + (pending or 0))


def normalize_numbers(text):
    toks = text.split()
    low = [unicodedata.normalize("NFC", t.lower().strip(",.!?")) for t in toks]
    out, i = [], 0
    while i < len(toks):
        if low[i] in _NUMWORDS and low[i] not in ("linh", "lẻ", "rưỡi"):
            j = i
            while j < len(toks) and low[j] in _NUMWORDS:
                j += 1
            run, nxt = low[i:j], (low[j] if j < len(toks) else "")
            convert = any(t in _MULT for t in run) or len(run) >= 3 or (len(run) == 1 and nxt in _UNITS_AFTER and run[0] != "không")
            val = _parse_run(run) if convert else None
            if val is not None:
                out.append(val); i = j; continue
        out.append(toks[i]); i += 1
    return " ".join(out)


_H = r"(\d{1,2}|mười một|mười hai|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)"
_PERIOD = r"(?:\s+(sáng|trưa|chiều|tối|đêm))?"
_TIME_RES = [
    (re.compile(rf"\b{_H}\s+giờ\s+kém\s+(\d{{1,2}})(?:\s+phút)?{_PERIOD}"), lambda h, m: (h - 1, 60 - m)),
    (re.compile(rf"\b{_H}(?:\s+giờ)?\s+rưỡi{_PERIOD}"), lambda h, m: (h, 30)),
    (re.compile(rf"\b{_H}\s+giờ\s+(\d{{1,2}})(?:\s+phút)?{_PERIOD}"), lambda h, m: (h, m)),
    (re.compile(rf"\b{_H}\s+giờ{_PERIOD}(?!\s*(?:đồng hồ|nữa|sau|trước))"), lambda h, m: (h, 0)),
]


def _hour_value(tok):
    return int(tok) if tok.isdigit() else int(_parse_run(tok.split()))


def _to_24h(h, period):
    if period in ("chiều", "tối") and h < 12:
        return h + 12
    if period == "trưa" and h < 11:
        return h + 12
    if period == "đêm" and 6 <= h < 12:
        return h + 12
    return h


def normalize_time(text):
    """'8 giờ kém 15 tối' → '19:45 tối', 'tám rưỡi sáng' → '08:30 sáng'. Chạy sau normalize_numbers."""
    text = unicodedata.normalize("NFC", text.lower())
    for rx, fn in _TIME_RES:
        def sub(m):
            if re.search(r"\b(trong|sau|mất|khoảng|thêm)\s*$", m.string[: m.start()]):
                return m.group(0)                       # "trong 2 giờ" là khoảng thời gian, không phải giờ đồng hồ
            g = m.groups()
            h = _hour_value(g[0])
            minute = int(g[1]) if len(g) == 3 and g[1] and g[1].isdigit() else 0
            h, minute = fn(h, minute)
            period = g[-1]
            if not (0 <= h <= 24 and 0 <= minute < 60):
                return m.group(0)
            return f"{_to_24h(h, period):02d}:{minute:02d}" + (f" {period}" if period else "")
        text = rx.sub(sub, text)
    return text


for s in ["chuyển cho mẹ năm trăm nghìn", "bật điều hòa hai mươi sáu độ", "nhắc tôi lúc ba giờ chiều mai", "chuyển một triệu rưỡi", "tắt đèn tầng một",
          "NHẮC CHỊ RA SÂN BAY LÚC TÁM GIỜ KÉM MƯỜI LĂM TỐI MAI", "GỌI LẠI CHO KHÁCH LÚC TÁM RƯỠI TỐI MAI", "BÁO THỨC NĂM RƯỠI SÁNG MAI", "họp lúc hai giờ ba mươi chiều"]:
    print(f"{s!r:58} → {normalize_time(normalize_numbers(s))!r}")

# %% [markdown]
# ## 7. Few-shot từ train
# Với mỗi sample, lấy `FEWSHOT_K` mẫu train có câu lệnh gần nhất (TF-IDF n-gram ký tự) và đưa chuỗi tool call đúng của chúng vào prompt. Khi chạy trên train thì bỏ chính sample đó.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

TRAIN = TRAIN_POOL
_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
_train_mat = _vec.fit_transform([t["audio_text"] for t in TRAIN])


def _fmt_calls(calls):
    # Viết đúng format <tool_call> của Qwen: run v1 hiển thị dạng name({...}) và model chép y nguyên dạng đó thay vì gọi tool.
    if not calls:
        return "OUT_OF_SCOPE"
    return "\n".join(f'<tool_call>\n{json.dumps({"name": c["name"], "arguments": c["arguments"]}, ensure_ascii=False)}\n</tool_call>' for c in calls)


def fewshot_block(sample, query):
    if CFG["FEWSHOT_K"] <= 0:
        return ""
    sims = (_train_mat @ _vec.transform([query]).T).toarray().ravel()
    picked = []
    for i in sims.argsort()[::-1]:
        if TRAIN[i]["id"] != sample["id"]:
            picked.append(TRAIN[i])
        if len(picked) == CFG["FEWSHOT_K"]:
            break
    parts = [f'Ví dụ {n + 1}: "{t["audio_text"]}"\n{_fmt_calls(t["label"]["tool_calls"])}' for n, t in enumerate(picked)]
    return ("Các ví dụ dưới đây là của người dùng KHÁC (ID và dữ liệu khác), chỉ để tham khảo cách chọn tool, thứ tự tool và "
            "những tham số nào được điền. Ví dụ liệt kê gộp mọi call; khi chạy thật, mỗi lượt bạn chỉ gọi các tool cần ngay lúc đó:\n\n"
            + "\n\n".join(parts))

# %% [markdown]
# ## 8. Prompt

# %%
SYSTEM_RULES = f"""Bạn là trợ lý giọng nói điều khiển các app của người dùng bằng cách gọi tool. Bạn chỉ hành động qua tool call; không cần trả lời bằng lời.

Quy tắc:
1. Chỉ dùng các tool được cung cấp, mỗi call viết trong <tool_call>{{"name": ..., "arguments": {{...}}}}</tool_call>. Nếu ngay từ đầu không app nào xử lý được yêu cầu, KHÔNG gọi tool nào (kể cả tool đọc), chỉ trả lời: OUT_OF_SCOPE. Đã gọi tool rồi thì không trả lời OUT_OF_SCOPE nữa. Người dùng huỷ ngay trong câu ("à thôi khỏi", "không cần") hoặc dặn đừng làm gì ("đừng tắt", "cứ để đó") thì cũng là OUT_OF_SCOPE, không gọi tool nào.
2. Nếu thiếu tham số bắt buộc mà không suy ra được từ yêu cầu, context, user profile, lịch sử hoặc kết quả tool, gọi {ASK_TOOL} với tool_name là tool định gọi và params là tên các tham số còn thiếu (đúng tên trong schema), rồi dừng. Không bịa giá trị. Thông tin tra được bằng tool đọc (số điện thoại, ID, số tiền hoá đơn, địa điểm trong lịch) thì tra, không hỏi. contacts.search trả về NHIỀU người và yêu cầu không nói rõ là ai thì gọi {ASK_TOOL} cho contact_id, không tự chọn.
3. Khi cần ID hay dữ liệu hiện tại (contact_id, device_id, event_id, số dư...), gọi tool đọc trước rồi dùng kết quả cho tool ghi. Làm theo cách sắp xếp tool trong các ví dụ. Không gọi tool đọc không cần cho yêu cầu, không gọi lại tool đọc đã có kết quả. Nhiều call ghi thì gọi theo đúng thứ tự người dùng nói; chọn theo tiêu chí ("hai cái rẻ nhất", "gần nhất") thì gọi theo thứ tự tiêu chí đó (rẻ nhất trước), không theo thứ tự trong kết quả tool.
4. Chỉ điền tham số bắt buộc và tham số người dùng nói rõ. Không tự thêm nhãn, ghi chú, bộ lọc, khoảng ngày. Tool đọc dạng liệt kê thì gọi không tham số.
4b. Tính giờ cẩn thận: "chín giờ tối" là 21:00, "hai giờ chiều" là 14:00. "Sớm hơn/muộn hơn N tiếng" tính từ giờ lấy trong kết quả tool. Số phút = giờ kết thúc trừ giờ bắt đầu (14:30 đến 15:15 là 45 phút; "tối nay học 1 tiếng" là 60). "Xong thì nhắc" là giờ kết thúc ghi trong kết quả tool.
5. Yêu cầu có điều kiện ("nếu ... thì ..."): gọi tool đọc để kiểm điều kiện, đọc kỹ kết quả; chỉ gọi tool ghi khi điều kiện ĐÚNG. Điều kiện sai thì dừng (DONE). Người dùng đổi ý giữa câu ("à không", "mà thôi") thì làm theo ý cuối. So số dư với số tiền cần trả: "nếu tài khoản thanh toán không đủ thì lấy từ tiết kiệm" nghĩa là chuyển với from_account_id là tài khoản tiết kiệm, không mở sổ tiết kiệm. Thiết bị nhà thông minh: lọc theo trường floor ("tầng 2" là floor 2), room và state trong kết quả get_status; "phòng đang bật điều hoà" là phòng có thiết bị climate với mode khác "off"; chỉ đổi thiết bị đang khác trạng thái yêu cầu.
5b. Chuyển khoản ngân hàng (banking.transfer): người nhận lấy từ banking.list_beneficiaries, không dùng contacts.search. contacts.search dùng cho nhắn tin SMS, gọi điện, ví điện tử, nạp điện thoại.
6. Chép nguyên giá trị từ kết quả tool và context (ID, địa chỉ, tên). "Nhà" là home_address, "công ty"/"cơ quan" là work_address trong user_profile. Đặt ngay ("bây giờ", "luôn") thì không điền giờ hẹn. Số ghi bằng số đúng kiểu schema. Thời gian quy đổi theo current_time và timezone, định dạng giống ví dụ (vd 2026-11-29T09:00:00+07:00); dùng bản đã chuẩn hoá giờ 24h nếu có.
7. Bản chép giọng nói (ASR) không dấu câu và có thể sai chính tả, tên riêng hoặc số. Khi tên gần giống một thực thể trong context hoặc kết quả tool, dùng tên đúng của thực thể đó. Người dùng có thể nói giọng địa phương: mô = đâu/nào, răng = sao, rứa = thế/vậy, ni = này, tê = kia, nớ = đó, chừ = bây giờ, bựa = hôm, mạ = mẹ, hông/hổng = không, dìa = về, mần = làm, bển = bên đó, ổng/bả = ông ấy/bà ấy, hen/nghen = nhé.
8. Khi đã gọi đủ tool để hoàn thành yêu cầu, trả lời ngắn: DONE."""


_WEEKDAY = ["thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm", "thứ Sáu", "thứ Bảy", "Chủ nhật"]


def date_table(current_time):
    # Run v2: model 4B tính sai "mai" ở 4/30 sample (vd hôm nay 24 -> ra 27), nên tính sẵn ngày tương đối.
    try:
        now = datetime.fromisoformat(current_time)
    except (TypeError, ValueError):
        return ""
    d = lambda k: (now + timedelta(days=k)).date().isoformat()
    lines = [f"- hôm nay: {d(0)} ({_WEEKDAY[now.weekday()]})", f"- mai: {d(1)}", f"- ngày kia/ngày mốt: {d(2)}", f"- hôm qua: {d(-1)}"]
    nxt = [f"{_WEEKDAY[(now.weekday() + k) % 7]} {d(k)}" for k in range(1, 8)]
    lines.append("- 7 ngày tới: " + ", ".join(nxt))
    return "Ngày tham chiếu (đã tính sẵn, dùng đúng các ngày này):\n" + "\n".join(lines)


def prereq_hint(allowed):
    lines = [f"- {w}: gọi trước {', '.join(p for p in ps if p in allowed)}" for w, ps in PREREQ.items()
             if w in allowed and any(p in allowed for p in ps)]
    return "Quy trình thường gặp (tool đọc cần gọi trước tool ghi):\n" + "\n".join(lines) if lines else ""


def build_system(sample, query):
    ctx = {k: v for k, v in sample["context"].items() if k != "installed_apps"}
    apps = "\n".join(f"- {a}: {APP_INFO.get(a, {}).get('name_vi', '')}. {APP_INFO.get(a, {}).get('description', '')}" for a in sample["context"]["installed_apps"])
    allowed = {n for app in sample["context"]["installed_apps"].values() for n in app["tools"]}
    extra = "\n\n".join(x for x in (date_table(sample["context"].get("current_time")), prereq_hint(allowed)) if x)
    return (f"{SYSTEM_RULES}\n\nContext:\n{json.dumps(ctx, ensure_ascii=False, indent=1)}\n\n{extra}\n\nApp đã cài:\n{apps}\n\n{fewshot_block(sample, query)}")


_TITLE_WORDS = {"cô", "chị", "anh", "bác", "em", "chú", "bà", "ông", "bạn", "con", "cháu", "dì", "cậu", "mợ", "thím", "giáo"}
_ORDINAL_WORDS = {"cái", "lần", "người", "tập", "cuốn", "chiếc", "số", "bài", "món", "phần", "tầng", "phòng", "buổi"}
_WD_TOKENS = {"hai": 0, "2": 0, "ba": 1, "3": 1, "tư": 2, "4": 2, "năm": 3, "5": 3, "sáu": 4, "6": 4, "bảy": 5, "7": 5}


def date_hints(text, current_time):
    # Run v6-gold: có bảng ngày trong system prompt mà model vẫn chọn lệch 1-2 ngày (mai -> ngày kia, thứ ba -> thứ năm),
    # nên gắn thẳng ngày đã tính cạnh câu. Train: thứ X (kể cả "tuần sau") là lần gặp kế tiếp trong 1-7 ngày tới (32/34).
    try:
        now = datetime.fromisoformat(current_time)
    except (TypeError, ValueError):
        return ""
    t = unicodedata.normalize("NFC", text.lower())
    d = lambda k: f"{(now + timedelta(days=k)).date().isoformat()} ({_WEEKDAY[(now.weekday() + k) % 7]})"
    prev = lambda m: (t[: m.start()].split() or [""])[-1]
    out = []
    if any(prev(m) not in _TITLE_WORDS for m in re.finditer(r"\bmai\b", t)):
        out.append(f"mai = {d(1)}")
    # "hai mốt", "ba mốt" là ngày 21, 31 chứ không phải ngày mốt
    if any(prev(m) not in {"hai", "ba", "mươi"} for m in re.finditer(r"\b(ngày kia|ngày mốt|mốt)\b", t)):
        out.append(f"ngày kia/mốt = {d(2)}")
    if re.search(r"\bhôm qua\b", t):
        out.append(f"hôm qua = {d(-1)}")
    for m in re.finditer(r"\b(thứ (hai|ba|tư|năm|sáu|bảy|[2-7])|chủ nhật)\b", t):
        if prev(m) in _ORDINAL_WORDS:
            continue
        wd = 6 if m.group(1) == "chủ nhật" else _WD_TOKENS[m.group(2)]
        out.append(f"{m.group(1)} = {d((wd - now.weekday()) % 7 or 7)}")
    return "Ngày tương ứng trong câu: " + "; ".join(dict.fromkeys(out)) if out else ""


def build_user_turn(transcript, current_time=None):
    norm = normalize_time(normalize_numbers(transcript))
    msg = f'Yêu cầu (bản chép ASR, có thể có lỗi): "{transcript}"'
    if norm != unicodedata.normalize("NFC", transcript.lower()):
        msg += f'\nBản đã chuẩn hoá số và giờ (24h): "{norm}"'
    hint = date_hints(transcript, current_time)
    return f"{msg}\n{hint}" if hint else msg


def history_messages(sample):
    msgs = []
    for k, turn in enumerate(sample["history"]):
        if turn.get("role") == "user":
            msgs.append({"role": "user", "content": build_user_turn(TRANSCRIPTS.get(f"{sample['id']}#{k}", ""), sample["context"].get("current_time"))})
        else:
            calls = turn.get("tool_calls") or []
            if calls:
                msgs.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
                for r in (turn.get("tool_results") or [])[: len(calls)]:
                    msgs.append({"role": "tool", "content": json.dumps(r, ensure_ascii=False)[: CFG["TOOL_RESULT_MAX_CHARS"]]})
            msgs.append({"role": "assistant", "content": turn.get("text", "")})
    return msgs


ROUTER_PROMPT = """Phân loại yêu cầu CUỐI của người dùng với các app đã cài. Trả lời đúng một nhãn:
- ACT: các app đã cài làm được (kể cả khi phải đọc dữ liệu trước).
- ASK: có app phù hợp nhưng người dùng chưa nói đủ thông tin bắt buộc.
- OUT_OF_SCOPE: không app đã cài nào làm được yêu cầu này.

App đã cài:
{apps}
{history}
{user}

Nhãn:"""

# %% [markdown]
# ## 9. LLM (vLLM, tensor parallel trên 2 GPU)
# Bỏ qua khi checkpoint đã đủ mọi sample.

# %%
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# T4 (SM 7.5): FlashAttention và nhánh paged của xformers đều cần SM 8.0+.
# Kernel Triton unified không kiểm SM và là backend vllm bản mới tự chọn cho T4.
# Từ vllm 0.11 tên backend này bị bỏ và vllm tự chọn Triton cho T4, nên chỉ đặt với bộ cài 0.10.2.
_VLLM_VER = getattr(sys.modules.get("vllm"), "__version__", "0.10.2")
if torch.cuda.get_device_capability(0) < (8, 0) and _VLLM_VER.startswith("0.10."):
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")
print("attention backend:", os.environ.get("VLLM_ATTENTION_BACKEND", "mặc định"))
from vllm import LLM, SamplingParams

def resolve_llm(name):
    """"input:<thư mục>" = model đã merge do notebook zalo-ai-sft xuất, gắn làm Input; còn lại là id trên HF."""
    if not name.startswith("input:"):
        return name
    hits = [p.parent for p in Path(CFG["INPUT_ROOT"]).rglob("config.json") if p.parent.name == name[len("input:"):]]
    assert hits, f"Không thấy {name} trong {CFG['INPUT_ROOT']}; gắn output của zalo-ai-sft làm Input"
    return str(hits[0])


if TODO:
    llm = LLM(model=resolve_llm(CFG["LLM"]), dtype="half", tensor_parallel_size=max(1, torch.cuda.device_count()),
              max_model_len=CFG["MAX_MODEL_LEN"], seed=CFG["SEED"],
              # Mặc định 256 seq: lúc warm-up sampler sort logits 256 x 151k vocab, vượt phần VRAM còn lại của T4.
              # Mỗi lần generate chỉ gửi tối đa một chunk, nên max_num_seqs = CHUNK_SIZE là đủ.
              max_num_seqs=CFG["CHUNK_SIZE"], gpu_memory_utilization=0.85,
              # Qwen3.5 kèm vision encoder: bỏ để dành VRAM cho KV cache, bài này chỉ có văn bản
              **({"language_model_only": True} if CFG["LLM_TEXT_ONLY"] else {}))
    TOK = llm.get_tokenizer()
    # Lỗi kernel attention chỉ lộ ra ở lần generate đầu, nên chạy thử ngay tại đây với batch lẫn prompt dài/ngắn
    # (giống lúc chạy thật) thay vì để engine chết giữa chunk.
    _t = time.time()
    _out = llm.generate(["xin chào " * 1500, "Trả lời ngắn: 1+1=?", "ok " * 300],
                        SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False)
    print(f"smoke test ok ({time.time() - _t:.1f}s):", repr(_out[1].outputs[0].text))
SP_AGENT = SamplingParams(temperature=0.0, max_tokens=CFG["MAX_NEW_TOKENS"], seed=CFG["SEED"])
SP_ROUTER = SamplingParams(temperature=0.0, max_tokens=8, seed=CFG["SEED"])


def render(messages, tools=None):
    kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return TOK.apply_chat_template(messages, tools=tools, **kw) if tools else TOK.apply_chat_template(messages, **kw)


_TC_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
# Run v1: model đôi khi viết call trần dạng `weather.get_forecast({"date": ...})` hoặc `sys.ask_missing_info{...}`
_BARE_RE = re.compile(r"(?<![\w.])([a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)\s*\(?\s*(?=\{)")
_JSON = json.JSONDecoder()


def _decode_at(text, pos):
    try:
        return _JSON.raw_decode(text, pos)[0]
    except json.JSONDecodeError:
        return None


_XML_FN_RE = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
_XML_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.S)


def _xml_value(raw):
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_tool_calls(text):
    # model có chế độ suy nghĩ: chỉ đọc phần sau </think>
    text = text.rsplit("</think>", 1)[-1]
    calls = []
    for m in _TC_RE.finditer(text):
        body = m.group(1)
        # định dạng qwen3_coder của Qwen3.5: <function=name><parameter=k>v</parameter></function>
        fn = _XML_FN_RE.search(body)
        if fn:
            calls.append({"name": fn.group(1), "arguments": {k: _xml_value(v) for k, v in _XML_PARAM_RE.findall(fn.group(2))}})
            continue
        start = body.find("{")
        obj = _decode_at(body, start) if start >= 0 else None
        if not isinstance(obj, dict):
            calls.append({"name": None, "arguments": {}}); continue
        calls.append({"name": obj.get("name"), "arguments": obj.get("arguments", obj.get("parameters", {}))})
    if calls:
        return calls
    for m in _BARE_RE.finditer(text):
        if m.group(1) not in TOOL_SPEC:
            continue
        args = _decode_at(text, m.end())
        ok = isinstance(args, dict)
        calls.append({"name": m.group(1) if ok else None, "arguments": args if ok else {}})
    if calls:
        return calls
    # Run v4 (train_0003): JSON {"name": ..., "arguments": ...} trần, không có thẻ <tool_call>
    for m in re.finditer(r"\{", text):
        obj = _decode_at(text, m.start())
        if isinstance(obj, dict) and obj.get("name") in TOOL_SPEC:
            calls.append({"name": obj["name"], "arguments": obj.get("arguments", obj.get("parameters", {}))})
    return calls

# %% [markdown]
# ## 9b. Probe kiểu Jev: xác suất của từng lựa chọn bước kế tiếp
# Đi theo chuỗi call gold (mỗi call một lượt, kết quả lấy từ label). Ở mỗi bước, lựa chọn = các tool được phép + DONE.
# Xác suất một lựa chọn = tích xác suất từng token của `<tool_call>\n{"name": "<tool>"` (hoặc `DONE`), đọc từ logprobs
# của các nút rẽ nhánh trên cây tiền tố, rồi chuẩn hoá lại trên tập lựa chọn. Không sinh token nào ngoài 1 token mỗi nút.

# %%
import math
from collections import defaultdict

SP_PROBE = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
DONE_OPT = "DONE"


def probe_contexts(s):
    st = init_state(s)
    gold, res = s["label"]["tool_calls"], s["label"]["tool_results"]
    labels = [c["name"] for c in gold] + ([] if gold and gold[-1]["name"] == ASK_TOOL else [DONE_OPT])
    out, msgs = [], [dict(m) for m in st["messages"]]
    for i, lab in enumerate(labels):
        ctx = {"messages": [dict(m) for m in msgs], "tools": st["tools"]}
        prompt = _fit_context(ctx)
        if prompt is not None:
            out.append({"id": s["id"], "step": i, "n_steps": len(labels), "gold": lab, "prompt": prompt,
                        "cands": sorted(st["allowed"]) + [DONE_OPT]})
        if i < len(gold):
            c = gold[i]
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}]})
            msgs.append({"role": "tool", "content": _short(res[i] if i < len(res) else {}, CFG["TOOL_RESULT_MAX_CHARS"])})
    return out


def _cand_ids(c):
    text = DONE_OPT if c == DONE_OPT else '<tool_call>\n{"name": "' + c + '"'
    return tuple(TOK(text, add_special_tokens=False).input_ids)


def probe_score(ctxs):
    """Trả về {lựa chọn: xác suất} cho mỗi context; chỉ hỏi model ở nút có từ 2 nhánh con."""
    seqs, nodes = [], []
    for ci, x in enumerate(ctxs):
        x["ids"] = TOK(x["prompt"], add_special_tokens=False).input_ids
        cs = {c: _cand_ids(c) for c in x["cands"]}
        kids = defaultdict(set)
        for ids in cs.values():
            for k in range(len(ids)):
                kids[ids[:k]].add(ids[k])
        seqs.append(cs)
        nodes += [(ci, pre) for pre, ch in kids.items() if len(ch) > 1]
    # nút gốc chạy trước để prefill prompt dài vào prefix cache, các nút sau chỉ tính vài token mới
    lp = {}
    for batch in ([n for n in nodes if not n[1]], [n for n in nodes if n[1]]):
        if not batch:
            continue
        outs = llm.generate([{"prompt_token_ids": ctxs[ci]["ids"] + list(pre)} for ci, pre in batch], SP_PROBE, use_tqdm=False)
        for n, o in zip(batch, outs):
            top = {t: v.logprob for t, v in o.outputs[0].logprobs[0].items()}
            lp[n] = (top, min(top.values()) - 2.0)
    result = []
    for ci, cs in enumerate(seqs):
        score = {}
        for c, ids in cs.items():
            total = 0.0
            for k in range(len(ids)):
                if (ci, ids[:k]) in lp:
                    top, floor = lp[(ci, ids[:k])]
                    total += top.get(ids[k], floor)
            score[c] = total
        m = max(score.values())
        z = sum(math.exp(v - m) for v in score.values())
        result.append({c: math.exp(v - m) / z for c, v in score.items()})
    return result


def _auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


# gọi ở cuối mục 10 vì cần init_state, _fit_context định nghĩa ở đó
def run_probe():
    assert SPLIT == "train", "probe cần label gold"
    t0, rows = time.time(), []
    ctxs = [x for s in SAMPLES for x in probe_contexts(s)]
    print(f"{len(ctxs)} context từ {len(SAMPLES)} sample")
    # 8 prompt khoảng 8K token vừa KV cache 2 x T4, để nút con còn trúng prefix cache của nút gốc
    for i in range(0, len(ctxs), 8):
        part = ctxs[i: i + 8]
        for x, p in zip(part, probe_score(part)):
            ranked = sorted(p.items(), key=lambda kv: -kv[1])
            rows.append({"id": x["id"], "step": x["step"], "n_steps": x["n_steps"], "gold": x["gold"],
                         "top": ranked[:8], "p_gold": p.get(x["gold"], 0.0), "rank": [c for c, _ in ranked].index(x["gold"]) + 1})
        if (i // 8) % 10 == 0:
            print(f"  {len(rows)}/{len(ctxs)} | {time.time() - t0:.0f}s", flush=True)
    (OUT_DIR / "jev_probe.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(f"probe xong {len(rows)} context trong {time.time() - t0:.0f}s")

    def report(name, rs):
        if not rs:
            return
        ok = [r for r in rs if r["rank"] == 1]
        print(f"{name:22s} n={len(rs):4d} top1={len(ok) / len(rs):.3f} top3={sum(r['rank'] <= 3 for r in rs) / len(rs):.3f} "
              f"top5={sum(r['rank'] <= 5 for r in rs) / len(rs):.3f} | p_top1 đúng={sum(r['top'][0][1] for r in ok) / max(1, len(ok)):.3f} "
              f"sai={sum(r['top'][0][1] for r in rs if r['rank'] > 1) / max(1, len(rs) - len(ok)):.3f} "
              f"AUROC={_auroc([r['top'][0][1] for r in ok], [r['top'][0][1] for r in rs if r['rank'] > 1]):.3f}")

    report("tất cả", rows)
    report("bước đầu", [r for r in rows if r["step"] == 0])
    report("bước sau", [r for r in rows if r["step"] > 0])
    report("gold = DONE", [r for r in rows if r["gold"] == DONE_OPT])
    report("gold = ask", [r for r in rows if r["gold"] == ASK_TOOL])
    report("gold = tool khác", [r for r in rows if r["gold"] not in (DONE_OPT, ASK_TOOL)])
    for th in (0.5, 0.7, 0.9, 0.97):
        sel = [r for r in rows if r["top"][0][1] >= th]
        print(f"  p_top1 >= {th}: phủ {len(sel) / len(rows):.3f}, top1 đúng {sum(r['rank'] == 1 for r in sel) / max(1, len(sel)):.3f}")
    by_id = defaultdict(list)
    for r in rows:
        by_id[r["id"]].append(r)
    print(f"sample đúng top1 mọi bước: {sum(all(r['rank'] == 1 for r in v) for v in by_id.values())}/{len(by_id)}")
    print("Nhầm hay gặp (gold -> top1):", Counter((r["gold"], r["top"][0][0]) for r in rows if r["rank"] > 1).most_common(12))

# %% [markdown]
# ## 10. Agent loop theo chunk, ghi checkpoint sau mỗi chunk
# Mỗi `ToolEnv` chỉ tạo một lần cho mỗi sample. Trong một chunk: router → vòng lặp batch theo bước (render prompt → sinh một lượt → kiểm call → `env.call` call hợp lệ → thêm kết quả vào hội thoại). Hết chunk thì ghi kết quả vào checkpoint. Quá `TIME_BUDGET_H` thì không nhận chunk mới.

# %%
def _short(obj, n):
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + "...(cắt bớt)"


def _strip_accents(s):
    s = unicodedata.normalize("NFD", s.lower()).replace("đ", "d")
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _call_key(c):
    # Run v3: sau khi tìm "Trần Văn Tuấn" không thấy, model thử lại "Tran Van Tuan"; bản bỏ dấu không bao giờ khớp hơn
    args = dict(c["arguments"])
    if c["name"].endswith(".search") and isinstance(args.get("query"), str):
        args["query"] = _strip_accents(args["query"])
    return json.dumps([c["name"], args], sort_keys=True, ensure_ascii=False)


def _lists(res):
    return [v for v in res.values() if isinstance(v, list)] if isinstance(res, dict) and res.get("ok") else []


def _no_hit(res):
    ls = _lists(res)
    return bool(ls) and all(not v for v in ls)


def _single_hit(res):
    ls = _lists(res)
    return len(ls) == 1 and len(ls[0]) == 1


def _unseen_id(c, seen):
    # Run v6-gold: model bịa ID (alm_001, cont_123, contact_003) thay vì gọi tool list/search để lấy
    args = c.get("arguments")
    if not isinstance(args, dict):
        return None
    bad = [f"{k}={v!r}" for k, v in args.items() if k.endswith("_id") and isinstance(v, str) and json.dumps(v, ensure_ascii=False) not in seen]
    if bad:
        return f"{c['name']}: {', '.join(bad)} không có trong context hay kết quả tool nào. Gọi tool đọc/tìm kiếm trước để lấy ID thật."
    return None


def _fit_context(st):
    """Cắt bớt kết quả tool khi prompt vượt context 16K."""
    budget = CFG["MAX_MODEL_LEN"] - CFG["MAX_NEW_TOKENS"] - 16
    limit = CFG["TOOL_RESULT_MAX_CHARS"]
    prompt = render(st["messages"], st["tools"])
    while len(TOK(prompt).input_ids) > budget and limit > 200:
        limit //= 2
        for m in st["messages"]:
            if m["role"] == "tool":
                m["content"] = _short(m["content"], limit)
        prompt = render(st["messages"], st["tools"])
    return prompt if len(TOK(prompt).input_ids) <= budget else None


def init_state(s):
    query = TRANSCRIPTS.get(s["id"], "")
    tools = tools_for_sample(s)
    messages = [{"role": "system", "content": build_system(s, query)}] + history_messages(s) + [{"role": "user", "content": build_user_turn(query, s["context"].get("current_time"))}]
    return {"id": s["id"], "sample": s, "env": ToolEnv(SPLIT, s["id"], root=str(ROOT)), "tools": tools,
            "allowed": {t["function"]["name"] for t in tools}, "messages": messages, "query": query,
            "done": False, "steps": 0, "retries": 0, "route": None, "trace": [],
            "last_result": None, "ask_guarded": False, "search_hinted": False, "prose_nudged": False, "log": [],
            # kết quả tool đầy đủ (không bị _short cắt) để kiểm ID model đưa ra có thật không
            "seen": json.dumps([s["context"], s["history"]], ensure_ascii=False)}


def route(states):
    prompts = []
    for st in states:
        apps = "\n".join(f"- {a}: {', '.join(v['tools'])}" for a, v in st["sample"]["context"]["installed_apps"].items())
        hist = "\n".join(m["content"] for m in history_messages(st["sample"]) if m["role"] == "user" or (m["role"] == "assistant" and m["content"]))
        hist = f"\nLịch sử hội thoại:\n{hist}\n" if hist else ""
        prompts.append(render([{"role": "user", "content": ROUTER_PROMPT.format(apps=apps, history=hist, user=build_user_turn(st["query"]))}]))
    for st, o in zip(states, llm.generate(prompts, SP_ROUTER, use_tqdm=False)):
        label = o.outputs[0].text.strip().upper()
        st["route"] = "OUT_OF_SCOPE" if "OUT" in label else ("ASK" if label.startswith("ASK") else "ACT")
        st["done"] = st["route"] == "OUT_OF_SCOPE"


def _read_barrier(calls, done_keys):
    # Run sft1-gold: 68/200 sample gọi tool đọc rồi tool ghi trong cùng lượt (vd gold_price.get_price + reminders.create
    # khi điều kiện giá chưa thoả), 42 sai hash. Call bị bỏ chưa vào lịch sử nên model gọi lại được ở lượt sau.
    # Ask sau tool đọc được giữ lại (trả về thứ hai) để quyết định sau khi có kết quả đọc, xem _keep_deferred_ask.
    # Tool đọc đã chạy ở lượt trước không tính: model hay lặp lại nó kèm call ghi, chặn thì lượt chỉ còn call trùng và agent dừng.
    def fresh_read(p):
        if TOOL_SPEC.get(p.get("name"), {}).get("kind") != "read":
            return False
        try:
            return _call_key(p) not in done_keys
        except (KeyError, TypeError, AttributeError):
            return True
    for i, c in enumerate(calls):
        kind = TOOL_SPEC.get(c.get("name"), {}).get("kind")
        if i and kind in ("write", "meta") and any(fresh_read(p) for p in calls[:i]):
            return calls[:i], (c if c.get("name") == ASK_TOOL else None)
    return calls, None


def _keep_deferred_ask(turn_calls, res):
    # Mô phỏng trên trace sft1-gold-rb2 (read + ask cùng lượt): gold là ask khi search trả về nhiều người/mục (11/17),
    # còn sau list/get hay search đúng 1 kết quả thì gold hầu như luôn là act (23/27). Chặn hết ask: ask 25.0; cho qua hết: act -8.
    # Lượt có từ 2 tool đọc trở lên rồi ask (vd get_status + contacts.search + ask): trên rb3 cả 6 sample gold đều có call
    # ghi cho tool đọc kia (light_set...) trước khi hỏi hoặc là act, nên bỏ ask để model đi tiếp.
    reads = [c for c in turn_calls if TOOL_SPEC.get(c["name"], {}).get("kind") == "read"]
    return len(reads) == 1 and turn_calls[-1] is reads[0] and _multi_search(turn_calls, res)


def _multi_search(turn_calls, res):
    return bool(turn_calls) and turn_calls[-1]["name"].endswith(".search") and any(len(v) > 1 for v in _lists(res))


def _apply_pending_ask(calls, pending):
    # Ask bị hoãn vì lượt có nhiều tool đọc (rb4 train_0207: get_status + contacts.search ra 2 "Hương" + ask sms.send):
    # lượt sau model làm light_set đúng rồi tự chọn một người cho sms.send. Gold là light_set rồi ask, nên thay call đó bằng ask.
    for i, c in enumerate(calls):
        if c.get("name") == pending["arguments"].get("tool_name"):
            return calls[:i] + [pending]
    return calls


_TIME_CUE = re.compile(r"\d|\b(giờ|rưỡi|phút|sáng|trưa|chiều|tối|đêm|khuya|mai|mốt|tuần|thứ|ngày|tháng|năm|lúc|nữa|kém|tiếng|chủ nhật|hôm|bữa"
                       r"|một|hai|ba|bốn|sáu|bảy|tám|chín|mười|xong|tới|khi|đó|đấy|ấy)\b", re.I)


def _ask_reminder_time(calls, query):
    # Model hay bịa giờ cho "nhắc mình gọi cho mẹ nhé" (dev train_0219, train_0360). Trên toàn train: câu không có từ chỉ
    # thời gian thì gold hỏi time 6/6; 3 ca act còn lại đều trỏ sự kiện trong lịch sử ("giặt xong", "nhắc đó", "xe tới").
    if _TIME_CUE.search(query or ""):
        return calls
    return [{"name": ASK_TOOL, "arguments": {"tool_name": "reminders.create", "params": ["time"]}} if c.get("name") == "reminders.create" else c
            for c in calls]


def agent_step(st, text):
    st["trace"].append(text)
    st["steps"] += 1
    calls = parse_tool_calls(text)
    deferred_ask = None
    if CFG["READ_BARRIER"]:
        calls, deferred_ask = _read_barrier(calls, {_call_key(c) for c in st["env"].get_tool_calls()})
    calls = _ask_reminder_time(calls, st["query"])
    if st.get("pending_ask"):
        calls = _apply_pending_ask(calls, st["pending_ask"])
    if not calls:
        # Run v6-gold: sau khi đọc dữ liệu, model hay diễn giải bằng lời ("hoá đơn nước -> trả bằng ngân hàng") rồi dừng mà không gọi tool ghi
        if not st["prose_nudged"] and st["env"].get_tool_calls() and re.sub(r"\W", "", text).upper() != "DONE":
            st["prose_nudged"] = True
            st["messages"].append({"role": "assistant", "content": text})
            st["messages"].append({"role": "user", "content": "Bạn vừa trả lời bằng lời, chưa có tool call nào được thực thi. Nếu theo kết quả đã đọc vẫn còn việc phải làm, hãy gọi tool đó ngay. Nếu không còn gì cần làm thì trả lời: DONE"})
            return
        st["done"] = True; return

    checked = [check_call(c, st["env"], st["allowed"], st["sample"]["context"].get("user_profile")) if c.get("name") else (c, "Không đọc được JSON của tool call.") for c in calls]
    checked = [(c, e or _unseen_id(c, st["seen"])) for c, e in checked]
    errors = [e for _, e in checked if e]
    if errors and st["retries"] < CFG["MAX_INVALID_RETRIES"]:
        st["retries"] += 1
        st["messages"].append({"role": "assistant", "content": text})
        st["messages"].append({"role": "user", "content": "Các tool call trên CHƯA được thực thi vì lỗi:\n- " + "\n- ".join(errors) + "\nHãy gửi lại toàn bộ tool call đã sửa."})
        return

    done_keys = {_call_key(c) for c in st["env"].get_tool_calls()}
    fresh = [c for c, e in checked if not e and _call_key(c) not in done_keys]
    if not fresh:
        dup_search = any(c["name"].endswith(".search") for c, e in checked if not e)
        if dup_search and st["retries"] < CFG["MAX_INVALID_RETRIES"]:
            st["retries"] += 1
            st["messages"].append({"role": "assistant", "content": text})
            st["messages"].append({"role": "user", "content": f"Tìm kiếm này đã gọi rồi (bỏ dấu không đổi kết quả). Dùng tool khác hoặc gọi {ASK_TOOL}."})
            return
        st["done"] = True; return

    ask = next((c for c in fresh if c["name"] == ASK_TOOL), None)
    if ask and not st["ask_guarded"] and _single_hit(st["last_result"]) and all(str(p).endswith("_id") for p in ask["arguments"].get("params") or []):
        # Run v2 (train_0018): contacts.search trả đúng 1 người nhưng model vẫn hỏi contact_id
        st["ask_guarded"] = True
        st["messages"].append({"role": "assistant", "content": text})
        st["messages"].append({"role": "user", "content": f"{ASK_TOOL} CHƯA được thực thi: kết quả tool vừa rồi có đúng một kết quả, ID cần dùng nằm trong đó. Hãy gọi tool cần thiết với ID đó. Nếu thật sự vẫn thiếu thông tin thì gọi lại {ASK_TOOL}."})
        return

    st["messages"].append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in fresh]})
    for c in fresh:
        res = st["env"].call(c["name"], c["arguments"])
        st["last_result"] = res
        st["seen"] += json.dumps(res, ensure_ascii=False)
        content = _short(res, CFG["TOOL_RESULT_MAX_CHARS"])
        if c["name"].endswith(".search") and _no_hit(res) and not st["search_hinted"]:
            st["search_hinted"] = True
            content += "\n(Không tìm thấy. Tên có thể bị ASR nghe sai: tìm lại MỘT lần bằng phần tên chính có dấu, bỏ chức danh như cô/chú/sếp."
            if "banking.list_beneficiaries" in st["allowed"]:
                content += " Người nhận chuyển khoản ngân hàng nằm trong banking.list_beneficiaries, không phải danh bạ."
            content += f" Vẫn không có thì gọi {ASK_TOOL}.)"
        st["messages"].append({"role": "tool", "content": content})
        st["log"].append((c, content))
    if deferred_ask and _keep_deferred_ask(fresh, st["last_result"]):
        ask, err = check_call(deferred_ask, st["env"], st["allowed"], st["sample"]["context"].get("user_profile"))
        if not err:
            st["messages"].append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": ask["name"], "arguments": ask["arguments"]}}]})
            st["messages"].append({"role": "tool", "content": _short(st["env"].call(ask["name"], ask["arguments"]), CFG["TOOL_RESULT_MAX_CHARS"])})
            fresh.append(ask)
    elif deferred_ask and _multi_search(fresh, st["last_result"]):
        st["pending_ask"] = deferred_ask
    if any(c["name"] == ASK_TOOL for c in fresh) or st["steps"] >= CFG["MAX_STEPS"]:
        st["done"] = True


VERIFY_PROMPT = """Bạn là bộ kiểm tra. Một trợ lý đã gọi tool để làm theo yêu cầu giọng nói. Với MỖI call ghi (đánh số), điền đúng khuôn sau, chỉ TRÍCH NGUYÊN VĂN từ câu người dùng, không tự diễn giải:
- "so": số thứ tự call.
- "nguoi_dung_noi": đoạn trong câu người dùng yêu cầu việc này.
- "loai_tru": đoạn trong câu người dùng loại trừ hoặc huỷ ĐÚNG đối tượng của call này (có chữ "trừ", "ngoại trừ", "khỏi", "thôi", "đừng", "không cần"). Không có thì "".
- "dieu_kien": đoạn "nếu ..." / "mà ... thì" áp cho call này. Không có thì "".
- "dieu_kien_dung": true/false theo kết quả tool đọc; null nếu không có điều kiện.
Các trường hợp mẫu:
1. "Tắt hết đèn tầng 2, trừ phòng nào đang bật điều hoà" + call tắt đèn phòng ngủ, kết quả đọc cho thấy điều hoà phòng ngủ mode "cool" -> loai_tru = "trừ phòng nào đang bật điều hoà".
2. "Bật đèn phòng khách 40 phần trăm" + call bật đèn phòng khách, đèn đang tắt -> loai_tru = "", dieu_kien = "" (đèn đang tắt thì càng phải bật).
3. "Nếu ví đủ thì trả tiền điện bằng ví" + số dư ví nhỏ hơn hoá đơn -> dieu_kien = "Nếu ví đủ", dieu_kien_dung = false.
4. "Nhắn Lan là anh tới trễ, à thôi khỏi nhắn" + call sms.send -> loai_tru = "à thôi khỏi nhắn".
5. "Đặt báo thức 6 giờ rồi gọi cho Minh" + call alarm.create -> loai_tru = "", dieu_kien = "".
Phần lớn call là đúng: chỉ điền loai_tru hoặc dieu_kien khi câu người dùng THẬT SỰ có đoạn đó cho đúng đối tượng của call.
Trả lời đúng một dòng JSON: {"kiem": [{"so": 1, "nguoi_dung_noi": "...", "loai_tru": "", "dieu_kien": "", "dieu_kien_dung": null}, ...]}"""
_EXCLUDE_CUE = re.compile(r"\b(trừ|ngoại trừ|khỏi|thôi|đừng|không cần|bỏ qua)\b", re.I)
_COND_CUE = re.compile(r"\b(nếu|hễ|lỡ|trường hợp)\b", re.I)
SP_VERIFY = SamplingParams(temperature=0.0, max_tokens=CFG["VERIFY_MAX_TOKENS"], seed=CFG["SEED"])


def _write_calls(calls):
    return [i for i, c in enumerate(calls) if TOOL_SPEC.get(c["name"], {}).get("kind") == "write"]


def verify_prompt(st, calls, writes):
    s = st["sample"]
    hist, pending = [], []
    for m in history_messages(s):
        if m.get("tool_calls"):
            pending = [t["function"] for t in m["tool_calls"]]
        elif m["role"] == "tool":
            f = pending.pop(0) if pending else {"name": "?", "arguments": {}}
            hist.append(f"Trợ lý gọi {f['name']} {json.dumps(f['arguments'], ensure_ascii=False)} -> {_short(m['content'], CFG['VERIFY_RESULT_CHARS'])}")
        elif m.get("content"):
            hist.append(f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content']}")
    reads = [f"- {c['name']} {json.dumps(c['arguments'], ensure_ascii=False)}\n  -> {_short(r, CFG['VERIFY_RESULT_CHARS'])}"
             for c, r in st["log"] if TOOL_SPEC.get(c["name"], {}).get("kind") == "read"]
    numbered = [f"{n}. {calls[i]['name']} {json.dumps(calls[i]['arguments'], ensure_ascii=False)}" for n, i in enumerate(writes, 1)]
    body = (f"Thời gian hiện tại: {s['context'].get('current_time')}\n"
            + ("Lịch sử hội thoại:\n" + "\n".join(hist[-14:]) + "\n\n" if hist else "")
            + build_user_turn(st["query"], s["context"].get("current_time"))
            + "\n\nCác tool đọc đã gọi và kết quả:\n" + ("\n".join(reads) if reads else "(không có)")
            + "\n\nCác call ghi cần kiểm:\n" + "\n".join(numbered))
    return render([{"role": "system", "content": VERIFY_PROMPT}, {"role": "user", "content": body}])


def _in_query(quote, query):
    norm = lambda s: " ".join(re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFC", str(s)).lower()).split())
    q = norm(quote)
    return len(q) >= 3 and q in norm(query)


def _parse_drop(text, n, query):
    """Model chỉ điền khuôn trích dẫn; quyết định bỏ do code: cụm trích phải có nguyên văn trong câu người dùng và chứa từ
    khoá loại trừ, hoặc là điều kiện mà model đánh giá sai. Run rb6v: để model tự quyết thì nó bỏ call 1 ở hầu hết sample."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        items = json.loads(m.group(0)).get("kiem") if m else None
    except (json.JSONDecodeError, AttributeError):
        return set()
    drop = set()
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not str(it.get("so", "")).strip().isdigit() or not 1 <= int(it["so"]) <= n:
            continue
        ex, cond = str(it.get("loai_tru") or ""), str(it.get("dieu_kien") or "")
        if ex and _EXCLUDE_CUE.search(ex) and _in_query(ex, query):
            drop.add(int(it["so"]))
        elif cond and it.get("dieu_kien_dung") is False and _COND_CUE.search(cond) and _in_query(cond, query):
            drop.add(int(it["so"]))
    return drop


def verify(states):
    """Vai thứ hai (cùng model): chỉ được bỏ call ghi sai rồi chạy lại các call còn lại trên ToolEnv mới để có hash mới.
    Bản gốc vẫn giữ trong base_calls/base_hash để so điểm."""
    jobs = []
    for st in states:
        calls = st["env"].get_tool_calls()
        st["base_calls"], st["base_hash"] = calls, st["env"].get_state_hash()
        st["final_calls"], st["final_hash"], st["verify_out"] = calls, st["base_hash"], None
        writes = _write_calls(calls)
        if writes:
            p = verify_prompt(st, calls, writes)
            if len(TOK(p).input_ids) <= CFG["MAX_MODEL_LEN"] - CFG["VERIFY_MAX_TOKENS"] - 16:
                jobs.append((st, calls, writes, p))
    if not jobs:
        return
    for (st, calls, writes, _), o in zip(jobs, llm.generate([j[3] for j in jobs], SP_VERIFY, use_tqdm=False)):
        text = o.outputs[0].text
        st["verify_out"] = text
        drop = {writes[n - 1] for n in _parse_drop(text, len(writes), st["query"])}
        if not drop:
            continue
        try:
            env2 = ToolEnv(SPLIT, st["id"], root=str(ROOT))
            for i, c in enumerate(calls):
                if i not in drop:
                    env2.call(c["name"], c["arguments"])
            st["final_calls"], st["final_hash"] = env2.get_tool_calls(), env2.get_state_hash()
        except Exception as e:  # replay hỏng thì giữ bản gốc, không để một sample làm hỏng cả chunk
            print("verify replay lỗi", st["id"], repr(e)[:200])


ASR_FIX_PROMPT = """Bạn tìm từ bị nghe nhầm trong bản chép giọng nói (ASR) tiếng Việt của một trợ lý điều khiển app. ASR hay nghe nhầm tên riêng, từ tiếng Anh và động từ thành từ khác có âm gần giống.
Làm theo đúng các bước:
B1. Xác định việc người dùng muốn làm và app đã cài nào làm được việc đó.
B2. Tìm từ trong câu làm việc đó trở nên vô lý với các app đã cài, hoặc tên người/địa điểm gần giống một tên trong danh sách "Tên đã biết".
B3. Chỉ sửa khi từ mới (a) có âm gần giống từ cũ VÀ (b) lấy đúng chữ từ danh sách "Tên đã biết" hoặc từ tên/mô tả app đã cài.
Các trường hợp mẫu:
1. "giặt câu này qua tiếng Anh", có app Dịch thuật -> [["giặt", "dịch"]] (không thể giặt một câu; có app dịch).
2. "cất cuốn sách này lên kệ" -> [] ("cất" hợp lý, âm khác hẳn "dịch").
3. "nhắn cho Lâm", Tên đã biết có "Lan" và không có "Lâm" -> [["Lâm", "Lan"]].
4. "vay mấy giờ chạy", có app Vé tàu/xe khách -> [["vay", "tàu"]].
5. "nhắn ít đêm ít cho Hương", có app Tin nhắn SMS -> [["ít đêm ít", "SMS"]].
6. "bật đèn phòng khách" -> [] (câu đã hợp lý).
Không đổi số, giờ, ngày; không viết lại cả câu; giữ từ địa phương (mô, răng, rứa...). Phần lớn câu không cần sửa.
Trả lời đúng một dòng JSON: {"viec": "<việc người dùng muốn>", "sua": [["từ cũ", "từ mới"], ...]}"""
SP_ASR_FIX = SamplingParams(temperature=0.0, max_tokens=CFG["ASR_FIX_MAX_TOKENS"], seed=CFG["SEED"])


def _known_names(s, limit=80):
    """Chuỗi ngắn trong context + lịch sử (tên người, địa điểm, tiêu đề) mà agent được phép thấy."""
    out = []

    def walk(v):
        if isinstance(v, dict):
            for k, x in v.items():
                if k not in ("id", "time", "timezone", "current_time", "audio") and not k.endswith("_id"):
                    walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str) and 2 <= len(v) <= 60 and re.search(r"[A-ZÀ-Ỹ]", v[:1]) and not re.fullmatch(r"[\w-]*\d[\w-]*", v):
            out.append(v)
    walk(s["context"].get("user_profile") or {})
    for t in s["history"]:
        walk(t.get("tool_results") or [])
        walk(t.get("tool_calls") or [])
    return list(dict.fromkeys(out))[:limit]


def asr_fix_prompt(st):
    s = st["sample"]
    apps = [f"- {APP_INFO[a]['name_vi']}: {APP_INFO[a]['description']}" for a in s["context"]["installed_apps"] if a in APP_INFO]
    hist = [m["content"] for m in history_messages(s) if m["role"] == "assistant" and m.get("content")]
    names = _known_names(s)
    body = ("App đã cài:\n" + "\n".join(apps)
            + ("\n\nTên đã biết (danh bạ, địa điểm, tiêu đề trong lịch sử):\n" + "; ".join(names) if names else "")
            + ("\n\nCâu trợ lý vừa nói:\n" + "\n".join(hist[-3:]) if hist else "")
            + f'\n\nBản chép ASR cần kiểm: "{st["query"]}"')
    return render([{"role": "system", "content": ASR_FIX_PROMPT}, {"role": "user", "content": body}])


def _fix_vocab(s):
    apps = [APP_INFO[a] for a in s["context"]["installed_apps"] if a in APP_INFO]
    text = " ".join(_known_names(s) + [a["name_vi"] + " " + a["description"] for a in apps]
                    + [t.get("description", "") if isinstance(t, dict) else TOOL_SPEC.get(t, {}).get("description", "") for a in apps for t in a["tools"]])
    return set(re.findall(r"\w+", unicodedata.normalize("NFC", text.lower())))


def apply_fixes(raw, pairs, vocab, max_pairs):
    """Chỉ thay khi từ cũ có trong câu và mọi chữ của từ mới có trong tên đã biết / từ vựng app đã cài, để model không bịa từ."""
    out, n = raw, 0
    for p in pairs if isinstance(pairs, list) else []:
        if n >= max_pairs or not (isinstance(p, (list, tuple)) and len(p) == 2):
            continue
        old, new = (unicodedata.normalize("NFC", str(x)).strip() for x in p)
        old_toks, new_toks = re.findall(r"\w+", old.lower()), re.findall(r"\w+", new.lower())
        if not old_toks or not new_toks or old.lower() == new.lower() or any(t not in vocab for t in new_toks):
            continue
        # số và giờ đã có bước chuẩn hoá riêng; model đổi "hai" -> "ba" thì sai nghĩa hẳn
        if re.search(r"\d", old + new) or any(t in _NUMWORDS for t in old_toks + new_toks):
            continue
        # nghe nhầm giữ số âm tiết ("giặt"/"dịch", "Lâm"/"Lan"); ngoại lệ là từ viết tắt tiếng Anh ("ít đêm ít" -> "SMS")
        acronym = len(new_toks) == 1 and new.isascii() and new.isupper()
        if (acronym and len(old_toks) < 2) or (len(old_toks) != len(new_toks) and not acronym):
            continue
        pat = re.compile(rf"(?<!\w){re.escape(old)}(?!\w)", re.I)
        if pat.search(out):
            out, n = pat.sub(new, out, count=1), n + 1
    return out


def asr_fix(states):
    """Vai sửa lời nghe (cùng model): model chỉ chỉ ra cặp [từ cũ, từ mới]; code kiểm từ vựng rồi tự thay.
    Agent chính nhận câu đã sửa kèm câu ASR gốc."""
    jobs = [st for st in states if not st["done"] and st["query"].strip()]
    for st, o in zip(jobs, llm.generate([asr_fix_prompt(st) for st in jobs], SP_ASR_FIX, use_tqdm=False)):
        text = o.outputs[0].text
        st["asr_fix_out"] = text
        m = re.search(r"\{.*\}", text, re.S)
        try:
            pairs = json.loads(m.group(0)).get("sua") if m else None
        except (json.JSONDecodeError, AttributeError):
            pairs = None
        raw = st["query"]
        fixed = apply_fixes(raw, pairs, _fix_vocab(st["sample"]), CFG["ASR_FIX_MAX_PAIRS"])
        if fixed == raw:
            continue
        st["asr_raw"], st["query"] = raw, fixed
        turn = build_user_turn(fixed, st["sample"]["context"].get("current_time"))
        st["messages"][-1]["content"] = turn.replace("Yêu cầu (bản chép ASR, có thể có lỗi)", "Yêu cầu (đã sửa lỗi nghe, có thể còn sai)", 1) + f'\nBản chép ASR gốc: "{raw}"'


def run_chunk(samples):
    states = [init_state(s) for s in samples]
    if CFG["USE_ROUTER"]:
        route(states)
    if CFG["ASR_FIX"]:
        asr_fix(states)
    while True:
        prompts, runnable = [], []
        for st in states:
            if st["done"]:
                continue
            p = _fit_context(st)
            if p is None:
                st["done"] = True; continue
            prompts.append(p); runnable.append(st)
        if not runnable:
            break
        for st, o in zip(runnable, llm.generate(prompts, SP_AGENT, use_tqdm=False)):
            agent_step(st, o.outputs[0].text)
    if CFG["VERIFY"]:
        verify(states)
    return states


def save_chunk(states):
    with open(RES_FILE, "a", encoding="utf-8") as fr, open(TRACE_FILE, "a", encoding="utf-8") as ft:
        for st in states:
            base_calls, base_hash = st["env"].get_tool_calls(), st["env"].get_state_hash()
            row = {"id": st["id"], "run_tag": CFG["RUN_TAG"], "tool_calls": st.get("final_calls", base_calls), "state_hash": st.get("final_hash", base_hash),
                   "base_calls": base_calls, "base_hash": base_hash,
                   "route": st["route"], "steps": st["steps"], "retries": st["retries"], "query": st["query"], "asr_raw": st.get("asr_raw")}
            DONE[st["id"]] = row
            fr.write(json.dumps(row, ensure_ascii=False) + "\n")
            ft.write(json.dumps({"id": st["id"], "llm_outputs": st["trace"], "verify_out": st.get("verify_out"), "asr_fix_out": st.get("asr_fix_out")}, ensure_ascii=False, default=str) + "\n")


budget_s = CFG["TIME_BUDGET_H"] * 3600
chunks = [] if CFG["JEV_PROBE"] else [TODO[i: i + CFG["CHUNK_SIZE"]] for i in range(0, len(TODO), CFG["CHUNK_SIZE"])]
for ci, chunk in enumerate(chunks):
    elapsed = time.time() - T_START
    if elapsed > budget_s:
        print(f"Hết ngân sách thời gian ({elapsed / 3600:.2f}h); dừng ở chunk {ci}/{len(chunks)}")
        break
    t0 = time.time()
    save_chunk(run_chunk(chunk))
    print(f"chunk {ci + 1}/{len(chunks)}: {len(chunk)} sample, {time.time() - t0:.0f}s | xong {len(DONE)}/{len(SAMPLES)} | tổng {(time.time() - T_START) / 3600:.2f}h", flush=True)
if CFG["JEV_PROBE"]:
    run_probe()

# %% [markdown]
# ## 11. Ghi submission (khi đủ mọi sample) và kiểm bằng validator của BTC

# %%
missing = [s["id"] for s in SAMPLES if s["id"] not in DONE]
sub_path = OUT_DIR / "submission.jsonl"
if missing:
    print(f"CHƯA ĐỦ: còn {len(missing)}/{len(SAMPLES)} sample. Lưu Version, rồi chạy lần sau với output này gắn làm Input để chạy tiếp.")
    if sub_path.exists():
        sub_path.unlink()
else:
    with open(sub_path, "w", encoding="utf-8") as f, open(OUT_DIR / "submission_base.jsonl", "w", encoding="utf-8") as fb:
        for s in SAMPLES:
            row = DONE[s["id"]]
            f.write(json.dumps({"id": s["id"], "tool_calls": row["tool_calls"], "state_hash": row["state_hash"]}, ensure_ascii=False) + "\n")
            fb.write(json.dumps({"id": s["id"], "tool_calls": row.get("base_calls", row["tool_calls"]),
                                 "state_hash": row.get("base_hash", row["state_hash"])}, ensure_ascii=False) + "\n")
    print(sub_path, "|", len(SAMPLES), "dòng | verifier đổi", sum(DONE[s["id"]].get("base_hash", DONE[s["id"]]["state_hash"]) != DONE[s["id"]]["state_hash"] for s in SAMPLES), "sample")
    print(Counter("oos" if not DONE[s["id"]]["tool_calls"] else ("ask" if any(c["name"] == ASK_TOOL for c in DONE[s["id"]]["tool_calls"]) else "act") for s in SAMPLES))
    r = subprocess.run([sys.executable, "-m", "zaic_tools", "--split", SPLIT, "--root", "..", "--validate", str(sub_path)],
                       cwd=SPLIT_DIR, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    print(r.stdout[-2000:], r.stderr[-2000:])

# %% [markdown]
# ## 12. Chấm cục bộ (split `train`)
# Theo công thức trong rule: gold rỗng → 1 khi nộp rỗng; sai hash → 0; còn lại F1 với P = M/(M+U), R = M/|G|. Tham số `x-match: fuzzy` so bằng token-F1 > 0.6 sau lowercase + NFC. Hash gold tính bằng cách chạy lại call gold trên một `ToolEnv` mới; việc này chỉ để chấm cục bộ.

# %%
def _norm(s):
    return unicodedata.normalize("NFC", str(s)).lower()


def token_f1(a, b):
    ta, tb = _norm(a).split(), _norm(b).split()
    if not ta or not tb:
        return float(ta == tb)
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def call_match(pred, gold):
    if pred.get("name") != gold.get("name"):
        return False
    pa, ga = pred.get("arguments") or {}, gold.get("arguments") or {}
    if set(pa) != set(ga):
        return False
    return all(token_f1(pa[k], ga[k]) > 0.6 if (gold["name"], k) in FUZZY else pa[k] == ga[k] for k in ga)


def sample_score(pred_calls, gold_calls, pred_hash, gold_hash):
    if not gold_calls:
        return 1.0 if not pred_calls else 0.0
    if pred_hash != gold_hash:
        return 0.0
    used, m = set(), 0
    for g in gold_calls:
        for i, p in enumerate(pred_calls):
            if i not in used and call_match(p, g):
                used.add(i); m += 1; break
    if m == 0:
        return 0.0
    prec, rec = m / len(pred_calls), m / len(gold_calls)
    return 2 * prec * rec / (prec + rec)


if SPLIT == "train":
    rows = []
    for s in SAMPLES:
        if s["id"] not in DONE:
            continue
        done = DONE[s["id"]]
        gold_calls = s["label"]["tool_calls"]
        genv = ToolEnv(SPLIT, s["id"], root=str(ROOT))
        for c in gold_calls:
            genv.call(c["name"], c["arguments"])
        sc = sample_score(done["tool_calls"], gold_calls, done["state_hash"], genv.get_state_hash())
        sc_base = sample_score(done.get("base_calls", done["tool_calls"]), gold_calls, done.get("base_hash", done["state_hash"]), genv.get_state_hash())
        kind = "oos" if not gold_calls else ("ask" if any(c["name"] == ASK_TOOL for c in gold_calls) else "act")
        rows.append({"id": s["id"], "kind": kind, "score": sc, "score_base": sc_base, "hash_ok": done["state_hash"] == genv.get_state_hash(),
                     "route": done["route"], "pred": done["tool_calls"], "gold": gold_calls, "query": done["query"], "audio_text": s["audio_text"]})
    (OUT_DIR / "train_eval.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
if SPLIT == "train" and rows:
    print(f"Điểm cục bộ: {100 * sum(r['score'] for r in rows) / len(rows):.2f} trên {len(rows)} sample"
          f" | trước verifier {100 * sum(r['score_base'] for r in rows) / len(rows):.2f}"
          f" | verifier lên {sum(r['score'] > r['score_base'] for r in rows)} / xuống {sum(r['score'] < r['score_base'] for r in rows)}")
    for kind in ("act", "ask", "oos"):
        ks = [r for r in rows if r["kind"] == kind]
        if ks:
            print(f"  {kind}: {100 * sum(r['score'] for r in ks) / len(ks):.2f} (n={len(ks)}, hash đúng {sum(r['hash_ok'] for r in ks)})")
    print("\nSample sai:")
    for r in [r for r in rows if r["score"] < 1][:8]:
        print(f"\n{r['id']} [{r['kind']}] score={r['score']:.2f} route={r['route']}\n  ASR : {r['query']}\n  text: {r['audio_text']}\n  pred: {_fmt_calls(r['pred'])}\n  gold: {_fmt_calls(r['gold'])}")

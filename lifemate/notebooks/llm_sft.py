# %% [markdown]
# # LifeMate: SFT LoRA cho LLM gọi tool
# Train LoRA trên hội thoại gold của train (bỏ 200 sample dev) do `scripts/make_sft.py` dựng bằng chính hàm prompt của notebook `zalo-ai`, nên lúc chạy thật model thấy đúng định dạng đã học. Loss chỉ tính trên các lượt assistant của lượt cuối (tool call và DONE).
#
# Chạy DDP trên 2 GPU T4 bằng `torchrun`. Xong thì merge LoRA vào model gốc, lưu ở `/kaggle/working/<OUT_NAME>` để notebook `zalo-ai` gắn làm Input (`LLM = "input:<OUT_NAME>"`).

# %%
!pip install -q "transformers==4.56.2" "huggingface_hub<1.0" "peft==0.17.1" accelerate

# %%
import json, os
from pathlib import Path

CFG = {
    "INPUT_ROOT": "/kaggle/input",
    "WORK_DIR": "/kaggle/working",
    "MODEL": "Qwen/Qwen3-4B-Instruct-2507",
    "SFT_FILE": "sft_train.jsonl",
    "OUT_NAME": "qwen3-4b-sft",
    "EPOCHS": 2,
    "LR": 1e-4,
    "GRAD_ACCUM": 8,                                    # mỗi GPU; batch hiệu dụng = 2 x GRAD_ACCUM
    "LORA_R": 32,
    "LORA_ALPHA": 64,
    "MAX_LEN": 10240,                                   # sft-smoke: OOM trên T4 ở chuỗi ~9.3K khi còn dropout; dài hơn bị bỏ (~5%)
    "WARMUP": 10,
    "TRAIN_BUDGET_H": 8.0,
    "LIMIT": None,                                      # chạy thử: số hội thoại DÀI NHẤT; None = tất cả
    "SEED": 42,
}
WORK = Path(CFG["WORK_DIR"])
sft_path = next(Path(CFG["INPUT_ROOT"]).rglob(CFG["SFT_FILE"]))
print("SFT data:", sft_path)
(WORK / "sft_cfg.json").write_text(json.dumps({**CFG, "SFT_PATH": str(sft_path)}), encoding="utf-8")

# %%
%%writefile train_sft.py
import json, math, os, random, sys, time
import torch, torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

CFG = json.load(open("/kaggle/working/sft_cfg.json"))
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
random.seed(CFG["SEED"]); torch.manual_seed(CFG["SEED"])
log = lambda *a: rank == 0 and print(*a, flush=True)

tok = AutoTokenizer.from_pretrained(CFG["MODEL"])
HEAD = "<|im_start|>assistant\n"
END = "<|im_end|>"


def encode(row):
    """Token hoá cả hội thoại, nhãn chỉ ở các lượt assistant sau phần prompt (bỏ lượt assistant trong history)."""
    text = tok.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False)
    n_hist = sum(m["role"] == "assistant" for m in row["messages"][: row["n_prompt"]])
    spans, pos, k = [], 0, 0
    while (i := text.find(HEAD, pos)) >= 0:
        j = text.find(END, i) + len(END)
        if k >= n_hist:
            spans.append((i + len(HEAD), j))
        pos, k = j, k + 1
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    labels = [t if any(a <= s < b for a, b in spans) else -100 for t, (s, _) in zip(enc.input_ids, enc.offset_mapping)]
    return enc.input_ids, labels


rows = [json.loads(l) for l in open(CFG["SFT_PATH"], encoding="utf-8")]
data = [encode(r) for r in rows]
n_all = len(data)
data = [d for d in data if len(d[0]) <= CFG["MAX_LEN"] and any(l != -100 for l in d[1])]
if CFG["LIMIT"]:
    # chạy thử lấy các chuỗi dài nhất để lộ OOM ngay, không phải đợi tới giữa epoch
    data = sorted(data, key=lambda d: -len(d[0]))[: CFG["LIMIT"]]
lens = sorted(len(d[0]) for d in data)
log(f"{len(data)}/{n_all} hội thoại | token median {lens[len(lens) // 2]}, max {lens[-1]} | nhãn tb {sum(sum(l != -100 for l in d[1]) for d in data) / len(data):.0f}")

model = AutoModelForCausalLM.from_pretrained(CFG["MODEL"], torch_dtype=torch.float16, attn_implementation="sdpa").cuda(rank)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
# lora_dropout > 0 giữ thêm một bản input cho mỗi lớp LoRA để backward, không đủ VRAM T4 với chuỗi dài
model = get_peft_model(model, LoraConfig(r=CFG["LORA_R"], lora_alpha=CFG["LORA_ALPHA"], lora_dropout=0.0, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
# T4 không có bf16: trọng số LoRA để fp32 cho GradScaler, model gốc giữ fp16
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
if rank == 0:
    model.print_trainable_parameters()


class MaskedLM(torch.nn.Module):
    """Chỉ đưa qua lm_head các vị trí có nhãn: logits cả chuỗi 12K x 151K vocab không vừa VRAM T4."""

    def __init__(self, peft_model):
        super().__init__()
        self.m = peft_model

    def forward(self, ids, labels):
        base = self.m.base_model.model
        h = base.model(input_ids=ids).last_hidden_state[0, :-1]
        tgt = labels[0, 1:]
        keep = tgt != -100
        logits = base.lm_head(h[keep]).float()
        return torch.nn.functional.cross_entropy(logits, tgt[keep])


net = torch.nn.parallel.DistributedDataParallel(MaskedLM(model), device_ids=[rank])
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=CFG["LR"], weight_decay=0.0)
steps_per_epoch = len(data) // (world * CFG["GRAD_ACCUM"])
total = CFG["EPOCHS"] * steps_per_epoch
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / CFG["WARMUP"]) * 0.5 * (1 + math.cos(math.pi * min(s, total) / max(total, 1))))
scaler = torch.amp.GradScaler("cuda")
net.train()
t0, step, stop = time.time(), 0, False
for ep in range(CFG["EPOCHS"]):
    order = list(range(len(data)))
    random.Random(CFG["SEED"] + ep).shuffle(order)
    mine = order[rank::world][: steps_per_epoch * CFG["GRAD_ACCUM"]]
    run = 0.0
    for i, idx in enumerate(mine):
        ids, lab = (torch.tensor([x], device=rank) for x in data[idx])
        sync = (i + 1) % CFG["GRAD_ACCUM"] == 0
        with net.no_sync() if not sync else torch.enable_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                loss = net(ids, lab) / CFG["GRAD_ACCUM"]
            scaler.scale(loss).backward()
        run += loss.item()
        if sync:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
            if step % 5 == 0:
                log(f"ep {ep} step {step}/{total} loss {run / 5:.4f} | {(time.time() - t0) / 60:.1f} phút")
                run = 0.0
            if time.time() - t0 > CFG["TRAIN_BUDGET_H"] * 3600:
                log("hết TRAIN_BUDGET_H, dừng train"); stop = True
        # mọi rank phải dừng cùng bước, không thì all-reduce của rank kia treo
        flag = torch.tensor([int(stop)], device=rank); dist.all_reduce(flag)
        if flag.item():
            break
    if stop:
        break
log(f"train xong {step} bước, {(time.time() - t0) / 60:.1f} phút")
if rank == 0:
    model.save_pretrained("/kaggle/working/sft_adapter")
dist.barrier()
dist.destroy_process_group()

# %%
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=2 train_sft.py

# %% [markdown]
# ## Merge LoRA vào model gốc (fp16) để vLLM nạp như model thường

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(CFG["MODEL"], torch_dtype=torch.float16)
merged = PeftModel.from_pretrained(base, str(WORK / "sft_adapter")).merge_and_unload()
out = WORK / CFG["OUT_NAME"]
merged.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(CFG["MODEL"]).save_pretrained(out)
print("merged ->", out, sorted(p.name for p in out.iterdir()))

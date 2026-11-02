"""Replay rb4 agent outputs, then run verify() with a scripted verifier reply to exercise prompt + replay paths.

  REPLY='{"kiem": []}' .venv/bin/python scripts/dev/verify_smoke.py [id ...]
  ASRFIX=1 .venv/bin/python scripts/dev/verify_smoke.py <id> ...
Needs data/lifemate and output/sft1-gold-rb4/checkpoint/train_trace.jsonl locally.
"""
import json, re, sys, types, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import make_sft

g = make_sft.load_notebook(ROOT / "data" / "lifemate", 0)
src = (ROOT / "notebooks" / "lifemate_pipeline.py").read_text(encoding="utf-8")
cells = [c.partition("\n")[2] for c in re.split(r"^# %%", src, flags=re.M)[1:] if not c.startswith(" [markdown]")]
start = next(i for i, c in enumerate(cells) if "from vllm import" in c)
agent = next(i for i, c in enumerate(cells) if "def agent_step" in c)
REPLY = os.environ.get("REPLY", '{"ly_do": "thử", "bo": [1]}')


class Tok:
    def apply_chat_template(self, messages, tools=None, **kw):
        return "\n".join(m.get("content") or "" for m in messages)

    def __call__(self, t, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[0] * (len(t) // 4))


class FakeLLM:
    def __init__(self, **kw): pass
    def get_tokenizer(self): return Tok()
    def generate(self, prompts, sp=None, use_tqdm=False):
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=REPLY, logprobs=None)]) for _ in prompts]


sys.modules["vllm"] = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **k: None, __version__="0.10.2")
g["torch"].cuda.get_device_capability = lambda i: (7, 5)
for c in cells[start:agent]:
    exec(c, g)
exec(cells[agent].split("\nbudget_s = ")[0], g)

if os.environ.get("ASRFIX"):
    by = {s["id"]: s for s in g["SAMPLES"]}
    for sid in sys.argv[1:]:
        g["TRANSCRIPTS"][sid] = by[sid]["audio_text"]
        st = g["init_state"](by[sid])
        print(g["asr_fix_prompt"](st))
        g["asr_fix"]([st])
        print("---> query:", st["query"], "| raw:", st.get("asr_raw"))
        print("---> user turn:", st["messages"][-1]["content"])
        print("names:", g["_known_names"](by[sid]))
    sys.exit()
T = {r["id"]: r for r in map(json.loads, open(ROOT / "output/sft1-gold-rb4/checkpoint/train_trace.jsonl"))}
ids = sys.argv[1:] or list(T)
by = {s["id"]: s for s in g["SAMPLES"]}
states = []
for sid in ids:
    s = by[sid]
    g["TRANSCRIPTS"][sid] = s["audio_text"]
    st = g["init_state"](s)
    for out in T[sid]["llm_outputs"]:
        if st["done"]:
            break
        g["agent_step"](st, out)
    states.append(st)
g["verify"](states)
changed = 0
for st in states:
    if st["final_hash"] != st["base_hash"]:
        changed += 1
    if len(ids) <= 3:
        calls, writes = st["base_calls"], g["_write_calls"](st["base_calls"])
        if writes:
            print(g["verify_prompt"](st, calls, writes))
        print(st["id"], "base", [c["name"] for c in st["base_calls"]], "\n   final", [c["name"] for c in st["final_calls"]])
print(len(states), "sample,", sum(bool(g["_write_calls"](st["base_calls"])) for st in states), "có call ghi,", changed, "đổi hash")

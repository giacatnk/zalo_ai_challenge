"""Replay recorded LLM outputs through the current agent_step (stub vLLM, real ToolEnv).

  RUN=sft1-gold-rb2 .venv/bin/python scripts/dev/barrier_smoke.py <id ...|all>
Needs data/lifemate and output/$RUN/checkpoint/train_trace.jsonl locally.
"""
import json, os, re, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import make_sft

g = make_sft.load_notebook(ROOT / "data" / "lifemate", 0)
src = (ROOT / "notebooks" / "lifemate_pipeline.py").read_text(encoding="utf-8")
cells = [c.partition("\n")[2] for c in re.split(r"^# %%", src, flags=re.M)[1:] if not c.startswith(" [markdown]")]
start = next(i for i, c in enumerate(cells) if "from vllm import" in c)
agent = next(i for i, c in enumerate(cells) if "def agent_step" in c)


class Tok:
    def apply_chat_template(self, messages, tools=None, **kw):
        return json.dumps(messages, ensure_ascii=False)[-3000:]

    def __call__(self, t, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[0] * (len(t) // 4))


class FakeLLM:
    def __init__(self, **kw): pass
    def get_tokenizer(self): return Tok()
    def generate(self, prompts, sp=None, use_tqdm=False):
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="DONE", logprobs=None)]) for _ in prompts]


sys.modules["vllm"] = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **k: None, __version__="0.10.2")
g["torch"].cuda.get_device_capability = lambda i: (7, 5)
for c in cells[start:agent]:
    exec(c, g)
exec(cells[agent].split("\ndef run_chunk")[0], g)

ids = sys.argv[1:]
T = {r["id"]: r for r in map(json.loads, open(ROOT / f"output/{os.environ.get('RUN', 'sft1-gold-rb2')}/checkpoint/train_trace.jsonl"))}
if ids == ["all"]:
    ids = list(T)
by = {s["id"]: s for s in g["SAMPLES"]}
for sid in ids:
    s = by[sid]
    g["TRANSCRIPTS"][sid] = s["audio_text"]
    st = g["init_state"](s)
    for out in T[sid]["llm_outputs"]:
        if st["done"]:
            break
        g["agent_step"](st, out)
    print(sid, [c["name"] for c in st["env"].get_tool_calls()], "done" if st["done"] else "continues")

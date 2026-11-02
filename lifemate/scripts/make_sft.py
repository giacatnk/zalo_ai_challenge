"""Tạo dữ liệu SFT (hội thoại gọi tool nhiều bước) từ gold của train, bỏ 200 sample dev.

Dựng prompt bằng chính các hàm của notebook pipeline (build_system, history_messages, tools_for_sample)
để lúc train và lúc chạy thật model thấy cùng một định dạng.

Dùng:
  .venv/bin/python scripts/make_sft.py data/lifemate data/sft/sft_train.jsonl [--fewshot-k 0]
"""
import argparse
import json
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "lifemate_pipeline.py"


def load_notebook(data_root, fewshot_k):
    """Chạy các ô của notebook tới trước ô nạp vLLM, trả về namespace chứa các hàm dựng prompt."""
    src = NB.read_text(encoding="utf-8")
    cells = [c.partition("\n")[2] for c in re.split(r"^# %%", src, flags=re.M)[1:] if not c.startswith(" [markdown]")]
    # máy local không có torch/vllm; các ô dựng prompt chỉ cần torch.manual_seed
    sys.modules["torch"] = types.SimpleNamespace(manual_seed=lambda x: None, Tensor=type("Tensor", (), {}),
                                                 cuda=types.SimpleNamespace(device_count=lambda: 0))
    g = {"__name__": "nb"}
    for c in cells:
        if "!pip" in c or c.lstrip().startswith("%%writefile"):
            continue
        if "from vllm import" in c:
            break
        c = (c.replace('"DATA_ROOT": None', f'"DATA_ROOT": {str(data_root)!r}')
              .replace('"WORK_DIR": "/kaggle/working"', '"WORK_DIR": "/tmp/lifemate_sft_work"')
              .replace('"INPUT_ROOT": "/kaggle/input"', '"INPUT_ROOT": "/tmp/lifemate_sft_input"')
              .replace('"TRANSCRIPT_SRC": "asr"', '"TRANSCRIPT_SRC": "gold"')
              .replace('"FEWSHOT_K": 3', f'"FEWSHOT_K": {fewshot_k}'))
        exec(c, g)
    return g


def _short(obj, n):
    # giống _short trong agent loop của notebook (ô đó nằm sau ô nạp vLLM nên không exec được ở đây)
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + "...(cắt bớt)"


def group_steps(calls, results, spec):
    """Gộp call liên tiếp vào cùng một lượt trừ khi call sau dùng giá trị chỉ xuất hiện trong kết quả của lượt hiện tại,
    hoặc call sau là ghi/ask mà lượt hiện tại đã có tool đọc (quyết định có thể phụ thuộc kết quả đọc)."""
    steps, cur, cur_out = [], [], ""
    for call, res in zip(calls, results):
        vals = [v for v in call["arguments"].values() if isinstance(v, (str, int)) and str(v)]
        after_read = spec.get(call["name"], {}).get("kind") != "read" and any(spec.get(c["name"], {}).get("kind") == "read" for c, _ in cur)
        if cur and (after_read or any(json.dumps(v, ensure_ascii=False) in cur_out for v in vals)):
            steps.append(cur); cur, cur_out = [], ""
        cur.append((call, res))
        cur_out += json.dumps(res, ensure_ascii=False)
    if cur:
        steps.append(cur)
    return steps


def build(g, sample):
    for k, turn in enumerate(sample["history"]):
        if turn.get("role") == "user" and turn.get("audio_text"):
            g["TRANSCRIPTS"][f"{sample['id']}#{k}"] = turn["audio_text"]
    query = sample["audio_text"]
    g["TRANSCRIPTS"][sample["id"]] = query
    tools = g["tools_for_sample"](sample)
    msgs = ([{"role": "system", "content": g["build_system"](sample, query)}] + g["history_messages"](sample)
            + [{"role": "user", "content": g["build_user_turn"](query, sample["context"].get("current_time"))}])
    n_prompt = len(msgs)
    calls, results = sample["label"]["tool_calls"], sample["label"]["tool_results"]
    for step in group_steps(calls, results, g["TOOL_SPEC"]):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c, _ in step]})
        for _, r in step:
            msgs.append({"role": "tool", "content": _short(r, g["CFG"]["TOOL_RESULT_MAX_CHARS"])})
    # ask_missing_info kết thúc lượt nên không có DONE sau nó
    if not calls or calls[-1]["name"] != g["ASK_TOOL"]:
        msgs.append({"role": "assistant", "content": "DONE"})
    elif msgs[-1]["role"] == "tool":
        msgs.pop()
    return {"id": sample["id"], "kind": g["kind_of"](sample), "tools": tools, "messages": msgs, "n_prompt": n_prompt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root")
    ap.add_argument("out")
    ap.add_argument("--fewshot-k", type=int, default=0)
    args = ap.parse_args()
    g = load_notebook(Path(args.data_root).resolve(), args.fewshot_k)
    pool = [s for s in g["TRAIN_ROWS"] if s["id"] not in g["DEV_IDS"]]
    g.setdefault("TRANSCRIPTS", {})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_steps, chars = [], []
    with out.open("w", encoding="utf-8") as f:
        for s in pool:
            row = build(g, s)
            n_steps.append(sum(m["role"] == "assistant" for m in row["messages"][row["n_prompt"]:]))
            chars.append(len(json.dumps(row, ensure_ascii=False)))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    chars.sort()
    print(f"{len(pool)} hội thoại -> {out}")
    print(f"lượt assistant/hội thoại: tb {sum(n_steps) / len(n_steps):.2f}, max {max(n_steps)}")
    print(f"ký tự: median {chars[len(chars) // 2]}, p95 {chars[int(len(chars) * .95)]}, max {chars[-1]}")


if __name__ == "__main__":
    main()

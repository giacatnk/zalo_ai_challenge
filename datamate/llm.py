"""One chat client for every LLM call, so the backend can be switched in one place.

Backends
  openrouter  development: OpenRouter cloud, key OPENROUTER_API_KEY from the environment or .env
  vllm        submission: local OpenAI-compatible vLLM server (no network in the final image)

The two differ in how they take a JSON schema and how Qwen3 thinking is switched off.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

DEFAULTS = {
    # one provider, no fallback: OpenRouter otherwise routes each call to a different host, and
    # some hosts return empty content with thinking + json_schema
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "qwen/qwen3-14b", "key_env": "OPENROUTER_API_KEY",
                   "provider": "DeepInfra"},
    "vllm": {"base_url": "http://localhost:8000/v1", "model": "datamate", "key_env": None},
}


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines, existing environment wins."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'"))


@dataclass
class LLM:
    backend: str = "openrouter"
    model: str | None = None
    base_url: str | None = None
    seed: int = 0
    timeout: float = 120
    retries: int = 4

    def __post_init__(self):
        cfg = DEFAULTS[self.backend]
        self.model = self.model or cfg["model"]
        self.base_url = self.base_url or cfg["base_url"]
        key = "EMPTY"
        if cfg["key_env"]:
            load_dotenv()
            key = os.environ.get(cfg["key_env"], "")
            if not key:
                raise RuntimeError(f"{cfg['key_env']} is not set (environment or .env)")
        self.client = OpenAI(base_url=self.base_url, api_key=key, timeout=self.timeout, max_retries=0)

    def _extra(self, schema: dict | None, think: bool, tools: bool = False) -> tuple[dict, dict]:
        """(top-level kwargs, extra_body) for this backend."""
        kw, extra = {}, {}
        if self.backend == "vllm":
            extra["chat_template_kwargs"] = {"enable_thinking": think}
            if schema:
                extra["guided_json"] = schema
        else:
            extra["reasoning"] = {"enabled": think} if think else {"enabled": False, "exclude": True}
            extra["provider"] = {"require_parameters": bool(schema) or tools}
            if DEFAULTS[self.backend].get("provider"):
                extra["provider"] |= {"order": [DEFAULTS[self.backend]["provider"]], "allow_fallbacks": False}
            if schema:
                kw["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "answer", "strict": True, "schema": schema}}
        return kw, extra

    def chat(self, messages: list[dict], *, schema: dict | None = None, think: bool = False,
             max_tokens: int = 1024, tools: list | None = None, temperature: float = 0,
             seed: int | None = None) -> dict:
        """Returns {"content", "tool_calls", "usage", "seconds", "finish", "provider"}. Retries rate
        limits and 5xx."""
        kw, extra = self._extra(schema, think, bool(tools))
        if schema:                             # the shape in words too: constrained decoding alone can run on blanks
            last = messages[-1]
            messages = messages[:-1] + [{**last, "content": f"{last['content']}\n\nTrả lời đúng một JSON dạng: "
                                         f"{json.dumps(skeleton(schema), ensure_ascii=False)}"}]
        if tools:
            kw["tools"] = tools
        last = None
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature,
                    seed=self.seed if seed is None else seed,
                    max_tokens=max_tokens, extra_body=extra, **kw)
                msg = resp.choices[0].message
                return {"content": msg.content or "", "tool_calls": msg.tool_calls,
                        "usage": resp.usage.model_dump() if resp.usage else None,
                        "seconds": round(time.time() - t0, 2), "finish": resp.choices[0].finish_reason,
                        "provider": getattr(resp, "provider", None)}
            except Exception as e:  # rate limit, upstream failure, timeout
                last = e
                status = getattr(e, "status_code", None)
                if status is not None and status < 500 and status != 429:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {self.retries} attempts: {last!r}")

    def json(self, prompt: str, schema: dict, **kw) -> tuple[dict, dict]:
        """One constrained call; returns (parsed JSON, raw response), {} when none can be read."""
        r = self.chat([{"role": "user", "content": prompt}], schema=schema, **kw)
        try:
            data = parse_json(r["content"])
        except ValueError:
            data = {}
        return data, r


def skeleton(schema: dict):
    """An example value of the schema's shape: keys kept, strings "...", arrays with one item."""
    t = schema.get("type")
    if "enum" in schema:
        return "|".join(map(str, schema["enum"]))
    if t == "object":
        return {k: skeleton(v) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [skeleton(schema.get("items", {}))]
    return {"string": "...", "number": 0, "integer": 0, "boolean": True}.get(t, "...")


def parse_json(text: str) -> dict:
    """The JSON object in a reply. A reply cut off by max_tokens (constrained decoding can run on
    whitespace until the limit) is read up to its last complete value, with the open arrays and
    objects closed. Raises ValueError when there is no object."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    text = text[start:]
    try:
        out = json.JSONDecoder().raw_decode(text)[0]
    except json.JSONDecodeError:
        out = None
        stack, in_str, esc, ends = [], False, False, []      # ends: (position after a value, open brackets)
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                    ends.append((i + 1, "".join(stack)))
            elif ch == '"':
                in_str = True
            elif ch in "[{":
                stack.append(ch)
            elif ch in "]}" and stack:
                stack.pop()
                ends.append((i + 1, "".join(stack)))
            elif ch.isalnum() and (i + 1 == len(text) or not text[i + 1].isalnum() and text[i + 1] != "."):
                ends.append((i + 1, "".join(stack)))   # end of a number, true, false, null
        for cut, open_ in reversed(ends[-200:]):
            tail = "".join("]" if c == "[" else "}" for c in reversed(open_))
            try:
                out = json.loads(text[:cut] + tail)
                break
            except json.JSONDecodeError:
                continue
        if out is None:
            raise ValueError("cut-off JSON could not be repaired")
    if not isinstance(out, dict):
        raise ValueError("reply is not a JSON object")
    return out

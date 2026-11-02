"""Question step (Q1): before the SQL agent, the model lists the abbreviations and typos of the
question with their full words, and states the date ranges the question asks about.

The model does not rewrite the question: code replaces only the listed words in the original, so
everything else (accents as typed, product names, codes) stays character for character. A fix
that only adds or removes accents is dropped (compared with accents stripped).

Context: only the rulebook sections that define time (picked by the model once per rulebook,
pick_time_sections), today's date and the weeks around it. The model writes each period as a
SELECT on the calendar table (solar date, weekday, lunar date, holiday); DuckDB runs it, so the
date arithmetic and the lunar lookup are exact. A query that fails or finds no day goes back to
the model. No word list and no time window is computed in code.

The periods are used twice: the SQL agent sees them as dates, and the answer check compares
them with the dates its query actually reads (solve.SqlBox.period_problems).
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

WEEKDAY_VI = ["thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm", "thứ Sáu", "thứ Bảy", "Chủ nhật"]
WEEKS_SHOWN = 9        # weeks of calendar before today, plus the current one
NORM_TRIES = 4         # calls: the first, fixes of period queries that failed or miscounted
QUERY_SECONDS = 10

NORM_SCHEMA = {"type": "object", "properties": {
    "question": {"type": "string"},
    "fixes": {"type": "array", "items": {"type": "object", "properties": {
        "from": {"type": "string"}, "to": {"type": "string"}},
        "required": ["from", "to"], "additionalProperties": False}, "maxItems": 12},
    "periods": {"type": "array", "items": {"type": "object", "properties": {
        "label": {"type": "string"}, "sql": {"type": "string"}},
        "required": ["label", "sql"], "additionalProperties": False}, "maxItems": 12},
    "reason": {"type": "string"}},
    "required": ["question", "fixes", "periods", "reason"], "additionalProperties": False}

# ---------------------------------------------------------------- the rulebook's time sections

PICK_SCHEMA = {"type": "object", "properties": {
    "sections": {"type": "array", "items": {"type": "string"}}, "reason": {"type": "string"}},
    "required": ["sections", "reason"], "additionalProperties": False}


def pick_time_sections(rb, llm) -> dict:
    """The model picks the rulebook sections that define periods of time (today, weeks, months,
    the lunar calendar...). One call per rulebook; the answer is cached by the caller."""
    heads = {}
    for u in rb.units:
        heads.setdefault(u.section, []).append(u.title[:60])
    listing = "\n".join(f"- {sec}: " + "; ".join(t for t in titles[:12]) for sec, titles in heads.items())
    prompt = ("Dưới đây là các section của một rulebook (tên section: tiêu đề các dòng trong đó). Chọn các section "
              "định nghĩa cách hiểu thời gian trong câu hỏi (hôm nay, tuần, tháng, quý, năm, N ngày gần nhất, âm "
              "lịch, ngày lễ...). Trả lời đúng tên section như danh sách.\n\n" + listing)
    try:
        data, _ = llm.json(prompt, PICK_SCHEMA, max_tokens=400)
    except (ValueError, RuntimeError) as e:
        data = {"sections": [], "reason": f"error: {e!r}"[:300]}
    picked = [x for x in data.get("sections", []) if x in heads]
    return {"sections": picked, "reason": data.get("reason"), "prompt": prompt}


def time_rules(rb, sections: list[str]) -> str:
    """The picked sections, rendered like the agent's rulebook. Nothing picked: every section."""
    out, last = [], None
    for u in rb.units:
        if sections and u.section not in sections:
            continue
        if u.section != last:
            out.append(f"\n### {u.section}")
            last = u.section
        out.append(f"- {u.text}")
    return "\n".join(out).strip()


def load_time_sections(cache: Path, rb, llm) -> list[str]:
    """<cache>/time_sections.json, made on first use."""
    path = cache / "time_sections.json"
    if not path.exists():
        path.write_text(json.dumps(pick_time_sections(rb, llm), ensure_ascii=False, indent=1), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))["sections"]

# ---------------------------------------------------------------- the calendar


class Calendar:
    """calendar.csv as a DuckDB table cal, for running the model's period queries."""

    def __init__(self, calendar: Path):
        self.con = duckdb.connect()
        cal = pd.read_csv(calendar, parse_dates=["date"])
        cal["date"] = cal["date"].dt.date
        self.con.register("_cal", cal)
        self.con.execute("CREATE TABLE cal AS SELECT * FROM _cal")
        self.con.unregister("_cal")
        self.con.execute("SET enable_external_access = false")

    def span(self, sql: str) -> tuple[str, str]:
        """(from, to) of a period query: the earliest and latest value of its date columns. Raises
        ValueError with a message for the model."""
        timer = threading.Timer(QUERY_SECONDS, self.con.interrupt)
        timer.start()
        try:
            df = self.con.execute(sql).df()
        except Exception as e:
            raise ValueError(f"{type(e).__name__}: {str(e).strip()[:300]}")
        finally:
            timer.cancel()
        dated = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
                 or df[c].map(lambda v: isinstance(v, date)).all()]
        days = pd.to_datetime(pd.Series(df[dated].values.ravel()), errors="coerce").dropna() if dated else pd.Series([], dtype="datetime64[ns]")
        if days.empty:
            raise ValueError("câu SQL không ra cột ngày nào")
        return f"{days.min():%Y-%m-%d}", f"{days.max():%Y-%m-%d}"

    def close(self):
        self.con.close()


def calendar_facts(asof: pd.Timestamp) -> str:
    """The weeks around today, Monday first."""
    start = asof.date() - timedelta(days=asof.weekday() + 7 * WEEKS_SHOWN)
    lines = []
    for w in range(WEEKS_SHOWN + 1):
        days = [start + timedelta(days=7 * w + i) for i in range(7)]
        lines.append(" | ".join(f"{WEEKDAY_VI[d.weekday()]} {d:%Y-%m-%d}" for d in days))
    return "Lịch các tuần gần đây:\n" + "\n".join(lines)

# ---------------------------------------------------------------- normalising one question


NORM_SYSTEM = """Bạn chuẩn hoá câu hỏi của chủ một cửa hàng nhỏ trước khi một người khác viết SQL trả lời. Trả về JSON:
{"question": "...", "fixes": [{"from": "...", "to": "..."}], "periods": [{"label": "...", "sql": "SELECT ... FROM cal ..."}], "reason": "..."}

- question: câu hỏi gốc viết lại đủ chữ, có dấu, để hiểu đúng nghĩa. Giữ nguyên ý, tên riêng, mã và con số. Viết xong thì đọc lại: câu mới phải là một câu hỏi rõ nghĩa của chủ cửa hàng, và mỗi chữ của câu gốc đều đã được viết đủ.
- fixes: các chữ viết tắt và chữ gõ sai của câu gốc (so câu gốc với question), mỗi chữ một mục: from là chữ đúng như trong câu gốc, to là chữ viết đủ. Chủ cửa hàng gõ nhanh kiểu nhắn tin. Viết tắt hay gặp: lấy chữ cái đầu mỗi từ (ntn = như thế nào, sl = số lượng), hoặc chữ cái đầu của từ trước ghép với cả từ sau (hkia = hôm kia, tgian = thời gian, bgio = bây giờ, nvien = nhân viên). Đọc từng chữ theo ngữ cảnh cả câu. Chữ chỉ thiếu dấu không phải lỗi, không ghi. Tên riêng (mặt hàng, khách, nhà cung cấp), mã và con số không ghi. Không có chữ nào cần sửa thì fixes = [].
- periods: mỗi khoảng ngày câu hỏi hỏi tới, theo hôm nay và các luật thời gian dưới đây. label là tên kỳ như câu hỏi gọi. sql là một câu SELECT (DuckDB) trên bảng lịch
    cal(date DATE, weekday INTEGER 1 = thứ Hai … 7 = Chủ nhật, lunar_day INTEGER, lunar_month INTEGER, lunar_year INTEGER, lunar_leap BOOLEAN, holiday VARCHAR)
  trả về các ngày của kỳ (hoặc ngày đầu và ngày cuối); hệ thống lấy ngày sớm nhất và muộn nhất làm khoảng from..to. Để DuckDB tính ngày (DATE '...', INTERVAL, date_trunc, cột âm lịch, holiday), không tự tính nhẩm. Kỳ tính theo một kỳ khác (ví dụ N ngày trước một kỳ) thì sql tính từ ngày đầu hoặc ngày cuối của kỳ đó. Câu so sánh nhiều kỳ thì ghi đủ từng kỳ theo thứ tự câu hỏi nêu. Câu hỏi không nói tới thời gian thì periods = [].
- reason: một câu nói các kỳ tính thế nào.

Luật thời gian:
{rules}

{calendar}"""


def norm_messages(q: dict, rules: str) -> list[dict]:
    asof = pd.Timestamp(q["as_of_date"])
    system = NORM_SYSTEM.replace("{rules}", rules).replace("{calendar}", calendar_facts(asof))
    user = f"Hôm nay: {asof:%Y-%m-%d} ({WEEKDAY_VI[asof.weekday()]}).\nCâu hỏi gốc: {q['question']}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strip(s: str) -> str:
    """Accents and đ stripped, case kept."""
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _bare(s: str) -> str:
    return _strip(s.lower())


def apply_fixes(question: str, fixes: list[dict]) -> tuple[str, list[dict]]:
    """The question with each fix replaced where it stands as a whole word; fixes not found in the
    question or that only change accents are skipped. A fix of a word typed without accents is
    written without accents too. Returns (question, fixes applied)."""
    done = []
    for f in fixes:
        old, new = str(f.get("from", "")).strip(), str(f.get("to", "")).strip()
        if not old or not new or _bare(old) == _bare(new):
            continue
        if old == _strip(old):
            new = _strip(new)
        pat = re.compile(rf"(?<!\w){re.escape(old)}(?!\w)", re.I)
        if pat.search(question):
            question = pat.sub(lambda m: new, question)
            done.append({"from": old, "to": new})
    return question, done


def reading_hints(question: str, fixes: list[dict]) -> list[dict]:
    """The model's readings of abbreviations and typos that stand in the question as whole words,
    as the model wrote them (accents included). They go to the SQL agent as hints next to the
    question as typed, so a wrong reading never changes the question itself."""
    out = []
    for f in fixes:
        old, new = str(f.get("from", "")).strip(), str(f.get("to", "")).strip()
        if old and new and old.lower() != new.lower() and re.search(rf"(?<!\w){re.escape(old)}(?!\w)", question, re.I):
            out.append({"from": old, "to": new})
    return out


def count_mismatch(periods: list[dict]) -> list[str]:
    """Periods whose label names a number of days N while the period holds N - 1 or N + 1 days:
    the usual off-by-one of counting a span. Only the numbers are compared."""
    out = []
    for p in periods:
        days = (date.fromisoformat(p["to"]) - date.fromisoformat(p["from"])).days + 1
        nums = {int(n) for n in re.findall(r"\d+", p["label"])}
        if days not in nums and any(n >= 2 and abs(days - n) == 1 for n in nums):
            out.append(f"- {p['label']}: {p['from']} đến {p['to']} là {days} ngày")
    return out


def normalize(q: dict, rules: str, calendar: Path, llm, think: bool = False, sample: dict | None = None) -> dict:
    """{"question", "fixes", "periods", "reason", "messages", "failures"}. Period queries that fail
    go back to the model; after NORM_TRIES calls the periods that ran are kept. A reply with no
    JSON (empty, cut off) is asked again without thinking. Falls back to the original question
    with no periods when no JSON comes back."""
    messages = norm_messages(q, rules)
    cal, data, periods, failures, think_now = Calendar(calendar), {}, [], [], think
    counted = False
    try:
        for _ in range(NORM_TRIES):
            r = {}
            try:
                r = llm.chat(messages, schema=NORM_SCHEMA, think=think_now, max_tokens=10000 if think_now else 1200,
                             **(sample or {}))
                got = json.loads(re.sub(r"<think>.*?</think>", "", r["content"], flags=re.S).strip())
            except (ValueError, RuntimeError) as e:     # empty or cut off: ask again without thinking
                failures.append(f"{e!r}"[:200] + f" finish={r.get('finish')} provider={r.get('provider')} think={think_now}")
                think_now = False
                continue
            messages.append({"role": "assistant", "content": json.dumps(got, ensure_ascii=False)})
            data, periods, errors = got, [], []
            for p in got.get("periods") or []:
                try:
                    lo, hi = cal.span(str(p.get("sql", "")))
                    periods.append({"label": str(p.get("label", "")), "from": lo, "to": hi, "sql": p.get("sql")})
                except ValueError as e:
                    errors.append(f"- {p.get('label')}: {e}")
            if not errors and not counted and (off := count_mismatch(periods)):
                counted = True
                messages.append({"role": "user", "content": "Số ngày của các kỳ sau lệch một so với số trong tên kỳ:\n" +
                                 "\n".join(off) + "\nĐếm lại theo luật thời gian (BETWEEN tính cả hai đầu) và trả lại "
                                 "toàn bộ JSON; kỳ nào đã đúng thì giữ nguyên."})
                continue
            if not errors:
                break
            messages.append({"role": "user", "content": "Một số sql của periods chạy không được:\n" +
                             "\n".join(errors) + "\nSửa và trả lại toàn bộ JSON."})
    finally:
        cal.close()
    question, fixes = apply_fixes(q["question"], data.get("fixes") or [])
    return {"question": question, "fixes": fixes, "hints": reading_hints(q["question"], data.get("fixes") or []), "periods": periods, "reason": data.get("reason"),
            "messages": messages, "failures": failures}


def render_periods(periods: list[dict]) -> str:
    if not periods:
        return "không giới hạn thời gian"
    return "; ".join(f"{p['label']}: {p['from']} đến {p['to']}" for p in periods)

# ---------------------------------------------------------------- reading it back


def _render(messages: list[dict]) -> list[str]:
    out = []
    for m in messages:
        role = m["role"].upper()
        if m.get("tool_calls"):
            calls = "\n".join(f"{c['function']['name']}({c['function']['arguments']})" for c in m["tool_calls"])
            out += [f"### {role} gọi tool\n", "```", (m.get("content") or "").strip(), calls, "```\n"]
        else:
            out += [f"### {role}\n", "```", m.get("content") or "", "```\n"]
    return out


def dump_markdown(path: Path, q: dict, norm: dict, messages: list[dict], tools: list, result: dict) -> None:
    """Every prompt of one question, in the order they were sent, for reading by hand."""
    out = [f"# {q['id']} ({q['business_id']}) · hôm nay {q['as_of_date']}\n", f"Câu gốc: {q['question']}\n",
           "## Bước 1: chuẩn hoá\n", *_render(norm["messages"]),
           f"Câu đã chuẩn hoá: {norm['question']}  \nKỳ (code chạy sql trên cal): {render_periods(norm['periods'])}\n",
           "## Bước 2: SQL agent\n", "### TOOLS\n", "```json", json.dumps(tools, ensure_ascii=False, indent=1), "```\n",
           *_render(messages)]
    out += ["## Kết quả\n", "```json", json.dumps(result, ensure_ascii=False, default=str, indent=1), "```"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(re.sub(r"\n{3,}", "\n\n", s) for s in out), encoding="utf-8")

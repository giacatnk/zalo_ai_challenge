"""Business layer B5: turn the loaded tables into a typed, read-only SQL database.

The model decides what each column is. One LLM call per table reads the column profile (the
shapes cells are written in, examples, README) and returns a spec: number, date, datetime or
text, plus the units written after numbers ("k" -> 1000) and the day/month order. Code only
applies the spec with the generic parsers and checks it: a column that does not parse on at
least MIN_PARSED of its cells stays text and the schema says so. Every converted column keeps
its original next to it as <name>_goc.

Names become ASCII snake_case (thanh_tien, ngay_dat_hang) so SQL needs no quoting and no
Vietnamese diacritics; the schema maps each one back to its original name.

Output per business: db.duckdb (tables + cal), spec.json (the model's spec and parse rates),
schema.md (what the question layer shows the model).
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .eda import render_column
from .parsing import parse_date, parse_number
from .profile import TableProfile
from .text import norm_text

MIN_PARSED = 0.99      # share of non-empty cells a conversion must read, else the column stays text
LOW_CARD = 30          # list every value of a text column with at most this many
GROUP_ROWS = 20        # a text column with at most this many rows per value groups rows (an order of lines)
KINDS = ("number", "date", "datetime", "text")
MAX_UNITS = 24         # units per column; caps a runaway list
SPEC_TRIES = 2         # calls before a table falls back to text

SPEC_SCHEMA = {"type": "object", "properties": {
    "columns": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"},
        "kind": {"type": "string", "enum": list(KINDS)},
        "units": {"type": "array", "items": {"type": "object", "properties": {
            "suffix": {"type": "string"}, "multiplier": {"type": "number"}},
            "required": ["suffix", "multiplier"], "additionalProperties": False}, "maxItems": MAX_UNITS},
        "date_order": {"type": "string", "enum": ["dmy", "mdy", "none"]}},
        "required": ["name", "kind", "units", "date_order"], "additionalProperties": False}}},
    "required": ["columns"], "additionalProperties": False}


# ---------------------------------------------------------------- names


def sql_name(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", norm_text(name)).strip("_") or "cot"
    if base[0].isdigit():
        base = f"c_{base}"
    out, n = base, 2
    while out in taken:
        out, n = f"{base}_{n}", n + 1
    taken.add(out)
    return out


# ---------------------------------------------------------------- the model's spec


def spec_prompt(tp: TableProfile, readme: str) -> str:
    cols = "\n".join(render_column(c) for c in tp.columns if c.non_null > 0)
    return (
        "Dưới đây là một bảng dữ liệu bán hàng của một cửa hàng, đọc từ file Excel/CSV, giá trị còn nguyên như "
        "trong file. Hãy cho biết mỗi cột nên đọc thành kiểu gì để truy vấn SQL.\n\n"
        "kind:\n"
        "- number: số đo, tiền, số lượng, tỉ lệ. units: mỗi phần chữ hay ký hiệu viết ngay sau số trong cột này, "
        'chỉ phần chữ, không kèm số, cùng hệ số nhân. Ví dụ ô "99k" -> suffix "k", multiplier 1000; ô "1.050.000đ" '
        '-> suffix "đ", multiplier 1; ô "2 ly" -> suffix "ly", multiplier 1. Số viết bình thường ("45,000.0000", '
        '"28.970.000") không cần units.\n'
        "- date: ngày; datetime: ngày kèm giờ. date_order: dmy (ngày/tháng/năm) hoặc mdy; dạng năm-tháng-ngày và số "
        "serial Excel (ví dụ 45886) được đọc tự động. Cột không phải ngày: date_order = none.\n"
        "- text: tên, mã, trạng thái, nhóm, ghi chú, số điện thoại, mã vạch, và mọi cột không chắc.\n"
        "Trả lời đủ mọi cột, đúng tên cột.\n\n"
        f"README của cửa hàng:\n{readme.strip()[:3000]}\n\n"
        f"Bảng {tp.name} ({tp.rows} dòng). Các cột (dạng viết thay chữ số bằng ví dụ thật, kèm tỉ lệ):\n{cols}")


def _read_spec(data: dict, names: list[str]) -> dict:
    by_name = {c.get("name"): c for c in data.get("columns", [])}
    cols = {}
    for name in names:
        got = by_name.get(name) or {}
        kind = got.get("kind") if got.get("kind") in KINDS else "text"
        units = {u["suffix"].strip().lower(): float(u["multiplier"]) for u in got.get("units", [])
                 if isinstance(u, dict) and str(u.get("suffix", "")).strip()
                 and re.match(r"[^\W\d]", str(u["suffix"]).strip())}   # a unit starts with a letter ('xe 4m3')
        cols[name] = {"kind": kind, "units": units, "date_order": got.get("date_order", "none")}
    return cols


def ask_spec(tp: TableProfile, readme: str, llm) -> dict:
    prompt = spec_prompt(tp, readme)
    for _ in range(SPEC_TRIES):
        try:
            data, r = llm.json(prompt, SPEC_SCHEMA, max_tokens=3000)
            break
        except (ValueError, RuntimeError) as e:      # malformed or runaway JSON: ask again, then every column stays text
            data, r = {}, {"usage": None, "seconds": None, "error": repr(e)[:300]}
    return {"columns": _read_spec(data, [c.name for c in tp.columns]), "prompt": prompt,
            "usage": r["usage"], "seconds": r["seconds"], "error": r.get("error")}


def retry_spec(spec: dict, misses: dict[str, dict], llm) -> dict:
    """One more call for the columns whose conversion read too few cells, showing the cells
    that failed. The model may add units, change the kind, or settle on text."""
    lines = [f'- {name}: đã chọn {m["kind"]}' + (f' với units {m["units"]}' if m["units"] else "") +
             (f', mọi ô là số nguyên {m["serial"]}; cột này là số đo hay là ngày?' if m.get("serial") else
              f', chỉ đọc được {m["parsed"]:.0%} ô; các ô không đọc được: ' + ", ".join(f'"{v}"' for v in m["failed"]))
             for name, m in misses.items()]
    prompt = (spec["prompt"] + "\n\nLần trước, các cột sau cần xem lại:\n" + "\n".join(lines) +
              "\nHãy trả lời lại cho riêng các cột này (thêm units còn thiếu, đổi kind, hoặc chọn text).")
    try:
        data, _ = llm.json(prompt, SPEC_SCHEMA, max_tokens=1500)
    except (ValueError, RuntimeError):               # keep the first answer; apply_spec leaves them text
        return {n: {k: v for k, v in m.items() if k in ("kind", "units", "date_order")} for n, m in misses.items()}
    return _read_spec(data, list(misses))


# ---------------------------------------------------------------- row groups (filling down)

FILL_SCHEMA = {"type": "object", "properties": {
    "fill_down": {"type": "array", "items": {"type": "string"}},
    "reason": {"type": "string"}},
    "required": ["fill_down", "reason"], "additionalProperties": False}
FILL_ROWS = 12         # rows shown to the model


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


def fill_candidates(df: pd.DataFrame) -> list[list[str]]:
    """Blocks of columns that look like one header per group of rows: two or more columns
    empty on exactly the same rows, the first row filled, and one of them never repeating a
    value on its filled rows (each filled row opens a new group). A column empty on its own
    is never a candidate."""
    blank = {c: _blank(df[c]) for c in df.columns if c != "_sheet"}
    blocks = {}
    for c, b in blank.items():
        if 0 < b.mean() < 1 and not b.iloc[0]:
            blocks.setdefault(b.values.tobytes(), []).append(c)
    out = []
    for cols in blocks.values():
        filled = ~blank[cols[0]]
        if len(cols) >= 2 and any(df.loc[filled, c].astype(str).str.strip().is_unique for c in cols):
            out.append(cols)
    return out


def fill_prompt(name: str, df: pd.DataFrame, blocks: list[list[str]], rules: str) -> str:
    sample = df.head(FILL_ROWS).drop(columns="_sheet", errors="ignore").fillna("")
    with pd.option_context("display.max_columns", 50, "display.width", 250, "display.max_colwidth", 30):
        rows = sample.to_string(index=False)
    lines = "\n".join(f"- {', '.join(cols)}: trống ở {_blank(df[cols[0]]).mean():.0%} số dòng, luôn trống cùng nhau, "
                      f"dòng đầu bảng có giá trị" for cols in blocks)
    return (
        "Bảng dưới đây có những nhóm cột luôn trống trên cùng các dòng. Đọc rulebook và các dòng mẫu, rồi cho biết "
        "cột nào phải điền xuống (ô trống lấy giá trị của dòng có giá trị gần nhất phía trên). Chỉ chọn trong các "
        "cột được nêu; cột trống vì không có thông tin thì không chọn. Không chắc thì không chọn.\n\n"
        f"Rulebook:\n{rules}\n\nBảng {name} ({len(df)} dòng). Các nhóm cột:\n{lines}\n\n"
        f"{FILL_ROWS} dòng đầu:\n{rows}")


def ask_fill(name: str, df: pd.DataFrame, rules: str, llm) -> dict:
    """The model's choice of columns to fill down, among the structural candidates. A reply with
    no JSON is asked again, up to SPEC_TRIES calls."""
    blocks = fill_candidates(df)
    if not blocks or llm is None:
        return {"candidates": blocks, "fill_down": [], "reason": None}
    allowed = {c for cols in blocks for c in cols}
    data, failures = {}, []
    for _ in range(SPEC_TRIES):
        try:
            data, r = llm.json(fill_prompt(name, df, blocks, rules), FILL_SCHEMA, max_tokens=800)
        except (ValueError, RuntimeError) as e:
            failures.append(f"error: {e!r}"[:300])
            continue
        if "fill_down" in data:
            break
        failures.append(f"no JSON: finish={r.get('finish')} provider={r.get('provider')} content={r['content'][:200]!r}")
    chosen = [c for c in data.get("fill_down", []) if c in allowed]
    return {"candidates": blocks, "fill_down": chosen, "reason": data.get("reason"), "failures": failures}


def fill_down(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Empty cells take the last filled value above, never across sheets."""
    if not cols:
        return df
    df = df.copy()
    for c in cols:
        s = df[c].where(~_blank(df[c]))
        df[c] = s.groupby(df["_sheet"]).ffill() if "_sheet" in df.columns else s.ffill()
    return df


# ---------------------------------------------------------------- applying it


def drop_copies(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rows equal to an earlier row in every cell (after fill-down) are file-export copies: one is
    kept. Returns (table, rows dropped)."""
    cols = [c for c in df.columns if c != "_sheet"]
    dup = df.astype(str).duplicated(subset=cols, keep="first")
    return df[~dup].reset_index(drop=True), int(dup.sum())


_NO_YEAR = re.compile(r"^\s*\d{1,2}\s*[/.-]\s*\d{1,2}\s*$")


def sheet_years(df: pd.DataFrame) -> pd.Series | None:
    """Per row, the year its sheet name holds (the last 4-digit number from 2000 on), if any."""
    if "_sheet" not in df.columns:
        return None
    y = df["_sheet"].astype(str).str.findall(r"(?<!\d)(20\d\d)(?!\d)").map(lambda xs: xs[-1] if xs else None)
    return y if y.notna().any() else None


def _convert(s: pd.Series, spec: dict, years: pd.Series | None = None) -> pd.Series:
    """A column by its spec. A date written without a year (day and month only) takes the year
    of its sheet's name when there is one."""
    if spec["kind"] == "number":
        return s.map(lambda v: parse_number(v, spec["units"])).astype(float)
    order = spec["date_order"] if spec["date_order"] in ("dmy", "mdy") else "dmy"
    def one(i, v):
        if years is not None and isinstance(v, str) and _NO_YEAR.match(v) and years.get(i):
            v = f"{v.strip()}/{years[i]}"
        return parse_date(v, order=order)
    out = pd.Series([one(i, v) for i, v in s.items()], index=s.index, dtype="datetime64[ns]")
    return out.dt.normalize() if spec["kind"] == "date" else out


def _check(s: pd.Series, sp: dict, years: pd.Series | None = None) -> tuple[pd.Series | None, float, list[str]]:
    """(converted column, share of non-empty cells read, examples that failed)."""
    nonempty = s[s.notna() & (s.astype(str).str.strip() != "")]
    if sp["kind"] == "text" or not len(nonempty):
        return None, 1.0, []
    conv = _convert(s, sp, years)
    ok = conv[nonempty.index].notna()
    return conv, round(float(ok.mean()), 4), [str(v) for v in nonempty[~ok].unique()[:8]]


SERIAL_RANGE = (36526, 54789)   # Excel day serials of 2000-01-01 .. 2049-12-31
SERIAL_SPAN = 3653              # a column of dates spans at most this many days
SERIAL_DISTINCT = 28            # and holds at least this many different days


def serial_dates(s: pd.Series) -> str | None:
    """When a number column looks like Excel day serials (whole numbers in the serial range of
    this century, spanning a few years, many different values), the dates they would be."""
    v = pd.to_numeric(s, errors="coerce").dropna()
    if len(v) < len(s.dropna()) or not len(v) or (v % 1 != 0).any():
        return None
    lo, hi = float(v.min()), float(v.max())
    if not (SERIAL_RANGE[0] <= lo and hi <= SERIAL_RANGE[1] and hi - lo <= SERIAL_SPAN and v.nunique() >= SERIAL_DISTINCT):
        return None
    day = lambda x: (pd.Timestamp("1899-12-30") + pd.Timedelta(days=x)).strftime("%Y-%m-%d")
    return f"{lo:.0f} .. {hi:.0f}, đọc theo số serial ngày của Excel thì là {day(lo)} .. {day(hi)}"


def settle_spec(df: pd.DataFrame, spec: dict, llm) -> dict:
    """Columns below MIN_PARSED get one retry with their failed cells; number columns that look
    like Excel day serials are asked about once more."""
    misses = {}
    for name in df.columns:
        sp = spec["columns"].get(name)
        if sp and sp["kind"] != "text":
            _, parsed, failed = _check(df[name], sp, sheet_years(df))
            if parsed < MIN_PARSED:
                misses[name] = {**sp, "parsed": parsed, "failed": failed}
        if sp and sp["kind"] == "number" and (dates := serial_dates(df[name])):
            misses[name] = {**sp, "parsed": 1.0, "failed": [], "serial": dates}
    if misses and llm is not None:
        spec["retried"] = {n: {**spec["columns"][n]} for n in misses}
        spec["columns"].update(retry_spec(spec, misses, llm))
    return spec


def _native(s: pd.Series) -> str | None:
    """'number' / 'datetime' when the file itself stores every cell that way (Excel numbers
    and dates): a fact of the file, kept whatever the spec says."""
    nn = s.dropna()
    if not len(nn):
        return None
    if nn.map(lambda v: isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))).all():
        return "number"
    if nn.map(lambda v: isinstance(v, (pd.Timestamp, datetime, date))).all():
        return "datetime"
    return None


def apply_spec(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, list[dict]]:
    """(typed table with ASCII names, one record per column for the schema)."""
    out, cols, taken = {}, [], set()
    years = sheet_years(df)
    for name in df.columns:
        s = df[name]
        sp = spec["columns"].get(name, {"kind": "text", "units": {}, "date_order": "none"})
        if native := _native(s):
            keep = sp["kind"] in ("date", "datetime") and (native == "datetime" or serial_dates(s))
            sp = {**sp, "kind": sp["kind"] if keep else native}
        col = {"name": sql_name(str(name), taken), "original": str(name), "kind": sp["kind"],
               "spec": sp, "parsed": None, "failed_examples": []}
        conv, parsed, failed = _check(s, sp, years)
        if conv is not None:
            col["parsed"], col["failed_examples"] = parsed, failed
            if parsed >= MIN_PARSED:
                out[col["name"]] = conv
                if s.map(lambda v: isinstance(v, str)).any():   # written as text: keep the original
                    col["raw"] = sql_name(f"{name}_goc", taken)
                    out[col["raw"]] = s.map(lambda v: None if pd.isna(v) else str(v))
            else:
                col["kind"] = "text"
                col["rejected"] = sp["kind"]
        if col["kind"] == "text":
            out[col["name"]] = s.map(lambda v: None if pd.isna(v) else str(v).strip())
        cols.append(col)
    return pd.DataFrame(out, index=df.index), cols


# ---------------------------------------------------------------- values that mean the same thing

GROUP_SCHEMA = {"type": "object", "properties": {
    "rule": {"type": "string"},
    "groups": {"type": "array", "items": {"type": "object", "properties": {
        "label": {"type": "string"}, "values": {"type": "array", "items": {"type": "string"}}},
        "required": ["label", "values"], "additionalProperties": False}},
    "reason": {"type": "string"}},
    "required": ["rule", "groups", "reason"], "additionalProperties": False}
NOVALUE_SCHEMA = {"type": "object", "properties": {
    "values": {"type": "array", "items": {"type": "object", "properties": {
        "value": {"type": "string"}, "names_one": {"type": "boolean"}},
        "required": ["value", "names_one"], "additionalProperties": False}}},
    "required": ["values"], "additionalProperties": False}
TOP_SHOWN = 12         # most frequent values shown for a text column with more than LOW_CARD values
GROUP_TOKENS = 8000    # thinking on: an offline call per column, where reading the rulebook pays
GROUP_WORKERS = 8      # columns asked at once
MIN_TOP_SHARE = 0.01   # a large column whose top value is rarer than this holds no placeholder
OUTLIER = 5            # a placeholder has at least this many times the rows of the median shown value
RUNAWAY = 50           # trailing whitespace of a cut-off reply that ran on blanks after finishing
GROUP_TRIES = 3        # calls per column: the first with thinking, then without


def group_prompt(table: str, col: dict, vc: pd.Series, rules: str) -> str:
    """One text column: its values with row counts, and the question that fits its size."""
    head = f'Rulebook:\n{rules}\n\nBảng {table}, cột {col["name"]} ("{col["original"]}"). '
    if len(vc) <= LOW_CARD:
        return (head + f"Tất cả {len(vc)} giá trị của cột, kèm số dòng:\n" + "\n".join(f"- '{v}' ({n})" for v, n in vc.items()) +
                "\n\nNgười viết SQL sẽ lọc và đếm theo các giá trị này, theo đúng các khái niệm của rulebook.\n"
                "- rule: nếu rulebook định nghĩa một tập loại cố định mà mỗi dòng của cột này thuộc đúng một loại, chép "
                "nguyên văn một đoạn ngắn của rulebook nói tập loại đó; rulebook không có thì rule = \"\".\n"
                "- groups: chỉ khi có rule: mỗi loại của rulebook là một nhóm, label là tên loại đúng như rulebook gọi, "
                "values là các giá trị của cột thuộc loại đó. Không có rule thì groups = [].\n"
                "Chép giá trị đúng từng chữ.")
    total = int(vc.sum())
    return (head + f"{TOP_SHOWN} giá trị hay gặp nhất trong {len(vc)} giá trị của cột, kèm số dòng và tỉ lệ trên {total} dòng:\n" +
            "\n".join(f"- '{v}' ({n}, {n / total:.1%})" for v, n in vc.head(TOP_SHOWN).items()) +
            "\n\nVới từng giá trị trong danh sách, names_one cho biết giá trị đó có phải tên (hay mã) của đúng một "
            "đối tượng riêng (một người, một mặt hàng, một đơn...) hay không. names_one = false khi giá trị là cách "
            "ghi chung cho nhiều đối tượng không biết tên, hoặc ghi thay cho chỗ không có thông tin; rulebook nói tới "
            "cách ghi đó thì theo rulebook. Chép giá trị đúng từng chữ.")


def _flat(t: str) -> str:
    return " ".join(str(t).replace("*", "").replace("`", "").split()).lower()


def _clean(v) -> str:
    return str(v).strip().strip("'\"")


def _kinds(data: dict, vc: pd.Series, rules: str) -> tuple[list[dict] | None, str]:
    """Checks a grouping of a small column. ([], why) when the model found no set of kinds in the
    rulebook, (groups, "") when it holds, (None, why) when it must be asked again: the quote is
    not in the rulebook, a label is not a rulebook word, or some value of the column has no group
    (each row of such a column is one of the kinds)."""
    rule = _flat(data.get("rule") or "")
    groups = [{"label": str(g.get("label") or "").strip(), "values": [_clean(v) for v in g.get("values") or []]}
              for g in data.get("groups") or []]
    groups = [g for g in groups if g["label"] and g["values"]]
    if not rule and not groups:
        return [], "no set of kinds"
    if len(rule) < 8 or rule not in _flat(rules):
        return None, "quote not in the rulebook"
    if len(groups) < 2 or not all(_flat(g["label"]) in _flat(rules) for g in groups):
        return None, "labels are not the rulebook's kinds"
    missing = set(vc.index) - {v for g in groups for v in g["values"]}
    if missing:
        return None, f"values with no kind: {sorted(missing)[:5]}"
    return groups, ""


def _placeholders(data: dict, vc: pd.Series) -> list[str]:
    """Values the model marked as not naming one thing, kept when they are few among the names
    shown and each holds far more rows than a typical named value (it stands for many)."""
    marked = [_clean(x.get("value")) for x in data.get("values") or [] if x.get("names_one") is False]
    if len(marked) > min(TOP_SHOWN, len(vc)) // 2:
        return []
    typical = float(vc.head(TOP_SHOWN).median())
    return [v for v in marked if v in vc.index and vc[v] >= OUTLIER * typical]


def _ask_column(table: str, col: dict, vc: pd.Series, rules: str, llm) -> tuple[dict, list[str]]:
    few = len(vc) <= LOW_CARD
    prompt, failures = group_prompt(table, col, vc, rules), []
    out = {"name": col["name"], "groups": [], "no_value": []}
    for attempt in range(GROUP_TRIES):             # thinking first; then without
        think = attempt == 0
        try:
            data, r = llm.json(prompt, GROUP_SCHEMA if few else NOVALUE_SCHEMA,
                               max_tokens=GROUP_TOKENS if think else 2000, think=think)
        except (ValueError, RuntimeError) as e:
            failures.append(f"{col['name']}: {e!r}"[:300])
            continue
        tail = len(r["content"]) - len(r["content"].rstrip())
        if r.get("finish") == "length" and tail < RUNAWAY:   # cut off inside the content: values may be missing
            failures.append(f"{col['name']}: cut off (think={think})")
            continue
        if few:
            groups, why = _kinds(data, vc, rules)
            if groups is None:
                failures.append(f"{col['name']}: {why} (think={think})")
                continue
            return {**out, "rule": data.get("rule"), "groups": groups, "reason": data.get("reason")}, failures
        if "values" in data:
            return {**out, "no_value": _placeholders(data, vc)}, failures
        failures.append(f"{col['name']}: no JSON, finish={r.get('finish')}")
    return out, failures


def ask_groups(name: str, df: pd.DataFrame, cols: list[dict], rules: str, llm) -> dict:
    """Per text column, the model's groups of same-meaning values (columns with at most LOW_CARD
    values) and placeholder values (larger columns). One call per column, run in parallel. A
    large column whose most frequent value holds under MIN_TOP_SHARE of the rows (codes, ids)
    has no placeholder to find and is skipped."""
    out = {"columns": [], "failures": []}
    if llm is None:
        return out
    jobs = []
    for col in cols:
        if col["kind"] != "text" or col["original"] == "_sheet":
            continue
        vc = df[col["name"]].dropna().value_counts()
        if len(vc) < 2 or (len(vc) > LOW_CARD and vc.iloc[0] / vc.sum() < MIN_TOP_SHARE):
            continue
        jobs.append((col, vc))
    with ThreadPoolExecutor(max_workers=GROUP_WORKERS) as pool:
        for got, failures in pool.map(lambda j: _ask_column(name, j[0], j[1], rules, llm), jobs):
            out["columns"].append(got)
            out["failures"] += failures
    return out


def apply_groups(df: pd.DataFrame, cols: list[dict], grouping: dict) -> tuple[pd.DataFrame, list[dict]]:
    """A <col>_nhom column next to each column the model grouped: the group label for grouped
    values, NULL for placeholder values, the value itself otherwise. Values the column does not
    hold are ignored; a group of one value with its own name changes nothing."""
    by_name = {c["name"]: c for c in cols}
    taken = set(df.columns)
    out_cols = list(cols)
    for g in grouping.get("columns", []):
        col = by_name.get(g.get("name"))
        if col is None or col["kind"] != "text" or col.get("group_of") or col["original"] == "_sheet":
            continue
        held = set(df[col["name"]].dropna())
        mapping, groups = {}, []
        for grp in g.get("groups") or []:
            vals = [v for v in dict.fromkeys(str(x).strip().strip("'\"") for x in grp.get("values") or [])
                    if v in held and v not in mapping]
            label = str(grp.get("label") or "").strip()
            if not label or not vals or (len(vals) == len(held) > 1):
                continue                             # one group swallowing the whole column
            mapping.update({v: label for v in vals})
            groups.append({"label": label, "values": vals})
        empty = [v for v in dict.fromkeys(str(x).strip().strip("'\"") for x in g.get("no_value") or [])
                 if v in held and v not in mapping]
        if not mapping and not empty:
            continue
        new = sql_name(f"{col['name']}_nhom", taken)
        df[new] = df[col["name"]].map(lambda v: None if v in empty else mapping.get(v, v))
        rec = {"name": new, "original": col["original"], "kind": "text", "group_of": col["name"],
               "groups": groups, "no_value": empty, "spec": None, "parsed": None, "failed_examples": []}
        out_cols.insert(out_cols.index(col) + 1, rec)
    return df, out_cols


# ---------------------------------------------------------------- schema shown to the model

SQL_TYPE = {"number": "DOUBLE", "date": "DATE", "datetime": "TIMESTAMP", "text": "VARCHAR"}


def _fmt(v) -> str:
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        t = pd.Timestamp(v)
        return t.strftime("%Y-%m-%d") if t == t.normalize() else t.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return f"{int(v)}"
    return str(v)


def column_doc(df: pd.DataFrame, col: dict) -> str:
    s = df[col["name"]].dropna()
    parts = [f'"{col["original"]}"']
    if col.get("group_of"):
        parts = [f'{col["group_of"]} đã gom các cách ghi cùng nghĩa'] + \
                [f"'{g['label']}' gồm " + ", ".join(f"'{v}'" for v in g["values"]) for g in col["groups"]] + \
                ([f"NULL khi {col['group_of']} là " + ", ".join(f"'{v}'" for v in col["no_value"]) + " (không ghi)"]
                 if col["no_value"] else []) + ["lọc, đếm, xếp hạng theo cột này"]
    if not len(s):
        return f"  {col['name']} {SQL_TYPE[col['kind']]},  -- \"{col['original']}\" · luôn trống"
    if col.get("filled"):
        parts.append(f"{col['filled']:.0%} ô trong file để trống, đã điền xuống theo dòng đầu nhóm")
    if len(df) and (empty := 1 - len(s) / len(df)) > 0:
        parts.append(f"trống {empty:.0%}" if empty >= 0.01 else "trống <1%")
    if col["kind"] in ("number", "date", "datetime") and len(s):
        parts.append(f"từ {_fmt(s.min())} đến {_fmt(s.max())}")
        if col.get("failed_examples"):
            parts.append(f"{1 - col['parsed']:.2%} ô không đọc được (NULL), ví dụ {', '.join(col['failed_examples'][:3])}")
    vc = s.value_counts()
    if col["kind"] == "text":
        if len(vc) <= LOW_CARD:
            parts.append(f"{len(vc)} giá trị: " + ", ".join(f"'{v}' ({n})" for v, n in vc.items()))
        else:
            parts.append(f"{len(vc)} giá trị khác nhau trên {len(s)} dòng, ví dụ " + ", ".join(f"'{v}'" for v in vc.index[:3]))
            if 1.1 <= (per := len(s) / len(vc)) <= GROUP_ROWS:
                parts.append(f"mỗi giá trị trung bình {per:.1f} dòng: đếm số {col['original']} thì dùng COUNT(DISTINCT {col['name']}), không đếm dòng")
        if col.get("rejected"):
            parts.append(f"để dạng chữ vì chỉ {col['parsed']:.0%} ô đọc được thành {col['rejected']}")
    elif col["kind"] == "number" and len(vc) <= 12:
        parts.append("giá trị: " + ", ".join(_fmt(v) for v in vc.index))
    return f"  {col['name']} {SQL_TYPE[col['kind']]},  -- " + " · ".join(parts)


def table_doc(name: str, df: pd.DataFrame, cols: list[dict], tp: TableProfile, copies: int = 0) -> str:
    lines = [f"-- {tp.name}: {len(df)} dòng" + (f"; {len(tp.sources)} sheet gộp lại, cột sheet là nguồn" if len(tp.sources) > 1 else "")
             + (f"; đã bỏ {copies} dòng giống hệt một dòng khác ở mọi ô (giữ một)" if copies else "")]
    for s in tp.suspect:
        lines.append(f"-- ⚠ không chắc cách đọc: {s}")
    lines.append(f"CREATE TABLE {name} (")
    for col in cols:
        lines.append(column_doc(df, col))
        if col.get("raw"):
            lines.append(f"  {col['raw']} VARCHAR,  -- \"{col['original']}\" nguyên văn trong file")
    lines.append(");")
    return "\n".join(lines)


CAL_DOC = ("CREATE TABLE cal (  -- lịch\n  date DATE, weekday INTEGER,  -- 1 = thứ Hai … 7 = Chủ nhật\n"
           "  lunar_day INTEGER, lunar_month INTEGER, lunar_year INTEGER, lunar_leap BOOLEAN, holiday VARCHAR\n);")


# ---------------------------------------------------------------- one business


def build_database(tables: dict, profiles: dict, readme: str, calendar: Path, out: Path, llm, rules: str = "") -> dict:
    """Writes out/db.duckdb, out/spec.json, out/schema.md. Returns the spec record.
    rules: the rulebook text the fill-down choice is made against."""
    db_path = out / "db.duckdb"
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    record, docs, taken = {}, [], {"cal"}
    try:
        for tname, t in tables.items():
            tp = profiles[tname]
            fill = ask_fill(tname, t.df, rules, llm)
            raw, copies = drop_copies(fill_down(t.df, fill["fill_down"]))
            spec = settle_spec(raw, ask_spec(tp, readme, llm), llm)
            df, cols = apply_spec(raw, spec)
            grouping = ask_groups(tname, df, cols, rules, llm)
            df, cols = apply_groups(df, cols, grouping)
            for col in cols:
                if col["original"] in fill["fill_down"] and not col.get("group_of"):
                    col["filled"] = round(float(_blank(t.df[col["original"]]).mean()), 4)
            sname = sql_name(tname, taken)
            con.register("_df", df)
            con.execute(f"CREATE TABLE {sname} AS SELECT * FROM _df")
            con.unregister("_df")
            docs.append(table_doc(sname, df, cols, tp, copies))
            record[tname] = {"sql_name": sname, "columns": cols, "retried": spec.get("retried", {}), "fill": fill,
                             "copies_dropped": copies,
                             "grouping": grouping,
                             "error": spec.get("error"),
                             "usage": spec["usage"], "seconds": spec["seconds"]}
        cal = pd.read_csv(calendar, parse_dates=["date"])
        cal["date"] = cal["date"].dt.date
        con.register("_cal", cal)
        con.execute("CREATE TABLE cal AS SELECT * FROM _cal")
        con.unregister("_cal")
    finally:
        con.close()
    (out / "spec.json").write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    schema = f"## README\n{readme.strip()}\n\n## Bảng (DuckDB)\n" + "\n\n".join(docs) + "\n\n" + CAL_DOC
    (out / "schema.md").write_text(schema, encoding="utf-8")
    return record

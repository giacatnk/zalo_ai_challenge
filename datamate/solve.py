"""Question layer: an agent loop with one SQL tool over the business's typed database.

Context per question: the rulebook minus the ### groups not picked for the business and minus
the answer-format table the formatter implements (system; the same for every question of a
business, so vLLM prefix caching keeps it cheap), then today's date, the question and the
schema (README + CREATE TABLE with one comment per column).

Tool
  sql(query)            runs one SELECT on a read-only DuckDB connection, returns the row count
                        and the first rows
Answer
  a reply without a tool call, holding {"type", "query", "money"}: the formatter takes values
  straight from the query's result (the model never copies a number by hand) and applies the
  §9 formats and rounding, which are the same in every split. Checks that fail go back to the
  model; a valid answer is final.

Before the agent, question.normalize rewrites the question and states its periods (Q1).
Every prompt of a question is written to <cache>/prompts/<id>.md.

Not here yet: abstain policy.

uv run python -m datamate.solve --split data_mate/train --cache cache/train --sample 8
uv run python -m datamate.solve --split data_mate/train --cache cache/train --ids q_001883280e83
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import difflib
import json
import math
import random
import re
import threading
import time
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import sqlglot
from sqlglot import exp

from .question import dump_markdown, load_time_sections, normalize, render_periods, time_rules
from .rulebook import Rulebook, Unit
from .text import norm_name, norm_text

MAX_STEPS = 8
GROUP_ROWS = 20         # a text column with at most this many rows per value groups rows (an order of lines)
LOW_VALUES = 30         # a column this small may be a join key the ON forgot (a unit, a kind)
KIND_VALUES = 12        # grouped columns this small are kinds (statuses...) shown when a query ignores them
VOTE_TEMPERATURE = 0.7  # sampled runs of the vote (the first run is greedy)
SHOW_ROWS = 20
OUTPUT_CHARS = 2000
QUERY_SECONDS = 20
LOW_CARD = 30          # text columns with at most this many values: "values not considered" feedback
NEAR_CARD = 5000       # text columns with at most this many values: nearest-value suggestions
WEEKDAY_VI = ["thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm", "thứ Sáu", "thứ Bảy", "Chủ nhật"]
TYPES = ("number", "percent", "date", "boolean", "string", "list", "table", "unanswerable")

TOOLS = [
    {"type": "function", "function": {
        "name": "sql",
        "description": "Chạy một câu SELECT (DuckDB) trên dữ liệu cửa hàng. Trả về số dòng và các dòng đầu.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]

SYSTEM = """Bạn là trợ lý phân tích dữ liệu bán hàng của một cửa hàng nhỏ. Trả lời câu hỏi bằng truy vấn SQL (DuckDB) trên dữ liệu của cửa hàng.

Tool:
  sql(query)  chạy một câu SELECT, trả về số dòng và tối đa 20 dòng đầu.

Đáp án: khi đã chắc, trả lời (không gọi tool) đúng một JSON:
  {"type": "...", "query": "SELECT ...", "money": true/false}
  - query: câu SELECT tính ra đáp án từ bảng dữ liệu; kết quả của câu đó chính là đáp án, không ghi số bằng tay.
  - type: number, percent, date, boolean, string: kết quả đúng 1 dòng 1 cột. list: 1 cột (hoặc 1 dòng), đúng thứ tự (ORDER BY).
    table: cột đầu là tên đối tượng, các cột sau là số đo.
  - money: true nếu số trong đáp án là tiền (kể cả trung bình của tiền), ngược lại false.
  - Không trả lời được: {"type": "unanswerable", "reason": "..."}
  Định dạng và làm tròn do hệ thống lo, câu SQL không cần ROUND.

Cách làm:
- Đọc câu hỏi, rulebook và schema; xác định các rule câu hỏi cần (định nghĩa, khoảng thời gian, đơn hợp lệ…).
- Cột tên kết thúc bằng _nhom (nếu có) là cột gốc đã gom các cách ghi cùng nghĩa theo rulebook, NULL ở các ô chỉ ghi thay cho "không có"; lọc, đếm, xếp hạng theo cột _nhom thay cho cột gốc.
- Lọc theo cột có ít giá trị (trạng thái, kênh…): schema liệt kê đủ mọi giá trị. Ghi rõ từng giá trị cần bỏ hoặc giữ bằng IN (...), xét cả cách ghi tiếng Anh, viết tắt; không lọc trạng thái bằng LIKE.
- So tên không phân biệt hoa thường và dấu: lower(strip_accents(cot)) = lower(strip_accents('tên')).
- Chạy sql(...) để kiểm tra trước khi nộp: số dòng sau bộ lọc, giá trị đã giữ / bỏ, khoảng ngày đã dùng. Ra 0 dòng thường là lọc sai.

Rulebook:
{rules}"""

ASK_JSON = ('Chưa có đáp án. Gọi sql để kiểm tra, hoặc trả lời đúng một JSON '
            '{"type": "...", "query": "SELECT ...", "money": true/false} (hoặc {"type": "unanswerable", "reason": "..."}).')


# ---------------------------------------------------------------- answers


def _round(f: float, places: int):
    """Half away from zero, on the decimal value (round(2.675, 2) would give 2.67)."""
    q = Decimal(repr(f)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return int(q) if q == q.to_integral_value() else float(q)


def _scalar(v, money: bool = False):
    """One answer value. §9: money rounds to whole đồng, every other number to 2 decimals."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float, Decimal)):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            raise ValueError("giá trị là NULL, NaN hoặc vô cực")
        return _round(f, 0 if money else 2)
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    if v is None:
        raise ValueError("giá trị là NULL")
    return v


def format_answer(type_: str, value, reason=None, money: bool = False) -> dict:
    """§9 formats. Raises ValueError with a message the agent can act on."""
    if type_ not in TYPES:
        raise ValueError(f"type phải là một trong {TYPES}")
    if type_ == "unanswerable":
        return {"type": "unanswerable", "reason": reason or ""}
    if type_ in ("number", "percent"):
        if isinstance(value, (str, bool)):
            raise ValueError(f"{type_} phải là một số")
        return {"type": type_, "answer": _scalar(float(value), money and type_ == "number")}
    if type_ == "date":
        v = _scalar(value)
        if not isinstance(v, str):
            raise ValueError('date phải là ngày, hoặc chuỗi "YYYY-MM-DD" / "YYYY-MM"')
        return {"type": "date", "answer": v}
    if type_ == "boolean":
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError("boolean phải là true hoặc false")
        return {"type": "boolean", "answer": bool(value)}
    if type_ == "string":
        return {"type": "string", "answer": str(_scalar(value)).strip()}
    if type_ == "list":
        return {"type": "list", "answer": [_scalar(v, money) for v in value]}
    df = value
    return {"type": "table", "answer": {"columns": [str(c) for c in df.columns],
                                        "rows": [[_scalar(v, money) for v in r] for r in df.itertuples(index=False)]}}


def _same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-6
    if isinstance(a, str) and isinstance(b, str):
        return norm_name(a) == norm_name(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def score(pred: dict | None, gold: dict) -> int:
    """+1 / -1 / 0 (abstain). Approximates the organisers' check: type first, then value;
    tables compared row-set wise; any alias accepted."""
    if pred is None or pred.get("type") == "abstain":
        return 0
    if pred["type"] != gold["type"]:
        return -1
    if gold["type"] == "unanswerable":
        return 1
    golds = [gold["answer"], *gold.get("alias", [])]
    if gold["type"] == "table":
        p = pred["answer"]
        rows = lambda t: sorted(json.dumps([norm_name(x) if isinstance(x, str) else x for x in r]) for r in t["rows"])
        return 1 if any(rows(p) == rows(g) for g in golds) else -1
    return 1 if any(_same(pred["answer"], g) for g in golds) else -1


# ---------------------------------------------------------------- database tools


class SqlBox:
    def __init__(self, db_path: Path):
        self.con = duckdb.connect(str(db_path), read_only=True, config={"enable_external_access": False})
        self.result: dict | None = None
        self.warned: set = set()                             # queries already told about a period / value problem
        self.values = self._text_values()
        self.dates = {}                                      # table -> its date / timestamp columns
        for t, c in self.con.execute("SELECT table_name, column_name FROM information_schema.columns "
                                     "WHERE data_type = 'DATE' OR data_type LIKE 'TIMESTAMP%'").fetchall():
            self.dates.setdefault(t, []).append(c)

    def _text_values(self) -> dict[tuple[str, str], list[str]]:
        """Distinct values of every text column with at most NEAR_CARD of them, most frequent first."""
        out = {}
        cols = self.con.execute("SELECT table_name, column_name FROM information_schema.columns "
                                "WHERE data_type = 'VARCHAR' AND table_name <> 'cal'").fetchall()
        for t, c in cols:
            n = self.con.execute(f'SELECT COUNT(DISTINCT "{c}") FROM "{t}"').fetchone()[0]
            if 0 < n <= NEAR_CARD:
                rows = self.con.execute(f'SELECT "{c}" FROM "{t}" WHERE "{c}" IS NOT NULL '
                                        f'GROUP BY 1 ORDER BY COUNT(*) DESC').fetchall()
                out[(t, c)] = [r[0] for r in rows]
        return out

    # -- feedback taken from the data, about the query the model just wrote

    @staticmethod
    def _literals(sql: str) -> tuple[list[tuple[str, frozenset]], list[tuple[str, bool, frozenset]]]:
        """((string, column names) compared by =, <>, IN; (pattern, ignore case, column names) of
        LIKE / ILIKE). Other strings (format strings, aliases, function arguments) are not data
        values."""
        try:
            tree = sqlglot.parse_one(sql, read="duckdb")
        except Exception:
            return [], []
        def text(e):
            return e.this if isinstance(e, exp.Literal) and e.is_string else None
        def cols(e):
            return frozenset(c.name for c in ([e] if isinstance(e, exp.Column) else e.find_all(exp.Column)))
        eq, like = [], []
        for node in tree.find_all(exp.EQ, exp.NEQ, exp.In, exp.Like, exp.ILike):
            if isinstance(node, exp.In):
                if names := cols(node.this):
                    eq += [(t, names) for e in node.expressions if (t := text(e)) is not None]
            elif isinstance(node, (exp.Like, exp.ILike)):
                if (names := cols(node.this)) and (t := text(node.expression)) is not None:
                    like.append((t, isinstance(node, exp.ILike), names))
            else:
                for a, b in ((node.this, node.expression), (node.expression, node.this)):
                    if (t := text(b)) is not None and (names := cols(a)):
                        eq.append((t, names))
        keep = lambda l: len(l.strip()) >= 2 and not re.fullmatch(r"[\d\s:./-]+", l)
        return [x for x in dict.fromkeys(eq) if keep(x[0])], list(dict.fromkeys(like))

    @staticmethod
    def _accent_free(sql: str, lit: str) -> bool:
        """Every use of the literal sits inside strip_accents(...), so it only has to match without accents."""
        quoted = "'" + lit.replace("'", "''") + "'"
        starts = [m.start() for m in re.finditer(re.escape(quoted), sql)]
        return bool(starts) and all(re.search(r"strip_accents\s*\(\s*(lower\s*\(\s*)?$", sql[:i], re.I) for i in starts)

    def literal_hints(self, sql: str) -> list[str]:
        """String literals of the query that no text column holds verbatim (or, inside
        strip_accents(...), accent-free), with the nearest values."""
        hints = []
        eq, like = self._literals(sql)
        pool = lambda names: {k: v for k, v in self.values.items() if k[1] in names}   # the compared columns' values
        for pat, nocase, names in like:
            values = pool(names)
            if not values:                                  # a column whose values are not listed: nothing to check
                continue
            rx = re.compile("".join(".*" if ch == "%" else "." if ch == "_" else re.escape(ch) for ch in pat),
                            re.I | re.S if nocase else re.S)
            if not any(rx.fullmatch(v) for vals in values.values() for v in vals):
                hints.append(f"mẫu {'ILIKE' if nocase else 'LIKE'} '{pat}' không khớp giá trị nào trong dữ liệu")
            elif not nocase:                                 # the same pattern ignoring case finds more values
                loose = re.compile(rx.pattern, re.I | re.S)
                more = [f"{c} = '{v}'" for (t, c), vals in values.items() for v in vals
                        if loose.fullmatch(v) and not rx.fullmatch(v)]
                if more:
                    hints.append(f"LIKE '{pat}' phân biệt hoa thường; không phân biệt (ILIKE) thì khớp thêm {len(more)} giá trị: "
                                 + "; ".join(more[:4]))
        for lit, names in eq:
            values = pool(names)
            if not values or any(lit in vals for vals in values.values()):
                continue
            key = norm_text(lit)
            same = [f"{c} = '{v}'" for (t, c), vals in values.items() for v in vals if norm_text(v) == key]
            if same and self._accent_free(sql, lit):
                continue
            if same:
                hints.append(f"'{lit}' không có nguyên văn trong dữ liệu; cùng chữ khi bỏ hoa thường và dấu: " + "; ".join(same[:4]))
                continue
            near = []
            for (t, c), vals in values.items():
                by = {norm_text(v): v for v in vals}
                near += [f"{c} = '{by[k]}'" for k in difflib.get_close_matches(key, list(by), n=2, cutoff=0.75)]
            hints.append(f"'{lit}' không có trong {', '.join(sorted(names))}" +
                         (f"; gần nhất: {'; '.join(near[:4])}" if near else
                          "; các giá trị: " + ", ".join(f"'{v}'" for vals in values.values() for v in vals[:12])
                          if sum(map(len, values.values())) <= 12 else ""))
        return hints

    def scopes(self, sql: str) -> list[dict]:
        """For each SELECT that reads a data table: the rows its FROM/WHERE keeps and the dates
        those rows span, measured by running that FROM/WHERE again."""
        try:
            tree = sqlglot.parse_one(sql, read="duckdb")
        except Exception:
            return []
        out = []
        for i, sel in enumerate(tree.find_all(exp.Select)):
            sources = [sel.args.get("from_")] + list(sel.args.get("joins") or [])
            tables = [t for src in sources if src is not None for t in [src.this] if isinstance(t, exp.Table)]
            if not tables or all(t.name == "cal" for t in tables):
                continue
            cols = [(t.alias_or_name, c) for t in tables for c in self.dates.get(t.name, [])]
            where = sel.args.get("where")
            used = {c.name for c in where.find_all(exp.Column)} if where else set()
            shown = [tc for tc in cols if tc[1] in used] or cols[:1]
            if not shown:
                continue
            probe = sel.copy()
            for k in ("group", "having", "order", "limit", "offset", "qualify", "distinct", "windows"):
                probe.set(k, None)
            a, c = shown[0]
            probe.set("expressions", [sqlglot.parse_one(e, read="duckdb") for e in
                                      ("COUNT(*)", f"MIN({a}.{c})", f"MAX({a}.{c})", f"COUNT(DISTINCT CAST({a}.{c} AS DATE))")])
            if tree.args.get("with") and probe.args.get("with") is None:
                probe.set("with", tree.args["with"].copy())
            try:
                rows, lo, hi, days = self.query(probe.sql(dialect="duckdb")).iloc[0].tolist()
            except Exception:
                continue
            by_cal = where is not None and any(
                c.table in {t.alias_or_name for t in tables if t.name == "cal"} for c in where.find_all(exp.Column))
            out.append({"name": "câu chính" if i == 0 else "truy vấn con", "tables": [t.name for t in tables],
                        "column": c, "rows": int(rows), "days": int(days),
                        "from": None if pd.isna(lo) else pd.Timestamp(lo).date(),
                        "to": None if pd.isna(hi) else pd.Timestamp(hi).date(),
                        "filtered": by_cal or bool(used & {c for _, c in cols})})
        return out

    def _kind_columns(self) -> dict[str, list[str]]:
        """table -> its <col>_nhom columns with at most KIND_VALUES values (statuses and the like,
        grouped along the rulebook at build time)."""
        if not hasattr(self, "_kinds"):
            self._kinds = {}
            for t, c in self.con.execute("SELECT table_name, column_name FROM information_schema.columns "
                                         "WHERE column_name LIKE '%\\_nhom' ESCAPE '\\'").fetchall():
                n = self.con.execute(f'SELECT COUNT(DISTINCT "{c}") FROM "{t}"').fetchone()[0]
                if 2 <= n <= KIND_VALUES:
                    self._kinds.setdefault(t, []).append(c)
        return self._kinds

    def _group_ids(self, table: str) -> list[tuple[str, float]]:
        """Text columns of a table whose values each cover a few rows (an order over its lines):
        (column, rows per value), as the schema notes them."""
        cache = self.__dict__.setdefault("_ids", {})
        if table not in cache:
            cache[table] = []
            for (c,) in self.con.execute("SELECT column_name FROM information_schema.columns WHERE table_name = ? "
                                         "AND data_type = 'VARCHAR'", [table]).fetchall():
                n, d = self.con.execute(f'SELECT COUNT("{c}"), COUNT(DISTINCT "{c}") FROM "{table}"').fetchone()
                if d > LOW_VALUES and 1.1 <= n / d <= GROUP_ROWS:
                    cache[table].append((c, n / d))
        return cache[table]

    def count_problems(self, sql: str) -> list[str]:
        """COUNT(*) or COUNT(col) over a table where one id covers several rows: it counts rows,
        not ids."""
        try:
            tree = sqlglot.parse_one(sql, read="duckdb")
        except Exception:
            return []
        out = []
        for sel in tree.find_all(exp.Select):
            counts = [c for e in sel.expressions for c in e.find_all(exp.Count) if not isinstance(c.this, exp.Distinct)]
            if not counts:
                continue
            sources = [sel.args.get("from_")] + list(sel.args.get("joins") or [])
            for t in [src.this for src in sources if src is not None and isinstance(src.this, exp.Table)]:
                for col, per in self._group_ids(t.name):
                    out.append(f"{counts[0].sql(dialect='duckdb')} trên {t.name} đếm dòng; mỗi {col} trung bình {per:.1f} dòng, "
                               f"đếm số {col} thì dùng COUNT(DISTINCT {col})")
        return out

    def kind_problems(self, sql: str) -> list[str]:
        """For each SELECT reading a table with grouped kind columns that the query never mentions
        (nor their source column): the rows its FROM/WHERE keeps, split by that column."""
        try:
            tree = sqlglot.parse_one(sql, read="duckdb")
        except Exception:
            return []
        named = {c.name for c in tree.find_all(exp.Column)}
        out = []
        for i, sel in enumerate(tree.find_all(exp.Select)):
            sources = [sel.args.get("from_")] + list(sel.args.get("joins") or [])
            tables = [t for src in sources if src is not None for t in [src.this] if isinstance(t, exp.Table)]
            for t in tables:
                for col in self._kind_columns().get(t.name, []):
                    if col in named or col.removesuffix("_nhom") in named:
                        continue
                    probe = sel.copy()
                    for k in ("group", "having", "order", "limit", "offset", "qualify", "distinct", "windows"):
                        probe.set(k, None)
                    ref = f"{t.alias_or_name}.{col}"
                    probe.set("expressions", [sqlglot.parse_one(e, read="duckdb") for e in (ref, "COUNT(*)")])
                    probe.set("group", exp.Group(expressions=[sqlglot.parse_one(ref, read="duckdb")]))
                    if tree.args.get("with") and probe.args.get("with") is None:
                        probe.set("with", tree.args["with"].copy())
                    try:
                        df = self.query(probe.sql(dialect="duckdb"))
                    except Exception:
                        continue
                    if len(df) > 1:
                        rows = df.sort_values(df.columns[1], ascending=False).values.tolist()
                        out.append(f"{'câu chính' if i == 0 else 'truy vấn con'} (FROM {t.name}) không lọc theo {col}, "
                                   "đang lấy: " + ", ".join(f"'{v}' {n} dòng" for v, n in rows))
        return out

    def join_problems(self, sql: str) -> list[str]:
        """JOINs whose ON leaves out a column pair that matches by value: a small text column of
        one side whose every value is also a value of a column of the other side (a unit, a kind)."""
        try:
            tree = sqlglot.parse_one(sql, read="duckdb")
        except Exception:
            return []
        out = []
        for sel in tree.find_all(exp.Select):
            base = sel.args.get("from_")
            left = [base.this] if base is not None and isinstance(base.this, exp.Table) else []
            for j in sel.args.get("joins") or []:
                right, on = j.this, j.args.get("on")
                if not isinstance(right, exp.Table) or on is None:
                    continue
                used = {c.name for c in on.find_all(exp.Column)}
                for lt in left:
                    for (t1, c1), v1 in self.values.items():
                        if t1 != right.name or c1 in used or not 0 < len(v1) <= LOW_VALUES:
                            continue
                        for (t2, c2), v2 in self.values.items():
                            if t2 == lt.name and c2 not in used and set(v1) <= set(v2):
                                out.append(f"JOIN {right.name} ON chưa nối {right.name}.{c1} với {lt.name}.{c2}, "
                                           f"dù mọi giá trị của {c1} ({', '.join(repr(v) for v in v1[:5])}) đều có trong {c2}")
                left.append(right)
        return out

    def period_problems(self, sql: str, periods: list[dict]) -> list[str]:
        """Disagreements between the periods the question step stated and the dates the query
        reads: no date filter at all, dates outside every period, or one day of a longer period."""
        if not periods:
            return []
        spans = [(date.fromisoformat(p["from"]), date.fromisoformat(p["to"]), p["label"]) for p in periods]
        out = []
        for sc in self.scopes(sql):
            where = f"{sc['name']} (FROM {', '.join(sc['tables'])})"
            if not sc["filtered"]:
                out.append(f"{where} không lọc theo ngày, đang lấy {sc['rows']} dòng từ {sc['from']} đến {sc['to']}")
            elif sc["from"] is not None:
                hit = [sp for sp in spans if sp[0] <= sc["from"] and sc["to"] <= sp[1]]
                if not hit:
                    out.append(f"{where} lấy {sc['column']} từ {sc['from']} đến {sc['to']}, không nằm trong kỳ nào")
                elif sc["days"] <= 1 and (hit[0][1] - hit[0][0]).days >= 6:
                    out.append(f"{where} chỉ lấy ngày {sc['from']}, trong khi kỳ {hit[0][2]} là {hit[0][0]} đến {hit[0][1]}")
        return out

    def close(self) -> None:
        self.con.close()

    def query(self, sql: str) -> pd.DataFrame:
        timer = threading.Timer(QUERY_SECONDS, self.con.interrupt)
        timer.start()
        try:
            return self.con.execute(sql).df()
        finally:
            timer.cancel()

    def run_sql(self, sql: str) -> str:
        try:
            df = self.query(sql)
        except Exception as e:
            return f"LỖI: {type(e).__name__}: {str(e).strip()[:600]}"
        with pd.option_context("display.max_columns", 50, "display.width", 250):
            body = df.head(SHOW_ROWS).to_string(index=False) if len(df) else "(không có dòng nào)"
        out = f"{len(df)} dòng\n{body}"
        if hints := self.literal_hints(sql):                 # a value typed wrong filters nothing
            out += "\nCHÚ Ý: " + "\n".join(hints)
        return out if len(out) <= OUTPUT_CHARS else out[:OUTPUT_CHARS] + "\n… (cắt)"

    def _value(self, type_: str, sql: str):
        df = self.query(sql)
        r, c = df.shape
        if type_ in ("number", "percent", "date", "boolean", "string"):
            if (r, c) != (1, 1):
                raise ValueError(f"{type_} cần kết quả đúng 1 dòng 1 cột, câu SQL đang ra {r} dòng × {c} cột")
            return df.iat[0, 0]
        if type_ == "list":
            if c == 1:
                return df.iloc[:, 0].tolist()
            if r == 1:
                return df.iloc[0].tolist()
            raise ValueError(f"list cần 1 cột (hoặc 1 dòng), câu SQL đang ra {r} dòng × {c} cột")
        return df

    def submit(self, a: dict, periods: list[dict] | None = None) -> str | None:
        """Checks and formats a JSON answer. None when it is accepted (self.result is set),
        else the message that goes back to the model. A value or period problem is raised once per
        query: sent again unchanged, the query is accepted (the check can be wrong)."""
        type_, sql = a.get("type", ""), a.get("query") or ""
        if type_ != "unanswerable" and not re.search(r"\bfrom\b", sql, re.I):
            return "LỖI: query của đáp án phải tính từ bảng dữ liệu (có FROM), không ghi thẳng giá trị."
        key = " ".join(sql.split()).lower()
        if type_ != "unanswerable" and ("v", key) not in self.warned and (bad := self.literal_hints(sql)):
            self.warned.add(("v", key))
            return ("KIỂM TRA GIÁ TRỊ: so các giá trị chữ của query với dữ liệu (giá trị lọc phải chép đúng từng chữ):\n" + "\n".join(bad) +
                    "\nNếu query đã đúng thì gửi lại y nguyên JSON đó.")
        if type_ != "unanswerable" and key not in self.warned and (probs := self.period_problems(sql, periods or [])):
            self.warned.add(key)
            return ("KIỂM TRA KỲ: kỳ của câu hỏi là " + render_periods(periods) + ".\n" + "\n".join(f"- {x}" for x in probs) +
                    "\nSửa query cho đúng kỳ. Nếu query đã đúng thì gửi lại y nguyên JSON đó.")
        if type_ != "unanswerable" and ("k", key) not in self.warned and (kinds := self.kind_problems(sql) + self.join_problems(sql) + self.count_problems(sql)):
            self.warned.add(("k", key))
            return ("KIỂM TRA PHẠM VI: " + "\n".join(f"- {x}" for x in kinds) +
                    "\nXem rulebook: câu hỏi có cần bỏ hay giữ riêng loại nào không. Nếu query đã đúng thì gửi lại y nguyên JSON đó.")
        try:
            value = None if type_ == "unanswerable" else self._value(type_, sql)
            self.result = format_answer(type_, value, a.get("reason"), bool(a.get("money")))
        except Exception as e:
            return f"LỖI: {type(e).__name__}: {str(e).strip()[:600]}"
        return None


# ---------------------------------------------------------------- one question


def _is_format_table(u: Unit) -> bool:
    """A row of the answer-format table: its first cell is one of the submission's answer types.
    format_answer() implements those formats, so the rows stay out of the prompt."""
    return u.kind == "row" and u.title.strip().lower() in TYPES


def render_rulebook(rb: Rulebook, industries: list[str], skip: list[str] = ()) -> str:
    """The rulebook in its own order, minus the ### groups not picked for this business, the
    answer-format table and the sections in skip (the time sections, once the question step
    has turned them into dates)."""
    out, last = [], None
    for u in rb.units:
        if (u.sub is not None and u.sub not in industries) or _is_format_table(u) or u.section in skip:
            continue
        where = f"{u.section} / {u.sub}" if u.sub else u.section
        if where != last:
            out.append(f"\n### {where}")
            last = where
        out.append(f"- {u.text}")
    return "\n".join(out).strip()


def build_messages(q: dict, rules: str, schema: str, norm: dict) -> list[dict]:
    system = SYSTEM.replace("{rules}", rules)
    asof = pd.Timestamp(q["as_of_date"])
    user = (f"Hôm nay: {asof:%Y-%m-%d} ({WEEKDAY_VI[asof.weekday()]}).\n"
            f"Câu hỏi: {q['question']}\n" +
            ("Cách đọc chữ viết tắt, gõ sai (gợi ý, có thể sai): " + "; ".join(f"{h['from']} = {h['to']}" for h in norm["hints"]) + "\n"
             if norm.get("hints") else "") +
            f"Kỳ thời gian của câu hỏi (đã tính sẵn, lọc đúng các ngày này): {render_periods(norm['periods'])}\n\n{schema}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _args(call) -> dict:
    try:
        a = json.loads(call.function.arguments or "{}")
        return a if isinstance(a, dict) else {}
    except json.JSONDecodeError:
        return {}


def _reply_json(text: str) -> dict | None:
    """The answer object in a reply: the last {...} that parses and has a type."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    for m in reversed(list(re.finditer(r"\{.*\}", text, re.S))):
        try:
            a = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(a, dict) and a.get("type"):
            return a
    return None


def solve(q: dict, cache: Path, rb: Rulebook, llm, calendar: Path, think: bool = False, norm_think: bool = True,
          prompts: Path | None = None, sample: dict | None = None, tag: str = "") -> dict:
    """think: thinking for the SQL agent; norm_think: for the question step. sample: temperature /
    seed for every call of this run (default greedy). Prompts are dumped to prompts/<id><tag>.md
    (default <cache>/prompts)."""
    biz = cache / q["business_id"]
    meta = json.loads((biz / "meta.json").read_text(encoding="utf-8"))
    industries = meta["industry"].get("industries") or []
    schema = (biz / "schema.md").read_text(encoding="utf-8")
    box = SqlBox(biz / "db.duckdb")
    time_sections = load_time_sections(cache, rb, llm)
    norm = normalize(q, time_rules(rb, time_sections), calendar, llm, think=norm_think, sample=sample)
    rules = render_rulebook(rb, industries, skip=time_sections)
    messages = build_messages(q, rules, schema, norm)
    steps, turns, t0, tokens = [], [], time.time(), 0
    try:
        for _ in range(MAX_STEPS):
            r = llm.chat(messages, tools=TOOLS, think=think, max_tokens=8000 if think else 2000, **(sample or {}))
            u = r["usage"] or {}
            tokens += u.get("total_tokens", 0)
            calls = r["tool_calls"] or []
            turns.append({"seconds": r["seconds"], "prompt": u.get("prompt_tokens"), "completion": u.get("completion_tokens"),
                          "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                          "calls": [c.function.name for c in calls]})
            if not calls:
                a = _reply_json(r["content"])
                out = ASK_JSON if a is None else box.submit(a, norm["periods"])
                steps.append({"text": r["content"], "output": out or "ĐÃ NỘP"})
                if out is None:
                    break
                messages += [{"role": "assistant", "content": r["content"]}, {"role": "user", "content": out}]
                continue
            messages.append({"role": "assistant", "content": r["content"] or None,
                             "tool_calls": [c.model_dump() for c in calls]})
            for c in calls:
                a = _args(c)
                out = box.run_sql(a.get("query", "")) if c.function.name == "sql" else f"LỖI: không có tool {c.function.name}"
                if out.startswith("LỖI") and any(s.get("call") == [c.function.name, a] for s in steps):
                    out += "\n(Giống hệt một lần gọi trước nên lỗi y như cũ. Sửa theo thông báo lỗi.)"
                steps.append({"call": [c.function.name, a], "output": out})
                messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
    finally:
        box.close()
    res = {"id": q["id"], "business_id": q["business_id"], "question": q["question"],
           "normalized": norm["question"], "fixes": norm["fixes"], "hints": norm.get("hints"), "periods": norm["periods"], "period_reason": norm["reason"],
           "as_of_date": q["as_of_date"], "industries": industries, "pred": box.result,
           "steps": steps, "turns": turns, "seconds": round(time.time() - t0, 1), "tokens": tokens,
           "prompt_chars": sum(len(m["content"] or "") for m in messages[:2])}
    dump_markdown((prompts or cache / "prompts") / f"{q['id']}{tag}.md", q, norm, messages, TOOLS, box.result)
    return res


def _vote_key(pred: dict | None) -> str | None:
    if not pred:
        return None
    if pred.get("type") == "unanswerable":
        return "unanswerable"
    return json.dumps(pred, ensure_ascii=False, sort_keys=True, default=str)


def solve_vote(q: dict, runs: int, **kw) -> dict:
    """Execution-based self-consistency: the run greedy plus runs - 1 sampled runs, and the
    answer most runs reach; a tie goes to the earliest run (the greedy one first). A run with no
    answer does not vote."""
    if runs <= 1:
        return solve(q, **kw)
    plans = [None] + [{"temperature": VOTE_TEMPERATURE, "seed": i} for i in range(1, runs)]
    with ThreadPoolExecutor(max_workers=runs) as pool:
        done = list(pool.map(lambda i: solve(q, **kw, sample=plans[i], tag=f"_s{i}" if i else ""), range(runs)))
    keys = [_vote_key(r["pred"]) for r in done]
    counts = {k: keys.count(k) for k in keys if k is not None}
    best = max(counts, key=lambda k: (counts[k], -keys.index(k))) if counts else None
    pick = done[keys.index(best)] if best is not None else done[0]
    return {**pick, "votes": counts, "samples": [r["pred"] for r in done],
            "seconds": round(sum(r["seconds"] for r in done), 1), "tokens": sum(r["tokens"] for r in done)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None, help="jsonl log (default: <cache>/solve_log.jsonl)")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--ids-file", help="file with one question id per line")
    ap.add_argument("--workers", type=int, default=1, help="questions solved at once")
    ap.add_argument("--vote", type=int, default=1, help="runs per question, answer by majority (1 = greedy only)")
    ap.add_argument("--biz", nargs="*", help="sample --sample questions from each of these businesses")
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", choices=["openrouter", "vllm"], default="openrouter")
    ap.add_argument("--model")
    ap.add_argument("--think", action="store_true", help="thinking on for the SQL agent (default off)")
    ap.add_argument("--norm-no-think", action="store_true", help="thinking off for the question step (default on)")
    a = ap.parse_args()

    from .llm import LLM
    split, cache = Path(a.split), Path(a.cache)
    qs = [json.loads(l) for l in open(split / "questions.jsonl", encoding="utf-8")]
    gold = {g["id"]: g for g in map(json.loads, open(split / "answers.jsonl", encoding="utf-8"))} \
        if (split / "answers.jsonl").exists() else {}
    if a.ids_file:
        a.ids = Path(a.ids_file).read_text().split()
    if a.ids:
        qs = [q for q in qs if q["id"] in set(a.ids)]
    elif a.biz:
        rng = random.Random(a.seed)
        qs = [q for b in a.biz for q in rng.sample([q for q in qs if q["business_id"] == b], a.sample)]
    else:
        qs = random.Random(a.seed).sample(qs, a.sample)
    rb = Rulebook((split / "rulebook.md").read_text(encoding="utf-8"))
    llm = LLM(a.llm, model=a.model)
    out = Path(a.out) if a.out else cache / "solve_log.jsonl"
    total = 0
    load_time_sections(cache, rb, llm)                       # made once, before the threads read it
    run = lambda q: solve_vote(q, a.vote, cache=cache, rb=rb, llm=llm, calendar=split / "calendar.csv", think=a.think,
                               norm_think=not a.norm_no_think, prompts=cache / f"prompts_{out.stem}")
    with open(out, "w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=a.workers) as pool:
        for q, res in zip(qs, pool.map(run, qs)):
            g = gold.get(q["id"])
            res["gold"], res["score"] = g, (score(res["pred"], g) if g else None)
            total += res["score"] or 0
            f.write(json.dumps(res, ensure_ascii=False, default=str) + "\n")
            f.flush()
            pa = res["pred"]["answer"] if res["pred"] and "answer" in res["pred"] else res["pred"]
            ga = g["answer"] if g and "answer" in g else g
            print(f"[{res['score']:+d}] {q['id']} {q['business_id']} {len(res['steps'])} bước {res['seconds']}s "
                  f"| {q['question'][:70]}\n      pred={json.dumps(pa, ensure_ascii=False)[:120]} "
                  f"gold={json.dumps(ga, ensure_ascii=False)[:120]}")
    print(f"score {total} / {len(qs)}   log: {out}")


if __name__ == "__main__":
    main()

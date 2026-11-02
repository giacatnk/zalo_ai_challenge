"""Business layer B3–B4: the EDA note (a fixed template, no LLM), the entity index, and the
one LLM call per business that picks its industry sections of the rulebook.

The note is split per table so the question layer can pick the tables a question needs.
"""

from __future__ import annotations

import re

from .profile import ColumnProfile, TableProfile
from .rulebook import Rulebook

STORED_VI = {"number": "lưu dạng số", "datetime": "lưu dạng ngày giờ", "mixed": "lưu lẫn số và chữ"}
MAX_VALUES_SHOWN = 12


def _n(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%" if x >= 0.01 else "<1%"


def render_column(c: ColumnProfile) -> str:
    parts = [STORED_VI[c.stored]] if c.stored in STORED_VI else []
    if c.null_ratio > 0:
        parts.append(f"trống {_pct(c.null_ratio)}")
    # formats only where the cells hold digits ("45,000.0000", "1 ly", "01/04/2025"); for names
    # the shapes are noise, and a column of real numbers has a single shape
    show = any("9" in s or s.startswith("<") for s, _, _ in c.formats)
    if show and (len(c.formats) > 1 or (c.formats and c.formats[0][0] != "<số>")):
        parts.append("dạng " + " · ".join(f'"{ex}" {_pct(sh)}' for _, sh, ex in c.formats[:4]))
    if c.values is not None:
        shown = ", ".join(f"{v} ({_n(k)})" for v, k in c.values[:MAX_VALUES_SHOWN])
        more = f", … {len(c.values) - MAX_VALUES_SHOWN} giá trị khác" if len(c.values) > MAX_VALUES_SHOWN else ""
        parts.append(f"{c.n_unique} giá trị: {shown}{more}")
    else:
        parts.append(f"{_n(c.n_unique)} giá trị, ví dụ " + ", ".join(c.samples))
    if c.min is not None and c.values is None:
        parts.append(f"từ {c.min} đến {c.max}")
    if c.repeated and c.values is None and c.letter_share >= 0.5:
        parts.append("lặp lại trên nhiều dòng")
    if c.variants:
        ex = " ~ ".join(f'"{v}"' for v in c.variants["examples"][0]) if c.variants["examples"] else ""
        parts.append(f"{c.variants['raw']} cách viết, {c.variants['norm']} tên sau khi bỏ hoa thường và dấu: {ex}")
    return f"- {c.name}: " + " · ".join(parts)


def render_table(tp: TableProfile) -> str:
    head = [f"### {tp.name}", f"{_n(tp.rows)} dòng"]
    if len(tp.sources) > 1:
        head.append(f"gộp {len(tp.sources)} sheet: {tp.sources[0]} … {tp.sources[-1]} (cột _sheet)")
    elif tp.sources:
        head.append(f"từ {tp.sources[0]}")
    lines = [" · ".join(head)]
    for s in tp.suspect:
        lines.append(f"⚠ Không chắc cách đọc: {s}. Xem bảng gốc bằng RAW(file, sheet) trước khi tính.")
    empty = [c.name for c in tp.columns if c.non_null == 0]
    lines += [render_column(c) for c in tp.columns if c.non_null > 0]
    if empty:
        lines.append(f"- Cột luôn trống: {', '.join(empty)}")
    if tp.duplicate_rows:
        lines.append(f"- {_n(tp.duplicate_rows)} dòng giống hệt một dòng khác (mọi cột)")
    for n in tp.notes:
        lines.append(f"- Khi đọc file: {n}")
    return "\n".join(lines)


def render_joins(groups: list[list[str]]) -> str:
    if not groups:
        return ""
    lines = ["## Cột cùng giá trị giữa các bảng (dò bằng trùng giá trị, dùng để nối)"]
    lines += ["- " + " ~ ".join(g) for g in groups]
    return "\n".join(lines)


def build_note(biz: str, readme: str, prof: dict) -> dict:
    """{"head": README, "tables": {name: block}, "joins": block}. The question layer
    joins head + the chosen tables + joins; full() gives everything."""
    head = f"# Dữ liệu {biz}\n\n## README\n{readme.strip()}".strip()
    return {"head": head,
            "tables": {n: render_table(tp) for n, tp in prof["tables"].items()},
            "joins": render_joins(prof["join_groups"])}


def full_note(note: dict) -> str:
    return "\n\n".join([note["head"], "## Bảng", *note["tables"].values(), note["joins"]]).strip()


# ---------------------------------------------------------------- entity index


def entity_index(prof: dict, tables: dict, max_unique: int = 5000) -> dict[str, list[str]]:
    """Distinct values of text columns of moderate cardinality: names of items, groups,
    customers, staff, suppliers… Chosen by statistics, not by column name."""
    out = {}
    for tn, tp in prof["tables"].items():
        for c in tp.columns:
            if c.letter_share >= 0.5 and 2 <= c.n_unique <= max_unique and c.non_null:
                vals = tables[tn].df[c.name].dropna().astype(str).str.strip()
                vals = sorted(v for v in vals.unique() if v)
                if sum(len(v) for v in vals) / len(vals) >= 2:
                    out[f"{tn}.{c.name}"] = vals
    return out


# ---------------------------------------------------------------- industry (LLM)


def industry_evidence(prof: dict, tables: dict, readme: str, n_cols: int = 6, n_vals: int = 10) -> str:
    """What the model sees to pick industries: the start of the README, column names, and the
    most frequent values of text columns that are not codes. Columns are ranked by number of
    distinct values, so an 80-row product catalogue beats the payment-method column of a big
    order table (a POS pharmacy had no drug names in its evidence otherwise)."""
    intro = " ".join(l.strip() for l in readme.splitlines() if l.strip() and not l.startswith("#"))[:300]
    cols = [f"{tn}: {', '.join(c.name for c in tp.columns if c.non_null)}" for tn, tp in prof["tables"].items()]
    cands = []
    for tn, tp in prof["tables"].items():
        for c in tp.columns:
            code_like = c.formats and c.formats[0][0] in ("a9", "a-9", "a_9") and c.formats[0][1] > 0.8
            if c.letter_share >= 0.5 and 5 <= c.n_unique <= 5000 and not code_like:
                cands.append((c.n_unique, tn, c.name))
    samples = []
    for _, tn, cn in sorted(cands, reverse=True)[:n_cols]:
        top = tables[tn].df[cn].dropna().astype(str).str.strip().value_counts().index[:n_vals]
        samples.append(f"{tn}.{cn}: " + "; ".join(top))
    return (f"README: {intro}\n\nCột:\n" + "\n".join(cols) +
            "\n\nGiá trị hay gặp:\n" + "\n".join(samples) +
            "\n\nDòng mẫu:\n" + sample_rows(tables))


def sample_rows(tables: dict, n_tables: int = 6, n_rows: int = 3, width: int = 40) -> str:
    """A few rows of each table (fixed random_state, so the same every run). Tables with more
    distinct text first, so catalogues and order lines come before lookup lists."""
    def richness(df):
        return sum(df[c].astype(str).nunique() for c in df.columns if df[c].dtype == object)
    out = []
    for name, t in sorted(tables.items(), key=lambda kv: -richness(kv[1].df))[:n_tables]:
        df = t.df.drop(columns=["_sheet"], errors="ignore").dropna(axis=1, how="all")
        rows = df.sample(min(n_rows, len(df)), random_state=0) if len(df) else df
        out.append(f"{name}:")
        for _, r in rows.iterrows():
            cells = [f"{c}={str(v)[:width]}" for c, v in r.items() if str(v) not in ("nan", "None", "")]
            out.append("  " + " | ".join(cells))
    return "\n".join(out)


def industry_prompt(rb: Rulebook, evidence: str) -> tuple[str, dict]:
    sections = []
    for sub in rb.industries:
        rules = [f"{u.title}: {u.body[:120]}" for u in rb.units if u.sub == sub and u.kind == "row"]
        sections.append(f"### {sub}\n" + "\n".join(f"- {r}" for r in rules))
    prompt = (
        "Rulebook có các mục luật riêng theo ngành dưới đây. Dựa vào dữ liệu của một cửa hàng, "
        "chọn các mục ngành áp dụng cho cửa hàng này: 0, 1 hoặc 2 mục. Chọn một mục chỉ khi cửa "
        "hàng thật sự thuộc ngành đó hoặc dữ liệu có đúng các khái niệm mà luật của mục đó nói tới. "
        "Cửa hàng thuộc ngành không có trong danh sách thì trả danh sách rỗng.\n"
        "Viết reason trước: cửa hàng bán gì, rồi với từng mục ngành, khái niệm trong luật của mục "
        "đó có xuất hiện trong dữ liệu không. Sau đó mới chọn industries theo đúng reason.\n\n"
        "Các mục ngành:\n" + "\n".join(sections) + "\n\nDữ liệu cửa hàng:\n" + evidence +
        '\n\nTrả lời JSON: {"reason": "...", "industries": [...]}')
    schema = {"type": "object", "properties": {
        "reason": {"type": "string"},
        "industries": {"type": "array", "items": {"type": "string", "enum": rb.industries}, "maxItems": 2}},
        "required": ["reason", "industries"], "additionalProperties": False}
    return prompt, schema


def detect_industry(rb: Rulebook, evidence: str, llm) -> dict:
    """One constrained LLM call. Returns {"industries": [...], "reason": ..., "usage": ...}."""
    prompt, schema = industry_prompt(rb, evidence)
    data, r = llm.json(prompt, schema, max_tokens=700)
    return {"industries": [i for i in data.get("industries", []) if i in rb.industries][:2],
            "reason": data.get("reason", ""), "raw": r["content"], "usage": r["usage"], "seconds": r["seconds"]}

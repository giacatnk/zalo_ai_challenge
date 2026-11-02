"""Business layer B2: statistics of every table and column, computed by code only.

Nothing here interprets meaning or guesses types (which column is a date, which is money,
which status means cancelled): it reports what the cells look like, as stored in the file,
so the agent can decide.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from .loader import Table
from .text import norm_text, shape

SAMPLE = 3000          # cells sampled for format detection
LOW_CARD = 30          # list every value when a column has at most this many


@dataclass
class ColumnProfile:
    name: str
    stored: str                     # how the file stores the cells: number | datetime | text | mixed
    letter_share: float             # share of non-null cells containing a letter
    non_null: int
    null_ratio: float
    n_unique: int
    formats: list[tuple[str, float, str]]   # (shape, share of non-null cells, example)
    values: list[tuple[str, int]] | None = None   # every value with its count (low cardinality)
    samples: list[str] = field(default_factory=list)
    min: str | None = None          # only when every cell is stored as a number or a datetime
    max: str | None = None
    variants: dict | None = None    # several spellings of one name: {"raw": n, "norm": m, "examples": [...]}
    repeated: bool = False          # values repeat across rows (ids / names of a group)


@dataclass
class TableProfile:
    name: str
    rows: int
    sources: list[str]
    columns: list[ColumnProfile]
    duplicate_rows: int = 0
    notes: list[str] = field(default_factory=list)
    suspect: list[str] = field(default_factory=list)


def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _fmt(v) -> str:
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(v).strftime("%Y-%m-%d") if pd.Timestamp(v).time() == datetime.min.time() \
            else pd.Timestamp(v).strftime("%Y-%m-%d %H:%M")
    if _is_num(v) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def _cell_shape(v) -> str:
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return "<ngày giờ Excel>"
    if _is_num(v):
        return "<số>"
    return shape(v)


def profile_column(name: str, s: pd.Series) -> ColumnProfile:
    nn = s.dropna()
    nn = nn[nn.map(lambda v: not (isinstance(v, str) and not v.strip()))]
    n = len(nn)
    as_str = nn.map(_fmt)
    vc = as_str.value_counts()
    sample = nn.sample(min(SAMPLE, n), random_state=0) if n > SAMPLE else nn
    shapes = Counter()
    example: dict[str, str] = {}
    for v in sample:
        k = _cell_shape(v)
        shapes[k] += 1
        example.setdefault(k, _fmt(v))
    formats = [(k, round(c / max(1, len(sample)), 3), example[k]) for k, c in shapes.most_common(5)]

    stored_as = Counter("datetime" if isinstance(v, (pd.Timestamp, datetime, date)) else
                        "number" if _is_num(v) else "text" for v in sample)
    stored = next(iter(stored_as)) if len(stored_as) == 1 else "mixed" if stored_as else "text"
    letters = np.mean([bool(re.search(r"[^\W\d_]", _fmt(v))) for v in sample]) if n else 0.0

    cp = ColumnProfile(name, stored, round(float(letters), 3), n, round(1 - n / max(1, len(s)), 3),
                       int(vc.size), formats, repeated=bool(n and vc.size < 0.5 * n))
    if vc.size <= LOW_CARD:
        cp.values = [(k, int(c)) for k, c in vc.items()]
    else:
        cp.samples = [str(k) for k in vc.index[:3]]
    if stored in ("number", "datetime") and n:
        cp.min, cp.max = _fmt(nn.min()), _fmt(nn.max())
    elif LOW_CARD < vc.size <= 20000:
        norm = Counter(norm_text(v) for v in vc.index)
        if len(norm) < vc.size:
            by = {}
            for v in vc.index:
                by.setdefault(norm_text(v), []).append(str(v))
            ex = [g for g in by.values() if len(g) > 1][:2]
            cp.variants = {"raw": int(vc.size), "norm": len(norm), "examples": [g[:3] for g in ex]}
    return cp


def profile_table(t: Table) -> TableProfile:
    df = t.df
    data_cols = [c for c in df.columns if c != "_sheet"]
    cols = [profile_column(c, df[c]) for c in data_cols]
    sources = [f"{f}[{s}]" if s else f for f, s in t.sources]
    tp = TableProfile(t.name, len(df), sources, cols, notes=list(t.notes), suspect=list(t.suspect))
    tp.duplicate_rows = int(df.duplicated(subset=data_cols).sum()) if len(df) < 2_000_000 else 0
    return tp


# ---------------------------------------------------------------- joins


def _key_values(s: pd.Series, limit: int = 20000) -> set[str]:
    vals = s.dropna().map(_fmt).map(norm_text)
    vals = vals[vals != ""]
    return set(vals.unique()[:limit])


def find_joins(tables: dict[str, Table], profiles: dict[str, TableProfile],
               min_contain: float = 0.9) -> list[dict]:
    """Columns that can be joined, found by value overlap, not by column name:
    ≥90% of the distinct values of column A (≥20 of them) appear in column B."""
    cands = []
    for tn, tp in profiles.items():
        for c in tp.columns:
            if c.n_unique >= 20:
                cands.append((tn, c.name, c.n_unique, c.non_null))
    sets = {(tn, cn): _key_values(tables[tn].df[cn]) for tn, cn, _, _ in cands}
    out = []
    for ta, ca, ua, _ in cands:
        A = sets[(ta, ca)]
        for tb, cb, ub, nb in cands:
            if ta == tb or not A:
                continue
            B = sets[(tb, cb)]
            if ub < 0.9 * ua * min_contain:   # B must be at least as rich as A
                continue
            share = len(A & B) / len(A)
            if share >= min_contain:
                out.append({"from": f"{ta}.{ca}", "to": f"{tb}.{cb}", "share": round(share, 3),
                            "to_is_key": ub >= 0.95 * nb})
    # values without letters (prices, quantities, dates) match each other by accident: keep
    # such pairs only when the target is a key
    return [j for j in out if j["to_is_key"] or not _letterless(profiles, j["from"])]


def _letterless(profiles, ref: str) -> bool:
    tn, _, cn = ref.rpartition(".")
    tp = profiles.get(tn)
    col = next((c for c in tp.columns if c.name == cn), None) if tp else None
    return bool(col and col.letter_share < 0.5)


def join_groups(joins: list[dict]) -> list[list[str]]:
    """Columns linked by joins, merged into groups of columns holding the same values."""
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for j in joins:
        parent[find(j["from"])] = find(j["to"])
    groups: dict[str, list[str]] = {}
    for x in parent:
        groups.setdefault(find(x), []).append(x)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def profile_business(tables: dict[str, Table]) -> dict:
    profiles = {n: profile_table(t) for n, t in tables.items()}
    joins = find_joins(tables, profiles)
    return {"tables": profiles, "joins": joins, "join_groups": join_groups(joins)}


def to_json(prof: dict) -> dict:
    return {"tables": {n: asdict(p) for n, p in prof["tables"].items()}, "joins": prof["joins"],
            "join_groups": prof["join_groups"]}

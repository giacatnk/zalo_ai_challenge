"""Business layer B1: read every file of a business into tables, without changing meaning.

Handles what the train/public data contains: title rows above the header, two-tier headers,
the header repeated inside a sheet, total / note rows at the bottom, and one table split over
several sheets or files (one per month or year). Every step is checked; a sheet that fails a
check is marked suspect, still loaded with the best guess, and its raw form stays reachable
through raw(). Everything removed is recorded in the table's notes.
"""

from __future__ import annotations

import csv
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .text import norm_text

HEADER_SCAN_ROWS = 25
KEY_FILL = 0.95        # a column filled on at least this share of rows is expected on every data row
SPARSE = 0.6          # a row with at most this share of the typical row's cells is sparse
MIN_BODY = 20         # too few rows to know what a typical row looks like


@dataclass
class Table:
    name: str
    df: pd.DataFrame
    sources: list[tuple[str, str]]            # (file, sheet) in concatenation order
    notes: list[str] = field(default_factory=list)
    suspect: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- raw readers


def _read_csv_raw(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            with open(path, encoding=enc) as f:
                sample = f.read(20000)
            break
        except UnicodeDecodeError:
            continue
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    return pd.read_csv(path, sep=sep, encoding=enc, header=None, dtype=str, keep_default_na=False,
                       na_values=[""], low_memory=False)


def read_raw(path: str | Path) -> dict[str, pd.DataFrame]:
    """sheet name -> cells exactly as stored, no header (header=None). sqlite tables keep
    their column names since a database has a real schema."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=None, header=None, engine="calamine")
    if ext in (".csv", ".tsv", ".txt"):
        return {"": _read_csv_raw(path)}
    if ext in (".sqlite", ".db", ".sqlite3"):
        con = sqlite3.connect(f"file:{path}?immutable=1", uri=True)
        try:
            names = [r[0] for r in con.execute("select name from sqlite_master where type in ('table','view')")]
            return {n: pd.read_sql_query(f'select * from "{n}"', con) for n in names}
        finally:
            con.close()
    return {}


def raw(data_dir: str | Path, file: str, sheet: str | None = None, nrows: int | None = None) -> pd.DataFrame:
    """The agent's RAW(file, sheet): an unprocessed sheet, for files the loader was unsure about."""
    sheets = read_raw(Path(data_dir) / file)
    df = sheets[sheet] if sheet is not None else next(iter(sheets.values()))
    return df.head(nrows) if nrows else df


# ---------------------------------------------------------------- header detection


def _is_label(v) -> bool:
    """A header cell: text where digits are less than half of the characters. Money and dates
    written as text ("7.000đ", "18632k", "01/04/2025") are data whatever their suffix."""
    if not isinstance(v, str) or not (t := re.sub(r"\s", "", v)):
        return False
    return sum(ch.isdigit() for ch in t) < 0.5 * len(t)


def _header_like(row: pd.Series, width: int) -> bool:
    vals = row.dropna()
    return len(vals) >= max(2, 0.6 * width) and sum(map(_is_label, vals)) / len(vals) >= 0.8


def _clean_columns(cells) -> tuple[list[str], int]:
    out, seen, generated = [], {}, 0
    for j, c in enumerate(cells):
        name = "" if c is None or (isinstance(c, float) and pd.isna(c)) else re.sub(r"\s+", " ", str(c)).strip()
        if not name:
            name, generated = f"col_{j}", generated + 1
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out, generated


def _cell_text(row: pd.Series) -> str:
    return " | ".join(str(v).strip() for v in row.dropna() if str(v).strip())


def process_sheet(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Header detection and junk-row removal for one sheet. Returns (df, notes, suspect)."""
    notes: list[str] = []
    suspect: list[str] = []
    df = raw_df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if df.empty:
        return df, notes, suspect
    width = int(df.head(HEADER_SCAN_ROWS).notna().sum(axis=1).max())
    cands = [i for i in range(min(HEADER_SCAN_ROWS, len(df))) if _header_like(df.iloc[i], width)]
    if not cands:
        h = 0
        suspect.append("không tìm thấy dòng nào giống header; đã dùng dòng đầu")
    else:
        h = cands[0]
        # a header sparser than the row below it is likely a group row of a two-tier header
        if h + 1 < len(df) and df.iloc[h].notna().sum() < 0.8 * df.iloc[h + 1].notna().sum():
            suspect.append(f"dòng header {h + 1} thưa hơn dòng {h + 2}; có thể là header hai tầng")
    for i in range(h):
        if t := _cell_text(df.iloc[i]):
            notes.append(f"phía trên header: {t}")
    cols, generated = _clean_columns(df.iloc[h].tolist())
    if generated > 0.2 * len(cols):
        suspect.append(f"{generated}/{len(cols)} cột không có tên")
    body = df.iloc[h + 1:].copy()
    body.columns = cols

    # header repeated inside the sheet (page breaks of an export)
    head_txt = [str(v).strip() if not pd.isna(v) else "" for v in df.iloc[h].tolist()]
    same = pd.Series(True, index=body.index)
    for c, t in zip(cols, head_txt):
        same &= body[c].astype(str).str.strip().where(body[c].notna(), "") == t
    if same.any():
        notes.append(f"đã bỏ {int(same.sum())} dòng lặp lại header giữa sheet")
        body = body[~same]

    # subtotal / total / note rows anywhere ("3,411,000 | Tổng ngày 01/03" after each day,
    # "Tong cong ..." or "Ghi chú: ..." at the end): told apart by shape, not by their words.
    # A data row fills the columns that almost every row fills; such a row leaves them empty
    # and holds far fewer cells than usual.
    if len(body) >= MIN_BODY:
        filled = body.notna()
        keys = [c for c in body.columns if filled[c].mean() >= KEY_FILL]
        nonnull = filled.sum(axis=1)
        junk = (nonnull <= SPARSE * nonnull.median()) & ~filled[keys].all(axis=1) if keys else None
        if junk is not None and junk.any():
            at_end = junk.iloc[::-1].cumprod().astype(bool).iloc[::-1]   # trailing run of junk rows
            for i in body.index[junk]:
                where = "dòng cuối" if at_end[i] else "dòng tổng hoặc ghi chú giữa sheet"
                notes.append(f"đã bỏ {where}: {_cell_text(body.loc[i])}")
            body = body[~junk]
    body = body.dropna(how="all").reset_index(drop=True)
    return body.infer_objects(), notes, suspect


# ---------------------------------------------------------------- business


def _group_name(keys: list[tuple[str, str]]) -> str:
    """Name for sheets of the same layout: common prefix of 'stem[sheet]', trailing digits and
    separators trimmed; the sheet name is kept when it is the same everywhere."""
    labels = [f"{Path(f).stem}[{s}]" if s else Path(f).stem for f, s in keys]
    prefix = os.path.commonprefix(labels)
    stems = {Path(f).stem for f, _ in keys}
    sheets = {s for _, s in keys}
    if len(stems) == 1:
        base = next(iter(stems))
        return base if len(sheets) == 1 else f"{base}[ALL]"
    base = re.sub(r"[\d_\-.\[ ]+$", "", os.path.commonprefix(sorted(stems))) or prefix.rstrip("[") or "table"
    sheet = next(iter(sheets))
    return f"{base}[{sheet}]" if len(sheets) == 1 and sheet and not re.fullmatch(r"sheet\d*", sheet, re.I) else base


def load_business(data_dir: str | Path) -> dict[str, Table]:
    data_dir = Path(data_dir)
    parts: list[tuple[tuple[str, str], pd.DataFrame, list[str], list[str]]] = []
    for path in sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() != ".md"):
        is_db = path.suffix.lower() in (".sqlite", ".db", ".sqlite3")
        for sheet, rdf in read_raw(path).items():
            if is_db:
                df, notes, suspect = rdf, [], []
            else:
                df, notes, suspect = process_sheet(rdf)
            if df.empty:
                continue
            parts.append(((path.name, sheet), df, notes, suspect))

    # group sheets (also across files) that share the exact same columns
    groups: dict[tuple, list[int]] = {}
    for i, (_, df, _, _) in enumerate(parts):
        groups.setdefault(tuple(df.columns), []).append(i)
    tables: dict[str, Table] = {}
    for idx in groups.values():
        keys = [parts[i][0] for i in idx]
        if len(idx) == 1:
            f, s = keys[0]
            n_sheets = sum(1 for k, *_ in parts if k[0] == f)
            name = f"{Path(f).stem}[{s}]" if s and n_sheets > 1 else Path(f).stem
            df = parts[idx[0]][1]
        else:
            name = _group_name(keys)
            df = pd.concat([parts[i][1].assign(_sheet=f"{k[0]}[{k[1]}]" if k[1] else k[0])
                            for i, k in zip(idx, keys)], ignore_index=True)
        notes, suspect = [], []
        for i, k in zip(idx, keys):
            where = f"{k[0]}[{k[1]}]" if k[1] else k[0]
            prefix = f"{where}: " if len(idx) > 1 else ""
            notes += [prefix + n for n in parts[i][2]]
            suspect += [f"{where}: {s}" for s in parts[i][3]]
        while name in tables:
            name += "_"
        tables[name] = Table(name, df, keys, _dedupe_notes(notes), suspect)
    return tables


def _dedupe_notes(notes: list[str]) -> list[str]:
    """Notes that repeat with only digits changing ("Từ ngày 01/10/2025 …" on every sheet,
    "TỔNG NGÀY 01/10" after every day) -> one line: first example and a count."""
    groups: dict[str, list[str]] = {}
    for n in notes:
        text = re.sub(r"^[^:]*\.(xlsx|xls|csv|sqlite)(\[[^\]]*\])?: ", "", n)
        groups.setdefault(re.sub(r"\d[\d.,]*(?:[^\W\d_]+[\d.,]*)?", "#", text), []).append(text)   # a number and its suffix
    return [texts[0] if len(texts) == 1 else f"{texts[0]} (và {len(texts) - 1} dòng cùng dạng)"
            for texts in groups.values()]

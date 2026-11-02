"""Parsers for cell values written as text: numbers in any separator style, numeric dates.

Preloaded in the agent's sandbox. They know no words: units and dates written with words are
read only when the caller says what they mean.
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import date, datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- numbers

# A number, then anything that is not a digit (a unit: "k", "tr", "đ", "₫", " ly"), then
# optionally more digits ("19tr990"), then optionally "/..." ("44.000 dong/tuyp").
_NUM_UNIT = re.compile(r"^([\d.,\s]*\d)\s*([^\d/]*?)\s*(\d*)\s*(/[^\d]*)?$")
_THOUSANDS = re.compile(r"^\d{1,3}([.,\s]\d{3})+$")                # 28.970.000  22,000  1 200 000


def _plain_number(s: str) -> float:
    if _THOUSANDS.match(s):
        return float(re.sub(r"[.,\s]", "", s))
    if s.count(",") == 1 and "." not in s:
        s = s.replace(",", ".")
    elif "," in s and "." in s:
        # the later separator is the decimal one: 45,000.0000 or 1.234,5
        s = s.replace(",", "") if s.rfind(".") > s.rfind(",") else s.replace(".", "").replace(",", ".")
    return float(s)


def parse_number(v, units: dict[str, float] | None = None) -> float:
    """A cell as a number. Plain numbers in any separator style: '45,000.0000'->45000,
    '28.970.000'->28970000. A unit after the number is read only when `units` names it, since
    what a suffix means depends on the shop (the EDA note shows each column's formats):
    parse_number('1386k', {'k': 1e3}) -> 1386000, parse_number('19tr990', {'tr': 1e6}) -> 19990000,
    parse_number('7.000đ', {'đ': 1}) -> 7000. NaN when empty, unparseable or the unit is not given."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = unicodedata.normalize("NFC", str(v)).strip().lower().replace("\u00a0", " ")
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    s = s.strip("-()+ ")
    units = {k.lower(): x for k, x in (units or {}).items()}
    for k in sorted((k for k in units if re.search(r"\d", k)), key=len, reverse=True):
        if s.endswith(k) and re.fullmatch(r"[\d.,\s]*\d\s*", s[:-len(k)]):    # a unit with digits: '1 xe 4m3'
            try:
                val = _plain_number(s[:-len(k)].strip()) * units[k]
            except ValueError:
                return np.nan
            return -val if neg else val
    m = _NUM_UNIT.match(s)
    if not m:
        return np.nan
    num, unit, tail = m.group(1).strip(), m.group(2).strip(), m.group(3)
    try:
        if not unit:
            if tail or m.group(4):
                return np.nan
            val = _plain_number(num)
        elif unit not in units:
            return np.nan
        elif tail:                                   # 19tr990, 1tr2: digits after the unit
            val = float(f"{num}.{tail}") * units[unit]
        else:
            val = _plain_number(num) * units[unit]
    except ValueError:
        return np.nan
    return -val if neg else val


def to_number(s, units: dict[str, float] | None = None):
    """A Series (or one cell) as numbers; see parse_number."""
    if not isinstance(s, pd.Series):
        return parse_number(s, units)
    return s.map(lambda v: parse_number(v, units)).astype(float)


to_money = to_number

# ---------------------------------------------------------------- dates

_DMY = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})(?:[/.\-](\d{2,4}))?(?:[ T]+(\d{1,2}):(\d{2})(?::(\d{2}))?)?")
_YMD = re.compile(r"^(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})(?:[ T]+(\d{1,2}):(\d{2})(?::(\d{2}))?)?")
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def _num_groups(s: str):
    """Day, month, year (+ time) from a date whose numbers are separated by words: three
    number groups with the year (4 digits) first or last, optionally followed by h:m(:s)."""
    g = re.findall(r"\d+", s)
    if len(g) not in (3, 5, 6) or any(len(x) > 4 for x in g):
        return None
    date, time_ = g[:3], [int(x) for x in g[3:]] + [0] * (6 - len(g))
    if len(date[2]) == 4:
        d, mo, y = date
    elif len(date[0]) == 4:
        y, mo, d = date
    else:
        return None
    return int(d), int(mo), int(y), *time_


def parse_date(v, default_year: int | None = None, order: str = "dmy"):
    """Excel serials (45717), datetime objects, yyyy-mm-dd, dd/mm/yyyy (day first),
    dd/mm without a year (uses default_year). order="mdy" reads 04/13/2025 as April 13. Numbers separated by words (25 thg 2 2026) are read the same way; month names are not.
    NaT otherwise."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return pd.NaT
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(v)
    if isinstance(v, (bool, np.bool_)):
        return pd.NaT
    if isinstance(v, (int, float, np.integer, np.floating)):
        return _EXCEL_EPOCH + pd.to_timedelta(float(v), unit="D") if 20000 <= v <= 80000 else pd.NaT
    s = str(v).strip()
    if re.fullmatch(r"\d{5}(\.\d+)?", s):
        return parse_date(float(s), order=order)
    y = mo = d = None
    hh = mm = ss = 0
    if m := _YMD.match(s):
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        hh, mm, ss = int(m[4] or 0), int(m[5] or 0), int(m[6] or 0)
    elif m := _DMY.match(s):
        d, mo = (int(m[2]), int(m[1])) if order == "mdy" else (int(m[1]), int(m[2]))
        y = int(m[3]) if m[3] else default_year
        hh, mm, ss = int(m[4] or 0), int(m[5] or 0), int(m[6] or 0)
    elif re.search(r"[^\W\d_]", s) and (g := _num_groups(s)):     # numbers among words: "25 thg 2 2026"
        d, mo, y, hh, mm, ss = g
        if order == "mdy":
            d, mo = mo, d
    else:
        return pd.NaT
    if y is None:
        return pd.NaT
    if y < 100:
        y += 2000
    try:
        return pd.Timestamp(year=y, month=mo, day=d, hour=hh, minute=mm, second=ss)
    except ValueError:
        return pd.NaT


def to_date(s: pd.Series, default_year=None, dayfirst: bool = True) -> pd.Series:
    """Vectorised parse_date. default_year: int, or a Series aligned with s
    (e.g. the year found in each row's sheet name). dayfirst is always assumed. One cell in,
    one Timestamp out."""
    if not isinstance(s, pd.Series):
        return parse_date(s, default_year)
    if isinstance(default_year, pd.Series):
        vals = [parse_date(v, None if pd.isna(y) else int(y)) for v, y in zip(s, default_year)]
        return pd.Series(vals, index=s.index, dtype="datetime64[ns]")
    cache: dict = {}

    def one(v):
        key = v if isinstance(v, (str, int, float)) else None
        if key is None:
            return parse_date(v, default_year)
        if key not in cache:
            cache[key] = parse_date(v, default_year)
        return cache[key]

    return pd.Series([one(v) for v in s], index=s.index, dtype="datetime64[ns]")


def year_in(name) -> int | None:
    """Year found in a sheet or file name such as 'T7-2025', 'BanHang_T1_2026'."""
    m = re.search(r"(20\d{2})", str(name))
    return int(m[1]) if m else None

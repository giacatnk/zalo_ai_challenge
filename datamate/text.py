"""Vietnamese text normalisation shared by every layer."""

from __future__ import annotations

import re
import unicodedata


def strip_accents(text) -> str:
    t = unicodedata.normalize("NFD", str(text))
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return t.replace("đ", "d").replace("Đ", "D")


def norm_text(text) -> str:
    """Lowercase, no accents, collapsed spaces: for matching names, statuses, keywords."""
    return re.sub(r"\s+", " ", strip_accents(text).lower()).strip()


def norm_name(text) -> str:
    """The organisers' string normalisation for scoring: NFC, lowercase, every char that is
    not a letter or digit becomes a space, whitespace collapsed. Accents are kept."""
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = "".join(c if c.isalnum() or c.isspace() else " " for c in t)
    return re.sub(r"\s+", " ", t).strip()


def shape(value) -> str:
    """Format fingerprint of a cell: digit runs -> 9, letter runs -> a, punctuation kept.
    '45,000.0000' -> '9,9.9'   '1 ly' -> '9 a'   '01/04/2025' -> '9/9/9'   '1386k' -> '9a'"""
    s = re.sub(r"\d+", "9", str(value).strip())
    s = re.sub(r"[^\W\d_]+", "a", s)
    return re.sub(r"\s+", " ", s)

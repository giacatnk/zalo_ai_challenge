"""Split layer: parse a split's rulebook.md into rule units (table rows and paragraphs).

Only the markdown structure is used, never the words of a heading, so a private rulebook that
renames, renumbers, adds or drops sections still parses the same way. ### subsections are
groups (industries in the train and public rulebooks): the business layer's LLM call picks
which groups apply to a business, and only those reach its prompt.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

from .text import norm_text

# ---------------------------------------------------------------- units


@dataclass
class Unit:
    id: str             # "s3.r10", "s6.ca-phe.r2", "s7.p1", "pre.p1"
    section: str        # section heading, e.g. "3. Xếp hạng"; "Mở đầu" before the first one
    sub: str | None     # ### subsection heading: a group (an industry in train/public)
    kind: str           # "row" | "para"
    title: str          # first table cell, bold label, or first words
    body: str

    @property
    def text(self) -> str:
        return f"{self.title}: {self.body}" if self.kind == "row" else self.body

    def render(self) -> str:
        where = f"{self.section} / {self.sub}" if self.sub else self.section
        return f"[{where}] {self.text}"


_BOLD_HEAD = re.compile(r"^\*\*(.+?)\*\*")
_MD = re.compile(r"\*\*|`")


def _strip_md(s: str) -> str:
    return _MD.sub("", s).strip()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", norm_text(s)).strip("-")


def parse_units(md: str) -> list[Unit]:
    units: list[Unit] = []
    section, sub, sid = "Mở đầu", None, "pre"
    n_row = n_par = 0
    in_table = False
    para: list[str] = []

    def flush():
        nonlocal para, n_par
        text = " ".join(l.strip() for l in para).strip()
        para = []
        if not text:
            return
        n_par += 1
        bold = _BOLD_HEAD.match(text)
        title = _strip_md(bold.group(1)).rstrip(":") if bold else " ".join(_strip_md(text).split()[:8])
        units.append(Unit(f"{sid}.p{n_par}", section, sub, "para", title, _strip_md(text)))

    for line in md.splitlines():
        s = line.strip()
        if s.startswith("# "):
            continue  # document title
        if s.startswith("#"):
            flush()
            head = s.lstrip("#").strip()
            if s.startswith("## "):
                section, sub = head, None
                m = re.match(r"(\d+)", head)
                sid = f"s{m.group(1)}" if m else f"s-{_slug(head)}"
            else:
                sub = head
                sid = f"{sid.split('.')[0]}.{_slug(head)}"
            n_row = n_par = 0
            in_table = False
            continue
        if s.startswith("|"):
            flush()
            if re.fullmatch(r"\|[\s:|-]+\|?", s):
                continue  # header separator
            if not in_table:  # first row of a table is its header
                in_table = True
                continue
            cells = [_strip_md(c) for c in s.strip("|").split("|")]
            n_row += 1
            units.append(Unit(f"{sid}.r{n_row}", section, sub, "row", cells[0],
                              " | ".join(c for c in cells[1:] if c)))
            continue
        in_table = False
        if not s:
            flush()
        elif s.startswith(("- ", "• ", "* ")):
            flush()
            para = [s[2:]]
            flush()
        else:
            para.append(s)
    flush()
    return units


# ---------------------------------------------------------------- rulebook

class Rulebook:
    def __init__(self, md: str):
        self.md = md
        self.units = parse_units(md)
        self.by_id = {u.id: u for u in self.units}

    @property
    def industries(self) -> list[str]:
        """### groups. The business layer's LLM call picks which of them apply."""
        return list(dict.fromkeys(u.sub for u in self.units if u.sub))

    # -- export

    def summary(self) -> dict:
        return {
            "industries": self.industries,
            "counts": dict(Counter(u.kind for u in self.units)),
            "units": [asdict(u) for u in self.units],
        }

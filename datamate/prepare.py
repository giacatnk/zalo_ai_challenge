"""Run the split and business layers for a split and cache the results.

uv run python -m datamate.prepare --split data_mate/public_test --out cache/public_test
uv run python -m datamate.prepare --split data_mate/train --out cache/train --llm openrouter
uv run python -m datamate.prepare --split data_mate/train --out cache/train --llm vllm   # final image

Output
  <out>/rulebook.json            rule units
  <out>/<biz>/note.md            full EDA note
  <out>/<biz>/note.json          the note split per table (for per-question selection)
  <out>/<biz>/profile.json       column statistics and joins
  <out>/<biz>/index.json         entity index (names for fuzzy matching)
  <out>/<biz>/meta.json          industries, timings, suspect sheets
  <out>/<biz>/tables.pkl         loaded tables, so the question layer does not re-read files
  <out>/<biz>/db.duckdb          typed tables + cal for the SQL agent (needs --llm: the model types columns)
  <out>/<biz>/spec.json          the model's column spec and parse rates
  <out>/<biz>/schema.md          README + CREATE TABLE schema shown to the SQL agent
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from .clean import build_database
from .eda import build_note, detect_industry, entity_index, full_note, industry_evidence
from .loader import load_business
from .profile import profile_business, to_json
from .rulebook import Rulebook
from .solve import render_rulebook


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def prepare_business(biz_dir: Path, out: Path, rb: Rulebook, llm=None) -> dict:
    t0 = time.time()
    tables = load_business(biz_dir)
    t_load = time.time() - t0
    prof = profile_business(tables)
    t_prof = time.time() - t0 - t_load
    readme_path = biz_dir / "README.md"
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    note = build_note(biz_dir.name, readme, prof)
    index = entity_index(prof, tables)
    evidence = industry_evidence(prof, tables, readme)

    industry = {"industries": None, "reason": "no LLM configured"}
    if llm is not None and rb.industries:
        industry = detect_industry(rb, evidence, llm)
    t_db = time.time()
    out.mkdir(parents=True, exist_ok=True)
    if llm is not None:
        rules = render_rulebook(rb, industry.get("industries") or [])
        build_database(tables, prof["tables"], readme, biz_dir.parent.parent / "calendar.csv", out, llm, rules)
    t_db = time.time() - t_db

    (out / "note.md").write_text(full_note(note), encoding="utf-8")
    _dump(out / "note.json", note)
    _dump(out / "profile.json", to_json(prof))
    _dump(out / "index.json", index)
    with open(out / "tables.pkl", "wb") as f:
        pickle.dump({n: t.df for n, t in tables.items()}, f)
    meta = {"business": biz_dir.name, "tables": {n: list(t.df.shape) for n, t in tables.items()},
            "suspect": {n: t.suspect for n, t in tables.items() if t.suspect},
            "industry": industry, "industry_evidence": evidence,
            "seconds": {"load": round(t_load, 2), "profile": round(t_prof, 2), "database": round(t_db, 2),
                        "total": round(time.time() - t0, 2)}}
    _dump(out / "meta.json", meta)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="split dir with data/, rulebook.md")
    ap.add_argument("--out", required=True)
    ap.add_argument("--biz", nargs="*", help="only these businesses")
    ap.add_argument("--llm", choices=["openrouter", "vllm"], help="backend for the industry call (omit to skip)")
    ap.add_argument("--model", help="override the backend's default model")
    ap.add_argument("--base-url", help="override the backend's default endpoint")
    a = ap.parse_args()

    split, out = Path(a.split), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rb = Rulebook((split / "rulebook.md").read_text(encoding="utf-8"))
    _dump(out / "rulebook.json", rb.summary())
    print(f"rulebook: {len(rb.units)} units, industries {rb.industries}")

    llm = None
    if a.llm:
        from .llm import LLM
        llm = LLM(a.llm, model=a.model, base_url=a.base_url)

    dirs = sorted(d for d in (split / "data").iterdir() if d.is_dir() and (not a.biz or d.name in a.biz))
    for d in dirs:
        m = prepare_business(d, out / d.name, rb, llm)
        sus = f" ⚠ {len(m['suspect'])} suspect" if m["suspect"] else ""
        print(f"{d.name}: {len(m['tables'])} tables, {m['seconds']['total']}s, "
              f"industry={m['industry']['industries']}{sus}")


if __name__ == "__main__":
    main()

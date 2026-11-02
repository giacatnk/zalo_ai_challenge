---
name: working-rules
description: Standing rules the user set for working on LifeMate (language, Kaggle safety, edits, publishing)
metadata:
  type: feedback
---

- Always answer in Vietnamese.
- Kaggle account must be **jasonhoang666**. Never read or print the Kaggle token. When CLI auth expires ("Permission kernels.get denied"), ask the user to run `! .venv/bin/kaggle auth login`; never do the OAuth consent step myself.
- Datasets uploaded to Kaggle must stay **Private** (`kaggle datasets create` is private by default).
- Do not download model weights (fetch excludes *.safetensors) unless the user allows it for that case.
- At most 2 Kaggle GPU sessions at once; do not poll in a tight loop (use background waits).
- Smoke-test locally before pushing (`scripts/dev/barrier_smoke.py`, `scripts/dev/verify_smoke.py`).
- Edit files with Edit/Write tools, not `sed -i`/heredoc writes (user reviews diffs). No em-dash in code comments; comments only explain non-obvious WHY.
- No claude.ai Artifacts unless asked.
- "cứ làm tốt nhất, license tính sau" (2026-11-01): optimise for score first.
- I must not submit to the competition portal; the user submits.
- Update PROGRESS.md after each run.

**Why:** set by the user across sessions 2026-10-29..11-02.
**How to apply:** follow on every task in this folder. Related: [[lifemate-kaggle-runs]], [[feedback-eta]]

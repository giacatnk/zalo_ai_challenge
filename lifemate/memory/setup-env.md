---
name: setup-env
description: FIRST step on any machine/session - check .venv + kaggle CLI installed and logged in; install if missing, ask user to log in
metadata:
  type: feedback
---

At the start of every session in this folder (new machine, fresh clone, or before any Kaggle command), run:

    bash scripts/setup_env.sh

- It creates `.venv` (Python 3.12 via uv, fallback python3 -m venv) and installs `kaggle` + `numpy` if missing, then checks auth with `kaggle kernels list --mine --page-size 1`.
- Exit 0: ready. Exit 2: not logged in / OAuth expired -> STOP and ask the user to run `! .venv/bin/kaggle auth login` (account **jasonhoang666**). Never do the OAuth consent myself, never read or print the token / `~/.kaggle/*`.
- Also check data: `data/lifemate/` and `models/` are git-ignored. If `data/lifemate` is missing, tell the user (download from organiser or private Kaggle dataset `jasonhoang666/lifemate-data`); do not download models to the machine.

**Why:** user moved the repo to GitHub to work on several machines (2026-11-02); `.venv` is git-ignored and the old one was lost in the move, so Kaggle commands failed.
**How to apply:** run the script before the first `kaggle_run.py` / `kaggle` call of a session; if it fails on auth, tell the user the exact login command and wait. Related: [[working-rules]], [[lifemate-kaggle-runs]]

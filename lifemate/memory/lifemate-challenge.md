---
name: lifemate-challenge
description: Zalo AI Challenge 2026 LifeMate track (Vietnamese voice -> tool calls); constraints and deadlines
metadata:
  type: project
---

User is entering Zalo AI Challenge 2026, LifeMate track: Vietnamese audio request -> ordered tool calls + state_hash via organiser's `zaic_tools`. Constraints: whole system < 6B params, LLM < 5B with <= 16K context, <= 30s/sample on 1xA30, fully offline, OSI-licensed open models released before 2026-10-26.

Deadlines (GMT+7): Docker solution 2026-11-16 23:59 (an older note said 13/11, check `paper-review/lifemate/00-rules.md`); private test released 2026-11-20 09:00; private results due 2026-11-23 23:59.

User runs everything on Kaggle (T4 x2, 12h sessions) and does not want models downloaded to their machine unless they say so; the dataset lives locally in `data/lifemate` (v3.2.0: train 1545 / public_test 634, git-ignored). Pipeline notebook resumes from a `checkpoint/` dir found under /kaggle/input (previous output attached as Input).

Scoring (verified in `zaic_tools/utils.py canon()`): state_hash = sha256 of canonical final STATE (not the call list); scalar lists sorted, lists of dicts keep order (so order of write calls into one list matters); call F1 is order-independent. Public leaderboard scores the jsonl only, so other teams may use big models there.

**Why:** these limits decide which models/papers are usable.
**How to apply:** check model size and license against them before recommending. Research workspace: `paper-review/lifemate/` (rules summary `00-rules.md`). Related: [[lifemate-kaggle-runs]]

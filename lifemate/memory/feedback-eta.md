---
name: feedback-eta
description: Always give the user an ETA (clock time, GMT+7) whenever pushing or reporting a Kaggle run
metadata:
  type: feedback
---

Every time I push a Kaggle run (or report one still running), state an ETA as a clock time in GMT+7, not only a duration, and note the queue risk.

**Why:** user asked on 2026-11-01 ("lần sau chạy thì nhớ báo tôi ETA") after runs sat in the GPU queue ~7.5 h unannounced.
**How to apply:** base it on past durations in PROGRESS.md (dev 200 with sft1 + READ_BARRIER ~2.4 h; public_test 634 ~4.0 h; ASR LoRA infer ~20 min; SFT train 8 h budget), and check `kaggle kernels list --mine --search zalo` lastRunTime (UTC) to see if a run actually started. Related: [[lifemate-kaggle-runs]]

---
name: lifemate-kaggle-runs
description: "How I run the LifeMate notebook on the user's Kaggle via CLI, and the score history per run tag"
metadata:
  type: project
---

All paths are relative to the `lifemate/` folder. Run `bash scripts/setup_env.sh` first ([[setup-env]]).

I push and run the pipeline myself: `.venv/bin/python scripts/kaggle_run.py --set 'SPLIT="train"' LIMIT=30 'RUN_TAG="vN"'` pushes `notebooks/lifemate_pipeline.py` as a new version of the user's notebook `jasonhoang666/zalo-ai` (T4x2, dataset `jasonhoang666/lifemate-data`, private), waits, downloads to `output/<tag>/`. Kaggle CLI auth is via `kaggle auth login` (never read/print the token). Each run uses the user's 30h/week GPU quota: say what runs and roughly how long before pushing.

Score history on the first 30 train samples (ASR text, not gold text):
- v1 12.56: few-shot format broke parsing, extra args, router.
- v2 (2026-10-30) 40.80: `<tool_call>` few-shot, arg pruning, time normalisation, router off. Remaining errors: wrong relative dates ("mai"), ASR name/verb errors, missing prerequisite reads (banking.get_balance), ask_missing_info after a single search hit.
- v3 47.76: adds date table (fixed 0008, 0016), prerequisite-read hints, empty-search retry hint, ask guard. contacts.search query is exact-match (not fuzzy), gold always capitalises the first letter.
- v4 (60 samples) 46.60 (first 30: 47.00, flat vs v3 = prompt tweaks plateaued): capitalise contacts.search query, dedupe accent-stripped searches, nhà/công ty -> profile addresses, beneficiary hint. Remaining: ASR errors (~half of failures), oos 2/5 (acts on cancelled/chitchat), conditionals (lights with AC on, bill source ví/ngân hàng), duplicate-name contacts. After v4 added bare-JSON parser fallback (not yet run).

- v5-text (first 60, gold text) 67.13 vs v4 ASR 46.60: ASR costs ~20 pts, LLM still loses ~33 on clean text (beneficiary vs contacts, balance conditionals, floor/AC light filtering, multi-match contacts -> ask (74/105 in gold), cancellation -> OOS).
- v6-gold: 200-dev, rules for those + query title-casing (lowercase only words seen lowercase in gold) + nhà/công ty substitution for ride_hailing/package_tracking. Result 49.71 (act .47, ask .52, oos .79); 83/200 wrong hash = 0, mean on hash-ok 0.85. Failure classes: model writes plan in prose and stops without the write call (~15), invented IDs (alm_001, cont_123), wrong date despite table, condition filtering (floor/AC lights, >500k bills), acts where gold asks after search (~7 ask samples).
- v7-gold (pushed 2026-10-30): date_hints() next to the user turn (weekday = next occurrence 1-7 days, 222/227 match gold), _unseen_id guard (ID must appear in context/history/raw tool results), one prose nudge when model answers in text after tool calls. Gold replay unchanged 98.67.

- asr1 (dev 200, 730 utts): vietasr WER .190 (best, 0.43 s/utt), phowhisper-large .267, phowhisper-medium .290, whisper-v3-turbo .359. VietASR wrong first/last word ~23% (edge truncation?). asr2 result: pad 0.5s + modified_beam_search 4 is best (WER .179, arg_recall .838 vs .815, first_err .18 vs .22, 0.50 s/utt); pad 1.0s no better. Edge errors are mostly not truncation.
- lora1: PhoWhisper-large + LoRA r32, 2 epochs (538 steps, 127 min on T4) -> dev WER .101, CER .061, first/last err .09, 0.50 s/utt. Clear winner over VietASR (.179). Adapter in output/lora1/phowhisper_lora_adapter. PhoWhisper-large 1.55B + Qwen3-4B ~4.0B is ~5.6B, under the 6B cap but tight. v7-lora pushed with this transcript.

- v7 results (dev 200): v7-gold 56.41 (hash_ok 133, +6.7 vs v6, 23 up / 2 down), v7-lora 44.00 (ASR costs ~12.4). ~10 of the 41 ASR losses have near-identical text (case/punctuation only) = LLM instability, the rest are real misrecognitions (names, rare words, dialect). 200 dev samples take ~2.2h on T4x2.

Goal set by user 2026-10-31: score >80. User asked to research/test existing models before training.
- Model research (2026-10-31): Qwen3.5-4B (Apache, Mar 2026, hybrid Gated DeltaNet, 4B LM + ~0.3B vision, BFCL-v4 50.3, TAU2 79.9, thinking on by default, qwen3_coder XML tool format, needs vllm>=0.17) is the only strong new candidate. Gemma 4 E4B is ~8B with embeddings (breaks LLM<5B). Thinking models are too slow for 30 s/sample. Pipeline now has `STACK["VLLM"]` (0.10.2 | 0.31.0, torch 2.13), `LLM_TEXT_ONLY`, and an XML tool-call parser. Run q35-gold40 on kernel zalo-ai-b.
- v7-gold failure classes (67 wrong hash): mixed 20, gold asks/pred acts 9, pred asks/gold acts 8, args differ 8, extra calls 8, stopped early 6. Prompt patches are plateauing; SFT is the planned path to >80.
- `scripts/make_sft.py data/lifemate <out.jsonl>` builds 1345 SFT conversations from non-dev train using the notebook's own prompt functions (FEWSHOT_K 0), consecutive independent calls grouped per turn, DONE at end. Output data/sft/sft_train.jsonl uploaded as private dataset `jasonhoang666/lifemate-sft`.
- SFT notebook `notebooks/llm_sft.py` -> kernel `jasonhoang666/zalo-ai-sft` (`--nb sft`, dir kaggle_sft/): torchrun DDP 2xT4, fp16 base + fp32 LoRA r32 all linear, loss only on final-turn assistant tokens (lm_head on labelled positions only), merges to /kaggle/working/qwen3-4b-sft. Pipeline loads it with `LLM="input:qwen3-4b-sft"` + `--source jasonhoang666/zalo-ai-sft`, FEWSHOT_K=0. sft-smoke OK after fixes (OOM at 9.3K tokens fixed by lora_dropout 0 + expandable_segments + MAX_LEN 10240; 24 longest convs fine). Full run sft1 pushed 2026-10-31 (2 epochs, ~6h).
- q35-gold40: Qwen3.5-4B runs on T4 (vllm 0.31, TRITON_ATTN, needs torchaudio uninstalled) but scores 50.05 vs Qwen3-4B 73.87 on the same 40 dev ids (acts before reading results, ignores conditions, invents times). Faster (15 s/sample batched). Decision: stay on Qwen3-4B-Instruct-2507 + SFT.
- Jev idea (user, 2026-10-31): "Jev" = TypeSafe's hosted decision model (closed, API only, unusable offline); open clones like FastJev just score options by causal-LM logits. Phase 1 probe = `JEV_PROBE=True` cell 9b in the pipeline. Result (base Qwen3-4B, 642 steps): top1 .841, top3 .935; first step .790, DONE .919, ask only .607; all-steps-correct 125/200. Confidence useless: wrong picks still p~0.999, AUROC .69. 4.3 s/step on T4. Verdict: no gain over generation for base model. Public test has 34/122 tools unseen in train, so fixed-label classifiers are out.
- asr3 result (dev 730 utts, zero-shot): Qwen3-ASR-1.7B WER .284, Gemma 4 E2B fp16 .456 / fp32 .406. Both far behind PhoWhisper-LoRA .101: end-to-end audio LLM idea dropped, keep PhoWhisper-LoRA.
- sft1: hit 8h TRAIN_BUDGET at step 120/166 (~1.45 ep), loss 1.09 -> ~0.05. sft1-gold = 49.86 vs v7-gold 56.41. Cause: make_sft group_steps put read + write in one assistant turn, so the model acts before seeing read results. Fixes: pipeline `READ_BARRIER` (drops only kind=="write" after a read in the same turn); make_sft now splits a step after reads. sft1-gold-rb (barrier on write+ask) 57.90 but ask collapsed -> barrier write-only. Oracle max(rb, v7) per sample = 72.3.
- Later runs (rb4..rb7, public 44.88, verifier/ASR_FIX experiments): see PROGRESS.md, it is the source of truth.
- Kaggle CLI OAuth expires: if status/output give "Permission kernels.get denied", ask the user to run `! .venv/bin/kaggle auth login` (I must not do the OAuth consent). Never download kernel outputs containing model weights: kaggle_run fetch excludes *.safetensors (user does not want models on the machine).

Kaggle quota quirk (user, 2026-11-02): a session started while any GPU quota remains may run its full 12h even past quota. So when quota is low: spend it on short tests, and save one last push for the final full public run.

Never run two versions of the same kernel at once: `kernels status`/`output` follow the latest version (fix by fetching `kernel/<version>`). For parallel pipeline runs use `--nb pipeline2` (kernel `jasonhoang666/zalo-ai-b`, dir `kaggle_b/`).
Kaggle allows at most 2 concurrent GPU batch sessions per account ("Maximum batch GPU session count of 2 reached"): queue the third push until one finishes.
Third notebook `notebooks/asr_lora.py` -> `jasonhoang666/zalo-ai-asr-lora` (`--nb lora`): LoRA r32 on PhoWhisper-large, trained on 4306 non-dev utterances with written-form labels, writes `asr_phowhisper-lora.json` + adapter.

Two parallel tracks (user agreed 2026-10-30): (A) ASR notebook `notebooks/asr_bench.py` -> kernel `jasonhoang666/zalo-ai-asr` (`--nb asr`); (B) agent in `zalo-ai`, `TRANSCRIPT_SRC` = "gold" | "asr" | "asr_<model>.json". Fixed dev set = 200 train samples by md5(id) per kind (`SUBSET="dev"`). Evaluate agent changes on both gold text and real ASR.

Open idea not done: snapping ASR contact names using env state; ask the user whether reading state.json is allowed by the rules first.

Related: [[lifemate-challenge]]

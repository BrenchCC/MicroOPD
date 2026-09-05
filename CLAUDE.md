# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

MicroOPD — a single-GPU (RTX 4060 16GB) reproduction + ablation of **On-Policy Distillation (OPD)** for math reasoning: Qwen3-8B (teacher, FP8) distills into Qwen3-1.7B (student, BF16 + LoRA r16) on GSM8K. Phase 1 is pure OPD; phase 2 (M6) swaps the teacher signal for GSM8K exact-match verifier reward (GRPO). Public distinction: it is a *reproduction/ablation*, not a method contribution — every number in the docs comes from real runs under `runs/`, negatives included.

**This project is complete.** Milestones M0–M7 have already run on GPU; results live in `runs/m7/` and are summarized in `README.md`. Expect to *read and extend*, not re-bootstrap.

## Canonical docs — read these first

- `AGENTS.md` — the primary agent guide: milestone acceptance checklist, memory-budget table, pit list, loss conventions, "surgical changes only" scope rules. **Authoritative on project constraints.**
- `README.md` — the math (LSM, naive top-K, entropy gate, GRPO, KL drift), result tables, and a "corrections log" (KL-direction fix, K=1 zero-gradient trap, 512-token protocol fix).
- `MicroOPD项目执行文档_v2.md` — the single design/execution doc: architecture decisions, VRAM budget, interview Q&A.

Do not repeat content from these; augment with code-level specifics. Scope is frozen — check the "做/不做" table before proposing new features.

## Commands

Environment: Python 3.11. The user's global rules require asking for a Conda env name before running Python (run with `conda run -n <ENV> python ...`).

```bash
# install
pip install -r requirements.txt

# models (HF unreachable on this machine → ModelScope, into pretrained/)
modelscope download --model Qwen/Qwen3-1.7B --local_dir pretrained/Qwen3-1.7B
modelscope download --model Qwen/Qwen3-8B-FP8 --local_dir pretrained/Qwen3-8B-FP8
# datasets/models are then loaded fully offline:
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

# tests — the only unit suite is the verifier (no GPU needed)
python -m pytest tests/ -q
python -m pytest tests/test_verifier.py::TestExtractAnswer::test_plain_integer -q   # single test

# M0 smoke (VRAM acceptance + Base full eval)
python scripts/smoke_m0.py

# M1 minimal closed loop (10 samples) — the M1 entry point
python scripts/run_iteration.py --config configs/base.yaml --num-samples 10

# experiments (hyperparams only in configs/*.yaml; `--smoke` shrinks scale for a self-check)
python scripts/run_experiment.py --config configs/base.yaml            # OPD LSM K=64
python scripts/run_experiment.py --config configs/sft.yaml             # SFT baseline
python scripts/run_experiment.py --config configs/ablation/k32.yaml    # K-ablation
python scripts/run_experiment.py --config configs/grpo.yaml --phase2   # phase-2 GRPO (needs init_adapter)
python scripts/run_experiment.py --config configs/opd_continue.yaml    # pure-OPD continue control

# evaluation + figures (post-training)
python scripts/final_eval_all.py
python scripts/make_figures.py
python scripts/eval_math500.py          # one-shot OOD transfer test
python scripts/eval_teacher.py          # teacher acc reference (offline, not in loop)
```

There is no linter/formatter configured (no ruff/black/Makefile). Run outputs land in `runs/<name>/` (`config.yaml`, `metrics.jsonl`, `eval.jsonl`, `adapters/`, `cache/`, `figs/`, `final_eval.json`).

## Architecture (big picture)

The whole system is one **PPO-isomorphic on-policy loop** per iteration, orchestrated by `scripts/run_experiment.py`, which dispatches on `experiment.mode` to `run_opd` / `run_sft` / `run_grpo`:

1. **Student samples** (`rollout/student_generate.py`): vLLM generates `y ~ π_S(·|x)` with `enable_thinking=False`.
2. **Teacher scores — one forward, never `generate()`** (`rollout/teacher_score.py`, `models/teacher.py`): the response is appended to the prompt and `prompt_logprobs` extracts the teacher distribution conditioned on the *student's own* tokens. This is what makes it on-policy (vs. offline SFT).
3. **Top-K Cache** (`cache/topk_cache.py`): per-position top-K support `ids [L,K]` + `logprobs [L,K]` serialized to `.npz`; the teacher is then unloaded so it doesn't hold the GPU during training. Derived `mass`/`entropy` are computed on read.
4. **Student LoRA update** (`trainer/distill_trainer.py` → `losses/*`): N = `inner_steps` of 8-bit AdamW over LoRA params only. Full-vocab logits are gathered in `chunk_size` segments (`losses/base.py::gather_support_stats`) to avoid a `[1,T,V]` materialization.
5. **KL-drift gate**: drift ≈ KL(π_old‖π_cur) vs the rollout-time policy; past `kl_drift_threshold` (0.05) it early-stops into the next rollout. This keeps the buffer roughly on-policy.

**Two-phase design.** Phase 1 (`mode: opd` / `sft`) is teacher-signal distillation. Phase 2 (`mode: grpo`, `--phase2`) drops the teacher entirely — reward comes from `evaluation/verifier.py` exact-match, group-normalized advantage, ratio clip, KL to a reference distribution obtained via `disable_adapter` on the same weights (`models/student.py::ref_context`). The two phases share the same rollout + trainer, differing only in loss and signal source.

**Losses are pluggable**, all with the same signature `loss_fn(model, sample, topk, cfg, device) -> (loss, stats)`, selected by `distill.loss`:
- `lsm` (`losses/lsm.py`): **reverse** KL = KL(student‖teacher), with both sides re-normalized inside the support set.
- `naive_topk` (`losses/naive_topk.py`): truncated teacher-weighted CE, no renormalization (control).
- `entropy_gate` (`losses/entropy_gate.py`): interpolates reverse/forward KL per token.
- `sft` (`losses/sft.py`): teacher-trajectory CE (offline baseline).
- `grpo` (`losses/grpo.py`): phase-2.

**GPU discipline (hard 16GB, invariant).** Teacher and student are time-sliced on one card — peak is max, not sum. Before an engine is created, the HF student is moved to CPU (`models/student.py::student_to_cpu`); unloading a vLLM engine requires an explicit `engine_core.shutdown()` and waiting for VRAM to fall (a `del` alone leaves ~11GB). `common.py` holds config loading (with `inherits:` deep-merge), prompt/chat-template building, and seed/VRAM utilities.

**Config conventions.** All hyperparameters, the eval protocol, and model paths live in `configs/*.yaml` (nothing hardcoded in code); ablation configs `inherits: base.yaml`. `common.py::load_config` resolves relative-inherit chains; `encode_prompt` derives the response boundary from the tokenizer's actual encoding (never hand-compute prompt/response split).

**Key invariants easy to break** (full list in `AGENTS.md` §6/§10):
- Teacher must score via `prompt_logprobs(prompt + student_response)`; `teacher.generate()` silently turns OPD back into SFT.
- Teacher scores exactly once, then is unloaded; GRPO never imports `models/teacher.py`.
- LSM is reverse KL with renorm on **both** sides; K=1 degenerates to teacher-argmax CE (`losses/base.py::k1_argmax_ce`), not to a zero-gradient KL.
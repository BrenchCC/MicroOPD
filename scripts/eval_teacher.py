"""教师模型 GSM8K accuracy 一次性评测（区间参照用，M0 遗留项）。

协议与学生评测完全一致（config: eval + generation）：贪婪解码、0-shot、
`####` 提取、max_new_tokens=512。教师 generate 仅用于本次离线评测，
不进 OPD 训练循环（架构原则 1 约束的是蒸馏闭环，不是评测）。

用法：
    python scripts/eval_teacher.py --config configs/base.yaml                # 全量 test 1319
    python scripts/eval_teacher.py --num-questions 20                        # 小规模自检
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import encode_prompt, load_config, set_seed, vram_peak_gb, vram_reset_peak
from evaluation.gsm8k import load_gsm8k
from evaluation.verifier import score_responses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--num-questions", type=int, default=None,
                    help="默认 None = 全量 eval_split")
    ap.add_argument("--out", default=None,
                    help="默认 runs/teacher_eval.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["student"])  # 师生同词表
    items = load_gsm8k(cfg, cfg["data"]["eval_split"])
    if args.num_questions:
        items = items[: args.num_questions]
    prompt_ids = [
        encode_prompt(tokenizer, it["question"], cfg["models"]["enable_thinking"])
        for it in items
    ]

    from models.teacher import load_teacher_engine, unload_teacher
    from rollout.teacher_score import generate_teacher_responses

    vram_reset_peak()
    t0 = time.time()
    engine = load_teacher_engine(cfg)
    responses = generate_teacher_responses(
        engine, prompt_ids,
        cfg["eval"]["temperature"], cfg["eval"]["top_p"],
        cfg["generation"]["max_new_tokens"],
        seed=cfg["experiment"]["seed"],
    )
    unload_teacher(engine)
    elapsed = time.time() - t0

    texts = [r["text"] for r in responses]
    lens = [len(r["response_ids"]) for r in responses]
    rewards, fail_rate = score_responses(texts, [it["gold_text"] for it in items])
    n = len(items)
    metrics = {
        "model": cfg["models"]["teacher"],
        "protocol": "greedy 0-shot #### (与学生评测同一管线)",
        "accuracy": sum(rewards) / n if n else 0.0,
        "avg_len": sum(lens) / n if n else 0.0,
        "extract_fail_rate": fail_rate,
        "n": n,
        "vram_peak_gb": round(vram_peak_gb(), 2),
        "wall_time_min": round(elapsed / 60, 1),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    out = args.out or os.path.join(cfg["logging"]["run_root"], "teacher_eval.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[done] 写入 {out}")


if __name__ == "__main__":
    main()

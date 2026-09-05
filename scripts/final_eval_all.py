"""对 runs/ 下各实验的最终 adapter 做全量 GSM8K test（1319 题）评测。

用法：python scripts/final_eval_all.py [run_name ...]（默认所有含 adapters/final 的 run）
产物：runs/<name>/full_eval.json（accuracy / avg_len / fail_rate / n=1319）
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import load_config
from evaluation.gsm8k import evaluate_accuracy, load_gsm8k
from models.student import load_tokenizer, unload
from rollout.student_generate import StudentGenerator


def main():
    run_root = "runs"
    names = sys.argv[1:]
    if not names:
        names = sorted(
            d for d in os.listdir(run_root)
            if os.path.isdir(os.path.join(run_root, d, "adapters", "final"))
        )
    print(f"全量评测 runs: {names}")

    for name in names:
        run_dir = os.path.join(run_root, name)
        out_path = os.path.join(run_dir, "full_eval.json")
        if os.path.exists(out_path):
            print(f"[skip] {name} 已有 full_eval.json")
            continue
        cfg = load_config(os.path.join(run_dir, "config.yaml"))
        tokenizer = load_tokenizer(cfg)
        items = load_gsm8k(cfg, cfg["data"]["eval_split"])
        sync = {"type": "lora", "path": os.path.join(run_dir, "adapters", "final")}
        gen = StudentGenerator(cfg, sync)
        res = evaluate_accuracy(gen, tokenizer, items, cfg, limit=None)  # 全量 1319
        gen.close()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[{name}] acc={res['accuracy']:.4f} len={res['avg_len']:.1f} "
              f"fail={res['extract_fail_rate']:.3f}")
        unload()


if __name__ == "__main__":
    main()

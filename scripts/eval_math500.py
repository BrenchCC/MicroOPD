"""MATH-500 一次性迁移测试（M7）：base / 最优 OPD / GRPO 三模型对比。

用法：python scripts/eval_math500.py
产物：runs/m7/math500.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import load_config
from evaluation.math500 import evaluate_math500
from models.student import load_tokenizer, unload
from rollout.student_generate import StudentGenerator


class BaseGenerator:  # 无 LoRA 的基座学生（与 StudentGenerator 同接口）
    def __init__(self, cfg):
        from vllm import LLM

        self.llm = LLM(
            model=cfg["models"]["student"],
            gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization_student"],
            max_model_len=cfg["vllm"]["max_model_len"],
            dtype="bfloat16",
            enforce_eager=cfg["vllm"].get("enforce_eager", False),
        )

    def generate(self, prompt_ids_list, temperature, top_p, max_new_tokens, n=1,
                 seed=None, want_logprobs=False):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        sp = SamplingParams(n=n, temperature=temperature, top_p=top_p,
                            max_tokens=max_new_tokens, seed=seed)
        outs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=i) for i in prompt_ids_list], sp)
        return [[{"response_ids": list(o.token_ids), "text": o.text,
                  "logprobs": None} for o in out.outputs] for out in outs]

    def close(self):
        unload(self.llm)


def main():
    cfg = load_config("configs/base.yaml")
    tokenizer = load_tokenizer(cfg)
    targets = {
        "base": None,
        "lsm_k8": "runs/lsm_k8/adapters/final",   # M3 最优 LSM（GRPO 的热启动点）
        "grpo": "runs/grpo/adapters/final",
    }
    results = {}
    for name, adapter in targets.items():
        if adapter is None:
            gen = BaseGenerator(cfg)
        else:
            gen = StudentGenerator(cfg, {"type": "lora", "path": adapter})
        res = evaluate_math500(gen, tokenizer, cfg)
        gen.close()
        results[name] = res
        print(f"[math500] {name}: acc={res['accuracy']:.4f} len={res['avg_len']:.1f}")
    os.makedirs("runs/m7", exist_ok=True)
    with open("runs/m7/math500.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

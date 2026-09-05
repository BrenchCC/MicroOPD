"""M0 冒烟：双模型显存验收 + 学生 baseline GSM8K 评测。

验收项（执行文档 M0）：
- [ ] 教师 FP8 推理显存 ≤ 10GB
- [ ] 学生 LoRA 训练态显存 ≤ 5GB
- [ ] 峰值合计（分时复用取 max）≤ 15GB
- [ ] 用自己的管线测出学生 baseline GSM8K acc 并记录（runs/m0/baseline_eval.json）

用法：
  python scripts/smoke_m0.py [--config configs/base.yaml] [--skip-eval] [--eval-limit N]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from common import encode_prompt, load_config, set_seed, vram_peak_gb, vram_reset_peak


def _nvml_used_gb():
    """整卡已用显存（GB）。vLLM V1 引擎在子进程，torch 统计抓不到，用 NVML。"""
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    used = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e9
    pynvml.nvmlShutdown()
    return used


def check_teacher(cfg):
    """教师 FP8 vLLM：加载 + 一次 prompt_logprobs 前向，报告峰值显存。"""
    from models.teacher import load_teacher_engine, unload_teacher
    from rollout.teacher_score import score_teacher

    engine = load_teacher_engine(cfg)
    # 一次真实打分前向（prompt+response 拼接），验证 FP8 推理路径
    ids = list(range(100, 300))
    scores = score_teacher(engine, [ids], [50], cfg["distill"]["K"])
    ids_top, lps = scores[0]
    assert ids_top.shape == (150, cfg["distill"]["K"])
    peak = _nvml_used_gb()  # vLLM 引擎在子进程，torch 统计抓不到，用 NVML 整卡读数
    unload_teacher(engine)
    print(f"[M0] 教师 FP8 峰值显存: {peak:.2f} GB（验收 ≤10GB）"
          + (" ✓" if peak <= 10 else " ✗ 超预算"))
    return peak


def check_student_train(cfg):
    """学生 LoRA 训练态：seq=256 一次 forward+backward，报告峰值显存。"""
    from models.student import build_optimizer, load_student, unload

    vram_reset_peak()
    model = load_student(cfg)
    optimizer = build_optimizer(model, cfg["distill"]["lr"])
    ids = torch.randint(0, 1000, (1, 256), device=cfg["experiment"]["device"])
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    peak = vram_peak_gb()
    del out, ids
    unload(model, optimizer)
    print(f"[M0] 学生 LoRA 训练态峰值显存: {peak:.2f} GB（验收 ≤5GB）"
          + (" ✓" if peak <= 5 else " ✗ 超预算"))
    return peak


def run_baseline_eval(cfg, limit=None):
    """学生基座（无 LoRA）GSM8K 评测：vLLM 直接加载基座权重。"""
    from vllm import LLM

    from evaluation.gsm8k import evaluate_accuracy, load_gsm8k
    from models.student import load_tokenizer, unload

    class _Gen:  # 与 StudentGenerator 同接口的最小包装（无 LoRA）
        def __init__(self, cfg):
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

    tokenizer = load_tokenizer(cfg)
    items = load_gsm8k(cfg, cfg["data"]["eval_split"])
    vram_reset_peak()
    gen = _Gen(cfg)
    res = evaluate_accuracy(gen, tokenizer, items, cfg, limit)
    res["vram_peak_gb"] = _nvml_used_gb()  # vLLM 子进程显存，NVML 整卡读数
    unload(gen.llm)
    print(f"[M0] 学生 baseline GSM8K: acc={res['accuracy']:.4f} "
          f"avg_len={res['avg_len']:.1f} fail_rate={res['extract_fail_rate']:.3f} "
          f"n={res['n']} vram={res['vram_peak_gb']:.2f}GB")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--eval-limit", type=int, default=None,
                    help="默认全量 test（1319）；调试用可设小")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    run_dir = os.path.join(cfg["logging"]["run_root"], "m0")
    os.makedirs(run_dir, exist_ok=True)

    teacher_peak = check_teacher(cfg)
    student_peak = check_student_train(cfg)
    combined = max(teacher_peak, student_peak)
    print(f"[M0] 分时复用峰值（取 max）: {combined:.2f} GB（验收 ≤15GB）"
          + (" ✓" if combined <= 15 else " ✗ 超预算"))

    result = {
        "teacher_fp8_vram_gb": teacher_peak,
        "student_train_vram_gb": student_peak,
        "combined_peak_gb": combined,
        "teacher_quantization": cfg["models"]["teacher_quantization"],
    }
    if not args.skip_eval:
        result["baseline_eval"] = run_baseline_eval(cfg, args.eval_limit)

    with open(os.path.join(run_dir, "m0_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[M0] 结果写入 {run_dir}/m0_results.json")


if __name__ == "__main__":
    main()

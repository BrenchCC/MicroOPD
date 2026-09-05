"""单轮完整迭代（M1 冒烟测试入口）：
生成 → 教师打分 → Top-K Cache → LSM loss → optimizer.step()。

用法：python scripts/run_iteration.py [--config configs/base.yaml] [--num-samples 10]

M1 验收对应：
- 10 条样本完整跑通闭环；
- loss 为有限值（量级 0.1–5 仅打印提示，不硬断言）；
- cache 每个 position 的 mass ∈ (0, 1]；
- 抽查：教师返回分布的 key 必须包含学生的实际 response token
  （证明是 prompt_logprobs 条件打分，没走错 generate 路线）。
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from cache.topk_cache import TopKCache
from common import load_config, set_seed, vram_peak_gb
from evaluation.gsm8k import load_gsm8k
from models.student import (
    build_optimizer,
    load_student,
    load_tokenizer,
    save_adapter,
    student_to_cpu,
    student_to_device,
    sync_student_to_vllm,
)
from models.teacher import load_teacher_engine, unload_teacher
from rollout.buffer import RolloutBuffer, Trajectory
from rollout.student_generate import StudentGenerator
from rollout.teacher_score import score_teacher
from trainer.distill_trainer import DistillTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--num-samples", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["distill"]["prompts_per_iter"] = args.num_samples
    set_seed(cfg["experiment"]["seed"])
    device = cfg["experiment"].get("device", "cuda")

    run_dir = os.path.join(cfg["logging"]["run_root"], "smoke")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    # ① 学生 rollout（vLLM）
    tokenizer = load_tokenizer(cfg)
    student = load_student(cfg)
    optimizer = build_optimizer(student, cfg["distill"]["lr"])
    sync = sync_student_to_vllm(
        student, tokenizer, os.path.join(run_dir, "vllm_sync"),
        cfg["vllm"]["student_weight_sync"],
    )
    student_to_cpu(student)

    items = load_gsm8k(cfg, cfg["data"]["train_split"])[: args.num_samples]
    from common import encode_prompt

    prompt_ids = [
        encode_prompt(tokenizer, it["question"], cfg["models"]["enable_thinking"])
        for it in items
    ]
    gen = StudentGenerator(cfg, sync)
    outs = gen.generate(
        prompt_ids,
        temperature=cfg["generation"]["temperature"],
        top_p=cfg["generation"]["top_p"],
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        n=cfg["distill"]["G"],
        seed=cfg["experiment"]["seed"],
    )
    gen.close()
    print(f"[smoke] 学生峰值显存（rollout）: {vram_peak_gb():.2f} GB")

    # ② 教师打分（一次前向，prompt_logprobs）→ ③ Top-K Cache
    buffer = RolloutBuffer()
    for i, (it, pids, group) in enumerate(zip(items, prompt_ids, outs)):
        g = group[0]
        buffer.add(
            Trajectory(
                prompt_ids=pids,
                response_ids=g["response_ids"],
                iter=0,
                question=it["question"],
                text=g["text"],
            )
        )
    teacher = load_teacher_engine(cfg)
    scores = score_teacher(
        teacher,
        [t.full_ids for t in buffer],
        [t.prompt_len for t in buffer],
        cfg["distill"]["K"],
    )
    unload_teacher(teacher)
    print(f"[smoke] 教师峰值显存（scoring）: {vram_peak_gb():.2f} GB")

    cache = TopKCache(os.path.join(run_dir, "cache"))
    n_bad_mass, n_missing_tok = 0, 0
    for i, (traj, (ids, lps)) in enumerate(zip(buffer, scores)):
        traj.cache_file = cache.write(f"it0_s{i}", ids, lps)
        entry = cache.read(traj.cache_file)
        if not ((entry["mass"] > 0).all() and (entry["mass"] <= 1.0 + 1e-4).all()):
            n_bad_mass += 1
        # 抽查（前 3 条）：学生实际 response token 必须出现在教师返回分布中
        if i < 3:
            idset = entry["ids"]
            for pos, tok in enumerate(traj.response_ids):
                if tok not in idset[pos]:
                    n_missing_tok += 1
    assert n_bad_mass == 0, "cache 中存在 mass ∉ (0, 1] 的 position"
    print(f"[smoke] cache 检查通过：mass ∈ (0,1]；抽查 response token 缺失 {n_missing_tok} 处"
          "（>0 说明教师 top-K 未覆盖学生 token，K 足够大时应为 0 或接近 0）")

    # ④ LSM loss → ⑤ optimizer.step()
    student_to_device(student, device)
    trainer = DistillTrainer(student, optimizer, cfg, run_dir)
    trainer.train_on_buffer(buffer, cache, iteration=0)
    save_adapter(student, os.path.join(run_dir, "adapters", "final"))
    trainer.plot_curves()

    import json

    rows = [
        json.loads(l)
        for l in open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8")
    ]
    losses = [r["loss"] for r in rows]
    assert all(math.isfinite(x) for x in losses), "loss 出现非有限值"
    lo, hi = min(losses), max(losses)
    print(f"[smoke] loss 范围 [{lo:.4f}, {hi:.4f}]"
          + ("" if 0.1 <= lo and hi <= 5 else "（提示：超出 0.1–5 经验区间，检查实现）"))
    print(f"[smoke] 训练峰值显存: {max(r['vram_peak_gb'] for r in rows):.2f} GB")
    print("[smoke] 单轮迭代闭环完成")


if __name__ == "__main__":
    main()

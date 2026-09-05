"""按 config 跑完整实验。

用法：
  一期 OPD / SFT：python scripts/run_experiment.py --config configs/base.yaml
  M3 消融：      python scripts/run_experiment.py --config configs/ablation/k32.yaml
  二期 GRPO：    python scripts/run_experiment.py --config configs/grpo.yaml --phase2
  快速自检：     --smoke（缩减 prompts/iterations/eval 题数）

产物（runs/<name>/）：config 副本、metrics.jsonl、eval.jsonl、adapters/、
cache/、figs/、final_eval.json。
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from cache.topk_cache import TopKCache
from common import encode_prompt, load_config, set_seed
from evaluation.agreement import evaluate_agreement
from evaluation.gsm8k import evaluate_accuracy, load_gsm8k
from evaluation.verifier import score_responses
from models.student import (
    build_optimizer,
    load_student,
    load_tokenizer,
    save_adapter,
    student_to_cpu,
    student_to_device,
    sync_student_to_vllm,
)
from rollout.buffer import RolloutBuffer, Trajectory
from rollout.student_generate import StudentGenerator
from rollout.teacher_score import generate_teacher_responses, score_teacher
from trainer.distill_trainer import DistillTrainer


def setup_run(cfg):
    run_dir = os.path.join(cfg["logging"]["run_root"], cfg["experiment"]["name"])
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return run_dir


def pick_items(items, iteration, n, seed):
    """每个 iteration 取训练集的一个轮换窗口（seed 固定的全局 shuffle 顺序）。"""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    start = (iteration * n) % len(order)
    return [items[order[(start + j) % len(order)]] for j in range(n)]


def rollout_student(cfg, tokenizer, model, run_dir, prompt_ids, n, want_logprobs=False,
                    temperature=None, top_p=None, tag="sync"):
    """同步权重 → HF 学生下卡 → vLLM 采样 → 关引擎 → HF 学生回卡。"""
    device = cfg["experiment"].get("device", "cuda")
    sync = sync_student_to_vllm(
        model, tokenizer, os.path.join(run_dir, "vllm_sync", tag),
        cfg["vllm"]["student_weight_sync"],
    )
    student_to_cpu(model)
    gen = StudentGenerator(cfg, sync)
    gcfg = cfg["generation"]
    outs = gen.generate(
        prompt_ids,
        temperature=temperature if temperature is not None else gcfg["temperature"],
        top_p=top_p if top_p is not None else gcfg["top_p"],
        max_new_tokens=gcfg["max_new_tokens"],
        n=n,
        seed=cfg["experiment"]["seed"],
        want_logprobs=want_logprobs,
    )
    gen.close()
    student_to_device(model, device)
    return outs


def run_eval(cfg, tokenizer, model, run_dir, eval_items, iteration):
    """周期性 GSM8K 评测（贪婪解码），结果追加 eval.jsonl。"""
    device = cfg["experiment"].get("device", "cuda")
    was_training = model.training
    model.eval()
    sync = sync_student_to_vllm(
        model, tokenizer, os.path.join(run_dir, "vllm_sync", "eval"),
        cfg["vllm"]["student_weight_sync"],
    )
    student_to_cpu(model)
    gen = StudentGenerator(cfg, sync)
    res = evaluate_accuracy(gen, tokenizer, eval_items, cfg, cfg["eval"]["num_questions"])
    gen.close()
    student_to_device(model, device)
    if was_training:
        model.train()
    rec = {"iter": iteration, **res}
    with open(os.path.join(run_dir, "eval.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[eval] iter={iteration} acc={res['accuracy']:.4f} "
          f"avg_len={res['avg_len']:.1f} fail_rate={res['extract_fail_rate']:.3f}")
    return res


def run_opd(cfg, tokenizer, model, optimizer, trainer, run_dir, train_items, eval_items):
    dcfg = cfg["distill"]
    n_iter = dcfg["iterations"]
    last_buffer, last_cache = None, None

    run_eval(cfg, tokenizer, model, run_dir, eval_items, iteration=-1)  # Base 行（M0）
    for it in range(n_iter):
        batch = pick_items(train_items, it, dcfg["prompts_per_iter"], cfg["experiment"]["seed"])
        prompt_ids = [
            encode_prompt(tokenizer, b["question"], cfg["models"]["enable_thinking"])
            for b in batch
        ]
        outs = rollout_student(cfg, tokenizer, model, run_dir, prompt_ids,
                               n=dcfg["G"], tag=f"iter_{it}")

        # 教师打分：只发生一次，进 cache 复用
        from models.teacher import load_teacher_engine, unload_teacher

        student_to_cpu(model)
        teacher = load_teacher_engine(cfg)
        buffer = RolloutBuffer()
        full_ids, plens = [], []
        for i, (b, pids, group) in enumerate(zip(batch, prompt_ids, outs)):
            for g in group:
                buffer.add(Trajectory(prompt_ids=pids, response_ids=g["response_ids"],
                                      iter=it, question=b["question"], text=g["text"]))
        scores = score_teacher(teacher, [t.full_ids for t in buffer],
                               [t.prompt_len for t in buffer], dcfg["K"])
        unload_teacher(teacher)
        student_to_device(model, cfg["experiment"].get("device", "cuda"))

        cache = TopKCache(os.path.join(run_dir, "cache", f"iter_{it}"))
        for i, (traj, (ids, lps)) in enumerate(zip(buffer, scores)):
            traj.cache_file = cache.write(f"it{it}_s{i}", ids, lps)

        early = trainer.train_on_buffer(buffer, cache, iteration=it)
        save_adapter(model, os.path.join(run_dir, "adapters", f"iter_{it}"))
        if early:
            print(f"[iter {it}] KL 漂移超阈值，提前进入下一轮 rollout")
        if (it + 1) % cfg["eval"]["every"] == 0 or it == n_iter - 1:
            run_eval(cfg, tokenizer, model, run_dir, eval_items, iteration=it)
        last_buffer, last_cache = buffer, cache

    save_adapter(model, os.path.join(run_dir, "adapters", "final"))
    return last_buffer, last_cache


def run_sft(cfg, tokenizer, model, optimizer, trainer, run_dir, train_items, eval_items):
    """M2 Baseline 1：教师 generate 造数据（离线，只造一次）→ 学生 SFT。

    - sft_data.jsonl 已存在则复用（教师造数只发生一次，重训不重复烧教师时间）；
    - sft.filter_no_answer: 过滤无 #### 可解析答案的轨迹（32.5% 教师轨迹顶满
      256 token 被截断，不滤会把"不收尾"风格教给学生 —— 实测 fail_rate 翻倍、
      acc 反降。拒绝采样式过滤是标准做法）。
    """
    from models.teacher import load_teacher_engine, unload_teacher

    dcfg = cfg["distill"]
    data_path = os.path.join(run_dir, "sft_data.jsonl")
    if os.path.exists(data_path):
        print(f"[sft] 复用已有教师数据 {data_path}")
        buffer = RolloutBuffer.load_jsonl(data_path)
    else:
        n_data = dcfg["prompts_per_iter"] * dcfg["iterations"]
        batch = pick_items(train_items, 0, n_data, cfg["experiment"]["seed"])
        prompt_ids = [
            encode_prompt(tokenizer, b["question"], cfg["models"]["enable_thinking"])
            for b in batch
        ]
        student_to_cpu(model)
        teacher = load_teacher_engine(cfg)
        responses = generate_teacher_responses(
            teacher, prompt_ids, cfg["generation"]["temperature"],
            cfg["generation"]["top_p"], cfg["generation"]["max_new_tokens"],
            seed=cfg["experiment"]["seed"],
        )
        unload_teacher(teacher)
        student_to_device(model, cfg["experiment"].get("device", "cuda"))

        buffer = RolloutBuffer()
        for b, pids, r in zip(batch, prompt_ids, responses):
            buffer.add(Trajectory(prompt_ids=pids, response_ids=r["response_ids"],
                                  iter=0, question=b["question"], text=r["text"]))
        buffer.save_jsonl(data_path)

    if cfg.get("sft", {}).get("filter_no_answer", False):
        from evaluation.verifier import extract_answer

        kept = [t for t in buffer if extract_answer(t.text) is not None]
        print(f"[sft] 拒绝采样过滤：{len(buffer)} → {len(kept)} 条"
              f"（{1 - len(kept) / max(len(buffer), 1):.1%} 无 #### 答案被滤掉）")
        buffer._items = kept

    run_eval(cfg, tokenizer, model, run_dir, eval_items, iteration=-1)
    # epoch 数独立配置（sft_epochs）：iterations 在 SFT 模式只决定教师造数规模
    n_epochs = cfg.get("sft_epochs", dcfg["iterations"])
    for epoch in range(n_epochs):
        trainer.train_on_buffer(buffer, cache=None, iteration=0)  # SFT 无 cache
        if (epoch + 1) % cfg["eval"]["every"] == 0 or epoch == n_epochs - 1:
            run_eval(cfg, tokenizer, model, run_dir, eval_items, iteration=epoch)
    save_adapter(model, os.path.join(run_dir, "adapters", "final"))
    return buffer, None


def run_grpo(cfg, tokenizer, model, optimizer, trainer, run_dir, train_items, eval_items):
    """二期 M6：教师完全不加载；reward 来自 verifier；ref 分布 disable_adapter。"""
    dcfg, gcfg = cfg["distill"], cfg["grpo"]
    n_iter = dcfg["iterations"]

    run_eval(cfg, tokenizer, model, run_dir, eval_items, iteration=-1)
    for it in range(n_iter):
        batch = pick_items(train_items, it, dcfg["prompts_per_iter"], cfg["experiment"]["seed"])
        prompt_ids = [
            encode_prompt(tokenizer, b["question"], cfg["models"]["enable_thinking"])
            for b in batch
        ]
        outs = rollout_student(cfg, tokenizer, model, run_dir, prompt_ids,
                               n=gcfg["G"], want_logprobs=True,
                               temperature=gcfg["temperature"], top_p=gcfg["top_p"],
                               tag=f"iter_{it}")

        groups, fail_rates = [], []
        for b, pids, group in zip(batch, prompt_ids, outs):
            texts = [g["text"] for g in group]
            rewards, fail_rate = score_responses(texts, [b["gold_text"]] * len(texts))
            fail_rates.append(fail_rate)
            groups.append([
                Trajectory(prompt_ids=pids, response_ids=g["response_ids"], iter=it,
                           question=b["question"], text=g["text"], reward=r,
                           old_logprobs=g["logprobs"])
                for g, r in zip(group, rewards)
            ])
        mean_fail = sum(fail_rates) / len(fail_rates)
        print(f"[iter {it}] verifier 提取失败率 {mean_fail:.3f}"
              + ("（>5%：先在 prompt 强约束 #### 格式再训）" if mean_fail > 0.05 else ""))

        trainer.train_grpo(groups, iteration=it)
        save_adapter(model, os.path.join(run_dir, "adapters", f"iter_{it}"))
        if (it + 1) % cfg["eval"]["every"] == 0 or it == n_iter - 1:
            run_eval(cfg, tokenizer, model, run_dir, eval_items, iteration=it)

    save_adapter(model, os.path.join(run_dir, "adapters", "final"))
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase2", action="store_true", help="进入 GRPO 阶段（config 需含 grpo 节）")
    ap.add_argument("--smoke", action="store_true", help="缩减规模快速自检")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.phase2:
        cfg["experiment"]["mode"] = "grpo"
        assert "grpo" in cfg, "--phase2 需要 config 中包含 grpo 节（见 configs/grpo.yaml）"
        assert cfg.get("init_adapter"), "GRPO 需从 OPD checkpoint 热启动：配置 init_adapter"
    if args.smoke:
        cfg["distill"]["prompts_per_iter"] = 8
        cfg["distill"]["iterations"] = 2
        cfg["distill"]["inner_steps"] = 2
        cfg["eval"]["num_questions"] = 16
        cfg["eval"]["every"] = 1
    set_seed(cfg["experiment"]["seed"])

    run_dir = setup_run(cfg)
    tokenizer = load_tokenizer(cfg)
    mode = cfg["experiment"]["mode"]
    # init_adapter：GRPO 热启动必需；OPD 可选（M6 对照组"pure OPD 续训"从同一 checkpoint 继续）
    init_adapter = cfg.get("init_adapter")
    model = load_student(cfg, adapter_path=init_adapter)
    optimizer = build_optimizer(model, cfg["distill"]["lr"])
    trainer = DistillTrainer(model, optimizer, cfg, run_dir)

    train_items = load_gsm8k(cfg, cfg["data"]["train_split"])
    eval_items = load_gsm8k(cfg, cfg["data"]["eval_split"])

    if mode == "opd":
        buffer, cache = run_opd(cfg, tokenizer, model, optimizer, trainer,
                                run_dir, train_items, eval_items)
    elif mode == "sft":
        buffer, cache = run_sft(cfg, tokenizer, model, optimizer, trainer,
                                run_dir, train_items, eval_items)
    elif mode == "grpo":
        buffer, cache = run_grpo(cfg, tokenizer, model, optimizer, trainer,
                                 run_dir, train_items, eval_items)
    else:
        raise ValueError(f"unknown mode: {mode}")

    # 收尾：agreement 诊断（仅 OPD）、曲线、最终 eval 汇总
    final = {"mode": mode, "name": cfg["experiment"]["name"]}
    if mode == "opd" and buffer is not None and cache is not None:
        model.eval()
        agree = evaluate_agreement(
            model, list(buffer), cache, cfg,
            cfg["experiment"].get("device", "cuda"), limit=16,
        )
        final["agreement"] = agree
        print(f"[final] top1_match={agree['top1_match']:.4f} "
              f"mass_coverage={agree['mass_coverage']:.4f}")
    eval_path = os.path.join(run_dir, "eval.jsonl")
    if os.path.exists(eval_path):
        final["eval_history"] = [
            json.loads(l) for l in open(eval_path, encoding="utf-8")
        ]
    with open(os.path.join(run_dir, "final_eval.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    trainer.plot_curves()
    print(f"[done] 产物见 {run_dir}")


if __name__ == "__main__":
    main()

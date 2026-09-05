"""Teacher Agreement 评测：Top1 match 与 Mass Coverage（评测 KL 与 Top5 已裁）。

- Mass Coverage：学生分布在教师 top-K 支撑集上的概率质量（定义见 losses.base）。
  它直接检验 LSM 的前提假设是否成立，是结果也是诊断。
- Top1 match：教师 top-1 token 是否为学生在支撑集内的 argmax。
  注意这是近似（学生真正的全词表 argmax 可能在支撑集外）；精确版需要 full-vocab
  argmax，代价不符，约定用支撑集内版本并在此注明。
"""
import torch

from losses.base import gather_support_stats, prepare_sample, student_mass_coverage


@torch.no_grad()
def evaluate_agreement(model, buffer, cache, cfg, device, limit=None):
    samples = buffer if not isinstance(buffer, list) else buffer
    if limit:
        samples = samples[:limit]
    chunk = cfg["distill"]["chunk_size"]
    top1_hits, masses, n_pos = 0, [], 0
    for s in samples:
        topk = cache.read(s.cache_file)
        input_ids, sup_ids, _ = prepare_sample(s, topk, device)
        stats = gather_support_stats(model, input_ids, s.prompt_len, sup_ids, chunk)
        s_argmax = stats["support_logits"].argmax(dim=-1)      # 支撑集内 argmax 的列号
        t_top1 = torch.zeros_like(s_argmax)                    # cache 已按 logprob 降序，第 0 列即教师 top1
        top1_hits += int((s_argmax == t_top1).sum().item())
        masses.append(student_mass_coverage(stats["support_logits"], stats["lse"]).mean().item())
        n_pos += s_argmax.numel()
    return {
        "top1_match": top1_hits / n_pos if n_pos else 0.0,
        "mass_coverage": sum(masses) / len(masses) if masses else 0.0,
        "n_samples": len(samples),
    }

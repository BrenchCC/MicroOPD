"""熵感知门控（M4 可选）：按 token 在 reverse / forward KL 之间 sigmoid 插值。

方向约定（与 LSM 论文 / MiniLLM 一致）：
- reverse KL = KL(学生‖教师)，期望在学生分布下取，mode-seeking；
- forward KL = KL(教师‖学生)，期望在教师分布下取，mass-covering。

动机：reverse KL 在低熵（mode-seeking）token 上合理，但 top-K 截断使熵系统性
偏低，高不确定 token 上纯 reverse 会过度 mode-seeking（输出坍缩风险）。
门控特征 f = H_norm + (1 − mass)：H_norm 为支撑集内重归一化熵 / logK；
mass 为**重归一化之前**的教师 top-K 原始概率质量（∈(0,1]），(1 − mass) 补
截断长尾信息（mass 小说明分布平、top-K 外还有质量）。
w = sigmoid(gain * (f − threshold))；loss = (1−w)·reverse + w·forward，
即低熵 token 走 reverse（mode-seeking），高熵 token 走 forward（mass-covering）。
"""
import math

import torch
import torch.nn.functional as F

from losses.base import gather_support_stats, prepare_sample, student_mass_coverage


def entropy_gate_loss(model, sample, topk, cfg, device):
    input_ids, sup_ids, t_logprobs = prepare_sample(sample, topk, device)
    stats = gather_support_stats(
        model, input_ids, sample.prompt_len, sup_ids, cfg["distill"]["chunk_size"]
    )
    s_logits = stats["support_logits"]
    K = sup_ids.shape[1]

    log_pt = F.log_softmax(t_logprobs, dim=-1)
    pt = log_pt.exp()
    log_ps = F.log_softmax(s_logits, dim=-1)
    ps = log_ps.exp()

    rev = (ps * (log_ps - log_pt)).sum(dim=-1)          # reverse KL(学生‖教师) [L]
    fwd = (pt * (log_pt - log_ps)).sum(dim=-1)          # forward KL(教师‖学生) [L]

    with torch.no_grad():
        h_norm = -(pt * log_pt).sum(dim=-1) / math.log(max(K, 2))  # [L] ∈ [0,1]
        mass = t_logprobs.exp().sum(dim=-1)              # 教师 top-K 原始 mass（重归一化之前）
        f = h_norm + (1.0 - mass)
        gcfg = cfg["entropy_gate"]
        w = torch.sigmoid(gcfg["gain"] * (f - gcfg["threshold"]))

    loss = ((1.0 - w) * rev + w * fwd).mean()

    with torch.no_grad():
        mc = student_mass_coverage(s_logits, stats["lse"]).mean()
    return loss, {
        "kl": rev.mean().item(),
        "mass_coverage": mc.item(),
        "gate_w_mean": w.mean().item(),   # M4 验收：验证低熵 token 上走 reverse（w→0）
        "entropy_norm_mean": h_norm.mean().item(),
        "resp_logp": stats["resp_logp"],
    }

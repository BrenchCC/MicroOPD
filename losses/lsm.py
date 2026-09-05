"""LSM（Local Support Matching）：支撑集内重归一化 reverse KL。

出处：Revisiting On-Policy Distillation（arXiv 2603.25562）Eq. (8)：
reverse KL = KL(学生‖教师)，期望在**学生**分布下取（student-weighted，
mode-seeking），与 MiniLLM 的 reverse KL 约定一致。
即 kl = Σ_v p̂_student(v) * (log p̂_student(v) − log q̂_teacher(v))。

- 教师、学生**两侧**都在支撑集内 softmax 重归一化（截断后两侧质量都 <1，
  不归一则两侧不可比，优化不稳；论文消融：去掉重归一化会快速坍缩）。
- K=1 特例：单元素支撑集上双侧重归一化使 KL 恒 0（无梯度），按退化链语义
  改为教师 argmax 交叉熵 ≈ SFT（见 losses.base.k1_argmax_ce）。
  退化链：SFT ⊂ top-K OPD ⊂ full-vocab OPD。
"""
import torch
import torch.nn.functional as F

from losses.base import (
    gather_support_stats,
    k1_argmax_ce,
    prepare_sample,
    student_mass_coverage,
)


def lsm_loss(model, sample, topk, cfg, device):
    input_ids, sup_ids, t_logprobs = prepare_sample(sample, topk, device)
    stats = gather_support_stats(
        model, input_ids, sample.prompt_len, sup_ids, cfg["distill"]["chunk_size"]
    )
    if sup_ids.shape[1] == 1:  # K=1：退化为教师 argmax CE（单元支撑集上 KL 恒 0）
        return k1_argmax_ce(stats)
    s_logits = stats["support_logits"]

    log_pt = F.log_softmax(t_logprobs, dim=-1)  # 教师侧重归一化
    log_ps = F.log_softmax(s_logits, dim=-1)    # 学生侧重归一化
    ps = log_ps.exp()
    kl = (ps * (log_ps - log_pt)).sum(dim=-1)   # [L]，reverse KL = KL(学生‖教师)
    loss = kl.mean()

    with torch.no_grad():
        mass = student_mass_coverage(s_logits, stats["lse"]).mean()
    return loss, {
        "kl": kl.mean().item(),
        "mass_coverage": mass.item(),
        "resp_logp": stats["resp_logp"],
    }

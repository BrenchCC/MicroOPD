"""naive top-K OPD（截断不归一化）—— M2/M3 对照 baseline。

方向说明：本损失是教师加权截断交叉熵（forward KL 方向，mass-covering），
即"不假思索的标准 KD 写法"；LSM（losses/lsm.py）则是 reverse KL
（KL(学生‖教师)，student-weighted）。两方向的对比本身是消融的一部分。

与 LSM 同样的 gather，但教师侧直接 exp(logprobs) 使用、不做支撑集内重归一化
（Σ p_teacher < 1，截断 KL；loss 发散/naive 不稳时先换 LSM 验证）。
学生侧仍在支撑集内归一化，保证目标是一个合法分布上的交叉熵。
K=1 时与 LSM 严格等价（教师 argmax CE，共用 losses.base.k1_argmax_ce），
该格消融两边共用，不重复训练。
"""
import torch.nn.functional as F

from losses.base import (
    gather_support_stats,
    k1_argmax_ce,
    prepare_sample,
    student_mass_coverage,
)


def naive_topk_loss(model, sample, topk, cfg, device):
    input_ids, sup_ids, t_logprobs = prepare_sample(sample, topk, device)
    stats = gather_support_stats(
        model, input_ids, sample.prompt_len, sup_ids, cfg["distill"]["chunk_size"]
    )
    if sup_ids.shape[1] == 1:  # K=1：与 LSM 严格等价（教师 argmax CE），共用同一实现
        return k1_argmax_ce(stats)
    s_logits = stats["support_logits"]

    pt = t_logprobs.exp()                       # 教师侧截断、不归一化
    log_ps = F.log_softmax(s_logits, dim=-1)    # 学生侧支撑集内归一化
    ce = -(pt * log_ps).sum(dim=-1)             # [L]，截断交叉熵
    loss = ce.mean()

    mass = student_mass_coverage(s_logits.detach(), stats["lse"].detach()).mean()
    return loss, {
        "kl": ce.mean().item(),  # 记录的是截断 CE，仅作曲线对照
        "mass_coverage": mass.item(),
        "resp_logp": stats["resp_logp"],
    }

"""二期 GRPO loss：组内 advantage + 重要性比 clip + KL-to-ref。

- advantage = (r − mean) / (std + eps)，组内（同 prompt 的 G 条）归一化；
  零方差组（全对/全错）advantage 全零无梯度，整组跳过（返回 None）。
- ratio = exp(logp_cur − logp_old)，clip(1±ε)，logp_old 来自 rollout 时 vLLM
  返回的采样 logprob（PPO 同构）。
- KL-to-ref 用 k3 估计子：exp(logp_ref − logp_cur) − (logp_ref − logp_cur) − 1 ≥ 0；
  ref 分布由同一模型 disable_adapter 免费获得（零额外显存），教师完全不加载。
- loss = −adv · ratio_clipped + β · KL；只给 exact-match 主 reward，防 reward hacking。
"""
import torch

from losses.base import gather_support_stats
from models.student import ref_context


def _response_logp(model, sample, chunk_size, device, grad=True):
    """学生对实际 response 的逐 token logp（支撑集即 response token 本身，K=1）。"""
    input_ids = torch.tensor([sample.full_ids], dtype=torch.long, device=device)
    sup = torch.tensor(sample.response_ids, dtype=torch.long, device=device).unsqueeze(-1)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        stats = gather_support_stats(
            model, input_ids, sample.prompt_len, sup, chunk_size
        )
        logp = stats["support_logits"].squeeze(-1) - stats["lse"]
    return logp


def grpo_group_loss(model, group, cfg, device):
    """对同 prompt 的一组 G 条轨迹算 GRPO loss。零方差组返回 None（跳过）。"""
    gcfg = cfg["grpo"]
    rewards = torch.tensor([s.reward for s in group], dtype=torch.float32)
    std = rewards.std(unbiased=False)
    if std < 1e-6:
        return None, {"skipped": True, "reward_mean": rewards.mean().item()}
    adv = (rewards - rewards.mean()) / (std + gcfg["adv_eps"])

    chunk = cfg["distill"]["chunk_size"]
    beta, eps = gcfg["beta"], gcfg["clip_eps"]
    losses, kl_means, ratio_means = [], [], []
    for sample, a in zip(group, adv):
        logp_cur = _response_logp(model, sample, chunk, device, grad=True)
        with ref_context(model):
            logp_ref = _response_logp(model, sample, chunk, device, grad=False)
        logp_old = torch.tensor(
            sample.old_logprobs, dtype=torch.float32, device=device
        )
        n = min(len(logp_cur), len(logp_old))  # 防御：长度不齐则截齐
        logp_cur, logp_ref, logp_old = logp_cur[:n], logp_ref[:n], logp_old[:n]

        ratio = torch.exp(logp_cur - logp_old)
        ratio_clip = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        # 标准 PPO surrogate：min 作用在乘过 advantage 的两项上，
        # 负 advantage 时自动取 max(ratio, clip) 那一支（悲观界）
        a_dev = a.to(device)
        pg = -torch.min(ratio * a_dev, ratio_clip * a_dev)
        kl = torch.exp(logp_ref - logp_cur) - (logp_ref - logp_cur) - 1.0  # k3 ≥ 0
        losses.append((pg + beta * kl).mean())
        kl_means.append(kl.mean().item())
        ratio_means.append(ratio.mean().item())

    loss = torch.stack(losses).mean()
    return loss, {
        "skipped": False,
        "reward_mean": rewards.mean().item(),
        "kl_to_ref": sum(kl_means) / len(kl_means),
        "ratio_mean": sum(ratio_means) / len(ratio_means),
    }

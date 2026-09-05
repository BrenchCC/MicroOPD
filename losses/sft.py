"""M2 Baseline 1：教师轨迹 SFT 的标准交叉熵。

response_mask 边界来自 tokenizer 实际编码（labels 前 prompt_len 位置 -100）。
SFT 是离线 baseline：数据由 rollout.teacher_score.generate_teacher_responses
一次性造出，训练期间教师不占卡，与 OPD 循环相互独立。
"""
import torch


def sft_loss(model, sample, topk, cfg, device):
    """签名与其他 loss 对齐；topk 恒为 None（SFT 无教师 cache）。"""
    input_ids = torch.tensor([sample.full_ids], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[0, : sample.prompt_len] = -100
    out = model(input_ids=input_ids, labels=labels)
    return out.loss, {"kl": 0.0, "mass_coverage": 0.0, "resp_logp": None}

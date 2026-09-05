"""损失基类与共享算子：学生 logits 分 chunk gather 到教师 top-K 支撑集。

显存纪律：不一次 materialize [1, T, V] 的 full-vocab logits，而是
backbone 前向一次拿 hidden states，再按 chunk_size（默认 64 token）逐段过
lm_head，每段只保留支撑集 gather 结果与 logsumexp（标量/position），防显存尖峰。

所有 OPD loss 的输入统一为：
  sample:  rollout.buffer.Trajectory（提供 full_ids / prompt_len）
  topk:    cache 读出的 dict（ids [L,K]、logprobs [L,K]，可选 mass/entropy）
返回 (loss_tensor, stats_dict)。
"""
import torch


def _unwrap_backbone_and_head(model):
    """兼容 PEFT 包装，取出 (backbone, lm_head)。"""
    causal = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        causal = model.base_model.model  # PeftModel -> LoraModel -> AutoModelForCausalLM
    return causal.model, causal.lm_head


def gather_support_stats(model, input_ids, prompt_len, support_ids, chunk_size=64):
    """学生对 response 各 position 的统计量（带梯度）。

    input_ids:   LongTensor [1, T]（prompt + response）
    prompt_len:  prompt 长度（response_mask 边界，来自 tokenizer 实际编码）
    support_ids: LongTensor [L, K]，教师 top-K 支撑集；L = T - prompt_len

    返回 dict：
      support_logits [L, K]  学生在支撑集 token 上的 logits
      lse            [L]     学生 full-vocab logits 的 logsumexp（算 Mass Coverage 用）
      resp_logp      [L]     学生对实际 response token 的 logp（KL 漂移监控 / GRPO 用）
    """
    device = input_ids.device
    backbone, lm_head = _unwrap_backbone_and_head(model)
    hidden = backbone(input_ids=input_ids).last_hidden_state  # [1, T, H]
    T = input_ids.shape[1]
    L = T - prompt_len
    if L <= 0:  # 防御：空 response
        z = torch.zeros(0, device=input_ids.device)
        return {
            "support_logits": torch.zeros(0, support_ids.shape[1], device=input_ids.device),
            "lse": z,
            "resp_logp": z,
        }
    pos0 = prompt_len - 1  # logits[i] 预测 token i+1；response 首 token 由 pos0 预测

    sup_chunks, lse_chunks, resp_chunks = [], [], []
    for s in range(pos0, T - 1, chunk_size):
        e = min(s + chunk_size, T - 1)
        logits = lm_head(hidden[:, s:e]).float()  # [1, c, V]，段内临时，用完即弃
        lse = torch.logsumexp(logits, dim=-1).squeeze(0)  # [c]
        sup = support_ids[s - pos0 : e - pos0].to(device)  # [c, K]
        sup_logit = logits.squeeze(0).gather(-1, sup)  # [c, K]
        next_tok = input_ids[0, s + 1 : e + 1]  # 实际 response token
        resp_logp = logits.squeeze(0).gather(-1, next_tok.unsqueeze(-1)).squeeze(-1) - lse
        sup_chunks.append(sup_logit)
        lse_chunks.append(lse)
        resp_chunks.append(resp_logp)
        del logits

    assert sum(c.shape[0] for c in sup_chunks) == L, "support 长度与 response 长度不一致"
    return {
        "support_logits": torch.cat(sup_chunks, dim=0),
        "lse": torch.cat(lse_chunks, dim=0),
        "resp_logp": torch.cat(resp_chunks, dim=0),
    }


def k1_argmax_ce(stats):
    """K=1 退化语义：教师 argmax 交叉熵 ≈ SFT（对全词表归一化）。

    严格在单元素支撑集内对师生两侧重归一化会得到 KL ≡ 0（无梯度），
    因此 K=1 时学生侧必须用 full-vocab 概率：CE = -(logit_argmax - logsumexp)。
    该格 naive ≡ LSM（实现上共用本函数，对应消融网格两边共用一个格子）。
    """
    ce = stats["lse"] - stats["support_logits"].squeeze(-1)  # [L] = -log p_s_full(argmax)
    loss = ce.mean()
    return loss, {
        "kl": ce.mean().item(),
        "mass_coverage": None,  # K=1 的 mass 由 agreement 评测给，训练日志不重复
        "resp_logp": stats["resp_logp"],
    }


def prepare_sample(sample, topk, device):
    """把 buffer 样本 + cache 条目转成张量，做长度对齐校验。"""
    input_ids = torch.tensor([sample.full_ids], dtype=torch.long, device=device)
    sup_ids = torch.tensor(topk["ids"], dtype=torch.long, device=device)
    t_logprobs = torch.tensor(topk["logprobs"], dtype=torch.float32, device=device)
    assert sup_ids.shape[0] == len(sample.response_ids), (
        f"cache 长度 {sup_ids.shape[0]} != response 长度 {len(sample.response_ids)}"
    )
    return input_ids, sup_ids, t_logprobs


def student_mass_coverage(support_logits, lse):
    """Mass Coverage：学生分布在教师 top-K 支撑集上的概率质量。
    = Σ_support p_student = exp(logsumexp(support_logits) - logsumexp(full_vocab))"""
    return torch.exp(torch.logsumexp(support_logits, dim=-1) - lse)

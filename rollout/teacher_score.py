"""教师打分：prompt_logprobs(prompt + student_response) 一次前向 → 每 position top-K。

绝不调用 generate 式解码（那是离线蒸馏）。做法：把 chat 化 prompt 与学生 response
的 token ids 拼接后整段作为 prompt 传入（TokensPrompt），max_tokens=1 仅为满足
vLLM 接口；从返回的 prompt_logprobs 中切出 response 区间的每 position top-K。

返回的 prompt_logprobs[i] 是「给定前 i 个 token 时第 i 个 token 的分布」，
因此 response token（full_ids[prompt_len:]）的分布在 prompt_logprobs[prompt_len:]。
"""
import numpy as np
from models.student import unload


def score_teacher(engine, full_ids_list, prompt_lens, K):
    """对一批 (prompt+response) 打分。
    返回 List[(topk_ids [L,K] int32, topk_logprobs [L,K] float32)]，L=len(response)。
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(
        prompt_logprobs=K,
        max_tokens=1,
        detokenize=False,
        temperature=0.0,
    )
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in full_ids_list]
    outputs = engine.generate(prompts, sp)

    results = []
    for out, plen in zip(outputs, prompt_lens):
        plp = out.prompt_logprobs
        ids_rows, lp_rows = [], []
        for pos in range(plen, len(plp)):
            dist = plp[pos]
            if dist is None:  # 防御：理论上只有 pos=0 为 None
                ids_rows.append([0] * K)
                lp_rows.append([-1e9] * K)
                continue
            # 字典含采样 token + top-K，按 logprob 降序取前 K
            items = sorted(dist.items(), key=lambda kv: kv[1].logprob, reverse=True)[:K]
            ids = [int(tid) for tid, _ in items]
            lps = [float(lp.logprob) for _, lp in items]
            while len(ids) < K:  # 极端防御：词表不可能小于 K，仅为形状稳定
                ids.append(0)
                lps.append(-1e9)
            ids_rows.append(ids)
            lp_rows.append(lps)
        results.append(
            (
                np.asarray(ids_rows, dtype=np.int32),
                np.asarray(lp_rows, dtype=np.float32),
            )
        )
    return results


def generate_teacher_responses(engine, prompt_ids_list, temperature, top_p, max_new_tokens, seed=None):
    """M2 Baseline 1（教师轨迹 SFT）专用：教师 generate 生成答案文本。
    注意：这是离线 SFT baseline 的造数步骤，与 OPD 打分（上方 prompt_logprobs）
    是两条独立路线，不得在 OPD 循环里调用本函数。
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(
        n=1, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, seed=seed
    )
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids_list]
    outputs = engine.generate(prompts, sp)
    return [
        {"response_ids": list(o.outputs[0].token_ids), "text": o.outputs[0].text}
        for o in outputs
    ]


def unload_scorer(engine):
    unload(engine)

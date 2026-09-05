"""MATH-500 一次性迁移测试（可选，仅 M6 执行后）。

答案解析用 math-verify 库，不自写解析器（文档纪律）。用途：GRPO 版在分布外
题目上仍涨 → 佐证「学到推理能力而非 GSM8K 格式特化」。
"""
from common import build_chat_prompt


def evaluate_math500(generator, tokenizer, cfg, limit=None):
    from datasets import load_dataset
    from math_verify import parse, verify

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if limit:
        ds = ds.select(range(limit))

    ecfg = cfg["eval"]
    prompt_ids = []
    for ex in ds:
        text = build_chat_prompt(
            tokenizer, ex["problem"], cfg["models"]["enable_thinking"]
        )
        prompt_ids.append(tokenizer(text, add_special_tokens=False)["input_ids"])
    outs = generator.generate(
        prompt_ids,
        temperature=ecfg["temperature"],
        top_p=ecfg["top_p"],
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        n=1,
        seed=cfg["experiment"]["seed"],
    )

    n_correct, lens = 0, []
    for ex, g in zip(ds, outs):
        text = g[0]["text"]
        lens.append(len(g[0]["response_ids"]))
        pred = parse(text)
        gold = parse(ex["solution"])
        try:
            if verify(gold, pred):
                n_correct += 1
        except Exception:
            pass  # 解析异常按错误计
    n = len(ds)
    return {
        "accuracy": n_correct / n if n else 0.0,
        "avg_len": sum(lens) / n if n else 0.0,
        "n": n,
    }

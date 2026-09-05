"""GSM8K 数据集加载与 accuracy/平均生成长度评测。

评测协议（config: eval）：贪婪解码（temperature=0）、0-shot、#### 提取规则。
官方 base 数字（1.7B≈75、8B≈90，4-shot CoT）只做区间参照；baseline 必须用
本管线自测（M0 验收项）。
"""
from common import build_chat_prompt
from evaluation.verifier import score_responses


def load_gsm8k(cfg, split):
    """返回 [{"question", "gold"(float), "gold_text"}]。"""
    from datasets import load_dataset

    ds = load_dataset(cfg["data"]["dataset"], cfg["data"]["subset"], split=split)
    from evaluation.verifier import extract_answer

    items = []
    for ex in ds:
        gold_text = ex["answer"]
        items.append(
            {
                "question": ex["question"],
                "gold": extract_answer(gold_text),
                "gold_text": gold_text,
            }
        )
    return items


def evaluate_accuracy(generator, tokenizer, items, cfg, limit=None):
    """generator: rollout.student_generate.StudentGenerator（已加载好权重）。
    返回 {"accuracy", "avg_len", "extract_fail_rate", "n"}。"""
    ecfg = cfg["eval"]
    if limit:
        items = items[:limit]
    prompt_ids = []
    for it in items:
        text = build_chat_prompt(
            tokenizer, it["question"], cfg["models"]["enable_thinking"]
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
    texts = [g[0]["text"] for g in outs]
    lens = [len(g[0]["response_ids"]) for g in outs]
    rewards, fail_rate = score_responses(texts, [it["gold_text"] for it in items])
    n = len(items)
    return {
        "accuracy": sum(rewards) / n if n else 0.0,
        "avg_len": sum(lens) / n if n else 0.0,
        "extract_fail_rate": fail_rate,
        "n": n,
    }

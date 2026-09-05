"""GSM8K 答案提取 + exact match（纯 Python，零显存）。

提取规则（M0 写死的评测协议）：取文本中最后一个 `####` 之后的第一个数值，
容忍 `$`、千分位逗号、小数、负号与简单分数 `a/b`。提取失败记 reward=0 并由
调用方统计失败率（>5% 先在 prompt 强约束 `####` 格式再训）。
"""
import re

_NUM_RE = re.compile(r"(-?\s*\$?\s*[\d,]*\.?\d+(?:\s*/\s*[\d,]*\.?\d+)?)")


def extract_answer(text):
    """返回 float 或 None（提取失败）。"""
    if not text:
        return None
    idx = text.rfind("####")
    if idx < 0:
        return None
    m = _NUM_RE.search(text[idx + 4 :])
    if not m:
        return None
    s = m.group(1).replace("$", "").replace(" ", "").replace(",", "")
    try:
        if "/" in s:
            num_s, den_s = s.split("/", 1)
            den = float(den_s)
            if den == 0:
                return None
            return float(num_s) / den
        return float(s)
    except ValueError:
        return None


def exact_match(text, gold):
    """gold 可以是数值或含 `####` 的数据集原始答案串。"""
    pred = extract_answer(text)
    if isinstance(gold, str):
        gold = extract_answer(gold)
    if pred is None or gold is None:
        return False
    return abs(pred - float(gold)) < 1e-6


def score_responses(texts, golds):
    """批量打分。返回 (rewards: List[float], extract_fail_rate: float)。"""
    rewards, n_fail = [], 0
    for t, g in zip(texts, golds):
        if extract_answer(t) is None:
            n_fail += 1
        rewards.append(1.0 if exact_match(t, g) else 0.0)
    return rewards, (n_fail / len(texts) if texts else 0.0)

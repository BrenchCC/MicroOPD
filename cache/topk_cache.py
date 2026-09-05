"""Top-K Cache：教师评分只发生一次，序列化后复用（学生训练期间教师不占卡）。

格式：每个样本一个 .npz（numpy 压缩），字段
  ids      [L, K] int32    教师 top-K token ids（支撑集）
  logprobs [L, K] float32  对应 logprob
派生量（读取时计算）：
  mass    [L]   top-K 概率质量 = Σ exp(logprob)，验收要求每个 position ∈ (0, 1]
  entropy [L]   支撑集内重归一化后的熵（注意：截断导致系统性偏低，门控时配合 mass 用）
"""
import os

import numpy as np


class TopKCache:
    def __init__(self, cache_dir):
        self.dir = cache_dir
        os.makedirs(self.dir, exist_ok=True)

    def path(self, sample_id):
        return os.path.join(self.dir, f"{sample_id}.npz")

    def write(self, sample_id, topk_ids, topk_logprobs):
        p = self.path(sample_id)
        np.savez_compressed(
            p,
            ids=np.asarray(topk_ids, dtype=np.int32),
            logprobs=np.asarray(topk_logprobs, dtype=np.float32),
        )
        return p

    def read(self, sample_id_or_path):
        p = (
            sample_id_or_path
            if str(sample_id_or_path).endswith(".npz")
            else self.path(sample_id_or_path)
        )
        z = np.load(p)
        ids, lps = z["ids"], z["logprobs"]
        return {
            "ids": ids,
            "logprobs": lps,
            "mass": self.mass(lps),
            "entropy": self.entropy(lps),
        }

    @staticmethod
    def mass(logprobs):
        # logprobs 含 padding -1e9，exp 后≈0，不影响 mass
        return np.exp(logprobs).sum(axis=-1)

    @staticmethod
    def entropy(logprobs):
        p = np.exp(logprobs)
        p_norm = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-12, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(p_norm > 0, p_norm * np.log(np.clip(p_norm, 1e-20, None)), 0.0)
        return -term.sum(axis=-1)

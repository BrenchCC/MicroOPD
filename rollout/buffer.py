"""Rollout buffer：{prompt_ids, response_ids, reward, iter, old_logprobs, cache_file}。

- 版本对齐：每条轨迹记录 iter（生成它的 iteration），训练只消费当前 iter 的数据
  （PPO 同构：每轮刷新数据 + N 步内更新，不做跨版本复用）。
- response_mask 不手算边界：由 prompt_len（tokenizer 实际编码结果）派生，
  前 prompt_len 位为 0，其余为 1。
- reward 字段 v2 起存在：一期 OPD 恒为 None，二期 GRPO 由 verifier 写入。
"""
import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class Trajectory:
    prompt_ids: List[int]
    response_ids: List[int]
    iter: int
    question: str = ""
    text: str = ""
    reward: Optional[float] = None
    old_logprobs: Optional[List[float]] = None  # GRPO：rollout 时策略的逐 token logp
    cache_file: Optional[str] = None            # Top-K Cache 落盘路径（一期）

    @property
    def full_ids(self):
        return self.prompt_ids + self.response_ids

    @property
    def prompt_len(self):
        return len(self.prompt_ids)

    @property
    def response_mask(self):
        return [0] * len(self.prompt_ids) + [1] * len(self.response_ids)


class RolloutBuffer:
    def __init__(self):
        self._items: List[Trajectory] = []

    def add(self, traj: Trajectory):
        self._items.append(traj)

    def clear(self):
        self._items = []

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def current(self, iteration):
        """只取指定 iteration 的轨迹（版本对齐检查）。"""
        return [t for t in self._items if t.iter == iteration]

    # ---- 可选落盘（断点排查用，非训练依赖） ----
    def save_jsonl(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for t in self._items:
                f.write(json.dumps(asdict(t)) + "\n")

    @classmethod
    def load_jsonl(cls, path):
        buf = cls()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                buf.add(Trajectory(**json.loads(line)))
        return buf

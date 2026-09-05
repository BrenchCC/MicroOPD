"""共享工具：配置加载（含 inherits 继承）、随机种子、显存统计、GSM8K chat prompt 构建。

模型下载走 HF_ENDPOINT 环境变量（本机须指向 hf-mirror），代码不覆盖该变量。
"""
import copy
import os
import random

import numpy as np
import yaml

# GSM8K 评测协议（M0 写死）：要求模型逐步推理并以 `#### <数值>` 结尾，
# verifier / 评测共用同一提取规则（evaluation/verifier.py）。
SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, "
    "then end your response with a final line exactly of the form '#### <number>' "
    "where <number> is the final numeric answer (no units, no commas)."
)


def load_config(path):
    """加载 yaml 配置；支持 `inherits: <相对路径>` 深合并父配置（子覆盖父）。"""
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent_path = cfg.pop("inherits", None)
    if parent_path:
        parent = load_config(os.path.join(os.path.dirname(path), parent_path))
        return _deep_merge(parent, cfg)
    return cfg


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_chat_prompt(tokenizer, question, enable_thinking=False):
    """统一走 chat template；Qwen3 模板支持 enable_thinking=False（non-thinking 模式）。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        # 模板不认 enable_thinking 时退回默认行为
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_prompt(tokenizer, question, enable_thinking=False):
    """chat 化 prompt 的 token ids。response 边界由它定位：mask 前段 0 后段 1，
    不手算边界（chat template 不一致是已知坑）。"""
    text = build_chat_prompt(tokenizer, question, enable_thinking=enable_thinking)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def vram_peak_gb():
    """当前 CUDA 峰值显存（GB）；无 CUDA 时返回 0。"""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except ImportError:
        pass
    return 0.0


def nvml_free_gb():
    """整卡空闲显存（GB，NVML 读数，含其他进程）。"""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        free = pynvml.nvmlDeviceGetMemoryInfo(h).free / 1e9
        pynvml.nvmlShutdown()
        return free
    except Exception:
        return -1.0


def ensure_free_gb(need_gb, timeout=60, tag=""):
    """引擎创建前确保整卡空闲 ≥ need_gb：gc + empty_cache + 等待子进程退出。"""
    import gc
    import time

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for _ in range(timeout):
        free = nvml_free_gb()
        if free < 0 or free >= need_gb:
            return free
        time.sleep(1)
    free = nvml_free_gb()
    print(f"[warn] {tag} 等待后空闲显存仅 {free:.2f} GB < 需求 {need_gb} GB")
    return free


def vram_reset_peak():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

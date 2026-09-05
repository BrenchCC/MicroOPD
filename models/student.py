"""学生模型（Qwen3-1.7B + LoRA r16）的加载、卸载与 vLLM 权重同步。

权重同步策略（config: vllm.student_weight_sync）：
- "lora"：model.save_pretrained(adapter_dir) 后，vLLM 学生引擎（enable_lora=True）
  用 LoRARequest 加载该 adapter 采样 —— 默认路径。
- "merge"：退路。merge_and_unload() 后整模落盘，vLLM 按普通模型加载。
  若 vLLM 对 Qwen3 LoRA 支持有问题，切这一项即可，同步逻辑只在本文件。

二期 ref 分布用同一权重 disable_adapter 获得（ref_context），零额外显存。
"""
import contextlib
import gc
import os

import torch


def load_tokenizer(cfg):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg["models"]["student"])


def load_student(cfg, adapter_path=None):
    """加载 HF+PEFT 学生（训练侧）。adapter_path 非空则从已有 LoRA adapter 继续训练。"""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        cfg["models"]["student"], torch_dtype=torch.bfloat16
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        lc = cfg["lora"]
        lora_cfg = LoraConfig(
            r=lc["r"],
            lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"],
            target_modules=list(lc["target_modules"]),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.config.use_cache = False  # 与 grad checkpointing 共存；训练不需要 KV cache
    return model.to(cfg["experiment"].get("device", "cuda"))


def build_optimizer(model, lr):
    """8-bit AdamW，只优化 LoRA 参数（其余 requires_grad=False 已被过滤）。"""
    import bitsandbytes as bnb

    params = [p for p in model.parameters() if p.requires_grad]
    return bnb.optim.PagedAdamW8bit(params, lr=lr)


def save_adapter(model, path):
    """只落盘 LoRA adapter（数十 MB），供 vLLM LoRARequest 或断点续训使用。"""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    return path


def sync_student_to_vllm(model, tokenizer, workdir, mode="lora"):
    """把当前学生权重同步成 vLLM 可用形态。返回 {"type": ..., "path": ...}。"""
    if mode == "lora":
        path = save_adapter(model, os.path.join(workdir, "adapter"))
        return {"type": "lora", "path": path}
    elif mode == "merge":
        # 退路：merge 后整模落盘。merge_and_unload 会改模型本体，
        # 因此只在 vLLM LoRA 路线不可用时使用（训练态从该目录重新加载继续）。
        path = os.path.join(workdir, "merged")
        os.makedirs(path, exist_ok=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(path)
        tokenizer.save_pretrained(path)
        del merged
        gc.collect()
        return {"type": "merged", "path": path}
    else:
        raise ValueError(f"unknown student_weight_sync mode: {mode}")


@contextlib.contextmanager
def ref_context(model):
    """disable LoRA adapter 取 ref 分布（二期 KL-to-ref，零额外副本）。"""
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield model
    else:
        yield model


def unload(*objs):
    """显式卸载并清卡。两个模型分时复用，峰值取 max 不取 sum。

    vLLM V1 引擎的计算核心在独立子进程，仅 del 父进程引用不会立即释放
    显存（已实测：残留 11GB 导致训练 OOM），必须显式 engine_core.shutdown()
    并等子进程退出、显存真正回落。
    """
    import time

    for o in objs:
        engine_core = getattr(getattr(o, "llm_engine", None), "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            try:
                engine_core.shutdown()
            except Exception:
                pass
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 等 vLLM 子进程退出、显存回落（最多 60s）
        try:
            import pynvml

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            for _ in range(60):
                free_gb = pynvml.nvmlDeviceGetMemoryInfo(h).free / 1e9
                if free_gb > 14.0:  # 16GB 卡，桌面占 ~0.1GB，引擎退出后应回到 ~15GB 空闲
                    break
                time.sleep(1)
            pynvml.nvmlShutdown()
        except Exception:
            time.sleep(3)


def student_to_cpu(model):
    """vLLM 引擎驻留期间把 HF 学生挪到 CPU，避免双份权重挤占 16GB。"""
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def student_to_device(model, device="cuda"):
    model.to(device)

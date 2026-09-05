"""教师模型（Qwen3-8B, FP8, vLLM）封装：加载/卸载 + prompt_logprobs 打分入口。

铁律（架构原则 1）：教师打分是 prompt_logprobs(prompt + student_response) 的一次前向，
绝不允许 teacher.generate()。本模块只暴露打分引擎的加载，不提供任何生成接口。
二期（GRPO）完全不 import 本模块。
"""
from models.student import unload  # 复用同一卸载纪律


def load_teacher_engine(cfg):
    """FP8 vLLM 引擎。打分阶段 OOM 时把 gpu_memory_utilization 降到 0.55–0.6（config）。"""
    from vllm import LLM

    from common import ensure_free_gb

    # 引擎子进程按整卡空闲显存做预算，父进程残留会导致 "No available memory"，
    # 创建前确保足够空闲（预算 = util × 15.57GB + 少量余量）
    need = cfg["vllm"]["gpu_memory_utilization_teacher"] * 15.57 + 0.5
    free = ensure_free_gb(need, tag="teacher-engine")
    print(f"[teacher] 引擎创建前空闲显存 {free:.2f} GB（需求 ≥{need:.1f}）")

    return LLM(
        model=cfg["models"]["teacher"],
        quantization=cfg["models"].get("teacher_quantization", "fp8"),
        gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization_teacher"],
        max_model_len=cfg["vllm"]["max_model_len"],
        max_logprobs=cfg["distill"]["K"],  # vLLM 默认上限 20，K=64 需显式放宽
        dtype="auto",
        enforce_eager=cfg["vllm"].get(
            "teacher_enforce_eager", cfg["vllm"].get("enforce_eager", False)
        ),
    )


def unload_teacher(engine):
    unload(engine)

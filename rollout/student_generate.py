"""学生 vLLM 批量采样（一期/二期复用；二期把 n=G 调大）。

引擎按 iteration 创建/销毁（与教师分时复用单卡），绝不逐条切换模型。
权重通过 models.student.sync_student_to_vllm 的产物加载：
- lora  模式：基座 + LoRARequest 指向刚 save 的 adapter；
- merge 模式：直接加载 merge 后的整模目录。
"""
from models.student import unload


class StudentGenerator:
    def __init__(self, cfg, sync):
        from vllm import LLM

        from common import ensure_free_gb

        vcfg = cfg["vllm"]
        # 同教师：创建前确保整卡空闲（父进程 HF 学生应已 student_to_cpu）
        need = vcfg["gpu_memory_utilization_student"] * 15.57 + 0.5
        free = ensure_free_gb(need, tag="student-engine")
        print(f"[student] 引擎创建前空闲显存 {free:.2f} GB（需求 ≥{need:.1f}）")

        common = dict(
            gpu_memory_utilization=vcfg["gpu_memory_utilization_student"],
            max_model_len=vcfg["max_model_len"],
            dtype="bfloat16",
            enforce_eager=vcfg.get("enforce_eager", False),
        )
        self.lora_request = None
        if sync["type"] == "lora":
            self.llm = LLM(
                model=cfg["models"]["student"],
                enable_lora=True,
                max_lora_rank=cfg["lora"]["r"],
                **common,
            )
            from vllm.lora.request import LoRARequest

            self.lora_request = LoRARequest("student", 1, sync["path"])
        elif sync["type"] == "merge":
            self.llm = LLM(model=sync["path"], **common)
        else:
            raise ValueError(f"unknown sync type: {sync['type']}")

    def generate(self, prompt_ids_list, temperature, top_p, max_new_tokens, n=1,
                 seed=None, want_logprobs=False):
        """对每条 prompt 采 n 条（二期 n=G≥8）。
        返回 List[List[dict]]：外层按 prompt，内层 n 条样本，字段
        {response_ids, text, logprobs}。logprobs 为逐 token 采样 logprob
        （GRPO 的旧策略 logp；一期不需要可关）。
        """
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        sp = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
            logprobs=1 if want_logprobs else None,
        )
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids_list]
        outputs = self.llm.generate(prompts, sp, lora_request=self.lora_request)

        results = []
        for out in outputs:
            group = []
            for o in out.outputs:
                lps = None
                if want_logprobs and o.logprobs is not None:
                    # vLLM logprobs 每项是 {token_id: Logprob}，取实际采样 token 的 logprob
                    lps = [
                        d[tid].logprob
                        for tid, d in zip(o.token_ids, o.logprobs)
                    ]
                group.append(
                    {
                        "response_ids": list(o.token_ids),
                        "text": o.text,
                        "logprobs": lps,
                    }
                )
            results.append(group)
        return results

    def close(self):
        unload(self.llm)
        self.llm = None

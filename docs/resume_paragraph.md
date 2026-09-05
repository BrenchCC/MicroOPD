# 简历段落（数字全部来自真实跑出的结果，原始记录见 runs/）

## 标准版

> **MicroOPD：单卡数学推理在线策略蒸馏系统** | PyTorch / vLLM / Transformers / LoRA · RTX 4060 Ti 16GB
>
> 在 16GB 单卡上完整跑通 Qwen3-8B（FP8 教师）→ Qwen3-1.7B（LoRA 学生）的数学推理在线策略蒸馏（OPD）全流程：学生采样、教师逐 token 分布打分、Top-K Cache 复用、漂移门控更新，整卡峰值 13.5GB。GSM8K 全量评测（1319 题）实证**在线蒸馏一致优于教师轨迹 SFT**（OPD 各变体 80.6–82.4% vs SFT 79.9%，Base 78.5%）；复现 LSM 支撑集内重归一化 reverse KL，训练 **loss 方差较 naive top-K 降低 67%**，Mass Coverage 0.9999 验证蒸馏假设；完成 K∈{1,8,32,64} 消融与 GRPO 规模复验（4× 长训 82.3% vs 锚点 82.0%，+0.4pt 噪声带内、倾向正面），MATH-500 分布外迁移为负结果亦如实记录。曾对照论文定义自查发现 KL 方向实现错误，修正后全链路重跑并更新全部结论。

# 简历项目经历（数字全部来自真实跑出的结果，原始记录见 runs/）

## 详细版（项目经历栏）

**MicroOPD：单卡数学推理在线策略蒸馏系统** | 个人项目
*PyTorch · vLLM · HuggingFace Transformers/PEFT · bitsandbytes · LoRA · FP8 量化*

- 在 RTX 4060 Ti **16GB 单卡**上搭建 Qwen3-8B（FP8 教师）→ Qwen3-1.7B（LoRA 学生）的**在线策略蒸馏（OPD）**流水线：学生 vLLM 采样 → 教师 `prompt_logprobs` 一次前向条件打分（非重新生成）→ Top-K Cache 序列化复用 → LoRA 更新；双模型分时复用显存，整卡峰值 **13.5GB**。
- 复现 **LSM（Local Support Matching，arXiv 2603.25562）**：教师 top-K 支撑集内师生双侧重归一化 reverse KL（KL(学生‖教师)），修复 top-K 截断偏差；训练 **loss 方差较 naive top-K 降低 67%**，Top1 match 0.945、Mass Coverage 0.9999 验证蒸馏假设成立。
- GSM8K 全量评测（1319 题）：Base 78.5% → SFT 79.9% → OPD 各变体 80.6–82.4%，**在线蒸馏一致优于教师轨迹 SFT**；K∈{1,8,32,64} 消融实证退化链 **SFT ≈ K=1 OPD ⊂ top-K OPD ⊂ full-vocab OPD**，LSM 的稳定性优势（方差 −67%）与论文主张一致，acc 优势在小规模设置下未复现（如实记录）。
- 训练循环按 **PPO 同构**设计（每轮刷新 rollout + N 步内更新 + KL 漂移 0.05 门控早停），二期仅换 loss 与信号源即接入 **GRPO**（组内 advantage + ratio clip + KL-to-ref，ref 用 `disable_adapter` 零显存获得）：短训相对锚点零提升（策略未动，命题未检验），**4× 规模长训后 82.3% vs 锚点 82.0%（+0.4pt，噪声带内、倾向正面）**，训练健康（KL-to-ref 累计 ~0.11，无 nan 无尖峰）。
- 工程排障：定位并修正生成预算与教师长度分布不自洽的协议缺陷（256→512，修正前全体方法 acc 被低估约 12pt）；**对照论文定义自查发现 LSM 的 KL 方向实现反了（forward 被当成 reverse）+ 熵门控特征死代码，修正后全链路重跑并如实更新全部结论**；发现 K=1 支撑集双归一化使 KL 恒零的退化陷阱；解决 vLLM V1 引擎子进程显存残留（11GB）问题。

## 一句话版（用于摘要/自我介绍）

> 在 16GB 单卡上复现并消融在线策略蒸馏（OPD/LSM），GSM8K 上实证在线蒸馏一致优于教师轨迹 SFT（80.6–82.4% vs 79.9%，Base 78.5%），LSM 重归一化 reverse KL 将训练 loss 方差降低 67%；全流程含 GRPO 对照、K 值消融与分布外负结果的诚实记录。

## 面试可追问点（均有实测数据支撑）

- OPD / 离线蒸馏 / RL 的信号来源区别；教师打分为什么用 prompt_logprobs 不用 generate
- reverse vs forward KL（KL(学生‖教师) vs KL(教师‖学生)，mode-seeking vs mass-covering）；top-K 截断问题与 LSM 修复；K=1 退化链
- "你怎么发现 KL 方向写反了" → 对照论文 Eq. (8) 逐项核对实现，修正后重跑全链路，旧结论（gate 最优、GRPO 突破）不成立则如实改写（README"修正记录"）
- "你的 buffer 还算 on-policy 吗" → PPO 同构 + KL 漂移锯齿图（runs/m7/figs/kl_drift.png）
- "LSM 比 naive 好在哪里" → 稳定性（loss 方差 −67%）而非 acc；小 K 略胜、K=64 反负的消融表
- 诚实面：MATH-500 分布外迁移为负结果（0.486 base → 0.468 OPD → 0.466 GRPO）；GRPO 长训 +0.4pt 在噪声带内，只敢说"倾向正面"

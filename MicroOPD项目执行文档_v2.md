# MicroOPD：单卡数学推理在线策略蒸馏系统 — 执行文档 v2

> 硬件约束：RTX 4060 16GB 单卡
> 定位：简历个人项目，标准是"稳定跑通 + 讲得清 + 有对比"，不追求方法创新
> 执行方式：分两期。一期 M0–M5 为纯 OPD（范围冻结），二期 M6 为混合阶段（OPD→GRPO 短训），M7 包装交付。每个里程碑有明确验收标准，前一个不过不进下一个

---

## 1. 项目定义

**一句话**：在 16G 单卡上实现 Qwen3-8B（instruct）→ Qwen3-1.7B（instruct）的数学推理在线策略蒸馏（OPD），核心组件为迭代式 rollout buffer + Top-K Cache + LSM 损失，评测指标含 Mass Coverage；二期在 OPD 收敛后接入 GSM8K 可验证奖励做短训 GRPO，验证"verifier 信号突破蒸馏天花板"的混合路线。

**最终范围（冻结，不再扩张）**：

| 做 | 不做 |
|---|---|
| 迭代式 rollout buffer 流水线 | reward model（GSM8K exact-match 即可，无需模型） |
| Top-K Cache（教师评分序列化复用） | 多教师、多数据集大规模训练 |
| LSM（支撑集内重归一化 KL） | 全参数训练作为必达目标（LoRA 为主） |
| K ∈ {1, 8, 32, 64} 消融 | K=256 |
| 熵感知 KL（flag 化，可选） | 组内正确率二阶门控 |
| Mass Coverage / Teacher Agreement 评测 | PPO critic / value network（GRPO 无 critic） |
| **M6：OPD→GRPO 顺序课程（仅 GSM8K，短训）** | 稠密+稀疏逐 token 混合 loss（调参成本高，留 future work） |

---

## 2. 系统架构

**一期（M0–M5）：纯 OPD 循环**

```
Iteration t:
  ┌─────────────────────────────────────────────────┐
  │ ① Student Rollout (vLLM, BF16)                  │
  │    GSM8K prompt 池 → 采样 G 条轨迹               │
  │         ↓                                        │
  │ ② Rollout Buffer (CPU)                          │
  │    {prompt, response_ids, response_mask, iter}   │
  │         ↓                                        │
  │ ③ Teacher Scoring (vLLM, FP8)                   │
  │    prompt_logprobs(prompt + student_response)    │
  │    一次前向，非重新生成                            │
  │         ↓                                        │
  │ ④ Top-K Cache (磁盘/内存)                        │
  │    {position, token_ids[K], logprobs[K], mass}   │
  │         ↓                                        │
  │ ⑤ Student Update (LoRA, 8-bit AdamW, N 步)       │
  │    LSM loss (+ 可选 entropy gate)                │
  │    监控 KL 漂移，超阈值提前进入下一轮              │
  └─────────────────────────────────────────────────┘
Iteration t+1: 用更新后的学生重新 rollout
```

**二期（M6）：混合阶段 —— 同一骨架，换掉 ③④⑤ 的信号源**

```
Iteration t（GRPO phase）:
  ┌─────────────────────────────────────────────────┐
  │ ① Student Rollout (vLLM)                        │
  │    每个 prompt 采样 G≥8 条（复用一期代码）         │
  │         ↓                                        │
  │ ② Verifier (纯 Python，零显存)                   │
  │    exact match：提取 #### 后的数值答案比对         │
  │         ↓                                        │
  │ ③ Buffer 写入 reward 字段                        │
  │    {prompt, response_ids, mask, reward, iter}    │
  │         ↓                                        │
  │ ④ Student Update (GRPO loss)                     │
  │    组内 advantage = (r − mean) / std             │
  │    重要性比 + clip + KL-to-ref 正则              │
  │    ref 分布 = 同一模型 disable LoRA adapter       │
  └─────────────────────────────────────────────────┘
关键事实：训练循环结构不变（PPO 同构），教师不再参与；
GRPO 不引入任何新常驻模型，verifier 是字符串匹配。
```

**两个必须守住的原则**（一期）：
1. 教师打分是 `prompt_logprobs(prompt + student_response)` 的**一次前向**，条件是学生的历史 token，绝不允许 `teacher.generate()`
2. 教师评分只发生一次、进 cache 复用；学生训练期间教师不占卡

**二期新增原则**：
3. GRPO 阶段教师模型**完全不加载**——二期的论点就是"教师信号到顶后，换 verifier 信号"，加载教师会破坏这个对照的干净性

---

## 3. 目录结构

```
MicroOPD/
├── configs/                  # 全部实验配置（yaml），长度/K 值/开关不许写死在代码里
│   ├── base.yaml
│   ├── ablation/
│   └── grpo.yaml             # 二期：G、β(KL系数)、clip、训练步数
├── models/
│   ├── student.py            # 加载/卸载学生，LoRA 注入；提供 disable_adapter 取 ref 分布
│   └── teacher.py            # 加载/卸载教师，FP8 推理封装（二期不调用）
├── rollout/
│   ├── student_generate.py   # vLLM 批量采样（两期复用，二期 G 调大）
│   ├── teacher_score.py      # prompt_logprobs 打分 → Top-K Cache
│   └── buffer.py             # rollout buffer：版本对齐、迭代管理；v2 起含 reward 字段
├── cache/
│   └── topk_cache.py         # 序列化格式、读写、mass/entropy 派生量
├── losses/
│   ├── base.py               # 损失基类：gather 学生 logits、mask
│   ├── naive_topk.py         # 截断不归一化（baseline）
│   ├── lsm.py                # 支撑集内重归一化 KL（reverse 方向，师生两侧归一化）
│   ├── entropy_gate.py       # 可选：reverse/forward KL 逐 token 插值
│   └── grpo.py               # 二期：组内 advantage + ratio clip + KL-to-ref
├── trainer/
│   └── distill_trainer.py    # N 步更新、KL 漂移监控、早停触发；二期复用同一 trainer 换 loss
├── evaluation/
│   ├── gsm8k.py
│   ├── math500.py            # 可选：仅 M6 执行后做一次性迁移测试（见 M7）
│   ├── agreement.py          # Top1 match、Mass Coverage（Top5/KL 已裁：与训练 loss 及 MC 信号重复）
│   └── verifier.py           # 二期：GSM8K 答案提取 + exact match（纯 Python）
├── scripts/
│   ├── run_iteration.py      # 单轮完整迭代（冒烟测试入口）
│   └── run_experiment.py     # 按 config 跑完整实验（含 --phase2 入口）
├── README.md
└── docs/
    └── architecture.png
```

---

## 4. 里程碑计划

### M0 — 环境与单模型冒烟

**任务**
- 模型选型写死：教师 Qwen3-8B（instruct）、学生 Qwen3-1.7B（instruct），同系列同词表；`enable_thinking=False`（non-thinking 模式与 max_new_tokens=256 预算自洽，thinking 长链会被截断、verifier 提取必败）
- vLLM 安装，验证 FP8 量化支持（Qwen3-8B 加载后报告显存占用）
- 学生 Qwen3-1.7B + LoRA (rank 16) 加载，一次 forward + backward 不 OOM
- GSM8K 加载与 prompt 模板（chat template 统一）
- 评测协议写死并进 config：解码参数（温度/top-p）、shot 数、答案提取规则（`####` 后数值）；官方 base 数字（1.7B≈75、8B≈90，4-shot CoT）只做区间参照，不做 baseline

**验收**
- [ ] 教师 FP8 推理显存 ≤ 10GB
- [ ] 学生 LoRA 训练态显存 ≤ 5GB
- [ ] 峰值合计（分时复用下取 max，不是 sum）≤ 15GB，留 1G 余量
- [ ] 用自己的管线测出学生 baseline GSM8K acc 并记录（主结果表第一行 Base 的出处；若与官方区间差距异常，先查协议再查模型）

**卡点预案**：FP8 不支持则退 4bit (bitsandbytes/AWQ)，在 README 记录精度差异。

---

### M1 — 最小蒸馏闭环（全项目最高风险点）

**任务**
- `student_generate.py`：10 条 prompt → 学生采样（max_new_tokens=256）
- `teacher_score.py`：prompt_logprobs 取 top-64，写 Top-K Cache
- `lsm.py` 初版：cache → gather 学生 logits → 支撑集内双归一化 → KL → backward
- `run_iteration.py` 串起单轮

**验收（必须全部通过才算闭环成立）**
- [ ] 10 条样本：生成 → 打分 → cache → loss → optimizer.step() 完整跑通
- [ ] loss 为有限值且量级合理（0.1–5）
- [ ] cache 中每个 position 的 mass ∈ (0, 1]
- [ ] 教师打分结果与"教师重新生成"的 token 分布一致（抽查 3 条，验证没走错 generate 路线）

**卡点预案**
- vLLM prompt_logprobs 只返回 prompt 部分 → 把 response 拼进 prompt 字段传入
- 学生 forward 的 full-vocab logits 显存尖峰 → 分 chunk 算（每 64 token 一段）

---

### M2 — Baseline 三件套 + 评测

**任务**
- Baseline 1：教师轨迹 SFT（教师 generate 生成答案，学生 SFT）
- Baseline 2：naive top-K OPD（截断不归一化）
- Baseline 3：LSM OPD（K=64）
- `agreement.py`：Top1 match、Mass Coverage 两项（评测 KL 与训练 loss 同量，Top5 与 Mass Coverage 高相关，均裁）
- `gsm8k.py`：accuracy + 平均生成长度

**验收**
- [ ] 三条曲线可同图比较：training step vs loss / Mass Coverage
- [ ] 每个 baseline 产出一行结果：GSM8K acc、平均长度、峰值显存、耗时
- [ ] 预期不强求：Base < SFT ≈ OPD 即可；若 OPD ≯ SFT，用 token agreement 上升 + 学生容量瓶颈做分析，写进 README

---

### M3 — K 值消融（核心产出图）

**任务**
- LSM 跑 K ∈ {1, 8, 32, 64}，naive 只跑 K ∈ {8, 32, 64}，共 7 组短训（固定 step 数）
- K=1 时支撑集单元化、重归一化是恒等变换，naive ≡ LSM，该格两边共用——重合本身就是退化链的直接证据，不重复烧一次训练
- K=1 理论上退化为教师 argmax 交叉熵 ≈ SFT——与 M2 的 SFT baseline 互证并写分析

**验收**
- [ ] 产出核心 figure：K vs accuracy/loss，naive 与 LSM 两条线
- [ ] 预期现象：K 越小 naive 越不稳定，LSM 越稳；若现象不出现，检查归一化实现而非强行编结论
- [ ] 能口述退化链条：SFT ⊂ top-K OPD ⊂ full-vocab OPD

---

### M4 — 熵感知门控（可选，有时间才启动）

**启动条件**：M3 验收通过且有余力。否则跳过，README 写明负结果/未做原因。

**任务**
- `entropy_gate.py`：从 cache 读 top-K 熵 + mass，sigmoid 门控插值 reverse/forward KL
- 缓解熵偏置：门控特征同时包含 top-K 概率质量（1 − mass 大 → 分布平）
- 对照实验：LSM vs LSM+gate，一行结果

**验收**
- [ ] 输出每类 token 的平均熵与权重分布（验证门控确实在低熵 token 上走 reverse KL）
- [ ] 结果好 → 写进主表；打平/变差 → 写负结果分析（允许，价值不减）

---

### M5 — Staleness 监控与打磨

**任务**
- `distill_trainer.py` 加 KL 漂移监控：每步估算当前学生分布 vs cache 时分布的 KL，超阈值（如 0.05）提前触发下一轮 rollout
- （可选）消融：N ∈ {4, 16} 对比——漂移曲线本身已支撑"N 小才配叫 on-policy"，此消融只是配图；不跑则 README 用漂移曲线 + PPO 类比论述

**验收**
- [ ] 能画出 KL 漂移曲线并解释
- [ ] 面试问题"你这还算 on-policy 吗"有标准答案（PPO 同构 + 漂移监控）——此答案同时是 M6 的架构依据

---

### M6 — 混合阶段：OPD → GRPO 顺序课程（二期）

**启动条件**：M3 验收通过（M4/M5 不阻塞）。若一期时间耗尽，M6 整体降级为 README future work，设计文档保留。

**前置论点（写进 README）**：纯蒸馏的目标函数以教师分布为最优解，天花板是教师；GRPO 阶段的实验目的是验证"接入 verifier 信号后，学生能突破纯 OPD 的收敛平台"。预期是超过 pure OPD 若干点，**不预期**超过 8B 教师。

**任务**
- `verifier.py`：GSM8K 答案提取（`####` 后数值）+ exact match；答案提取失败记 reward=0 并统计失败率（>5% 则先在 prompt 里强约束输出格式）
- `buffer.py`：trajectory 记录加 `reward` 字段
- `grpo.py`：组内 advantage（G≥8）、重要性比 clip、KL-to-ref（β 起步 0.01–0.05）；ref 分布用 `disable_adapter` 从当前模型免费获得，不另存副本
- `configs/grpo.yaml`：从 M3 最优 LSM checkpoint 热启动；lr 比一期低一个量级（1e-6–1e-7）；短训固定 step 数
- 对照：同一 checkpoint，继续 pure OPD 等步数 vs 切 GRPO 等步数

**验收**
- [ ] verifier 单测通过（含提取失败 case），训练集上 reward 分布非退化（不全 0 不全 1）
- [ ] GRPO 曲线无 nan、无梯度尖峰失控（grad norm 监控 + clip）
- [ ] 产出一行对比：pure OPD 续训 vs OPD→GRPO 的 GSM8K acc；打平也算有效结果，写分析（样本量/步数/容量）
- [ ] 面试问题"怎么超过教师"有实证答案：天花板曲线 + verifier 信号接入后的变化

**卡点预案**
- 组内全对/全错 → advantage 全零无梯度：跳过该组；大面积出现则调温度或换难度分桶采样
- reward hacking（输出膨胀刷格式分）：只给 exact-match 主 reward，格式 reward 权重 ≤0.1 或不设
- 训练崩（KL 爆）：β 调大一档，lr 减半，仍崩则回退到 M3 checkpoint 记录负结果

---

### M7 — 包装交付

- [ ] 主结果表（Base / SFT / naive / LSM / LSM+gate / OPD→GRPO 六行；GRPO 行未跑出则标注"设计见 M6，未执行"）
- [ ] K 消融图、Mass Coverage 曲线、显存吞吐表；二期加"蒸馏天花板 vs 混合突破"对比图
- [ ] （可选，仅当 M6 执行）MATH-500 一次性迁移测试：答案解析用 math-verify 库，不自写解析器；GRPO 版在分布外题目上仍涨 → 佐证"学到推理能力而非 GSM8K 格式特化"；未执行则 README future work 说明
- [ ] README：架构图、复现命令、负结果分析、limitation 列表；引用 OPD/LSM 出处文献
- [ ] 简历段落（数字全部来自真实跑出的结果，未跑出不写数）
- [ ] 面试 Q&A 自测通过（见第 6 节）

---

## 5. 显存预算表

| 组件 | 配置 | 估算 |
|---|---|---|
| 教师 Qwen3-8B | FP8，vLLM | ~9.5 GB（仅一期打分阶段驻留；二期不加载） |
| 学生 Qwen3-1.7B | BF16 + LoRA r16 | ~3.5 GB |
| 优化器 | 8-bit AdamW（LoRA 参数） | <0.5 GB |
| 激活 | grad checkpointing, seq 256, bs 1 | ~1 GB |
| 学生 forward logits | 分 chunk（64 token/段） | 每段 <0.1 GB |
| 二期 ref 模型 | 同一权重 disable LoRA adapter | 0（无额外副本） |
| 二期 verifier | 纯 Python 字符串匹配 | 0（不占卡） |

**纪律**：两个模型分时复用，峰值取 max 不取 sum；任何阶段超过 15GB 就降 seq 或 chunk，不硬扛。二期常驻显存低于一期（无教师），余量可让给更大的 G。

---

## 6. 面试 Q&A（自测清单）

1. **OPD / 离线蒸馏 / RL 三者的信号来源区别？**
   教师逐 token 分布 vs 教师文本 vs verifier 标量；OPD 不需要标准答案也不需要环境。
2. **教师打分为什么用 prompt_logprobs 而不是 generate？**
   要在学生历史 token 条件下算 P(x_t|x_{<t})；generate 得到的是教师自己的轨迹，那就退化成离线蒸馏了。且生成是串行解码，打分是 teacher forcing 一次并行前向，成本差约一个序列长度的倍数。
3. **reverse KL vs forward KL？**
   mode-seeking vs mode-covering；推理关键 token 要低熵对齐，所以主用 reverse。
4. **为什么教师学生必须同词表？**（KL 需要在相同支撑集上逐 token 比较；这是"逐点比高度"的前提，本质是保持两个分布同相）
5. **top-K 截断有什么问题，LSM 怎么修？**
   截断后概率和 <1 破坏 KL 等价性；支撑集内师生各自重归一化再算 KL。LSM 即文献中的 Local Support Matching，本项目是复现+消融验证，不声称方法创新。
6. **top-K 熵为什么偏低？你怎么处理？**
   长尾被截断；门控时同时用 top-K mass 作为辅助特征。（主动提 limitation 加分）
7. **你的 rollout buffer 还算 on-policy 吗？**
   与 PPO 同构：每个 iteration 刷新数据 + N 步内更新 + KL 漂移监控，超阈值提前刷新。**推论**：正因为训练循环是 PPO 同构的，M6 接入 GRPO 只换了 loss 和信号源，没动架构。
8. **K=1 时你的方法退化成什么？**
   教师 argmax 交叉熵 ≈ SFT；整条链 SFT ⊂ top-K OPD ⊂ full-vocab OPD。
9. **Mass Coverage 是什么，为什么重要？**
   学生分布在教师 top-K 支撑集上的概率质量；它直接检验 LSM 的前提假设是否成立，是结果也是诊断。verl 等框架的 OPD 实现内置同款统计（student_mass）。
10. **LoRA 会不会让蒸馏失效？**
    词表和输出头不变，分布层面蒸馏信号完整；容量受限是 fair 的 trade-off，16G 约束下的合理选择。附带好处：ref 分布可用 disable_adapter 免费获得（M6 的 KL 正则零成本）。
11. **蒸馏能超过教师吗？你的混合阶段怎么回答这个问题？**
    不能——KL 目标的最优解就是教师分布，且容量差使复制都不完全。超越需要教师之外的信号源：verifier 对错、环境反馈、搜索计算量。M6 用 GSM8K exact-match（零显存 verifier）接入标量 reward，实证验证"pure OPD 平台期 → GRPO 再抬一截"的突破机制；这就是工业界 off-policy SFT → OPD → 短 RL 三段式配方的最小复现。

---

## 7. 坑清单（遇到先查这里）

| 症状 | 原因 | 处理 |
|---|---|---|
| OOM 在教师打分阶段 | vLLM KV cache 预留过多 | 降 `gpu_memory_utilization` 到 0.55–0.6 |
| loss=nan 或发散 | naive top-K 未归一化 / 学习率过大 | 先换 LSM 验证；lr 降到 1e-5 量级 |
| 教师打分结果像重新生成 | 误用 generate | 改 prompt_logprobs，M1 验收第 4 条防的就是这个 |
| prompt/response 边界错位 | chat template 不一致 | response_mask 用 tokenizer 实际编码结果定位，别手算 |
| Mass Coverage 不升反降 | K 太小或 LSM 实现错误 | 检查重归一化是否对师生**两侧**都做了；KL 方向是否写成 forward（LSM 是 reverse） |
| 训练极慢 | 每步都在加载/卸载模型 | 检查是否退化成逐条切换；正确做法是分阶段批量 |
| 学生输出复读/坍缩 | reverse KL 在低熵 token 上过度 mode-seeking | 降 lr；若开了 entropy gate，检查门控方向是否写反；仍塌则试 stop-gradient 版 top-K 目标（一行 detach） |
| **GRPO 无梯度/曲线平直** | 组内全对或全错，advantage 全零 | 跳过零方差组；大面积出现则调采样温度或按难度分桶 |
| **GRPO 输出长度膨胀** | reward hacking / 格式分漏洞 | 只保留 exact-match 主 reward；必要时加长度惩罚 |
| **GRPO 训练崩（KL 爆、grad 尖峰）** | β 太小或 lr 相对 RL 过大 | grad clip；β 调大一档；lr 降到 1e-6 量级；回退 M3 checkpoint |
| **reward 恒 0** | 答案提取失败（格式不符） | 统计提取失败率；>5% 先在 prompt 强约束 `####` 格式再训 |

---

## 8. 简历段落模板（数字待真实结果填入）

> **MicroOPD：单卡数学推理在线策略蒸馏系统** — 基于 PyTorch / Transformers / vLLM 在 RTX 4060 16GB 上搭建在线策略蒸馏流水线，Qwen3-8B 教师对 Qwen3-1.7B 学生的数学推理轨迹做逐 token 分布监督。设计迭代式 Rollout–Scoring–Update 机制与 Top-K Cache（教师评分序列化复用，避免重复前向），复现支撑集内重归一化（LSM）修复 top-K 截断的 KL 偏差，并以 Mass Coverage 等指标验证蒸馏假设。在 GSM8K 上与教师轨迹 SFT、naive top-K OPD 对比（若执行 M6，另以 MATH-500 做一次性迁移测试），K ∈ {1, 8, 32, 64} 消融表明 ______（填真实结论，如：top-32 下 LSM 较 naive 提升 X 个百分点，峰值显存 XX GB）。训练循环按 PPO 同构设计，二期接入 GSM8K 可验证奖励热启动 GRPO 短训，验证 verifier 信号对蒸馏天花板的突破：______（填真实结论，如：pure OPD 平台期 XX% 基础上再提升 X 个百分点）。

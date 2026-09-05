# AGENTS.md — MicroOPD

> 本文件面向 AI 编码代理。读者默认对本项目一无所知。

## 1. 项目概览

**MicroOPD：单卡数学推理在线策略蒸馏（OPD）系统** —— 一个简历级个人项目，标准是"稳定跑通 + 讲得清 + 有对比"，不追求方法创新。

- **硬件约束**：RTX 4060 16GB 单卡。一切设计围绕此约束（分时复用、LoRA、FP8/8-bit、分 chunk）。
- **核心目标**：在 16G 单卡上实现 Qwen3-8B（instruct，教师）→ Qwen3-1.7B（instruct，学生）的数学推理在线策略蒸馏，数据集 GSM8K。
- **核心组件**：迭代式 rollout buffer + Top-K Cache（教师评分序列化复用）+ LSM 损失（Local Support Matching，支撑集内重归一化 KL），评测含 Mass Coverage。
- **分两期执行**：
  - 一期 M0–M5：纯 OPD（范围冻结）。里程碑顺序：M0 环境冒烟 → M1 最小蒸馏闭环（全项目最高风险点）→ M2 Baseline 三件套+评测 → M3 K 值消融（核心产出）→ M4 熵感知门控（可选）→ M5 Staleness 监控。
  - 二期 M6：OPD→GRPO 顺序课程（GSM8K exact-match 可验证奖励，短训），验证"verifier 信号突破蒸馏天花板"。M7 包装交付。
- **执行纪律**：每个里程碑有明确验收标准，前一个不过不进下一个。范围冻结，不再扩张（详见执行文档第 1 节的"做/不做"表）。

**唯一的设计与执行文档**：`MicroOPD项目执行文档_v2.md`（项目根目录）。所有架构决策、里程碑验收标准、显存预算、坑清单、面试 Q&A 都在其中。实现前先读它。

## 2. 当前状态

代码已落地并**全部里程碑（M0–M7）已在 GPU 上真实跑完**，数字见 README.md 主结果表与 `runs/` 原始记录。关键实测结论（2026-08 reverse KL 方向修正后重跑）：Base 0.785 < SFT 0.799 < OPD 各变体 0.806–0.824（naive K=64 最高），LSM 的明确优势在训练稳定性（loss 方差 −67%）而非 acc，entropy gate 无增益，OPD→GRPO 短训 0.820 与 pure OPD 续训 0.820 打平（策略几乎未动，命题未被检验），4× 规模长训复验 0.8234（+0.4pt vs 锚点，噪声带内、倾向正面，见 configs/grpo_long.yaml）。曾发现并修正 LSM 的 KL 方向实现与论文定义相反（forward 误作 reverse）及 entropy gate 的 mass 死代码，修正后全链路重跑、结论如实改写（见 README"修正记录"）。改代码前先读执行文档与 README 的"修正记录"（含 512 协议修正、K=1 恒零陷阱、KL 方向、vLLM 子进程显存释放等已踩过的坑），不要自行扩张范围。

## 3. 技术栈

- **语言/框架**：Python、PyTorch、HuggingFace Transformers、vLLM（推理与教师打分）。
- **模型**：教师 Qwen3-8B（instruct，FP8 量化，vLLM）；学生 Qwen3-1.7B（instruct，BF16 + LoRA rank 16）。同系列同词表是硬性前提（KL 需要在相同支撑集上逐 token 比较）。
- **生成模式**：`enable_thinking=False`；max_new_tokens=512（初版 256 与教师实际长度分布不自洽，M2 实证后修正，见 README"关键工程发现"）。
- **训练**：LoRA 为主（全参数训练不是必达目标）；8-bit AdamW；grad checkpointing；学生 full-vocab logits 分 chunk 算（每 64 token 一段）。
- **数据**：GSM8K；答案提取规则为 `####` 后数值，chat template 统一。可选 MATH-500 一次性迁移测试（答案解析用 math-verify 库，不自写解析器）。

## 4. 计划目录结构（来自执行文档第 3 节）

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
│   ├── math500.py            # 可选：仅 M6 执行后做一次性迁移测试
│   ├── agreement.py          # Top1 match、Mass Coverage
│   └── verifier.py           # 二期：GSM8K 答案提取 + exact match（纯 Python）
├── scripts/
│   ├── run_iteration.py      # 单轮完整迭代（冒烟测试入口）
│   └── run_experiment.py     # 按 config 跑完整实验（含 --phase2 入口）
├── README.md
└── docs/
    └── architecture.png
```

## 5. 构建与运行命令

依赖清单为根目录 `requirements.txt`（vllm==0.8.5.post1、transformers>=4.51,<5、peft、accelerate、datasets、bitsandbytes、pyyaml、matplotlib、pytest、math-verify、numpy）。安装：

```bash
pip install -r requirements.txt
# 本机 huggingface.co 不可达且 hf-mirror 大文件 CDN 超时：模型走 ModelScope 下载到 pretrained/（configs 已指向本地路径）
# 运行时数据集/模型全部本地缓存：export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

- M0 冒烟（显存验收 + Base 全量评测）：`python scripts/smoke_m0.py`
- 冒烟测试（M1 入口）：`python scripts/run_iteration.py --config configs/base.yaml --num-samples 10`
- 完整实验：`python scripts/run_experiment.py --config configs/<name>.yaml`（`--smoke` 缩减规模自检）
- M3 消融：`python scripts/run_experiment.py --config configs/ablation/{k1,k8,k32,k64,naive_k8,naive_k32,naive_k64}.yaml`
- SFT baseline：`python scripts/run_experiment.py --config configs/sft.yaml`
- 二期 GRPO：`python scripts/run_experiment.py --config configs/grpo.yaml --phase2`（`init_adapter` 已指向 M3 最优 adapter runs/lsm_k8/adapters/final）
- 全量评测：`python scripts/final_eval_all.py`；出图：`python scripts/make_figures.py`；MATH-500：`python scripts/eval_math500.py`；教师 acc 一次性评测（离线参照，不进训练闭环）：`python scripts/eval_teacher.py`
- verifier 单测：`python -m pytest tests/ -q`

代码已落地（目录结构同第 3 节；另加根目录 `common.py` 承载配置继承加载/种子/prompt 模板等共享工具，以及 `losses/sft.py` 承载 M2 教师轨迹 SFT baseline 的 CE 损失）。配置支持 `inherits:` 相对路径继承（消融配置即基于 `base.yaml` 覆盖）。实验产物统一在 `runs/<name>/`：config 副本、metrics.jsonl、eval.jsonl、adapters/、cache/、figs/、final_eval.json。

## 6. 必须守住的架构原则

1. **教师打分是一次前向**：`prompt_logprobs(prompt + student_response)`，条件是学生的历史 token，**绝不允许 `teacher.generate()`**（那会使其退化为离线蒸馏）。若 vLLM 的 prompt_logprobs 只返回 prompt 部分，把 response 拼进 prompt 字段传入。
2. **教师评分只发生一次**，进 Top-K Cache 复用；学生训练期间教师不占卡（两模型分时复用）。
3. **GRPO 阶段教师模型完全不加载** —— 二期论点就是"教师信号到顶后换 verifier 信号"，加载教师会破坏对照的干净性。
4. **训练循环 PPO 同构**：每个 iteration 刷新 rollout 数据 + N 步内更新 + KL 漂移监控（阈值如 0.05），超阈值提前进入下一轮。这是"还算 on-policy"的标准答案，也是 M6 能只换 loss 和信号源就接入 GRPO 的架构依据。

## 7. 显存纪律（16GB 硬约束）

| 组件 | 配置 | 估算 |
|---|---|---|
| 教师 Qwen3-8B | FP8，vLLM | ~9.5 GB（仅一期打分阶段驻留） |
| 学生 Qwen3-1.7B | BF16 + LoRA r16 | ~3.5 GB |
| 优化器 | 8-bit AdamW（LoRA 参数） | <0.5 GB |
| 激活 | grad checkpointing, seq 256, bs 1 | ~1 GB |

- 两个模型分时复用，**峰值取 max 不取 sum**；任何阶段超过 15GB 就降 seq 或 chunk，不硬扛。
- 教师打分 OOM：降 vLLM `gpu_memory_utilization` 到 0.55–0.6。
- FP8 不支持则退 4bit（bitsandbytes/AWQ），并在 README 记录精度差异。
- 二期 ref 分布用同一权重 `disable_adapter` 获得，不另存副本（零额外显存）。

## 8. 测试与验收策略

项目以**里程碑验收清单**代替传统测试套件（详见执行文档第 4 节），关键验收点：

- **M0**：教师 FP8 推理显存 ≤ 10GB；学生 LoRA 训练态 ≤ 5GB；峰值合计 ≤ 15GB；用自己的管线测出学生 baseline GSM8K acc 并记录。
- **M1（最高风险点）**：10 条样本完整跑通闭环；loss 为有限值且量级 0.1–5；cache 每个 position 的 mass ∈ (0, 1]；抽查 3 条验证打分结果与教师重新生成的 token 分布一致（防走错 generate 路线）。
- **M2**：三条 baseline（教师轨迹 SFT / naive top-K OPD / LSM OPD K=64）曲线同图可比较；每个 baseline 产出一行结果（GSM8K acc、平均长度、峰值显存、耗时）。
- **M3**：K ∈ {1, 8, 32, 64} 消融共 7 组（K=1 时 naive ≡ LSM，该格两边共用，不重复训练）；产出核心 figure：K vs accuracy/loss 两条线。
- **M6**：verifier 单测（含提取失败 case）；reward 分布非退化；GRPO 曲线无 nan、无梯度尖峰失控（grad norm 监控 + clip）。
- 评测指标：GSM8K accuracy + 平均生成长度；Top1 match；Mass Coverage（学生分布在教师 top-K 支撑集上的概率质量）。评测 KL 与 Top5 已裁（与训练 loss 及 MC 信号重复）。

## 9. 代码与配置约定

- **一切超参进 yaml**：长度、K 值、开关不许写死在代码里；评测协议（解码温度/top-p、shot 数、答案提取规则）也写死进 config。
- **loss 约定**：LSM 是 reverse KL = KL(学生‖教师)（期望在学生分布下取，student-weighted / mode-seeking，与 LSM 论文 arXiv 2603.25562 Eq. (8) 及 MiniLLM 约定一致），师生**两侧**都在支撑集内重归一化；naive top-K 是教师加权截断 CE（forward 方向），截断不归一化（作为对照 baseline）。K=1 退化为教师 argmax 交叉熵 ≈ SFT，退化链：SFT ⊂ top-K OPD ⊂ full-vocab OPD。
- **response_mask 用 tokenizer 实际编码结果定位**，别手算 prompt/response 边界（chat template 不一致是已知坑）。
- **GRPO（二期）**：G≥8；组内 advantage = (r − mean) / std；重要性比 + clip + KL-to-ref（β 起步 0.01–0.05）；lr 比一期低一个量级（1e-6–1e-7）；只给 exact-match 主 reward，格式 reward 权重 ≤0.1 或不设（防 reward hacking）；组内全对/全错（零方差）的组跳过。
- **模型切换纪律**：分阶段批量加载/卸载，绝不逐条切换（训练极慢的根因）。
- 本项目是 OPD/LSM 文献的复现+消融验证，**不声称方法创新**；README 需引用出处文献。

## 10. 常见问题速查（详见执行文档第 7 节坑清单）

- loss=nan 或发散 → 先换 LSM 验证；lr 降到 1e-5 量级。
- 教师打分结果像重新生成 → 误用了 generate，改 prompt_logprobs。
- Mass Coverage 不升反降 → 检查重归一化是否师生两侧都做了；KL 方向是否误写成 KL(教师‖学生)（那是 forward；LSM 的 reverse 是 KL(学生‖教师)，期望在学生分布下取）。
- 学生输出复读/坍缩 → 降 lr；开了 entropy gate 则检查门控方向；仍塌则试 stop-gradient 版 top-K 目标（一行 detach）。
- GRPO 无梯度 → 组内零方差；调采样温度或按难度分桶。
- reward 恒 0 → 答案提取失败；失败率 >5% 先在 prompt 强约束 `####` 格式。

## 11. 结果报告纪律

- 简历与 README 中的数字**全部来自真实跑出的结果**，未跑出不写数；负结果/打平结果允许写，价值不减。
- 官方 base 数字（1.7B≈75、8B≈90，4-shot CoT）只做区间参照，不做 baseline；baseline 必须用自己的管线测出。
- 若预期现象不出现（如 K 越小 naive 越不稳定），检查实现而非强行编结论。

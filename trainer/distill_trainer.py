"""DistillTrainer：N 步内层更新 + KL 漂移监控 + 早停；二期复用同一 trainer 换 loss。

PPO 同构（M5 验收 / 面试题 7 的标准答案）：
- 每个 iteration 刷新 rollout 数据后，在 buffer 上做 N=inner_steps 步更新；
- 每步估算「当前学生 vs cache/rollout 时学生」的 KL 漂移：
  用 rollout 时的 response token logp（old）与当前 logp（cur），
  drift ≈ E[logp_old − logp_cur] ≈ KL(pi_old ‖ pi_cur)（样本来自旧策略）；
- drift 超阈值（config: kl_drift_threshold，默认 0.05）提前触发下一轮 rollout。

日志：runs/<name>/metrics.jsonl，每 optimizer step 一行
（iter/step/loss/mass_coverage/kl_drift/grad_norm/vram_peak_gb + loss 自带统计）。
"""
import json
import os
import random

import torch

from common import vram_peak_gb, vram_reset_peak
from losses.base import gather_support_stats
from losses.entropy_gate import entropy_gate_loss
from losses.grpo import grpo_group_loss
from losses.lsm import lsm_loss
from losses.naive_topk import naive_topk_loss
from losses.sft import sft_loss

LOSS_FNS = {
    "lsm": lsm_loss,
    "naive_topk": naive_topk_loss,
    "entropy_gate": entropy_gate_loss,
    "sft": sft_loss,
}


class DistillTrainer:
    def __init__(self, model, optimizer, cfg, run_dir):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = cfg["experiment"].get("device", "cuda")
        self.run_dir = run_dir
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self.global_step = 0
        self.loss_fn = LOSS_FNS[cfg["distill"]["loss"]]
        self.trainable = [p for p in model.parameters() if p.requires_grad]

    # ---------- 一期 OPD / SFT ----------
    def train_on_buffer(self, buffer, cache, iteration):
        """在当前 iteration 的 buffer 上做 N 步更新。返回是否因 KL 漂移提前停止。"""
        dcfg = self.cfg["distill"]
        samples = buffer.current(iteration)
        assert samples, f"iteration {iteration} 的 buffer 为空（版本对齐失败）"
        is_opd = self.cfg["distill"]["loss"] != "sft"

        # rollout 时学生的 response logp（no_grad），作为 KL 漂移基准
        old_logps = {}
        if is_opd:
            old_logps = self._snapshot_student_logps(samples)

        early_stopped = False
        order = list(range(len(samples)))
        for step in range(dcfg["inner_steps"]):
            random.shuffle(order)
            self.optimizer.zero_grad()
            accum_stats, n_accum = [], 0  # (样本索引, stats)，乱序对齐用
            for i in order:
                s = samples[i]
                topk = cache.read(s.cache_file) if s.cache_file else None
                loss, stats = self.loss_fn(self.model, s, topk, self.cfg, self.device)
                (loss / dcfg["grad_accum"]).backward()
                accum_stats.append((i, stats))
                n_accum += 1
                if n_accum % dcfg["grad_accum"] == 0 or i == order[-1]:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.trainable, dcfg["max_grad_norm"]
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    drift = self._estimate_drift(accum_stats, old_logps)
                    rec = {
                        "iter": iteration,
                        "step": self.global_step,
                        "loss": float(loss.item()),
                        "kl_drift": drift,
                        "grad_norm": float(grad_norm),
                        "vram_peak_gb": vram_peak_gb(),
                        "mass_coverage": _mean_stat(accum_stats, "mass_coverage"),
                        "kl": _mean_stat(accum_stats, "kl"),
                    }
                    for k in ("gate_w_mean", "entropy_norm_mean"):
                        v = _mean_stat(accum_stats, k)
                        if v is not None:
                            rec[k] = v
                    self.log_metric(rec)
                    accum_stats = []
                    vram_reset_peak()
                    if drift is not None and drift > dcfg["kl_drift_threshold"]:
                        early_stopped = True
                        break
            if early_stopped:
                break
        return early_stopped

    def _snapshot_student_logps(self, samples):
        """no_grad 快照 rollout 时学生的 response 逐 token logp。"""
        old = {}
        chunk = self.cfg["distill"]["chunk_size"]
        with torch.no_grad():
            for idx, s in enumerate(samples):
                input_ids = torch.tensor(
                    [s.full_ids], dtype=torch.long, device=self.device
                )
                sup = torch.tensor(
                    s.response_ids, dtype=torch.long, device=self.device
                ).unsqueeze(-1)
                stats = gather_support_stats(
                    self.model, input_ids, s.prompt_len, sup, chunk
                )
                old[idx] = (stats["support_logits"].squeeze(-1) - stats["lse"]).cpu()
        return old

    @staticmethod
    def _estimate_drift(accum_stats, old_logps):
        """drift ≈ E[logp_old − logp_cur]（KL(pi_old‖pi_cur) 的采样估计）。
        accum_stats 为 (样本索引, stats)，与 old_logps 的键直接对齐。"""
        if not old_logps:
            return None
        vals = []
        for idx, st in accum_stats:
            if st.get("resp_logp") is None or idx not in old_logps:
                continue
            cur = st["resp_logp"].detach().cpu()
            old = old_logps[idx]
            n = min(len(cur), len(old))
            vals.append((old[:n] - cur[:n]).mean().item())
        return sum(vals) / len(vals) if vals else None

    # ---------- 二期 GRPO ----------
    def train_grpo(self, groups, iteration):
        """groups: List[List[Trajectory]]（按 prompt 分组，组内 G 条，含 reward/old_logprobs）。
        零方差组由 grpo_group_loss 返回 None 跳过。"""
        dcfg = self.cfg["distill"]
        n_skipped = 0
        order = list(range(len(groups)))
        for step in range(dcfg["inner_steps"]):
            random.shuffle(order)
            self.optimizer.zero_grad()
            batch_stats, n_accum = [], 0
            for gi in order:
                loss, stats = grpo_group_loss(self.model, groups[gi], self.cfg, self.device)
                if loss is None:
                    n_skipped += 1
                    continue
                (loss / dcfg["grad_accum"]).backward()
                batch_stats.append(stats)
                n_accum += 1
                if n_accum % dcfg["grad_accum"] == 0 or gi == order[-1]:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.trainable, dcfg["max_grad_norm"]
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    kl_ref = _mean_stat(batch_stats, "kl_to_ref")
                    self.log_metric(
                        {
                            "iter": iteration,
                            "step": self.global_step,
                            "loss": float(loss.item()),
                            "kl_drift": kl_ref,  # GRPO 侧漂移量即 KL-to-ref
                            "grad_norm": float(grad_norm),
                            "vram_peak_gb": vram_peak_gb(),
                            "reward_mean": _mean_stat(batch_stats, "reward_mean"),
                            "ratio_mean": _mean_stat(batch_stats, "ratio_mean"),
                            "groups_skipped": n_skipped,
                        }
                    )
                    batch_stats = []
                    vram_reset_peak()
                    if kl_ref is not None and kl_ref > dcfg["kl_drift_threshold"] * 10:
                        # KL 爆：β 调大一档、lr 减半（坑清单），此处先早停保现场
                        return True
        return False

    # ---------- 日志与出图 ----------
    def log_metric(self, rec):
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def plot_curves(self):
        """metrics.jsonl → runs/<name>/figs/（loss / mass_coverage / kl_drift）。"""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not os.path.exists(self.metrics_path):
            return
        rows = [json.loads(l) for l in open(self.metrics_path, encoding="utf-8")]
        figs = os.path.join(self.run_dir, "figs")
        os.makedirs(figs, exist_ok=True)
        for key in ("loss", "mass_coverage", "kl_drift", "grad_norm", "reward_mean"):
            xs = [r["step"] for r in rows if r.get(key) is not None]
            ys = [r[key] for r in rows if r.get(key) is not None]
            if not xs:
                continue
            plt.figure()
            plt.plot(xs, ys)
            plt.xlabel("optimizer step")
            plt.ylabel(key)
            plt.title(f"{self.cfg['experiment']['name']} — {key}")
            plt.savefig(os.path.join(figs, f"{key}.png"), dpi=120, bbox_inches="tight")
            plt.close()


def _mean_stat(stats_list, key):
    """stats_list 元素为 stats dict 或 (样本索引, stats) 元组。"""
    vals = []
    for item in stats_list:
        s = item[1] if isinstance(item, tuple) else item
        if s.get(key) is not None:
            vals.append(s[key])
    return sum(vals) / len(vals) if vals else None

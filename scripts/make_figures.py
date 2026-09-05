"""M7 包装：主结果表 + 全部图（K 消融、Mass Coverage、KL 漂移、天花板对比、显存吞吐）。

数据源（全部为真实跑出的产物）：
- runs/<name>/full_eval.json      全量 1319 题 acc/len/fail
- runs/<name>/metrics.jsonl       loss/mass/drift/grad_norm/vram
- runs/<name>/final_eval.json     agreement（top1_match/mass_coverage）
- runs/m0/m0_results.json         Base 行 + 显存验收
- runs/m7/math500.json            MATH-500 迁移（可选）

产物：runs/m7/{main_table.md,main_table.json} 与 runs/m7/figs/*.png
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 注册中文字体（系统 Noto Sans CJK），否则中文标题变方块
for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RUNS = "runs"
OUT = os.path.join(RUNS, "m7")
FIGS = os.path.join(OUT, "figs")

# 主表行（顺序即展示顺序）；label=展示名，run=runs/ 下的目录
MAIN_ROWS = [
    ("Base（Qwen3-1.7B 未训练）", None),
    ("SFT（教师轨迹）", "sft"),
    ("naive top-K OPD（K=64）", "naive_k64"),
    ("LSM OPD（K=64）", "base"),
    ("LSM + entropy gate（K=64）", "lsm_gate_k64"),
    ("OPD→GRPO（M6）", "grpo"),
    ("pure OPD 续训（M6 对照）", "opd_continue"),
]


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def metrics(run):
    return [json.loads(l) for l in open(os.path.join(RUNS, run, "metrics.jsonl"), encoding="utf-8")]


def run_wall_hours(run):
    """目录创建到 final_eval.json 的墙钟时间（小时，近似）。"""
    d = os.path.join(RUNS, run)
    try:
        t0 = os.path.getmtime(os.path.join(d, "config.yaml"))
        t1 = os.path.getmtime(os.path.join(d, "final_eval.json"))
        return (t1 - t0) / 3600
    except OSError:
        return None


def build_main_table():
    m0 = load_json(os.path.join(RUNS, "m0", "m0_results.json"))
    rows = []
    base_eval = m0["baseline_eval"]
    rows.append({
        "label": MAIN_ROWS[0][0], "acc": base_eval["accuracy"],
        "avg_len": base_eval["avg_len"], "fail": base_eval["extract_fail_rate"],
        "vram_peak": None, "hours": None,
    })
    for label, run in MAIN_ROWS[1:]:
        fe = load_json(os.path.join(RUNS, run, "full_eval.json"))
        ms = metrics(run)
        rows.append({
            "label": label,
            "acc": fe["accuracy"], "avg_len": fe["avg_len"],
            "fail": fe["extract_fail_rate"],
            "vram_peak": max(r["vram_peak_gb"] for r in ms),
            "hours": run_wall_hours(run),
        })
    lines = ["| 方法 | GSM8K acc（全量 1319） | 平均长度 | 提取失败率 | 训练峰值显存 GB | 墙钟 h |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        v = f"{r['vram_peak']:.2f}" if r["vram_peak"] else "—"
        h = f"{r['hours']:.1f}" if r["hours"] else "—"
        lines.append(f"| {r['label']} | {r['acc']:.4f} | {r['avg_len']:.0f} | "
                     f"{r['fail']:.3f} | {v} | {h} |")
    table = "\n".join(lines)
    with open(os.path.join(OUT, "main_table.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    with open(os.path.join(OUT, "main_table.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(table)
    return rows


def fig_k_ablation():
    """核心产出图：K vs acc（全量）与 K vs loss std，naive/LSM 两条线；K=1 两边共用。"""
    ks = [1, 8, 32, 64]
    grid = {("lsm", 1): "lsm_k1", ("lsm", 8): "lsm_k8", ("lsm", 32): "lsm_k32",
            ("lsm", 64): "base", ("naive", 8): "naive_k8", ("naive", 32): "naive_k32",
            ("naive", 64): "naive_k64"}
    acc, lstd, top1 = {"lsm": {}, "naive": {}}, {"lsm": {}, "naive": {}}, {"lsm": {}, "naive": {}}
    for (kind, k), run in grid.items():
        acc[kind][k] = load_json(os.path.join(RUNS, run, "full_eval.json"))["accuracy"]
        import statistics as st
        lstd[kind][k] = st.stdev([r["loss"] for r in metrics(run)])
        fe = load_json(os.path.join(RUNS, run, "final_eval.json"))
        if "agreement" in fe:
            top1[kind][k] = fe["agreement"]["top1_match"]
    acc["naive"][1] = acc["lsm"][1]   # K=1 共用格
    lstd["naive"][1] = lstd["lsm"][1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for kind, marker, label in [("lsm", "o", "LSM"), ("naive", "s", "naive top-K")]:
        xs = [k for k in ks if k in acc[kind]]
        axes[0].plot(xs, [acc[kind][k] for k in xs], marker + "-", label=label)
        axes[1].plot(xs, [lstd[kind][k] for k in xs], marker + "-", label=label)
    axes[0].axhline(0.7847, ls="--", c="gray", lw=1, label="Base（未训练）")
    axes[0].set_xscale("log", base=2); axes[0].set_xticks(ks); axes[0].set_xticklabels(ks)
    axes[0].set_xlabel("K"); axes[0].set_ylabel("GSM8K acc（全量）"); axes[0].legend()
    axes[1].set_xscale("log", base=2); axes[1].set_xticks(ks); axes[1].set_xticklabels(ks)
    axes[1].set_xlabel("K"); axes[1].set_ylabel("训练 loss 标准差"); axes[1].legend()
    fig.suptitle("K 值消融：naive vs LSM（K=1 两边共用同一格）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "k_ablation.png"), dpi=150)
    plt.close(fig)
    print("fig: k_ablation.png")


def fig_kl_drift():
    """M5：KL 漂移锯齿曲线（base run）——迭代内漂移上升、rollout 刷新归零，PPO 同构。"""
    ms = metrics("base")
    xs = [r["step"] for r in ms if r.get("kl_drift") is not None]
    ys = [r["kl_drift"] for r in ms if r.get("kl_drift") is not None]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, lw=0.8)
    ax.axhline(0.05, ls="--", c="r", lw=1, label="阈值 0.05（触发提前 rollout）")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("KL(pi_old ‖ pi_cur) 采样估计")
    ax.set_title("KL 漂移监控（LSM K=64）：迭代内上升，刷新 rollout 后归零")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "kl_drift.png"), dpi=150)
    plt.close(fig)
    print("fig: kl_drift.png")


def fig_mass_coverage():
    """Mass Coverage 训练曲线（各 K 的 LSM 组）。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    for run, label in [("lsm_k8", "K=8"), ("lsm_k32", "K=32"), ("base", "K=64")]:
        ms = metrics(run)
        pts = [(r["step"], r["mass_coverage"]) for r in ms if r.get("mass_coverage") is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=0.8, label=label)
    ax.set_xlabel("optimizer step"); ax.set_ylabel("Mass Coverage")
    ax.set_title("Mass Coverage：学生在教师 top-K 支撑集上的概率质量")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "mass_coverage.png"), dpi=150)
    plt.close(fig)
    print("fig: mass_coverage.png")


def fig_ceiling():
    """蒸馏天花板 vs 混合突破：主表关键行 + MATH-500（若已跑出）。"""
    rows = load_json(os.path.join(OUT, "main_table.json"))
    pick = ["Base（Qwen3-1.7B 未训练）", "SFT（教师轨迹）", "LSM OPD（K=64）",
            "LSM + entropy gate（K=64）", "pure OPD 续训（M6 对照）", "OPD→GRPO（M6）"]
    sel = [r for r in rows if r["label"] in pick]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [r["label"].replace("（", "\n（") for r in sel]
    vals = [r["acc"] for r in sel]
    colors = ["#888", "#5aa", "#5aa", "#5aa", "#a85", "#a5a"]
    ax.bar(labels, vals, color=colors[:len(sel)])
    ax.axhline(vals[0], ls="--", c="gray", lw=1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(min(vals) - 0.02, max(vals) + 0.02)
    ax.set_ylabel("GSM8K acc（全量 1319）")
    ax.set_title("蒸馏天花板 vs verifier 信号（OPD→GRPO）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ceiling.png"), dpi=150)
    plt.close(fig)
    print("fig: ceiling.png")

    m5 = os.path.join(RUNS, "m7", "math500.json")
    if os.path.exists(m5):
        res = load_json(m5)
        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(res.keys())
        ax.bar(names, [res[n]["accuracy"] for n in names], color="#5aa")
        for i, n in enumerate(names):
            ax.text(i, res[n]["accuracy"] + 0.003, f"{res[n]['accuracy']:.3f}", ha="center")
        ax.set_ylabel("MATH-500 acc")
        ax.set_title("MATH-500 一次性迁移测试（分布外）")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "math500.png"), dpi=150)
        plt.close(fig)
        print("fig: math500.png")


def main():
    os.makedirs(FIGS, exist_ok=True)
    build_main_table()
    fig_k_ablation()
    fig_kl_drift()
    fig_mass_coverage()
    fig_ceiling()
    print(f"产物见 {OUT}")


if __name__ == "__main__":
    main()

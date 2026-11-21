"""Stage 2 mechanistic trajectory analysis, checkpoint ablations, and figure generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, spearmanr
import torch

# Support `python -m induction.src.analysis`, `python -m src.analysis`, and direct script execution
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data import (
        DATA_SEED,
        N_PROBE,
        N_VAL,
        PROBE_HALF_LEN,
        PROBE_SEED,
        SEQ_LEN,
        VAL_SEED,
        VOCAB_SIZE,
        ValidationSet,
        make_base_transition_matrix,
        make_prefix_matching_probe,
        make_validation_set,
    )
    from src.metrics import (
        compute_prefix_matching_score,
        compute_prev_token_score,
        evaluate_validation_loss,
    )
    from src.model import InductionTransformer, TransformerConfig
    from src.overnight_sweep import SweepJob, build_sweep_matrix
else:
    from .data import (
        DATA_SEED,
        N_PROBE,
        N_VAL,
        PROBE_HALF_LEN,
        PROBE_SEED,
        SEQ_LEN,
        VAL_SEED,
        VOCAB_SIZE,
        ValidationSet,
        make_base_transition_matrix,
        make_prefix_matching_probe,
        make_validation_set,
    )
    from .metrics import (
        compute_prefix_matching_score,
        compute_prev_token_score,
        evaluate_validation_loss,
    )
    from .model import InductionTransformer, TransformerConfig
    from .overnight_sweep import SweepJob, build_sweep_matrix

# Publication-clean matplotlib defaults
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "legend.fontsize": 9,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "1L-attn": "#64748B",
    "2L-attn": "#2563EB",
    "2L-full": "#059669",
    "kappa_0.3": "#7C3AED",
    "kappa_1.0": "#2563EB",
    "kappa_3.0": "#D97706",
    "wd_0.0": "#0D9488",
    "wd_0.01": "#2563EB",
    "wd_0.1": "#DC2626",
    "bigram": "#475569",
    "bayes": "#111827",
}

CONDITION_SPECS = [
    ("1L-attn_k1_wd0.01", "1L-attn", 1.0, 0.01),
    ("2L-attn_k1_wd0.01", "2L-attn", 1.0, 0.01),
    ("2L-full_k1_wd0.01", "2L-full", 1.0, 0.01),
    ("2L-attn_k0.3_wd0.01", "2L-attn", 0.3, 0.01),
    ("2L-attn_k3_wd0.01", "2L-attn", 3.0, 0.01),
    ("2L-attn_k1_wd0", "2L-attn", 1.0, 0.0),
    ("2L-attn_k1_wd0.1", "2L-attn", 1.0, 0.1),
]


def cond_key(arch: str, kappa: float, wd: float) -> str:
    return f"{arch}_k{kappa:g}_wd{wd:g}"


def load_run_data(run_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load summary.json and metrics.jsonl for a single run."""
    with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    metrics = []
    with (run_dir / "metrics.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return summary, metrics


def load_checkpoint_model(ckpt_path: Path, device: torch.device) -> InductionTransformer:
    """Reconstruct InductionTransformer from compact final.pt checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = TransformerConfig(**ckpt["model_config"])
    model = InductionTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def compute_t_phase(
    metrics: List[Dict[str, Any]],
    icl_baseline: float,
    slope_window_steps: int = 500,
) -> Dict[str, Any]:
    """Compute phase-transition timing metrics from a run's metrics.jsonl trajectory.

    Returns:
        - t_phase_icl50: First step where icl_score crosses icl_baseline + 0.5*(icl_final - icl_baseline).
        - t_phase_peak_slope_mid: Midpoint of the 500-step window with maximum negative slope of val_loss_late.
        - t_phase_peak_slope_end: Right endpoint of that 500-step window.
        - peak_neg_slope_per_1k: Maximum drop rate of val_loss_late (nats per 1,000 steps).
    """
    steps = np.array([m["step"] for m in metrics], dtype=np.int64)
    icl = np.array([m["icl_score"] for m in metrics], dtype=np.float64)
    late = np.array([m["val_loss_late"] for m in metrics], dtype=np.float64)

    icl_final = float(icl[-1])
    gain = icl_final - icl_baseline
    target_icl = icl_baseline + 0.5 * gain

    t_icl50: Optional[int] = None
    if gain > 0.05:
        hits = np.where(icl >= target_icl)[0]
        if len(hits) > 0:
            t_icl50 = int(steps[hits[0]])

    # 500-step window negative slope of val_loss_late
    # metrics logged every 100 steps -> window offset = 5 indices
    step_map = {int(s): i for i, s in enumerate(steps)}
    best_drop = -1e9
    best_start = None
    best_end = None
    for idx, s in enumerate(steps):
        s_end = int(s) + slope_window_steps
        if s_end in step_map:
            idx_end = step_map[s_end]
            # Exclude initial embedding-learning drop at step 0->500 when looking for induction cliff
            # unless the model transitions immediately (2L-full transitions around 500-1500)
            if int(s) < 200:
                continue
            drop = float(late[idx] - late[idx_end])
            if drop > best_drop:
                best_drop = drop
                best_start = int(s)
                best_end = s_end

    return {
        "icl_baseline_used": float(icl_baseline),
        "icl_target_50": float(target_icl),
        "t_phase_icl50": t_icl50,
        "t_phase_peak_slope_start": best_start,
        "t_phase_peak_slope_mid": int(0.5 * (best_start + best_end)) if best_start is not None else None,
        "t_phase_peak_slope_end": best_end,
        "peak_neg_slope_per_1k": float(best_drop * (1000.0 / slope_window_steps)) if best_start is not None else None,
    }


@torch.no_grad()
def run_checkpoint_ablation_diagnostics(
    ckpt_path: Path,
    val_tokens_dev: torch.Tensor,
    probe_tokens_dev: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Run progressive cumulative L1 head ablation (0..4) and single-head L0 ablation on final.pt."""
    model = load_checkpoint_model(ckpt_path, device=device)
    n_heads = model.cfg.n_heads

    # Compute exact per-head probe scores directly from checkpoint
    _, attn_patterns = model(probe_tokens_dev, return_attn=True)
    l0_prev_scores = compute_prev_token_score(attn_patterns[0]).cpu().tolist()
    l0_pm_scores = compute_prefix_matching_score(attn_patterns[0], half_len=PROBE_HALF_LEN).cpu().tolist()
    l1_pm_scores = compute_prefix_matching_score(attn_patterns[1], half_len=PROBE_HALF_LEN).cpu().tolist()

    # Baseline (0 heads ablated)
    base = evaluate_validation_loss(model, val_tokens_dev, eval_batch_size=128)
    base_icl = base["icl_score"]
    base_late = base["val_loss_late"]

    # Order L1 heads descending by prefix-matching score
    l1_order = sorted(range(n_heads), key=lambda h: l1_pm_scores[h], reverse=True)

    # 1. Progressive cumulative L1 ablation: k = 0, 1, 2, 3, 4 heads
    cumulative_l1: List[Dict[str, Any]] = [
        {
            "k_ablated": 0,
            "ablated_heads": [],
            "val_loss": base["val_loss"],
            "val_loss_early": base["val_loss_early"],
            "val_loss_late": base_late,
            "icl_score": base_icl,
            "icl_drop": 0.0,
            "late_loss_increase": 0.0,
        }
    ]
    for k in range(1, n_heads + 1):
        heads_to_ablate = [(1, h) for h in l1_order[:k]]
        res = evaluate_validation_loss(
            model, val_tokens_dev, eval_batch_size=128, ablate_heads=heads_to_ablate
        )
        cumulative_l1.append(
            {
                "k_ablated": k,
                "ablated_heads": [f"L1H{h}" for h in l1_order[:k]],
                "val_loss": res["val_loss"],
                "val_loss_early": res["val_loss_early"],
                "val_loss_late": res["val_loss_late"],
                "icl_score": res["icl_score"],
                "icl_drop": base_icl - res["icl_score"],
                "late_loss_increase": res["val_loss_late"] - base_late,
            }
        )

    # 2. Single-head L1 ablations (verify exact match & compute sum of single drops)
    single_l1: Dict[str, Dict[str, float]] = {}
    for h in range(n_heads):
        res = evaluate_validation_loss(
            model, val_tokens_dev, eval_batch_size=128, ablate_heads=[(1, h)]
        )
        single_l1[f"L1H{h}"] = {
            "prefix_match_score": float(l1_pm_scores[h]),
            "ablated_icl_score": res["icl_score"],
            "ablated_val_loss_late": res["val_loss_late"],
            "icl_drop": base_icl - res["icl_score"],
            "late_loss_increase": res["val_loss_late"] - base_late,
        }

    sum_single_l1_icl_drop = sum(v["icl_drop"] for v in single_l1.values())
    all4_l1_icl_drop = cumulative_l1[-1]["icl_drop"]
    sum_single_l1_late_inc = sum(v["late_loss_increase"] for v in single_l1.values())
    all4_l1_late_inc = cumulative_l1[-1]["late_loss_increase"]

    # Super-additivity / backup redundancy ratio:
    # > 1.0 => super-additive total drop (remaining heads compensate when 1 head is ablated)
    # ~ 1.0 => linearly additive independent heads
    # < 1.0 => sub-additive
    redundancy_ratio_icl = all4_l1_icl_drop / max(1e-9, sum_single_l1_icl_drop)
    redundancy_ratio_late = all4_l1_late_inc / max(1e-9, sum_single_l1_late_inc)

    # 3. Single-head L0 previous-token ablations (L0H0..L0H3) + all-4 L0 ablation
    single_l0: Dict[str, Dict[str, float]] = {}
    for h in range(n_heads):
        res = evaluate_validation_loss(
            model, val_tokens_dev, eval_batch_size=128, ablate_heads=[(0, h)]
        )
        single_l0[f"L0H{h}"] = {
            "prev_token_score": float(l0_prev_scores[h]),
            "ablated_icl_score": res["icl_score"],
            "ablated_val_loss_late": res["val_loss_late"],
            "icl_drop": base_icl - res["icl_score"],
            "late_loss_increase": res["val_loss_late"] - base_late,
        }

    all4_l0 = evaluate_validation_loss(
        model, val_tokens_dev, eval_batch_size=128, ablate_heads=[(0, h) for h in range(n_heads)]
    )
    sum_single_l0_icl_drop = sum(v["icl_drop"] for v in single_l0.values())
    all4_l0_icl_drop = base_icl - all4_l0["icl_score"]

    top_l0_h = int(np.argmax(l0_prev_scores))
    ctrl_l0_drops = [single_l0[f"L0H{h}"]["icl_drop"] for h in range(n_heads) if h != top_l0_h]

    return {
        "base_icl_score": base_icl,
        "base_val_loss_late": base_late,
        "per_head_prev_token_L0": [float(x) for x in l0_prev_scores],
        "per_head_prefix_match_L0": [float(x) for x in l0_pm_scores],
        "per_head_prefix_match_L1": [float(x) for x in l1_pm_scores],
        "l1_order_desc_pm": [f"L1H{h}" for h in l1_order],
        "cumulative_l1": cumulative_l1,
        "single_l1": single_l1,
        "sum_single_l1_icl_drop": float(sum_single_l1_icl_drop),
        "all4_l1_icl_drop": float(all4_l1_icl_drop),
        "redundancy_ratio_icl": float(redundancy_ratio_icl),
        "sum_single_l1_late_inc": float(sum_single_l1_late_inc),
        "all4_l1_late_inc": float(all4_l1_late_inc),
        "redundancy_ratio_late": float(redundancy_ratio_late),
        "single_l0": single_l0,
        "top_l0_head": f"L0H{top_l0_h}",
        "top_l0_prev_token_score": float(l0_prev_scores[top_l0_h]),
        "top_l0_icl_drop": float(single_l0[f"L0H{top_l0_h}"]["icl_drop"]),
        "control_l0_mean_icl_drop": float(np.mean(ctrl_l0_drops)),
        "sum_single_l0_icl_drop": float(sum_single_l0_icl_drop),
        "all4_l0_icl_drop": float(all4_l0_icl_drop),
        "l0_redundancy_ratio_icl": float(all4_l0_icl_drop / max(1e-9, sum_single_l0_icl_drop)),
    }


def agg_stats(vals: List[float]) -> Dict[str, float]:
    arr = np.array(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def stack_trajectory(runs: List[Dict[str, Any]], key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (steps, values_matrix) of shape (N_steps,) and (N_seeds, N_steps)."""
    steps = np.array([m["step"] for m in runs[0]["metrics"]], dtype=np.int64)
    mat = np.array([[m[key] for m in r["metrics"]] for r in runs], dtype=np.float64)
    return steps, mat


def plot_fig1_phase_transition_depth_barrier(
    grouped: Dict[str, List[Dict[str, Any]]],
    baselines_k1: Dict[str, float],
    out_path: Path,
) -> None:
    """Figure 1: 3-panel shared x-axis (0..10k steps) for Axis 1 (kappa=1.0, wd=0.01)."""
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.2), sharex=True)
    plt.subplots_adjust(hspace=0.18)

    archs = [
        ("1L-attn_k1_wd0.01", "1L-attn (1 layer, no MLP/LN)", COLORS["1L-attn"]),
        ("2L-attn_k1_wd0.01", "2L-attn (2 layers, attn-only)", COLORS["2L-attn"]),
        ("2L-full_k1_wd0.01", "2L-full (2 layers, Pre-LN + MLP)", COLORS["2L-full"]),
    ]

    # Panel A: val_loss_late
    ax = axes[0]
    for key, label, color in archs:
        steps, mat = stack_trajectory(grouped[key], "val_loss_late")
        mean, std = mat.mean(0), mat.std(0)
        ax.plot(steps, mean, color=color, lw=2.2, label=label)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.16)

    ax.axhline(
        baselines_k1["bigram_val_loss_late"],
        color=COLORS["bigram"],
        ls=":",
        lw=1.6,
        label=f"Global Bigram ({baselines_k1['bigram_val_loss_late']:.3f})",
    )
    ax.axhline(
        baselines_k1["bayes_val_loss_late"],
        color=COLORS["bayes"],
        ls="--",
        lw=1.6,
        label=f"Bayes-Optimal ICL ({baselines_k1['bayes_val_loss_late']:.3f})",
    )
    ax.set_ylabel("Late Val Loss (pos 192–254)")
    ax.set_title("A. Depth Barrier & Phase Transition in Late-Context Loss (κ = 1.0, wd = 0.01)")
    ax.set_ylim(1.45, 4.05)
    ax.legend(loc="center right", frameon=True, framealpha=0.92, ncol=1)

    # Panel B: icl_score
    ax = axes[1]
    for key, label, color in archs:
        steps, mat = stack_trajectory(grouped[key], "icl_score")
        mean, std = mat.mean(0), mat.std(0)
        ax.plot(steps, mean, color=color, lw=2.2, label=label.split(" (")[0])
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.16)

    ax.axhline(
        baselines_k1["bayes_icl_score"],
        color=COLORS["bayes"],
        ls="--",
        lw=1.6,
        label=f"Bayes-Optimal ICL ({baselines_k1['bayes_icl_score']:.3f})",
    )
    ax.set_ylabel("ICL Score (Early − Late)")
    ax.set_title("B. In-Context Learning Score Emergence vs. Exact Bayes-Optimal Limit")
    ax.set_ylim(-0.05, 1.72)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, ncol=2)

    # Panel C: Circuit progress measures aligned to the cliff
    ax = axes[2]
    for key, short_name, color in [
        ("2L-attn_k1_wd0.01", "2L-attn", COLORS["2L-attn"]),
        ("2L-full_k1_wd0.01", "2L-full", COLORS["2L-full"]),
    ]:
        steps, prev_m = stack_trajectory(grouped[key], "max_prev_token_score_L0")
        _, pm_m = stack_trajectory(grouped[key], "max_prefix_match_score")
        _, kc_m = stack_trajectory(grouped[key], "max_k_composition_score")

        ax.plot(steps, prev_m.mean(0), color=color, ls="-", lw=2.2, label=f"{short_name}: prev-token L0")
        ax.fill_between(steps, prev_m.mean(0) - prev_m.std(0), prev_m.mean(0) + prev_m.std(0), color=color, alpha=0.12)
        ax.plot(steps, pm_m.mean(0), color=color, ls="--", lw=1.9, label=f"{short_name}: prefix-match")
        ax.fill_between(steps, pm_m.mean(0) - pm_m.std(0), pm_m.mean(0) + pm_m.std(0), color=color, alpha=0.10)
        ax.plot(steps, kc_m.mean(0), color=color, ls=":", lw=1.7, label=f"{short_name}: K-composition")

    # 1L-attn control baseline on prefix-match
    steps, pm_1l = stack_trajectory(grouped["1L-attn_k1_wd0.01"], "max_prefix_match_score")
    ax.plot(steps, pm_1l.mean(0), color=COLORS["1L-attn"], ls="--", lw=1.5, label="1L-attn: prefix-match (~0.01)")

    ax.set_ylabel("Circuit Score")
    ax.set_xlabel("Training Step (batch size 64, seq len 256; 10k steps ≈ 164M tokens)")
    ax.set_title("C. Mechanistic Progress Measures Aligned to the Phase-Transition Cliff")
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlim(0, 10000)
    ax.legend(
        loc="center right",
        bbox_to_anchor=(0.995, 0.72),
        frameon=True,
        framealpha=0.94,
        ncol=2,
        fontsize=8.2,
    )

    fig.savefig(out_path)
    plt.close(fig)


def plot_fig2_kappa_scaling(
    grouped: Dict[str, List[Dict[str, Any]]],
    val_sets: Dict[float, ValidationSet],
    out_path: Path,
) -> None:
    """Figure 2: 2-panel prior concentration (kappa) scaling for 2L-attn (wd=0.01)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.7))
    plt.subplots_adjust(wspace=0.28)

    kappas = [
        (0.3, "2L-attn_k0.3_wd0.01", COLORS["kappa_0.3"], "κ = 0.3 (strong ICL signal)"),
        (1.0, "2L-attn_k1_wd0.01", COLORS["kappa_1.0"], "κ = 1.0 (balanced prior)"),
        (3.0, "2L-attn_k3_wd0.01", COLORS["kappa_3.0"], "κ = 3.0 (prior-dominated)"),
    ]

    # Panel A: Normalized late-loss recovery trajectories
    for kappa, key, color, label in kappas:
        bl = val_sets[kappa].baselines
        bg = bl["bigram_val_loss_late"]
        by = bl["bayes_val_loss_late"]
        steps, late_mat = stack_trajectory(grouped[key], "val_loss_late")
        rec_mat = 100.0 * (bg - late_mat) / (bg - by)
        mean, std = rec_mat.mean(0), rec_mat.std(0)
        ax1.plot(steps, mean, color=color, lw=2.3, label=label)
        ax1.fill_between(steps, mean - std, mean + std, color=color, alpha=0.16)

    ax1.axhline(100.0, color=COLORS["bayes"], ls="--", lw=1.3, label="Bayes-Optimal (100%)")
    ax1.axhline(0.0, color=COLORS["bigram"], ls=":", lw=1.3, label="Global Bigram (0%)")
    ax1.annotate(
        "Still climbing at 10k\n(Δlate = −0.133 / 1k)",
        xy=(9600, 56),
        xytext=(5400, 48),
        fontsize=8.6,
        color=COLORS["kappa_3.0"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["kappa_3.0"], lw=1.4),
    )
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Late-Loss Recovery (%)  [(Bigram − Model) / (Bigram − Bayes)]")
    ax1.set_title("A. Normalized ICL Recovery Trajectories (2L-attn)", fontsize=10.8)
    ax1.set_xlim(0, 10000)
    ax1.set_ylim(-5, 108)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=8.4)

    # Panel B: Final recovery fraction, prev_token_L0, prefix_match vs kappa
    x_pos = np.arange(3)
    kappa_labels = ["κ = 0.3", "κ = 1.0", "κ = 3.0"]
    metric_specs = [
        ("recovery", "Late-Loss Recovery (frac)", "#2563EB", "o", -0.18),
        ("prev", "Max Prev-Token L0", "#059669", "s", 0.0),
        ("pm", "Max Prefix-Match", "#D97706", "^", 0.18),
    ]

    data_by_metric = {m[0]: [] for m in metric_specs}
    for kappa, key, _, _ in kappas:
        bl = val_sets[kappa].baselines
        bg, by = bl["bigram_val_loss_late"], bl["bayes_val_loss_late"]
        recs = [(bg - r["summary"]["val_loss_late"]) / (bg - by) for r in grouped[key]]
        prevs = [r["summary"]["max_prev_token_score_L0"] for r in grouped[key]]
        pms = [r["summary"]["max_prefix_match_score"] for r in grouped[key]]
        data_by_metric["recovery"].append(recs)
        data_by_metric["prev"].append(prevs)
        data_by_metric["pm"].append(pms)

    for m_key, m_label, color, marker, dx in metric_specs:
        means = [np.mean(vals) for vals in data_by_metric[m_key]]
        stds = [np.std(vals) for vals in data_by_metric[m_key]]
        ax2.errorbar(
            x_pos + dx,
            means,
            yerr=stds,
            color=color,
            marker=marker,
            markersize=7,
            lw=2.0,
            capsize=4,
            label=m_label,
            zorder=4,
        )
        # Plot individual seeds
        for i, vals in enumerate(data_by_metric[m_key]):
            jitter = np.linspace(-0.035, 0.035, len(vals))
            ax2.scatter(
                np.full(len(vals), x_pos[i] + dx) + jitter,
                vals,
                color=color,
                alpha=0.45,
                s=22,
                zorder=3,
            )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(kappa_labels)
    ax2.set_ylabel("Score / Recovery Fraction at Step 10,000")
    ax2.set_title("B. Final Recovery & Circuit Scores vs. κ", fontsize=10.8)
    ax2.set_ylim(0.0, 1.08)
    ax2.legend(loc="lower left", frameon=True, framealpha=0.92)

    fig.savefig(out_path)
    plt.close(fig)


def plot_fig3_wd_contrast_and_kcomp(
    grouped: Dict[str, List[Dict[str, Any]]],
    t_phase_by_run: Dict[str, Dict[str, Any]],
    friedman_results: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """Figure 3: 2-panel paired seed spaghetti plots for Axis 3 (wd in {0.0, 0.01, 0.1})."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.7))
    plt.subplots_adjust(wspace=0.38)

    wds = [
        (0.0, "2L-attn_k1_wd0", "0.0"),
        (0.01, "2L-attn_k1_wd0.01", "0.01"),
        (0.1, "2L-attn_k1_wd0.1", "0.1"),
    ]
    x = np.arange(3)
    wd_labels = [w[2] for w in wds]

    # Build paired (5 seeds x 3 wd) arrays
    def paired_mat(extractor) -> np.ndarray:
        mat = np.zeros((5, 3), dtype=np.float64)
        for j, (_, key, _) in enumerate(wds):
            by_seed = {r["summary"]["config"]["seed"]: r for r in grouped[key]}
            for s in range(5):
                mat[s, j] = extractor(by_seed[s])
        return mat

    icl_mat = paired_mat(lambda r: r["summary"]["icl_score"])
    prev_mat = paired_mat(lambda r: r["summary"]["max_prev_token_score_L0"])
    kcomp_mat = paired_mat(lambda r: r["summary"]["max_k_composition_score"])
    tphase_mat = paired_mat(lambda r: t_phase_by_run[r["run_name"]]["t_phase_icl50"])

    # Panel A: icl_score (left axis) and max_prev_token_score_L0 (right axis)
    ax1_r = ax1.twinx()
    ax1_r.spines["top"].set_visible(False)

    for s in range(5):
        ax1.plot(x, icl_mat[s], color=COLORS["2L-attn"], alpha=0.35, lw=1.2, marker="o", ms=4)
        ax1_r.plot(x, prev_mat[s], color=COLORS["2L-full"], alpha=0.35, lw=1.2, ls="--", marker="s", ms=4)

    l1 = ax1.plot(
        x,
        icl_mat.mean(0),
        color=COLORS["2L-attn"],
        lw=2.6,
        marker="o",
        ms=7,
        label=f"ICL Score (mean; Friedman p = {friedman_results['icl_score']['p']:.4f})",
    )
    l2 = ax1_r.plot(
        x,
        prev_mat.mean(0),
        color=COLORS["2L-full"],
        lw=2.6,
        ls="--",
        marker="s",
        ms=7,
        label=f"Prev-Token L0 (mean; Friedman p = {friedman_results['max_prev_token_score_L0']['p']:.4f})",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"wd = {w}" for w in wd_labels])
    ax1.set_ylabel("Final ICL Score (Early − Late)", color=COLORS["2L-attn"])
    ax1_r.set_ylabel("Max Prev-Token Score (L0)", color=COLORS["2L-full"])
    ax1.tick_params(axis="y", labelcolor=COLORS["2L-attn"])
    ax1_r.tick_params(axis="y", labelcolor=COLORS["2L-full"])
    ax1.set_ylim(1.24, 1.46)
    ax1_r.set_ylim(0.32, 0.86)
    ax1.set_title("A. Weight Decay Damps ICL & L0 Attention (5/5 Seeds)", fontsize=10.6)
    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="lower left", frameon=True, framealpha=0.93, fontsize=8.3)

    # Panel B: max_k_composition_score (left) and t_phase_icl50 (right)
    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)

    for s in range(5):
        ax2.plot(x, kcomp_mat[s], color=COLORS["wd_0.1"], alpha=0.35, lw=1.2, marker="D", ms=4)
        ax2_r.plot(x, tphase_mat[s], color="#7C3AED", alpha=0.35, lw=1.2, ls="--", marker="^", ms=4)

    r1 = ax2.plot(
        x,
        kcomp_mat.mean(0),
        color=COLORS["wd_0.1"],
        lw=2.6,
        marker="D",
        ms=7,
        label=f"Max K-Comp (mean; Friedman p = {friedman_results['max_k_composition_score']['p']:.4f})",
    )
    r2 = ax2_r.plot(
        x,
        tphase_mat.mean(0),
        color="#7C3AED",
        lw=2.6,
        ls="--",
        marker="^",
        ms=7,
        label=f"Phase Step t_phase (Friedman p = {friedman_results['t_phase_icl50']['p']:.4f})",
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"wd = {w}" for w in wd_labels])
    ax2.set_ylabel("Max K-Composition Score", color=COLORS["wd_0.1"])
    ax2_r.set_ylabel("Phase-Transition Step t_phase (50% ICL gain)", color="#7C3AED")
    ax2.tick_params(axis="y", labelcolor=COLORS["wd_0.1"])
    ax2_r.tick_params(axis="y", labelcolor="#7C3AED")
    ax2.set_ylim(0.165, 0.258)
    ax2_r.set_ylim(3400, 4650)
    ax2.set_title("B. Higher wd Raises K-Comp Yet Delays t_phase", fontsize=10.6)
    lines2 = r1 + r2
    ax2.legend(lines2, [ln.get_label() for ln in lines2], loc="upper left", frameon=True, framealpha=0.93, fontsize=8.3)

    fig.savefig(out_path)
    plt.close(fig)


def plot_fig4_distributed_induction_ablation(
    grouped: Dict[str, List[Dict[str, Any]]],
    ablation_by_run: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    """Figure 4: 2-panel resolution of the causal ablation paradox (distributed heads + cumulative ablation)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    plt.subplots_adjust(wspace=0.28)

    # Panel A: Grouped dot plot + mean bars of sorted per-head L0 prev-token and L1 prefix-match
    ranks = np.arange(4)
    rank_labels = ["Rank 1\n(top)", "Rank 2", "Rank 3", "Rank 4\n(lowest)"]

    series = [
        ("2L-attn_k1_wd0.01", "per_head_prev_token_L0", "2L-attn: L0 prev-token", COLORS["2L-attn"], "o", -0.24),
        ("2L-full_k1_wd0.01", "per_head_prev_token_L0", "2L-full: L0 prev-token", COLORS["2L-full"], "s", -0.08),
        ("2L-attn_k1_wd0.01", "per_head_prefix_match_L1", "2L-attn: L1 prefix-match", "#60A5FA", "^", 0.08),
        ("2L-full_k1_wd0.01", "per_head_prefix_match_L1", "2L-full: L1 prefix-match", "#34D399", "D", 0.24),
    ]

    for cond, field, label, color, marker, dx in series:
        sorted_mat = np.array(
            [sorted(ablation_by_run[r["run_name"]][field], reverse=True) for r in grouped[cond]],
            dtype=np.float64,
        )  # (5, 4)
        means = sorted_mat.mean(0)
        stds = sorted_mat.std(0)
        ax1.bar(
            ranks + dx,
            means,
            width=0.14,
            color=color,
            alpha=0.72,
            edgecolor="#1E293B",
            linewidth=0.6,
            label=label,
            zorder=2,
        )
        ax1.errorbar(
            ranks + dx,
            means,
            yerr=stds,
            fmt="none",
            ecolor="#1E293B",
            elinewidth=1.0,
            capsize=2.5,
            zorder=4,
        )
        for j in range(4):
            ax1.scatter(
                np.full(5, ranks[j] + dx),
                sorted_mat[:, j],
                color="#0F172A",
                s=11,
                alpha=0.65,
                zorder=5,
            )

    ax1.set_xticks(ranks)
    ax1.set_xticklabels(rank_labels)
    ax1.set_ylabel("Probe Attention Score (Step 10,000)")
    ax1.set_ylim(0.0, 1.12)
    ax1.set_title("A. Specialized L0 vs. Distributed 4-Head L1 Induction", fontsize=10.6)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8.3)

    # Panel B: Cumulative L1 head ablation curve (0, 1, 2, 3, 4 heads ablated)
    ks = np.arange(5)
    for cond, label, color, marker in [
        ("2L-attn_k1_wd0.01", "2L-attn (attn-only)", COLORS["2L-attn"], "o"),
        ("2L-full_k1_wd0.01", "2L-full (Pre-LN + MLP)", COLORS["2L-full"], "s"),
    ]:
        mat = np.array(
            [[step_d["icl_score"] for step_d in ablation_by_run[r["run_name"]]["cumulative_l1"]] for r in grouped[cond]],
            dtype=np.float64,
        )  # (5, 5)
        ratios = [ablation_by_run[r["run_name"]]["redundancy_ratio_icl"] for r in grouped[cond]]
        mean, std = mat.mean(0), mat.std(0)
        ax2.plot(
            ks,
            mean,
            color=color,
            lw=2.4,
            marker=marker,
            ms=7,
            label=f"{label} (all4 / Σsingle = {np.mean(ratios):.2f}×)",
        )
        ax2.fill_between(ks, mean - std, mean + std, color=color, alpha=0.16)
        # Linear additivity reference line connecting k=0 and k=4
        ax2.plot([0, 4], [mean[0], mean[4]], color=color, ls=":", lw=1.3, alpha=0.7)

    # 1L-attn ceiling reference
    icl_1l = np.mean([r["summary"]["icl_score"] for r in grouped["1L-attn_k1_wd0.01"]])
    ax2.axhline(
        icl_1l,
        color=COLORS["1L-attn"],
        ls="--",
        lw=1.6,
        label=f"1L-attn ceiling ({icl_1l:.3f})",
    )
    ax2.annotate(
        "2L-full backup redundancy\n(1.99× super-additive collapse)",
        xy=(1.05, 1.33),
        xytext=(0.05, 0.56),
        fontsize=8.5,
        color=COLORS["2L-full"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["2L-full"], lw=1.4),
    )

    ax2.set_xticks(ks)
    ax2.set_xticklabels(["0\n(intact)", "1\n(top)", "2", "3", "4\n(all L1)"])
    ax2.set_xlabel("Number of Layer-1 Heads Zero-Ablated (Descending Prefix-Match Order)")
    ax2.set_ylabel("Remaining Validation ICL Score")
    ax2.set_ylim(-0.12, 1.72)
    ax2.set_title("B. Progressive L1 Ablation: Redundancy vs. Additivity", fontsize=10.6)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.93, fontsize=8.2)

    fig.savefig(out_path)
    plt.close(fig)


@torch.no_grad()
def plot_fig5_circuit_heatmaps(
    sweep_root: Path,
    probe_tokens: torch.Tensor,
    device: torch.device,
    out_path: Path,
) -> None:
    """Figure 5: Attention heatmaps on the repeated-token probe for seed=0 (2L-attn vs 2L-full)."""
    probe_dev = probe_tokens.to(device)

    specs = [
        ("2L-attn", sweep_root / "2L-attn_kappa1_wd0.01_seed0" / "final.pt"),
        ("2L-full", sweep_root / "2L-full_kappa1_wd0.01_seed0" / "final.pt"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.2))
    plt.subplots_adjust(wspace=0.30, hspace=0.30)

    for row_idx, (arch_name, ckpt_path) in enumerate(specs):
        model = load_checkpoint_model(ckpt_path, device=device)
        _, attns = model(probe_dev, return_attn=True)

        # Average attention patterns across all 128 probe sequences: (H, 128, 128)
        l0_mean = attns[0].mean(dim=0).cpu().numpy()
        l1_mean = attns[1].mean(dim=0).cpu().numpy()

        l0_prev_scores = compute_prev_token_score(attns[0]).cpu().numpy()
        l1_pm_scores = compute_prefix_matching_score(attns[1], half_len=PROBE_HALF_LEN).cpu().numpy()

        top_l0 = int(np.argmax(l0_prev_scores))
        top_l1 = int(np.argmax(l1_pm_scores))

        # Left panel: Layer 0 top previous-token head (zoomed 0..64 for crystal-clear off-diagonal stripe)
        ax_l0 = axes[row_idx, 0]
        im_l0 = ax_l0.imshow(
            l0_mean[top_l0, :64, :64],
            origin="upper",
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax_l0.set_title(
            f"{arch_name} — Layer 0 Head {top_l0} (prev-token = {l0_prev_scores[top_l0]:.3f})",
            fontsize=10.5,
        )
        ax_l0.set_xlabel("Key Position (0..63)")
        ax_l0.set_ylabel("Query Position (0..63)")
        ax_l0.grid(False)
        fig.colorbar(im_l0, ax=ax_l0, fraction=0.046, pad=0.03)

        # Right panel: Layer 1 top induction head (shared vmax=0.35 so pos-0 sink does not dim induction stripe)
        ax_l1 = axes[row_idx, 1]
        im_l1 = ax_l1.imshow(
            l1_mean[top_l1],
            origin="upper",
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=0.35,
            interpolation="nearest",
        )
        ax_l1.axhline(63.5, color="white", ls="--", lw=1.0, alpha=0.8)
        ax_l1.axvline(63.5, color="white", ls="--", lw=1.0, alpha=0.8)
        ax_l1.annotate(
            "Induction stripe\n(k = t − 63)",
            xy=(32, 95),
            xytext=(70, 35),
            color="white",
            fontsize=8.8,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="white", lw=1.3),
        )
        ax_l1.set_title(
            f"{arch_name} — Layer 1 Head {top_l1} (prefix-match = {l1_pm_scores[top_l1]:.3f})",
            fontsize=10.5,
        )
        ax_l1.set_xlabel("Key Position (0..127)")
        ax_l1.set_ylabel("Query Position (0..127)")
        ax_l1.grid(False)
        fig.colorbar(im_l1, ax=ax_l1, fraction=0.046, pad=0.03, extend="max")

    fig.suptitle(
        "Probe-Averaged Attention Patterns (seed = 0, κ = 1.0, wd = 0.01; N = 128 Repeated Sequences)",
        fontsize=12.2,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(out_path)
    plt.close(fig)


def run_stage2_analysis(
    sweep_root: Path,
    figures_dir: Path,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete Stage 2 analysis, write stage2_analysis.json, and generate all 5 figures."""
    t0 = time.perf_counter()
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

    jobs: List[SweepJob] = build_sweep_matrix()
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    all_runs: Dict[str, Dict[str, Any]] = {}

    for job in jobs:
        run_dir = sweep_root / job.run_name
        summary, metrics = load_run_data(run_dir)
        entry = {
            "job": job,
            "run_name": job.run_name,
            "run_dir": run_dir,
            "summary": summary,
            "metrics": metrics,
        }
        all_runs[job.run_name] = entry
        ck = cond_key(job.arch, job.kappa, job.wd)
        grouped.setdefault(ck, []).append(entry)

    # Sort each group by seed
    for ck in grouped:
        grouped[ck].sort(key=lambda r: r["summary"]["config"]["seed"])

    # Build deterministic validation sets & probe once (matching train.py seeds identically)
    p_base = make_base_transition_matrix(vocab_size=VOCAB_SIZE, data_seed=DATA_SEED)
    probe_tokens = make_prefix_matching_probe(
        vocab_size=VOCAB_SIZE,
        n_probe=N_PROBE,
        half_len=PROBE_HALF_LEN,
        probe_seed=PROBE_SEED,
    )
    val_sets: Dict[float, ValidationSet] = {}
    val_tokens_dev: Dict[float, torch.Tensor] = {}
    for kappa in (0.3, 1.0, 3.0):
        vs = make_validation_set(p_base=p_base, kappa=kappa, n_val=N_VAL, seq_len=SEQ_LEN, val_seed=VAL_SEED)
        val_sets[kappa] = vs
        val_tokens_dev[kappa] = vs.tokens.to(device)
    probe_tokens_dev = probe_tokens.to(device)

    # 1. Phase-transition timing (t_phase) & convergence check
    icl_1l_k1_mean = float(np.mean([r["summary"]["icl_score"] for r in grouped["1L-attn_k1_wd0.01"]]))
    # For kappa != 1.0, compute pre-transition 1L-unigram plateau directly from early pre-cliff steps
    # (steps 1500..2000 where max_prefix_match_score < 0.03)
    pre_plateau_by_kappa: Dict[float, float] = {1.0: icl_1l_k1_mean}
    for kappa, ck in [(0.3, "2L-attn_k0.3_wd0.01"), (3.0, "2L-attn_k3_wd0.01")]:
        pre_vals = []
        for r in grouped[ck]:
            for m in r["metrics"]:
                if 1500 <= m["step"] <= 2000 and m["max_prefix_match_score"] < 0.03:
                    pre_vals.append(m["icl_score"])
        pre_plateau_by_kappa[kappa] = float(np.mean(pre_vals))

    t_phase_by_run: Dict[str, Dict[str, Any]] = {}
    convergence_by_run: Dict[str, Dict[str, float]] = {}

    for run_name, entry in all_runs.items():
        job = entry["job"]
        metrics = entry["metrics"]
        by_step = {m["step"]: m for m in metrics}
        convergence_by_run[run_name] = {
            "d_late_8k_to_10k": float(by_step[10000]["val_loss_late"] - by_step[8000]["val_loss_late"]),
            "d_late_9k_to_10k": float(by_step[10000]["val_loss_late"] - by_step[9000]["val_loss_late"]),
            "d_icl_8k_to_10k": float(by_step[10000]["icl_score"] - by_step[8000]["icl_score"]),
            "d_icl_9k_to_10k": float(by_step[10000]["icl_score"] - by_step[9000]["icl_score"]),
            "d_prevL0_9k_to_10k": float(
                by_step[10000]["max_prev_token_score_L0"] - by_step[9000]["max_prev_token_score_L0"]
            ),
            "d_pmatch_9k_to_10k": float(
                by_step[10000]["max_prefix_match_score"] - by_step[9000]["max_prefix_match_score"]
            ),
        }
        if job.arch == "1L-attn":
            t_phase_by_run[run_name] = {
                "icl_baseline_used": icl_1l_k1_mean,
                "t_phase_icl50": None,
                "t_phase_peak_slope_mid": None,
                "t_phase_peak_slope_end": None,
                "peak_neg_slope_per_1k": None,
            }
        else:
            t_phase_by_run[run_name] = compute_t_phase(
                metrics=metrics,
                icl_baseline=pre_plateau_by_kappa[job.kappa],
                slope_window_steps=500,
            )

    # 2. Checkpoint causal ablation diagnostics across all 30 2L checkpoints
    print(f"Running progressive L1 & single-head L0 ablations across 30 2L checkpoints on {device}...", flush=True)
    ablation_by_run: Dict[str, Dict[str, Any]] = {}
    for run_name, entry in all_runs.items():
        job = entry["job"]
        if job.arch == "1L-attn":
            continue
        ckpt_path = entry["run_dir"] / "final.pt"
        ablation_by_run[run_name] = run_checkpoint_ablation_diagnostics(
            ckpt_path=ckpt_path,
            val_tokens_dev=val_tokens_dev[job.kappa],
            probe_tokens_dev=probe_tokens_dev,
            device=device,
        )

    # 3. Aggregate per-condition statistics
    condition_summaries: Dict[str, Any] = {}
    for ck, arch, kappa, wd in CONDITION_SPECS:
        runs = grouped[ck]
        bl = val_sets[kappa].baselines
        bg_late = bl["bigram_val_loss_late"]
        by_late = bl["bayes_val_loss_late"]

        cond_dict: Dict[str, Any] = {
            "arch": arch,
            "kappa": kappa,
            "wd": wd,
            "n_seeds": len(runs),
            "baselines": bl,
            "val_loss": agg_stats([r["summary"]["val_loss"] for r in runs]),
            "val_loss_early": agg_stats([r["summary"]["val_loss_early"] for r in runs]),
            "val_loss_late": agg_stats([r["summary"]["val_loss_late"] for r in runs]),
            "icl_score": agg_stats([r["summary"]["icl_score"] for r in runs]),
            "late_recovery_pct": agg_stats(
                [100.0 * (bg_late - r["summary"]["val_loss_late"]) / (bg_late - by_late) for r in runs]
            ),
            "max_prev_token_score_L0": agg_stats([r["summary"]["max_prev_token_score_L0"] for r in runs]),
            "max_prefix_match_score": agg_stats([r["summary"]["max_prefix_match_score"] for r in runs]),
            "weight_l2_norm": agg_stats([r["summary"]["weight_l2_norm"] for r in runs]),
            "d_late_9k_to_10k": agg_stats([convergence_by_run[r["run_name"]]["d_late_9k_to_10k"] for r in runs]),
            "d_icl_9k_to_10k": agg_stats([convergence_by_run[r["run_name"]]["d_icl_9k_to_10k"] for r in runs]),
        }

        if arch != "1L-attn":
            cond_dict["max_k_composition_score"] = agg_stats(
                [r["summary"]["max_k_composition_score"] for r in runs]
            )
            cond_dict["t_phase_icl50"] = agg_stats(
                [t_phase_by_run[r["run_name"]]["t_phase_icl50"] for r in runs]
            )
            cond_dict["t_phase_peak_slope_mid"] = agg_stats(
                [t_phase_by_run[r["run_name"]]["t_phase_peak_slope_mid"] for r in runs]
            )
            cond_dict["peak_neg_slope_per_1k"] = agg_stats(
                [t_phase_by_run[r["run_name"]]["peak_neg_slope_per_1k"] for r in runs]
            )

            # Distributed head statistics (sorted rank 1..4 across seeds)
            l1_pm_sorted = np.array(
                [sorted(ablation_by_run[r["run_name"]]["per_head_prefix_match_L1"], reverse=True) for r in runs],
                dtype=np.float64,
            )  # (5, 4)
            l0_prev_sorted = np.array(
                [sorted(ablation_by_run[r["run_name"]]["per_head_prev_token_L0"], reverse=True) for r in runs],
                dtype=np.float64,
            )  # (5, 4)
            cond_dict["l1_prefix_match_by_rank"] = [
                agg_stats(l1_pm_sorted[:, j].tolist()) for j in range(4)
            ]
            cond_dict["l0_prev_token_by_rank"] = [
                agg_stats(l0_prev_sorted[:, j].tolist()) for j in range(4)
            ]
            cond_dict["l1_all_heads_mean_pm"] = float(l1_pm_sorted.mean())
            cond_dict["l1_min_head_pm_across_all_seeds"] = float(l1_pm_sorted.min())

            # Cumulative L1 ablation statistics (k = 0..4)
            cum_summary = []
            for k in range(5):
                cum_summary.append(
                    {
                        "k_ablated": k,
                        "icl_score": agg_stats(
                            [ablation_by_run[r["run_name"]]["cumulative_l1"][k]["icl_score"] for r in runs]
                        ),
                        "icl_drop": agg_stats(
                            [ablation_by_run[r["run_name"]]["cumulative_l1"][k]["icl_drop"] for r in runs]
                        ),
                        "val_loss_late": agg_stats(
                            [ablation_by_run[r["run_name"]]["cumulative_l1"][k]["val_loss_late"] for r in runs]
                        ),
                    }
                )
            cond_dict["cumulative_l1_ablation"] = cum_summary
            cond_dict["sum_single_l1_icl_drop"] = agg_stats(
                [ablation_by_run[r["run_name"]]["sum_single_l1_icl_drop"] for r in runs]
            )
            cond_dict["all4_l1_icl_drop"] = agg_stats(
                [ablation_by_run[r["run_name"]]["all4_l1_icl_drop"] for r in runs]
            )
            cond_dict["redundancy_ratio_icl"] = agg_stats(
                [ablation_by_run[r["run_name"]]["redundancy_ratio_icl"] for r in runs]
            )
            cond_dict["redundancy_ratio_late"] = agg_stats(
                [ablation_by_run[r["run_name"]]["redundancy_ratio_late"] for r in runs]
            )
            cond_dict["top_l0_icl_drop"] = agg_stats(
                [ablation_by_run[r["run_name"]]["top_l0_icl_drop"] for r in runs]
            )
            cond_dict["control_l0_mean_icl_drop"] = agg_stats(
                [ablation_by_run[r["run_name"]]["control_l0_mean_icl_drop"] for r in runs]
            )
            cond_dict["l0_redundancy_ratio_icl"] = agg_stats(
                [ablation_by_run[r["run_name"]]["l0_redundancy_ratio_icl"] for r in runs]
            )

        condition_summaries[ck] = cond_dict

    # 4. Paired Friedman tests across Axis 3 (wd in {0.0, 0.01, 0.1}, seeds 0..4)
    wd_keys = ["2L-attn_k1_wd0", "2L-attn_k1_wd0.01", "2L-attn_k1_wd0.1"]
    friedman_specs = {
        "icl_score": lambda r: r["summary"]["icl_score"],
        "val_loss_late": lambda r: r["summary"]["val_loss_late"],
        "val_loss": lambda r: r["summary"]["val_loss"],
        "max_prev_token_score_L0": lambda r: r["summary"]["max_prev_token_score_L0"],
        "max_prefix_match_score": lambda r: r["summary"]["max_prefix_match_score"],
        "max_k_composition_score": lambda r: r["summary"]["max_k_composition_score"],
        "weight_l2_norm": lambda r: r["summary"]["weight_l2_norm"],
        "t_phase_icl50": lambda r: t_phase_by_run[r["run_name"]]["t_phase_icl50"],
        "t_phase_peak_slope_mid": lambda r: t_phase_by_run[r["run_name"]]["t_phase_peak_slope_mid"],
    }
    friedman_results: Dict[str, Dict[str, Any]] = {}
    for metric_name, fn in friedman_specs.items():
        cols = []
        for wk in wd_keys:
            by_seed = {r["summary"]["config"]["seed"]: r for r in grouped[wk]}
            cols.append([float(fn(by_seed[s])) for s in range(5)])
        stat, pval = friedmanchisquare(cols[0], cols[1], cols[2])
        # Count strictly monotonic seeds across (0.0 -> 0.01 -> 0.1)
        inc_count = sum(1 for s in range(5) if cols[0][s] < cols[1][s] < cols[2][s])
        dec_count = sum(1 for s in range(5) if cols[0][s] > cols[1][s] > cols[2][s])
        friedman_results[metric_name] = {
            "chi2": float(stat),
            "p": float(pval),
            "wd0_mean": float(np.mean(cols[0])),
            "wd0.01_mean": float(np.mean(cols[1])),
            "wd0.1_mean": float(np.mean(cols[2])),
            "strictly_increasing_seeds": inc_count,
            "strictly_decreasing_seeds": dec_count,
            "per_seed_wd0_wd0.01_wd0.1": [[cols[0][s], cols[1][s], cols[2][s]] for s in range(5)],
        }

    # L0 prev_token_score vs single-head L0 icl_drop Spearman correlation across all 30 2L runs (120 heads)
    l0_scores_all, l0_drops_all = [], []
    for ab in ablation_by_run.values():
        for h_info in ab["single_l0"].values():
            l0_scores_all.append(h_info["prev_token_score"])
            l0_drops_all.append(h_info["icl_drop"])
    l0_rho, l0_p = spearmanr(l0_scores_all, l0_drops_all)

    # Save complete stage2_analysis.json
    output = {
        "meta": {
            "sweep_root": str(sweep_root),
            "n_runs_total": len(all_runs),
            "n_2l_checkpoints_ablated": len(ablation_by_run),
            "icl_1l_ceiling_k1": icl_1l_k1_mean,
            "pre_transition_icl_plateau_by_kappa": {str(k): v for k, v in pre_plateau_by_kappa.items()},
        },
        "conditions": condition_summaries,
        "friedman_axis3_wd": friedman_results,
        "l0_prev_token_vs_ablation_drop_spearman": {
            "n_heads": len(l0_scores_all),
            "rho": float(l0_rho),
            "p": float(l0_p),
        },
        "per_run_t_phase": t_phase_by_run,
        "per_run_convergence": convergence_by_run,
        "per_run_ablation": ablation_by_run,
    }

    json_path = sweep_root / "stage2_analysis.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {json_path}", flush=True)

    # 5. Generate all 5 publication-clean figures
    print(f"Generating 5 figures into {figures_dir}...", flush=True)
    plot_fig1_phase_transition_depth_barrier(
        grouped=grouped,
        baselines_k1=val_sets[1.0].baselines,
        out_path=figures_dir / "fig1_phase_transition_depth_barrier.png",
    )
    plot_fig2_kappa_scaling(
        grouped=grouped,
        val_sets=val_sets,
        out_path=figures_dir / "fig2_kappa_scaling.png",
    )
    plot_fig3_wd_contrast_and_kcomp(
        grouped=grouped,
        t_phase_by_run=t_phase_by_run,
        friedman_results=friedman_results,
        out_path=figures_dir / "fig3_wd_contrast_and_kcomp.png",
    )
    plot_fig4_distributed_induction_ablation(
        grouped=grouped,
        ablation_by_run=ablation_by_run,
        out_path=figures_dir / "fig4_distributed_induction_ablation.png",
    )
    plot_fig5_circuit_heatmaps(
        sweep_root=sweep_root,
        probe_tokens=probe_tokens,
        device=device,
        out_path=figures_dir / "fig5_circuit_heatmaps.png",
    )

    elapsed = time.perf_counter() - t0
    print(f"Stage 2 analysis + 5 figures completed in {elapsed:.1f}s.", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 2 trajectory analysis, cumulative ablations, and figure generation."
    )
    in_pkg_root = (Path.cwd() / "pyproject.toml").is_file()
    parser.add_argument(
        "--sweep-root",
        type=str,
        default="runs/overnight_sweep" if in_pkg_root else "induction/runs/overnight_sweep",
        help="Directory containing the 35 completed overnight sweep runs.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="artifacts/figures" if in_pkg_root else "induction/artifacts/figures",
        help="Output directory for the 5 publication-clean PNG figures.",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stage2_analysis(
        sweep_root=Path(args.sweep_root),
        figures_dir=Path(args.figures_dir),
        device_str=args.device,
    )


if __name__ == "__main__":
    main()

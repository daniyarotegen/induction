"""Stage 3 mechanistic dissection: LN self-repair, attention sinks, arch factorization, and kappa=3 25k."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from einops import einsum
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import torch
import torch.nn.functional as F

# Support `python -m induction.src.dissect`, `python -m src.dissect`, and direct execution
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.analysis import agg_stats, compute_t_phase, load_checkpoint_model, load_run_data
    from src.data import (
        DATA_SEED,
        EARLY_POS_END,
        EARLY_POS_START,
        LATE_POS_END,
        LATE_POS_START,
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
    from src.metrics import compute_prefix_matching_score, compute_prev_token_score
    from src.model import InductionTransformer
else:
    from .analysis import agg_stats, compute_t_phase, load_checkpoint_model, load_run_data
    from .data import (
        DATA_SEED,
        EARLY_POS_END,
        EARLY_POS_START,
        LATE_POS_END,
        LATE_POS_START,
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
    from .metrics import compute_prefix_matching_score, compute_prev_token_score
    from .model import InductionTransformer

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.6,
        "xtick.labelsize": 9.3,
        "ytick.labelsize": 9.3,
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
    "2L-attn-ln": "#E11D48",
    "2L-full": "#059669",
    "kappa_0.3": "#7C3AED",
    "kappa_1.0": "#2563EB",
    "kappa_3.0": "#D97706",
    "bigram": "#475569",
    "bayes": "#111827",
}


@torch.no_grad()
def evaluate_validation_loss_with_ln_mode(
    model: InductionTransformer,
    val_tokens: torch.Tensor,
    ablate_heads: Optional[Iterable[Tuple[int, int]]] = None,
    freeze_ln_scales: bool = False,
    eval_batch_size: int = 128,
) -> Dict[str, float]:
    """Evaluate validation loss under standard or Frozen-LN-scale zero-ablation."""
    model.eval()
    device = next(model.parameters()).device
    n_val, seq_len = val_tokens.shape
    n_pred = seq_len - 1
    early_end = min(EARLY_POS_END, n_pred)
    late_start = min(LATE_POS_START, n_pred)
    late_end = min(LATE_POS_END, n_pred)

    total_sum, total_cnt = 0.0, 0
    early_sum, early_cnt = 0.0, 0
    late_sum, late_cnt = 0.0, 0

    ablate_list = list(ablate_heads) if ablate_heads is not None else None

    for start in range(0, n_val, eval_batch_size):
        batch = val_tokens[start : start + eval_batch_size].to(device, non_blocking=True)  # (b, T)
        if freeze_ln_scales and model.cfg.use_ln and ablate_list:
            _, clean_cache = model(batch, return_cache=True)
            logits = model(batch, ablate_heads=ablate_list, frozen_ln_stds=clean_cache["ln_stds"])
        else:
            logits = model(batch, ablate_heads=ablate_list)

        pred_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        targets = batch[:, 1:].reshape(-1)
        loss_per_pos = F.cross_entropy(pred_logits, targets, reduction="none").view(
            batch.shape[0], n_pred
        )

        total_sum += float(loss_per_pos.sum().item())
        total_cnt += loss_per_pos.numel()

        early_slice = loss_per_pos[:, EARLY_POS_START:early_end]
        early_sum += float(early_slice.sum().item())
        early_cnt += early_slice.numel()

        late_slice = loss_per_pos[:, late_start:late_end]
        late_sum += float(late_slice.sum().item())
        late_cnt += late_slice.numel()

    val_loss = total_sum / max(1, total_cnt)
    val_loss_early = early_sum / max(1, early_cnt)
    val_loss_late = late_sum / max(1, late_cnt)
    return {
        "val_loss": val_loss,
        "val_loss_early": val_loss_early,
        "val_loss_late": val_loss_late,
        "icl_score": val_loss_early - val_loss_late,
    }


def compute_prior_occurrence_counts(tokens: torch.Tensor) -> torch.Tensor:
    """Compute prior occurrence count c_u(b, t) = sum_{s < t} 1[x_{b, s} == x_{b, t}].

    Args:
        tokens: Long tensor of shape (N, T).

    Returns:
        counts: Long tensor of shape (N, T) on CPU.
    """
    tok_cpu = tokens.detach().to("cpu", dtype=torch.long)             # (N, T)
    n_seq, seq_len = tok_cpu.shape
    prior_counts = torch.zeros((n_seq, seq_len), dtype=torch.long)    # (N, T)
    seen = torch.zeros((n_seq, VOCAB_SIZE), dtype=torch.long)         # (N, V)
    row_idx = torch.arange(n_seq)                                     # (N,)
    for t in range(seq_len):
        u = tok_cpu[:, t]                                             # (N,)
        prior_counts[:, t] = seen[row_idx, u]
        seen[row_idx, u] += 1
    return prior_counts


@torch.no_grad()
def dissect_checkpoint(
    ckpt_path: Path,
    val_tokens_dev: torch.Tensor,
    prior_counts_dev: torch.Tensor,
    probe_tokens_dev: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Run Extension A (LN self-repair + active-dormant sink + value-state drain) on one checkpoint."""
    model = load_checkpoint_model(ckpt_path, device=device)
    n_heads = model.cfg.n_heads

    # 1. Probe head rankings
    _, probe_attns = model(probe_tokens_dev, return_attn=True)
    l0_prev_scores = compute_prev_token_score(probe_attns[0]).cpu().tolist()
    l1_pm_scores = compute_prefix_matching_score(probe_attns[1], half_len=PROBE_HALF_LEN).cpu().tolist()
    l1_order = sorted(range(n_heads), key=lambda h: l1_pm_scores[h], reverse=True)

    # 2. Standard vs Frozen-LN cumulative & single-head L1 ablations
    def run_l1_suite(freeze_ln: bool) -> Dict[str, Any]:
        base = evaluate_validation_loss_with_ln_mode(
            model, val_tokens_dev, ablate_heads=None, freeze_ln_scales=freeze_ln
        )
        base_icl = base["icl_score"]
        base_late = base["val_loss_late"]

        cum = [
            {
                "k_ablated": 0,
                "icl_score": base_icl,
                "icl_drop": 0.0,
                "val_loss_late": base_late,
                "late_loss_increase": 0.0,
            }
        ]
        for k in range(1, n_heads + 1):
            heads = [(1, h) for h in l1_order[:k]]
            res = evaluate_validation_loss_with_ln_mode(
                model, val_tokens_dev, ablate_heads=heads, freeze_ln_scales=freeze_ln
            )
            cum.append(
                {
                    "k_ablated": k,
                    "icl_score": res["icl_score"],
                    "icl_drop": base_icl - res["icl_score"],
                    "val_loss_late": res["val_loss_late"],
                    "late_loss_increase": res["val_loss_late"] - base_late,
                }
            )

        single_icl_drops = []
        single_late_incs = []
        for h in range(n_heads):
            res = evaluate_validation_loss_with_ln_mode(
                model, val_tokens_dev, ablate_heads=[(1, h)], freeze_ln_scales=freeze_ln
            )
            single_icl_drops.append(base_icl - res["icl_score"])
            single_late_incs.append(res["val_loss_late"] - base_late)

        sum_single_icl = float(sum(single_icl_drops))
        all4_icl = float(cum[-1]["icl_drop"])
        sum_single_late = float(sum(single_late_incs))
        all4_late = float(cum[-1]["late_loss_increase"])

        return {
            "cumulative_l1": cum,
            "single_icl_drops": [float(x) for x in single_icl_drops],
            "sum_single_icl_drop": sum_single_icl,
            "all4_icl_drop": all4_icl,
            "redundancy_ratio_icl": all4_icl / max(1e-9, sum_single_icl),
            "sum_single_late_inc": sum_single_late,
            "all4_late_inc": all4_late,
            "redundancy_ratio_late": all4_late / max(1e-9, sum_single_late),
        }

    std_suite = run_l1_suite(freeze_ln=False)
    frz_suite = run_l1_suite(freeze_ln=True) if model.cfg.use_ln else std_suite

    # 3. Active-Dormant Attention Sink (attn_pos0 stratified by c_u on t >= 1) + Value-State Drain
    bucket_keys = ["0", "1", "2", "3", ">=4"]
    sink_sum = {bk: 0.0 for bk in bucket_keys}
    sink_cnt = {bk: 0 for bk in bucket_keys}
    sink_per_head_sum = {bk: np.zeros(n_heads, dtype=np.float64) for bk in bucket_keys}

    v_pos0_sum = np.zeros(n_heads, dtype=np.float64)
    v_pos0_cnt = 0
    v_rest_sum = np.zeros(n_heads, dtype=np.float64)
    v_rest_cnt = 0

    ov_pos0_sum = np.zeros(n_heads, dtype=np.float64)
    ov_rest_sum = np.zeros(n_heads, dtype=np.float64)

    w_v1 = model.blocks[1].attn.W_V                                   # (H, D, K)
    w_o1 = model.blocks[1].attn.W_O                                   # (H, K, D)

    n_val = val_tokens_dev.shape[0]
    eval_bs = 64
    for start in range(0, n_val, eval_bs):
        batch = val_tokens_dev[start : start + eval_bs]               # (b, T)
        c_batch = prior_counts_dev[start : start + eval_bs]           # (b, T)
        _, cache = model(batch, return_cache=True)

        attn1 = cache["attn_patterns"][1]                             # (b, H, T, T)
        # Exclude t = 0 where causal attention trivially puts 1.0 on key position 0
        attn1_pos0 = attn1[:, :, 1:, 0]                               # (b, H, T - 1)
        c_q = c_batch[:, 1:]                                          # (b, T - 1)

        masks = {
            "0": c_q == 0,
            "1": c_q == 1,
            "2": c_q == 2,
            "3": c_q == 3,
            ">=4": c_q >= 4,
        }
        for bk, m in masks.items():
            cnt = int(m.sum().item())
            if cnt > 0:
                # attn1_pos0 transposed to (b, T-1, H) for clean boolean indexing
                selected = attn1_pos0.permute(0, 2, 1)[m]             # (cnt, H)
                sink_per_head_sum[bk] += selected.sum(dim=0).cpu().numpy()
                sink_sum[bk] += float(selected.mean(dim=-1).sum().item())
                sink_cnt[bk] += cnt

        # Value-state drain on Layer 1 attention input: (b, T, D)
        attn_in_1 = cache["attn_inputs"][1]                           # (b, T, D)
        v = einsum(attn_in_1, w_v1, "b s d, h d k -> b h s k")        # (b, H, T, K)
        ov = einsum(v, w_o1, "b h s k, h k d -> b h s d")             # (b, H, T, D)

        v_norm = torch.linalg.vector_norm(v, dim=-1)                  # (b, H, T)
        ov_norm = torch.linalg.vector_norm(ov, dim=-1)                # (b, H, T)

        b_sz = batch.shape[0]
        t_rest = batch.shape[1] - 1
        v_pos0_sum += v_norm[:, :, 0].sum(dim=0).cpu().numpy()
        v_rest_sum += v_norm[:, :, 1:].sum(dim=(0, 2)).cpu().numpy()
        ov_pos0_sum += ov_norm[:, :, 0].sum(dim=0).cpu().numpy()
        ov_rest_sum += ov_norm[:, :, 1:].sum(dim=(0, 2)).cpu().numpy()
        v_pos0_cnt += b_sz
        v_rest_cnt += b_sz * t_rest

    attn_pos0_by_cu = {bk: sink_sum[bk] / max(1, sink_cnt[bk]) for bk in bucket_keys}
    attn_pos0_per_head_by_cu = {
        bk: (sink_per_head_sum[bk] / max(1, sink_cnt[bk])).tolist() for bk in bucket_keys
    }

    v_pos0_mean_h = v_pos0_sum / max(1, v_pos0_cnt)                   # (H,)
    v_rest_mean_h = v_rest_sum / max(1, v_rest_cnt)                   # (H,)
    ov_pos0_mean_h = ov_pos0_sum / max(1, v_pos0_cnt)                 # (H,)
    ov_rest_mean_h = ov_rest_sum / max(1, v_rest_cnt)                 # (H,)

    v_drain_ratio = float(v_pos0_mean_h.mean() / max(1e-9, v_rest_mean_h.mean()))
    ov_drain_ratio = float(ov_pos0_mean_h.mean() / max(1e-9, ov_rest_mean_h.mean()))

    return {
        "arch": model.cfg.arch,
        "per_head_prev_token_L0": [float(x) for x in l0_prev_scores],
        "per_head_prefix_match_L1": [float(x) for x in l1_pm_scores],
        "standard_ablation": std_suite,
        "frozen_ln_ablation": frz_suite,
        "attn_pos0_by_cu": attn_pos0_by_cu,
        "attn_pos0_per_head_by_cu": attn_pos0_per_head_by_cu,
        "bucket_token_counts": sink_cnt,
        "v_norm_pos0": float(v_pos0_mean_h.mean()),
        "v_norm_rest": float(v_rest_mean_h.mean()),
        "v_drain_ratio": v_drain_ratio,
        "v_drain_ratio_per_head": (v_pos0_mean_h / np.maximum(1e-9, v_rest_mean_h)).tolist(),
        "ov_norm_pos0": float(ov_pos0_mean_h.mean()),
        "ov_norm_rest": float(ov_rest_mean_h.mean()),
        "ov_drain_ratio": ov_drain_ratio,
        "ov_drain_ratio_per_head": (ov_pos0_mean_h / np.maximum(1e-9, ov_rest_mean_h)).tolist(),
    }


def stack_traj(runs: List[Dict[str, Any]], key: str) -> Tuple[np.ndarray, np.ndarray]:
    steps = np.array([m["step"] for m in runs[0]["metrics"]], dtype=np.int64)
    mat = np.array([[m[key] for m in r["metrics"]] for r in runs], dtype=np.float64)
    return steps, mat


def plot_fig6_arch_factorization(
    arch_runs: Dict[str, List[Dict[str, Any]]],
    dissect_by_arch: Dict[str, List[Dict[str, Any]]],
    t_phase_by_arch: Dict[str, List[Dict[str, Any]]],
    baselines_k1: Dict[str, float],
    out_path: Path,
) -> None:
    """Figure 6: 3-panel architectural factorization (2L-attn vs 2L-attn-ln vs 2L-full)."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    plt.subplots_adjust(wspace=0.28)

    order = [
        ("2L-attn", "2L-attn (no LN, no MLP)", COLORS["2L-attn"]),
        ("2L-attn-ln", "2L-attn-ln (Pre-LN, no MLP)", COLORS["2L-attn-ln"]),
        ("2L-full", "2L-full (Pre-LN + GELU MLP)", COLORS["2L-full"]),
    ]

    # Panel A: val_loss_late trajectories
    ax = axes[0]
    for arch, label, color in order:
        steps, mat = stack_traj(arch_runs[arch], "val_loss_late")
        mean, std = mat.mean(0), mat.std(0)
        ax.plot(steps, mean, color=color, lw=2.3, label=label)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.16)
    ax.axhline(
        baselines_k1["bigram_val_loss_late"],
        color=COLORS["bigram"],
        ls=":",
        lw=1.4,
        label=f"Global Bigram ({baselines_k1['bigram_val_loss_late']:.3f})",
    )
    ax.axhline(
        baselines_k1["bayes_val_loss_late"],
        color=COLORS["bayes"],
        ls="--",
        lw=1.5,
        label=f"Bayes-Optimal ({baselines_k1['bayes_val_loss_late']:.3f})",
    )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Late Val Loss (pos 192–254)")
    ax.set_title("A. Late-Loss Trajectories: LN vs. MLP", fontsize=10.6)
    ax.set_xlim(0, 10000)
    ax.set_ylim(1.50, 3.95)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8.0)

    # Panel B: Sorted L0 prev-token scores (Rank 1..4) across the 3 architectures
    ax = axes[1]
    ranks = np.arange(4)
    offsets = [-0.24, 0.0, 0.24]
    for (arch, _, color), dx in zip(order, offsets):
        mat = np.array(
            [sorted(d["per_head_prev_token_L0"], reverse=True) for d in dissect_by_arch[arch]],
            dtype=np.float64,
        )  # (5, 4)
        ax.bar(
            ranks + dx,
            mat.mean(0),
            width=0.22,
            color=color,
            alpha=0.78,
            edgecolor="#1E293B",
            lw=0.6,
            label=arch,
            zorder=2,
        )
        ax.errorbar(
            ranks + dx,
            mat.mean(0),
            yerr=mat.std(0),
            fmt="none",
            ecolor="#1E293B",
            elinewidth=1.0,
            capsize=2.5,
            zorder=4,
        )
        for j in range(4):
            ax.scatter(
                np.full(5, ranks[j] + dx), mat[:, j], color="#0F172A", s=11, alpha=0.65, zorder=5
            )
    ax.set_xticks(ranks)
    ax.set_xticklabels(["Rank 1\n(top)", "Rank 2", "Rank 3", "Rank 4"])
    ax.set_ylabel("Layer-0 Previous-Token Score")
    ax.set_ylim(0.0, 1.12)
    ax.set_title("B. L0 Head Specialization (Single-Head Collapse)", fontsize=10.6)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)

    # Panel C: Final Gap to Bayes-Optimal (left axis) & t_icl50 (right axis)
    ax = axes[2]
    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    x = np.arange(3)
    bayes_late = baselines_k1["bayes_val_loss_late"]

    gap_means, gap_stds = [], []
    tp_means, tp_stds = [], []
    bar_colors = []
    for i, (arch, _, color) in enumerate(order):
        gaps = [r["summary"]["val_loss_late"] - bayes_late for r in arch_runs[arch]]
        tps = [tp["t_phase_icl50"] for tp in t_phase_by_arch[arch]]
        gap_means.append(float(np.mean(gaps)))
        gap_stds.append(float(np.std(gaps)))
        tp_means.append(float(np.mean(tps)))
        tp_stds.append(float(np.std(tps)))
        bar_colors.append(color)
        ax.scatter(np.full(5, i - 0.16), gaps, color="#0F172A", s=14, alpha=0.7, zorder=5)
        ax_r.scatter(np.full(5, i + 0.16), tps, color="#4C1D95", s=14, alpha=0.7, zorder=5)

    b1 = ax.bar(
        x - 0.16,
        gap_means,
        yerr=gap_stds,
        width=0.30,
        color=bar_colors,
        alpha=0.78,
        edgecolor="#1E293B",
        lw=0.7,
        capsize=3,
        label="Gap to Bayes (late nats)",
        zorder=3,
    )
    b2 = ax_r.bar(
        x + 0.16,
        tp_means,
        yerr=tp_stds,
        width=0.30,
        color="#A78BFA",
        alpha=0.75,
        edgecolor="#4C1D95",
        lw=0.7,
        capsize=3,
        hatch="//",
        label="Phase Step t_icl50",
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["2L-attn", "2L-attn-ln", "2L-full"])
    ax.set_ylabel("Final Gap to Bayes Late Loss (nats)")
    ax_r.set_ylabel("Phase-Transition Step t_icl50", color="#5B21B6")
    ax_r.tick_params(axis="y", labelcolor="#5B21B6")
    ax.set_ylim(0.0, max(gap_means) * 1.38)
    ax_r.set_ylim(0, max(tp_means) * 1.30)
    ax.set_title("C. Final Bayes Gap & Transition Timing", fontsize=10.6)
    ax.legend(
        handles=[b1, b2],
        labels=["Gap to Bayes (nats)", "Phase Step t_icl50"],
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        fontsize=8.1,
    )

    fig.savefig(out_path)
    plt.close(fig)


def plot_fig7_self_repair_and_sink(
    dissect_by_arch: Dict[str, List[Dict[str, Any]]],
    icl_1l_ceiling: float,
    out_path: Path,
) -> None:
    """Figure 7: Panel A = Standard vs Frozen-LN cumulative L1 ablation; Panel B = Sink vs c_u + drain inset."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))
    plt.subplots_adjust(wspace=0.28)

    ks = np.arange(5)

    # Panel A: Standard (solid) vs Frozen-LN (dashed) cumulative L1 ablation
    specs = [
        ("2L-attn", COLORS["2L-attn"], "o", False),
        ("2L-attn-ln", COLORS["2L-attn-ln"], "^", True),
        ("2L-full", COLORS["2L-full"], "s", True),
    ]
    for arch, color, marker, has_ln in specs:
        std_mat = np.array(
            [[c["icl_score"] for c in d["standard_ablation"]["cumulative_l1"]] for d in dissect_by_arch[arch]],
            dtype=np.float64,
        )
        std_ratio = float(np.mean([d["standard_ablation"]["redundancy_ratio_icl"] for d in dissect_by_arch[arch]]))
        ax1.plot(
            ks,
            std_mat.mean(0),
            color=color,
            lw=2.3,
            marker=marker,
            ms=6.5,
            label=f"{arch} Std (ratio {std_ratio:.2f}×)",
        )
        ax1.fill_between(ks, std_mat.mean(0) - std_mat.std(0), std_mat.mean(0) + std_mat.std(0), color=color, alpha=0.12)

        if has_ln:
            frz_mat = np.array(
                [[c["icl_score"] for c in d["frozen_ln_ablation"]["cumulative_l1"]] for d in dissect_by_arch[arch]],
                dtype=np.float64,
            )
            frz_ratio = float(
                np.mean([d["frozen_ln_ablation"]["redundancy_ratio_icl"] for d in dissect_by_arch[arch]])
            )
            ax1.plot(
                ks,
                frz_mat.mean(0),
                color=color,
                lw=2.0,
                ls="--",
                marker=marker,
                ms=5.5,
                mfc="white",
                label=f"{arch} Frozen-LN ({frz_ratio:.2f}×)",
            )

    ax1.axhline(
        icl_1l_ceiling,
        color=COLORS["1L-attn"],
        ls=":",
        lw=1.5,
        label=f"1L-attn ceiling ({icl_1l_ceiling:.3f})",
    )
    ax1.set_xticks(ks)
    ax1.set_xticklabels(["0\n(intact)", "1\n(top)", "2", "3", "4\n(all L1)"])
    ax1.set_xlabel("Layer-1 Heads Zero-Ablated (Descending Prefix-Match Order)")
    ax1.set_ylabel("Remaining Validation ICL Score")
    ax1.set_ylim(-0.15, 1.72)
    ax1.set_title("A. LayerNorm Self-Repair: Standard vs. Frozen-LN Ablation", fontsize=10.6)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.93, fontsize=7.9)

    # Panel B: Layer-1 attn_pos0 stratified by prior occurrence count c_u
    buckets = ["0", "1", "2", "3", ">=4"]
    x_b = np.arange(len(buckets))
    for arch, color, marker, _ in specs:
        mat = np.array(
            [[d["attn_pos0_by_cu"][bk] for bk in buckets] for d in dissect_by_arch[arch]],
            dtype=np.float64,
        )  # (5, 5)
        ax2.plot(
            x_b,
            mat.mean(0),
            color=color,
            lw=2.4,
            marker=marker,
            ms=7,
            label=f"{arch} (c_u=0: {mat[:, 0].mean():.3f} → ≥4: {mat[:, -1].mean():.3f})",
        )
        ax2.fill_between(x_b, mat.mean(0) - mat.std(0), mat.mean(0) + mat.std(0), color=color, alpha=0.14)

    ax2.set_xticks(x_b)
    ax2.set_xticklabels(["0\n(unseen)", "1", "2", "3", "≥ 4"])
    ax2.set_xlabel("Prior Context Occurrences c_u of Query Token (t ≥ 1)")
    ax2.set_ylabel("Mean Layer-1 Attention Weight on Key Pos 0")
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_title("B. Active-Dormant Pos-0 Sink & Value-State Drain", fontsize=10.6)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=8.0)

    # Clean inset bar chart placed in empty upper-right whitespace below legend and above sink curves
    ax_in = inset_axes(
        ax2,
        width="100%",
        height="100%",
        bbox_to_anchor=(0.54, 0.44, 0.43, 0.33),
        bbox_transform=ax2.transAxes,
        loc="lower left",
    )
    ax_in.patch.set_alpha(0.96)
    ix = np.arange(3)
    v_ratios = [np.mean([d["v_drain_ratio"] for d in dissect_by_arch[a]]) for a in ("2L-attn", "2L-attn-ln", "2L-full")]
    ov_ratios = [
        np.mean([d["ov_drain_ratio"] for d in dissect_by_arch[a]]) for a in ("2L-attn", "2L-attn-ln", "2L-full")
    ]
    bar_cols = [COLORS["2L-attn"], COLORS["2L-attn-ln"], COLORS["2L-full"]]
    ax_in.bar(ix - 0.17, v_ratios, width=0.32, color=bar_cols, alpha=0.85, edgecolor="#1E293B", lw=0.5, label="||v₀||/||vₜ||")
    ax_in.bar(
        ix + 0.17,
        ov_ratios,
        width=0.32,
        color=bar_cols,
        alpha=0.45,
        edgecolor="#1E293B",
        lw=0.5,
        hatch="///",
        label="||W_O v₀||/||W_O vₜ||",
    )
    ax_in.axhline(1.0, color="#475569", ls=":", lw=0.9)
    ax_in.set_xticks(ix)
    ax_in.set_xticklabels(["attn", "attn+LN", "full"], fontsize=7.6)
    ax_in.set_ylabel("Drain Ratio", fontsize=7.8)
    ax_in.set_title("L1 Value Drain (pos 0 / t≥1)", fontsize=8.0, fontweight="bold")
    ax_in.tick_params(axis="y", labelsize=7.5)
    ax_in.set_ylim(0.0, 1.48)
    ax_in.legend(loc="upper right", fontsize=6.4, frameon=True, framealpha=0.92, ncol=1)

    fig.savefig(out_path)
    plt.close(fig)


def plot_fig8_kappa3_25k(
    k03_runs: List[Dict[str, Any]],
    k10_runs: List[Dict[str, Any]],
    k30_25k_runs: List[Dict[str, Any]],
    val_sets: Dict[float, ValidationSet],
    out_path: Path,
) -> None:
    """Figure 8: Extended 25,000-step convergence of 2L-attn at kappa=3.0 alongside kappa=0.3 and 1.0."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.7))
    plt.subplots_adjust(wspace=0.28)

    # Panel A: Normalized recovery trajectories up to 25k steps
    for kappa, runs, color, label in [
        (0.3, k03_runs, COLORS["kappa_0.3"], "κ = 0.3 (0–10k)"),
        (1.0, k10_runs, COLORS["kappa_1.0"], "κ = 1.0 (0–10k)"),
        (3.0, k30_25k_runs, COLORS["kappa_3.0"], "κ = 3.0 (extended 0–25k)"),
    ]:
        bl = val_sets[kappa].baselines
        bg, by = bl["bigram_val_loss_late"], bl["bayes_val_loss_late"]
        steps, late_mat = stack_traj(runs, "val_loss_late")
        rec_mat = 100.0 * (bg - late_mat) / (bg - by)
        mean, std = rec_mat.mean(0), rec_mat.std(0)
        ax1.plot(steps, mean, color=color, lw=2.3, label=f"{label}: {mean[-1]:.1f}%")
        ax1.fill_between(steps, mean - std, mean + std, color=color, alpha=0.16)
        if steps[-1] == 10000:
            ax1.plot([10000, 25000], [mean[-1], mean[-1]], color=color, ls=":", lw=1.2, alpha=0.65)

    ax1.axvline(10000, color="#64748B", ls="--", lw=1.2, alpha=0.8)
    ax1.text(10450, 42, "10k sweep\ncutoff", fontsize=8.5, color="#475569", fontweight="bold")
    ax1.axhline(100.0, color=COLORS["bayes"], ls="--", lw=1.3, label="Bayes-Optimal (100%)")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Late-Loss Recovery (%)  [(Bigram − Model) / (Bigram − Bayes)]")
    ax1.set_title("A. Normalized Late-Loss Recovery: κ = 3.0 to 25k Steps", fontsize=10.6)
    ax1.set_xlim(0, 25000)
    ax1.set_ylim(-5, 108)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=8.3)

    # Panel B: κ = 3.0 mechanistic convergence (val_loss_late left axis, prev_token & prefix_match right axis)
    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)

    steps3, late3 = stack_traj(k30_25k_runs, "val_loss_late")
    _, prev3 = stack_traj(k30_25k_runs, "max_prev_token_score_L0")
    _, pm3 = stack_traj(k30_25k_runs, "max_prefix_match_score")
    bl3 = val_sets[3.0].baselines

    l1 = ax2.plot(steps3, late3.mean(0), color=COLORS["kappa_3.0"], lw=2.4, label="val_loss_late (κ=3.0)")
    ax2.fill_between(steps3, late3.mean(0) - late3.std(0), late3.mean(0) + late3.std(0), color=COLORS["kappa_3.0"], alpha=0.16)
    l2 = ax2.axhline(bl3["bayes_val_loss_late"], color=COLORS["bayes"], ls="--", lw=1.4, label=f"Bayes late ({bl3['bayes_val_loss_late']:.3f})")
    l3 = ax2.axhline(bl3["bigram_val_loss_late"], color=COLORS["bigram"], ls=":", lw=1.4, label=f"Bigram late ({bl3['bigram_val_loss_late']:.3f})")

    r1 = ax2_r.plot(steps3, prev3.mean(0), color=COLORS["2L-full"], lw=2.0, ls="--", label="prev-token L0")
    r2 = ax2_r.plot(steps3, pm3.mean(0), color=COLORS["2L-attn"], lw=2.0, ls="-.", label="prefix-match L1")

    ax2.axvline(10000, color="#64748B", ls="--", lw=1.1, alpha=0.7)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Late Val Loss (nats)", color=COLORS["kappa_3.0"])
    ax2_r.set_ylabel("Circuit Score (prev-token / prefix-match)", color=COLORS["2L-full"])
    ax2.tick_params(axis="y", labelcolor=COLORS["kappa_3.0"])
    ax2_r.tick_params(axis="y", labelcolor=COLORS["2L-full"])
    ax2.set_xlim(0, 25000)
    ax2.set_ylim(2.55, 3.85)
    ax2_r.set_ylim(0.0, 1.08)
    ax2.set_title("B. κ = 3.0 Loss & Circuit Convergence (0–25k Steps)", fontsize=10.6)

    handles = l1 + [l2, l3] + r1 + r2
    ax2.legend(handles, [h.get_label() for h in handles], loc="center left", bbox_to_anchor=(0.01, 0.45), frameon=True, framealpha=0.94, fontsize=7.8)

    fig.savefig(out_path)
    plt.close(fig)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_stage3_dissection(
    sweep_root: Path,
    ext_b_root: Path,
    ext_c_root: Path,
    figures_dir: Path,
    data_dir: Path,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute Stage 3 Extensions A, B, C analysis, CSV export, and Figures 6–8."""
    t0 = time.perf_counter()
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Fixed deterministic validation sets & probe
    p_base = make_base_transition_matrix(vocab_size=VOCAB_SIZE, data_seed=DATA_SEED)
    probe_tokens = make_prefix_matching_probe(
        vocab_size=VOCAB_SIZE, n_probe=N_PROBE, half_len=PROBE_HALF_LEN, probe_seed=PROBE_SEED
    )
    val_sets: Dict[float, ValidationSet] = {}
    for kappa in (0.3, 1.0, 3.0):
        val_sets[kappa] = make_validation_set(
            p_base=p_base, kappa=kappa, n_val=N_VAL, seq_len=SEQ_LEN, val_seed=VAL_SEED
        )

    val_k1_dev = val_sets[1.0].tokens.to(device)
    prior_counts_k1_dev = compute_prior_occurrence_counts(val_sets[1.0].tokens).to(device)
    probe_dev = probe_tokens.to(device)

    # Load 1L-attn ceiling at kappa=1.0
    runs_1l = []
    for seed in range(5):
        sm, mt = load_run_data(sweep_root / f"1L-attn_kappa1_wd0.01_seed{seed}")
        runs_1l.append({"summary": sm, "metrics": mt})
    icl_1l_ceiling = float(np.mean([r["summary"]["icl_score"] for r in runs_1l]))

    # Load 3 architectures at kappa=1.0, wd=0.01, seeds 0..4
    arch_dirs = {
        "2L-attn": [sweep_root / f"2L-attn_kappa1_wd0.01_seed{s}" for s in range(5)],
        "2L-attn-ln": [ext_b_root / f"2L-attn-ln_kappa1_wd0.01_seed{s}" for s in range(5)],
        "2L-full": [sweep_root / f"2L-full_kappa1_wd0.01_seed{s}" for s in range(5)],
    }

    arch_runs: Dict[str, List[Dict[str, Any]]] = {}
    t_phase_by_arch: Dict[str, List[Dict[str, Any]]] = {}
    dissect_by_arch: Dict[str, List[Dict[str, Any]]] = {}

    for arch, dirs in arch_dirs.items():
        arch_runs[arch] = []
        t_phase_by_arch[arch] = []
        dissect_by_arch[arch] = []
        print(f"Dissecting {arch} (5 checkpoints) on {device}...", flush=True)
        for seed, run_dir in enumerate(dirs):
            sm, mt = load_run_data(run_dir)
            arch_runs[arch].append({"seed": seed, "run_dir": run_dir, "summary": sm, "metrics": mt})
            t_phase_by_arch[arch].append(compute_t_phase(mt, icl_baseline=icl_1l_ceiling))
            dissect_by_arch[arch].append(
                dissect_checkpoint(
                    ckpt_path=run_dir / "final.pt",
                    val_tokens_dev=val_k1_dev,
                    prior_counts_dev=prior_counts_k1_dev,
                    probe_tokens_dev=probe_dev,
                    device=device,
                )
            )

    # Aggregate Extension A & B statistics + export CSVs
    bl_k1 = val_sets[1.0].baselines
    bg_late_k1, by_late_k1 = bl_k1["bigram_val_loss_late"], bl_k1["bayes_val_loss_late"]

    ext_ab_summary: Dict[str, Any] = {}
    ext_b_rows: List[Dict[str, Any]] = []
    ext_a_rows: List[Dict[str, Any]] = []

    for arch in ("2L-attn", "2L-attn-ln", "2L-full"):
        runs = arch_runs[arch]
        dis = dissect_by_arch[arch]
        tps = t_phase_by_arch[arch]

        l0_ranks = np.array([sorted(d["per_head_prev_token_L0"], reverse=True) for d in dis], dtype=np.float64)
        l1_ranks = np.array([sorted(d["per_head_prefix_match_L1"], reverse=True) for d in dis], dtype=np.float64)

        std_ratios = [d["standard_ablation"]["redundancy_ratio_icl"] for d in dis]
        frz_ratios = [d["frozen_ln_ablation"]["redundancy_ratio_icl"] for d in dis]
        std_ratios_late = [d["standard_ablation"]["redundancy_ratio_late"] for d in dis]
        frz_ratios_late = [d["frozen_ln_ablation"]["redundancy_ratio_late"] for d in dis]

        ext_ab_summary[arch] = {
            "n_params": runs[0]["summary"]["config"]["n_params"],
            "t_icl50": agg_stats([tp["t_phase_icl50"] for tp in tps]),
            "t_slope_mid": agg_stats([tp["t_phase_peak_slope_mid"] for tp in tps]),
            "val_loss": agg_stats([r["summary"]["val_loss"] for r in runs]),
            "val_loss_late": agg_stats([r["summary"]["val_loss_late"] for r in runs]),
            "gap_to_bayes_late": agg_stats([r["summary"]["val_loss_late"] - by_late_k1 for r in runs]),
            "late_recovery_pct": agg_stats(
                [100.0 * (bg_late_k1 - r["summary"]["val_loss_late"]) / (bg_late_k1 - by_late_k1) for r in runs]
            ),
            "icl_score": agg_stats([r["summary"]["icl_score"] for r in runs]),
            "l0_prev_token_by_rank": [agg_stats(l0_ranks[:, j].tolist()) for j in range(4)],
            "l1_prefix_match_by_rank": [agg_stats(l1_ranks[:, j].tolist()) for j in range(4)],
            "max_k_composition_score": agg_stats([r["summary"]["max_k_composition_score"] for r in runs]),
            "std_cum_icl": [
                agg_stats([d["standard_ablation"]["cumulative_l1"][k]["icl_score"] for d in dis]) for k in range(5)
            ],
            "frz_cum_icl": [
                agg_stats([d["frozen_ln_ablation"]["cumulative_l1"][k]["icl_score"] for d in dis]) for k in range(5)
            ],
            "std_sum_single_icl_drop": agg_stats([d["standard_ablation"]["sum_single_icl_drop"] for d in dis]),
            "std_all4_icl_drop": agg_stats([d["standard_ablation"]["all4_icl_drop"] for d in dis]),
            "std_redundancy_ratio_icl": agg_stats(std_ratios),
            "frz_sum_single_icl_drop": agg_stats([d["frozen_ln_ablation"]["sum_single_icl_drop"] for d in dis]),
            "frz_all4_icl_drop": agg_stats([d["frozen_ln_ablation"]["all4_icl_drop"] for d in dis]),
            "frz_redundancy_ratio_icl": agg_stats(frz_ratios),
            "std_redundancy_ratio_late": agg_stats(std_ratios_late),
            "frz_redundancy_ratio_late": agg_stats(frz_ratios_late),
            "attn_pos0_by_cu": {
                bk: agg_stats([d["attn_pos0_by_cu"][bk] for d in dis]) for bk in ("0", "1", "2", "3", ">=4")
            },
            "v_norm_pos0": agg_stats([d["v_norm_pos0"] for d in dis]),
            "v_norm_rest": agg_stats([d["v_norm_rest"] for d in dis]),
            "v_drain_ratio": agg_stats([d["v_drain_ratio"] for d in dis]),
            "ov_norm_pos0": agg_stats([d["ov_norm_pos0"] for d in dis]),
            "ov_norm_rest": agg_stats([d["ov_norm_rest"] for d in dis]),
            "ov_drain_ratio": agg_stats([d["ov_drain_ratio"] for d in dis]),
        }

        for s in range(5):
            sm = runs[s]["summary"]
            d = dis[s]
            tp = tps[s]
            l0_s = sorted(d["per_head_prev_token_L0"], reverse=True)
            ext_b_rows.append(
                {
                    "arch": arch,
                    "seed": s,
                    "n_params": sm["config"]["n_params"],
                    "t_icl50": tp["t_phase_icl50"],
                    "t_slope_mid": tp["t_phase_peak_slope_mid"],
                    "val_loss": sm["val_loss"],
                    "val_loss_late": sm["val_loss_late"],
                    "gap_to_bayes_late": sm["val_loss_late"] - by_late_k1,
                    "late_recovery_pct": 100.0 * (bg_late_k1 - sm["val_loss_late"]) / (bg_late_k1 - by_late_k1),
                    "icl_score": sm["icl_score"],
                    "l0_rank1": l0_s[0],
                    "l0_rank2": l0_s[1],
                    "l0_rank3": l0_s[2],
                    "l0_rank4": l0_s[3],
                    "max_prefix_match_score": sm["max_prefix_match_score"],
                    "max_k_composition_score": sm["max_k_composition_score"],
                }
            )
            ext_a_rows.append(
                {
                    "arch": arch,
                    "seed": s,
                    "std_sum_single_icl": d["standard_ablation"]["sum_single_icl_drop"],
                    "std_all4_icl": d["standard_ablation"]["all4_icl_drop"],
                    "std_ratio_icl": d["standard_ablation"]["redundancy_ratio_icl"],
                    "frz_sum_single_icl": d["frozen_ln_ablation"]["sum_single_icl_drop"],
                    "frz_all4_icl": d["frozen_ln_ablation"]["all4_icl_drop"],
                    "frz_ratio_icl": d["frozen_ln_ablation"]["redundancy_ratio_icl"],
                    "std_ratio_late": d["standard_ablation"]["redundancy_ratio_late"],
                    "frz_ratio_late": d["frozen_ln_ablation"]["redundancy_ratio_late"],
                    "attn_pos0_cu0": d["attn_pos0_by_cu"]["0"],
                    "attn_pos0_cu1": d["attn_pos0_by_cu"]["1"],
                    "attn_pos0_cu2": d["attn_pos0_by_cu"]["2"],
                    "attn_pos0_cu3": d["attn_pos0_by_cu"]["3"],
                    "attn_pos0_cu4plus": d["attn_pos0_by_cu"][">=4"],
                    "v_drain_ratio": d["v_drain_ratio"],
                    "ov_drain_ratio": d["ov_drain_ratio"],
                }
            )

    write_csv(data_dir / "ext_b_arch_factorization.csv", list(ext_b_rows[0].keys()), ext_b_rows)
    write_csv(data_dir / "ext_a_self_repair_sinks.csv", list(ext_a_rows[0].keys()), ext_a_rows)

    # Extension C: kappa=3.0 25,000-step convergence
    k03_runs = []
    for s in range(5):
        sm, mt = load_run_data(sweep_root / f"2L-attn_kappa0.3_wd0.01_seed{s}")
        k03_runs.append({"seed": s, "summary": sm, "metrics": mt})

    k30_25k_runs = []
    ext_c_rows = []
    bl3 = val_sets[3.0].baselines
    bg3, by3 = bl3["bigram_val_loss_late"], bl3["bayes_val_loss_late"]

    for s in range(5):
        run_dir = ext_c_root / f"2L-attn_kappa3_wd0.01_seed{s}"
        sm, mt = load_run_data(run_dir)
        by_step = {m["step"]: m for m in mt}
        k30_25k_runs.append({"seed": s, "summary": sm, "metrics": mt})
        rec_10k = 100.0 * (bg3 - by_step[10000]["val_loss_late"]) / (bg3 - by3)
        rec_25k = 100.0 * (bg3 - by_step[25000]["val_loss_late"]) / (bg3 - by3)
        ext_c_rows.append(
            {
                "seed": s,
                "steps": 25000,
                "val_loss_10k": by_step[10000]["val_loss"],
                "val_loss_late_10k": by_step[10000]["val_loss_late"],
                "recovery_pct_10k": rec_10k,
                "icl_10k": by_step[10000]["icl_score"],
                "val_loss_25k": by_step[25000]["val_loss"],
                "val_loss_late_25k": by_step[25000]["val_loss_late"],
                "recovery_pct_25k": rec_25k,
                "icl_25k": by_step[25000]["icl_score"],
                "prev_token_L0_25k": by_step[25000]["max_prev_token_score_L0"],
                "prefix_match_25k": by_step[25000]["max_prefix_match_score"],
                "k_comp_25k": by_step[25000]["max_k_composition_score"],
                "d_late_24k_to_25k": by_step[25000]["val_loss_late"] - by_step[24000]["val_loss_late"],
                "d_icl_24k_to_25k": by_step[25000]["icl_score"] - by_step[24000]["icl_score"],
            }
        )

    write_csv(data_dir / "ext_c_kappa3_25k_summary.csv", list(ext_c_rows[0].keys()), ext_c_rows)

    ext_c_summary = {
        "baselines_kappa3": bl3,
        "val_loss_late_10k": agg_stats([r["val_loss_late_10k"] for r in ext_c_rows]),
        "recovery_pct_10k": agg_stats([r["recovery_pct_10k"] for r in ext_c_rows]),
        "val_loss_late_25k": agg_stats([r["val_loss_late_25k"] for r in ext_c_rows]),
        "recovery_pct_25k": agg_stats([r["recovery_pct_25k"] for r in ext_c_rows]),
        "icl_25k": agg_stats([r["icl_25k"] for r in ext_c_rows]),
        "prev_token_L0_25k": agg_stats([r["prev_token_L0_25k"] for r in ext_c_rows]),
        "prefix_match_25k": agg_stats([r["prefix_match_25k"] for r in ext_c_rows]),
        "k_comp_25k": agg_stats([r["k_comp_25k"] for r in ext_c_rows]),
        "d_late_24k_to_25k": agg_stats([r["d_late_24k_to_25k"] for r in ext_c_rows]),
        "d_icl_24k_to_25k": agg_stats([r["d_icl_24k_to_25k"] for r in ext_c_rows]),
    }

    # Save master JSON
    out_json = sweep_root / "stage3_dissection.json"
    result = {
        "ext_ab_arch_factorization_and_dissection": ext_ab_summary,
        "ext_c_kappa3_25k": ext_c_summary,
        "per_seed_dissection": {arch: dissect_by_arch[arch] for arch in ("2L-attn", "2L-attn-ln", "2L-full")},
        "per_seed_ext_c": ext_c_rows,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_json}", flush=True)

    # Generate Figures 6, 7, 8
    print(f"Generating Figures 6–8 into {figures_dir}...", flush=True)
    plot_fig6_arch_factorization(
        arch_runs=arch_runs,
        dissect_by_arch=dissect_by_arch,
        t_phase_by_arch=t_phase_by_arch,
        baselines_k1=bl_k1,
        out_path=figures_dir / "fig6_arch_factorization_ln_vs_mlp.png",
    )
    plot_fig7_self_repair_and_sink(
        dissect_by_arch=dissect_by_arch,
        icl_1l_ceiling=icl_1l_ceiling,
        out_path=figures_dir / "fig7_self_repair_and_attention_sink.png",
    )
    plot_fig8_kappa3_25k(
        k03_runs=k03_runs,
        k10_runs=arch_runs["2L-attn"],
        k30_25k_runs=k30_25k_runs,
        val_sets=val_sets,
        out_path=figures_dir / "fig8_kappa3_25k_convergence.png",
    )
    print(f"Stage 3 dissection + Figures 6–8 completed in {time.perf_counter() - t0:.1f}s.", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    in_pkg_root = (Path.cwd() / "pyproject.toml").is_file()
    p = "runs" if in_pkg_root else "induction/runs"
    a = "artifacts" if in_pkg_root else "induction/artifacts"
    parser = argparse.ArgumentParser(description="Run Stage 3 mechanistic dissection and Figures 6–8.")
    parser.add_argument("--sweep-root", type=str, default=f"{p}/overnight_sweep")
    parser.add_argument("--ext-b-root", type=str, default=f"{p}/ext_b_2l_attn_ln")
    parser.add_argument("--ext-c-root", type=str, default=f"{p}/ext_c_kappa3_25k")
    parser.add_argument("--figures-dir", type=str, default=f"{a}/figures")
    parser.add_argument("--data-dir", type=str, default=f"{a}/data")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stage3_dissection(
        sweep_root=Path(args.sweep_root),
        ext_b_root=Path(args.ext_b_root),
        ext_c_root=Path(args.ext_c_root),
        figures_dir=Path(args.figures_dir),
        data_dir=Path(args.data_dir),
        device_str=args.device,
    )


if __name__ == "__main__":
    main()

"""CPU-only consultant review diagnostics (T1-T4) for induction Stages 2-3.

Outputs suffixed with `_v2` into `induction/artifacts/data/` without overwriting
any existing CSVs, figures, or README.md.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import friedmanchisquare
import torch
import torch.nn.functional as F

# Support `python -m induction.src.review_diagnostics_v2`, `python -m src.review_diagnostics_v2`, or direct run
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.analysis import load_checkpoint_model, load_run_data
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
        make_base_transition_matrix,
        make_prefix_matching_probe,
        make_validation_set,
    )
    from src.dissect import compute_prior_occurrence_counts
    from src.metrics import compute_prefix_matching_score
    from src.model import InductionTransformer
else:
    from .analysis import load_checkpoint_model, load_run_data
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
        make_base_transition_matrix,
        make_prefix_matching_probe,
        make_validation_set,
    )
    from .dissect import compute_prior_occurrence_counts
    from .metrics import compute_prefix_matching_score
    from .model import InductionTransformer


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(vals: List[float]) -> Tuple[float, float, float]:
    """Return (mean, sample_sd_ddof1, pop_sd_ddof0)."""
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)), float(np.std(arr, ddof=0))


# ---------------------------------------------------------------------------
# T1: kappa in absolute terms + kappa=3.0 exponential fit (15k-25k)
# ---------------------------------------------------------------------------

def fit_exp_decay_15k_25k(steps: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Fit y(t) = a + b * exp(-t / tau) over t in [15000, 25000].

    Uses shifted u = t - 15000 for numerical conditioning:
        y(t) = a + b_15k * exp(-(t - 15000) / tau),   b = b_15k * exp(15000 / tau).
    Tests multiple initial tau guesses and selects the global minimum SSE fit.
    """
    u = (steps - 15000.0).astype(np.float64)

    def model_fn(u_val: np.ndarray, a: float, b_15k: float, tau: float) -> np.ndarray:
        return a + b_15k * np.exp(-u_val / tau)

    best_sse = float("inf")
    best_popt: Optional[np.ndarray] = None

    y_end = float(y[-1])
    y_span = max(1e-4, float(y[0] - y[-1]))

    for tau_init in (500.0, 1500.0, 3000.0, 6000.0, 15000.0, 50000.0):
        for a_init in (max(0.1, y_end - 0.05), max(0.1, y_end - 0.01), max(0.1, y_end - 0.15)):
            b_init = max(1e-4, float(y[0]) - a_init)
            try:
                popt, _ = curve_fit(
                    model_fn,
                    u,
                    y,
                    p0=[a_init, b_init, tau_init],
                    bounds=([0.0, 0.0, 100.0], [4.0, 4.0, 1e6]),
                    maxfev=20000,
                )
                pred = model_fn(u, *popt)
                sse = float(np.sum((y - pred) ** 2))
                if sse < best_sse:
                    best_sse = sse
                    best_popt = popt
            except Exception:
                continue

    if best_popt is None:
        raise RuntimeError("Exponential curve_fit failed for all initializations.")

    a_hat, b15_hat, tau_hat = [float(x) for x in best_popt]
    pred = model_fn(u, a_hat, b15_hat, tau_hat)
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (best_sse / max(1e-15, ss_tot))

    return {
        "a": a_hat,
        "b_15k": b15_hat,
        "tau": tau_hat,
        "rmse": rmse,
        "r2": r2,
        "y_15k": float(y[0]),
        "y_25k": float(y[-1]),
        "remaining_to_a_at_25k": float(y[-1] - a_hat),
    }


def run_t1(sweep_root: Path, ext_c_root: Path, data_dir: Path, val_sets: Dict[float, Any]) -> None:
    print("=" * 88)
    print("T1: KAPPA IN ABSOLUTE TERMS + KAPPA=3.0 EXPONENTIAL FIT (15k-25k)")
    print("=" * 88)

    conds = [
        (0.3, 10000, [sweep_root / f"2L-attn_kappa0.3_wd0.01_seed{s}" for s in range(5)]),
        (1.0, 10000, [sweep_root / f"2L-attn_kappa1_wd0.01_seed{s}" for s in range(5)]),
        (3.0, 10000, [sweep_root / f"2L-attn_kappa3_wd0.01_seed{s}" for s in range(5)]),
        (3.0, 25000, [ext_c_root / f"2L-attn_kappa3_wd0.01_seed{s}" for s in range(5)]),
    ]

    t1_rows: List[Dict[str, Any]] = []
    print(
        f"{'kappa':>5s} {'steps':>6s} | {'late_loss (mean ± sd1)':>22s} | {'bayes_late':>10s} | "
        f"{'gap_to_bayes (nats)':>22s} | {'recovery % (mean ± sd1)':>23s}"
    )
    print("-" * 100)

    for kappa, steps, dirs in conds:
        bl = val_sets[kappa].baselines
        bg_late = bl["bigram_val_loss_late"]
        by_late = bl["bayes_val_loss_late"]

        lates, gaps, recs = [], [], []
        for d in dirs:
            sm, _ = load_run_data(d)
            late = float(sm["val_loss_late"])
            lates.append(late)
            gaps.append(late - by_late)
            recs.append(100.0 * (bg_late - late) / (bg_late - by_late))

        lm, ls1, ls0 = mean_sd(lates)
        gm, gs1, gs0 = mean_sd(gaps)
        rm, rs1, rs0 = mean_sd(recs)
        print(
            f"{kappa:5.1f} {steps:6d} | {lm:9.4f} ± {ls1:8.4f} | {by_late:10.4f} | "
            f"{gm:9.4f} ± {gs1:8.4f} | {rm:9.2f}% ± {rs1:6.2f}%"
        )
        t1_rows.append(
            {
                "kappa": kappa,
                "steps": steps,
                "n_seeds": len(dirs),
                "bigram_late": bg_late,
                "bayes_late": by_late,
                "late_loss_mean": lm,
                "late_loss_sd_ddof1": ls1,
                "late_loss_sd_ddof0": ls0,
                "gap_to_bayes_mean": gm,
                "gap_to_bayes_sd_ddof1": gs1,
                "gap_to_bayes_sd_ddof0": gs0,
                "recovery_pct_mean": rm,
                "recovery_pct_sd_ddof1": rs1,
                "recovery_pct_sd_ddof0": rs0,
            }
        )

    write_csv(data_dir / "t1_kappa_absolute_v2.csv", list(t1_rows[0].keys()), t1_rows)

    # Exponential fit on kappa=3.0 over 15k-25k per seed + last 5k slope
    by_late_k3 = val_sets[3.0].baselines["bayes_val_loss_late"]
    bg_late_k3 = val_sets[3.0].baselines["bigram_val_loss_late"]
    fit_rows: List[Dict[str, Any]] = []
    asym_gaps: List[float] = []
    slopes_ols_per_1k: List[float] = []
    slopes_endpt_per_1k: List[float] = []

    print("\nkappa=3.0 per-seed exponential fit y(t) = a + b_15k * exp(-(t-15000)/tau) over t in [15k, 25k]:")
    print(
        f"{'seed':>4s} | {'a (asym)':>9s} | {'asym_gap':>9s} | {'b_15k':>8s} | {'tau':>8s} | "
        f"{'rmse':>8s} | {'R2':>7s} | {'y25k-a':>8s} | {'OLS_slope_20k-25k (/1k)':>23s} | {'endpt_slope (/1k)':>17s}"
    )
    print("-" * 118)

    for s in range(5):
        _, mt = load_run_data(ext_c_root / f"2L-attn_kappa3_wd0.01_seed{s}")
        steps_arr = np.array([m["step"] for m in mt], dtype=np.float64)
        late_arr = np.array([m["val_loss_late"] for m in mt], dtype=np.float64)

        mask_fit = (steps_arr >= 15000) & (steps_arr <= 25000)
        fit = fit_exp_decay_15k_25k(steps_arr[mask_fit], late_arr[mask_fit])
        asym_gap = fit["a"] - by_late_k3
        asym_gaps.append(asym_gap)

        mask_5k = (steps_arr >= 20000) & (steps_arr <= 25000)
        t_k = steps_arr[mask_5k] / 1000.0
        y_5k = late_arr[mask_5k]
        ols_slope = float(np.polyfit(t_k, y_5k, 1)[0])                # nats per 1k steps
        endpt_slope = float((y_5k[-1] - y_5k[0]) / (t_k[-1] - t_k[0]))  # nats per 1k steps
        slopes_ols_per_1k.append(ols_slope)
        slopes_endpt_per_1k.append(endpt_slope)

        print(
            f"{s:4d} | {fit['a']:9.5f} | {asym_gap:9.5f} | {fit['b_15k']:8.5f} | {fit['tau']:8.1f} | "
            f"{fit['rmse']:8.5f} | {fit['r2']:7.4f} | {fit['remaining_to_a_at_25k']:+8.5f} | "
            f"{ols_slope:+23.6f} | {endpt_slope:+17.6f}"
        )
        fit_rows.append(
            {
                "seed": s,
                "bayes_late": by_late_k3,
                "a_asymptote": fit["a"],
                "asym_gap_nats": asym_gap,
                "asym_recovery_pct": 100.0 * (bg_late_k3 - fit["a"]) / (bg_late_k3 - by_late_k3),
                "b_15k": fit["b_15k"],
                "tau_steps": fit["tau"],
                "rmse": fit["rmse"],
                "r2": fit["r2"],
                "y_15k": fit["y_15k"],
                "y_25k": fit["y_25k"],
                "remaining_to_a_at_25k": fit["remaining_to_a_at_25k"],
                "ols_slope_20k_25k_per_1k": ols_slope,
                "endpt_slope_20k_25k_per_1k": endpt_slope,
            }
        )

    # Bootstrap CI over the 5 seeds (10,000 Monte Carlo resamples + exact 5^5 = 3,125 enumeration)
    gaps_arr = np.asarray(asym_gaps, dtype=np.float64)
    rng = np.random.default_rng(0)
    mc_means = rng.choice(gaps_arr, size=(10000, len(gaps_arr)), replace=True).mean(axis=1)
    mc_lo, mc_hi = float(np.percentile(mc_means, 2.5)), float(np.percentile(mc_means, 97.5))

    exact_means = np.array([np.mean(tup) for tup in itertools.product(gaps_arr, repeat=len(gaps_arr))])
    ex_lo, ex_hi = float(np.percentile(exact_means, 2.5)), float(np.percentile(exact_means, 97.5))

    ag_m, ag_s1, ag_s0 = mean_sd(asym_gaps)
    os_m, os_s1, _ = mean_sd(slopes_ols_per_1k)
    es_m, es_s1, _ = mean_sd(slopes_endpt_per_1k)

    print("-" * 118)
    print(f"Fitted asymptotic gap (a - Bayes_late): mean = {ag_m:.5f} ± {ag_s1:.5f} (sd_ddof1; sd_ddof0 = {ag_s0:.5f}) nats")
    print(f"  Bootstrap 95% CI (B=10000, seed=0): [{mc_lo:.5f}, {mc_hi:.5f}] nats")
    print(f"  Exact 5^5=3125 bootstrap 95% CI   : [{ex_lo:.5f}, {ex_hi:.5f}] nats")
    print(f"  Implied asymptotic recovery %     : {100.0 * (bg_late_k3 - (by_late_k3 + ag_m)) / (bg_late_k3 - by_late_k3):.2f}%")
    print(f"Last-5k slope (20k->25k) OLS        : {os_m:+.6f} ± {os_s1:.6f} nats / 1k steps ({os_m / 1000.0:+.3e} nats/step)")
    print(f"Last-5k slope (20k->25k) endpoint   : {es_m:+.6f} ± {es_s1:.6f} nats / 1k steps ({es_m / 1000.0:+.3e} nats/step)")

    fit_rows.append(
        {
            "seed": "SUMMARY",
            "bayes_late": by_late_k3,
            "a_asymptote": by_late_k3 + ag_m,
            "asym_gap_nats": ag_m,
            "asym_recovery_pct": 100.0 * (bg_late_k3 - (by_late_k3 + ag_m)) / (bg_late_k3 - by_late_k3),
            "b_15k": f"sd1={ag_s1:.5f}",
            "tau_steps": f"boot95_mc=[{mc_lo:.5f},{mc_hi:.5f}]",
            "rmse": f"boot95_exact=[{ex_lo:.5f},{ex_hi:.5f}]",
            "r2": "",
            "y_15k": "",
            "y_25k": "",
            "remaining_to_a_at_25k": "",
            "ols_slope_20k_25k_per_1k": os_m,
            "endpt_slope_20k_25k_per_1k": es_m,
        }
    )
    write_csv(data_dir / "t1_kappa3_exp_fit_v2.csv", list(fit_rows[0].keys()), fit_rows)


# ---------------------------------------------------------------------------
# T2: Self-repair null + Correct-token logit progressive L1 ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_loss_and_logits_cpu(
    model: InductionTransformer,
    val_tokens: torch.Tensor,
    ablate_heads: Optional[Iterable[Tuple[int, int]]] = None,
    freeze_ln_scales: bool = False,
    clean_ln_stds_by_batch: Optional[List[Dict[str, torch.Tensor]]] = None,
    eval_batch_size: int = 128,
) -> Dict[str, float]:
    """Evaluate validation losses, ICL score, and exact float64 correct-token logits on CPU."""
    model.eval()
    n_val, seq_len = val_tokens.shape
    n_pred = seq_len - 1
    early_end = min(EARLY_POS_END, n_pred)
    late_start = min(LATE_POS_START, n_pred)
    late_end = min(LATE_POS_END, n_pred)

    ablate_list = list(ablate_heads) if ablate_heads is not None else None

    loss_early_sum, loss_early_cnt = 0.0, 0
    loss_late_sum, loss_late_cnt = 0.0, 0
    logit_all_sum, logit_all_cnt = 0.0, 0
    logit_early_sum, logit_early_cnt = 0.0, 0
    logit_late_sum, logit_late_cnt = 0.0, 0
    centered_late_sum, centered_late_cnt = 0.0, 0

    for b_idx, start in enumerate(range(0, n_val, eval_batch_size)):
        batch = val_tokens[start : start + eval_batch_size]           # (b, T)
        if freeze_ln_scales and model.cfg.use_ln and ablate_list:
            assert clean_ln_stds_by_batch is not None
            logits = model(
                batch,
                ablate_heads=ablate_list,
                frozen_ln_stds=clean_ln_stds_by_batch[b_idx],
            )                                                         # (b, T, V)
        else:
            logits = model(batch, ablate_heads=ablate_list)           # (b, T, V)

        pred_logits = logits[:, :-1, :]                               # (b, T - 1, V)
        targets = batch[:, 1:]                                        # (b, T - 1)

        # Cross-entropy losses
        ce = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).view(batch.shape[0], n_pred)                                # (b, T - 1)
        ce_e = ce[:, EARLY_POS_START:early_end]
        ce_l = ce[:, late_start:late_end]
        loss_early_sum += float(ce_e.to(torch.float64).sum().item())
        loss_early_cnt += ce_e.numel()
        loss_late_sum += float(ce_l.to(torch.float64).sum().item())
        loss_late_cnt += ce_l.numel()

        # Exact float64 correct-token logits
        pred_f64 = pred_logits.to(torch.float64)                      # (b, T - 1, V)
        correct_logit = pred_f64.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (b, T - 1)
        centered_logit = correct_logit - pred_f64.mean(dim=-1)        # (b, T - 1)

        logit_all_sum += float(correct_logit.sum().item())
        logit_all_cnt += correct_logit.numel()

        cl_e = correct_logit[:, EARLY_POS_START:early_end]
        cl_l = correct_logit[:, late_start:late_end]
        logit_early_sum += float(cl_e.sum().item())
        logit_early_cnt += cl_e.numel()
        logit_late_sum += float(cl_l.sum().item())
        logit_late_cnt += cl_l.numel()

        c_l = centered_logit[:, late_start:late_end]
        centered_late_sum += float(c_l.sum().item())
        centered_late_cnt += c_l.numel()

    val_loss_early = loss_early_sum / loss_early_cnt
    val_loss_late = loss_late_sum / loss_late_cnt
    logit_y_early = logit_early_sum / logit_early_cnt
    logit_y_late = logit_late_sum / logit_late_cnt
    return {
        "val_loss_early": val_loss_early,
        "val_loss_late": val_loss_late,
        "icl_score": val_loss_early - val_loss_late,
        "logit_y_all": logit_all_sum / logit_all_cnt,
        "logit_y_early": logit_y_early,
        "logit_y_late": logit_y_late,
        "logit_y_icl_diff": logit_y_late - logit_y_early,
        "centered_logit_y_late": centered_late_sum / centered_late_cnt,
    }


def run_t2_and_t3(
    sweep_root: Path,
    ext_b_root: Path,
    data_dir: Path,
    val_tokens_k1: torch.Tensor,
    probe_tokens: torch.Tensor,
) -> None:
    print("\n" + "=" * 88)
    print("T2: SELF-REPAIR NULLS & CORRECT-TOKEN LOGIT PROGRESSIVE L1 ABLATION (CPU)")
    print("=" * 88)

    device = torch.device("cpu")
    prior_counts = compute_prior_occurrence_counts(val_tokens_k1)     # (N, T) on CPU
    eval_bs = 128

    arch_dirs = {
        "2L-attn": [sweep_root / f"2L-attn_kappa1_wd0.01_seed{s}" for s in range(5)],
        "2L-attn-ln": [ext_b_root / f"2L-attn-ln_kappa1_wd0.01_seed{s}" for s in range(5)],
        "2L-full": [sweep_root / f"2L-full_kappa1_wd0.01_seed{s}" for s in range(5)],
    }

    # Storage for T2
    t2_per_seed: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arch_dirs}
    t2_csv_rows: List[Dict[str, Any]] = []

    # Storage for T3 (computed from the exact same clean forward pass!)
    pos_bins = [
        ("1-31", 1, 31),
        ("32-95", 32, 95),
        ("96-191", 96, 191),
        ("192-254", 192, 254),
    ]
    cu_buckets = ["0", "1", "2", "3", ">=4"]
    t3_cell_by_arch: Dict[str, Dict[Tuple[str, str], List[float]]] = {
        a: {(pb[0], bk): [] for pb in pos_bins for bk in cu_buckets} for a in arch_dirs
    }
    t3_posbin_by_arch: Dict[str, Dict[str, List[float]]] = {
        a: {pb[0]: [] for pb in pos_bins} for a in arch_dirs
    }
    t3_pos_curve_by_arch: Dict[str, List[np.ndarray]] = {a: [] for a in arch_dirs}
    t3_cell_counts: Dict[Tuple[str, str], int] = {}
    t3_posbin_counts: Dict[str, int] = {}

    # Precompute boolean masks on (N_val, T) for T3 once
    t_grid = torch.arange(val_tokens_k1.shape[1]).unsqueeze(0).expand_as(val_tokens_k1)  # (N, T)
    cu_masks = {
        "0": prior_counts == 0,
        "1": prior_counts == 1,
        "2": prior_counts == 2,
        "3": prior_counts == 3,
        ">=4": prior_counts >= 4,
    }
    for pb_name, lo, hi in pos_bins:
        pm = (t_grid >= lo) & (t_grid <= hi)
        t3_posbin_counts[pb_name] = int(pm.sum().item())
        for bk in cu_buckets:
            t3_cell_counts[(pb_name, bk)] = int((pm & cu_masks[bk]).sum().item())

    for arch, dirs in arch_dirs.items():
        for seed, run_dir in enumerate(dirs):
            model = load_checkpoint_model(run_dir / "final.pt", device=device)
            n_heads = model.cfg.n_heads

            # Descending prefix-match ordering on probe
            with torch.no_grad():
                _, probe_attns = model(probe_tokens, return_attn=True)
            l1_pm = compute_prefix_matching_score(probe_attns[1], half_len=PROBE_HALF_LEN).tolist()
            l1_order = sorted(range(n_heads), key=lambda h: l1_pm[h], reverse=True)

            # Single clean pass: cache LN stds for T2 Frozen-LN AND collect L1 attn_pos0 for T3
            clean_ln_stds_batches: List[Dict[str, torch.Tensor]] = []
            attn0_chunks: List[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, val_tokens_k1.shape[0], eval_bs):
                    batch = val_tokens_k1[start : start + eval_bs]
                    _, cache = model(batch, return_cache=True)
                    clean_ln_stds_batches.append(cache["ln_stds"])
                    # Mean over 4 L1 heads of attention weight on key position 0: (b, T)
                    attn1 = cache["attn_patterns"][1]                 # (b, H, T, T)
                    attn0_chunks.append(attn1[:, :, :, 0].mean(dim=1))

            a0_full = torch.cat(attn0_chunks, dim=0).to(torch.float64)  # (N_val, T)
            t3_pos_curve_by_arch[arch].append(a0_full[:, 1:255].mean(dim=0).numpy())
            for pb_name, lo, hi in pos_bins:
                pm = (t_grid >= lo) & (t_grid <= hi)
                t3_posbin_by_arch[arch][pb_name].append(float(a0_full[pm].mean().item()))
                for bk in cu_buckets:
                    cm = pm & cu_masks[bk]
                    if cm.any():
                        t3_cell_by_arch[arch][(pb_name, bk)].append(float(a0_full[cm].mean().item()))

            # T2 ablation runner for a given mode (standard or frozen-LN)
            base_res = eval_loss_and_logits_cpu(model, val_tokens_k1, ablate_heads=None, eval_batch_size=eval_bs)

            def eval_mode(freeze_ln: bool) -> Dict[str, Any]:
                single_res: Dict[int, Dict[str, float]] = {}
                for h in range(n_heads):
                    single_res[h] = eval_loss_and_logits_cpu(
                        model,
                        val_tokens_k1,
                        ablate_heads=[(1, h)],
                        freeze_ln_scales=freeze_ln,
                        clean_ln_stds_by_batch=clean_ln_stds_batches,
                        eval_batch_size=eval_bs,
                    )

                cum_res: List[Dict[str, float]] = [base_res, single_res[l1_order[0]]]
                for k in range(2, n_heads + 1):
                    heads = [(1, h) for h in l1_order[:k]]
                    cum_res.append(
                        eval_loss_and_logits_cpu(
                            model,
                            val_tokens_k1,
                            ablate_heads=heads,
                            freeze_ln_scales=freeze_ln,
                            clean_ln_stds_by_batch=clean_ln_stds_batches,
                            eval_batch_size=eval_bs,
                        )
                    )

                out: Dict[str, Any] = {"cum": cum_res}
                # Compute additivity ratios for ICL score, val_loss_late, and all 4 logit metrics
                metric_signs = [
                    ("icl_score", 1.0),              # drop = base - abl
                    ("val_loss_late", -1.0),         # increase = abl - base
                    ("logit_y_late", 1.0),           # drop = base - abl
                    ("logit_y_all", 1.0),            # drop = base - abl
                    ("logit_y_icl_diff", 1.0),       # drop = base - abl
                    ("centered_logit_y_late", 1.0),  # drop = base - abl
                ]
                for m_name, sign in metric_signs:
                    s_drops = [sign * (base_res[m_name] - single_res[h][m_name]) for h in range(n_heads)]
                    sum_s = float(sum(s_drops))
                    all4 = float(sign * (base_res[m_name] - cum_res[4][m_name]))
                    out[f"{m_name}_sum_single"] = sum_s
                    out[f"{m_name}_all4"] = all4
                    out[f"{m_name}_ratio"] = all4 / sum_s
                    out[f"{m_name}_cum_drops"] = [
                        float(sign * (base_res[m_name] - cum_res[k][m_name])) for k in range(5)
                    ]
                    out[f"{m_name}_cum_vals"] = [float(cum_res[k][m_name]) for k in range(5)]
                return out

            std_out = eval_mode(freeze_ln=False)
            frz_out = eval_mode(freeze_ln=True) if model.cfg.use_ln else std_out

            # Mandatory exact linearity sanity checks
            if arch == "2L-attn":
                for m_chk in ("logit_y_late", "logit_y_all", "logit_y_icl_diff", "centered_logit_y_late"):
                    assert abs(std_out[f"{m_chk}_ratio"] - 1.0) < 1e-5, (
                        f"Sanity check failed for 2L-attn seed={seed} {m_chk}: {std_out[f'{m_chk}_ratio']}"
                    )
            if arch == "2L-attn-ln":
                for m_chk in ("logit_y_late", "logit_y_all", "logit_y_icl_diff", "centered_logit_y_late"):
                    assert abs(frz_out[f"{m_chk}_ratio"] - 1.0) < 1e-5, (
                        f"Sanity check failed for 2L-attn-ln Frozen-LN seed={seed} {m_chk}: {frz_out[f'{m_chk}_ratio']}"
                    )

            t2_per_seed[arch].append({"seed": seed, "std": std_out, "frz": frz_out})

            for mode_name, m_out in (("standard", std_out), ("frozen_ln", frz_out)):
                if arch == "2L-attn" and mode_name == "frozen_ln":
                    continue
                t2_csv_rows.append(
                    {
                        "arch": arch,
                        "mode": mode_name,
                        "seed": seed,
                        "icl_ratio": m_out["icl_score_ratio"],
                        "late_loss_ratio": m_out["val_loss_late_ratio"],
                        "logit_y_late_base": m_out["logit_y_late_cum_vals"][0],
                        "logit_y_late_k1": m_out["logit_y_late_cum_vals"][1],
                        "logit_y_late_k2": m_out["logit_y_late_cum_vals"][2],
                        "logit_y_late_k3": m_out["logit_y_late_cum_vals"][3],
                        "logit_y_late_k4": m_out["logit_y_late_cum_vals"][4],
                        "logit_y_late_sum_single_drop": m_out["logit_y_late_sum_single"],
                        "logit_y_late_all4_drop": m_out["logit_y_late_all4"],
                        "logit_y_late_ratio": m_out["logit_y_late_ratio"],
                        "logit_y_all_sum_single_drop": m_out["logit_y_all_sum_single"],
                        "logit_y_all_all4_drop": m_out["logit_y_all_all4"],
                        "logit_y_all_ratio": m_out["logit_y_all_ratio"],
                        "logit_y_icl_diff_ratio": m_out["logit_y_icl_diff_ratio"],
                        "centered_logit_y_late_ratio": m_out["centered_logit_y_late_ratio"],
                    }
                )

    write_csv(data_dir / "t2_logit_ablation_v2.csv", list(t2_csv_rows[0].keys()), t2_csv_rows)

    # T2 Part A: LN share against nulls {1.0, 2L-attn mean, 2L-attn-ln mean}
    null_rows: List[Dict[str, Any]] = []
    print("\nPart A — LN share of 2L-full excess redundancy: (r_full_std - r_full_frz) / (r_full_std - r_null)")
    print(
        f"{'metric':>14s} | {'null_name':>18s} | {'r_null':>8s} | {'r_full_std':>10s} | "
        f"{'r_full_frz':>10s} | {'excess':>8s} | {'LN_delta':>8s} | {'LN_share %':>10s}"
    )
    print("-" * 104)

    for metric_key, metric_label in (("icl_score", "icl_score"), ("val_loss_late", "val_loss_late")):
        r_full_std = float(np.mean([x["std"][f"{metric_key}_ratio"] for x in t2_per_seed["2L-full"]]))
        r_full_frz = float(np.mean([x["frz"][f"{metric_key}_ratio"] for x in t2_per_seed["2L-full"]]))
        r_attn_std = float(np.mean([x["std"][f"{metric_key}_ratio"] for x in t2_per_seed["2L-attn"]]))
        r_attn_ln_std = float(np.mean([x["std"][f"{metric_key}_ratio"] for x in t2_per_seed["2L-attn-ln"]]))
        r_attn_ln_frz = float(np.mean([x["frz"][f"{metric_key}_ratio"] for x in t2_per_seed["2L-attn-ln"]]))

        for null_name, r_null in [
            ("1.0", 1.0),
            ("2L-attn mean", r_attn_std),
            ("2L-attn-ln (std)", r_attn_ln_std),
            ("2L-attn-ln (frz)", r_attn_ln_frz),
        ]:
            excess = r_full_std - r_null
            ln_delta = r_full_std - r_full_frz
            ln_share = 100.0 * ln_delta / excess
            print(
                f"{metric_label:>14s} | {null_name:>18s} | {r_null:8.4f} | {r_full_std:10.4f} | "
                f"{r_full_frz:10.4f} | {excess:8.4f} | {ln_delta:8.4f} | {ln_share:9.2f}%"
            )
            null_rows.append(
                {
                    "metric": metric_label,
                    "null_name": null_name,
                    "r_null": r_null,
                    "r_full_std": r_full_std,
                    "r_full_frz": r_full_frz,
                    "excess_over_null": excess,
                    "ln_delta": ln_delta,
                    "ln_share_pct": ln_share,
                }
            )

    write_csv(data_dir / "t2_self_repair_null_v2.csv", list(null_rows[0].keys()), null_rows)

    # T2 Part B: Progressive L1 ablation on correct-token logit
    print("\nPart B — Progressive L1 ablation on correct-token logit (mean ± sd_ddof1 across 5 seeds):")
    print(
        f"{'arch':>10s} {'mode':>9s} | {'logit_y_late ratio':>21s} | {'logit_y_all ratio':>21s} | "
        f"{'logit_icl_diff ratio':>21s} | {'centered_late ratio':>21s}"
    )
    print("-" * 116)
    for arch in ("2L-attn", "2L-attn-ln", "2L-full"):
        modes = [("standard", "std")] if arch == "2L-attn" else [("standard", "std"), ("frozen_ln", "frz")]
        for m_label, m_key in modes:
            r_late = mean_sd([x[m_key]["logit_y_late_ratio"] for x in t2_per_seed[arch]])
            r_all = mean_sd([x[m_key]["logit_y_all_ratio"] for x in t2_per_seed[arch]])
            r_icl = mean_sd([x[m_key]["logit_y_icl_diff_ratio"] for x in t2_per_seed[arch]])
            r_cen = mean_sd([x[m_key]["centered_logit_y_late_ratio"] for x in t2_per_seed[arch]])
            print(
                f"{arch:>10s} {m_label:>9s} | {r_late[0]:9.6f} ± {r_late[1]:8.6f} | "
                f"{r_all[0]:9.6f} ± {r_all[1]:8.6f} | {r_icl[0]:9.6f} ± {r_icl[1]:8.6f} | "
                f"{r_cen[0]:9.6f} ± {r_cen[1]:8.6f}"
            )

    print("\nProgressive correct-token logit (late pos 192-254) values & drops (mean across 5 seeds):")
    print(
        f"{'arch':>10s} {'mode':>9s} | {'k=0':>7s} {'k=1':>7s} {'k=2':>7s} {'k=3':>7s} {'k=4':>7s} | "
        f"{'sum_single':>10s} {'all4_drop':>10s} {'ratio':>10s}"
    )
    print("-" * 96)
    for arch in ("2L-attn", "2L-attn-ln", "2L-full"):
        modes = [("standard", "std")] if arch == "2L-attn" else [("standard", "std"), ("frozen_ln", "frz")]
        for m_label, m_key in modes:
            ks = [float(np.mean([x[m_key]["logit_y_late_cum_vals"][k] for x in t2_per_seed[arch]])) for k in range(5)]
            ss = float(np.mean([x[m_key]["logit_y_late_sum_single"] for x in t2_per_seed[arch]]))
            a4 = float(np.mean([x[m_key]["logit_y_late_all4"] for x in t2_per_seed[arch]]))
            rat = float(np.mean([x[m_key]["logit_y_late_ratio"] for x in t2_per_seed[arch]]))
            print(
                f"{arch:>10s} {m_label:>9s} | {ks[0]:7.4f} {ks[1]:7.4f} {ks[2]:7.4f} {ks[3]:7.4f} {ks[4]:7.4f} | "
                f"{ss:10.5f} {a4:10.5f} {rat:10.6f}"
            )

    # Also compute LN share on logit_y_late and logit_y_all for exact parity with Part A
    for l_key in ("logit_y_late", "logit_y_all", "centered_logit_y_late"):
        rf_s = float(np.mean([x["std"][f"{l_key}_ratio"] for x in t2_per_seed["2L-full"]]))
        rf_f = float(np.mean([x["frz"][f"{l_key}_ratio"] for x in t2_per_seed["2L-full"]]))
        ra_ln_s = float(np.mean([x["std"][f"{l_key}_ratio"] for x in t2_per_seed["2L-attn-ln"]]))
        print(
            f"  [{l_key}] 2L-full std={rf_s:.6f}, frz={rf_f:.6f}, 2L-attn-ln std={ra_ln_s:.6f} | "
            f"LN share vs 1.0 = {100.0 * (rf_s - rf_f) / (rf_s - 1.0):.2f}%"
        )

    # -----------------------------------------------------------------------
    # T3: Sink position control
    # -----------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("T3: SINK POSITION CONTROL — KEY-0 ATTENTION BY c_u WITHIN POSITION BINS (CPU)")
    print("=" * 88)

    t3_cell_rows: List[Dict[str, Any]] = []
    for arch in ("2L-attn", "2L-attn-ln", "2L-full"):
        print(f"\nArchitecture: {arch} (mean ± sd_ddof1 across 5 seeds)")
        header = f"{'pos_bin':>8s} | " + " | ".join(f"{'c_u=' + bk:>19s}" for bk in cu_buckets)
        print(header)
        print("-" * len(header))
        for pb_name, _, _ in pos_bins:
            cells_str = []
            for bk in cu_buckets:
                vals = t3_cell_by_arch[arch][(pb_name, bk)]
                cnt = t3_cell_counts[(pb_name, bk)]
                if vals:
                    m, s1, s0 = mean_sd(vals)
                    cells_str.append(f"{m:6.4f}±{s1:6.4f}({cnt:5d})")
                    t3_cell_rows.append(
                        {
                            "arch": arch,
                            "pos_bin": pb_name,
                            "c_u": bk,
                            "n_tokens": cnt,
                            "attn_pos0_mean": m,
                            "attn_pos0_sd_ddof1": s1,
                            "attn_pos0_sd_ddof0": s0,
                        }
                    )
                else:
                    cells_str.append(f"{'n/a':>19s}")
            print(f"{pb_name:>8s} | " + " | ".join(cells_str))

    write_csv(data_dir / "t3_sink_pos_x_cu_v2.csv", list(t3_cell_rows[0].keys()), t3_cell_rows)

    print("\nKey-0 attention vs. position bin alone (marginalized over c_u; mean ± sd_ddof1 across 5 seeds):")
    print(f"{'pos_bin':>8s} | {'n_tokens':>8s} | {'2L-attn':>19s} | {'2L-attn-ln':>19s} | {'2L-full':>19s}")
    print("-" * 86)
    t3_pos_rows: List[Dict[str, Any]] = []
    for pb_name, _, _ in pos_bins:
        cnt = t3_posbin_counts[pb_name]
        row_d: Dict[str, Any] = {"pos_bin": pb_name, "n_tokens": cnt}
        strs = []
        for arch in ("2L-attn", "2L-attn-ln", "2L-full"):
            m, s1, s0 = mean_sd(t3_posbin_by_arch[arch][pb_name])
            strs.append(f"{m:8.5f} ± {s1:7.5f}")
            row_d[f"{arch}_mean"] = m
            row_d[f"{arch}_sd_ddof1"] = s1
            row_d[f"{arch}_sd_ddof0"] = s0
        print(f"{pb_name:>8s} | {cnt:8d} | " + " | ".join(strs))
        t3_pos_rows.append(row_d)

    write_csv(data_dir / "t3_sink_vs_position_v2.csv", list(t3_pos_rows[0].keys()), t3_pos_rows)

    # Fine-grained per-position CSV (t = 1..254)
    pos_t_rows: List[Dict[str, Any]] = []
    curves = {a: np.stack(t3_pos_curve_by_arch[a], axis=0) for a in ("2L-attn", "2L-attn-ln", "2L-full")}
    for idx, t_val in enumerate(range(1, 255)):
        r = {"pos_t": t_val}
        for arch in ("2L-attn", "2L-attn-ln", "2L-full"):
            r[f"{arch}_mean"] = float(curves[arch][:, idx].mean())
            r[f"{arch}_sd_ddof1"] = float(curves[arch][:, idx].std(ddof=1))
        pos_t_rows.append(r)
    write_csv(data_dir / "t3_sink_vs_position_by_t_v2.csv", list(pos_t_rows[0].keys()), pos_t_rows)


# ---------------------------------------------------------------------------
# T4: K-composition numerator & denominator norms for wd in {0, 0.01, 0.1}
# ---------------------------------------------------------------------------

def run_t4(sweep_root: Path, data_dir: Path) -> None:
    print("\n" + "=" * 88)
    print("T4: K-COMPOSITION NUMERATOR & DENOMINATOR NORMS (2L-attn, kappa=1.0, step 10k, CPU)")
    print("=" * 88)

    device = torch.device("cpu")
    wds = [("0", 0.0), ("0.01", 0.01), ("0.1", 0.1)]
    rows: List[Dict[str, Any]] = []
    by_wd: Dict[float, List[Dict[str, Any]]] = {w: [] for _, w in wds}

    for wd_str, wd_val in wds:
        for seed in range(5):
            run_dir = sweep_root / f"2L-attn_kappa1_wd{wd_str}_seed{seed}"
            sm, _ = load_run_data(run_dir)
            model = load_checkpoint_model(run_dir / "final.pt", device=device)

            with torch.no_grad():
                w_ov_0 = model.get_W_OV(0).to(torch.float64)          # (4, D, D)
                w_qk_1 = model.get_W_QK(1).to(torch.float64)          # (4, D, D)

                ov_norms = torch.linalg.matrix_norm(w_ov_0, ord="fro")  # (4,)
                qk_norms = torch.linalg.matrix_norm(w_qk_1, ord="fro")  # (4,)

                num_mat = torch.empty((4, 4), dtype=torch.float64)
                den_mat = torch.empty((4, 4), dtype=torch.float64)
                rat_mat = torch.empty((4, 4), dtype=torch.float64)
                for h0 in range(4):
                    for h1 in range(4):
                        prod = w_qk_1[h1] @ w_ov_0[h0]
                        num_mat[h0, h1] = torch.linalg.matrix_norm(prod, ord="fro")
                        den_mat[h0, h1] = qk_norms[h1] * ov_norms[h0]
                        rat_mat[h0, h1] = num_mat[h0, h1] / (den_mat[h0, h1] + 1e-12)

            flat_idx = int(torch.argmax(rat_mat).item())
            h0_star, h1_star = divmod(flat_idx, 4)

            max_rat = float(rat_mat[h0_star, h1_star].item())
            # Cross-check against logged summary.json
            logged_rat = float(sm["max_k_composition_score"])
            assert abs(max_rat - logged_rat) < 1e-5, f"Mismatch {max_rat} vs {logged_rat}"

            rec = {
                "wd": wd_val,
                "seed": seed,
                "argmax_pair": f"L0H{h0_star}->L1H{h1_star}",
                "argmax_ratio": max_rat,
                "argmax_num_fro": float(num_mat[h0_star, h1_star].item()),
                "argmax_qk_fro": float(qk_norms[h1_star].item()),
                "argmax_ov_fro": float(ov_norms[h0_star].item()),
                "argmax_denom_prod": float(den_mat[h0_star, h1_star].item()),
                "all16_ratio_mean": float(rat_mat.mean().item()),
                "all16_num_fro_mean": float(num_mat.mean().item()),
                "all4_qk_fro_mean": float(qk_norms.mean().item()),
                "all4_ov_fro_mean": float(ov_norms.mean().item()),
                "all16_denom_prod_mean": float(den_mat.mean().item()),
            }
            rows.append(rec)
            by_wd[wd_val].append(rec)

    write_csv(data_dir / "t4_kcomp_decomposition_v2.csv", list(rows[0].keys()), rows)

    print("\nPart A — Argmax (h0*, h1*) pair achieving max_k_composition_score (mean ± sd_ddof1 across 5 seeds):")
    print(
        f"{'wd':>5s} | {'ratio':>17s} | {'num ||W_QK W_OV||_F':>21s} | {'||W_QK||_F':>19s} | "
        f"{'||W_OV||_F':>19s} | {'denom prod':>21s}"
    )
    print("-" * 116)
    for _, wd_val in wds:
        r_m = mean_sd([x["argmax_ratio"] for x in by_wd[wd_val]])
        n_m = mean_sd([x["argmax_num_fro"] for x in by_wd[wd_val]])
        q_m = mean_sd([x["argmax_qk_fro"] for x in by_wd[wd_val]])
        o_m = mean_sd([x["argmax_ov_fro"] for x in by_wd[wd_val]])
        d_m = mean_sd([x["argmax_denom_prod"] for x in by_wd[wd_val]])
        print(
            f"{wd_val:5.2f} | {r_m[0]:7.4f} ± {r_m[1]:7.4f} | {n_m[0]:9.4f} ± {n_m[1]:9.4f} | "
            f"{q_m[0]:8.4f} ± {q_m[1]:8.4f} | {o_m[0]:8.4f} ± {o_m[1]:8.4f} | "
            f"{d_m[0]:9.4f} ± {d_m[1]:9.4f}"
        )

    print("\nPart B — Mean over all 16 (h0, h1) pairs & all 4 heads per layer (mean ± sd_ddof1 across 5 seeds):")
    print(
        f"{'wd':>5s} | {'ratio (16)':>17s} | {'num (16)':>21s} | {'||W_QK||_F (4)':>19s} | "
        f"{'||W_OV||_F (4)':>19s} | {'denom prod (16)':>21s}"
    )
    print("-" * 116)
    for _, wd_val in wds:
        r_m = mean_sd([x["all16_ratio_mean"] for x in by_wd[wd_val]])
        n_m = mean_sd([x["all16_num_fro_mean"] for x in by_wd[wd_val]])
        q_m = mean_sd([x["all4_qk_fro_mean"] for x in by_wd[wd_val]])
        o_m = mean_sd([x["all4_ov_fro_mean"] for x in by_wd[wd_val]])
        d_m = mean_sd([x["all16_denom_prod_mean"] for x in by_wd[wd_val]])
        print(
            f"{wd_val:5.2f} | {r_m[0]:7.4f} ± {r_m[1]:7.4f} | {n_m[0]:9.4f} ± {n_m[1]:9.4f} | "
            f"{q_m[0]:8.4f} ± {q_m[1]:8.4f} | {o_m[0]:8.4f} ± {o_m[1]:8.4f} | "
            f"{d_m[0]:9.4f} ± {d_m[1]:9.4f}"
        )

    print("\nPaired Friedman tests across wd in {0.0, 0.01, 0.1} (n = 5 seeds):")
    for field in (
        "argmax_ratio",
        "argmax_num_fro",
        "argmax_qk_fro",
        "argmax_ov_fro",
        "argmax_denom_prod",
        "all16_ratio_mean",
        "all16_num_fro_mean",
        "all4_qk_fro_mean",
        "all4_ov_fro_mean",
        "all16_denom_prod_mean",
    ):
        c0 = [x[field] for x in by_wd[0.0]]
        c1 = [x[field] for x in by_wd[0.01]]
        c2 = [x[field] for x in by_wd[0.1]]
        chi2, pval = friedmanchisquare(c0, c1, c2)
        inc = sum(1 for i in range(5) if c0[i] < c1[i] < c2[i])
        dec = sum(1 for i in range(5) if c0[i] > c1[i] > c2[i])
        print(f"  {field:22s}: chi2={chi2:5.2f}, p={pval:.4f}, inc={inc}/5, dec={dec}/5")


def main() -> None:
    in_pkg_root = (Path.cwd() / "pyproject.toml").is_file()
    p = "runs" if in_pkg_root else "induction/runs"
    a = "artifacts" if in_pkg_root else "induction/artifacts"

    parser = argparse.ArgumentParser(description="CPU-only consultant review diagnostics T1-T4.")
    parser.add_argument("--sweep-root", type=str, default=f"{p}/overnight_sweep")
    parser.add_argument("--ext-b-root", type=str, default=f"{p}/ext_b_2l_attn_ln")
    parser.add_argument("--ext-c-root", type=str, default=f"{p}/ext_c_kappa3_25k")
    parser.add_argument("--data-dir", type=str, default=f"{a}/data")
    args = parser.parse_args()

    t0 = time.perf_counter()
    assert not torch.cuda.is_available(), "Must run on CPU only (set CUDA_VISIBLE_DEVICES='')"

    sweep_root = Path(args.sweep_root)
    ext_b_root = Path(args.ext_b_root)
    ext_c_root = Path(args.ext_c_root)
    data_dir = Path(args.data_dir)

    p_base = make_base_transition_matrix(vocab_size=VOCAB_SIZE, data_seed=DATA_SEED)
    probe_tokens = make_prefix_matching_probe(
        vocab_size=VOCAB_SIZE, n_probe=N_PROBE, half_len=PROBE_HALF_LEN, probe_seed=PROBE_SEED
    )
    val_sets = {
        k: make_validation_set(p_base=p_base, kappa=k, n_val=N_VAL, seq_len=SEQ_LEN, val_seed=VAL_SEED)
        for k in (0.3, 1.0, 3.0)
    }

    run_t1(sweep_root=sweep_root, ext_c_root=ext_c_root, data_dir=data_dir, val_sets=val_sets)
    run_t2_and_t3(
        sweep_root=sweep_root,
        ext_b_root=ext_b_root,
        data_dir=data_dir,
        val_tokens_k1=val_sets[1.0].tokens,
        probe_tokens=probe_tokens,
    )
    run_t4(sweep_root=sweep_root, data_dir=data_dir)
    print(f"\nCompleted T1-T4 on CPU in {time.perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()

"""Resumable overnight runner for the full 35-run Induction Heads sweep matrix."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Support `python -m induction.src.overnight_sweep`, `python -m src.overnight_sweep`, and direct script execution
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
    from src.train import run_training
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
    from .train import run_training


@dataclass(frozen=True)
class SweepJob:
    """Single run specification in the 35-run sweep matrix."""

    axis: str
    arch: str
    kappa: float
    wd: float
    seed: int

    @property
    def run_name(self) -> str:
        return f"{self.arch}_kappa{self.kappa:g}_wd{self.wd:g}_seed{self.seed}"


def build_sweep_matrix() -> List[SweepJob]:
    """Construct the exact 35-run matrix from induction/EXPERIMENT_SPEC.md."""
    seeds = (0, 1, 2, 3, 4)
    jobs: List[SweepJob] = []

    # Axis 1: Architecture & Depth (kappa = 1.0, wd = 0.01) -> 15 runs
    for arch in ("1L-attn", "2L-attn", "2L-full"):
        for seed in seeds:
            jobs.append(SweepJob(axis="axis1_arch", arch=arch, kappa=1.0, wd=0.01, seed=seed))

    # Axis 2: Prior Concentration (2L-attn, wd = 0.01; kappa=1.0 reused from Axis 1) -> 10 runs
    for kappa in (0.3, 3.0):
        for seed in seeds:
            jobs.append(SweepJob(axis="axis2_kappa", arch="2L-attn", kappa=kappa, wd=0.01, seed=seed))

    # Axis 3: Weight Decay (2L-attn, kappa = 1.0; wd=0.01 reused from Axis 1) -> 10 runs
    for wd in (0.0, 0.1):
        for seed in seeds:
            jobs.append(SweepJob(axis="axis3_wd", arch="2L-attn", kappa=1.0, wd=wd, seed=seed))

    # Sanity guard: ensure all 35 jobs have unique run names
    names = [j.run_name for j in jobs]
    assert len(jobs) == 35, f"Expected 35 jobs, got {len(jobs)}"
    assert len(set(names)) == 35, "Duplicate run_name detected in sweep matrix"
    return jobs


def format_duration(seconds: float) -> str:
    """Format seconds as a compact human-readable duration (e.g. '1h 12m 05s')."""
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs}h {mins:02d}m {secs:02d}s"
    if mins > 0:
        return f"{mins}m {secs:02d}s"
    return f"{seconds:.1f}s"


def is_run_completed(run_dir: Path) -> Tuple[bool, Optional[dict]]:
    """Return (True, summary_dict) if run_dir/summary.json exists and parses as valid JSON."""
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return False, None
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "val_loss" in data:
            return True, data
    except Exception:
        return False, None
    return False, None


def write_sweep_summary_csv(out_root: Path, jobs: List[SweepJob]) -> None:
    """Aggregate all completed summary.json files into out_root/sweep_summary.csv."""
    fieldnames = [
        "run_name",
        "axis",
        "arch",
        "kappa",
        "wd",
        "seed",
        "steps",
        "val_loss",
        "val_loss_early",
        "val_loss_late",
        "icl_score",
        "bigram_val_loss_late",
        "bayes_val_loss_late",
        "bayes_icl_score",
        "max_prev_token_score_L0",
        "max_prefix_match_score",
        "max_k_composition_score",
        "weight_l2_norm",
        "ablation_icl_drop_top_head",
        "ablation_icl_drop_control",
        "total_wall_sec",
        "sec_per_1k_steps",
        "peak_vram_mb",
    ]
    rows = []
    for job in jobs:
        done, summary = is_run_completed(out_root / job.run_name)
        if not done or summary is None:
            continue
        bl = summary.get("baselines", {})
        timing = summary.get("timing", {})
        cfg = summary.get("config", {})
        rows.append(
            {
                "run_name": job.run_name,
                "axis": job.axis,
                "arch": job.arch,
                "kappa": job.kappa,
                "wd": job.wd,
                "seed": job.seed,
                "steps": cfg.get("steps"),
                "val_loss": summary.get("val_loss"),
                "val_loss_early": summary.get("val_loss_early"),
                "val_loss_late": summary.get("val_loss_late"),
                "icl_score": summary.get("icl_score"),
                "bigram_val_loss_late": bl.get("bigram_val_loss_late"),
                "bayes_val_loss_late": bl.get("bayes_val_loss_late"),
                "bayes_icl_score": bl.get("bayes_icl_score"),
                "max_prev_token_score_L0": summary.get("max_prev_token_score_L0"),
                "max_prefix_match_score": summary.get("max_prefix_match_score"),
                "max_k_composition_score": summary.get("max_k_composition_score"),
                "weight_l2_norm": summary.get("weight_l2_norm"),
                "ablation_icl_drop_top_head": summary.get("ablation_icl_drop_top_head"),
                "ablation_icl_drop_control": summary.get("ablation_icl_drop_control"),
                "total_wall_sec": timing.get("total_wall_sec"),
                "sec_per_1k_steps": timing.get("sec_per_1k_steps"),
                "peak_vram_mb": timing.get("peak_vram_mb"),
            }
        )

    csv_path = out_root / "sweep_summary.csv"
    csv_tmp = out_root / "sweep_summary.csv.tmp"
    with csv_tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    csv_tmp.replace(csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the resumable 35-run Induction Heads overnight sweep."
    )
    default_root = (
        "runs/overnight_sweep"
        if (Path.cwd() / "pyproject.toml").is_file()
        else "induction/runs/overnight_sweep"
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=default_root,
        help="Root output directory for all 35 run folders.",
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print all 35 jobs and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    jobs = build_sweep_matrix()

    already_done = sum(1 for j in jobs if is_run_completed(out_root / j.run_name)[0])
    print(
        f"=== Induction Heads Overnight Sweep ===\n"
        f"Total jobs: {len(jobs)} | Already completed: {already_done} | "
        f"Remaining: {len(jobs) - already_done}\n"
        f"Steps/run: {args.steps:,} | Out root: {out_root.resolve()}",
        flush=True,
    )

    if args.dry_run:
        for idx, job in enumerate(jobs, start=1):
            done, _ = is_run_completed(out_root / job.run_name)
            status = "SKIP (done)" if done else "PENDING"
            print(f"  [{idx:02d}/{len(jobs)}] {status:11s} {job.run_name} ({job.axis})")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    # Precompute shared fixed deterministic data assets once (P_base, probe, val sets per kappa)
    print("Precomputing fixed base transition matrix, probe, and validation sets...", flush=True)
    p_base = make_base_transition_matrix(vocab_size=VOCAB_SIZE, data_seed=DATA_SEED)
    probe_tokens = make_prefix_matching_probe(
        vocab_size=VOCAB_SIZE,
        n_probe=N_PROBE,
        half_len=PROBE_HALF_LEN,
        probe_seed=PROBE_SEED,
    )
    val_sets: Dict[float, ValidationSet] = {}
    for kappa in sorted({j.kappa for j in jobs}):
        vs = make_validation_set(
            p_base=p_base,
            kappa=kappa,
            n_val=N_VAL,
            seq_len=SEQ_LEN,
            val_seed=VAL_SEED,
        )
        val_sets[kappa] = vs
        bl = vs.baselines
        print(
            f"  kappa={kappa:<4g}: Bigram late={bl['bigram_val_loss_late']:.4f} | "
            f"Bayes late={bl['bayes_val_loss_late']:.4f} | "
            f"Bayes ICL={bl['bayes_icl_score']:.4f}",
            flush=True,
        )

    sweep_start = time.perf_counter()
    executed_times: List[float] = []
    n_skipped = 0
    n_succeeded = 0
    n_failed = 0
    error_log = out_root / "sweep_errors.jsonl"

    for idx, job in enumerate(jobs, start=1):
        run_dir = out_root / job.run_name
        done, existing = is_run_completed(run_dir)
        if done and existing is not None:
            n_skipped += 1
            print(
                f"[{idx:02d}/{len(jobs)}] SKIP {job.run_name} "
                f"(val={existing['val_loss']:.4f}, icl={existing['icl_score']:+.4f}, "
                f"pmatch={existing['max_prefix_match_score']:.3f})",
                flush=True,
            )
            continue

        run_t0 = time.perf_counter()
        try:
            summary = run_training(
                arch=job.arch,
                kappa=job.kappa,
                wd=job.wd,
                seed=job.seed,
                steps=args.steps,
                out_dir=run_dir,
                batch_size=args.batch_size,
                lr=args.lr,
                eval_every=args.eval_every,
                device_str=args.device,
                p_base=p_base,
                val_set=val_sets[job.kappa],
                probe_tokens=probe_tokens,
                verbose=False,
            )
            run_sec = time.perf_counter() - run_t0
            executed_times.append(run_sec)
            n_succeeded += 1

            remaining = sum(
                1 for later in jobs[idx:] if not is_run_completed(out_root / later.run_name)[0]
            )
            avg_sec = sum(executed_times) / len(executed_times)
            eta_sec = avg_sec * remaining
            elapsed_total = time.perf_counter() - sweep_start

            abl_top = summary["ablation_icl_drop_top_head"]
            abl_str = f"{abl_top:+.3f}" if abl_top is not None else "N/A"
            print(
                f"[{idx:02d}/{len(jobs)}] DONE {job.run_name} in {format_duration(run_sec)} | "
                f"val={summary['val_loss']:.4f} late={summary['val_loss_late']:.4f} "
                f"icl={summary['icl_score']:+.4f} pmatch={summary['max_prefix_match_score']:.3f} "
                f"abl={abl_str} | elapsed={format_duration(elapsed_total)} "
                f"ETA={format_duration(eta_sec)}",
                flush=True,
            )
            write_sweep_summary_csv(out_root, jobs)

        except Exception as exc:
            run_sec = time.perf_counter() - run_t0
            n_failed += 1
            tb = traceback.format_exc()
            err_record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "job_index": idx,
                "run_name": job.run_name,
                "arch": job.arch,
                "kappa": job.kappa,
                "wd": job.wd,
                "seed": job.seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": tb,
            }
            with error_log.open("a", encoding="utf-8") as ef:
                ef.write(json.dumps(err_record) + "\n")
            print(
                f"[{idx:02d}/{len(jobs)}] ERROR {job.run_name} after {format_duration(run_sec)}: "
                f"{type(exc).__name__}: {exc} (logged to {error_log}; continuing)",
                file=sys.stderr,
                flush=True,
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_sweep_summary_csv(out_root, jobs)
    total_elapsed = time.perf_counter() - sweep_start
    print(
        f"\n=== Sweep Finished in {format_duration(total_elapsed)} ===\n"
        f"Succeeded: {n_succeeded} | Skipped: {n_skipped} | Failed: {n_failed}\n"
        f"Aggregated table: {(out_root / 'sweep_summary.csv').resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()

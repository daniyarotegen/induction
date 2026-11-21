"""Single-run trainer for Project #2: Induction Heads & Emergent ICL."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

# Support `python -m induction.src.train`, `python -m src.train`, and direct script execution
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
        OnlineDirichletMarkovSampler,
        ValidationSet,
        make_base_transition_matrix,
        make_prefix_matching_probe,
        make_validation_set,
        set_seed,
    )
    from src.metrics import evaluate_causal_head_ablation, evaluate_step_metrics
    from src.model import VALID_ARCHS, InductionTransformer, TransformerConfig
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
        OnlineDirichletMarkovSampler,
        ValidationSet,
        make_base_transition_matrix,
        make_prefix_matching_probe,
        make_validation_set,
        set_seed,
    )
    from .metrics import evaluate_causal_head_ablation, evaluate_step_metrics
    from .model import VALID_ARCHS, InductionTransformer, TransformerConfig


def get_git_info(repo_dir: Path) -> Dict[str, Any]:
    """Return current git commit hash and dirty status (best-effort)."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_dir),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=str(repo_dir),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        return {"commit": commit, "dirty": bool(status)}
    except Exception:
        return {"commit": "unknown", "dirty": None}


def build_optimizer(model: InductionTransformer, lr: float, wd: float) -> torch.optim.AdamW:
    """Create AdamW with betas=(0.9, 0.98) applying weight decay to >=2D tensors only."""
    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups = [{"params": decay_params, "weight_decay": float(wd)}]
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=float(lr), betas=(0.9, 0.98))


def get_lr_for_step(step: int, total_steps: int, base_lr: float, schedule: str) -> float:
    """Return learning rate for 1-indexed optimization step."""
    if schedule == "constant":
        return base_lr
    if schedule == "cosine":
        progress = min(1.0, max(0.0, (step - 1) / max(1, total_steps - 1)))
        return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown lr_schedule '{schedule}'. Expected 'constant' or 'cosine'.")


def run_training(
    arch: str,
    kappa: float,
    wd: float,
    seed: int,
    steps: int,
    out_dir: Path | str,
    batch_size: int = 64,
    lr: float = 1e-3,
    lr_schedule: str = "constant",
    eval_every: int = 100,
    device_str: Optional[str] = None,
    p_base: Optional[torch.Tensor] = None,
    val_set: Optional[ValidationSet] = None,
    probe_tokens: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute one deterministic training run and write metrics.jsonl, final.pt, summary.json."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # 1. Deterministic fixed data assets (isolated RNG, independent of training seed)
    if p_base is None:
        p_base = make_base_transition_matrix(vocab_size=VOCAB_SIZE, data_seed=DATA_SEED)
    if val_set is None:
        val_set = make_validation_set(
            p_base=p_base,
            kappa=kappa,
            n_val=N_VAL,
            seq_len=SEQ_LEN,
            val_seed=VAL_SEED,
        )
    if probe_tokens is None:
        probe_tokens = make_prefix_matching_probe(
            vocab_size=VOCAB_SIZE,
            n_probe=N_PROBE,
            half_len=PROBE_HALF_LEN,
            probe_seed=PROBE_SEED,
        )

    # 2. Seed everything for model initialization & online batch stream
    set_seed(seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    cfg = TransformerConfig(arch=arch, vocab_size=VOCAB_SIZE, max_len=SEQ_LEN)
    model = InductionTransformer(cfg).to(device)
    optimizer = build_optimizer(model, lr=lr, wd=wd)

    sampler = OnlineDirichletMarkovSampler(
        p_base=p_base,
        kappa=kappa,
        batch_size=batch_size,
        seq_len=SEQ_LEN,
        device=device,
        chunk_batches=16,
    )

    val_tokens_dev = val_set.tokens.to(device)
    probe_tokens_dev = probe_tokens.to(device)

    metrics_file = out_path / "metrics.jsonl"
    start_time = time.perf_counter()
    last_eval_metrics: Dict[str, Any] = {}

    if verbose:
        bl = val_set.baselines
        print(
            f"[START] arch={arch} kappa={kappa} wd={wd} seed={seed} steps={steps} "
            f"params={model.count_parameters():,} device={device} | "
            f"Bigram={bl['bigram_val_loss']:.3f} (late {bl['bigram_val_loss_late']:.3f}) | "
            f"Bayes={bl['bayes_val_loss']:.3f} (late {bl['bayes_val_loss_late']:.3f}, "
            f"ICL {bl['bayes_icl_score']:.3f})",
            flush=True,
        )

    with metrics_file.open("w", encoding="utf-8") as mf:
        # Step 0 evaluation before any gradient updates
        step0 = evaluate_step_metrics(
            model=model,
            val_tokens=val_tokens_dev,
            probe_tokens=probe_tokens_dev,
        )
        rec0 = {
            "step": 0,
            "tokens_seen": 0,
            "train_loss": None,
            "lr": lr,
            "elapsed_sec": time.perf_counter() - start_time,
            **step0,
        }
        mf.write(json.dumps(rec0) + "\n")
        mf.flush()
        last_eval_metrics = rec0

        model.train()
        running_train_loss = 0.0
        running_train_count = 0

        for step in range(1, steps + 1):
            cur_lr = get_lr_for_step(step, steps, lr, lr_schedule)
            if lr_schedule != "constant":
                for group in optimizer.param_groups:
                    group["lr"] = cur_lr

            batch = sampler.next_batch()                              # (B, T)
            logits = model(batch, return_attn=False)                  # (B, T, V)

            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, cfg.vocab_size),        # (B * (T - 1), V)
                batch[:, 1:].reshape(-1),                             # (B * (T - 1),)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_train_loss += float(loss.item())
            running_train_count += 1

            should_eval = (step % eval_every == 0) or (step == steps)
            if should_eval:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                avg_train_loss = running_train_loss / max(1, running_train_count)
                running_train_loss = 0.0
                running_train_count = 0

                eval_metrics = evaluate_step_metrics(
                    model=model,
                    val_tokens=val_tokens_dev,
                    probe_tokens=probe_tokens_dev,
                )
                elapsed = time.perf_counter() - start_time
                record = {
                    "step": step,
                    "tokens_seen": step * batch_size * SEQ_LEN,
                    "train_loss": avg_train_loss,
                    "lr": cur_lr,
                    "elapsed_sec": elapsed,
                    **eval_metrics,
                }
                mf.write(json.dumps(record) + "\n")
                mf.flush()
                last_eval_metrics = record

                if verbose and (step % max(eval_every * 5, 500) == 0 or step == steps):
                    k_comp_str = (
                        f"{eval_metrics['max_k_composition_score']:.3f}"
                        if eval_metrics["max_k_composition_score"] is not None
                        else "N/A"
                    )
                    print(
                        f"  step={step:5d}/{steps} | train={avg_train_loss:.4f} "
                        f"val={eval_metrics['val_loss']:.4f} "
                        f"early={eval_metrics['val_loss_early']:.4f} "
                        f"late={eval_metrics['val_loss_late']:.4f} "
                        f"icl={eval_metrics['icl_score']:+.4f} | "
                        f"prevL0={eval_metrics['max_prev_token_score_L0']:.3f} "
                        f"pmatch={eval_metrics['max_prefix_match_score']:.3f} "
                        f"kcomp={k_comp_str} | {elapsed:.1f}s",
                        flush=True,
                    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_wall_sec = time.perf_counter() - start_time

    # 3. End-of-run causal head ablation on validation set
    ablation = evaluate_causal_head_ablation(
        model=model,
        val_tokens=val_tokens_dev,
        probe_tokens=probe_tokens_dev,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    else:
        peak_vram_mb = 0.0

    total_wall_sec = time.perf_counter() - start_time
    sec_per_1k = (train_wall_sec / max(1, steps)) * 1000.0
    tokens_per_sec = (steps * batch_size * SEQ_LEN) / max(1e-6, train_wall_sec)

    # 4. Save compact final checkpoint (model weights + metadata only, ~0.5-1.5 MB)
    ckpt_path = out_path / "final.pt"
    ckpt_tmp = out_path / "final.pt.tmp"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": cfg.to_dict(),
            "step": steps,
            "seed": seed,
            "kappa": float(kappa),
            "wd": float(wd),
        },
        ckpt_tmp,
    )
    os.replace(ckpt_tmp, ckpt_path)

    # 5. Assemble and atomically write summary.json
    git_info = get_git_info(Path(__file__).resolve().parent)
    gpu_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda" and torch.cuda.is_available()
        else None
    )
    summary: Dict[str, Any] = {
        "run_name": out_path.name,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "git_commit": git_info["commit"],
        "config": {
            "arch": arch,
            "kappa": float(kappa),
            "wd": float(wd),
            "seed": int(seed),
            "steps": int(steps),
            "batch_size": int(batch_size),
            "seq_len": SEQ_LEN,
            "vocab_size": VOCAB_SIZE,
            "lr": float(lr),
            "lr_schedule": lr_schedule,
            "betas": [0.9, 0.98],
            "eval_every": int(eval_every),
            "device": str(device),
            "data_seed": DATA_SEED,
            "val_seed": VAL_SEED,
            "probe_seed": PROBE_SEED,
            "n_params": model.count_parameters(),
            "model_config": cfg.to_dict(),
        },
        "git": git_info,
        "baselines": val_set.baselines,
        "final_metrics": last_eval_metrics,
        "ablation": ablation,
        # Top-level convenience fields for sweep aggregation
        "val_loss": last_eval_metrics["val_loss"],
        "val_loss_early": last_eval_metrics["val_loss_early"],
        "val_loss_late": last_eval_metrics["val_loss_late"],
        "icl_score": last_eval_metrics["icl_score"],
        "max_prev_token_score_L0": last_eval_metrics["max_prev_token_score_L0"],
        "max_prefix_match_score": last_eval_metrics["max_prefix_match_score"],
        "max_k_composition_score": last_eval_metrics["max_k_composition_score"],
        "weight_l2_norm": last_eval_metrics["weight_l2_norm"],
        "ablation_icl_drop_top_head": ablation["ablation_icl_drop_top_head"],
        "ablation_icl_drop_control": ablation["ablation_icl_drop_control"],
        "timing": {
            "train_wall_sec": train_wall_sec,
            "total_wall_sec": total_wall_sec,
            "sec_per_1k_steps": sec_per_1k,
            "tokens_per_sec": tokens_per_sec,
            "peak_vram_mb": peak_vram_mb,
        },
    }

    summary_path = out_path / "summary.json"
    summary_tmp = out_path / "summary.json.tmp"
    with summary_tmp.open("w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)
    os.replace(summary_tmp, summary_path)

    if verbose:
        abl_str = (
            f"top_head=L1H{ablation['ablation_top_head_idx']} "
            f"icl_drop={ablation['ablation_icl_drop_top_head']:+.4f} "
            f"ctrl_drop={ablation['ablation_icl_drop_control']:+.4f}"
            if ablation["ablation_icl_drop_top_head"] is not None
            else "N/A (1L)"
        )
        print(
            f"[DONE]  {out_path.name} in {total_wall_sec:.1f}s "
            f"({sec_per_1k:.2f} s/1k steps, peak VRAM {peak_vram_mb:.1f} MB) | "
            f"ablation: {abl_str}",
            flush=True,
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a single Induction Heads Transformer on Zipfian Dirichlet-Markov chains."
    )
    parser.add_argument("--arch", type=str, required=True, choices=VALID_ARCHS)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", type=str, default="constant", choices=("constant", "cosine"))
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        arch=args.arch,
        kappa=args.kappa,
        wd=args.wd,
        seed=args.seed,
        steps=args.steps,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        eval_every=args.eval_every,
        device_str=args.device,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

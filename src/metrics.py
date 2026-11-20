"""Mechanistic circuit progress measures and causal head-ablation evaluator."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from induction.src.data import (
    EARLY_POS_END,
    EARLY_POS_START,
    LATE_POS_END,
    LATE_POS_START,
    PROBE_HALF_LEN,
)
from induction.src.model import InductionTransformer


def compute_prev_token_score(attn: torch.Tensor) -> torch.Tensor:
    """Compute per-head previous-token attention score: mean weight on (i, i - 1).

    Args:
        attn: Attention probability tensor of shape (B, H, T, T).

    Returns:
        scores: Tensor of shape (H,) with mean attention on (i, i - 1) for i in 1..T-1.
    """
    if attn.ndim != 4:
        raise ValueError(f"Expected 4D attention tensor (B, H, T, T), got shape {tuple(attn.shape)}")
    seq_len = attn.shape[-1]
    if seq_len < 2:
        raise ValueError(f"Sequence length must be >= 2 for previous-token score, got {seq_len}")

    q_idx = torch.arange(1, seq_len, device=attn.device)              # (T - 1,)
    k_idx = q_idx - 1                                                 # (T - 1,)
    # Extract sub-diagonal (i, i - 1) across batch and heads: (B, H, T - 1)
    subdiag = attn[:, :, q_idx, k_idx]
    return subdiag.mean(dim=(0, 2))                                   # (H,)


def compute_prefix_matching_score(
    attn: torch.Tensor,
    half_len: int = PROBE_HALF_LEN,
) -> torch.Tensor:
    """Compute Olsson et al. per-head prefix-matching score on a repeated-token probe.

    For probe sequences [r_0..r_{L-1}, r_0..r_{L-1}] of length 2L, measures the mean
    attention from second-half query position t in [L, 2L - 2] to first-half key
    position k = t - L + 1 in [1, L - 1] (the token immediately following the prior
    occurrence of r_{t - L}).

    Args:
        attn: Attention probability tensor of shape (B, H, 2L, 2L).
        half_len: Length L of each repeated half (default 64 for length-128 probe).

    Returns:
        scores: Tensor of shape (H,) with mean prefix-matching attention weight.
    """
    if attn.ndim != 4:
        raise ValueError(f"Expected 4D attention tensor (B, H, T, T), got shape {tuple(attn.shape)}")
    seq_len = attn.shape[-1]
    if seq_len != 2 * half_len:
        raise ValueError(f"Expected probe length 2*half_len={2 * half_len}, got {seq_len}")

    # Query positions in second half whose induction target lies strictly in first half
    q_idx = torch.arange(half_len, 2 * half_len - 1, device=attn.device)  # (L - 1,) = 64..126
    k_idx = q_idx - half_len + 1                                          # (L - 1,) = 1..63
    matched = attn[:, :, q_idx, k_idx]                                    # (B, H, L - 1)
    return matched.mean(dim=(0, 2))                                       # (H,)


@torch.no_grad()
def compute_k_composition_matrix(model: InductionTransformer) -> Optional[torch.Tensor]:
    """Compute the (H0, H1) K-composition score matrix for 2-layer models.

    For every Layer 0 head h0 and Layer 1 head h1:
        K_comp(h0, h1) = ||W_QK^(1, h1) @ W_OV^(0, h0)||_F /
                         (||W_QK^(1, h1)||_F * ||W_OV^(0, h0)||_F)

    Returns None for 1-layer models.
    """
    if model.cfg.n_layers < 2:
        return None

    w_ov_0 = model.get_W_OV(0)                                        # (H, D, D)
    w_qk_1 = model.get_W_QK(1)                                        # (H, D, D)
    n_heads = model.cfg.n_heads

    ov_norms = torch.linalg.matrix_norm(w_ov_0, ord="fro")            # (H,)
    qk_norms = torch.linalg.matrix_norm(w_qk_1, ord="fro")            # (H,)

    # Pairwise product M[h0, h1] = W_QK^(1, h1) @ W_OV^(0, h0): (H0, H1, D, D)
    scores = torch.empty((n_heads, n_heads), device=w_ov_0.device, dtype=w_ov_0.dtype)
    for h0 in range(n_heads):
        for h1 in range(n_heads):
            prod = w_qk_1[h1] @ w_ov_0[h0]                            # (D, D)
            num = torch.linalg.matrix_norm(prod, ord="fro")
            den = qk_norms[h1] * ov_norms[h0] + 1e-12
            scores[h0, h1] = num / den
    return scores                                                     # (H0, H1)


@torch.no_grad()
def compute_weight_l2_norm(model: InductionTransformer) -> float:
    """Compute global Euclidean L2 norm across all model parameters."""
    sq_sum = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float64)
    for p in model.parameters():
        sq_sum += p.detach().to(torch.float64).pow(2).sum()
    return float(torch.sqrt(sq_sum).item())


@torch.no_grad()
def evaluate_validation_loss(
    model: InductionTransformer,
    val_tokens: torch.Tensor,
    eval_batch_size: int = 128,
    ablate_heads: Optional[Iterable[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """Evaluate exact total, early (1..31), late (192..254) NLL and ICL score on val_tokens."""
    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    n_val, seq_len = val_tokens.shape
    n_pred = seq_len - 1
    early_end = min(EARLY_POS_END, n_pred)
    late_start = min(LATE_POS_START, n_pred)
    late_end = min(LATE_POS_END, n_pred)

    total_sum = 0.0
    total_count = 0
    early_sum = 0.0
    early_count = 0
    late_sum = 0.0
    late_count = 0

    for start in range(0, n_val, eval_batch_size):
        batch = val_tokens[start : start + eval_batch_size].to(device, non_blocking=True)  # (b, T)
        logits = model(batch, return_attn=False, ablate_heads=ablate_heads)  # (b, T, V)

        # Next-token cross-entropy per position t in 0..T-2 predicting x_{t+1}
        pred_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])  # (b * (T - 1), V)
        targets = batch[:, 1:].reshape(-1)                             # (b * (T - 1),)
        loss_per_pos = F.cross_entropy(pred_logits, targets, reduction="none").view(
            batch.shape[0], n_pred
        )                                                              # (b, T - 1)

        total_sum += float(loss_per_pos.sum().item())
        total_count += loss_per_pos.numel()

        early_slice = loss_per_pos[:, EARLY_POS_START:early_end]
        early_sum += float(early_slice.sum().item())
        early_count += early_slice.numel()

        late_slice = loss_per_pos[:, late_start:late_end]
        late_sum += float(late_slice.sum().item())
        late_count += late_slice.numel()

    if was_training:
        model.train()

    val_loss = total_sum / max(1, total_count)
    val_loss_early = early_sum / max(1, early_count)
    val_loss_late = late_sum / max(1, late_count)
    return {
        "val_loss": val_loss,
        "val_loss_early": val_loss_early,
        "val_loss_late": val_loss_late,
        "icl_score": val_loss_early - val_loss_late,
    }


@torch.no_grad()
def evaluate_step_metrics(
    model: InductionTransformer,
    val_tokens: torch.Tensor,
    probe_tokens: torch.Tensor,
    eval_batch_size: int = 128,
    probe_half_len: int = PROBE_HALF_LEN,
) -> Dict[str, Any]:
    """Compute all spec evaluation metrics at a single training step."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    # 1. Validation losses & ICL score
    loss_metrics = evaluate_validation_loss(
        model=model,
        val_tokens=val_tokens,
        eval_batch_size=eval_batch_size,
    )

    # 2. Attention-based circuit metrics on the repeated-token probe (no consecutive duplicates)
    probe_dev = probe_tokens.to(device, non_blocking=True)            # (N_probe, 2L)
    _, attn_patterns = model(probe_dev, return_attn=True)             # list of (N_probe, H, 2L, 2L)

    prev_scores_l0 = compute_prev_token_score(attn_patterns[0])       # (H,)
    max_prev_token_score_l0 = float(prev_scores_l0.max().item())

    per_head_prefix_match: List[List[float]] = []
    max_prefix_match_score = -1.0
    for attn_l in attn_patterns:
        pm_l = compute_prefix_matching_score(attn_l, half_len=probe_half_len)  # (H,)
        pm_list = [float(x) for x in pm_l.cpu().tolist()]
        per_head_prefix_match.append(pm_list)
        max_prefix_match_score = max(max_prefix_match_score, max(pm_list))

    # 3. K-composition score (2L models only)
    k_comp_mat = compute_k_composition_matrix(model)
    max_k_comp = float(k_comp_mat.max().item()) if k_comp_mat is not None else None

    # 4. Global parameter L2 norm
    weight_l2_norm = compute_weight_l2_norm(model)

    if was_training:
        model.train()

    return {
        **loss_metrics,
        "max_prev_token_score_L0": max_prev_token_score_l0,
        "per_head_prev_token_L0": [float(x) for x in prev_scores_l0.cpu().tolist()],
        "max_prefix_match_score": max_prefix_match_score,
        "per_head_prefix_match": per_head_prefix_match,
        "max_k_composition_score": max_k_comp,
        "weight_l2_norm": weight_l2_norm,
    }


@torch.no_grad()
def evaluate_causal_head_ablation(
    model: InductionTransformer,
    val_tokens: torch.Tensor,
    probe_tokens: torch.Tensor,
    eval_batch_size: int = 128,
    probe_half_len: int = PROBE_HALF_LEN,
) -> Dict[str, Any]:
    """Zero-ablate the top Layer 1 prefix-matching head vs. control Layer 1 heads on val_tokens.

    For 1-layer models (`n_layers < 2`), returns None for Layer 1 ablation metrics.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    base_metrics = evaluate_validation_loss(
        model=model,
        val_tokens=val_tokens,
        eval_batch_size=eval_batch_size,
    )

    if model.cfg.n_layers < 2:
        if was_training:
            model.train()
        return {
            "ablation_base_icl_score": base_metrics["icl_score"],
            "ablation_top_head_idx": None,
            "ablation_top_head_prefix_match": None,
            "ablation_icl_drop_top_head": None,
            "ablation_icl_drop_control": None,
            "ablation_per_head_L1": None,
        }

    # Identify the single Layer 1 head with the highest prefix-matching score on the probe
    probe_dev = probe_tokens.to(device, non_blocking=True)
    _, attn_patterns = model(probe_dev, return_attn=True)
    l1_prefix_scores = compute_prefix_matching_score(attn_patterns[1], half_len=probe_half_len)  # (H,)
    top_head_idx = int(torch.argmax(l1_prefix_scores).item())
    top_head_pm = float(l1_prefix_scores[top_head_idx].item())

    per_head_results: Dict[str, Dict[str, float]] = {}
    control_drops: List[float] = []
    top_drop = 0.0

    for h in range(model.cfg.n_heads):
        abl_metrics = evaluate_validation_loss(
            model=model,
            val_tokens=val_tokens,
            eval_batch_size=eval_batch_size,
            ablate_heads=[(1, h)],
        )
        icl_drop = base_metrics["icl_score"] - abl_metrics["icl_score"]
        loss_increase = abl_metrics["val_loss"] - base_metrics["val_loss"]
        late_loss_increase = abl_metrics["val_loss_late"] - base_metrics["val_loss_late"]

        per_head_results[f"L1H{h}"] = {
            "prefix_match_score": float(l1_prefix_scores[h].item()),
            "ablated_val_loss": abl_metrics["val_loss"],
            "ablated_val_loss_late": abl_metrics["val_loss_late"],
            "ablated_icl_score": abl_metrics["icl_score"],
            "icl_drop": icl_drop,
            "val_loss_increase": loss_increase,
            "val_loss_late_increase": late_loss_increase,
        }

        if h == top_head_idx:
            top_drop = icl_drop
        else:
            control_drops.append(icl_drop)

    control_mean_drop = sum(control_drops) / max(1, len(control_drops))

    if was_training:
        model.train()

    return {
        "ablation_base_icl_score": base_metrics["icl_score"],
        "ablation_top_head_idx": top_head_idx,
        "ablation_top_head_prefix_match": top_head_pm,
        "ablation_icl_drop_top_head": top_drop,
        "ablation_icl_drop_control": control_mean_drop,
        "ablation_per_head_L1": per_head_results,
    }

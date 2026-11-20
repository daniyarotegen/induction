"""Synthetic Zipfian Dirichlet-Markov data generator, analytical baselines, and probes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

VOCAB_SIZE = 64
SEQ_LEN = 256
ZIPF_EXPONENT = 0.8
DATA_SEED = 12345
VAL_SEED = 99999
PROBE_SEED = 77777
N_VAL = 512
N_PROBE = 128
PROBE_HALF_LEN = 64

# 0-indexed next-token prediction positions t in [0, SEQ_LEN - 2] = [0, 254],
# where position t predicts x_{t+1} given x_{0..t}.
EARLY_POS_START = 1
EARLY_POS_END = 32   # inclusive 1..31
LATE_POS_START = 192
LATE_POS_END = 255   # inclusive 192..254


def set_seed(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch CPU/CUDA random seeds for deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_base_transition_matrix(
    vocab_size: int = VOCAB_SIZE,
    zipf_exponent: float = ZIPF_EXPONENT,
    data_seed: int = DATA_SEED,
) -> torch.Tensor:
    """Build fixed global base transition matrix P_base of shape (V, V) in float64.

    Each row is an independent deterministic random permutation of the normalized
    Zipfian distribution rank^(-zipf_exponent) for rank = 1..V.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(data_seed)
        ranks = torch.arange(1, vocab_size + 1, dtype=torch.float64)  # (V,)
        zipf = ranks.pow(-zipf_exponent)                              # (V,)
        zipf = zipf / zipf.sum()                                      # (V,)

        p_base = torch.empty((vocab_size, vocab_size), dtype=torch.float64)  # (V, V)
        for row in range(vocab_size):
            perm = torch.randperm(vocab_size)                         # (V,)
            p_base[row] = zipf[perm]
    return p_base


def sample_dirichlet_markov_batch(
    p_base: torch.Tensor,
    kappa: float,
    batch_size: int,
    seq_len: int = SEQ_LEN,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample fresh sequences from per-document Dirichlet(kappa * P_base) Markov chains.

    Uses the exact log-space Gamma boost identity:
        Gamma(alpha, 1) = Gamma(alpha + 1, 1) * U^(1 / alpha),  U ~ Uniform(0, 1)
    followed by softmax in log-space, preventing float32 underflow when kappa * P_base
    is small (e.g. kappa = 0.3).

    Args:
        p_base: Base transition matrix of shape (V, V).
        kappa: Dirichlet prior concentration scalar (> 0).
        batch_size: Number of independent sequences B.
        seq_len: Sequence length T.
        device: Target torch device.
        dtype: Floating dtype for Gamma/CDF math.

    Returns:
        tokens: Long tensor of shape (B, T) with values in [0, V - 1].
    """
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}")

    vocab_size = p_base.shape[0]
    device = torch.device(device)

    # alpha: (B, V, V)
    alpha = (p_base.to(device=device, dtype=dtype) * float(kappa)).unsqueeze(0).expand(
        batch_size, vocab_size, vocab_size
    )

    # Exact log-space Dirichlet sampling via Gamma(alpha + 1, 1) * U^(1 / alpha)
    g_boost = torch._standard_gamma(alpha + 1.0).clamp_(min=1e-30)    # (B, V, V)
    u_boost = torch.rand(alpha.shape, device=device, dtype=dtype).clamp_(min=1e-30)
    log_g = torch.log(g_boost) + torch.log(u_boost) / alpha           # (B, V, V)
    trans_probs = torch.softmax(log_g, dim=-1)                        # (B, V, V)

    # Precompute row CDFs for fast vectorized inverse-CDF Markov rollout
    cdf = trans_probs.cumsum(dim=-1)                                  # (B, V, V)
    cdf[:, :, -1] = 1.0

    u_step = torch.rand((batch_size, seq_len - 1), device=device, dtype=dtype)  # (B, T - 1)
    tokens = torch.empty((batch_size, seq_len), device=device, dtype=torch.long)  # (B, T)
    cur = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=torch.long)  # (B,)
    tokens[:, 0] = cur

    batch_idx = torch.arange(batch_size, device=device)               # (B,)
    for t in range(seq_len - 1):
        row_cdf = cdf[batch_idx, cur]                                 # (B, V)
        # Count how many CDF thresholds are <= u_step[:, t]; clamp handles u == 1.0
        cur = (u_step[:, t : t + 1] >= row_cdf).sum(dim=-1).clamp_(max=vocab_size - 1)  # (B,)
        tokens[:, t + 1] = cur

    return tokens


class OnlineDirichletMarkovSampler:
    """Fast GPU-resident online sampler yielding fresh batches without sequence reuse.

    Samples `chunk_batches * batch_size` sequences per GPU rollout to amortize the
    255-step Python loop overhead (~0.4 ms/batch on RTX 5070), while strictly never
    reusing any sequence.
    """

    def __init__(
        self,
        p_base: torch.Tensor,
        kappa: float,
        batch_size: int = 64,
        seq_len: int = SEQ_LEN,
        device: torch.device | str = "cuda",
        chunk_batches: int = 16,
    ) -> None:
        self.p_base = p_base.to(device=device, dtype=torch.float32)   # (V, V)
        self.kappa = float(kappa)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.device = torch.device(device)
        self.chunk_batches = max(1, int(chunk_batches))
        self._buffer: torch.Tensor | None = None
        self._cursor: int = 0

    def next_batch(self) -> torch.Tensor:
        """Return a fresh batch of shape (batch_size, seq_len) on self.device."""
        if self._buffer is None or self._cursor >= self._buffer.shape[0]:
            total = self.batch_size * self.chunk_batches
            self._buffer = sample_dirichlet_markov_batch(
                p_base=self.p_base,
                kappa=self.kappa,
                batch_size=total,
                seq_len=self.seq_len,
                device=self.device,
                dtype=torch.float32,
            )  # (chunk_batches * B, T)
            self._cursor = 0

        batch = self._buffer[self._cursor : self._cursor + self.batch_size]  # (B, T)
        self._cursor += self.batch_size
        return batch


def compute_analytical_baselines(
    val_tokens: torch.Tensor,
    p_base: torch.Tensor,
    kappa: float,
) -> Dict[str, float]:
    """Compute exact Global Bigram and Bayes-Optimal ICL NLL baselines on val_tokens.

    Args:
        val_tokens: Long tensor of shape (N, T) on CPU or GPU.
        p_base: Base transition matrix of shape (V, V).
        kappa: Dirichlet prior concentration.

    Returns:
        Dictionary with exact total, early (1..31), late (192..254), and ICL scores
        for both the Global Bigram and Exact Bayes-Optimal Dirichlet-Multinomial models.
    """
    tokens_cpu = val_tokens.detach().to("cpu", dtype=torch.long)      # (N, T)
    p_base_f64 = p_base.detach().to("cpu", dtype=torch.float64)       # (V, V)
    n_seq, seq_len = tokens_cpu.shape
    vocab_size = p_base_f64.shape[0]
    n_pred = seq_len - 1

    # 1. Global Bigram NLL: -log P_base[x_t, x_{t+1}]
    u_tok = tokens_cpu[:, :-1]                                        # (N, T - 1)
    v_tok = tokens_cpu[:, 1:]                                         # (N, T - 1)
    bigram_probs = p_base_f64[u_tok, v_tok]                           # (N, T - 1)
    bigram_nll = -torch.log(bigram_probs)                             # (N, T - 1)

    # 2. Exact Bayes-Optimal ICL NLL:
    #    p(x_{t+1} = v | x_{0..t}) = (kappa * P_base[u, v] + counts[u, v]) / (kappa + counts[u].sum())
    counts = torch.zeros((n_seq, vocab_size, vocab_size), dtype=torch.float64)  # (N, V, V)
    row_totals = torch.zeros((n_seq, vocab_size), dtype=torch.float64)          # (N, V)
    bayes_nll = torch.empty((n_seq, n_pred), dtype=torch.float64)               # (N, T - 1)
    seq_idx = torch.arange(n_seq)                                               # (N,)

    kappa_f64 = float(kappa)
    for t in range(n_pred):
        ut = tokens_cpu[:, t]                                         # (N,)
        vt = tokens_cpu[:, t + 1]                                     # (N,)
        num = kappa_f64 * p_base_f64[ut, vt] + counts[seq_idx, ut, vt]          # (N,)
        den = kappa_f64 + row_totals[seq_idx, ut]                               # (N,)
        bayes_nll[:, t] = -torch.log(num / den)
        counts[seq_idx, ut, vt] += 1.0
        row_totals[seq_idx, ut] += 1.0

    early_end = min(EARLY_POS_END, n_pred)
    late_start = min(LATE_POS_START, n_pred)
    late_end = min(LATE_POS_END, n_pred)

    bigram_total = float(bigram_nll.mean().item())
    bigram_early = float(bigram_nll[:, EARLY_POS_START:early_end].mean().item())
    bigram_late = float(bigram_nll[:, late_start:late_end].mean().item())

    bayes_total = float(bayes_nll.mean().item())
    bayes_early = float(bayes_nll[:, EARLY_POS_START:early_end].mean().item())
    bayes_late = float(bayes_nll[:, late_start:late_end].mean().item())

    return {
        "bigram_val_loss": bigram_total,
        "bigram_val_loss_early": bigram_early,
        "bigram_val_loss_late": bigram_late,
        "bigram_icl_score": bigram_early - bigram_late,
        "bayes_val_loss": bayes_total,
        "bayes_val_loss_early": bayes_early,
        "bayes_val_loss_late": bayes_late,
        "bayes_icl_score": bayes_early - bayes_late,
    }


@dataclass(frozen=True)
class ValidationSet:
    """Fixed deterministic validation sequences and exact analytical baselines."""

    tokens: torch.Tensor              # (N_val, T)
    baselines: Dict[str, float]
    kappa: float
    val_seed: int


def make_validation_set(
    p_base: torch.Tensor,
    kappa: float,
    n_val: int = N_VAL,
    seq_len: int = SEQ_LEN,
    val_seed: int = VAL_SEED,
) -> ValidationSet:
    """Generate deterministic fixed validation set on CPU and precompute exact baselines."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(val_seed)
        val_tokens = sample_dirichlet_markov_batch(
            p_base=p_base,
            kappa=kappa,
            batch_size=n_val,
            seq_len=seq_len,
            device="cpu",
            dtype=torch.float64,
        )  # (N_val, T)

    baselines = compute_analytical_baselines(val_tokens=val_tokens, p_base=p_base, kappa=kappa)
    return ValidationSet(tokens=val_tokens, baselines=baselines, kappa=float(kappa), val_seed=val_seed)


def make_prefix_matching_probe(
    vocab_size: int = VOCAB_SIZE,
    n_probe: int = N_PROBE,
    half_len: int = PROBE_HALF_LEN,
    probe_seed: int = PROBE_SEED,
) -> torch.Tensor:
    """Build repeated-token probe [r_0..r_{L-1}, r_0..r_{L-1}] without consecutive duplicates.

    Draws r_0..r_{L-1} uniformly subject to r_t != r_{t-1} (and r_{L-1} != r_0 so the
    concatenation seam has no consecutive duplicate either).

    Returns:
        probe_tokens: Long tensor of shape (n_probe, 2 * half_len).
    """
    if vocab_size < 3:
        raise ValueError(f"vocab_size must be >= 3 to avoid seam duplicates, got {vocab_size}")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(probe_seed)
        first_half = torch.empty((n_probe, half_len), dtype=torch.long)  # (N_probe, L)
        first_half[:, 0] = torch.randint(0, vocab_size, (n_probe,))

        for pos in range(1, half_len - 1):
            prev = first_half[:, pos - 1]                             # (N_probe,)
            # Draw from {0..V-2} and shift >= prev up by 1 for exact uniform over V \ {prev}
            raw = torch.randint(0, vocab_size - 1, (n_probe,))        # (N_probe,)
            first_half[:, pos] = raw + (raw >= prev).long()

        # Final token of first half must avoid both r_{L-2} and r_0 (seam)
        for b in range(n_probe):
            prev_tok = int(first_half[b, half_len - 2].item())
            first_tok = int(first_half[b, 0].item())
            while True:
                cand = int(torch.randint(0, vocab_size, (1,)).item())
                if cand != prev_tok and cand != first_tok:
                    first_half[b, half_len - 1] = cand
                    break

        probe_tokens = torch.cat([first_half, first_half], dim=1)     # (N_probe, 2 * L)
    return probe_tokens

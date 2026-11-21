"""From-scratch 1L-attn, 2L-attn, and 2L-full Transformers in PyTorch + einops."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

VALID_ARCHS = ("1L-attn", "2L-attn", "2L-attn-ln", "2L-full")


def apply_layernorm(
    ln: nn.LayerNorm,
    x: torch.Tensor,
    frozen_std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply LayerNorm, optionally freezing the per-token denominator std from a clean pass.

    Args:
        ln: PyTorch LayerNorm module.
        x: Input tensor of shape (B, T, D).
        frozen_std: Optional cached denominator tensor of shape (B, T, 1).

    Returns:
        y: Normalized output of shape (B, T, D).
        std: Denominator sqrt(var + eps) actually used, shape (B, T, 1).
    """
    mean = x.mean(dim=-1, keepdim=True)                               # (B, T, 1)
    if frozen_std is None:
        var = x.var(dim=-1, unbiased=False, keepdim=True)             # (B, T, 1)
        std = torch.sqrt(var + ln.eps)                                # (B, T, 1)
    else:
        std = frozen_std                                              # (B, T, 1)
    y = (x - mean) / std                                              # (B, T, D)
    if ln.elementwise_affine:
        y = y * ln.weight + ln.bias                                   # (B, T, D)
    return y, std


@dataclass(frozen=True)
class TransformerConfig:
    """Architecture configuration for controlled induction-head experiments."""

    arch: str = "2L-attn"
    vocab_size: int = 64
    max_len: int = 256
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.arch not in VALID_ARCHS:
            raise ValueError(f"Unknown arch '{self.arch}'. Expected one of {VALID_ARCHS}.")
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError(
                f"n_heads * d_head ({self.n_heads * self.d_head}) must equal d_model ({self.d_model})."
            )

    @property
    def n_layers(self) -> int:
        return 1 if self.arch == "1L-attn" else 2

    @property
    def use_mlp(self) -> bool:
        return self.arch == "2L-full"

    @property
    def use_ln(self) -> bool:
        return self.arch in ("2L-attn-ln", "2L-full")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class CausalMultiHeadAttention(nn.Module):
    """Multi-head causal self-attention with directly accessible W_Q, W_K, W_V, W_O."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.scale = 1.0 / math.sqrt(cfg.d_head)

        # Directly accessible per-head projection parameters (bias-free for exact circuit math)
        self.W_Q = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))  # (H, D, K)
        self.W_K = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))  # (H, D, K)
        self.W_V = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))  # (H, D, K)
        self.W_O = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model))  # (H, K, D)

        # Upper-triangular boolean mask where True marks forbidden future positions (s > t)
        causal_mask = torch.triu(
            torch.ones((cfg.max_len, cfg.max_len), dtype=torch.bool),
            diagonal=1,
        )  # (max_len, max_len)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self._reset_parameters(cfg.init_std)

    def _reset_parameters(self, std: float) -> None:
        nn.init.normal_(self.W_Q, mean=0.0, std=std)
        nn.init.normal_(self.W_K, mean=0.0, std=std)
        nn.init.normal_(self.W_V, mean=0.0, std=std)
        nn.init.normal_(self.W_O, mean=0.0, std=std)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D)
        ablate_head_indices: Optional[Set[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run causal multi-head attention and return (residual_update, attn_pattern).

        Args:
            x: Residual stream tensor of shape (B, T, d_model).
            ablate_head_indices: Optional set of head indices in this layer to zero-ablate.

        Returns:
            out: Attention output of shape (B, T, d_model).
            attn: Post-softmax causal attention weights of shape (B, n_heads, T, T).
        """
        seq_len = x.shape[1]

        # Project residual stream into per-head queries, keys, values
        q = einsum(x, self.W_Q, "b t d, h d k -> b h t k")            # (B, H, T, d_head)
        k = einsum(x, self.W_K, "b s d, h d k -> b h s k")            # (B, H, T, d_head)
        v = einsum(x, self.W_V, "b s d, h d k -> b h s k")            # (B, H, T, d_head)

        # Scaled dot-product attention logits with strict causal masking
        scores = einsum(q, k, "b h t k, b h s k -> b h t s") * self.scale  # (B, H, T, T)
        mask = self.causal_mask[:seq_len, :seq_len]                   # (T, T)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)                              # (B, H, T, T)

        # Weighted sum of values per head
        z = einsum(attn, v, "b h t s, b h s k -> b h t k")            # (B, H, T, d_head)

        # Optional single-head (or multi-head) zero ablation before W_O projection
        if ablate_head_indices:
            head_mask = torch.ones((1, self.n_heads, 1, 1), device=z.device, dtype=z.dtype)
            for h in ablate_head_indices:
                if not (0 <= h < self.n_heads):
                    raise ValueError(f"Head index {h} out of range for n_heads={self.n_heads}")
                head_mask[:, h, :, :] = 0.0
            z = z * head_mask                                         # (B, H, T, d_head)

        # Project heads back to residual stream and sum across heads
        out = einsum(z, self.W_O, "b h t k, h k d -> b t d")          # (B, T, d_model)
        return out, attn


class GELUMLP(nn.Module):
    """Standard 2-layer Transformer MLP with GELU activation (used in 2L-full)."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.W_in = nn.Parameter(torch.empty(cfg.d_model, cfg.d_mlp))     # (D, M)
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))                  # (M,)
        self.W_out = nn.Parameter(torch.empty(cfg.d_mlp, cfg.d_model))    # (M, D)
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))               # (D,)

        nn.init.normal_(self.W_in, mean=0.0, std=cfg.init_std)
        nn.init.normal_(self.W_out, mean=0.0, std=cfg.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU feed-forward network to x of shape (B, T, D)."""
        hidden = einsum(x, self.W_in, "b t d, d m -> b t m") + self.b_in  # (B, T, d_mlp)
        hidden = F.gelu(hidden)                                           # (B, T, d_mlp)
        out = einsum(hidden, self.W_out, "b t m, m d -> b t d") + self.b_out  # (B, T, d_model)
        return out


class TransformerBlock(nn.Module):
    """Single Transformer block: attention-only or Pre-LayerNorm attention + GELU MLP."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.use_ln = cfg.use_ln
        self.use_mlp = cfg.use_mlp

        self.ln1: Optional[nn.LayerNorm] = nn.LayerNorm(cfg.d_model) if cfg.use_ln else None
        self.attn = CausalMultiHeadAttention(cfg)
        self.ln2: Optional[nn.LayerNorm] = nn.LayerNorm(cfg.d_model) if (cfg.use_ln and cfg.use_mlp) else None
        self.mlp: Optional[GELUMLP] = GELUMLP(cfg) if cfg.use_mlp else None

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D)
        ablate_head_indices: Optional[Set[int]] = None,
        layer_idx: int = 0,
        frozen_ln_stds: Optional[Dict[str, torch.Tensor]] = None,
        ln_stds_out: Optional[Dict[str, torch.Tensor]] = None,
        attn_inputs_out: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.ln1 is not None:
            key1 = f"block{layer_idx}_ln1"
            if frozen_ln_stds is not None or ln_stds_out is not None:
                attn_in, std1 = apply_layernorm(
                    self.ln1, x, frozen_std=frozen_ln_stds.get(key1) if frozen_ln_stds else None
                )
                if ln_stds_out is not None:
                    ln_stds_out[key1] = std1
            else:
                attn_in = self.ln1(x)                                 # (B, T, D)
        else:
            attn_in = x                                               # (B, T, D)

        if attn_inputs_out is not None:
            attn_inputs_out.append(attn_in)

        attn_out, attn_pattern = self.attn(attn_in, ablate_head_indices=ablate_head_indices)
        x = x + attn_out                                              # (B, T, D)

        if self.mlp is not None:
            if self.ln2 is not None:
                key2 = f"block{layer_idx}_ln2"
                if frozen_ln_stds is not None or ln_stds_out is not None:
                    mlp_in, std2 = apply_layernorm(
                        self.ln2, x, frozen_std=frozen_ln_stds.get(key2) if frozen_ln_stds else None
                    )
                    if ln_stds_out is not None:
                        ln_stds_out[key2] = std2
                else:
                    mlp_in = self.ln2(x)                              # (B, T, D)
            else:
                mlp_in = x                                            # (B, T, D)
            x = x + self.mlp(mlp_in)                                  # (B, T, D)

        return x, attn_pattern


class InductionTransformer(nn.Module):
    """From-scratch causal Transformer supporting 1L-attn, 2L-attn, and 2L-full.

    Exposes directly accessible `W_E`, `W_pos`, `W_U`, and per-layer
    `blocks[l].attn.{W_Q, W_K, W_V, W_O}` for exact mechanistic circuit analysis.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Directly accessible embedding and unembedding matrices
        self.W_E = nn.Parameter(torch.empty(cfg.vocab_size, cfg.d_model))   # (V, D)
        self.W_pos = nn.Parameter(torch.empty(cfg.max_len, cfg.d_model))    # (max_len, D)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f: Optional[nn.LayerNorm] = nn.LayerNorm(cfg.d_model) if cfg.use_ln else None
        self.W_U = nn.Parameter(torch.empty(cfg.d_model, cfg.vocab_size))   # (D, V)

        nn.init.normal_(self.W_E, mean=0.0, std=cfg.init_std)
        nn.init.normal_(self.W_pos, mean=0.0, std=cfg.init_std)
        nn.init.normal_(self.W_U, mean=0.0, std=cfg.init_std)

    def get_W_QK(self, layer_idx: int) -> torch.Tensor:
        """Return per-head QK circuit matrices W_QK^(l, h) = W_Q @ W_K^T of shape (H, D, D).

        In column-vector residual notation (Elhage et al., 2021), pre-softmax attention
        score between query x_q and key x_k is (x_q^T W_QK x_k) / sqrt(d_head).
        """
        attn = self.blocks[layer_idx].attn
        return einsum(attn.W_Q, attn.W_K, "h d1 k, h d2 k -> h d1 d2")  # (H, D, D)

    def get_W_OV(self, layer_idx: int) -> torch.Tensor:
        """Return per-head OV circuit matrices W_OV^(l, h) = W_O^T @ W_V^T of shape (H, D, D).

        In column-vector residual notation, a head reading source vector x_s writes
        delta_x = W_OV x_s to the residual stream. Consequently, K-composition between
        Layer 0 head h0 and Layer 1 head h1 is exactly W_QK^(1, h1) @ W_OV^(0, h0).
        """
        attn = self.blocks[layer_idx].attn
        return einsum(attn.W_O, attn.W_V, "h k d1, h d2 k -> h d1 d2")  # (H, D, D)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        tokens: torch.Tensor,  # (B, T)
        return_attn: bool = False,
        ablate_heads: Optional[Iterable[Tuple[int, int]]] = None,
        frozen_ln_stds: Optional[Dict[str, torch.Tensor]] = None,
        return_cache: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, List[torch.Tensor]],
        Tuple[torch.Tensor, Dict[str, object]],
    ]:
        """Forward pass with optional attention patterns, head zero-ablation, and frozen LN scales.

        Args:
            tokens: Long tensor of token IDs of shape (B, T), T <= max_len.
            return_attn: If True (and return_cache=False), return (logits, attn_patterns).
            ablate_heads: Optional iterable of (layer_idx, head_idx) pairs to zero-ablate.
            frozen_ln_stds: Optional dict of cached LayerNorm denominator tensors (B, T, 1)
                keyed by "block{l}_ln1", "block{l}_ln2", "ln_f" (for Frozen-LN-scale ablations).
            return_cache: If True, return (logits, cache_dict) containing "attn_patterns",
                "ln_stds", and "attn_inputs".

        Returns:
            logits of shape (B, T, V), or (logits, attn_patterns), or (logits, cache_dict).
        """
        seq_len = tokens.shape[1]
        if seq_len > self.cfg.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.cfg.max_len}")

        # Group requested head ablations by layer index
        ablate_by_layer: Dict[int, Set[int]] = {}
        if ablate_heads is not None:
            for layer_idx, head_idx in ablate_heads:
                if not (0 <= layer_idx < self.cfg.n_layers):
                    raise ValueError(
                        f"Layer index {layer_idx} out of range for n_layers={self.cfg.n_layers}"
                    )
                ablate_by_layer.setdefault(layer_idx, set()).add(head_idx)

        # Token + learned positional embeddings: (B, T, D)
        tok_emb = self.W_E[tokens]                                    # (B, T, D)
        pos_emb = self.W_pos[:seq_len].unsqueeze(0)                   # (1, T, D)
        x = tok_emb + pos_emb                                         # (B, T, D)

        attn_patterns: List[torch.Tensor] = []
        ln_stds_out: Optional[Dict[str, torch.Tensor]] = {} if return_cache else None
        attn_inputs_out: Optional[List[torch.Tensor]] = [] if return_cache else None

        for layer_idx, block in enumerate(self.blocks):
            x, attn = block(
                x,
                ablate_head_indices=ablate_by_layer.get(layer_idx),
                layer_idx=layer_idx,
                frozen_ln_stds=frozen_ln_stds,
                ln_stds_out=ln_stds_out,
                attn_inputs_out=attn_inputs_out,
            )
            if return_attn or return_cache:
                attn_patterns.append(attn)

        if self.ln_f is not None:
            if frozen_ln_stds is not None or ln_stds_out is not None:
                x, std_f = apply_layernorm(
                    self.ln_f, x, frozen_std=frozen_ln_stds.get("ln_f") if frozen_ln_stds else None
                )
                if ln_stds_out is not None:
                    ln_stds_out["ln_f"] = std_f
            else:
                x = self.ln_f(x)                                      # (B, T, D)

        logits = einsum(x, self.W_U, "b t d, d v -> b t v")           # (B, T, V)

        if return_cache:
            return logits, {
                "attn_patterns": attn_patterns,
                "ln_stds": ln_stds_out,
                "attn_inputs": attn_inputs_out,
            }
        if return_attn:
            return logits, attn_patterns
        return logits

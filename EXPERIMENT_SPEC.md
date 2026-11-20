# Experiment Spec: Induction Heads & Emergent ICL Phase Transitions (Olsson et al., 2022)

## Objective
Reproduce the induction-head phase transition from scratch in PyTorch + `einops` on Zipfian In-Context Markov Chains (ICMC), verify the 1-layer vs. 2-layer depth barrier with exact circuit progress measures and causal head ablations, and sweep prior concentration (`kappa`) and weight decay (`wd`) across multiple seeds overnight.

## Synthetic Data Distribution (Zero External Downloads)
- Vocabulary size `V = 64`, sequence length `T = 256`.
- Fixed global base transition matrix `P_base` (shape `(64, 64)`): each row is a deterministic random permutation of a Zipfian distribution `rank^(-0.8)` normalized to sum to 1 (generated once from a fixed `data_seed = 12345`).
- Per-sequence ("document") transition matrix: each row `i` is drawn fresh from `Dirichlet(kappa * P_base[i])`.
- Online training batches (`batch_size = 64`, fresh sequences sampled on GPU via `Gamma` normalization to avoid overfitting/memorization of fixed training files).
- Fixed deterministic validation set (`n_val = 512` sequences, `val_seed = 99999`) for each `kappa`, with exact precomputed analytical baselines:
  - **Global bigram loss** using `P_base[x_t, x_{t+1}]`
  - **Exact Bayes-optimal ICL loss** using the online Dirichlet-Multinomial posterior `(kappa * P_base[u] + counts[u]) / (kappa + counts[u].sum())`
- Fixed **prefix-matching probe set** (`n_probe = 128` random repeated sequences `[r_0..r_63, r_0..r_63]` of length `128`, drawn uniformly without consecutive duplicates) to measure exact Olsson et al. prefix-matching attention scores.

## Model Architectures (From Scratch, PyTorch + `einops`)
- Shared dimensions: `d_model = 128`, `n_heads = 4` (`d_head = 32`), learned positional embeddings (`max_len = 256`), causal attention mask, directly accessible `W_E`, `W_pos`, `W_Q`, `W_K`, `W_V`, `W_O`, `W_U`.
- Three controlled variants:
  1. `1L-attn`: 1-layer attention-only, no MLP, no LayerNorm (structural negative control).
  2. `2L-attn`: 2-layer attention-only, no MLP, no LayerNorm (clean composition math).
  3. `2L-full`: 2-layer full Transformer with Pre-LayerNorm + `d_mlp = 512` GELU MLP.

## Optimization & Logging
- Optimizer: AdamW, `lr = 1e-3`, `betas = (0.9, 0.98)`, cosine or constant LR (constant `1e-3` default so phase-transition timing isn't confounded by schedule decay).
- Training budget: `10_000` steps per run (`batch_size = 64`, `~164M` tokens/run).
- Evaluate & log to `metrics.jsonl` every `100` steps:
  - `val_loss`, `val_loss_early` (pos `1..31`), `val_loss_late` (pos `192..254`), `icl_score = val_loss_early - val_loss_late`
  - `max_prev_token_score_L0` (max across L0 heads of mean attention weight on `(i, i-1)`)
  - `max_prefix_match_score` (max across all heads of mean attention from second-half pos `t` to first-half pos `t - 64 + 1` on the repeated-token probe) and `per_head_prefix_match`
  - `max_k_composition_score` (for 2L models: max over `(h0, h1)` of `||W_QK^(1,h1) @ W_OV^(0,h0)||_F / (||W_QK^(1,h1)||_F * ||W_OV^(0,h0)||_F)`)
  - `weight_l2_norm`
- At the end of each run, perform **causal head ablation** on the validation set: zero-ablate the single highest prefix-matching head in Layer 1 vs. mean across non-induction Layer 1 heads, logging `ablation_icl_drop_top_head` and `ablation_icl_drop_control`.
- Compact disk footprint: save only the final checkpoint (`final.pt`, ~1–2 MB) plus `summary.json` and `metrics.jsonl` per run.

## Mandatory Pre-Run Sanity Suite (`induction/tests/test_sanity.py`)
1. Output shapes `(B, T, V)` and causal mask invariance for `1L-attn`, `2L-attn`, and `2L-full`.
2. Analytical Dirichlet-Markov verifier: Bayes-optimal ICL loss is strictly lower than global bigram loss on `val_loss_late` by at least `1.5` nats at `kappa = 1.0`.
3. Prefix-matching and previous-token attention score unit tests on synthetic ground-truth attention matrices (score equals `1.0` on ideal patterns, `0.0` on uniform/diagonal mismatch).
4. K-composition matrix identity check against explicit `W_Q`, `W_K`, `W_V`, `W_O` products.
5. Determinism check on CUDA for identical seeds.

## Overnight Resumable Sweep Matrix (`35` runs total, `seeds = 0..4`)
- **Axis 1 — Architecture & Depth (`kappa = 1.0`, `wd = 0.01`):** `1L-attn`, `2L-attn`, `2L-full` (15 runs).
- **Axis 2 — Prior Concentration (`2L-attn`, `wd = 0.01`):** `kappa = 0.3` and `kappa = 3.0` (10 runs; `kappa = 1.0` reused from Axis 1).
- **Axis 3 — Weight Decay (`2L-attn`, `kappa = 1.0`):** `wd = 0.0` and `wd = 0.1` (10 runs; `wd = 0.01` reused from Axis 1).

"""Mandatory pre-run sanity suite for Project #2 (Induction Heads & Emergent ICL)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

try:
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
        make_base_transition_matrix,
        make_prefix_matching_probe,
        make_validation_set,
        set_seed,
    )
    from src.metrics import (
        compute_k_composition_matrix,
        compute_prefix_matching_score,
        compute_prev_token_score,
    )
    from src.model import VALID_ARCHS, InductionTransformer, TransformerConfig
    from src.train import build_optimizer
except ModuleNotFoundError:
    from induction.src.data import (
        DATA_SEED,
        N_PROBE,
        N_VAL,
        PROBE_HALF_LEN,
        PROBE_SEED,
        SEQ_LEN,
        VAL_SEED,
        VOCAB_SIZE,
        OnlineDirichletMarkovSampler,
        make_base_transition_matrix,
        make_prefix_matching_probe,
        make_validation_set,
        set_seed,
    )
    from induction.src.metrics import (
        compute_k_composition_matrix,
        compute_prefix_matching_score,
        compute_prev_token_score,
    )
    from induction.src.model import VALID_ARCHS, InductionTransformer, TransformerConfig
    from induction.src.train import build_optimizer


@pytest.mark.parametrize("arch", VALID_ARCHS)
def test_1_output_shapes_and_causal_invariance(arch: str) -> None:
    """1. Verify (B, T, V) output shapes, attention shapes, and strict causal mask invariance."""
    set_seed(101)
    cfg = TransformerConfig(arch=arch)
    model = InductionTransformer(cfg)
    model.eval()

    B, T = 4, 64
    tokens = torch.randint(0, VOCAB_SIZE, (B, T), dtype=torch.long)

    with torch.no_grad():
        logits, attns = model(tokens, return_attn=True)

    assert logits.shape == (B, T, VOCAB_SIZE)
    assert len(attns) == cfg.n_layers
    for attn in attns:
        assert attn.shape == (B, cfg.n_heads, T, T)
        # Rows must sum to 1 and strictly upper-triangular entries must be 0
        assert torch.allclose(attn.sum(dim=-1), torch.ones((B, cfg.n_heads, T)), atol=1e-5)
        upper = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1)
        assert torch.all(attn[:, :, upper] == 0.0)

    # Causal invariance: modifying tokens strictly after t_cut must not change logits[:t_cut+1]
    t_cut = 25
    perturbed = tokens.clone()
    perturbed[:, t_cut + 1 :] = (perturbed[:, t_cut + 1 :] + 17) % VOCAB_SIZE

    with torch.no_grad():
        logits_perturbed = model(perturbed, return_attn=False)

    max_diff = (logits[:, : t_cut + 1, :] - logits_perturbed[:, : t_cut + 1, :]).abs().max().item()
    assert max_diff < 1e-6, f"Causal leakage detected in {arch}: max_diff={max_diff}"

    # Verify single-head zero ablation alters output cleanly and handles head index bounds
    with torch.no_grad():
        logits_abl = model(tokens, ablate_heads=[(0, 0)])
    assert logits_abl.shape == (B, T, VOCAB_SIZE)
    assert not torch.allclose(logits, logits_abl, atol=1e-7)


def test_2_analytical_dirichlet_markov_baselines_and_probe() -> None:
    """2. Verify Bayes-optimal ICL beats global bigram by >= 1.5 nats on val_loss_late at kappa=1.0."""
    p_base = make_base_transition_matrix(vocab_size=VOCAB_SIZE, data_seed=DATA_SEED)
    assert p_base.shape == (VOCAB_SIZE, VOCAB_SIZE)
    assert torch.allclose(p_base.sum(dim=-1), torch.ones(VOCAB_SIZE, dtype=torch.float64), atol=1e-12)

    val_set = make_validation_set(
        p_base=p_base,
        kappa=1.0,
        n_val=N_VAL,
        seq_len=SEQ_LEN,
        val_seed=VAL_SEED,
    )
    assert val_set.tokens.shape == (N_VAL, SEQ_LEN)

    bl = val_set.baselines
    late_gap = bl["bigram_val_loss_late"] - bl["bayes_val_loss_late"]
    assert late_gap >= 1.5, (
        f"Expected Bayes-optimal val_loss_late to beat Global Bigram by >= 1.5 nats at kappa=1.0, "
        f"got gap={late_gap:.4f} (bigram={bl['bigram_val_loss_late']:.4f}, bayes={bl['bayes_val_loss_late']:.4f})"
    )
    assert bl["bayes_icl_score"] > 1.0, f"Expected strong positive Bayes ICL score, got {bl['bayes_icl_score']:.4f}"

    # Verify repeated-token probe structure: [r_0..r_63, r_0..r_63] with zero consecutive duplicates
    probe = make_prefix_matching_probe(
        vocab_size=VOCAB_SIZE,
        n_probe=N_PROBE,
        half_len=PROBE_HALF_LEN,
        probe_seed=PROBE_SEED,
    )
    assert probe.shape == (N_PROBE, 2 * PROBE_HALF_LEN)
    assert torch.equal(probe[:, :PROBE_HALF_LEN], probe[:, PROBE_HALF_LEN:])
    assert torch.all(probe[:, 1:] != probe[:, :-1]), "Consecutive duplicate found in prefix-matching probe"


def test_3_prefix_matching_and_prev_token_score_unit_tests() -> None:
    """3. Synthetic ground-truth attention tests: 1.0 on ideal patterns, 0.0 on diagonal mismatch."""
    B, H = 3, 4
    L = PROBE_HALF_LEN
    T = 2 * L  # 128

    # Construct synthetic attention matrices for each of the 4 heads:
    # Head 0: ideal previous-token head -> weight 1.0 on (i, i - 1) for i >= 1, (0, 0) = 1.0
    # Head 1: ideal induction (prefix-matching) head -> weight 1.0 on (t, t - L + 1) for t in L..2L-2
    # Head 2: diagonal mismatch -> weight 1.0 on (i, i)
    # Head 3: uniform causal attention -> 1 / (i + 1) for j <= i
    attn = torch.zeros((B, H, T, T), dtype=torch.float32)

    # Head 0: ideal previous-token
    attn[:, 0, 0, 0] = 1.0
    idx_prev = torch.arange(1, T)
    attn[:, 0, idx_prev, idx_prev - 1] = 1.0

    # Head 1: ideal prefix-matching on second half, diagonal on first half & final pos
    for i in range(T):
        if L <= i <= 2 * L - 2:
            attn[:, 1, i, i - L + 1] = 1.0
        else:
            attn[:, 1, i, i] = 1.0

    # Head 2: pure diagonal mismatch
    idx_all = torch.arange(T)
    attn[:, 2, idx_all, idx_all] = 1.0

    # Head 3: uniform causal
    for i in range(T):
        attn[:, 3, i, : i + 1] = 1.0 / float(i + 1)

    # Verify all heads are valid probability distributions
    assert torch.allclose(attn.sum(dim=-1), torch.ones((B, H, T)), atol=1e-6)

    prev_scores = compute_prev_token_score(attn)                      # (H,)
    pm_scores = compute_prefix_matching_score(attn, half_len=L)       # (H,)

    # Ideal patterns must equal 1.0 exactly
    assert prev_scores[0].item() == pytest.approx(1.0, abs=1e-6)
    assert pm_scores[1].item() == pytest.approx(1.0, abs=1e-6)

    # Diagonal mismatch must equal 0.0 exactly for both metrics
    assert prev_scores[2].item() == pytest.approx(0.0, abs=1e-6)
    assert pm_scores[2].item() == pytest.approx(0.0, abs=1e-6)

    # Cross-pattern mismatch: ideal previous-token head has 0.0 prefix-matching score (since L=64 > 2)
    assert pm_scores[0].item() == pytest.approx(0.0, abs=1e-6)
    # Ideal prefix-matching head has 0.0 previous-token score (k = t - 63 != t - 1 and first half is diagonal)
    assert prev_scores[1].item() == pytest.approx(0.0, abs=1e-6)

    # Uniform causal baseline matches exact analytical harmonic mean
    expected_unif_prev = sum(1.0 / (i + 1) for i in range(1, T)) / (T - 1)
    expected_unif_pm = sum(1.0 / (t + 1) for t in range(L, 2 * L - 1)) / (L - 1)
    assert prev_scores[3].item() == pytest.approx(expected_unif_prev, abs=1e-6)
    assert pm_scores[3].item() == pytest.approx(expected_unif_pm, abs=1e-6)
    assert prev_scores[3].item() < 0.05
    assert pm_scores[3].item() < 0.02


def test_4_k_composition_matrix_identity() -> None:
    """4. Verify K-composition matrix matches explicit W_Q, W_K, W_V, W_O products and circuit action."""
    set_seed(202)
    model_1l = InductionTransformer(TransformerConfig(arch="1L-attn"))
    assert compute_k_composition_matrix(model_1l) is None

    model_2l = InductionTransformer(TransformerConfig(arch="2L-attn"))
    k_comp = compute_k_composition_matrix(model_2l)
    assert k_comp is not None
    assert k_comp.shape == (model_2l.cfg.n_heads, model_2l.cfg.n_heads)

    attn0 = model_2l.blocks[0].attn
    attn1 = model_2l.blocks[1].attn

    # Random query vector x_q and Layer 0 source vector x_s (row vectors)
    x_q = torch.randn(5, model_2l.cfg.d_model)                        # (5, D)
    x_s = torch.randn(5, model_2l.cfg.d_model)                        # (5, D)

    for h0 in range(model_2l.cfg.n_heads):
        W_V0 = attn0.W_V[h0]                                          # (D, K)
        W_O0 = attn0.W_O[h0]                                          # (K, D)
        # Column-vector OV map: W_OV = W_O^T @ W_V^T
        W_OV0_explicit = W_O0.T @ W_V0.T                              # (D, D)
        assert torch.allclose(model_2l.get_W_OV(0)[h0], W_OV0_explicit, atol=1e-6)

        for h1 in range(model_2l.cfg.n_heads):
            W_Q1 = attn1.W_Q[h1]                                      # (D, K)
            W_K1 = attn1.W_K[h1]                                      # (D, K)
            W_QK1_explicit = W_Q1 @ W_K1.T                            # (D, D)
            assert torch.allclose(model_2l.get_W_QK(1)[h1], W_QK1_explicit, atol=1e-6)

            # 1) Explicit matrix product check
            M_explicit = W_QK1_explicit @ W_OV0_explicit              # (D, D)
            expected_score = torch.linalg.matrix_norm(M_explicit, ord="fro") / (
                torch.linalg.matrix_norm(W_QK1_explicit, ord="fro")
                * torch.linalg.matrix_norm(W_OV0_explicit, ord="fro")
                + 1e-12
            )
            assert k_comp[h0, h1].item() == pytest.approx(expected_score.item(), rel=1e-5, abs=1e-6)

            # 2) Functional circuit identity check:
            #    L0 reads x_s and writes delta_k = x_s @ W_V0 @ W_O0 into the residual stream;
            #    L1 computes attention score between query x_q @ W_Q1 and key delta_k @ W_K1.
            delta_k = (x_s @ W_V0) @ W_O0                             # (5, D)
            q1 = x_q @ W_Q1                                           # (5, K)
            k1 = delta_k @ W_K1                                       # (5, K)
            functional_score = (q1 * k1).sum(dim=-1)                  # (5,)
            matrix_score = ((x_q @ M_explicit) * x_s).sum(dim=-1)     # (5,)
            assert torch.allclose(functional_score, matrix_score, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_5_cuda_determinism() -> None:
    """5. Identical seeds produce identical batches, weights, and logits on CUDA."""
    p_base = make_base_transition_matrix(vocab_size=VOCAB_SIZE, data_seed=DATA_SEED)
    device = torch.device("cuda")

    def run_short(seed: int):
        set_seed(seed)
        model = InductionTransformer(TransformerConfig(arch="2L-attn")).to(device)
        opt = build_optimizer(model, lr=1e-3, wd=0.01)
        sampler = OnlineDirichletMarkovSampler(
            p_base=p_base,
            kappa=1.0,
            batch_size=16,
            seq_len=64,
            device=device,
            chunk_batches=2,
        )
        losses = []
        last_batch = None
        for _ in range(4):
            batch = sampler.next_batch()
            last_batch = batch
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, VOCAB_SIZE),
                batch[:, 1:].reshape(-1),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        with torch.no_grad():
            final_logits = model(last_batch)
        return losses, final_logits.cpu(), {k: v.detach().cpu() for k, v in model.state_dict().items()}

    losses_a, logits_a, state_a = run_short(seed=7)
    losses_b, logits_b, state_b = run_short(seed=7)
    losses_c, logits_c, _ = run_short(seed=8)

    assert losses_a == losses_b, f"Loss trajectory diverged: {losses_a} vs {losses_b}"
    assert torch.equal(logits_a, logits_b), "Final CUDA logits differed for identical seeds"
    for name in state_a:
        assert torch.equal(state_a[name], state_b[name]), f"Parameter {name} differed for identical seeds"

    assert not torch.allclose(logits_a, logits_c), "Different seeds unexpectedly produced identical logits"


def test_6_single_batch_overfit() -> None:
    """Extra AGENTS.md check: 2L-full can rapidly overfit a single tiny batch."""
    set_seed(303)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InductionTransformer(TransformerConfig(arch="2L-full")).to(device)
    opt = build_optimizer(model, lr=3e-3, wd=0.0)

    batch = torch.randint(0, VOCAB_SIZE, (4, 32), device=device, dtype=torch.long)
    initial_loss = None
    final_loss = None
    for step in range(120):
        logits = model(batch)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, VOCAB_SIZE),
            batch[:, 1:].reshape(-1),
        )
        if step == 0:
            initial_loss = float(loss.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final_loss = float(loss.item())

    assert initial_loss is not None and final_loss is not None
    assert final_loss < 0.05, f"Failed single-batch overfit: initial={initial_loss:.3f}, final={final_loss:.3f}"


@pytest.mark.parametrize("arch", ("2L-attn-ln", "2L-full"))
def test_7_frozen_ln_scale_hook_and_2l_attn_ln_structure(arch: str) -> None:
    """Verify 2L-attn-ln structure (LN without MLP) and exact Frozen-LN-scale ablation hook."""
    set_seed(404)
    cfg = TransformerConfig(arch=arch)
    model = InductionTransformer(cfg)
    model.eval()

    if arch == "2L-attn-ln":
        assert cfg.n_layers == 2 and cfg.use_ln and not cfg.use_mlp
        assert model.ln_f is not None
        for block in model.blocks:
            assert block.ln1 is not None
            assert block.ln2 is None and block.mlp is None

    tokens = torch.randint(0, VOCAB_SIZE, (4, 32), dtype=torch.long)
    with torch.no_grad():
        clean_logits = model(tokens)
        cached_logits, cache = model(tokens, return_cache=True)
        ln_stds = cache["ln_stds"]

        # 1) Cached forward matches native nn.LayerNorm forward to float32 precision
        assert torch.allclose(clean_logits, cached_logits, atol=1e-5)
        assert "ln_f" in ln_stds and "block0_ln1" in ln_stds and "block1_ln1" in ln_stds
        if arch == "2L-full":
            assert "block1_ln2" in ln_stds
        else:
            assert "block1_ln2" not in ln_stds

        # 2) Passing clean frozen_ln_stds into an unablated pass reproduces clean logits
        replayed_logits = model(tokens, frozen_ln_stds=ln_stds)
        assert torch.allclose(clean_logits, replayed_logits, atol=1e-5)

        # 3) Under L1 head ablation, frozen LN scales differ from standard recomputed LN scales
        #    (amplifying weights slightly so residual variance shift under ablation is pronounced)
        model.blocks[1].attn.W_O.data.mul_(10.0)
        _, cache_amp = model(tokens, return_cache=True)
        std_abl = model(tokens, ablate_heads=[(1, 0), (1, 1)])
        frz_abl = model(tokens, ablate_heads=[(1, 0), (1, 1)], frozen_ln_stds=cache_amp["ln_stds"])
        assert not torch.allclose(std_abl, frz_abl, atol=1e-5)


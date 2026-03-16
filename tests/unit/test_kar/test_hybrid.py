"""Tests for KARModel (hybrid composition)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from src.config import DeepFMConfig, ExpertConfig, FusionConfig, GatingConfig, KARConfig, LightGCNConfig
from src.kar.hybrid import KARInput, KARModel, compute_d_backbone
from src.models.deepfm import DeepFM, DeepFMInput
from src.models.lightgcn import LightGCN, LightGCNInput, build_normalized_adj
import numpy as np


@pytest.fixture
def rngs():
    return nnx.Rngs(params=0, dropout=1)


@pytest.fixture
def kar_config():
    return KARConfig(
        expert=ExpertConfig(d_enc=768, d_hidden=64, d_rec=32),
        gating=GatingConfig(variant="g2"),
        fusion=FusionConfig(variant="f2"),
    )


@pytest.fixture
def deepfm_backbone(rngs):
    """Small DeepFM for testing."""
    field_dims = [8, 5, 5, 10, 8, 6, 10, 4]
    config = DeepFMConfig(d_embed=4, dnn_hidden_dims=(32, 16), dropout_rate=0.0)
    return DeepFM(field_dims, n_numerical=10, config=config, rngs=rngs)


@pytest.fixture
def deepfm_kar(deepfm_backbone, kar_config, rngs):
    """KARModel wrapping DeepFM."""
    d_backbone = compute_d_backbone("deepfm", deepfm_backbone)
    return KARModel(
        backbone=deepfm_backbone,
        backbone_name="deepfm",
        kar_config=kar_config,
        d_backbone=d_backbone,
        rngs=rngs,
    )


@pytest.fixture
def fake_kar_input():
    """Fake KARInput for DeepFM."""
    B = 4
    return KARInput(
        base_input=DeepFMInput(
            user_cat=jnp.ones((B, 3), dtype=jnp.int32),
            user_num=jnp.ones((B, 8), dtype=jnp.float32),
            item_cat=jnp.ones((B, 5), dtype=jnp.int32),
            item_num=jnp.ones((B, 2), dtype=jnp.float32),
        ),
        h_fact=jnp.ones((B, 768)),
        h_reason=jnp.ones((B, 768)),
    )


def test_kar_forward_shape(deepfm_kar, fake_kar_input):
    """KARModel forward returns (B,) logits."""
    deepfm_kar.eval()
    logits = deepfm_kar(fake_kar_input)
    assert logits.shape == (4,)


def test_kar_forward_with_intermediates(deepfm_kar, fake_kar_input):
    """forward_with_intermediates returns logits + dict."""
    deepfm_kar.eval()
    logits, intermediates = deepfm_kar.forward_with_intermediates(fake_kar_input)
    assert logits.shape == (4,)
    assert "e_fact" in intermediates
    assert "e_reason" in intermediates
    assert "g_fact" in intermediates
    assert "g_reason" in intermediates
    assert "x_backbone_flat" in intermediates
    assert intermediates["e_fact"].shape == (4, 32)  # d_rec
    assert intermediates["g_fact"].shape == (4, 1)


def test_kar_gradient_flows(deepfm_kar, fake_kar_input):
    """Gradients flow through KARModel."""

    def loss_fn(model):
        logits = model(fake_kar_input)
        return jnp.mean(logits ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(deepfm_kar)
    grad_leaves = jax.tree.leaves(grads)
    assert any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, 'shape'))


def test_kar_jit_compatible(deepfm_kar, fake_kar_input):
    """KARModel can be JIT-compiled."""

    @nnx.jit
    def forward(model, x):
        return model(x)

    logits = forward(deepfm_kar, fake_kar_input)
    assert logits.shape == (4,)


def test_kar_get_expert_outputs(deepfm_kar):
    """get_expert_outputs returns (e_fact, e_reason)."""
    h_fact = jnp.ones((4, 768))
    h_reason = jnp.ones((4, 768))
    deepfm_kar.eval()
    e_fact, e_reason = deepfm_kar.get_expert_outputs(h_fact, h_reason)
    assert e_fact.shape == (4, 32)
    assert e_reason.shape == (4, 32)


def test_kar_align_targets(deepfm_kar):
    """get_align_targets projects to d_rec."""
    d_backbone = deepfm_kar._d_backbone
    x = jnp.ones((4, d_backbone))
    proj = deepfm_kar.get_align_targets(x)
    assert proj.shape == (4, 32)  # d_rec


def test_compute_d_backbone_deepfm(deepfm_backbone):
    """compute_d_backbone for DeepFM."""
    d = compute_d_backbone("deepfm", deepfm_backbone)
    # (8 cat + 10 num) * 4 d_embed = 72
    assert d == 72


def test_kar_backbone_unchanged(deepfm_backbone, fake_kar_input, kar_config, rngs):
    """Standalone backbone __call__ still works (backward compatible)."""
    standalone_out = deepfm_backbone(fake_kar_input.base_input)
    assert standalone_out.shape == (4,)


def test_kar_no_nan_output(deepfm_kar, fake_kar_input):
    """KARModel output has no NaN."""
    deepfm_kar.eval()
    logits = deepfm_kar(fake_kar_input)
    assert not jnp.any(jnp.isnan(logits))


def test_kar_different_gating_variants(deepfm_backbone, rngs):
    """KARModel works with all non-context gating variants."""
    for variant in ["g1", "g2", "g4"]:
        config = KARConfig(
            expert=ExpertConfig(d_enc=768, d_hidden=64, d_rec=32),
            gating=GatingConfig(variant=variant),
            fusion=FusionConfig(variant="f2"),
        )
        d_backbone = compute_d_backbone("deepfm", deepfm_backbone)
        model = KARModel(
            backbone=deepfm_backbone,
            backbone_name="deepfm",
            kar_config=config,
            d_backbone=d_backbone,
            rngs=rngs,
        )
        model.eval()
        inp = KARInput(
            base_input=DeepFMInput(
                user_cat=jnp.ones((2, 3), dtype=jnp.int32),
                user_num=jnp.ones((2, 8)),
                item_cat=jnp.ones((2, 5), dtype=jnp.int32),
                item_num=jnp.ones((2, 2)),
            ),
            h_fact=jnp.ones((2, 768)),
            h_reason=jnp.ones((2, 768)),
        )
        logits = model(inp)
        assert logits.shape == (2,)


def test_kar_different_fusion_variants(deepfm_backbone, rngs):
    """KARModel works with f2, f3, f4 (f1 changes dim)."""
    for variant in ["f2", "f3", "f4"]:
        config = KARConfig(
            expert=ExpertConfig(d_enc=768, d_hidden=64, d_rec=32),
            gating=GatingConfig(variant="g2"),
            fusion=FusionConfig(variant=variant),
        )
        d_backbone = compute_d_backbone("deepfm", deepfm_backbone)
        model = KARModel(
            backbone=deepfm_backbone,
            backbone_name="deepfm",
            kar_config=config,
            d_backbone=d_backbone,
            rngs=rngs,
        )
        model.eval()
        inp = KARInput(
            base_input=DeepFMInput(
                user_cat=jnp.ones((2, 3), dtype=jnp.int32),
                user_num=jnp.ones((2, 8)),
                item_cat=jnp.ones((2, 5), dtype=jnp.int32),
                item_num=jnp.ones((2, 2)),
            ),
            h_fact=jnp.ones((2, 768)),
            h_reason=jnp.ones((2, 768)),
        )
        logits = model(inp)
        assert logits.shape == (2,)


def test_kar_lightgcn_backbone(rngs, kar_config):
    """KARModel works with LightGCN backbone."""
    n_users, n_items = 10, 20
    user_idx = np.array([0, 1, 2, 3], dtype=np.int32)
    item_idx = np.array([0, 1, 2, 3], dtype=np.int32)
    adj = build_normalized_adj(user_idx, item_idx, n_users, n_items)
    config = LightGCNConfig(d_embed=16, n_layers=2)
    backbone = LightGCN(n_users, n_items, adj, config, rngs=rngs)

    d_backbone = compute_d_backbone("lightgcn", backbone)
    assert d_backbone == 32  # 16 * 2

    model = KARModel(
        backbone=backbone,
        backbone_name="lightgcn",
        kar_config=kar_config,
        d_backbone=d_backbone,
        rngs=rngs,
    )
    model.eval()
    inp = KARInput(
        base_input=LightGCNInput(
            user_idx=jnp.array([0, 1], dtype=jnp.int32),
            item_idx=jnp.array([0, 1], dtype=jnp.int32),
        ),
        h_fact=jnp.ones((2, 768)),
        h_reason=jnp.ones((2, 768)),
    )
    logits = model(inp)
    assert logits.shape == (2,)


def test_kar_multi_stage_gradient_flow(deepfm_kar, fake_kar_input):
    """Stage 2/3 loss gradient flows through experts."""
    from src.losses import kar_total_loss

    def loss_fn(model):
        logits, intermediates = model.forward_with_intermediates(fake_kar_input)
        x_proj = model.get_align_targets(intermediates["x_backbone_flat"])
        total, _ = kar_total_loss(
            logits, jnp.ones(4), intermediates["e_fact"], x_proj,
            intermediates["e_reason"], x_proj, include_rec_loss=True,
        )
        return total

    loss, grads = nnx.value_and_grad(loss_fn)(deepfm_kar)
    assert loss > 0
    grad_leaves = jax.tree.leaves(grads)
    assert any(jnp.any(g != 0) for g in grad_leaves if hasattr(g, 'shape'))

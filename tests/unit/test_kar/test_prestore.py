"""Tests for prestore computation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from src.config import DeepFMConfig, ExpertConfig, FusionConfig, GatingConfig, KARConfig
from src.kar.hybrid import KARModel, compute_d_backbone
from src.models.deepfm import DeepFM
from src.serving.prestore import _batch_expert_forward, compute_prestore, load_prestore


@pytest.fixture
def rngs():
    return nnx.Rngs(params=0, dropout=1)


@pytest.fixture
def kar_model(rngs):
    """Small KARModel for testing."""
    field_dims = [8, 5, 5, 10, 8, 6, 10, 4]
    backbone = DeepFM(
        field_dims, n_numerical=10,
        config=DeepFMConfig(d_embed=4, dnn_hidden_dims=(16,), dropout_rate=0.0),
        rngs=rngs,
    )
    kar_config = KARConfig(
        expert=ExpertConfig(d_enc=768, d_hidden=32, d_rec=16),
        gating=GatingConfig(variant="g2"),
        fusion=FusionConfig(variant="f2"),
    )
    d_backbone = compute_d_backbone("deepfm", backbone)
    return KARModel(
        backbone=backbone, backbone_name="deepfm",
        kar_config=kar_config, d_backbone=d_backbone, rngs=rngs,
    )


def test_batch_expert_forward_shape(kar_model):
    """_batch_expert_forward produces correct shape."""
    embeddings = np.random.randn(100, 768).astype(np.float32)
    out = _batch_expert_forward(kar_model.factual_expert, embeddings, batch_size=32)
    assert out.shape == (100, 16)


def test_compute_prestore_saves_files(kar_model):
    """compute_prestore saves item_expert.npz and user_expert.npz."""
    item_emb = np.random.randn(50, 768).astype(np.float32)
    user_emb = np.random.randn(30, 768).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        item_path, user_path = compute_prestore(
            kar_model, item_emb, user_emb, output_dir, batch_size=16
        )
        assert item_path.exists()
        assert user_path.exists()

        item_data = np.load(item_path)
        assert item_data["expert_outputs"].shape == (50, 16)

        user_data = np.load(user_path)
        assert user_data["expert_outputs"].shape == (30, 16)


def test_load_prestore(kar_model):
    """load_prestore loads saved files correctly."""
    item_emb = np.random.randn(50, 768).astype(np.float32)
    user_emb = np.random.randn(30, 768).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        compute_prestore(kar_model, item_emb, user_emb, output_dir, batch_size=16)

        loaded_item, loaded_user = load_prestore(output_dir)
        assert loaded_item.shape == (50, 16)
        assert loaded_user.shape == (30, 16)


def test_prestore_no_nan(kar_model):
    """Prestore outputs have no NaN."""
    item_emb = np.random.randn(20, 768).astype(np.float32)
    user_emb = np.random.randn(10, 768).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        compute_prestore(kar_model, item_emb, user_emb, Path(tmpdir), batch_size=8)
        item_out, user_out = load_prestore(Path(tmpdir))
        assert not np.any(np.isnan(item_out))
        assert not np.any(np.isnan(user_out))

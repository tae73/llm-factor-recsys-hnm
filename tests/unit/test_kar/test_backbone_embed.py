"""Tests for backbone embed()/predict_from_embedding() backward compatibility."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from src.config import DCNv2Config, DeepFMConfig, DINConfig, LightGCNConfig, SASRecConfig
from src.models.dcnv2 import DCNv2
from src.models.deepfm import DeepFM, DeepFMInput
from src.models.din import DIN, DINInput
from src.models.lightgcn import LightGCN, LightGCNInput, build_normalized_adj
from src.models.sasrec import SASRec, SASRecInput


@pytest.fixture
def rngs():
    return nnx.Rngs(params=0, dropout=1)


# --- DeepFM ---

def test_deepfm_embed_predict_equals_call(rngs):
    """DeepFM: embed() + predict_from_embedding() == __call__()."""
    field_dims = [8, 5, 5, 10, 8, 6, 10, 4]
    model = DeepFM(field_dims, 10, DeepFMConfig(d_embed=4, dnn_hidden_dims=(16,), dropout_rate=0.0), rngs=rngs)
    model.eval()

    x = DeepFMInput(
        user_cat=jnp.ones((4, 3), dtype=jnp.int32),
        user_num=jnp.ones((4, 8)),
        item_cat=jnp.ones((4, 5), dtype=jnp.int32),
        item_num=jnp.ones((4, 2)),
    )

    direct = model(x)
    stacked, first_order = model.embed(x)
    split = model.predict_from_embedding(stacked, first_order)

    assert jnp.allclose(direct, split, atol=1e-5)


# --- DCNv2 ---

def test_dcnv2_embed_predict_equals_call(rngs):
    """DCNv2: embed() + predict_from_embedding() == __call__()."""
    field_dims = [8, 5, 5, 10, 8, 6, 10, 4]
    model = DCNv2(
        field_dims, 10,
        DCNv2Config(d_embed=4, n_cross_layers=2, n_experts=2, d_low_rank=8,
                    dnn_hidden_dims=(16,), dropout_rate=0.0),
        rngs=rngs,
    )
    model.eval()

    x = DeepFMInput(
        user_cat=jnp.ones((4, 3), dtype=jnp.int32),
        user_num=jnp.ones((4, 8)),
        item_cat=jnp.ones((4, 5), dtype=jnp.int32),
        item_num=jnp.ones((4, 2)),
    )

    direct = model(x)
    stacked = model.embed(x)
    split = model.predict_from_embedding(stacked)

    assert jnp.allclose(direct, split, atol=1e-5)


# --- LightGCN ---

def test_lightgcn_embed_predict_equals_call(rngs):
    """LightGCN: embed() + predict_from_embedding() == __call__()."""
    n_users, n_items = 10, 20
    u_idx = np.array([0, 1, 2], dtype=np.int32)
    i_idx = np.array([0, 1, 2], dtype=np.int32)
    adj = build_normalized_adj(u_idx, i_idx, n_users, n_items)
    model = LightGCN(n_users, n_items, adj, LightGCNConfig(d_embed=8, n_layers=2), rngs=rngs)
    model.eval()

    x = LightGCNInput(
        user_idx=jnp.array([0, 1], dtype=jnp.int32),
        item_idx=jnp.array([0, 1], dtype=jnp.int32),
    )

    direct = model(x)
    u, i = model.embed(x)
    split = model.predict_from_embedding(u, i)

    assert jnp.allclose(direct, split, atol=1e-5)


# --- DIN ---

def test_din_embed_predict_equals_call(rngs):
    """DIN: embed() + predict_from_embedding() == __call__()."""
    field_dims = [8, 5, 5, 10, 8, 6, 10, 4]
    model = DIN(
        field_dims, 10, n_items=20, max_seq_len=5,
        config=DINConfig(d_embed=4, attention_hidden_dims=(8,), dnn_hidden_dims=(16,), dropout_rate=0.0),
        rngs=rngs,
    )
    model.eval()

    x = DINInput(
        user_cat=jnp.ones((2, 3), dtype=jnp.int32),
        user_num=jnp.ones((2, 8)),
        item_cat=jnp.ones((2, 5), dtype=jnp.int32),
        item_num=jnp.ones((2, 2)),
        history=jnp.ones((2, 5), dtype=jnp.int32),
        hist_len=jnp.array([3, 4], dtype=jnp.int32),
    )

    direct = model(x)
    ui, tq, sf = model.embed(x)
    split = model.predict_from_embedding(ui, tq, sf)

    assert jnp.allclose(direct, split, atol=1e-5)


# --- SASRec ---

def test_sasrec_embed_predict_equals_call(rngs):
    """SASRec: embed() + predict_from_embedding() == __call__()."""
    model = SASRec(
        n_items=20,
        config=SASRecConfig(d_embed=8, n_heads=2, n_blocks=1, max_seq_len=5, dropout_rate=0.0),
        rngs=rngs,
    )
    model.eval()

    x = SASRecInput(
        history=jnp.array([[1, 2, 3, 0, 0]], dtype=jnp.int32),
        hist_len=jnp.array([3], dtype=jnp.int32),
    )
    target = jnp.array([5], dtype=jnp.int32)

    direct = model(x, target)
    u_emb, t_emb = model.embed(x, target)
    split = model.predict_from_embedding(u_emb, t_emb)

    assert jnp.allclose(direct, split, atol=1e-5)

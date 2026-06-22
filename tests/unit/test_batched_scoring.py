"""Track B regression: batched full-catalog scoring == per-user scoring.

The validation bottleneck fix replaced the per-user (batch_size=1) scoring loop
with a vectorized chunked scorer in src/training/trainer.generate_predictions.
These tests assert the batched path returns the SAME top-K recommendations as the
reference per-user score_full_catalog, across all four scorer branches
(feature-based, DIN feature+history, graph, sequential).
"""

from __future__ import annotations

import numpy as np
import pytest
from flax import nnx

from src.config import DeepFMConfig, DINConfig, LightGCNConfig, SASRecConfig
from src.models.deepfm import DeepFM
from src.models.din import DIN
from src.models.lightgcn import LightGCN, build_normalized_adj
from src.models.sasrec import SASRec
from src.training.trainer import generate_predictions, score_full_catalog

N_USERS = 6
N_ITEMS = 20
K = 5
USER_CAT_DIMS = [5, 3, 3]
ITEM_CAT_DIMS = [10, 6, 4, 8, 3]
FIELD_DIMS = USER_CAT_DIMS + ITEM_CAT_DIMS
N_NUMERICAL = 10  # 8 user + 2 item
T = 5


def _rand_cat(rng, n_rows, dims):
    return np.stack([rng.integers(0, d, size=n_rows) for d in dims], axis=1).astype(np.int32)


def _feature_dicts(rng):
    user_features = {
        "categorical": _rand_cat(rng, N_USERS, USER_CAT_DIMS),
        "numerical": rng.random((N_USERS, 8)).astype(np.float32),
    }
    item_features = {
        "categorical": _rand_cat(rng, N_ITEMS, ITEM_CAT_DIMS),
        "numerical": rng.random((N_ITEMS, 2)).astype(np.float32),
    }
    return user_features, item_features


def _per_user(model, backbone, uf, itf, seq, lens):
    out = {}
    for j in range(N_USERS):
        idxs = score_full_catalog(
            model, j, uf, itf, k=K, backbone_name=backbone,
            sequences=seq, seq_lengths=lens,
        )
        out[f"u{j}"] = [str(int(i)) for i in idxs]
    return out


def _batched(model, backbone, uf, itf, seq, lens, batch_size):
    user_ids = [f"u{j}" for j in range(N_USERS)]
    user_to_idx = {f"u{j}": j for j in range(N_USERS)}
    idx_to_item = {i: str(i) for i in range(N_ITEMS)}
    return generate_predictions(
        model, user_ids, uf, itf, user_to_idx, idx_to_item, k=K,
        backbone_name=backbone, sequences=seq, seq_lengths=lens, batch_size=batch_size,
    )


def _assert_equal(per_user, batched):
    assert set(per_user) == set(batched)
    for uid in per_user:
        assert per_user[uid] == batched[uid], (
            f"{uid}: per-user {per_user[uid]} != batched {batched[uid]}"
        )


@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_deepfm_batched_equals_per_user(batch_size):
    rng = np.random.default_rng(0)
    uf, itf = _feature_dicts(rng)
    config = DeepFMConfig(d_embed=4, dnn_hidden_dims=(8, 4), dropout_rate=0.0, use_batch_norm=False)
    model = DeepFM(FIELD_DIMS, N_NUMERICAL, config, rngs=nnx.Rngs(params=1, dropout=2))
    pu = _per_user(model, "deepfm", uf, itf, None, None)
    bt = _batched(model, "deepfm", uf, itf, None, None, batch_size)
    _assert_equal(pu, bt)


@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_deepfm_id_embed_batched_equals_per_user(batch_size):
    """With id embeddings on, batched top-K must still match per-user top-K.

    Asserts the (n_batch * n_items) idx layout in _score_users_chunk matches the
    per-user broadcast in _score_full_catalog_features.
    """
    rng = np.random.default_rng(7)
    uf, itf = _feature_dicts(rng)
    config = DeepFMConfig(
        d_embed=4, dnn_hidden_dims=(8, 4), dropout_rate=0.0, use_batch_norm=False,
        use_id_embed=True, n_users=N_USERS, n_items=N_ITEMS,
    )
    model = DeepFM(FIELD_DIMS, N_NUMERICAL, config, rngs=nnx.Rngs(params=1, dropout=2))
    assert model._use_id_embed
    pu = _per_user(model, "deepfm", uf, itf, None, None)
    bt = _batched(model, "deepfm", uf, itf, None, None, batch_size)
    _assert_equal(pu, bt)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_din_batched_equals_per_user(batch_size):
    rng = np.random.default_rng(1)
    uf, itf = _feature_dicts(rng)
    seq = rng.integers(1, N_ITEMS, size=(N_USERS, T)).astype(np.int32)
    lens = rng.integers(1, T + 1, size=N_USERS).astype(np.int32)
    config = DINConfig(d_embed=4, attention_hidden_dims=(8, 4), dnn_hidden_dims=(8, 4),
                       dropout_rate=0.0, use_batch_norm=False)
    model = DIN(field_dims=FIELD_DIMS, n_numerical=N_NUMERICAL, n_items=N_ITEMS,
                max_seq_len=T, config=config, rngs=nnx.Rngs(params=1, dropout=2))
    pu = _per_user(model, "din", uf, itf, seq, lens)
    bt = _batched(model, "din", uf, itf, seq, lens, batch_size)
    _assert_equal(pu, bt)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_lightgcn_batched_equals_per_user(batch_size):
    rng = np.random.default_rng(2)
    uf, itf = _feature_dicts(rng)  # only item_features["categorical"].shape[0] used
    edges = 30
    u_idx = rng.integers(0, N_USERS, size=edges).astype(np.int32)
    i_idx = rng.integers(0, N_ITEMS, size=edges).astype(np.int32)
    adj = build_normalized_adj(u_idx, i_idx, N_USERS, N_ITEMS)
    config = LightGCNConfig(d_embed=8, n_layers=2)
    model = LightGCN(N_USERS, N_ITEMS, adj, config, rngs=nnx.Rngs(params=1, dropout=2))
    pu = _per_user(model, "lightgcn", uf, itf, None, None)
    bt = _batched(model, "lightgcn", uf, itf, None, None, batch_size)
    _assert_equal(pu, bt)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_sasrec_batched_equals_per_user(batch_size):
    rng = np.random.default_rng(3)
    uf, itf = _feature_dicts(rng)  # only item_features["categorical"].shape[0] used
    seq = np.zeros((N_USERS, T), dtype=np.int32)
    for j in range(N_USERS):
        n = rng.integers(1, T + 1)
        seq[j, T - n:] = rng.integers(1, N_ITEMS, size=n)
    lens = (seq > 0).sum(axis=1).astype(np.int32)
    config = SASRecConfig(d_embed=16, n_heads=2, n_blocks=1, max_seq_len=T, dropout_rate=0.0)
    model = SASRec(n_items=N_ITEMS, config=config, rngs=nnx.Rngs(params=1, dropout=2))
    pu = _per_user(model, "sasrec", uf, itf, seq, lens)
    bt = _batched(model, "sasrec", uf, itf, seq, lens, batch_size)
    _assert_equal(pu, bt)

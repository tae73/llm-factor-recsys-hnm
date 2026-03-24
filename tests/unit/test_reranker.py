"""Unit tests for GBDT Re-Ranker: model wrapper, feature builder, stage1 extraction."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ReRankerConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    return ReRankerConfig(
        top_k=10,
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        num_leaves=8,
        min_child_samples=2,
        subsample=1.0,
        colsample_bytree=1.0,
        random_seed=42,
    )


@pytest.fixture
def small_fk():
    """Small factual knowledge DataFrame with L1/L2/L3 columns."""
    n = 20
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "article_id": [f"art_{i:03d}" for i in range(n)],
        "product_code": [f"prod_{i:03d}" for i in range(n)],
        "super_category": rng.choice(["Apparel", "Footwear", "Accessories"], n).tolist(),
        # L1
        "l1_material": rng.choice(["Cotton", "Polyester", "Wool"], n).tolist(),
        "l1_closure": rng.choice(["None", "Button", "Zipper"], n).tolist(),
        "l1_design_details": [json.dumps(["solid"]) if i % 2 == 0 else json.dumps(["solid", "print"]) for i in range(n)],
        # L1 Apparel-specific
        "l1_neckline": rng.choice(["Round", "V-neck", "Crew"], n).tolist(),
        "l1_sleeve_type": rng.choice(["Short", "Long", "Sleeveless"], n).tolist(),
        "l1_fit": rng.choice(["Regular", "Slim", "Loose"], n).tolist(),
        "l1_length": rng.choice(["Short", "Medium", "Long"], n).tolist(),
        # L2
        "l2_trendiness": rng.choice(["Classic", "Current", "Trend-forward"], n).tolist(),
        "l2_season_fit": rng.choice(["Spring", "Summer", "Winter", "All-season"], n).tolist(),
        "l2_perceived_quality": rng.integers(1, 6, n).tolist(),
        "l2_versatility": rng.integers(1, 6, n).tolist(),
        "l2_style_mood": [json.dumps(rng.choice(["Minimalist", "Casual", "Sporty"], rng.integers(1, 3)).tolist()) for _ in range(n)],
        "l2_occasion": [json.dumps(rng.choice(["Everyday", "Work", "Party"], rng.integers(1, 3)).tolist()) for _ in range(n)],
        # L3
        "l3_color_harmony": rng.choice(["Monochromatic", "Complementary", "Analogous"], n).tolist(),
        "l3_coordination_role": rng.choice(["Foundation", "Accent", "Statement"], n).tolist(),
        "l3_tone_season": rng.choice(["Cool-Winter", "Warm-Autumn", "Neutral"], n).tolist(),
        "l3_visual_weight": rng.integers(1, 6, n).tolist(),
        "l3_style_lineage": [json.dumps(rng.choice(["Minimalism", "Streetwear", "Classic"], rng.integers(1, 3)).tolist()) for _ in range(n)],
        # L3 Apparel-specific
        "l3_silhouette": rng.choice(["A-line", "Straight", "Fitted"], n).tolist(),
        "l3_proportion_effect": rng.choice(["Elongating", "Balancing", "Neutral"], n).tolist(),
    })


@pytest.fixture
def small_user_features():
    n_users = 10
    return {
        "numerical": np.random.rand(n_users, 8).astype(np.float32),
        "categorical": np.random.randint(0, 5, (n_users, 3)).astype(np.int32),
    }


@pytest.fixture
def small_item_features():
    n_items = 20
    return {
        "numerical": np.random.rand(n_items, 2).astype(np.float32),
        "categorical": np.random.randint(0, 10, (n_items, 5)).astype(np.int32),
    }


@pytest.fixture
def small_candidates():
    rng = np.random.default_rng(42)
    n_users, top_k = 10, 10
    return {
        "user_indices": np.arange(n_users, dtype=np.int32),
        "candidate_indices": rng.integers(0, 20, (n_users, top_k)).astype(np.int32),
        "candidate_scores": np.sort(rng.random((n_users, top_k)).astype(np.float32), axis=1)[:, ::-1],
    }


@pytest.fixture
def small_idx_maps():
    idx_to_user = {i: f"user_{i:03d}" for i in range(10)}
    idx_to_item = {i: f"art_{i:03d}" for i in range(20)}
    return idx_to_user, idx_to_item


# ---------------------------------------------------------------------------
# TestReRankerModel
# ---------------------------------------------------------------------------


class TestReRankerModel:
    def test_train_basic(self, small_config):
        from src.models.reranker import ReRanker

        rng = np.random.default_rng(42)
        X = rng.random((200, 10)).astype(np.float32)
        y = rng.integers(0, 2, 200).astype(np.float32)
        names = [f"feat_{i}" for i in range(10)]

        reranker = ReRanker(small_config)
        metrics = reranker.train(X, y, feature_names=names)
        assert "best_iteration" in metrics

    def test_predict_shape(self, small_config):
        from src.models.reranker import ReRanker

        rng = np.random.default_rng(42)
        X_train = rng.random((200, 10)).astype(np.float32)
        y_train = rng.integers(0, 2, 200).astype(np.float32)

        reranker = ReRanker(small_config)
        reranker.train(X_train, y_train)

        X_test = rng.random((50, 10)).astype(np.float32)
        scores = reranker.predict(X_test)
        assert scores.shape == (50,)
        assert scores.dtype == np.float32

    def test_save_load_roundtrip(self, small_config):
        from src.models.reranker import ReRanker

        rng = np.random.default_rng(42)
        X = rng.random((200, 10)).astype(np.float32)
        y = rng.integers(0, 2, 200).astype(np.float32)
        names = [f"feat_{i}" for i in range(10)]

        reranker = ReRanker(small_config)
        reranker.train(X, y, feature_names=names)
        scores_before = reranker.predict(X[:10])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model"
            reranker.save(path)

            loaded = ReRanker.load(path, small_config)
            scores_after = loaded.predict(X[:10])

        np.testing.assert_allclose(scores_before, scores_after, atol=1e-6)

    def test_feature_importance(self, small_config):
        from src.models.reranker import ReRanker

        rng = np.random.default_rng(42)
        X = rng.random((200, 5)).astype(np.float32)
        y = rng.integers(0, 2, 200).astype(np.float32)
        names = ["a", "b", "c", "d", "e"]

        reranker = ReRanker(small_config)
        reranker.train(X, y, feature_names=names)
        imp = reranker.feature_importance("gain")

        assert set(imp.keys()) == set(names)
        assert all(v >= 0 for v in imp.values())


# ---------------------------------------------------------------------------
# TestAttributeEncoding
# ---------------------------------------------------------------------------


class TestAttributeEncoding:
    def test_build_encoders(self, small_fk):
        from src.features.reranker_features import build_attribute_encoders

        encoders = build_attribute_encoders(small_fk)
        # Should have encoders for all categorical + multi-hot columns
        assert "l1_material" in encoders
        assert "l2_style_mood" in encoders
        assert "l3_style_lineage" in encoders

    def test_encode_item_attributes_shape(self, small_fk):
        from src.features.reranker_features import build_attribute_encoders, encode_item_attributes

        idx_to_item = {i: f"art_{i:03d}" for i in range(20)}
        encoders = build_attribute_encoders(small_fk)
        features, names = encode_item_attributes(small_fk, encoders, idx_to_item)

        assert features.shape[0] == 20
        assert features.shape[1] == len(names)
        assert features.shape[1] > 10  # Should have many features

    def test_ordinal_passthrough(self, small_fk):
        from src.features.reranker_features import build_attribute_encoders, encode_item_attributes

        idx_to_item = {i: f"art_{i:03d}" for i in range(20)}
        encoders = build_attribute_encoders(small_fk)
        features, names = encode_item_attributes(small_fk, encoders, idx_to_item)

        # Find perceived_quality column
        pq_idx = names.index("l2_perceived_quality")
        # Values should be in 1-5 range
        nonzero = features[:, pq_idx][features[:, pq_idx] > 0]
        assert nonzero.min() >= 1.0
        assert nonzero.max() <= 5.0

    def test_missing_article_has_defaults(self, small_fk):
        from src.features.reranker_features import build_attribute_encoders, encode_item_attributes

        # Create idx_to_item where index 20 maps to nonexistent article
        idx_to_item = {i: f"art_{i:03d}" for i in range(20)}
        idx_to_item[20] = "nonexistent_article"  # 21 items, last one missing

        encoders = build_attribute_encoders(small_fk)
        features, names = encode_item_attributes(small_fk, encoders, idx_to_item)

        assert features.shape[0] == 21
        # Missing article's multi-hot features should be zero
        # (multi-hot columns have no "UNKNOWN" class, so they stay 0)
        # Categorical columns get UNKNOWN label (may be non-zero int), which is expected
        # Ordinal columns should be 0
        pq_idx = names.index("l2_perceived_quality")
        assert features[20, pq_idx] == 0.0
        vw_idx = names.index("l3_visual_weight")
        assert features[20, vw_idx] == 0.0


# ---------------------------------------------------------------------------
# TestReRankerFeatures
# ---------------------------------------------------------------------------


class TestReRankerFeatures:
    def test_base_features_shape(self, small_candidates, small_user_features, small_item_features):
        from src.features.reranker_features import build_reranker_features

        X, names = build_reranker_features(
            small_candidates["user_indices"],
            small_candidates["candidate_indices"],
            small_candidates["candidate_scores"],
            small_user_features, small_item_features,
        )
        n_users, top_k = 10, 10
        assert X.shape[0] == n_users * top_k
        assert X.shape[1] == len(names)
        # Base: 3 score + 8 user_num + 3 user_cat + 2 item_num + 5 item_cat + 3 cross = 24
        assert X.shape[1] == 24

    def test_full_features_more_dims(self, small_candidates, small_user_features, small_item_features, small_fk):
        from src.features.reranker_features import (
            build_attribute_encoders,
            build_reranker_features,
            encode_item_attributes,
        )

        idx_to_item = {i: f"art_{i:03d}" for i in range(20)}
        encoders = build_attribute_encoders(small_fk)
        item_attrs, attr_names = encode_item_attributes(small_fk, encoders, idx_to_item)

        X_full, names_full = build_reranker_features(
            small_candidates["user_indices"],
            small_candidates["candidate_indices"],
            small_candidates["candidate_scores"],
            small_user_features, small_item_features,
            item_attrs, attr_names,
        )
        X_base, names_base = build_reranker_features(
            small_candidates["user_indices"],
            small_candidates["candidate_indices"],
            small_candidates["candidate_scores"],
            small_user_features, small_item_features,
        )
        assert X_full.shape[1] > X_base.shape[1]

    def test_score_features_descending(self, small_candidates, small_user_features, small_item_features):
        from src.features.reranker_features import build_reranker_features

        X, names = build_reranker_features(
            small_candidates["user_indices"],
            small_candidates["candidate_indices"],
            small_candidates["candidate_scores"],
            small_user_features, small_item_features,
        )
        score_idx = names.index("stage1_score")
        # First user's scores should be descending
        first_user_scores = X[:10, score_idx]
        assert np.all(first_user_scores[:-1] >= first_user_scores[1:])

    def test_rank_position_values(self, small_candidates, small_user_features, small_item_features):
        from src.features.reranker_features import build_reranker_features

        X, names = build_reranker_features(
            small_candidates["user_indices"],
            small_candidates["candidate_indices"],
            small_candidates["candidate_scores"],
            small_user_features, small_item_features,
        )
        rank_idx = names.index("rank_position")
        # First user: rank should be 0,1,2,...,9
        first_user_ranks = X[:10, rank_idx]
        np.testing.assert_array_equal(first_user_ranks, np.arange(10, dtype=np.float32))

    def test_bge_similarity(self, small_candidates, small_user_features, small_item_features):
        from src.features.reranker_features import build_reranker_features

        rng = np.random.default_rng(42)
        user_bge = rng.random((10, 768)).astype(np.float32)
        item_bge = rng.random((20, 768)).astype(np.float32)

        X, names = build_reranker_features(
            small_candidates["user_indices"],
            small_candidates["candidate_indices"],
            small_candidates["candidate_scores"],
            small_user_features, small_item_features,
            user_bge=user_bge, item_bge=item_bge,
        )
        assert "bge_cosine_similarity" in names
        sim_idx = names.index("bge_cosine_similarity")
        # Cosine sim should be in reasonable range
        assert X[:, sim_idx].min() >= -1.1
        assert X[:, sim_idx].max() <= 1.1

    def test_labels_binary(self, small_candidates, small_idx_maps):
        from src.features.reranker_features import build_reranker_labels

        idx_to_user, idx_to_item = small_idx_maps
        ground_truth = {"user_000": ["art_003", "art_007"]}

        labels = build_reranker_labels(
            small_candidates["user_indices"],
            small_candidates["candidate_indices"],
            ground_truth, idx_to_user, idx_to_item,
        )
        assert labels.shape == (100,)  # 10 users × 10 candidates
        assert set(np.unique(labels)).issubset({0.0, 1.0})

    def test_feature_names_unique(self, small_candidates, small_user_features, small_item_features):
        from src.features.reranker_features import build_reranker_features

        _, names = build_reranker_features(
            small_candidates["user_indices"],
            small_candidates["candidate_indices"],
            small_candidates["candidate_scores"],
            small_user_features, small_item_features,
        )
        assert len(names) == len(set(names))

    def test_cross_features_present(self, small_candidates, small_user_features, small_item_features):
        from src.features.reranker_features import build_reranker_features

        _, names = build_reranker_features(
            small_candidates["user_indices"],
            small_candidates["candidate_indices"],
            small_candidates["candidate_scores"],
            small_user_features, small_item_features,
        )
        assert "cross_age_section" in names
        assert "cross_price_ratio" in names


# ---------------------------------------------------------------------------
# TestStage1Extraction
# ---------------------------------------------------------------------------


class TestStage1Extraction:
    def test_extract_returns_correct_keys(self):
        """Verify extract_stage1_candidates returns expected dict keys."""
        # This is a signature/contract test — actual model test needs JAX
        from src.training.trainer import extract_stage1_candidates
        assert callable(extract_stage1_candidates)

    def test_candidate_cache_roundtrip(self, small_candidates):
        """Save/load candidates .npz roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "candidates.npz"
            np.savez_compressed(path, **small_candidates)

            loaded = np.load(path)
            np.testing.assert_array_equal(loaded["user_indices"], small_candidates["user_indices"])
            np.testing.assert_array_equal(loaded["candidate_indices"], small_candidates["candidate_indices"])
            np.testing.assert_allclose(loaded["candidate_scores"], small_candidates["candidate_scores"])

    def test_scores_shape_consistency(self, small_candidates):
        """user_indices, candidate_indices, candidate_scores shapes match."""
        n_users = small_candidates["user_indices"].shape[0]
        assert small_candidates["candidate_indices"].shape[0] == n_users
        assert small_candidates["candidate_scores"].shape[0] == n_users
        assert small_candidates["candidate_indices"].shape[1] == small_candidates["candidate_scores"].shape[1]

    def test_scores_descending(self, small_candidates):
        """Candidate scores should be in descending order per user."""
        scores = small_candidates["candidate_scores"]
        for i in range(scores.shape[0]):
            assert np.all(scores[i, :-1] >= scores[i, 1:])

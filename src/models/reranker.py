"""LightGBM-based GBDT Re-Ranker for 2-stage recommendation.

NOT a JAX/Flax model — uses scikit-learn compatible API.
NOT registered in BackboneRegistry (different paradigm: Stage 2 re-ranking).

Usage:
    reranker = ReRanker(config)
    result = reranker.train(X_train, y_train, X_val, y_val, feature_names)
    scores = reranker.predict(X_test)
    reranker.save(path)
    reranker = ReRanker.load(path)
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.config import ReRankerConfig


class ReRanker:
    """LightGBM wrapper for recommendation re-ranking.

    Args:
        config: ReRankerConfig with hyperparameters.
    """

    def __init__(self, config: ReRankerConfig = ReRankerConfig()):
        self._config = config
        self._model: lgb.LGBMClassifier | None = None
        self._booster: lgb.Booster | None = None
        self._feature_names: list[str] = []

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Train LightGBM classifier with early stopping.

        Args:
            X_train: (N, D) training features.
            y_train: (N,) binary labels.
            X_val: (M, D) validation features or None.
            y_val: (M,) validation labels or None.
            feature_names: Feature names for importance analysis.

        Returns:
            dict with training metrics (best_iteration, val_auc).
        """
        cfg = self._config
        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        self._model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            is_unbalance=True,
            random_state=cfg.random_seed,
            verbosity=-1,
        )

        fit_kwargs: dict = {"feature_name": self._feature_names}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_metric"] = "auc"
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=50),
            ]

        self._model.fit(X_train, y_train, **fit_kwargs)

        best_iter = self._model.best_iteration_ if hasattr(self._model, "best_iteration_") else cfg.n_estimators
        val_auc = 0.0
        if hasattr(self._model, "best_score_") and self._model.best_score_:
            val_scores = self._model.best_score_.get("valid_0", {})
            val_auc = val_scores.get("auc", 0.0)

        return {"best_iteration": best_iter, "val_auc": val_auc}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return prediction scores (N,) for re-ranking.

        Uses predict_proba positive class probability, or raw booster for loaded models.
        """
        if self._model is None and self._booster is None:
            raise RuntimeError("Model not trained. Call train() first.")
        if self._booster is not None:
            # Loaded model — use booster directly (sigmoid applied)
            raw = self._booster.predict(X)
            return raw.astype(np.float32)
        return self._model.predict_proba(X)[:, 1].astype(np.float32)

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """Return feature importance as {name: importance}.

        Args:
            importance_type: "gain" or "split".
        """
        booster = self._booster or (self._model.booster_ if self._model else None)
        if booster is None:
            raise RuntimeError("Model not trained. Call train() first.")
        importances = booster.feature_importance(importance_type=importance_type)
        return dict(zip(self._feature_names, map(float, importances)))

    def save(self, path: Path) -> None:
        """Save model to booster.txt (portable LightGBM format)."""
        booster = self._booster or (self._model.booster_ if self._model else None)
        if booster is None:
            raise RuntimeError("Model not trained. Call train() first.")
        path.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(path / "booster.txt"))
        # Save feature names
        import json
        (path / "feature_names.json").write_text(json.dumps(self._feature_names))

    @classmethod
    def load(cls, path: Path, config: ReRankerConfig = ReRankerConfig()) -> ReRanker:
        """Load model from booster.txt."""
        import json
        instance = cls(config)
        instance._booster = lgb.Booster(model_file=str(path / "booster.txt"))
        feature_names_path = path / "feature_names.json"
        if feature_names_path.exists():
            instance._feature_names = json.loads(feature_names_path.read_text())
        return instance

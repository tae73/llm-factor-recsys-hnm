"""UserKNN baseline using implicit ALS (Alternating Least Squares).

Wraps the implicit library's AlternatingLeastSquares model.
"""

from implicit.als import AlternatingLeastSquares

from src.config import BaselineConfig, InteractionData


def train_als(
    interaction_data: InteractionData,
    config: BaselineConfig = BaselineConfig(),
) -> AlternatingLeastSquares:
    """Train ALS model on the interaction matrix.

    implicit >= 0.5 expects user-item matrix for .fit() and .recommend().
    """
    model = AlternatingLeastSquares(
        factors=config.als_factors,
        regularization=config.als_regularization,
        iterations=config.als_iterations,
        random_state=42,
    )
    model.fit(interaction_data.matrix)
    return model

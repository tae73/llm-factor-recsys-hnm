"""BPR-MF baseline using implicit's BayesianPersonalizedRanking.

Wraps the implicit library's BPR model for pairwise learning.
"""

from implicit.bpr import BayesianPersonalizedRanking

from src.config import BaselineConfig, InteractionData


def train_bpr(
    interaction_data: InteractionData,
    config: BaselineConfig = BaselineConfig(),
) -> BayesianPersonalizedRanking:
    """Train BPR model on the interaction matrix.

    implicit >= 0.5 expects user-item matrix for .fit() and .recommend().
    """
    model = BayesianPersonalizedRanking(
        factors=config.bpr_factors,
        learning_rate=config.bpr_learning_rate,
        iterations=config.bpr_iterations,
        random_state=42,
    )
    model.fit(interaction_data.matrix)
    return model

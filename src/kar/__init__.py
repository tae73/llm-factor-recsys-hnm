"""KAR (Knowledge-Augmented Recommendation) module.

Implements the 2-Expert architecture from Xi et al. 2023:
- Expert: MLP encoding BGE embeddings to d_rec
- Gating: Dynamic weighting of factual/reasoning experts (G1-G4)
- Fusion: Merging augmented vector into backbone embedding (F1-F4)
- KARModel: Composition wrapper (backbone + experts + gating + fusion)
"""

from src.kar.expert import Expert
from src.kar.fusion import (
    F1ConcatFusion,
    F2AdditionFusion,
    F3GatedFusion,
    F4CrossAttentionFusion,
    create_fusion,
)
from src.kar.gating import (
    G1FixedGating,
    G2ExpertGating,
    G3ContextGating,
    G4CrossGating,
    create_gating,
)
from src.kar.hybrid import KARInput, KARModel, compute_d_backbone

__all__ = [
    "Expert",
    "G1FixedGating",
    "G2ExpertGating",
    "G3ContextGating",
    "G4CrossGating",
    "create_gating",
    "F1ConcatFusion",
    "F2AdditionFusion",
    "F3GatedFusion",
    "F4CrossAttentionFusion",
    "create_fusion",
    "KARInput",
    "KARModel",
    "compute_d_backbone",
]

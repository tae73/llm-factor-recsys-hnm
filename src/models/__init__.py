"""Recommendation model backbones.

Provides a registry mapping backbone names to their model, input, and config classes.
"""

from __future__ import annotations

from typing import NamedTuple

from src.config import DCNv2Config, DeepFMConfig, DINConfig, LightGCNConfig, SASRecConfig
from src.models.dcnv2 import DCNv2
from src.models.deepfm import DeepFM, DeepFMInput
from src.models.din import DIN, DINInput
from src.models.lightgcn import LightGCN, LightGCNInput
from src.models.sasrec import SASRec, SASRecInput


class BackboneSpec(NamedTuple):
    """Specification for a recommendation backbone."""

    model_cls: type
    input_cls: type
    config_cls: type
    needs_graph: bool = False  # True for graph-based models (LightGCN)
    needs_sequence: bool = False  # True for sequential models (DIN, SASRec)


BACKBONE_REGISTRY: dict[str, BackboneSpec] = {
    "deepfm": BackboneSpec(DeepFM, DeepFMInput, DeepFMConfig),
    "dcnv2": BackboneSpec(DCNv2, DeepFMInput, DCNv2Config),
    "lightgcn": BackboneSpec(LightGCN, LightGCNInput, LightGCNConfig, needs_graph=True),
    "din": BackboneSpec(DIN, DINInput, DINConfig, needs_sequence=True),
    "sasrec": BackboneSpec(SASRec, SASRecInput, SASRecConfig, needs_sequence=True),
}


def get_backbone(name: str) -> BackboneSpec:
    """Look up a backbone by name.

    Args:
        name: Backbone name (deepfm, dcnv2, lightgcn, din, sasrec).

    Returns:
        BackboneSpec with model_cls, input_cls, config_cls, needs_graph, needs_sequence.

    Raises:
        ValueError: If name not in registry.
    """
    if name not in BACKBONE_REGISTRY:
        valid = ", ".join(sorted(BACKBONE_REGISTRY.keys()))
        raise ValueError(f"Unknown backbone '{name}'. Choose from: {valid}")
    return BACKBONE_REGISTRY[name]


def is_kar_model(model) -> bool:
    """Check if a model is a KARModel wrapper.

    Args:
        model: Any nnx.Module.

    Returns:
        True if the model is a KARModel instance.
    """
    return type(model).__name__ == "KARModel"

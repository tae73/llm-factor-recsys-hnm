"""Factual knowledge extraction: L1 (product) + L2 (perceptual) + L3 (theory) attributes.

Per-item integrated extraction via GPT-4.1-nano with Structured Outputs.
"""

from src.knowledge.factual.cache import ProductCodeCache
from src.knowledge.factual.text_composer import LAYER_COMBOS, construct_factual_text
from src.knowledge.factual.validator import ValidationResult, validate_knowledge

__all__ = [
    "ProductCodeCache",
    "LAYER_COMBOS",
    "ValidationResult",
    "construct_factual_text",
    "validate_knowledge",
]

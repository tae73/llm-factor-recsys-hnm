"""User-level reasoning knowledge: L1 aggregation + L2/L3 LLM Factorization Prompting.

Produces per-user reasoning_text for the KAR Reasoning Expert input.
"""

from src.knowledge.reasoning.cache import CustomerCache
from src.knowledge.reasoning.prompts import compose_reasoning_text

__all__ = [
    "CustomerCache",
    "compose_reasoning_text",
]

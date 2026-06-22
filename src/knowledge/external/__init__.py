"""External (open-world styling) knowledge extraction.

Realizes KAR's open-world premise (validated in probe_16 / probe_19): per
PRODUCT_CODE, a single GPT-4.1-nano structured-output call extracts BOTH a prose
complement description and structured complement attributes (product_type /
colour / material / style). The structured part is rendered into L1's
attribute-list genre (format-unified) so BGE encodes it coherently with L1.

Importing this package has NO API side effects.
"""

from src.knowledge.external.extractor import (
    estimate_external_cost,
    extract_external_knowledge,
    group_representatives,
    load_checkpoint,
    propagate_to_articles,
    save_checkpoint,
)
from src.knowledge.external.prompts import (
    JSON_SCHEMA,
    SYSTEM_PROMPT,
    build_messages,
    render_external_text,
    render_structured_text,
)

__all__ = [
    "estimate_external_cost",
    "extract_external_knowledge",
    "group_representatives",
    "load_checkpoint",
    "propagate_to_articles",
    "save_checkpoint",
    "JSON_SCHEMA",
    "SYSTEM_PROMPT",
    "build_messages",
    "render_external_text",
    "render_structured_text",
]

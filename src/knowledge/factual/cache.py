"""Product-code-based extraction result cache with Parquet checkpoints.

~47K unique product_codes map to ~105K article SKUs.
Cache operates at the product_code level to avoid redundant API calls.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ProductCodeCache:
    """In-memory dict cache backed by Parquet checkpoints.

    Usage:
        cache = ProductCodeCache(checkpoint_dir=Path("data/knowledge/factual/checkpoint"))
        loaded = cache.load_checkpoint()  # Resume from last checkpoint
        if cache.get("0123456") is None:
            knowledge = extract(...)
            cache.put("0123456", knowledge)
        cache.save_checkpoint()
    """

    def __init__(self, checkpoint_dir: Path | None = None) -> None:
        self._store: dict[str, dict] = {}
        self._checkpoint_dir = checkpoint_dir
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get(self, product_code: str) -> dict | None:
        """Return cached knowledge for product_code, or None."""
        return self._store.get(product_code)

    def put(self, product_code: str, knowledge: dict) -> None:
        """Store extracted knowledge for a product_code."""
        self._store[product_code] = knowledge

    @property
    def size(self) -> int:
        """Number of cached product_codes."""
        return len(self._store)

    def save_checkpoint(self) -> None:
        """Persist current cache to a Parquet checkpoint file."""
        if self._checkpoint_dir is None:
            logger.warning("No checkpoint_dir configured; skipping save.")
            return
        if not self._store:
            return

        # Convert to DataFrame: each dict is a row
        rows = []
        for product_code, knowledge in self._store.items():
            row = {"product_code": product_code}
            for k, v in knowledge.items():
                # Serialize lists/dicts as JSON strings for Parquet
                row[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
            rows.append(row)

        df = pd.DataFrame(rows)
        checkpoint_path = self._checkpoint_dir / "checkpoint.parquet"
        df.to_parquet(checkpoint_path, index=False)
        logger.info("Saved checkpoint: %d product_codes → %s", len(rows), checkpoint_path)

    def load_checkpoint(self) -> int:
        """Load cache from Parquet checkpoint. Returns number of items loaded."""
        if self._checkpoint_dir is None:
            return 0

        checkpoint_path = self._checkpoint_dir / "checkpoint.parquet"
        if not checkpoint_path.exists():
            return 0

        df = pd.read_parquet(checkpoint_path)
        loaded = 0
        for _, row in df.iterrows():
            product_code = row["product_code"]
            knowledge = {}
            for col in df.columns:
                if col == "product_code":
                    continue
                val = row[col]
                if pd.isna(val):
                    knowledge[col] = None
                elif isinstance(val, str) and val.startswith(("[", "{")):
                    try:
                        knowledge[col] = json.loads(val)
                    except json.JSONDecodeError:
                        knowledge[col] = val
                else:
                    knowledge[col] = val
            self._store[product_code] = knowledge
            loaded += 1

        logger.info("Loaded checkpoint: %d product_codes from %s", loaded, checkpoint_path)
        return loaded

    def keys(self) -> set[str]:
        """Return set of cached product_codes."""
        return set(self._store.keys())

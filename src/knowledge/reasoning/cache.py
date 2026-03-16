"""Customer-ID-based profiling result cache with Parquet checkpoints.

~876K active users + ~421K sparse users = ~1.3M total.
Cache operates at the customer_id level for resume support.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class CustomerCache:
    """In-memory dict cache backed by Parquet checkpoints.

    Usage:
        cache = CustomerCache(checkpoint_dir=Path("data/knowledge/reasoning/checkpoint"))
        loaded = cache.load_checkpoint()
        if cache.get("abc123") is None:
            profile = extract(...)
            cache.put("abc123", profile)
        cache.save_checkpoint()
    """

    def __init__(self, checkpoint_dir: Path | None = None) -> None:
        self._store: dict[str, dict] = {}
        self._checkpoint_dir = checkpoint_dir
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get(self, customer_id: str) -> dict | None:
        """Return cached profile for customer_id, or None."""
        return self._store.get(customer_id)

    def put(self, customer_id: str, profile: dict) -> None:
        """Store profiling result for a customer_id."""
        self._store[customer_id] = profile

    @property
    def size(self) -> int:
        """Number of cached customer_ids."""
        return len(self._store)

    def save_checkpoint(self) -> None:
        """Persist current cache to a Parquet checkpoint file."""
        if self._checkpoint_dir is None:
            logger.warning("No checkpoint_dir configured; skipping save.")
            return
        if not self._store:
            return

        rows = []
        for customer_id, profile in self._store.items():
            row = {"customer_id": customer_id}
            for k, v in profile.items():
                row[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
            rows.append(row)

        df = pd.DataFrame(rows)
        checkpoint_path = self._checkpoint_dir / "checkpoint.parquet"
        df.to_parquet(checkpoint_path, index=False)
        logger.info("Saved checkpoint: %d customers → %s", len(rows), checkpoint_path)

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
            customer_id = row["customer_id"]
            profile: dict = {}
            for col in df.columns:
                if col == "customer_id":
                    continue
                val = row[col]
                if pd.isna(val):
                    profile[col] = None
                elif isinstance(val, str) and val.startswith(("[", "{")):
                    try:
                        profile[col] = json.loads(val)
                    except json.JSONDecodeError:
                        profile[col] = val
                else:
                    profile[col] = val
            self._store[customer_id] = profile
            loaded += 1

        logger.info("Loaded checkpoint: %d customers from %s", loaded, checkpoint_path)
        return loaded

    def keys(self) -> set[str]:
        """Return set of cached customer_ids."""
        return set(self._store.keys())

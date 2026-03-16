"""Batch API wrapper for user reasoning knowledge extraction.

Reuses src.knowledge.factual.batch infrastructure (run_batch_pipeline, parse_batch_results).
Reasoning-specific: JSONL preparation with user message construction.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import ReasoningConfig
from src.knowledge.reasoning.prompts import build_reasoning_request_line

logger = logging.getLogger(__name__)


def prepare_reasoning_batch_jsonl_chunked(
    user_data: list[dict],
    config: ReasoningConfig,
    output_dir: Path,
    max_bytes: int | None = None,
    max_requests: int | None = None,
) -> list[Path]:
    """Generate chunked Batch API input JSONL files for user profiling.

    Each line = one customer's profiling request.

    Args:
        user_data: List of dicts with keys:
            - customer_id: str
            - l1_summary: dict (L1 aggregated stats)
            - recent_items_l2: list[dict] (recent items with L2 attrs)
            - l3_distributions: dict (L3 distribution summaries)
        config: Reasoning configuration.
        output_dir: Directory to write chunk files.
        max_bytes: Max bytes per chunk. Defaults to config.batch_max_bytes.
        max_requests: Max requests per chunk. Defaults to config.batch_max_requests.

    Returns:
        List of paths to generated JSONL chunk files.
    """
    if max_bytes is None:
        max_bytes = config.batch_max_bytes
    if max_requests is None:
        max_requests = config.batch_max_requests

    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths: list[Path] = []
    chunk_idx = 0
    current_bytes = 0
    current_file = None
    n_requests_in_chunk = 0

    def _open_new_chunk() -> None:
        nonlocal chunk_idx, current_bytes, current_file, n_requests_in_chunk
        if current_file is not None:
            current_file.close()
            logger.info(
                "Chunk %d: %d requests, %.1f MB → %s",
                chunk_idx - 1,
                n_requests_in_chunk,
                chunk_paths[-1].stat().st_size / 1_000_000,
                chunk_paths[-1],
            )
        path = output_dir / f"input_{chunk_idx:03d}.jsonl"
        chunk_paths.append(path)
        current_file = open(path, "wb")  # noqa: SIM115
        current_bytes = 0
        n_requests_in_chunk = 0
        chunk_idx += 1

    _open_new_chunk()

    for user in user_data:
        line_bytes = build_reasoning_request_line(
            customer_id=user["customer_id"],
            l1_summary=user["l1_summary"],
            recent_items_l2=user["recent_items_l2"],
            l3_distributions=user["l3_distributions"],
            config=config,
        )
        line_size = len(line_bytes)

        if n_requests_in_chunk > 0 and (
            current_bytes + line_size > max_bytes or n_requests_in_chunk >= max_requests
        ):
            _open_new_chunk()

        current_file.write(line_bytes)  # type: ignore[union-attr]
        current_bytes += line_size
        n_requests_in_chunk += 1

    # Close final chunk
    if current_file is not None:
        current_file.close()
        logger.info(
            "Chunk %d: %d requests, %.1f MB → %s",
            chunk_idx - 1,
            n_requests_in_chunk,
            chunk_paths[-1].stat().st_size / 1_000_000,
            chunk_paths[-1],
        )

    logger.info(
        "Prepared %d chunked JSONL files for %d users → %s",
        len(chunk_paths),
        len(user_data),
        output_dir,
    )
    return chunk_paths

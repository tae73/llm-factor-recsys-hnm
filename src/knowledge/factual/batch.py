"""OpenAI Batch API wrapper for full-scale knowledge extraction.

Batch API provides 50% cost discount with 24-hour turnaround.
Flow: prepare JSONL → upload file → submit batch → poll status → download results.

Multi-chunk support: JSONL files > 200MB are split into 150MB chunks,
each submitted as a separate batch job, then results are merged.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import openai

from src.config import ExtractionConfig
from src.knowledge.factual.image_utils import get_image_for_article
from src.knowledge.factual.prompts import build_user_message, get_prompt_and_schema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-file helpers (backwards compatible)
# ---------------------------------------------------------------------------


def _build_request_line(
    product: dict,
    images_dir: Path,
    config: ExtractionConfig,
) -> bytes:
    """Build a single JSONL request line as UTF-8 bytes (including trailing newline)."""
    product_code = product["product_code"]
    rep = product["representative"]
    super_cat = product["super_category"]
    article_id = str(rep["article_id"])

    system_prompt, json_schema = get_prompt_and_schema(super_cat)
    image_b64 = get_image_for_article(images_dir, article_id, config.image_max_size)
    user_content = build_user_message(rep, rep.get("detail_desc", ""), image_b64)

    request = {
        "custom_id": product_code,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": config.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": f"knowledge_{super_cat.lower()}",
                    "schema": json_schema,
                    "strict": True,
                }
            },
        },
    }
    return (json.dumps(request) + "\n").encode("utf-8")


def prepare_batch_jsonl(
    products: list[dict],
    images_dir: Path,
    config: ExtractionConfig,
    output_path: Path,
) -> Path:
    """Generate Batch API input JSONL. Each line = one product_code.

    Args:
        products: List of dicts with keys: product_code, representative (row dict),
                  super_category.
        images_dir: Path to product images.
        config: Extraction config.
        output_path: Path to write the JSONL file.

    Returns:
        Path to the generated JSONL file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for product in products:
            product_code = product["product_code"]
            rep = product["representative"]
            super_cat = product["super_category"]
            article_id = str(rep["article_id"])

            system_prompt, json_schema = get_prompt_and_schema(super_cat)
            image_b64 = get_image_for_article(images_dir, article_id, config.image_max_size)
            user_content = build_user_message(rep, rep.get("detail_desc", ""), image_b64)

            request = {
                "custom_id": product_code,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": config.model,
                    "input": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": f"knowledge_{super_cat.lower()}",
                            "schema": json_schema,
                            "strict": True,
                        }
                    },
                },
            }
            f.write(json.dumps(request) + "\n")

    logger.info("Prepared batch JSONL: %d requests → %s", len(products), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Multi-chunk JSONL preparation
# ---------------------------------------------------------------------------


def prepare_batch_jsonl_chunked(
    products: list[dict],
    images_dir: Path,
    config: ExtractionConfig,
    output_dir: Path,
    max_bytes: int | None = None,
    max_requests: int | None = None,
) -> list[Path]:
    """Generate chunked Batch API input JSONL files, each under size/request limits.

    Splits products into multiple JSONL files based on cumulative byte size
    OR request count — whichever limit is reached first.

    Args:
        products: List of dicts with keys: product_code, representative, super_category.
        images_dir: Path to product images.
        config: Extraction config (uses config.batch_max_bytes / batch_max_requests
                if max_bytes / max_requests is None).
        output_dir: Directory to write chunk files (input_000.jsonl, input_001.jsonl, ...).
        max_bytes: Maximum bytes per chunk file. Defaults to config.batch_max_bytes.
        max_requests: Maximum requests per chunk file. Defaults to config.batch_max_requests.

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

    for product in products:
        line_bytes = _build_request_line(product, images_dir, config)
        line_size = len(line_bytes)

        # Split if byte limit OR request count limit would be exceeded (non-empty chunk)
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
        "Prepared %d chunked JSONL files for %d products → %s",
        len(chunk_paths),
        len(products),
        output_dir,
    )
    return chunk_paths


# ---------------------------------------------------------------------------
# Single batch submit / poll (backwards compatible)
# ---------------------------------------------------------------------------


def submit_batch(jsonl_path: Path, client: openai.OpenAI | None = None) -> str:
    """Upload JSONL file and submit batch job.

    Returns:
        batch_id for polling.
    """
    if client is None:
        client = openai.OpenAI()

    # Upload file
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    logger.info("Uploaded file: %s (%s)", file_obj.id, file_obj.filename)

    # Create batch
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    logger.info("Submitted batch: %s (status: %s)", batch.id, batch.status)
    return batch.id


def poll_batch(
    batch_id: str,
    client: openai.OpenAI | None = None,
    poll_interval: int = 60,
    timeout: int = 86400,
) -> Path:
    """Poll batch until completion, then download results.

    Args:
        batch_id: Batch job ID.
        client: OpenAI client.
        poll_interval: Seconds between polls.
        timeout: Maximum wait time in seconds (default: 24h).

    Returns:
        Path to downloaded results JSONL.
    """
    if client is None:
        client = openai.OpenAI()

    start = time.monotonic()
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        elapsed = time.monotonic() - start

        logger.info(
            "Batch %s: status=%s, completed=%s/%s (%.0fs elapsed)",
            batch_id,
            status,
            batch.request_counts.completed if batch.request_counts else "?",
            batch.request_counts.total if batch.request_counts else "?",
            elapsed,
        )

        if status == "completed":
            # Download output
            output_file_id = batch.output_file_id
            content = client.files.content(output_file_id)
            output_path = Path(f"batch_output_{batch_id}.jsonl")
            output_path.write_bytes(content.content)
            logger.info("Downloaded batch results → %s", output_path)
            return output_path

        if status in ("failed", "cancelled", "expired"):
            error_msg = f"Batch {batch_id} ended with status: {status}"
            if batch.errors:
                error_msg += f" — errors: {batch.errors}"
            raise RuntimeError(error_msg)

        if elapsed > timeout:
            raise TimeoutError(
                f"Batch {batch_id} did not complete within {timeout}s"
            )

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Multi-batch submit / poll
# ---------------------------------------------------------------------------


def submit_multi_batch(
    jsonl_paths: list[Path],
    batch_dir: Path,
    client: openai.OpenAI | None = None,
) -> list[str]:
    """Upload and submit multiple JSONL chunk files as separate batch jobs.

    Saves batch_ids.json to batch_dir for resume support. If a chunk fails
    to submit, it is recorded as failed and remaining chunks continue.

    Args:
        jsonl_paths: List of chunked JSONL file paths.
        batch_dir: Directory to save batch_ids.json.
        client: OpenAI client.

    Returns:
        List of batch IDs (empty string for failed chunks).
    """
    if client is None:
        client = openai.OpenAI()

    batch_ids: list[str] = []
    failed_indices: list[int] = []

    for idx, path in enumerate(jsonl_paths):
        try:
            bid = submit_batch(path, client=client)
            batch_ids.append(bid)
            logger.info("Chunk %d/%d submitted: %s", idx + 1, len(jsonl_paths), bid)
        except Exception as e:
            logger.error("Chunk %d/%d failed to submit: %s", idx + 1, len(jsonl_paths), e)
            batch_ids.append("")
            failed_indices.append(idx)

    # Persist batch_ids.json for resume
    manifest = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "n_chunks": len(jsonl_paths),
        "n_products": sum(
            _count_lines(p) for p in jsonl_paths
        ),
        "batch_ids": batch_ids,
        "failed_indices": failed_indices,
    }
    manifest_path = batch_dir / "batch_ids.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Saved batch manifest → %s (%d batch IDs)", manifest_path, len(batch_ids))

    return batch_ids


def poll_multi_batch(
    batch_ids: list[str],
    output_dir: Path,
    client: openai.OpenAI | None = None,
    poll_interval: int = 60,
    timeout: int = 86400,
) -> list[Path]:
    """Poll all batch jobs until completion, download results.

    Polls each non-empty batch ID sequentially. Already-downloaded results
    (from a previous run) are skipped.

    Args:
        batch_ids: List of batch IDs (empty strings are skipped).
        output_dir: Directory to save downloaded result JSONL files.
        client: OpenAI client.
        poll_interval: Seconds between polls per batch.
        timeout: Maximum wait time per batch in seconds.

    Returns:
        List of paths to downloaded result JSONL files.
    """
    if client is None:
        client = openai.OpenAI()

    output_dir.mkdir(parents=True, exist_ok=True)
    result_paths: list[Path] = []

    for idx, bid in enumerate(batch_ids):
        if not bid:
            logger.warning("Chunk %d: no batch ID (submission failed), skipping", idx)
            continue

        output_path = output_dir / f"output_{idx:03d}.jsonl"

        # Skip already-downloaded results
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info("Chunk %d: already downloaded → %s, skipping poll", idx, output_path)
            result_paths.append(output_path)
            continue

        logger.info("Polling chunk %d/%d: %s", idx + 1, len(batch_ids), bid)
        try:
            raw_path = poll_batch(bid, client=client, poll_interval=poll_interval, timeout=timeout)
            # Move to organized output location
            raw_path.rename(output_path)
            result_paths.append(output_path)
            logger.info("Chunk %d complete → %s", idx, output_path)
        except (RuntimeError, TimeoutError) as e:
            logger.error("Chunk %d (%s) failed: %s", idx, bid, e)

    logger.info(
        "Multi-batch poll complete: %d/%d chunks downloaded",
        len(result_paths),
        len([b for b in batch_ids if b]),
    )
    return result_paths


# ---------------------------------------------------------------------------
# Sequential batch pipeline
# ---------------------------------------------------------------------------


def run_batch_pipeline(
    jsonl_paths: list[Path],
    batch_dir: Path,
    client: openai.OpenAI | None = None,
    poll_interval: int = 60,
    timeout: int = 86400,
) -> list[Path]:
    """Submit, poll, and download batch chunks sequentially (one at a time).

    Processes each chunk: submit → poll → download → next chunk.
    This avoids exceeding the org-level enqueued token limit by having
    at most 1 batch job enqueued at any time.

    Resume support: if ``output_NNN.jsonl`` already exists for a chunk,
    that chunk is skipped. A stale manifest (``n_chunks`` mismatch) is
    discarded and the pipeline starts fresh.

    Args:
        jsonl_paths: List of chunked JSONL file paths (input_000.jsonl, ...).
        batch_dir: Directory for manifest and output files.
        client: OpenAI client (created if None).
        poll_interval: Seconds between polls per batch.
        timeout: Maximum wait time per batch in seconds (default: 24h).

    Returns:
        List of paths to successfully downloaded result JSONL files.
    """
    if not jsonl_paths:
        return []

    if client is None:
        client = openai.OpenAI()

    batch_dir.mkdir(parents=True, exist_ok=True)

    # Load or initialize manifest
    manifest = load_batch_manifest(batch_dir)
    if manifest is not None and manifest.get("n_chunks") != len(jsonl_paths):
        logger.warning(
            "Stale manifest (n_chunks=%d, expected=%d) — discarding",
            manifest.get("n_chunks", -1),
            len(jsonl_paths),
        )
        manifest = None

    if manifest is not None and "statuses" not in manifest:
        manifest["statuses"] = ["pending"] * manifest["n_chunks"]
        _save_pipeline_manifest(manifest, batch_dir)

    if manifest is None:
        manifest = {
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "n_chunks": len(jsonl_paths),
            "batch_ids": [""] * len(jsonl_paths),
            "statuses": ["pending"] * len(jsonl_paths),
        }
        _save_pipeline_manifest(manifest, batch_dir)

    result_paths: list[Path] = []

    for idx, jsonl_path in enumerate(jsonl_paths):
        output_path = batch_dir / f"output_{idx:03d}.jsonl"

        # Skip already-completed chunks
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info("Chunk %d/%d: already downloaded → %s, skipping", idx + 1, len(jsonl_paths), output_path)
            result_paths.append(output_path)
            manifest["statuses"][idx] = "completed"
            _save_pipeline_manifest(manifest, batch_dir)
            continue

        # Submit
        logger.info("Chunk %d/%d: submitting %s", idx + 1, len(jsonl_paths), jsonl_path)
        try:
            batch_id = submit_batch(jsonl_path, client=client)
            manifest["batch_ids"][idx] = batch_id
            manifest["statuses"][idx] = "submitted"
            _save_pipeline_manifest(manifest, batch_dir)
        except Exception as e:
            logger.error("Chunk %d/%d: submission failed — %s", idx + 1, len(jsonl_paths), e)
            manifest["statuses"][idx] = "failed"
            _save_pipeline_manifest(manifest, batch_dir)
            continue

        # Poll until completion
        try:
            raw_path = poll_batch(batch_id, client=client, poll_interval=poll_interval, timeout=timeout)
            raw_path.rename(output_path)
            result_paths.append(output_path)
            manifest["statuses"][idx] = "completed"
            logger.info("Chunk %d/%d: complete → %s", idx + 1, len(jsonl_paths), output_path)
        except (RuntimeError, TimeoutError) as e:
            logger.error("Chunk %d/%d (%s): poll failed — %s", idx + 1, len(jsonl_paths), batch_id, e)
            manifest["statuses"][idx] = "failed"

        _save_pipeline_manifest(manifest, batch_dir)

    logger.info(
        "Pipeline complete: %d/%d chunks downloaded",
        len(result_paths),
        len(jsonl_paths),
    )
    return result_paths


def _save_pipeline_manifest(manifest: dict, batch_dir: Path) -> None:
    """Save pipeline manifest to batch_ids.json."""
    manifest_path = batch_dir / "batch_ids.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def parse_batch_results(results_path: Path | list[Path]) -> dict[str, dict]:
    """Parse batch output JSONL into {custom_id: knowledge} dict.

    Args:
        results_path: Single Path or list of Paths to result JSONL files.
            When a list is provided, all files are merged into one dict.

    Returns:
        Dict mapping custom_id (product_code) to parsed knowledge dict.
    """
    paths = results_path if isinstance(results_path, list) else [results_path]

    results: dict[str, dict] = {}
    errors: list[str] = []

    for path in paths:
        with open(path) as f:
            for line in f:
                entry = json.loads(line.strip())
                custom_id = entry["custom_id"]

                if entry.get("error"):
                    errors.append(f"{custom_id}: {entry['error']}")
                    continue

                try:
                    response_body = entry["response"]["body"]
                    output_text = response_body["output"][0]["content"][0]["text"]
                    knowledge = json.loads(output_text)
                    results[custom_id] = knowledge
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    errors.append(f"{custom_id}: parse error — {e}")

    if errors:
        logger.warning("Batch parse errors (%d): %s", len(errors), errors[:5])

    logger.info(
        "Parsed batch results: %d success, %d errors (from %d files)",
        len(results),
        len(errors),
        len(paths),
    )
    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def load_batch_manifest(batch_dir: Path) -> dict | None:
    """Load batch_ids.json manifest if it exists.

    Returns:
        Manifest dict or None if not found.
    """
    manifest_path = batch_dir / "batch_ids.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def _count_lines(path: Path) -> int:
    """Count lines in a file efficiently."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count

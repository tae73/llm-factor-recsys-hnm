"""CLI entry point for user reasoning knowledge extraction (L1 aggregation + L2/L3 LLM reasoning).

Builds reasoning_text for all users: active (LLM) + sparse (template).

Usage:
    # Pilot (200 users, real-time API)
    python scripts/extract_reasoning_knowledge.py \
        --data-dir data/processed \
        --fk-dir data/knowledge/factual \
        --output-dir data/knowledge/reasoning \
        --pilot

    # Full batch (876K active users, Batch API)
    python scripts/extract_reasoning_knowledge.py \
        --data-dir data/processed \
        --fk-dir data/knowledge/factual \
        --output-dir data/knowledge/reasoning \
        --batch-api \
        --max-cost 80

    # Resume interrupted batch
    python scripts/extract_reasoning_knowledge.py \
        --data-dir data/processed \
        --fk-dir data/knowledge/factual \
        --output-dir data/knowledge/reasoning \
        --batch-api \
        --resume

    # Retry failed batch results + assemble final output
    python scripts/extract_reasoning_knowledge.py \
        --data-dir data/processed \
        --fk-dir data/knowledge/factual \
        --output-dir data/knowledge/reasoning \
        --batch-api \
        --retry-failed
"""

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

load_dotenv()

from src.config import ReasoningConfig  # noqa: E402
from src.knowledge.factual.batch import (  # noqa: E402
    load_batch_manifest,
    parse_batch_results,
    run_batch_pipeline,
)
from src.knowledge.reasoning.batch import prepare_reasoning_batch_jsonl_chunked  # noqa: E402
from src.knowledge.reasoning.cache import CustomerCache  # noqa: E402
from src.knowledge.reasoning.extractor import (  # noqa: E402
    aggregate_l1_profiles,
    build_sparse_user_profiles,
    compute_l3_distributions_batch,
    get_recent_items_batch,
)
from src.knowledge.reasoning.prompts import compose_reasoning_text  # noqa: E402

app = typer.Typer(help="Extract user reasoning knowledge: L1 aggregation + L2/L3 reasoning text")

# Token costs (GPT-4.1-nano Batch API)
_INPUT_COST_PER_1M = 0.10
_OUTPUT_COST_PER_1M = 0.40
_BATCH_DISCOUNT = 0.5


def _estimate_cost(input_tokens: int, output_tokens: int, is_batch: bool = False) -> float:
    discount = _BATCH_DISCOUNT if is_batch else 1.0
    return (
        input_tokens / 1_000_000 * _INPUT_COST_PER_1M * discount
        + output_tokens / 1_000_000 * _OUTPUT_COST_PER_1M * discount
    )


@app.command()
def main(
    data_dir: Path = typer.Option(
        "data/processed",
        help="Directory containing train_transactions.parquet, articles.parquet, customer IDs",
    ),
    fk_dir: Path = typer.Option(
        "data/knowledge/factual",
        help="Directory containing factual_knowledge.parquet",
    ),
    output_dir: Path = typer.Option(
        "data/knowledge/reasoning",
        help="Output directory for user reasoning knowledge",
    ),
    model: str = typer.Option("gpt-4.1-nano", help="OpenAI model name"),
    batch_api: bool = typer.Option(False, help="Use Batch API (50%% discount, 24h)"),
    max_cost: float = typer.Option(80.0, help="Cost limit in USD"),
    min_purchases: int = typer.Option(5, help="Min purchases for active user"),
    pilot: bool = typer.Option(False, help="Pilot mode (200 users, real-time API)"),
    resume: bool = typer.Option(False, help="Resume from checkpoint"),
    retry_failed: bool = typer.Option(False, help="Retry failed batch responses + template fallback + assemble"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
) -> None:
    """Extract user reasoning knowledge: L1 aggregation + L2/L3 reasoning text."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger(__name__)

    # Mutual exclusivity
    if retry_failed and (pilot or resume):
        logger.error("--retry-failed is mutually exclusive with --pilot and --resume")
        raise typer.Exit(1)

    config = ReasoningConfig(
        model=model,
        use_batch_api=batch_api,
        max_cost_usd=max_cost,
        min_purchases=min_purchases,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    txn_path = data_dir / "train_transactions.parquet"
    articles_path = data_dir / "articles.parquet"
    fk_path = fk_dir / "factual_knowledge.parquet"
    active_ids_path = data_dir / "active_customer_ids.json"
    sparse_ids_path = data_dir / "sparse_customer_ids.json"

    for p in [txn_path, articles_path, fk_path, active_ids_path, sparse_ids_path]:
        if not p.exists():
            logger.error("Required file not found: %s", p)
            raise typer.Exit(1)

    # Load customer ID lists
    active_ids = json.loads(active_ids_path.read_text())
    sparse_ids = json.loads(sparse_ids_path.read_text())
    logger.info("Loaded %d active + %d sparse customer IDs", len(active_ids), len(sparse_ids))

    # Fast-path: retry failed batch responses
    if batch_api and retry_failed:
        logger.info("=== Retry Failed Mode ===")

        # 1) Load checkpoint
        cache = CustomerCache(checkpoint_dir=output_dir / "checkpoint")
        n_loaded = cache.load_checkpoint()
        logger.info("Loaded checkpoint: %d cached profiles", n_loaded)

        # 2) Collect failed IDs
        failed_ids = _collect_failed_ids(active_ids, cache)
        logger.info("Found %d failed IDs to retry", len(failed_ids))

        if failed_ids:
            # 3) L1 aggregation (needed for _assemble_output)
            logger.info("Stage A: L1 aggregation (for assembly)...")
            start = time.monotonic()
            l1_df = aggregate_l1_profiles(txn_path, articles_path, fk_path, config)
            logger.info("L1 aggregation: %d users (%.1fs)", len(l1_df), time.monotonic() - start)

            # 4) Prepare user data for failed IDs only
            user_data_list = _prepare_user_data(
                failed_ids, l1_df, txn_path, fk_path, config, logger,
            )
            logger.info("Prepared %d users for retry", len(user_data_list))

            if user_data_list:
                # 5) Generate JSONL in batch/retry/ subdirectory
                retry_dir = output_dir / "batch" / "retry"
                retry_dir.mkdir(parents=True, exist_ok=True)

                chunk_paths = prepare_reasoning_batch_jsonl_chunked(
                    user_data_list, config, output_dir=retry_dir,
                )
                logger.info("Prepared %d retry chunks", len(chunk_paths))

                # 6) Submit + poll
                try:
                    result_paths = run_batch_pipeline(
                        chunk_paths, batch_dir=retry_dir, poll_interval=60,
                    )
                except KeyboardInterrupt:
                    logger.info("Retry pipeline interrupted — saving progress...")
                    result_paths = []

                # 7) Process results → checkpoint
                if result_paths:
                    _process_batch_results(result_paths, cache, logger)
                else:
                    # Check for any downloaded outputs
                    downloaded = sorted(
                        p for p in retry_dir.glob("output_*.jsonl") if p.stat().st_size > 0
                    )
                    if downloaded:
                        _process_batch_results(downloaded, cache, logger)

                cache.save_checkpoint()

            # 8) Template fallback for still-failed users
            still_failed = _collect_failed_ids(active_ids, cache)
            if still_failed:
                logger.info("Applying template fallback for %d still-failed users", len(still_failed))
                _apply_template_fallback(still_failed, txn_path, fk_path, cache, logger)
                cache.save_checkpoint()
        else:
            # No failed IDs — still need L1 for assembly
            logger.info("No failed IDs found. Proceeding to assembly...")
            l1_df = aggregate_l1_profiles(txn_path, articles_path, fk_path, config)

        # 9) Stage C: sparse user profiles
        logger.info("Stage C: Building sparse user profiles (%d users)...", len(sparse_ids))
        sparse_df = build_sparse_user_profiles(txn_path, fk_path, sparse_ids)
        logger.info("Sparse profiles complete: %d users", len(sparse_df))

        # 10) Assemble final output
        logger.info("Assembling final profiles...")
        _assemble_output(l1_df, cache, sparse_df, active_ids, output_dir, logger)
        logger.info("Done! Output: %s", output_dir)
        raise typer.Exit(0)

    # Fast-path: batch resume — skip L1 + LLM input prep if chunks already exist
    if batch_api and resume:
        batch_dir = output_dir / "batch"
        manifest = load_batch_manifest(batch_dir)
        existing_chunks = sorted(batch_dir.glob("input_*.jsonl"))

        if (
            existing_chunks
            and manifest is not None
            and manifest.get("n_chunks") == len(existing_chunks)
        ):
            logger.info(
                "Batch resume fast-path: %d existing chunks found, skipping L1/input prep",
                len(existing_chunks),
            )

            # 1) Cache 생성 + 기존 checkpoint 로드 (파이프라인 전)
            cache = CustomerCache(checkpoint_dir=output_dir / "checkpoint")
            n_loaded = cache.load_checkpoint()
            logger.info("Loaded %d cached profiles", n_loaded)

            # 2) 이미 다운로드된 outputs → 즉시 checkpoint 갱신
            downloaded = sorted(
                p for p in batch_dir.glob("output_*.jsonl") if p.stat().st_size > 0
            )
            if downloaded:
                logger.info("Processing %d already-downloaded outputs", len(downloaded))
                _process_batch_results(downloaded, cache, logger)

            # 3) 나머지 chunks 파이프라인 실행 (KeyboardInterrupt 핸들링)
            downloaded_set = set(downloaded)
            try:
                result_paths = run_batch_pipeline(
                    existing_chunks, batch_dir=batch_dir, poll_interval=60
                )
            except KeyboardInterrupt:
                logger.info("Pipeline interrupted — saving progress...")
                result_paths = []

            # 4) 새로 다운로드된 outputs만 처리
            new_outputs = sorted(
                p for p in batch_dir.glob("output_*.jsonl")
                if p.stat().st_size > 0 and p not in downloaded_set
            )
            if new_outputs:
                _process_batch_results(new_outputs, cache, logger)

            # 5) 최종 checkpoint
            cache.save_checkpoint()
            logger.info("Batch resume complete.")
            raise typer.Exit(0)

    # -----------------------------------------------------------------------
    # Stage A: L1 Direct Aggregation (all users)
    # -----------------------------------------------------------------------
    logger.info("Stage A: L1 aggregation...")
    start = time.monotonic()
    l1_df = aggregate_l1_profiles(txn_path, articles_path, fk_path, config)
    logger.info("L1 aggregation complete: %d users (%.1fs)", len(l1_df), time.monotonic() - start)

    # -----------------------------------------------------------------------
    # Stage B: LLM Reasoning (active users)
    # -----------------------------------------------------------------------
    # Load cache
    cache = CustomerCache(checkpoint_dir=output_dir / "checkpoint")
    if resume:
        n_loaded = cache.load_checkpoint()
        logger.info("Resumed: %d users from checkpoint", n_loaded)

    # Determine target users
    if pilot:
        target_ids = [uid for uid in active_ids if cache.get(uid) is None][:config.pilot_size]
        logger.info("Pilot mode: %d users", len(target_ids))
    else:
        target_ids = [uid for uid in active_ids if cache.get(uid) is None]
        logger.info("Full mode: %d users to process (%d cached)", len(target_ids), cache.size)

    if target_ids:
        # Prepare LLM input data
        user_data_list = _prepare_user_data(
            target_ids, l1_df, txn_path, fk_path, config, logger,
        )
        logger.info("Total users prepared for LLM: %d", len(user_data_list))

        if batch_api:
            # Batch API path
            batch_dir = output_dir / "batch"
            batch_dir.mkdir(exist_ok=True)

            # Check for existing chunks (resume)
            manifest = load_batch_manifest(batch_dir)
            existing_chunks = sorted(batch_dir.glob("input_*.jsonl"))

            if (
                existing_chunks
                and manifest is not None
                and manifest.get("n_chunks") == len(existing_chunks)
            ):
                chunk_paths = existing_chunks
                logger.info("Reusing %d existing chunks (resume)", len(chunk_paths))
            else:
                chunk_paths = prepare_reasoning_batch_jsonl_chunked(
                    user_data_list, config, output_dir=batch_dir,
                )
                logger.info("Prepared %d chunks", len(chunk_paths))

            # Run sequential pipeline
            result_paths = run_batch_pipeline(chunk_paths, batch_dir=batch_dir, poll_interval=60)

            if result_paths:
                _process_batch_results(result_paths, cache, logger)
            else:
                logger.warning("No batch results downloaded.")

        elif pilot:
            # Real-time API for pilot
            _run_realtime_pilot(user_data_list, config, cache, logger)

        cache.save_checkpoint()

    # -----------------------------------------------------------------------
    # Stage C: Sparse User Fallback
    # -----------------------------------------------------------------------
    logger.info("Stage C: Building sparse user profiles (%d users)...", len(sparse_ids))
    sparse_df = build_sparse_user_profiles(txn_path, fk_path, sparse_ids)
    logger.info("Sparse profiles complete: %d users", len(sparse_df))

    # -----------------------------------------------------------------------
    # Assemble Final Output
    # -----------------------------------------------------------------------
    logger.info("Assembling final profiles...")
    _assemble_output(l1_df, cache, sparse_df, active_ids, output_dir, logger)

    logger.info("Done! Output: %s", output_dir)


def _prepare_user_data(
    target_ids: list[str],
    l1_df: pd.DataFrame,
    txn_path: Path,
    fk_path: Path,
    config: ReasoningConfig,
    logger: logging.Logger,
    chunk_size: int = 10_000,
) -> list[dict]:
    """Prepare LLM input data (recent items + L3 distributions) for target users."""
    logger.info("Preparing LLM input: recent items + L3 distributions...")
    start = time.monotonic()

    user_data_list: list[dict] = []
    l1_indexed = l1_df.set_index("customer_id")

    for i in range(0, len(target_ids), chunk_size):
        chunk_ids = target_ids[i : i + chunk_size]

        recent_items = get_recent_items_batch(
            txn_path, fk_path, chunk_ids, limit=config.recent_items_limit
        )
        l3_dists = compute_l3_distributions_batch(txn_path, fk_path, chunk_ids)

        for uid in chunk_ids:
            if uid not in l1_indexed.index:
                continue
            l1_row = l1_indexed.loc[uid].to_dict()
            user_data_list.append({
                "customer_id": uid,
                "l1_summary": l1_row,
                "recent_items_l2": recent_items.get(uid, []),
                "l3_distributions": l3_dists.get(uid, {"shared": {}, "by_category": {}}),
            })

        logger.info(
            "Prepared %d/%d users (%.1fs)",
            min(i + chunk_size, len(target_ids)),
            len(target_ids),
            time.monotonic() - start,
        )

    return user_data_list


def _collect_failed_ids(active_ids: list[str], cache: CustomerCache) -> list[str]:
    """Return active user IDs not present in cache."""
    cached = cache.keys()
    return [uid for uid in active_ids if uid not in cached]


def _apply_template_fallback(
    failed_ids: list[str],
    txn_path: Path,
    fk_path: Path,
    cache: CustomerCache,
    logger: logging.Logger,
) -> None:
    """Apply template-based fallback profiles for users that still failed after retry."""
    sparse_df = build_sparse_user_profiles(txn_path, fk_path, failed_ids)
    n_applied = 0
    for _, row in sparse_df.iterrows():
        uid = row["customer_id"]
        cache.put(uid, {
            "reasoning_json": json.loads(row["reasoning_json"]) if isinstance(row["reasoning_json"], str) else row["reasoning_json"],
            "reasoning_text": row["reasoning_text"],
            "profile_source": "template_fallback",
        })
        n_applied += 1
    logger.info("Template fallback applied: %d users", n_applied)


def _process_batch_results(
    result_paths: list[Path],
    cache: CustomerCache,
    logger: logging.Logger,
    checkpoint_every: int = 10,
) -> None:
    """Parse batch results and populate cache with periodic checkpointing."""
    n_valid = 0

    for batch_start in range(0, len(result_paths), checkpoint_every):
        batch_paths = result_paths[batch_start : batch_start + checkpoint_every]
        parsed = parse_batch_results(batch_paths)

        for customer_id, reasoning_json in parsed.items():
            reasoning_text = compose_reasoning_text(reasoning_json)
            cache.put(customer_id, {
                "reasoning_json": reasoning_json,
                "reasoning_text": reasoning_text,
                "profile_source": "llm",
            })
            n_valid += 1

        cache.save_checkpoint()
        logger.info(
            "Checkpoint saved: %d/%d result files processed (%d valid profiles)",
            min(batch_start + checkpoint_every, len(result_paths)),
            len(result_paths),
            n_valid,
        )

    logger.info("Batch results processed: %d valid profiles total", n_valid)


def _run_realtime_pilot(
    user_data: list[dict],
    config: ReasoningConfig,
    cache: CustomerCache,
    logger: logging.Logger,
) -> None:
    """Run real-time API pilot extraction for a small number of users."""
    import openai

    client = openai.OpenAI()

    from src.knowledge.reasoning.prompts import (
        REASONING_SCHEMA,
        SYSTEM_PROMPT,
        build_reasoning_user_message,
    )

    total_input_tokens = 0
    total_output_tokens = 0

    for i, user in enumerate(user_data):
        customer_id = user["customer_id"]
        if cache.get(customer_id) is not None:
            continue

        user_message = build_reasoning_user_message(
            user["l1_summary"], user["recent_items_l2"], user["l3_distributions"]
        )

        try:
            response = client.responses.create(
                model=config.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "user_reasoning_profile",
                        "schema": REASONING_SCHEMA,
                        "strict": True,
                    }
                },
            )

            reasoning_json = json.loads(response.output_text)
            reasoning_text = compose_reasoning_text(reasoning_json)

            cache.put(customer_id, {
                "reasoning_json": reasoning_json,
                "reasoning_text": reasoning_text,
                "profile_source": "llm",
            })

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            if (i + 1) % 50 == 0:
                cache.save_checkpoint()
                cost = _estimate_cost(total_input_tokens, total_output_tokens)
                logger.info(
                    "Pilot progress: %d/%d users, $%.4f",
                    i + 1, len(user_data), cost,
                )

        except Exception as e:
            logger.warning("Failed for %s: %s", customer_id, e)

    cost = _estimate_cost(total_input_tokens, total_output_tokens)
    logger.info(
        "Pilot complete: %d users, %d input + %d output tokens, $%.4f",
        len(user_data), total_input_tokens, total_output_tokens, cost,
    )


def _assemble_output(
    l1_df: pd.DataFrame,
    cache: CustomerCache,
    sparse_df: pd.DataFrame,
    active_ids: list[str],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Assemble final output parquets from L1 stats, LLM cache, and sparse profiles."""
    # Active user profiles from cache
    active_rows = []
    for uid in active_ids:
        cached = cache.get(uid)
        if cached is not None:
            active_rows.append({
                "customer_id": uid,
                "reasoning_text": cached.get("reasoning_text", ""),
                "reasoning_json": json.dumps(cached.get("reasoning_json", {})),
                "profile_source": cached.get("profile_source", "llm"),
            })

    active_profiles_df = pd.DataFrame(active_rows) if active_rows else pd.DataFrame(
        columns=["customer_id", "reasoning_text", "reasoning_json", "profile_source"]
    )

    # Merge L1 stats with reasoning
    l1_active = l1_df[l1_df["customer_id"].isin(set(active_ids))]
    if not active_profiles_df.empty:
        merged_active = l1_active.merge(active_profiles_df, on="customer_id", how="left")
    else:
        merged_active = l1_active.assign(
            reasoning_text="", reasoning_json="{}", profile_source=""
        )
    merged_active = merged_active.assign(is_active=True)

    # Sparse profiles — merge with L1 stats
    l1_sparse = l1_df[~l1_df["customer_id"].isin(set(active_ids))]
    if not sparse_df.empty:
        merged_sparse = l1_sparse.merge(
            sparse_df[["customer_id", "reasoning_text", "reasoning_json", "profile_source"]],
            on="customer_id",
            how="left",
        )
    else:
        merged_sparse = l1_sparse.assign(
            reasoning_text="", reasoning_json="{}", profile_source="template"
        )
    merged_sparse = merged_sparse.assign(is_active=False)

    # Combine
    all_profiles = pd.concat([merged_active, merged_sparse], ignore_index=True)

    # Save outputs
    profiles_path = output_dir / "user_profiles.parquet"
    all_profiles.to_parquet(profiles_path, index=False)
    logger.info("Saved %d user profiles → %s", len(all_profiles), profiles_path)

    # reasoning_texts.parquet — KAR input (customer_id → reasoning_text)
    reasoning_df = all_profiles[["customer_id", "reasoning_text"]].dropna(
        subset=["reasoning_text"]
    )
    reasoning_df = reasoning_df[reasoning_df["reasoning_text"].str.len() > 0]
    reasoning_path = output_dir / "reasoning_texts.parquet"
    reasoning_df.to_parquet(reasoning_path, index=False)
    logger.info("Saved %d reasoning texts → %s", len(reasoning_df), reasoning_path)

    # Quality report
    n_active_with_llm = len(active_profiles_df)
    n_sparse = len(sparse_df)
    n_template_fallback = (
        active_profiles_df["profile_source"].eq("template_fallback").sum()
        if not active_profiles_df.empty and "profile_source" in active_profiles_df.columns
        else 0
    )
    quality_report = {
        "n_total_users": len(all_profiles),
        "n_active_users": len(merged_active),
        "n_sparse_users": len(merged_sparse),
        "n_active_with_reasoning": n_active_with_llm,
        "n_active_template_fallback": int(n_template_fallback),
        "n_sparse_with_reasoning": n_sparse,
        "reasoning_coverage": (n_active_with_llm + n_sparse) / max(len(all_profiles), 1),
    }
    report_path = output_dir / "quality_report.json"
    report_path.write_text(json.dumps(quality_report, indent=2))
    logger.info("Quality report → %s", report_path)


if __name__ == "__main__":
    app()

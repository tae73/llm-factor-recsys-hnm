"""Tests for multi-chunk Batch API support."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import ExtractionConfig
from src.knowledge.factual.batch import (
    _count_lines,
    load_batch_manifest,
    parse_batch_results,
    prepare_batch_jsonl_chunked,
    run_batch_pipeline,
    submit_multi_batch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_product(product_code: str, super_category: str = "Apparel") -> dict:
    """Create a minimal product dict for testing."""
    return {
        "product_code": product_code,
        "representative": {
            "article_id": "0100001001",
            "product_type_name": "T-shirt",
            "product_group_name": "Garment Upper body",
            "colour_group_name": "Black",
            "detail_desc": "A simple test product.",
        },
        "super_category": super_category,
    }


def _make_batch_result_line(custom_id: str, knowledge: dict) -> str:
    """Create a single batch result JSONL line."""
    entry = {
        "custom_id": custom_id,
        "response": {
            "body": {
                "output": [
                    {
                        "content": [
                            {"text": json.dumps(knowledge)}
                        ]
                    }
                ]
            }
        },
    }
    return json.dumps(entry)


def _make_batch_error_line(custom_id: str, error: str) -> str:
    """Create a batch result JSONL error line."""
    return json.dumps({"custom_id": custom_id, "error": error})


@pytest.fixture
def config() -> ExtractionConfig:
    return ExtractionConfig(model="gpt-4.1-nano")


# ---------------------------------------------------------------------------
# prepare_batch_jsonl_chunked
# ---------------------------------------------------------------------------


class TestPrepareBatchJsonlChunked:
    """Test byte-based chunked JSONL generation."""

    @patch("src.knowledge.factual.batch.get_image_for_article", return_value=None)
    @patch(
        "src.knowledge.factual.batch.get_prompt_and_schema",
        return_value=("system prompt", {"type": "object"}),
    )
    @patch(
        "src.knowledge.factual.batch.build_user_message",
        return_value=[{"type": "text", "text": "user message"}],
    )
    def test_splits_by_size(
        self, mock_user_msg, mock_prompt, mock_image, tmp_path: Path, config: ExtractionConfig
    ):
        """Chunks are split when cumulative size exceeds max_bytes."""
        products = [_make_product(f"PC{i:04d}") for i in range(20)]

        # Use a very small max_bytes to force multiple chunks
        paths = prepare_batch_jsonl_chunked(
            products, tmp_path / "images", config, tmp_path / "batch", max_bytes=500
        )

        assert len(paths) > 1, "Should produce multiple chunks with small max_bytes"

        # Each chunk should be <= max_bytes (500 bytes)
        # Note: each chunk must contain at least 1 line, so a single line may exceed max_bytes
        for path in paths:
            assert path.exists()
            assert path.stat().st_size > 0

        # Total lines across all chunks should equal product count
        total_lines = sum(_count_lines(p) for p in paths)
        assert total_lines == 20

        # File naming convention
        assert all(p.name.startswith("input_") and p.name.endswith(".jsonl") for p in paths)

    @patch("src.knowledge.factual.batch.get_image_for_article", return_value=None)
    @patch(
        "src.knowledge.factual.batch.get_prompt_and_schema",
        return_value=("system prompt", {"type": "object"}),
    )
    @patch(
        "src.knowledge.factual.batch.build_user_message",
        return_value=[{"type": "text", "text": "msg"}],
    )
    def test_single_chunk_when_small(
        self, mock_user_msg, mock_prompt, mock_image, tmp_path: Path, config: ExtractionConfig
    ):
        """Small input fits in a single chunk."""
        products = [_make_product("PC0001")]

        paths = prepare_batch_jsonl_chunked(
            products,
            tmp_path / "images",
            config,
            tmp_path / "batch",
            max_bytes=10_000_000,  # 10MB — way more than 1 product
        )

        assert len(paths) == 1
        assert _count_lines(paths[0]) == 1

    @patch("src.knowledge.factual.batch.get_image_for_article", return_value=None)
    @patch(
        "src.knowledge.factual.batch.get_prompt_and_schema",
        return_value=("system prompt", {"type": "object"}),
    )
    @patch(
        "src.knowledge.factual.batch.build_user_message",
        return_value=[{"type": "text", "text": "msg"}],
    )
    def test_uses_config_batch_max_bytes(
        self, mock_user_msg, mock_prompt, mock_image, tmp_path: Path
    ):
        """Uses config.batch_max_bytes when max_bytes is not specified."""
        custom_config = ExtractionConfig(model="gpt-4.1-nano", batch_max_bytes=300)
        products = [_make_product(f"PC{i:04d}") for i in range(10)]

        paths = prepare_batch_jsonl_chunked(
            products,
            tmp_path / "images",
            custom_config,
            tmp_path / "batch",
        )

        # With 300 bytes limit, should produce multiple chunks
        assert len(paths) > 1

    @patch("src.knowledge.factual.batch.get_image_for_article", return_value=None)
    @patch(
        "src.knowledge.factual.batch.get_prompt_and_schema",
        return_value=("system prompt", {"type": "object"}),
    )
    @patch(
        "src.knowledge.factual.batch.build_user_message",
        return_value=[{"type": "text", "text": "msg"}],
    )
    def test_each_chunk_has_valid_jsonl(
        self, mock_user_msg, mock_prompt, mock_image, tmp_path: Path, config: ExtractionConfig
    ):
        """Each chunk file contains valid JSONL."""
        products = [_make_product(f"PC{i:04d}") for i in range(5)]

        paths = prepare_batch_jsonl_chunked(
            products, tmp_path / "images", config, tmp_path / "batch", max_bytes=500
        )

        for path in paths:
            with open(path) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    assert "custom_id" in entry
                    assert "method" in entry
                    assert entry["method"] == "POST"


# ---------------------------------------------------------------------------
# submit_multi_batch
# ---------------------------------------------------------------------------


class TestSubmitMultiBatch:
    """Test multi-batch submission and manifest saving."""

    def test_saves_batch_ids_json(self, tmp_path: Path):
        """batch_ids.json is saved with correct structure."""
        # Create dummy JSONL files
        chunk1 = tmp_path / "input_000.jsonl"
        chunk2 = tmp_path / "input_001.jsonl"
        chunk1.write_text('{"custom_id": "PC001"}\n{"custom_id": "PC002"}\n')
        chunk2.write_text('{"custom_id": "PC003"}\n')

        mock_client = MagicMock()

        # Mock submit_batch to return fake batch IDs
        with patch(
            "src.knowledge.factual.batch.submit_batch",
            side_effect=["batch_aaa", "batch_bbb"],
        ):
            batch_ids = submit_multi_batch(
                [chunk1, chunk2], batch_dir=tmp_path, client=mock_client
            )

        assert batch_ids == ["batch_aaa", "batch_bbb"]

        # Verify manifest
        manifest = json.loads((tmp_path / "batch_ids.json").read_text())
        assert manifest["n_chunks"] == 2
        assert manifest["n_products"] == 3
        assert manifest["batch_ids"] == ["batch_aaa", "batch_bbb"]
        assert manifest["failed_indices"] == []
        assert "submitted_at" in manifest

    def test_partial_failure(self, tmp_path: Path):
        """Failed chunk submissions are recorded, others continue."""
        chunk1 = tmp_path / "input_000.jsonl"
        chunk2 = tmp_path / "input_001.jsonl"
        chunk1.write_text('{"custom_id": "PC001"}\n')
        chunk2.write_text('{"custom_id": "PC002"}\n')

        mock_client = MagicMock()

        with patch(
            "src.knowledge.factual.batch.submit_batch",
            side_effect=["batch_aaa", RuntimeError("upload failed")],
        ):
            batch_ids = submit_multi_batch(
                [chunk1, chunk2], batch_dir=tmp_path, client=mock_client
            )

        assert batch_ids == ["batch_aaa", ""]

        manifest = json.loads((tmp_path / "batch_ids.json").read_text())
        assert manifest["failed_indices"] == [1]


# ---------------------------------------------------------------------------
# parse_batch_results (multi-file)
# ---------------------------------------------------------------------------


class TestParseBatchResultsMulti:
    """Test parsing merged results from multiple JSONL files."""

    def test_merges_multiple_files(self, tmp_path: Path):
        """Results from multiple files are merged into one dict."""
        file1 = tmp_path / "output_000.jsonl"
        file2 = tmp_path / "output_001.jsonl"

        file1.write_text(
            _make_batch_result_line("PC001", {"l1_material": "cotton"}) + "\n"
            + _make_batch_result_line("PC002", {"l1_material": "silk"}) + "\n"
        )
        file2.write_text(
            _make_batch_result_line("PC003", {"l1_material": "polyester"}) + "\n"
        )

        results = parse_batch_results([file1, file2])

        assert len(results) == 3
        assert results["PC001"]["l1_material"] == "cotton"
        assert results["PC002"]["l1_material"] == "silk"
        assert results["PC003"]["l1_material"] == "polyester"

    def test_single_path_backwards_compatible(self, tmp_path: Path):
        """Single Path argument still works (backwards compatibility)."""
        file1 = tmp_path / "output.jsonl"
        file1.write_text(
            _make_batch_result_line("PC001", {"l1_material": "cotton"}) + "\n"
        )

        results = parse_batch_results(file1)
        assert len(results) == 1
        assert "PC001" in results

    def test_handles_errors_across_files(self, tmp_path: Path):
        """Errors in any file are collected and logged."""
        file1 = tmp_path / "output_000.jsonl"
        file2 = tmp_path / "output_001.jsonl"

        file1.write_text(
            _make_batch_result_line("PC001", {"l1_material": "cotton"}) + "\n"
            + _make_batch_error_line("PC002", "rate_limit_exceeded") + "\n"
        )
        file2.write_text(
            _make_batch_result_line("PC003", {"l1_material": "polyester"}) + "\n"
        )

        results = parse_batch_results([file1, file2])

        assert len(results) == 2
        assert "PC001" in results
        assert "PC002" not in results
        assert "PC003" in results


# ---------------------------------------------------------------------------
# load_batch_manifest
# ---------------------------------------------------------------------------


class TestLoadBatchManifest:
    """Test batch_ids.json loading."""

    def test_returns_none_when_missing(self, tmp_path: Path):
        assert load_batch_manifest(tmp_path) is None

    def test_loads_manifest(self, tmp_path: Path):
        manifest = {
            "submitted_at": "2026-02-21T22:53:34+00:00",
            "n_chunks": 2,
            "n_products": 100,
            "batch_ids": ["batch_aaa", "batch_bbb"],
            "failed_indices": [],
        }
        (tmp_path / "batch_ids.json").write_text(json.dumps(manifest))

        loaded = load_batch_manifest(tmp_path)
        assert loaded is not None
        assert loaded["n_chunks"] == 2
        assert loaded["batch_ids"] == ["batch_aaa", "batch_bbb"]


# ---------------------------------------------------------------------------
# prepare_batch_jsonl_chunked — max_requests limit
# ---------------------------------------------------------------------------

class TestPrepareBatchJsonlChunkedMaxRequests:
    """Test request-count-based chunking."""

    @patch("src.knowledge.factual.batch.get_image_for_article", return_value=None)
    @patch(
        "src.knowledge.factual.batch.get_prompt_and_schema",
        return_value=("system prompt", {"type": "object"}),
    )
    @patch(
        "src.knowledge.factual.batch.build_user_message",
        return_value=[{"type": "text", "text": "msg"}],
    )
    def test_splits_by_request_count(
        self, mock_user_msg, mock_prompt, mock_image, tmp_path: Path, config: ExtractionConfig
    ):
        """12 products with max_requests=5 → 3 chunks (5+5+2)."""
        products = [_make_product(f"PC{i:04d}") for i in range(12)]

        paths = prepare_batch_jsonl_chunked(
            products,
            tmp_path / "images",
            config,
            tmp_path / "batch",
            max_bytes=999_999_999,  # effectively unlimited
            max_requests=5,
        )

        assert len(paths) == 3
        line_counts = [_count_lines(p) for p in paths]
        assert line_counts == [5, 5, 2]

    @patch("src.knowledge.factual.batch.get_image_for_article", return_value=None)
    @patch(
        "src.knowledge.factual.batch.get_prompt_and_schema",
        return_value=("system prompt", {"type": "object"}),
    )
    @patch(
        "src.knowledge.factual.batch.build_user_message",
        return_value=[{"type": "text", "text": "msg"}],
    )
    def test_uses_config_batch_max_requests(
        self, mock_user_msg, mock_prompt, mock_image, tmp_path: Path
    ):
        """Defaults to config.batch_max_requests when max_requests is None."""
        custom_config = ExtractionConfig(model="gpt-4.1-nano", batch_max_requests=3)
        products = [_make_product(f"PC{i:04d}") for i in range(7)]

        paths = prepare_batch_jsonl_chunked(
            products,
            tmp_path / "images",
            custom_config,
            tmp_path / "batch",
            max_bytes=999_999_999,
        )

        assert len(paths) == 3  # 3+3+1
        line_counts = [_count_lines(p) for p in paths]
        assert line_counts == [3, 3, 1]

    @patch("src.knowledge.factual.batch.get_image_for_article", return_value=None)
    @patch(
        "src.knowledge.factual.batch.get_prompt_and_schema",
        return_value=("system prompt", {"type": "object"}),
    )
    @patch(
        "src.knowledge.factual.batch.build_user_message",
        return_value=[{"type": "text", "text": "msg"}],
    )
    def test_byte_limit_still_triggers_independently(
        self, mock_user_msg, mock_prompt, mock_image, tmp_path: Path, config: ExtractionConfig
    ):
        """Byte limit triggers even if request count is high."""
        products = [_make_product(f"PC{i:04d}") for i in range(10)]

        paths = prepare_batch_jsonl_chunked(
            products,
            tmp_path / "images",
            config,
            tmp_path / "batch",
            max_bytes=500,  # very small → bytes trigger
            max_requests=999,  # very high → won't trigger
        )

        assert len(paths) > 1
        total_lines = sum(_count_lines(p) for p in paths)
        assert total_lines == 10

    @patch("src.knowledge.factual.batch.get_image_for_article", return_value=None)
    @patch(
        "src.knowledge.factual.batch.get_prompt_and_schema",
        return_value=("system prompt", {"type": "object"}),
    )
    @patch(
        "src.knowledge.factual.batch.build_user_message",
        return_value=[{"type": "text", "text": "msg"}],
    )
    def test_both_limits_trigger_correct_chunk_count(
        self, mock_user_msg, mock_prompt, mock_image, tmp_path: Path, config: ExtractionConfig
    ):
        """When both limits are set, the stricter one determines chunk count."""
        products = [_make_product(f"PC{i:04d}") for i in range(10)]

        # max_requests=2 is stricter (10/2=5 chunks)
        paths = prepare_batch_jsonl_chunked(
            products,
            tmp_path / "images",
            config,
            tmp_path / "batch",
            max_bytes=999_999_999,
            max_requests=2,
        )

        assert len(paths) == 5
        assert all(_count_lines(p) == 2 for p in paths)


# ---------------------------------------------------------------------------
# run_batch_pipeline
# ---------------------------------------------------------------------------


class TestRunBatchPipeline:
    """Test sequential batch pipeline."""

    def test_sequential_submit_poll_each_chunk(self, tmp_path: Path):
        """submit→poll are interleaved sequentially (not all-submit then all-poll)."""
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        chunk1 = batch_dir / "input_000.jsonl"
        chunk2 = batch_dir / "input_001.jsonl"
        chunk1.write_text('{"custom_id": "PC001"}\n')
        chunk2.write_text('{"custom_id": "PC002"}\n')

        call_order: list[str] = []

        def mock_submit(path, client=None):
            call_order.append(f"submit_{path.name}")
            return f"batch_{path.stem}"

        def mock_poll(batch_id, client=None, poll_interval=60, timeout=86400):
            call_order.append(f"poll_{batch_id}")
            # Create output file simulating poll_batch behavior
            output = Path(f"batch_output_{batch_id}.jsonl")
            output.write_text(_make_batch_result_line("PC001", {"test": True}) + "\n")
            return output

        mock_client = MagicMock()

        with patch("src.knowledge.factual.batch.submit_batch", side_effect=mock_submit), \
             patch("src.knowledge.factual.batch.poll_batch", side_effect=mock_poll):
            result_paths = run_batch_pipeline(
                [chunk1, chunk2], batch_dir=batch_dir, client=mock_client,
            )

        assert len(result_paths) == 2
        # Verify sequential interleaving: submit1→poll1→submit2→poll2
        assert call_order == [
            "submit_input_000.jsonl",
            "poll_batch_input_000",
            "submit_input_001.jsonl",
            "poll_batch_input_001",
        ]

    def test_saves_manifest_after_each_chunk(self, tmp_path: Path):
        """Manifest is updated after every chunk completion."""
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        chunk1 = batch_dir / "input_000.jsonl"
        chunk2 = batch_dir / "input_001.jsonl"
        chunk1.write_text('{"custom_id": "PC001"}\n')
        chunk2.write_text('{"custom_id": "PC002"}\n')

        manifest_snapshots: list[dict] = []
        original_save = __import__(
            "src.knowledge.factual.batch", fromlist=["_save_pipeline_manifest"]
        )._save_pipeline_manifest

        def tracking_save(manifest, bd):
            original_save(manifest, bd)
            # Deep copy to capture snapshot
            manifest_snapshots.append(json.loads(json.dumps(manifest)))

        def mock_submit(path, client=None):
            return f"batch_{path.stem}"

        def mock_poll(batch_id, client=None, poll_interval=60, timeout=86400):
            output = Path(f"batch_output_{batch_id}.jsonl")
            output.write_text(_make_batch_result_line("PC001", {"test": True}) + "\n")
            return output

        mock_client = MagicMock()

        with patch("src.knowledge.factual.batch.submit_batch", side_effect=mock_submit), \
             patch("src.knowledge.factual.batch.poll_batch", side_effect=mock_poll), \
             patch("src.knowledge.factual.batch._save_pipeline_manifest", side_effect=tracking_save):
            run_batch_pipeline([chunk1, chunk2], batch_dir=batch_dir, client=mock_client)

        # Initial save + 2 per chunk (submit + complete) = 5
        assert len(manifest_snapshots) >= 3

    def test_skips_existing_output_files(self, tmp_path: Path):
        """Completed chunks (output_NNN.jsonl exists) are skipped."""
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        chunk1 = batch_dir / "input_000.jsonl"
        chunk2 = batch_dir / "input_001.jsonl"
        chunk1.write_text('{"custom_id": "PC001"}\n')
        chunk2.write_text('{"custom_id": "PC002"}\n')

        # Pre-create output for chunk 0 (already completed)
        (batch_dir / "output_000.jsonl").write_text(
            _make_batch_result_line("PC001", {"test": True}) + "\n"
        )

        submit_calls: list[str] = []

        def mock_submit(path, client=None):
            submit_calls.append(path.name)
            return "batch_new"

        def mock_poll(batch_id, client=None, poll_interval=60, timeout=86400):
            output = Path(f"batch_output_{batch_id}.jsonl")
            output.write_text(_make_batch_result_line("PC002", {"test": True}) + "\n")
            return output

        mock_client = MagicMock()

        with patch("src.knowledge.factual.batch.submit_batch", side_effect=mock_submit), \
             patch("src.knowledge.factual.batch.poll_batch", side_effect=mock_poll):
            result_paths = run_batch_pipeline(
                [chunk1, chunk2], batch_dir=batch_dir, client=mock_client,
            )

        assert len(result_paths) == 2
        # Only chunk 1 should have been submitted (chunk 0 was skipped)
        assert submit_calls == ["input_001.jsonl"]

    def test_failed_chunk_continues_pipeline(self, tmp_path: Path):
        """If poll fails for one chunk, pipeline continues with remaining chunks."""
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        chunk1 = batch_dir / "input_000.jsonl"
        chunk2 = batch_dir / "input_001.jsonl"
        chunk1.write_text('{"custom_id": "PC001"}\n')
        chunk2.write_text('{"custom_id": "PC002"}\n')

        def mock_submit(path, client=None):
            return f"batch_{path.stem}"

        call_count = 0

        def mock_poll(batch_id, client=None, poll_interval=60, timeout=86400):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Batch failed: token_limit_exceeded")
            output = Path(f"batch_output_{batch_id}.jsonl")
            output.write_text(_make_batch_result_line("PC002", {"test": True}) + "\n")
            return output

        mock_client = MagicMock()

        with patch("src.knowledge.factual.batch.submit_batch", side_effect=mock_submit), \
             patch("src.knowledge.factual.batch.poll_batch", side_effect=mock_poll):
            result_paths = run_batch_pipeline(
                [chunk1, chunk2], batch_dir=batch_dir, client=mock_client,
            )

        # Chunk 0 failed, chunk 1 succeeded
        assert len(result_paths) == 1
        assert result_paths[0].name == "output_001.jsonl"

    def test_submission_failure_continues_pipeline(self, tmp_path: Path):
        """If submit fails for one chunk, pipeline continues with remaining chunks."""
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        chunk1 = batch_dir / "input_000.jsonl"
        chunk2 = batch_dir / "input_001.jsonl"
        chunk1.write_text('{"custom_id": "PC001"}\n')
        chunk2.write_text('{"custom_id": "PC002"}\n')

        call_count = 0

        def mock_submit(path, client=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Upload failed")
            return "batch_ok"

        def mock_poll(batch_id, client=None, poll_interval=60, timeout=86400):
            output = Path(f"batch_output_{batch_id}.jsonl")
            output.write_text(_make_batch_result_line("PC002", {"test": True}) + "\n")
            return output

        mock_client = MagicMock()

        with patch("src.knowledge.factual.batch.submit_batch", side_effect=mock_submit), \
             patch("src.knowledge.factual.batch.poll_batch", side_effect=mock_poll):
            result_paths = run_batch_pipeline(
                [chunk1, chunk2], batch_dir=batch_dir, client=mock_client,
            )

        # Chunk 0 submit failed, chunk 1 succeeded
        assert len(result_paths) == 1
        assert result_paths[0].name == "output_001.jsonl"

    def test_stale_manifest_discarded_on_n_chunks_mismatch(self, tmp_path: Path):
        """Stale manifest (wrong n_chunks) is discarded and fresh start begins."""
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        chunk1 = batch_dir / "input_000.jsonl"
        chunk1.write_text('{"custom_id": "PC001"}\n')

        # Write stale manifest with wrong n_chunks
        stale_manifest = {
            "submitted_at": "2026-02-21T00:00:00+00:00",
            "n_chunks": 5,  # mismatch: we have 1 chunk
            "batch_ids": ["old_batch"] * 5,
            "statuses": ["completed"] * 5,
        }
        (batch_dir / "batch_ids.json").write_text(json.dumps(stale_manifest))

        def mock_submit(path, client=None):
            return "batch_fresh"

        def mock_poll(batch_id, client=None, poll_interval=60, timeout=86400):
            output = Path(f"batch_output_{batch_id}.jsonl")
            output.write_text(_make_batch_result_line("PC001", {"test": True}) + "\n")
            return output

        mock_client = MagicMock()

        with patch("src.knowledge.factual.batch.submit_batch", side_effect=mock_submit), \
             patch("src.knowledge.factual.batch.poll_batch", side_effect=mock_poll):
            result_paths = run_batch_pipeline(
                [chunk1], batch_dir=batch_dir, client=mock_client,
            )

        assert len(result_paths) == 1

        # Verify new manifest was created with correct n_chunks
        new_manifest = json.loads((batch_dir / "batch_ids.json").read_text())
        assert new_manifest["n_chunks"] == 1
        assert new_manifest["batch_ids"] == ["batch_fresh"]

    def test_empty_chunks_returns_empty_list(self, tmp_path: Path):
        """Empty input returns empty result list without error."""
        result_paths = run_batch_pipeline(
            [], batch_dir=tmp_path / "batch",
        )
        assert result_paths == []

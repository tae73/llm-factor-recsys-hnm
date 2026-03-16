"""Tests for CustomerCache with Parquet checkpoints."""

from pathlib import Path

from src.knowledge.reasoning.cache import CustomerCache


class TestInMemoryCache:
    """Test basic in-memory cache operations."""

    def test_put_and_get(self):
        cache = CustomerCache()
        cache.put("cust_001", {"reasoning_text": "prefers casual"})
        assert cache.get("cust_001") == {"reasoning_text": "prefers casual"}

    def test_get_missing(self):
        cache = CustomerCache()
        assert cache.get("nonexistent") is None

    def test_size(self):
        cache = CustomerCache()
        assert cache.size == 0
        cache.put("cust_001", {"reasoning_text": "a"})
        cache.put("cust_002", {"reasoning_text": "b"})
        assert cache.size == 2

    def test_overwrite(self):
        cache = CustomerCache()
        cache.put("cust_001", {"reasoning_text": "old"})
        cache.put("cust_001", {"reasoning_text": "new"})
        assert cache.get("cust_001")["reasoning_text"] == "new"
        assert cache.size == 1

    def test_keys(self):
        cache = CustomerCache()
        cache.put("cust_001", {})
        cache.put("cust_002", {})
        assert cache.keys() == {"cust_001", "cust_002"}


class TestCheckpoint:
    """Test Parquet checkpoint save/load."""

    def test_save_and_load(self, tmp_path: Path):
        cache1 = CustomerCache(checkpoint_dir=tmp_path)
        cache1.put("cust_001", {"reasoning_text": "casual lover", "n_purchases": 25})
        cache1.put("cust_002", {"reasoning_text": "formal style", "n_purchases": 10})
        cache1.save_checkpoint()

        cache2 = CustomerCache(checkpoint_dir=tmp_path)
        loaded = cache2.load_checkpoint()
        assert loaded == 2
        assert cache2.size == 2
        assert cache2.get("cust_001")["reasoning_text"] == "casual lover"
        assert cache2.get("cust_002")["n_purchases"] == 10

    def test_load_preserves_lists(self, tmp_path: Path):
        cache1 = CustomerCache(checkpoint_dir=tmp_path)
        cache1.put("cust_001", {"top_categories": ["T-shirt", "Trousers"]})
        cache1.save_checkpoint()

        cache2 = CustomerCache(checkpoint_dir=tmp_path)
        cache2.load_checkpoint()
        assert cache2.get("cust_001")["top_categories"] == ["T-shirt", "Trousers"]

    def test_load_preserves_dicts(self, tmp_path: Path):
        cache1 = CustomerCache(checkpoint_dir=tmp_path)
        cache1.put("cust_001", {"reasoning_json": {"style_mood": "Casual", "trend": "Classic"}})
        cache1.save_checkpoint()

        cache2 = CustomerCache(checkpoint_dir=tmp_path)
        cache2.load_checkpoint()
        assert cache2.get("cust_001")["reasoning_json"]["style_mood"] == "Casual"

    def test_load_nonexistent_checkpoint(self, tmp_path: Path):
        cache = CustomerCache(checkpoint_dir=tmp_path)
        loaded = cache.load_checkpoint()
        assert loaded == 0
        assert cache.size == 0

    def test_save_empty_cache(self, tmp_path: Path):
        cache = CustomerCache(checkpoint_dir=tmp_path)
        cache.save_checkpoint()
        assert not (tmp_path / "checkpoint.parquet").exists()

    def test_no_checkpoint_dir(self):
        cache = CustomerCache(checkpoint_dir=None)
        cache.put("cust_001", {})
        cache.save_checkpoint()
        assert cache.load_checkpoint() == 0

    def test_checkpoint_creates_dir(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        CustomerCache(checkpoint_dir=nested)
        assert nested.exists()

    def test_none_values_preserved(self, tmp_path: Path):
        cache1 = CustomerCache(checkpoint_dir=tmp_path)
        cache1.put("cust_001", {"reasoning_text": "hello", "optional_field": None})
        cache1.save_checkpoint()

        cache2 = CustomerCache(checkpoint_dir=tmp_path)
        cache2.load_checkpoint()
        assert cache2.get("cust_001")["optional_field"] is None

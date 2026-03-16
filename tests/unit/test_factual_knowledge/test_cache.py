"""Tests for ProductCodeCache with Parquet checkpoints."""

from pathlib import Path

from src.knowledge.factual.cache import ProductCodeCache


class TestInMemoryCache:
    """Test basic in-memory cache operations."""

    def test_put_and_get(self):
        cache = ProductCodeCache()
        cache.put("001", {"l1_material": "cotton"})
        assert cache.get("001") == {"l1_material": "cotton"}

    def test_get_missing(self):
        cache = ProductCodeCache()
        assert cache.get("nonexistent") is None

    def test_size(self):
        cache = ProductCodeCache()
        assert cache.size == 0
        cache.put("001", {"l1_material": "cotton"})
        cache.put("002", {"l1_material": "polyester"})
        assert cache.size == 2

    def test_overwrite(self):
        cache = ProductCodeCache()
        cache.put("001", {"l1_material": "cotton"})
        cache.put("001", {"l1_material": "silk"})
        assert cache.get("001")["l1_material"] == "silk"
        assert cache.size == 1

    def test_keys(self):
        cache = ProductCodeCache()
        cache.put("001", {})
        cache.put("002", {})
        assert cache.keys() == {"001", "002"}


class TestCheckpoint:
    """Test Parquet checkpoint save/load."""

    def test_save_and_load(self, tmp_path: Path):
        # Save
        cache1 = ProductCodeCache(checkpoint_dir=tmp_path)
        cache1.put("001", {"l1_material": "cotton", "l2_style_mood": ["Casual"]})
        cache1.put("002", {"l1_material": "polyester", "l2_perceived_quality": 3})
        cache1.save_checkpoint()

        # Load into new cache
        cache2 = ProductCodeCache(checkpoint_dir=tmp_path)
        loaded = cache2.load_checkpoint()
        assert loaded == 2
        assert cache2.size == 2
        assert cache2.get("001")["l1_material"] == "cotton"
        assert cache2.get("002")["l2_perceived_quality"] == 3

    def test_load_preserves_lists(self, tmp_path: Path):
        cache1 = ProductCodeCache(checkpoint_dir=tmp_path)
        cache1.put("001", {"l2_style_mood": ["Casual", "Minimalist"]})
        cache1.save_checkpoint()

        cache2 = ProductCodeCache(checkpoint_dir=tmp_path)
        cache2.load_checkpoint()
        assert cache2.get("001")["l2_style_mood"] == ["Casual", "Minimalist"]

    def test_load_nonexistent_checkpoint(self, tmp_path: Path):
        cache = ProductCodeCache(checkpoint_dir=tmp_path)
        loaded = cache.load_checkpoint()
        assert loaded == 0
        assert cache.size == 0

    def test_save_empty_cache(self, tmp_path: Path):
        cache = ProductCodeCache(checkpoint_dir=tmp_path)
        cache.save_checkpoint()  # Should not crash
        assert not (tmp_path / "checkpoint.parquet").exists()

    def test_no_checkpoint_dir(self):
        cache = ProductCodeCache(checkpoint_dir=None)
        cache.put("001", {})
        cache.save_checkpoint()  # Should not crash
        assert cache.load_checkpoint() == 0

    def test_checkpoint_creates_dir(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        ProductCodeCache(checkpoint_dir=nested)
        assert nested.exists()

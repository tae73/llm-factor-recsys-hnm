"""Tests for image loading, resizing, and base64 encoding."""

import base64
import io
from pathlib import Path

from PIL import Image

from src.knowledge.factual.image_utils import (
    find_article_image,
    get_image_for_article,
    load_and_encode_image,
)


def _create_test_image(path: Path, width: int = 800, height: int = 1000) -> None:
    """Create a test JPEG image at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color="red")
    img.save(path, format="JPEG")


class TestFindArticleImage:
    """Test image path resolution for H&M folder structure."""

    def test_finds_existing_image(self, tmp_path: Path):
        # H&M structure: images/0XX/0XXXXXXXXX.jpg
        article_id = "108775015"
        padded = article_id.zfill(10)  # "0108775015"
        folder = padded[:3]  # "010"
        image_path = tmp_path / folder / f"{padded}.jpg"
        _create_test_image(image_path)

        result = find_article_image(tmp_path, article_id)
        assert result == image_path

    def test_returns_none_for_missing(self, tmp_path: Path):
        result = find_article_image(tmp_path, "999999999")
        assert result is None

    def test_zero_padded_article_id(self, tmp_path: Path):
        article_id = "0123456789"
        folder = "012"
        image_path = tmp_path / folder / f"{article_id}.jpg"
        _create_test_image(image_path)

        result = find_article_image(tmp_path, article_id)
        assert result == image_path


class TestLoadAndEncodeImage:
    """Test image loading, resizing, and encoding."""

    def test_encodes_to_base64(self, tmp_path: Path):
        image_path = tmp_path / "test.jpg"
        _create_test_image(image_path)

        encoded = load_and_encode_image(image_path)
        assert encoded is not None
        # Verify it's valid base64
        decoded = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(decoded))
        assert img.format == "JPEG"

    def test_resizes_to_max_size(self, tmp_path: Path):
        image_path = tmp_path / "large.jpg"
        _create_test_image(image_path, width=2000, height=3000)

        encoded = load_and_encode_image(image_path, max_size=512)
        decoded = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(decoded))
        assert max(img.size) <= 512

    def test_small_image_not_upscaled(self, tmp_path: Path):
        image_path = tmp_path / "small.jpg"
        _create_test_image(image_path, width=200, height=150)

        encoded = load_and_encode_image(image_path, max_size=512)
        decoded = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(decoded))
        assert img.size[0] <= 200
        assert img.size[1] <= 150

    def test_returns_none_for_missing(self, tmp_path: Path):
        result = load_and_encode_image(tmp_path / "nonexistent.jpg")
        assert result is None


class TestGetImageForArticle:
    """Test combined find + encode."""

    def test_existing_image(self, tmp_path: Path):
        article_id = "108775015"
        padded = article_id.zfill(10)
        folder = padded[:3]
        _create_test_image(tmp_path / folder / f"{padded}.jpg")

        result = get_image_for_article(tmp_path, article_id)
        assert result is not None
        assert isinstance(result, str)

    def test_missing_image(self, tmp_path: Path):
        result = get_image_for_article(tmp_path, "999999999")
        assert result is None

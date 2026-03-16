"""Image loading, resizing, and base64 encoding for multimodal LLM extraction.

H&M image folder structure: images/{padded[:3]}/{padded}.jpg  (padded = article_id.zfill(10))
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def find_article_image(images_dir: Path, article_id: str) -> Path | None:
    """Locate product image for an article_id.

    H&M images are stored as: images/{padded[:3]}/{padded}.jpg  (padded = article_id.zfill(10))
    """
    padded = article_id.zfill(10)
    folder = padded[:3]
    image_path = images_dir / folder / f"{padded}.jpg"
    return image_path if image_path.exists() else None


def load_and_encode_image(image_path: Path, max_size: int = 512) -> str | None:
    """Load image, resize to max_size, and return base64-encoded JPEG string.

    Returns None if the image cannot be loaded.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            # Resize maintaining aspect ratio
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        logger.warning("Failed to load image: %s", image_path)
        return None


def get_image_for_article(
    images_dir: Path, article_id: str, max_size: int = 512
) -> str | None:
    """Find and encode image for an article. Returns base64 string or None."""
    path = find_article_image(images_dir, article_id)
    if path is None:
        return None
    return load_and_encode_image(path, max_size=max_size)

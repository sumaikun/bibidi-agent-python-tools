"""
Shared image utilities — decoding, hashing, saving.
"""

import base64
import hashlib
from datetime import datetime
from io import BytesIO

from PIL import Image

from vision_api.config import SCREENSHOTS_DIR


def page_key(url: str) -> str:
    """URL → short hash key for indexing."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def image_size(b64: str) -> tuple[int, int]:
    """Base64 image → (width, height)."""
    data = base64.b64decode(b64)
    img = Image.open(BytesIO(data))
    return img.size


def center(bbox: list[float], w: int, h: int) -> dict:
    """Normalized bbox [x1, y1, x2, y2] → pixel center {x, y}."""
    return {
        "x": int((bbox[0] + bbox[2]) / 2 * w),
        "y": int((bbox[1] + bbox[3]) / 2 * h),
    }


def save_images(key: str, original_b64: str, annotated_b64: str | None) -> tuple[str, str]:
    """Save original and annotated screenshots to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_path = SCREENSHOTS_DIR / f"{key}_{timestamp}_original.png"
    annotated_path = SCREENSHOTS_DIR / f"{key}_{timestamp}_annotated.png"
    original_path.write_bytes(base64.b64decode(original_b64))
    if annotated_b64:
        annotated_path.write_bytes(base64.b64decode(annotated_b64))
    return str(original_path), str(annotated_path)


def elements_to_text(elements: list) -> list[tuple]:
    """Elements list → txtai-indexable (id, text) tuples."""
    return [
        (str(i), f"{el.get('content', el.get('label', ''))} {el.get('type', '')} {el.get('source', '')}")
        for i, el in enumerate(elements)
    ]

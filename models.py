"""
Model registry — eager load all models at startup.

Call `load_all()` from the FastAPI lifespan handler.
After that, import any model handle directly:

    from vision_api.models import ui_yolo, sam3_finder, ...
"""

import easyocr
import torch
from txtai.embeddings import Embeddings
from ultralytics import YOLO

from vision_api.config import YOLO_MODEL_PATH, CAPTCHA_MODEL_PATH
from vision_api.services.segment import SAM3UIFinder

# --- Global handles (None until load_all is called) ---

ui_yolo: YOLO | None = None
captcha_yolo: YOLO | None = None
ocr_reader: easyocr.Reader | None = None
sam3_finder: SAM3UIFinder | None = None
embeddings: Embeddings | None = None

# --- txtai in-memory index (page_key -> elements) ---

element_index: dict[str, list[dict]] = {}


def load_all():
    """Load every model eagerly. Called once at startup."""
    global ui_yolo, captcha_yolo, ocr_reader, sam3_finder, embeddings

    print("=" * 60)
    print("  Loading all models...")
    print("=" * 60)

    print("[1/5] UI YOLO...")
    ui_yolo = YOLO(YOLO_MODEL_PATH)

    print("[2/5] CAPTCHA YOLO...")
    captcha_yolo = YOLO(CAPTCHA_MODEL_PATH)

    print("[3/5] EasyOCR...")
    ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    ocr_reader.detector.float()

    print("[4/5] SAM3...")
    sam3_finder = SAM3UIFinder()
    torch.set_default_dtype(torch.float32)

    print("[5/5] txtai embeddings...")
    embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})

    print("=" * 60)
    print("  All models loaded. Ready.")
    print("=" * 60)


def shutdown():
    """Clean shutdown."""
    if sam3_finder:
        sam3_finder.shutdown()

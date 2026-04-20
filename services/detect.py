"""
Detection service — OmniParser + custom UI YOLO + EasyOCR.
Saves annotated image + element JSON for debugging.
"""

import base64
import json
from datetime import datetime
from io import BytesIO

import httpx
import numpy as np
import torch
from PIL import Image

from vision_api.config import OMNIPARSER_URL, YOLO_CONFIDENCE, INTERACTIVE_CLASSES, SCREENSHOTS_DIR
from vision_api import models
from vision_api.utils import center, image_size, page_key


def run_yolo(image_bytes: bytes, img_w: int, img_h: int) -> list[dict]:
    """Run custom UI YOLO + RapidOCR on raw image bytes. Skips elements with no OCR text."""
    img_np = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
    results = models.ui_yolo.predict(source=img_np, conf=YOLO_CONFIDENCE, verbose=False)
    elements = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cls = result.names[int(box.cls)]
            conf = round(float(box.conf), 3)

            crop = img_np[y1:y2, x1:x2]

            # RapidOCR swap — no more autocast wrapper, no more torch contamination
            if crop.size > 0:
                ocr_result, _ = models.ocr_reader(crop)
                texts = [item[1] for item in (ocr_result or [])]
            else:
                texts = []

            text = " ".join(texts).strip()
            
            #print("text", text)

            if not text:
                continue

            elements.append({
                "content": text,
                "label": text,
                "type": cls,
                "interactivity": cls in INTERACTIVE_CLASSES,
                "source": "yolo",
                "confidence": conf,
                "bbox": [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h],
                "x": (x1 + x2) // 2,
                "y": (y1 + y2) // 2,
            })

    return elements


def _save_detection(key: str, screenshot_b64: str, annotated_b64: str | None, elements: list) -> dict:
    """Save original image, annotated image, and element JSON. Returns file paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{key}_{timestamp}"

    original_path = SCREENSHOTS_DIR / f"{prefix}_original.png"
    annotated_path = SCREENSHOTS_DIR / f"{prefix}_annotated.png"
    json_path = SCREENSHOTS_DIR / f"{prefix}_elements.json"

    original_path.write_bytes(base64.b64decode(screenshot_b64))

    if annotated_b64:
        annotated_path.write_bytes(base64.b64decode(annotated_b64))

    # Save elements with index numbers matching the annotated image
    numbered = [
        {
            "index": i,
            "content": el.get("content", ""),
            "type": el.get("type", ""),
            "source": el.get("source", ""),
            "interactivity": el.get("interactivity", False),
            "x": el.get("x", 0),
            "y": el.get("y", 0),
            "bbox": el.get("bbox", []),
        }
        for i, el in enumerate(elements)
    ]

    json_path.write_text(json.dumps({
        "page_key": key,
        "timestamp": timestamp,
        "total": len(elements),
        "elements": numbered,
    }, indent=2))

    return {
        "original_image": str(original_path),
        "annotated_image": str(annotated_path) if annotated_b64 else None,
        "elements_json": str(json_path),
    }


async def detect(screenshot_b64: str, url: str) -> dict:
    """
    Full detection pipeline: OmniParser + UI YOLO.
    Returns elements, annotated image (base64), and saves all artifacts to disk.
    """
    image_bytes = base64.b64decode(screenshot_b64)
    w, h = image_size(screenshot_b64)
    key = page_key(url)

    omniparser_elements = []
    annotated_b64 = None

    # --- OmniParser (external) ---
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{OMNIPARSER_URL}/parse/",
                json={"base64_image": screenshot_b64},
            )
        if resp.status_code == 200:
            data = resp.json()
            annotated_b64 = data.get("som_image_base64")
            omniparser_elements = [
                {
                    "content": el["content"],
                    "label": el["content"],
                    "type": el.get("type", "text"),
                    "interactivity": el.get("interactivity", False),
                    "source": "omniparser",
                    "bbox": el["bbox"],
                    **center(el["bbox"], w, h),
                }
                for el in data.get("parsed_content_list", [])
            ]
    except Exception as e:
        print(f"OmniParser error: {e}")

    # --- UI YOLO ---
    yolo_elements = run_yolo(image_bytes, w, h)

    # --- Merge ---
    all_elements = omniparser_elements + yolo_elements

    # --- Save all artifacts (images + JSON) ---
    paths = _save_detection(key, screenshot_b64, annotated_b64, all_elements)

    # --- Build summary ---
    summary = _build_summary(all_elements)

    return {
        "page_key": key,
        "elements": all_elements,
        "annotated_b64": annotated_b64,
        "total": len(all_elements),
        "omniparser": len(omniparser_elements),
        "yolo": len(yolo_elements),
        **paths,
        "summary": summary,
    }


def _build_summary(elements: list) -> dict:
    """Build a structured summary of detected elements."""
    interactive = []
    text_fragments = []

    for i, el in enumerate(elements):
        entry = {
            "index": i,
            "content": el.get("content", ""),
            "type": el.get("type", ""),
            "source": el.get("source", ""),
            "x": el.get("x", 0),
            "y": el.get("y", 0),
        }

        if el.get("interactivity"):
            interactive.append(entry)
        else:
            text_fragments.append(entry)

    return {
        "interactive": {
            "count": len(interactive),
            "elements": interactive,
        },
        "text": {
            "count": len(text_fragments),
            "elements": text_fragments,
        },
    }
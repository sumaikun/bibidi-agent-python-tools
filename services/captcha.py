"""
CAPTCHA solving service.

Pipeline: determine object + dims (VLM) → segment object (SAM3 direct) →
detect grid (YOLO) → map mask to grid cells → click targets.

Order note: SAM3 runs before YOLO/VLM when an object_hint is provided.
The route must be declared `async def` so it runs on the event loop —
a sync `def` route drops the call into a threadpool worker where SAM3's
first forward pass fails with a bf16/fp32 mismatch.
"""

import base64
import json
import os
import shutil
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vision_api.config import CAPTCHA_OUTPUT_DIR, YOLO_CONFIDENCE
from vision_api import models
from vision_api.vlm import vlm_see, text_chat


# ============================================
# GRID DETECTION
# ============================================

def detect_grid(image_path: str) -> dict | None:
    """Detect CAPTCHA grid bounding box using dedicated YOLO model."""
    results = models.captcha_yolo(image_path, conf=YOLO_CONFIDENCE, verbose=False)
    best = None
    best_conf = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best = {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": x2 - x1, "height": y2 - y1,
                    "confidence": round(conf, 3),
                }
    return best


# ============================================
# VLM ANALYSIS
# ============================================

def ask_vlm(image_path: str) -> dict:
    """Two-stage: VLM analyzes image → text LLM parses to JSON."""
    image_data = open(image_path, "rb").read()

    print("  [VLM] Stage 1: analyzing image...")
    description = vlm_see(
        "This is a CAPTCHA image. Describe:\n"
        "- What object does it ask you to select?\n"
        "- How many rows and columns of image tiles are in the grid? "
        "Count them carefully.",
        image_data=image_data,
    )
    #print(f"  [VLM] says: {description[:200]}...")
    print(f"  [VLM] says: {description}")

    print("  [VLM] Stage 2: parsing with text LLM...")
    text = text_chat(
        "Extract the following from this CAPTCHA description and "
        "respond ONLY with valid JSON, nothing else:\n"
        '{"object": "single word object name", "rows": number, "cols": number}\n\n'
        "IMPORTANT: 'object' must be a single word — just the object itself. "
        "Examples: 'bus', 'bicycle', 'crosswalk', 'motorcycle', 'traffic light', 'hydrant'.\n"
        "Do NOT include phrases like 'squares containing' or 'images with'.\n\n"
        f"Description:\n{description}"
    )

    text = text.strip()
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        result = json.loads(text[start:end])
        obj = result.get("object", "")
        for prefix in [
            "squares containing ", "images with ", "images of ",
            "pictures of ", "all ", "select ",
        ]:
            if obj.lower().startswith(prefix):
                obj = obj[len(prefix):]
        result["object"] = obj.strip()
        return result
    except (ValueError, json.JSONDecodeError):
        print(f"  [VLM] Parse failed: {text}")
        return {}


# ============================================
# SAM3 DIRECT SEGMENTATION
# ============================================

def segment_object(image_path: str, prompt: str) -> list[dict]:
    """Segment all instances of `prompt` in the image via SAM3."""
    raw_results = models.sam3_finder.find_multiple(image_path, [prompt])

    return [
        {
            "found": True,
            "prompt": r["prompt"],
            "instance_id": r.get("instance_id", 0),
            "bbox": r["bbox"],
            "mask": r["mask"],
            "mask_area_px": r["mask_area_px"],
        }
        for r in raw_results
        if r["found"]
    ]


# ============================================
# MASK → GRID CELL MAPPING
# ============================================

def map_mask_to_cells(
    masks: list[np.ndarray], grid: dict,
    rows: int, cols: int, min_overlap_px: int = 20,
) -> list[dict]:
    """Map merged masks onto grid cells, return all cells with hit status."""
    gx1, gy1 = grid["x1"], grid["y1"]
    gx2, gy2 = grid["x2"], grid["y2"]

    if rows <= 0 or cols <= 0 or gx2 <= gx1 or gy2 <= gy1:
        print(f"  [grid] Invalid grid: {rows}x{cols} bbox=({gx1},{gy1})->({gx2},{gy2})")
        return []

    cell_w = (gx2 - gx1) / cols
    cell_h = (gy2 - gy1) / rows

    h = max(m.shape[0] for m in masks)
    w = max(m.shape[1] for m in masks)
    merged = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        merged[:m.shape[0], :m.shape[1]] = np.maximum(
            merged[:m.shape[0], :m.shape[1]], m
        )

    print(f"  [grid] Merged mask: {merged.shape}, total px: {int(np.sum(merged > 0))}")
    print(f"  [grid] Grid: ({gx1},{gy1})->({gx2},{gy2}) {rows}x{cols} cell={cell_w:.0f}x{cell_h:.0f}")

    cells = []
    for r in range(rows):
        for c in range(cols):
            cx1 = int(gx1 + c * cell_w)
            cy1 = int(gy1 + r * cell_h)
            cx2 = int(gx1 + (c + 1) * cell_w)
            cy2 = int(gy1 + (r + 1) * cell_h)

            my1 = max(0, min(cy1, merged.shape[0]))
            my2 = max(0, min(cy2, merged.shape[0]))
            mx1 = max(0, min(cx1, merged.shape[1]))
            mx2 = max(0, min(cx2, merged.shape[1]))

            cell_mask = merged[my1:my2, mx1:mx2]
            overlap = int(np.sum(cell_mask > 0))
            cell_area = (mx2 - mx1) * (my2 - my1)
            pct = round(overlap / cell_area * 100, 1) if cell_area > 0 else 0

            hit = overlap >= min_overlap_px
            tag = "✓ CLICK" if hit else ""
            print(f"    [{r},{c}] ({cx1},{cy1})->({cx2},{cy2}) overlap={overlap}px ({pct}%) {tag}")

            cells.append({
                "row": r, "col": c,
                "x": (cx1 + cx2) // 2, "y": (cy1 + cy2) // 2,
                "x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2,
                "overlap": overlap, "pct": pct, "click": hit,
            })
    return cells


# ============================================
# VISUALIZATION
# ============================================

def draw_captcha_analysis(
    image_path: str, grid: dict, cells: list[dict],
    masks: list[np.ndarray], prompt: str, rows: int, cols: int,
) -> str:
    """Draw grid, mask overlay, and X marks on cells to click."""
    img = cv2.imread(image_path)

    for mask in masks:
        overlay = img.copy()
        overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        img = overlay

    cv2.rectangle(img, (grid["x1"], grid["y1"]), (grid["x2"], grid["y2"]), (255, 255, 0), 2)

    for cell in cells:
        cv2.rectangle(img, (cell["x1"], cell["y1"]), (cell["x2"], cell["y2"]), (200, 200, 200), 1)
        if cell["click"]:
            pad = 10
            cv2.line(img, (cell["x1"] + pad, cell["y1"] + pad),
                     (cell["x2"] - pad, cell["y2"] - pad), (0, 0, 255), 4)
            cv2.line(img, (cell["x2"] - pad, cell["y1"] + pad),
                     (cell["x1"] + pad, cell["y2"] - pad), (0, 0, 255), 4)
            cv2.putText(img, f"{cell['overlap']}px",
                        (cell["x1"] + 5, cell["y2"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    click_count = sum(1 for c in cells if c["click"])
    title = f"'{prompt}' {rows}x{cols} -> {click_count} cells to click"
    cv2.putText(img, title, (10, img.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = str(CAPTCHA_OUTPUT_DIR / f"{timestamp}_analysis.jpg")
    cv2.imwrite(out, img)
    return out


# ============================================
# FULL SOLVE PIPELINE
# ============================================

def solve(screenshot_b64: str, object_hint: str = "", rows: int = 0, cols: int = 0, min_overlap: int = 20) -> dict:
    """
    Full CAPTCHA solving pipeline:
    1. Detect grid (YOLO)
    2. Determine object + grid dims (VLM)
    3. Segment object (SAM3 direct)
    4. Map mask to grid cells
    5. Draw annotated debug image
    6. Return click targets
    """
    if models.sam3_finder is None:
        return {"solved": False, "error": "SAM3 model not loaded yet", "click_targets": []}

    try:
        image_bytes = base64.b64decode(screenshot_b64)
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"solved": False, "error": f"Invalid image bytes: {e}", "click_targets": []}

    img_w, img_h = pil_img.size

    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3_solve_"))
    image_path = tmp_dir / "input.jpg"

    try:
        pil_img.save(str(image_path))

        print(f"\n{'=' * 50}")
        print(f"  [solve] image: {img_w}x{img_h}")
        print(f"{'=' * 50}")

        # --- Determine target (VLM only if caller didn't provide hints) ---
        prompt = object_hint
        if not prompt or rows == 0 or cols == 0:
            print("\n[1] Asking VLM for target + grid dims...")
            try:
                vlm = ask_vlm(str(image_path))
                prompt = prompt or vlm.get("object", "")
                rows = rows or vlm.get("rows", 0)
                cols = cols or vlm.get("cols", 0)
            except Exception as e:
                return {
                    "solved": False,
                    "error": f"VLM failed: {e}. Retry or pass object_hint/rows/cols manually.",
                    "click_targets": [],
                    "vlm_error": True,
                }

        if not prompt:
            return {
                "solved": False,
                "error": f"Could not determine target object: '{prompt}'",
                "click_targets": [],
            }

        # Normalize grid — reCAPTCHA is always 3x3 or 4x4
        if rows and cols and rows != cols:
            # VLM is more reliable counting rows than cols — trust rows as the
            # canonical grid size (reCAPTCHA is always square: 3x3 or 4x4).
            cols = rows
            # Clamp to valid reCAPTCHA sizes
            if rows not in (3, 4):
                rows = cols = 4 if rows > 3 else 3
            print(f"  Grid normalized to rows: {rows}x{cols}")

        # --- SAM3 segmentation ---
        print(f"\n[2] Running SAM3 for '{prompt}'...")
        found = segment_object(str(image_path), prompt)
        if not found:
            return {
                "solved": False,
                "error": f"SAM3 found nothing for '{prompt}'",
                "click_targets": [],
                "prompt": prompt, "rows": rows, "cols": cols,
            }

        masks = [r["mask"] for r in found if r.get("mask") is not None]
        print(f"  ✓ {len(found)} instance(s), {len(masks)} mask(s)")
        for i, r in enumerate(found):
            print(f"    instance {i}: area={r['mask_area_px']}px "
                  f"bbox=({r['bbox']['x1']},{r['bbox']['y1']})->"
                  f"({r['bbox']['x2']},{r['bbox']['y2']})")

        # --- Grid detection (YOLO) ---
        print("\n[3] Detecting grid...")
        grid = detect_grid(str(image_path))
        if grid is None:
            return {
                "solved": False,
                "error": "No CAPTCHA grid detected",
                "click_targets": [],
                "prompt": prompt, "sam3_instances": len(found),
            }
        print(f"  ✓ Grid: ({grid['x1']},{grid['y1']})->({grid['x2']},{grid['y2']}) "
              f"size={grid['width']}x{grid['height']} conf={grid['confidence']}")

        if rows == 0 or cols == 0:
            return {
                "solved": False,
                "error": f"Missing grid dims: rows={rows} cols={cols}",
                "click_targets": [], "grid": grid,
                "prompt": prompt,
            }

        print(f"  ✓ Mapping '{prompt}' in {rows}x{cols} grid")

        # --- Map masks to grid cells ---
        print(f"\n[4] Mapping masks to {rows}x{cols} grid...")
        cells = map_mask_to_cells(masks, grid, rows, cols, min_overlap)

        click_cells = [c for c in cells if c["click"]]
        print(f"\n  → {len(click_cells)} cell(s) to CLICK")
        for c in click_cells:
            print(f"    [{c['row']},{c['col']}] center=({c['x']},{c['y']}) overlap={c['overlap']}px")

        # --- Draw annotated debug image ---
        print("\n[5] Drawing analysis...")
        annotated_path = draw_captcha_analysis(str(image_path), grid, cells, masks, prompt, rows, cols)
        print(f"  ✓ Saved: {annotated_path}")

        click_targets = [
            {"x": c["x"], "y": c["y"], "row": c["row"], "col": c["col"], "overlap": c["overlap"]}
            for c in click_cells
        ]

        return {
            "solved": True,
            "prompt": prompt,
            "rows": rows, "cols": cols,
            "grid": grid,
            "click_targets": click_targets,
            "total_mask_px": sum(r["mask_area_px"] for r in found),
            "sam3_instances": len(found),
            "image_size": {"width": img_w, "height": img_h},
            "annotated_image": annotated_path,
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
"""
Route: /segment
"""

import base64
import shutil
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, HTTPException
from PIL import Image

from vision_api.config import SAM3_OUTPUT_DIR
from vision_api import models 
from vision_api.schemas import SegmentRequest
from vision_api.services.captcha import map_mask_to_cells
from vision_api.services.segment import draw_sam3_results

router = APIRouter()


@router.post("/segment")
async def segment(req: SegmentRequest):
    if models.sam3_finder is None:
        raise HTTPException(503, "SAM3 model not loaded yet")

    try:
        image_bytes = base64.b64decode(req.screenshot)
    except Exception:
        raise HTTPException(400, "Invalid base64 in 'screenshot' field")

    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3_api_"))
    image_path = tmp_dir / "input.jpg"

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img.save(str(image_path))

        raw_results = models.sam3_finder.find_multiple(str(image_path), req.prompts)

        results = []
        for r in raw_results:
            polygon = r.get("polygon")
            if polygon and len(polygon) > 20:
                step = len(polygon) // 20
                polygon = polygon[::step][:20]

            click_target = None
            click_targets = None

            if r["found"] and r["bbox"]:
                if req.grid:
                    mask = r.get("mask")
                    if mask is not None:
                        cells = map_mask_to_cells(
                            [mask], req.grid,
                            req.grid.get("rows", 3), req.grid.get("cols", 3),
                        )
                        click_targets = [{"x": c["x"], "y": c["y"]} for c in cells if c["click"]]
                    else:
                        click_target = {"x": r["bbox"]["center_x"], "y": r["bbox"]["center_y"]}
                else:
                    click_target = {"x": r["bbox"]["center_x"], "y": r["bbox"]["center_y"]}

            result_entry = {
                "prompt": r["prompt"],
                "instance_id": r.get("instance_id", 0),
                "found": r["found"],
                "bbox": r["bbox"],
                "polygon": polygon,
                "mask_area_px": r["mask_area_px"],
                "click_target": click_target,
            }
            if click_targets is not None:
                result_entry["click_targets"] = click_targets

            results.append(result_entry)

        found = [r for r in results if r["found"]]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompts_slug = "_".join(p.replace(" ", "-") for p in req.prompts[:3])[:40]
        original_path = str(SAM3_OUTPUT_DIR / f"{timestamp}_{prompts_slug}_original.jpg")
        img.save(original_path)

        annotated_path = None
        if req.save_annotated:
            annotated_out = str(SAM3_OUTPUT_DIR / f"{timestamp}_{prompts_slug}_annotated.jpg")
            annotated_path = draw_sam3_results(str(image_path), raw_results, annotated_out)

        return {
            "results": results,
            "found": len(found),
            "total": len(results),
            "original_image": original_path,
            "annotated_image": annotated_path,
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

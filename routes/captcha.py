"""
Routes: /detect_grid, /solve
"""

import base64
import os
import tempfile

from fastapi import APIRouter
from PIL import Image

from vision_api.schemas import DetectGridRequest, SolveRequest
from vision_api.services import captcha as captcha_svc

router = APIRouter()


@router.post("/detect_grid")
def api_detect_grid(req: DetectGridRequest):
    image_bytes = base64.b64decode(req.screenshot)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        img_pil = Image.open(tmp_path)
        print(f"[detect_grid] image: size={img_pil.size}, mode={img_pil.mode}")

        grid = captcha_svc.detect_grid(tmp_path)
        print(f"[detect_grid] result: {grid}")

        return {
            "found": grid is not None,
            "grid": grid,
            "image_size": {"width": img_pil.size[0], "height": img_pil.size[1]},
        }
    finally:
        os.unlink(tmp_path)


@router.post("/solve")
def api_solve(req: SolveRequest):
    return captcha_svc.solve(
        req.screenshot, req.object_hint, req.rows, req.cols, req.min_overlap,
    )

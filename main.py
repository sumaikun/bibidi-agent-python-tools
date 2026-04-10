"""
Vision API — Unified entrypoint.

Single FastAPI service combining:
  - OmniParser + YOLO detection
  - Ollama VLM description
  - txtai semantic element search
  - SAM3 segmentation
  - CAPTCHA solving pipeline

Usage:
    python -m vision_api.main

OmniParser remains an external dependency (separate server).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from vision_api import models
from vision_api.config import PORT
from vision_api.middleware import ObserverMiddleware
from vision_api.routes.vision import router as vision_router
from vision_api.routes.segment import router as segment_router
from vision_api.routes.captcha import router as captcha_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    models.load_all()
    yield
    models.shutdown()


app = FastAPI(
    title="Vision API — Unified",
    description="OmniParser + YOLO + txtai + SAM3 + CAPTCHA solver",
    lifespan=lifespan,
)

app.include_router(vision_router)
app.include_router(segment_router)
app.include_router(captcha_router)


@app.get("/probe")
def probe():
    return {
        "status": "ok",
        "models": {
            "ui_yolo": models.ui_yolo is not None,
            "captcha_yolo": models.captcha_yolo is not None,
            "ocr": models.ocr_reader is not None,
            "sam3": models.sam3_finder is not None,
            "txtai": models.embeddings is not None,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("vision_api.main:app", host="0.0.0.0", port=PORT, reload=False)
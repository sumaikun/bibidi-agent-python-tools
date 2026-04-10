"""
VLM / Text LLM provider wrapper.

Supports: ollama (local), ollama-cloud, anthropic.
All config comes from config.py.

Usage:
    from vision_api.vlm import vlm_see, text_chat

    result = vlm_see("What is on this page?", image_data=b"...")
    result = text_chat("Extract JSON from: ...")
"""

import base64
from typing import Optional

import httpx
import ollama
from ollama import Client

from vision_api.config import (
    VLM_PROVIDER, VLM_MODEL,
    TEXT_PROVIDER, TEXT_MODEL,
    OLLAMA_BASE_URL, OLLAMA_CLOUD_URL, OLLAMA_API_KEY,
    ANTHROPIC_API_KEY,
)


# ============================================
# OLLAMA LOCAL
# ============================================

def _ollama_local_vision(model: str, prompt: str, image_data: bytes) -> str:
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_data],
        }],
    )
    return response["message"]["content"]


def _ollama_local_text(model: str, prompt: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# ============================================
# OLLAMA CLOUD
# ============================================

def _ollama_cloud_client() -> Client:
    if not OLLAMA_CLOUD_URL:
        raise ValueError("OLLAMA_CLOUD_URL not set")
    headers = {}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    return Client(host=OLLAMA_CLOUD_URL, headers=headers)


def _ollama_cloud_vision(model: str, prompt: str, image_data: bytes) -> str:
    client = _ollama_cloud_client()
    img_b64 = base64.b64encode(image_data).decode("utf-8")
    response = client.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [img_b64],
        }],
    )
    return response["message"]["content"]


def _ollama_cloud_text(model: str, prompt: str) -> str:
    client = _ollama_cloud_client()
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# ============================================
# ANTHROPIC
# ============================================

def _detect_media_type(image_data: bytes) -> str:
    if image_data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_data[:4] == b"\x89PNG":
        return "image/png"
    if image_data[:4] == b"RIFF":
        return "image/webp"
    return "image/png"


def _anthropic_vision(model: str, prompt: str, image_data: bytes) -> str:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": _detect_media_type(image_data),
                            "data": base64.b64encode(image_data).decode("utf-8"),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def _anthropic_text(model: str, prompt: str) -> str:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


# ============================================
# DISPATCH
# ============================================

_VISION = {
    "ollama":       _ollama_local_vision,
    "ollama-cloud": _ollama_cloud_vision,
    "anthropic":    _anthropic_vision,
}

_TEXT = {
    "ollama":       _ollama_local_text,
    "ollama-cloud": _ollama_cloud_text,
    "anthropic":    _anthropic_text,
}


# ============================================
# PUBLIC API
# ============================================

def vlm_see(
    prompt: str,
    image_data: Optional[bytes] = None,
    image_b64: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Send image + prompt to vision LLM, return text response.

    Accepts either raw bytes (image_data) or base64 string (image_b64).
    """
    provider = provider or VLM_PROVIDER
    model = model or VLM_MODEL

    if image_data is None and image_b64 is None:
        raise ValueError("Provide either image_data or image_b64")
    if image_data is None:
        image_data = base64.b64decode(image_b64)

    fn = _VISION.get(provider)
    if fn is None:
        raise ValueError(f"Unknown VLM provider: {provider}")

    return fn(model, prompt, image_data)


def text_chat(
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Send text prompt to LLM, return text response."""
    provider = provider or TEXT_PROVIDER
    model = model or TEXT_MODEL

    fn = _TEXT.get(provider)
    if fn is None:
        raise ValueError(f"Unknown text provider: {provider}")

    return fn(model, prompt)
"""
Compare two screenshots (before/after a click) using VLM.

Returns a structured verdict: what changed, what disappeared, what appeared,
and whether the expected action (e.g. "close popup") likely succeeded.
"""

import base64

from vision_api.vlm import vlm_see


COMPARE_PROMPT = """You are verifying if a browser action worked by comparing two screenshots.

IMAGE 1 (LEFT): The page BEFORE the action.
IMAGE 2 (RIGHT): The page AFTER the action.

The user expected this to happen: "{expectation}"

Analyze both images and respond with EXACTLY this format:

VERDICT: SUCCESS | FAILED | UNCLEAR
CHANGED: brief description of what visually changed between the two screenshots
DISAPPEARED: list elements that were in the BEFORE but not in the AFTER (or "nothing")
APPEARED: list elements that are in the AFTER but not in the BEFORE (or "nothing")
SUMMARY: one sentence explaining whether the expected action succeeded and why

Rules:
- Focus on the EXPECTED ACTION, not unrelated page changes
- If a popup/modal was supposed to close, check if it's gone — ignore other ads or banners
- If a dropdown was supposed to open, check if new options appeared
- Be specific about what you see, not what you assume
"""


def compare(before_b64: str, after_b64: str, expectation: str) -> dict:
    """
    Compare two screenshots and return a verdict.

    Args:
        before_b64: base64 screenshot before the action
        after_b64:  base64 screenshot after the action
        expectation: what the agent expected to happen (e.g. "close the popup")

    Returns:
        dict with verdict, changed, disappeared, appeared, summary
    """
    before_bytes = base64.b64decode(before_b64)
    after_bytes = base64.b64decode(after_b64)

    # Combine both images side by side by sending them in sequence
    # Most VLMs handle multiple images or we can use a combined prompt
    prompt = COMPARE_PROMPT.format(expectation=expectation)

    # Try sending as two separate images via the anthropic multi-image approach
    # Fall back to sending them one at a time if needed
    try:
        result = _compare_two_images(prompt, before_bytes, after_bytes)
    except Exception:
        # Fallback: send after image with context about what to look for
        fallback_prompt = f"""The user just clicked something on a webpage.
Expected result: "{expectation}"
Is this screenshot showing that the action succeeded? Describe what you see.
Respond with: VERDICT: SUCCESS | FAILED | UNCLEAR followed by SUMMARY: one sentence."""
        result = vlm_see(fallback_prompt, image_data=after_bytes)

    return _parse_verdict(result, expectation)


def _compare_two_images(prompt: str, before: bytes, after: bytes) -> str:
    """Send two images to VLM for comparison."""
    import httpx
    from vision_api.config import (
        VLM_PROVIDER, VLM_MODEL,
        ANTHROPIC_API_KEY,
        OLLAMA_BASE_URL,
    )
    from vision_api.vlm import _detect_media_type

    if VLM_PROVIDER == "anthropic":
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": VLM_MODEL,
                "max_tokens": 1024,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "IMAGE 1 - BEFORE the action:"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": _detect_media_type(before),
                                "data": base64.b64encode(before).decode("utf-8"),
                            },
                        },
                        {
                            "type": "text",
                            "text": "IMAGE 2 - AFTER the action:"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": _detect_media_type(after),
                                "data": base64.b64encode(after).decode("utf-8"),
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

    else:
        # Ollama: send both images in one message
        before_b64 = base64.b64encode(before).decode("utf-8")
        after_b64 = base64.b64encode(after).decode("utf-8")

        import ollama as ollama_lib
        if VLM_PROVIDER == "ollama-cloud":
            from vision_api.vlm import _ollama_cloud_client
            client = _ollama_cloud_client()
        else:
            client = ollama_lib

        response = client.chat(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": f"IMAGE 1 - BEFORE:\n(see first image)\n\nIMAGE 2 - AFTER:\n(see second image)\n\n{prompt}",
                "images": [before_b64, after_b64],
            }],
        )
        return response["message"]["content"]


def _parse_verdict(raw: str, expectation: str) -> dict:
    """Parse VLM response into structured dict."""
    lines = raw.strip().split("\n")

    result = {
        "expectation": expectation,
        "verdict": "UNCLEAR",
        "changed": "",
        "disappeared": "",
        "appeared": "",
        "summary": "",
        "raw": raw,
    }

    for line in lines:
        line = line.strip()
        if line.startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().upper()
            if "SUCCESS" in v:
                result["verdict"] = "SUCCESS"
            elif "FAILED" in v:
                result["verdict"] = "FAILED"
            else:
                result["verdict"] = "UNCLEAR"
        elif line.startswith("CHANGED:"):
            result["changed"] = line.split(":", 1)[1].strip()
        elif line.startswith("DISAPPEARED:"):
            result["disappeared"] = line.split(":", 1)[1].strip()
        elif line.startswith("APPEARED:"):
            result["appeared"] = line.split(":", 1)[1].strip()
        elif line.startswith("SUMMARY:"):
            result["summary"] = line.split(":", 1)[1].strip()

    return result
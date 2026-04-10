"""
Deep search service — visual element search using annotated screenshot + VLM.

Pipeline:
  1. Run /detect to get elements + annotated image with numbered labels
  2. Send annotated image + question to VLM
  3. Extract bbox index from VLM response (regex)
  4. Lookup element by index → return coordinates
"""

import json
import re
import base64
from datetime import datetime

from vision_api.vlm import vlm_see
from vision_api.services import detect as detect_svc
from vision_api.config import SCREENSHOTS_DIR


async def deep_search(screenshot_b64: str, question: str, url: str) -> dict:
    """
    Visual element search: detect → annotate → VLM → extract index → return target.
    """
    print(f"\n{'='*60}")
    print(f"  DEEP SEARCH")
    print(f"{'='*60}")
    print(f"  Question: {question}")
    print(f"  URL: {url}")

    # 1. Detect + annotate
    print(f"\n  [1/4] Running detection pipeline...")
    detect_result = await detect_svc.detect(screenshot_b64, url)
    elements = detect_result.get("elements", [])
    annotated_b64 = detect_result.get("annotated_b64")
    print(f"  Detected {len(elements)} elements")
    print(f"  Annotated image: {'yes' if annotated_b64 else 'NO — using plain screenshot'}")

    # Save screenshot for debugging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = SCREENSHOTS_DIR / f"deep_search_{timestamp}.png"
    if annotated_b64:
        debug_path.write_bytes(base64.b64decode(annotated_b64))
        print(f"  Debug image saved: {debug_path}")

    # 2. VLM — natural language answer referencing bbox index
    print(f"\n  [2/4] Asking VLM...")
    vlm_prompt = (
        f"{question}\n\n"
        "The image has numbered bounding boxes on detected elements. "
        "Which bbox index number should I interact with? "
        "Answer naturally and mention the bbox index."
    )
    print(f"  Prompt: {vlm_prompt[:200]}...")

    image_for_vlm = annotated_b64 or screenshot_b64
    description = vlm_see(vlm_prompt, image_b64=image_for_vlm)
    print(f"  VLM response: {description[:300]}{'...' if len(description) > 300 else ''}")

    # 3. Extract index from VLM response
    print(f"\n  [3/4] Extracting bbox index...")
    index = _extract_index(description)
    print(f"  Extracted index: {index}")

    # 4. Lookup element
    print(f"\n  [4/4] Looking up element...")
    target = None
    if index is not None and 0 <= index < len(elements):
        target = elements[index]
        print(f"  ✓ Found: [{target.get('x')}, {target.get('y')}] "
              f"{target.get('content', '')[:60]} "
              f"(source={target.get('source')}, type={target.get('type')})")
    elif index is not None:
        print(f"  ✗ Index {index} out of range (0-{len(elements)-1})")
    else:
        print(f"  ✗ No index extracted from VLM response")

    print(f"\n{'='*60}")
    print(f"  DEEP SEARCH {'OK' if target else 'FAILED'}")
    print(f"{'='*60}\n")

    return {
        "description": description,
        "target": target,
        "index": index,
        "debug_image": str(debug_path) if annotated_b64 else None,
        "total_elements": len(elements),
    }


def _extract_index(vlm_response: str) -> int | None:
    """
    Extract bbox index from VLM response.
    Tries regex first (fast), returns None if ambiguous.
    """
    patterns = [
        r"(?:bbox\s*)?index\s*(\d+)",
        r"\bbbox\s+(\d+)\b",
        r"element\s+(?:with\s+)?(?:bbox\s+)?(?:index\s+)?(\d+)",
        r"\[(\d+)\]",
        r"#(\d+)\b",
    ]

    found = set()
    for pattern in patterns:
        matches = re.findall(pattern, vlm_response.lower())
        if matches:
            print(f"    regex '{pattern}' → matched: {matches}")
        found.update(int(m) for m in matches)

    print(f"    All indices found: {found}")

    if len(found) == 1:
        result = found.pop()
        print(f"    → Single match: {result}")
        return result

    if len(found) > 1:
        # Try direct action pattern
        direct = re.search(
            r"(?:click|tap|press|select|close|open)\s+(?:on\s+)?(?:the\s+)?(?:element\s+)?(?:with\s+)?(?:bbox\s+)?(?:index\s+)?(\d+)",
            vlm_response.lower()
        )
        if direct:
            result = int(direct.group(1))
            print(f"    → Action pattern match: {result}")
            return result

        result = min(found)
        print(f"    → Multiple found, using smallest: {result}")
        return result

    print(f"    → No index found")
    return None
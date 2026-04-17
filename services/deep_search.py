"""
Deep search service — hybrid visual element search.

Pipeline:
  1. Detect + annotate + embed
  2. VLM describes what to interact with
  3. Hybrid resolution:
     a. Extract bbox index from VLM → verify content matches VLM description
     b. Extract keyword from VLM → txtai semantic search
     c. Merge and deduplicate candidates
  4. Return ranked candidates (agent decides)
"""

import re
import base64
from datetime import datetime

from vision_api.vlm import vlm_see
from vision_api.services import detect as detect_svc
from vision_api.services import embed as embed_svc
from vision_api.config import SCREENSHOTS_DIR


async def deep_search(screenshot_b64: str, question: str, url: str) -> dict:
    """
    Hybrid element search: detect → embed → VLM → resolve (index + txtai) → candidates.
    """
    print(f"\n{'='*60}")
    print(f"  DEEP SEARCH")
    print(f"{'='*60}")
    print(f"  Question: {question}")
    print(f"  URL: {url}")

    # 1. Detect + annotate
    print(f"\n  [1/5] Running detection pipeline...")
    detect_result = await detect_svc.detect(screenshot_b64, url)
    elements = detect_result.get("elements", [])
    annotated_b64 = detect_result.get("annotated_b64")
    print(f"  Detected {len(elements)} elements")
    print(f"  Annotated image: {'yes' if annotated_b64 else 'NO'}")

    # Save debug screenshot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = SCREENSHOTS_DIR / f"deep_search_{timestamp}.png"
    if annotated_b64:
        debug_path.write_bytes(base64.b64decode(annotated_b64))
        print(f"  Debug image saved: {debug_path}")

    # 2. Embed elements into txtai
    print(f"\n  [2/5] Embedding {len(elements)} elements...")
    embed_result = embed_svc.embed(url, elements)
    print(f"  Stored: {embed_result['stored']}, Skipped: {embed_result['skipped']}")

    # 3. VLM — natural answer, but allowed to mention bbox index
    print(f"\n  [3/5] Asking VLM...")
    vlm_prompt = (
        f"{question}\n\n"
        "The image has numbered bounding boxes on detected elements. "
        "Describe the element I should interact with — mention its visible text or label. "
        "If you can identify the bbox index, mention it too."
    )
    print(f"  Prompt: {vlm_prompt[:200]}")

    image_for_vlm = annotated_b64 or screenshot_b64
    description = vlm_see(vlm_prompt, image_b64=image_for_vlm)
    print(f"  VLM response: {description[:300]}{'...' if len(description) > 300 else ''}")

    # 4. Hybrid resolution — both paths, merge candidates
    print(f"\n  [4/5] Resolving candidates...")
    candidates = []

    # --- Path A: VLM bbox index → verify against description ---
    index = _extract_index(description)
    keyword = _extract_keyword(description, question)
    print(f"  VLM index: {index}")
    print(f"  VLM keyword: '{keyword}'")

    if index is not None and 0 <= index < len(elements):
        el = elements[index]
        content = (el.get("content") or "").lower()
        kw_lower = keyword.lower() if keyword else ""

        # Verify: does the element content match what the VLM described?
        match = (
            kw_lower and kw_lower in content
            or content and content in kw_lower
            or _fuzzy_match(content, kw_lower)
        )

        if match:
            print(f"  ✓ VLM index {index} VERIFIED: '{el.get('content', '')[:40]}' matches '{keyword}'")
            candidates.append({
                **el,
                "resolution": "vlm_verified",
                "confidence": "high",
            })
        else:
            print(f"  ✗ VLM index {index} MISMATCH: '{el.get('content', '')[:40]}' vs '{keyword}'")
            candidates.append({
                **el,
                "resolution": "vlm_unverified",
                "confidence": "low",
            })

    # --- Path B: txtai semantic search ---
    search_terms = set()
    if keyword:
        search_terms.add(keyword)
    # Also search with raw question keywords
    question_kw = _extract_keyword(question, question)
    if question_kw and question_kw != keyword:
        search_terms.add(question_kw)

    for term in search_terms:
        print(f"  Searching txtai for '{term}'...")
        find_result = embed_svc.find(url, term, limit=3)
        if find_result and find_result.get("matches"):
            for m in find_result["matches"]:
                # Skip if already in candidates (same x,y)
                already = any(
                    c.get("x") == m.get("x") and c.get("y") == m.get("y")
                    for c in candidates
                )
                if not already:
                    print(f"    [{m.get('x')}, {m.get('y')}] {m.get('content', '')[:40]} "
                          f"score={m.get('score')} source={m.get('source')}")
                    candidates.append({
                        **m,
                        "resolution": "txtai",
                        "confidence": "high" if m.get("score", 0) > 0.5 else "medium",
                    })

    # 5. Rank candidates
    print(f"\n  [5/5] Ranking {len(candidates)} candidates...")
    candidates = _rank_candidates(candidates)

    for i, c in enumerate(candidates):
        print(f"  [{i}] ({c.get('confidence')}) [{c.get('x')}, {c.get('y')}] "
              f"{c.get('content', '')[:40]} via {c.get('resolution')}")

    # Best target
    target = candidates[0] if candidates else None

    print(f"\n{'='*60}")
    print(f"  DEEP SEARCH {'OK' if target else 'FAILED'} — {len(candidates)} candidate(s)")
    print(f"{'='*60}\n")

    return {
        "description": description,
        "target": target,
        "candidates": candidates[:5],  # top 5 for the agent
        "keyword": keyword,
        "debug_image": str(debug_path) if annotated_b64 else None,
        "total_elements": len(elements),
    }


# ============================================
# Resolution helpers
# ============================================

def _rank_candidates(candidates: list) -> list:
    """Rank candidates: vlm_verified > txtai high > vlm_unverified > txtai medium."""
    priority = {
        ("vlm_verified", "high"): 0,
        ("txtai", "high"): 1,
        ("vlm_unverified", "low"): 2,
        ("txtai", "medium"): 3,
    }

    def sort_key(c):
        key = (c.get("resolution", ""), c.get("confidence", ""))
        return priority.get(key, 99)

    return sorted(candidates, key=sort_key)


def _fuzzy_match(a: str, b: str, threshold: float = 0.6) -> bool:
    """Simple character overlap check for OCR typos (e.g. 'Cerra' vs 'Cerrar')."""
    if not a or not b:
        return False
    a, b = a.lower().strip(), b.lower().strip()
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < 2:
        return False
    # Check if shorter is a substring-ish of longer
    if shorter in longer or longer in shorter:
        return True
    # Character overlap ratio
    overlap = sum(1 for c in shorter if c in longer)
    ratio = overlap / max(len(shorter), 1)
    return ratio >= threshold


# ============================================
# Extraction helpers
# ============================================

def _extract_index(vlm_response: str) -> int | None:
    """Extract bbox index from VLM response via regex."""
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
            print(f"    index regex '{pattern}' → {matches}")
        found.update(int(m) for m in matches)

    if len(found) == 1:
        return found.pop()

    if len(found) > 1:
        direct = re.search(
            r"(?:click|tap|press|select|close|open)\s+(?:on\s+)?(?:the\s+)?(?:element\s+)?(?:with\s+)?(?:bbox\s+)?(?:index\s+)?(\d+)",
            vlm_response.lower()
        )
        if direct:
            return int(direct.group(1))
        return min(found)

    return None


def _extract_keyword(vlm_response: str, original_question: str) -> str:
    """Extract the key element name/text from VLM response or question."""

    # 1. Quoted text
    quoted = re.findall(r'["\']([^"\']{2,30})["\']', vlm_response)
    if quoted:
        generic = {"close", "open", "click", "button", "the", "popup", "yes", "no"}
        meaningful = [q for q in quoted if q.lower().strip() not in generic]
        if meaningful:
            print(f"    keyword quoted: {meaningful}")
            return meaningful[0]
        print(f"    keyword quoted (generic): {quoted}")
        return quoted[0]

    # 2. Label patterns
    label_patterns = [
        r'(?:labeled?|says?|called|named|titled|reading)\s+["\']?([A-Za-zÀ-ÿ\s]{2,30})["\']?',
        r'(?:the|a)\s+["\']?([A-Za-zÀ-ÿ]{3,20})["\']?\s+(?:button|link|icon|element|text)',
        r'["\']?([A-Za-zÀ-ÿ]{3,20})["\']?\s+(?:button|link|icon|element)',
    ]
    for pattern in label_patterns:
        match = re.search(pattern, vlm_response, re.IGNORECASE)
        if match:
            result = match.group(1).strip()
            print(f"    keyword label: '{result}'")
            return result

    # 3. Bold markdown
    bold = re.findall(r'\*\*([^*]{2,30})\*\*', vlm_response)
    if bold:
        short = [b for b in bold if len(b.split()) <= 4]
        if short:
            print(f"    keyword bold: {short}")
            return short[0]

    # 4. Fallback — question keywords
    stop_words = {"how", "do", "i", "can", "you", "the", "a", "an", "is", "where",
                  "what", "close", "open", "click", "find", "in", "on", "current",
                  "page", "popup", "please", "help", "me", "to"}
    words = original_question.lower().split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    if keywords:
        result = " ".join(keywords[:2])
        print(f"    keyword fallback: '{result}'")
        return result

    return original_question
"""
Embedding service — txtai index for semantic element search.

Filters elements before indexing:
  - Keeps: interactive elements with text, DOM inputs, text with 4+ chars
  - Skips: empty labels, single-word text fragments, generic class names
"""

from vision_api import models
from vision_api.utils import page_key


# Generic class names that aren't useful search targets
_GENERIC_LABELS = {"button", "link", "icon", "text", "image", "input"}


def _is_indexable(el: dict) -> bool:
    """Decide if an element is worth indexing for search."""
    content = (el.get("content") or "").strip()

    # Skip empty content
    if not content:
        return False

    # Skip generic class-name-only content (YOLO fallback with no OCR text)
    if content.lower() in _GENERIC_LABELS:
        return False

    # DOM inputs are always valuable — they have selectors and placeholders
    if el.get("source") == "dom":
        return True

    # Interactive elements with real text are always valuable
    if el.get("interactivity") and len(content) > 2:
        return True

    # Non-interactive text: only index if it's meaningful (not a fragment like "en" or "del")
    if len(content) >= 4:
        return True

    return False


def _element_to_text(el: dict) -> str:
    """Build a search-friendly text string for an element."""
    content = el.get("content", el.get("label", ""))
    el_type = el.get("type", "")

    # For interactive elements, emphasize what it is
    if el.get("interactivity"):
        return f"{content} [{el_type}]"

    # For DOM inputs, include selector info
    if el.get("source") == "dom":
        selector = el.get("selector", "")
        return f"{content} {el_type} {selector}".strip()

    # For text elements, just the content
    return content


def embed(url: str, elements: list) -> dict:
    """Filter and index elements for a URL. Overwrites existing."""
    key = page_key(url)

    # Filter to indexable elements only
    indexable = [el for el in elements if _is_indexable(el)]

    # Store filtered elements — these are what /find will return
    models.element_index[key] = indexable

    # Build search text
    docs = [(str(i), _element_to_text(el)) for i, el in enumerate(indexable)]
    models.embeddings.index(docs)

    # Stats
    sources = {}
    for el in indexable:
        src = el.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    skipped = len(elements) - len(indexable)

    return {
        "page_key": key,
        "stored": len(indexable),
        "skipped": skipped,
        "total_received": len(elements),
        "sources": sources,
    }


def find(url: str, query: str, limit: int = 5) -> dict:
    """Find top-N elements similar to query for a given URL."""
    key = page_key(url)

    if key not in models.element_index:
        return None

    elements = models.element_index[key]
    results = models.embeddings.search(query, limit)

    matches = []
    for idx, score in results:
        i = int(idx)
        if i < len(elements):
            matches.append({**elements[i], "score": round(score, 3)})

    return {"query": query, "matches": matches, "total": len(matches)}
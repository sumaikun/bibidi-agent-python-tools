"""
Request/response logging middleware.

Logs every API call with:
  - Method, path, timing
  - Request body (screenshots truncated)
  - Response status + body summary
  - Errors with full traceback

Add to app in main.py:
    from vision_api.middleware import ObserverMiddleware
    app.add_middleware(ObserverMiddleware)
"""

import json
import time
import traceback
from datetime import datetime

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


# Keys that contain large base64 data — truncate in logs
_B64_KEYS = {"screenshot", "base64_image", "annotated_image", "original_image", "som_image_base64"}

# Keys with internal data — summarize instead of dumping
_LIST_KEYS = {"elements", "matches", "results", "click_targets"}


def _clean_for_log(obj, max_str_len=200):
    """Recursively clean an object for logging — truncate base64, summarize lists."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if k in _B64_KEYS and isinstance(v, str) and len(v) > 100:
                cleaned[k] = f"<base64 {len(v)} chars>"
            elif k in _LIST_KEYS and isinstance(v, list):
                cleaned[k] = f"<list len={len(v)}>"
            else:
                cleaned[k] = _clean_for_log(v, max_str_len)
        return cleaned
    elif isinstance(obj, list):
        if len(obj) > 5:
            return [_clean_for_log(obj[0], max_str_len), f"... +{len(obj)-1} more"]
        return [_clean_for_log(item, max_str_len) for item in obj]
    elif isinstance(obj, str) and len(obj) > max_str_len:
        return obj[:max_str_len] + f"... ({len(obj)} chars)"
    return obj


def _format_json(obj):
    """Pretty-print cleaned JSON."""
    try:
        return json.dumps(obj, indent=2, default=str)
    except (TypeError, ValueError):
        return str(obj)


class ObserverMiddleware(BaseHTTPMiddleware):
    """Logs request/response details for every API call."""

    async def dispatch(self, request: Request, call_next):
        # Skip probe/health
        if request.url.path == "/probe":
            return await call_next(request)

        start = time.time()
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        method = request.method
        path = request.url.path

        # --- Log request ---
        req_body = None
        if method in ("POST", "PUT", "PATCH"):
            try:
                raw = await request.body()
                req_body = json.loads(raw)
            except Exception:
                req_body = "<unreadable>"

        cleaned_req = _clean_for_log(req_body) if req_body else None
        print(f"\n{'='*70}")
        print(f"[{ts}] → {method} {path}")
        if cleaned_req:
            print(f"[REQUEST]\n{_format_json(cleaned_req)}")

        # --- Execute ---
        try:
            response = await call_next(request)
            elapsed = round((time.time() - start) * 1000)

            # Read response body for logging
            resp_body = b""
            async for chunk in response.body_iterator:
                resp_body += chunk if isinstance(chunk, bytes) else chunk.encode()

            # Log response
            status = response.status_code
            try:
                resp_json = json.loads(resp_body)
                cleaned_resp = _clean_for_log(resp_json)
            except (json.JSONDecodeError, ValueError):
                cleaned_resp = resp_body.decode("utf-8", errors="replace")[:500]

            status_icon = "✓" if 200 <= status < 400 else "✗"
            print(f"[{ts}] ← {status_icon} {status} ({elapsed}ms)")
            print(f"[RESPONSE]\n{_format_json(cleaned_resp)}")
            print(f"{'='*70}\n")

            # Reconstruct response since we consumed the body
            return Response(
                content=resp_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        except Exception as e:
            elapsed = round((time.time() - start) * 1000)
            print(f"[{ts}] ✗ ERROR ({elapsed}ms): {e}")
            print(f"[TRACEBACK]\n{traceback.format_exc()}")
            print(f"{'='*70}\n")
            raise
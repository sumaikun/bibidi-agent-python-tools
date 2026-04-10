"""
Vision service — describe screenshot using VLM.
"""

import base64
from datetime import datetime

from vision_api.vlm import vlm_see
from vision_api.config import SCREENSHOTS_DIR


def see(screenshot_b64: str, question: str, url: str = None) -> dict:
    """Send screenshot + question to VLM, return description."""
    # Save screenshot for debugging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SCREENSHOTS_DIR / f"see_{timestamp}.png"
    path.write_bytes(base64.b64decode(screenshot_b64))

    description = vlm_see(question, image_b64=screenshot_b64)
    return {"description": description, "screenshot_path": str(path)}
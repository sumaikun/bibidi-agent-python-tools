"""
Configuration — env vars, paths, constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# --- External services ---
OMNIPARSER_URL = os.getenv("OMNIPARSER_URL", "http://localhost:8000")

# --- Model weights ---
YOLO_MODEL_PATH    = os.getenv("YOLO_MODEL_PATH",    "/mnt/c/Users/user/Desktop/AppDev/computer-vision/scraper_vision/ui_dataset/runs/detect/ui_detector/v1/weights/best.pt")
CAPTCHA_MODEL_PATH = os.getenv("CAPTCHA_MODEL_PATH", "/mnt/c/Users/user/Desktop/AppDev/ethermed/CustomNN/captcha_detector/v13/weights/best.pt")

# --- VLM (vision) ---
VLM_PROVIDER       = os.getenv("VLM_PROVIDER",       "ollama")
VLM_MODEL          = os.getenv("VLM_MODEL",          "qwen3-vl:8b-thinking")

# --- Text LLM ---
TEXT_PROVIDER       = os.getenv("TEXT_PROVIDER",      "ollama")
TEXT_MODEL          = os.getenv("TEXT_MODEL",         "qwen2.5:7b-instruct")

# --- Ollama ---
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")
OLLAMA_CLOUD_URL    = os.getenv("OLLAMA_CLOUD_URL",   "")
OLLAMA_API_KEY      = os.getenv("OLLAMA_API_KEY",     "")

# --- Anthropic ---
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY",  "")

# --- Output directories ---
SCREENSHOTS_DIR    = Path(os.getenv("SCREENSHOTS_DIR",    "screenshots"))
SAM3_OUTPUT_DIR    = Path(os.getenv("SAM3_OUTPUT_DIR",    "sam3_outputs"))
CAPTCHA_OUTPUT_DIR = Path(os.getenv("CAPTCHA_OUTPUT_DIR", "captcha_analysis"))

SCREENSHOTS_DIR.mkdir(exist_ok=True)
SAM3_OUTPUT_DIR.mkdir(exist_ok=True)
CAPTCHA_OUTPUT_DIR.mkdir(exist_ok=True)

# --- Thresholds ---
MIN_MASK_AREA_PX = 50
YOLO_CONFIDENCE  = 0.25

# --- Server ---
PORT = int(os.getenv("PORT", 5001))


INTERACTIVE_CLASSES = {
    "Button", "Button-Outlined", "Button-filled", "button",
    "TextButton", "Text Button",
    "Call to Action",
    "Link", "link",
    "EditText", "Input", "Text-Input",
    "CheckBox", "CheckedTextView",
    "RadioButton",
    "DropDown", "Spinner",
    "Toggle-Checked", "Toggle-Unchecked", "On-Off Switch", "Switch",
    "Slider", "SeekBar",
    "ImageButton",
    "Icon",
    "Multi_Tab",
    "List Item",
}
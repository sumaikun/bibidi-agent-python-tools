# Vision API

A unified Python microservice that powers [Autopilot](../autopilot/README.md)'s visual understanding. It bundles multiple computer vision pipelines — UI element detection, OCR, semantic search, object segmentation, CAPTCHA solving, and visual language reasoning — behind a single HTTP API.

---

## Pipelines

### 1. Custom YOLO — UI Element Detector

A YOLOv8 model fine-tuned on **48 UI element classes** for detecting interactive components in web pages.

| Detail | Value |
|---|---|
| Classes | Text, Icon, Image, Link, Button, Input, Dropdown, Checkbox, Radio, Toggle, etc. (48 total) |
| OCR engine | EasyOCR (extracts text content from detected elements) |
| Confidence threshold | `0.25` |
| Output | Bounding boxes + class label + OCR text per element |

**Dataset:** [UI Element Detect](https://universe.roboflow.com/uied/ui-element-detect) on Roboflow Universe.

### 2. OmniParser (Florence2 + YOLO)

A general-purpose UI screen parser based on Microsoft's [OmniParser](https://github.com/microsoft/OmniParser). Combines Florence2 for captioning/grounding with YOLO for element detection.

Best suited for pages where the custom YOLO model hasn't been fine-tuned.

### 3. txtai — Semantic Search

Indexes detected UI elements and enables natural-language queries over them (e.g., *"the login button"*, *"email input field"*). Useful for the Autopilot agent to locate elements by intent rather than exact label.

### 4. SAM 3 — Segment Anything Model 3

Meta's open-vocabulary segmentation model. Used in Autopilot for **non-UI physical objects** — CAPTCHAs containing real-world images, product photos, document scans, etc.

SAM 3 accepts text prompts or image exemplars and returns precise segmentation masks for all matching instances. See the [SAM 3 Access Requirements](#sam-3-access-requirements) section below for setup details.

### 5. CAPTCHA Solver

A dedicated YOLO model trained to detect and classify CAPTCHA regions, combined with SAM 3 segmentation for image-based challenges.

**Dataset:** [CAPTCHA Area](https://app.roboflow.com/jesus-projects/captcha-area) on Roboflow (private project).

### 6. VLM — Vision-Language Model

A multimodal model for complex visual reasoning tasks that go beyond detection — describing page state, interpreting ambiguous UI, answering questions about screenshots.

---

## Architecture

```
                        ┌──────────────────────┐
                        │    Vision API        │
                        │   Flask / port 5001  │
                        └──────────┬───────────┘
                                   │
        ┌──────────┬───────────┬───┴────┬──────────┬─────────┐
        ▼          ▼           ▼        ▼          ▼         ▼
   ┌────────┐ ┌─────────┐ ┌──────┐ ┌──────┐ ┌─────────┐ ┌─────┐
   │  YOLO  │ │OmniParser│ │ txtai│ │ SAM3 │ │ CAPTCHA │ │ VLM │
   │48 class│ │Florence2 │ │search│ │ Meta │ │  YOLO   │ │     │
   └────────┘ └──────────┘ └──────┘ └──────┘ └─────────┘ └─────┘
       │                                │
    EasyOCR                      HuggingFace
  (text extraction)           (gated model access)
```

All pipelines share a single GPU and are served behind one Flask application. The Autopilot Elixir agent calls this API over HTTP.

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA with 8 GB VRAM | NVIDIA RTX 4080 SUPER (16 GB) or higher |
| RAM | 16 GB | 32 GB+ |
| Storage | 25 GB free | 50 GB+ free |
| CUDA | 12.x | 12.6 |

### Software

| Dependency | Version |
|---|---|
| Python | >= 3.10 (3.12 recommended) |
| PyTorch | >= 2.7 with CUDA support |
| Ultralytics | latest |
| EasyOCR | latest |
| Flask | latest |
| txtai | latest |
| huggingface_hub | latest |

---

## SAM 3 Access Requirements

SAM 3 model weights are **gated** — you need to request access before downloading them.

### Step 1 — Accept the license

Go to the SAM 3 Hugging Face repository and accept Meta's SAM License:

> **https://huggingface.co/facebook/sam3**

Click **"Agree and access repository"**. Approval is typically immediate.

### Step 2 — Create a Hugging Face token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **Read** access
3. Save the token

### Step 3 — Authenticate locally

```bash
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted
```

### Step 4 — Install SAM 3

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

The first time you run inference, model weights (~3.4 GB) will be downloaded automatically.

### Step 5 — Verify

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

from sam3.model_builder import build_sam3
print("SAM 3 loaded successfully")
```

### SAM 3 notes

- SAM 3 is released under [Meta's SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE) — review it for commercial use terms.
- Typical VRAM usage during inference is ~4 GB, but 16 GB+ is recommended when running alongside the other pipelines.
- CPU-only execution is possible but significantly slower.
- For Ultralytics integration: `pip install -U ultralytics` (SAM 3 available since v8.3.237).

---

## Datasets

### UI Element Detection

> **https://universe.roboflow.com/uied/ui-element-detect**

Public dataset on Roboflow Universe. Used to train the custom YOLO model on 48 UI element classes. You can download it in YOLO format directly from Roboflow:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("uied").project("ui-element-detect")
dataset = project.version(1).download("yolov8")
```

### CAPTCHA Area Detection

> **https://app.roboflow.com/jesus-projects/captcha-area**

Private dataset for training the CAPTCHA detection model. Contact the project maintainer for access, or request an invitation via Roboflow.

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("jesus-projects").project("captcha-area")
dataset = project.version(1).download("yolov8")
```

---

## Getting Started

### 1. Set up the environment

```bash
# Create a virtual environment
conda create -n vision-api python=3.12
conda activate vision-api

# Install PyTorch with CUDA
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install dependencies
pip install -r requirements.txt

# Authenticate with Hugging Face (required for SAM 3)
huggingface-cli login
```

### 2. Download model weights

```bash
# YOLO UI Detector — place weights in the expected path
# Default: ./weights/ui_detector/best.pt

# SAM 3 — auto-downloads on first use after HF authentication

# OmniParser — follow the OmniParser repo instructions
```

### 3. Configure

Create a `.env` file or export environment variables:

```env
VISION_PORT=5001
YOLO_WEIGHTS_PATH=./weights/ui_detector/best.pt
YOLO_CONF_THRESHOLD=0.25
CAPTCHA_WEIGHTS_PATH=./weights/captcha/best.pt
```

### 4. Run the server

```bash
python server.py
# API available at http://localhost:5001
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/detect` | Run YOLO UI detection + OCR on a screenshot |
| `POST` | `/omniparser` | Run OmniParser (Florence2 + YOLO) |
| `POST` | `/search` | Semantic search over detected elements (txtai) |
| `POST` | `/segment` | Run SAM 3 segmentation with a text or visual prompt |
| `POST` | `/captcha` | Detect and solve CAPTCHA challenges |
| `POST` | `/vlm` | Visual question answering on a screenshot |
| `GET`  | `/health` | Health check |

All endpoints accept images as base64-encoded strings or multipart file uploads.

---

## Training Your Own Models

### UI Detector

```bash
yolo detect train \
  data=./datasets/ui-element-detect/data.yaml \
  model=yolov8m.pt \
  epochs=100 \
  imgsz=640 \
  conf=0.25
```

### CAPTCHA Detector

```bash
yolo detect train \
  data=./datasets/captcha-area/data.yaml \
  model=yolov8m.pt \
  epochs=50 \
  imgsz=640
```

---

## Project Structure

```
vision-api/
├── server.py               # Flask app entry point
├── requirements.txt
├── .env
├── pipelines/
│   ├── yolo_detector.py     # Custom YOLO + EasyOCR
│   ├── omniparser.py        # Florence2 + YOLO
│   ├── txtai_search.py      # Semantic element search
│   ├── sam3_segmenter.py    # SAM 3 integration
│   ├── captcha_solver.py    # CAPTCHA detection + solving
│   └── vlm.py               # Vision-Language Model
├── weights/
│   ├── ui_detector/
│   │   └── best.pt
│   └── captcha/
│       └── best.pt
└── datasets/                # Downloaded Roboflow datasets
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `CUDA out of memory` | Reduce batch size or run fewer pipelines concurrently. SAM 3 alone uses ~4 GB. |
| SAM 3 download fails | Ensure you accepted the license at `huggingface.co/facebook/sam3` and ran `huggingface-cli login`. |
| EasyOCR slow on first run | It downloads language models on first use (~100 MB). Subsequent runs are fast. |
| Low detection accuracy | Check `YOLO_CONF_THRESHOLD` — lower values catch more elements but increase false positives. |

---

## License

This project is licensed under the [MIT License](LICENSE).

SAM 3 model weights are subject to [Meta's SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE).

---

## References

- [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719) — Meta AI, 2025
- [OmniParser](https://github.com/microsoft/OmniParser) — Microsoft
- [UI Element Detect Dataset](https://universe.roboflow.com/uied/ui-element-detect) — Roboflow Universe
- [CAPTCHA Area Dataset](https://app.roboflow.com/jesus-projects/captcha-area) — Roboflow
- [Ultralytics YOLO](https://docs.ultralytics.com/) — YOLOv8 framework
- [SAM 3 on Hugging Face](https://huggingface.co/facebook/sam3) — Gated model access
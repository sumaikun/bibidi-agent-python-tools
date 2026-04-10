"""
SAM3 segmentation service.

SAM3UIFinder uses the video predictor in single-frame mode:
feed one frame as a "video", prompt by text, get binary masks.
"""

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sam3.model_builder import build_sam3_video_predictor

from vision_api.config import MIN_MASK_AREA_PX, SAM3_OUTPUT_DIR


# ============================================
# MASK UTILITIES
# ============================================

def tensor_to_numpy_mask(mask_tensor) -> np.ndarray:
    """Convert SAM3 mask tensor to clean 2D numpy uint8 array."""
    if isinstance(mask_tensor, torch.Tensor):
        mask = mask_tensor.cpu().numpy()
    else:
        mask = mask_tensor
    mask = (mask > 0).astype(np.uint8)
    if mask.ndim > 2:
        mask = mask[0]
    return mask


def mask_to_bbox(mask: np.ndarray, img_w: int, img_h: int) -> dict | None:
    """Binary mask → bbox dict with pixel + normalized coordinates."""
    rows, cols = np.where(mask > 0)
    if len(rows) < MIN_MASK_AREA_PX:
        return None
    y1, y2 = int(rows.min()), int(rows.max())
    x1, x2 = int(cols.min()), int(cols.max())
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w // 2
    cy = y1 + h // 2
    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "center_x": cx, "center_y": cy,
        "width": w, "height": h,
        "cx_norm": round(cx / img_w, 6),
        "cy_norm": round(cy / img_h, 6),
        "w_norm": round(w / img_w, 6),
        "h_norm": round(h / img_h, 6),
    }


def mask_to_polygon(mask: np.ndarray) -> list | None:
    """Binary mask → simplified polygon contour points (pixel coords)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_MASK_AREA_PX:
        return None
    approx = cv2.approxPolyDP(largest, 2.0, True)
    if len(approx) < 3:
        return None
    return approx.reshape(-1, 2).tolist()


# ============================================
# SAM3 UI FINDER
# ============================================

class SAM3UIFinder:
    """
    Find UI elements by text prompt using SAM3's video predictor
    in single-frame mode. Feeds one frame as a 'video'.
    """

    def __init__(self):
        print("Loading SAM3 video predictor...")
        self.predictor = build_sam3_video_predictor(
            gpus_to_use=[torch.cuda.current_device()]
        )
        print("SAM3 ready.")

    def find_multiple(self, image_path: str, prompts: list[str]) -> list[dict]:
        """
        Find ALL instances of each prompt in the image.
        Returns a flat list of detections with masks.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        H, W = img.shape[:2]

        temp_dir = Path(tempfile.mkdtemp(prefix="sam3_ui_"))
        frame_path = temp_dir / "frame_000000.jpg"
        cv2.imwrite(str(frame_path), img)

        all_results = []
        try:
            for prompt in prompts:
                prompt = prompt.strip()
                instances = self._run_single_prompt(temp_dir, prompt, W, H)
                all_results.extend(instances)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return all_results

    def _run_single_prompt(self, frames_dir: Path, prompt: str, W: int, H: int) -> list[dict]:
        """
        Run SAM3 session for a single text prompt on a single frame.
        Splits merged masks into individual instances via connected components.
        """
        instances = []

        try:
            response = self.predictor.handle_request(dict(
                type="start_session", resource_path=str(frames_dir),
            ))
            session_id = response["session_id"]

            self.predictor.handle_request(dict(
                type="add_prompt", session_id=session_id,
                frame_index=0, text=prompt, obj_id=0,
            ))

            raw_masks = []
            for resp in self.predictor.handle_stream_request(dict(
                type="propagate_in_video", session_id=session_id,
            )):
                outputs = resp.get("outputs", {})
                if "out_binary_masks" in outputs:
                    for m_tensor in outputs["out_binary_masks"]:
                        raw_masks.append(tensor_to_numpy_mask(m_tensor))

            print(f"  SAM3 returned {len(raw_masks)} raw mask(s) for '{prompt}'")

            # Split merged masks into individual instances
            all_individual_masks = []
            for mask in raw_masks:
                area = int(np.sum(mask > 0))
                if area < MIN_MASK_AREA_PX:
                    continue
                num_labels, labels = cv2.connectedComponents(mask)
                print(f"    Mask area={area}px, connected_components={num_labels - 1}")
                if num_labels <= 2:
                    all_individual_masks.append(mask)
                else:
                    for label_id in range(1, num_labels):
                        component_mask = (labels == label_id).astype(np.uint8)
                        comp_area = int(np.sum(component_mask > 0))
                        if comp_area >= MIN_MASK_AREA_PX:
                            all_individual_masks.append(component_mask)
                            print(f"      Component {label_id}: area={comp_area}px")

            for idx, mask in enumerate(all_individual_masks):
                bbox = mask_to_bbox(mask, W, H)
                if bbox is not None:
                    instances.append({
                        "found": True, "prompt": prompt, "instance_id": idx,
                        "bbox": bbox, "polygon": mask_to_polygon(mask),
                        "mask": mask, "mask_area_px": int(np.sum(mask > 0)),
                    })

            self.predictor.handle_request(dict(
                type="close_session", session_id=session_id,
            ))

        except Exception as e:
            print(f"  ERROR processing '{prompt}': {e}")
        finally:
            torch.cuda.empty_cache()

        if not instances:
            instances.append({
                "found": False, "prompt": prompt, "instance_id": 0,
                "bbox": None, "polygon": None, "mask": None, "mask_area_px": 0,
            })

        print(f"  → {len([i for i in instances if i['found']])} instance(s) of '{prompt}'")
        return instances

    def shutdown(self):
        self.predictor.shutdown()


# ============================================
# VISUALIZATION
# ============================================

def draw_sam3_results(image_path: str, results: list[dict], output_path: str = None) -> str:
    """Draw bounding boxes and mask overlays for SAM3 results."""
    img = Image.open(image_path)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    colors = ["#00FF00", "#FF4444", "#4488FF", "#FFAA00", "#FF44FF", "#44FFFF"]
    img_np = np.array(img)

    for i, res in enumerate(results):
        if not res["found"]:
            print(f"  ✗ '{res['prompt']}' — NOT FOUND (mask_area={res['mask_area_px']}px)")
            continue

        color = colors[i % len(colors)]
        b = res["bbox"]

        # Semi-transparent mask overlay
        if res.get("mask") is not None:
            mask = res["mask"]
            r_c = int(color[1:3], 16)
            g_c = int(color[3:5], 16)
            b_c = int(color[5:7], 16)
            overlay = img_np.copy()
            overlay[mask > 0] = (
                overlay[mask > 0] * 0.5 + np.array([r_c, g_c, b_c]) * 0.5
            ).astype(np.uint8)
            img_np = overlay

        pil_img = Image.fromarray(img_np)
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=color, width=3)

        if res.get("polygon"):
            poly_flat = [(p[0], p[1]) for p in res["polygon"]]
            if len(poly_flat) >= 3:
                draw.polygon(poly_flat, outline=color)

        inst = res.get("instance_id", 0)
        label = f"{res['prompt']}#{inst} ({res['mask_area_px']}px)"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(b["y1"] - text_h - 6, 0)
        draw.rectangle(
            [b["x1"], label_y, b["x1"] + text_w + 8, label_y + text_h + 6], fill=color,
        )
        draw.text((b["x1"] + 4, label_y + 3), label, fill="white", font=font)
        img_np = np.array(pil_img)

        print(f"  ✓ '{res['prompt']}' — center=({b['center_x']}, {b['center_y']}) "
              f"size={b['width']}x{b['height']} area={res['mask_area_px']}px")

    if output_path is None:
        p = Path(image_path)
        output_path = str(SAM3_OUTPUT_DIR / f"{p.stem}_sam3_found{p.suffix}")

    Image.fromarray(img_np).save(output_path)
    print(f"\nAnnotated image saved to: {output_path}")
    return output_path

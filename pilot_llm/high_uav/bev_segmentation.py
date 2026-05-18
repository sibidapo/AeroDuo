"""
bev_segmentation.py — Run Grounded-SAM2 on a single BEV image.

Usage:
    python bev_segmentation.py <bev_image_path> <instruction_json_path> [--output-dir <dir>]

Saves:
    <output_dir>/<stem>_seg.json   — structured segmentation metadata
    <output_dir>/<stem>_masks.npz  — compressed numpy masks (keyed by object index)
    <output_dir>/<stem>_vis.png    — visualisation overlay

The JSON output is designed to be consumed by the downstream DINOv2 feature
extraction step without re-running detection.
"""

import os
import sys
import json
import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

_RUNTIME_CACHE_ROOT = Path(os.environ.get("AERODUO_CACHE_DIR", Path.cwd() / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_RUNTIME_CACHE_ROOT / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_RUNTIME_CACHE_ROOT))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import cv2
import torch

# ─── Path setup: make Grounded-SAM-2 importable ───────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent   # aeroduo/pilot_llm/
_GSAM2_DIR = _REPO_ROOT / "Grounded-SAM-2"
if str(_GSAM2_DIR) not in sys.path:
    sys.path.insert(0, str(_GSAM2_DIR))

from torchvision.ops import box_convert, nms as torchvision_nms
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

# Local module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from noun_extractor import parse_instruction, build_prompt_list

# ─── Default model paths (relative to Grounded-SAM-2 repo root) ───────────────
_SAM2_CHECKPOINT    = str(_GSAM2_DIR / "checkpoints" / "sam2.1_hiera_large.pt")
_SAM2_CONFIG        = "configs/sam2.1/sam2.1_hiera_l.yaml"
_GDINO_CONFIG       = str(_GSAM2_DIR / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
_GDINO_CHECKPOINT   = str(_GSAM2_DIR / "gdino_checkpoints" / "groundingdino_swint_ogc.pth")

# ─── Thresholds ───────────────────────────────────────────────────────────────
BOX_THRESHOLD  = 0.40   # GroundingDINO box confidence
TEXT_THRESHOLD = 0.40   # GroundingDINO text confidence
NMS_IOU_THRESH = 0.50   # IoU threshold for suppressing duplicate boxes

# ─── Visualisation palette ────────────────────────────────────────────────────
# 12 distinct BGR colours (OpenCV); index 0 reserved for goal object (bright)
_COLOURS = [
    (0,   255, 50),    # goal: bright green
    (255, 100,  20),   # contextual colours below
    (0,   180, 255),
    (200,  60, 255),
    (255, 220,   0),
    (0,   255, 200),
    (255,  80, 160),
    (80,  180, 255),
    (255, 160,  80),
    (60,  255, 120),
    (200, 120, 255),
    (255, 255,  80),
]


# ─── Model loading (cached at module level) ───────────────────────────────────

_sam2_predictor: Optional[SAM2ImagePredictor] = None
_grounding_model = None

def load_models(device: str = "auto"):
    """Load SAM2 and GroundingDINO models (once; cached)."""
    global _sam2_predictor, _grounding_model

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _sam2_predictor is None:
        sam2 = build_sam2(_SAM2_CONFIG, _SAM2_CHECKPOINT, device=device)
        _sam2_predictor = SAM2ImagePredictor(sam2)

    if _grounding_model is None:
        _grounding_model = load_model(
            model_config_path=_GDINO_CONFIG,
            model_checkpoint_path=_GDINO_CHECKPOINT,
            device=device,
        )

    return _sam2_predictor, _grounding_model, device


# ─── NMS helper ───────────────────────────────────────────────────────────────

def _apply_nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float):
    """Return indices of detections surviving NMS."""
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=int)
    t_boxes  = torch.from_numpy(boxes_xyxy.astype(np.float32))
    t_scores = torch.from_numpy(scores.astype(np.float32))
    keep = torchvision_nms(t_boxes, t_scores, iou_thresh)
    return keep.numpy()


def _normalise_category(category: Optional[str]) -> Optional[str]:
    """Normalise prompt/category names for stable comparison and storage."""
    if category is None:
        return None
    cleaned = category.strip().lower().rstrip(".")
    return cleaned or None


def _detect_per_prompt(
    image_tensor: torch.Tensor,
    image_width: int,
    image_height: int,
    prompt_list: list[str],
    goal_object: Optional[str],
    grounding_model,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> list[dict]:
    """
    Run GroundingDINO once per prompt so categories stay tied to the extracted nouns.
    """
    candidates = []
    goal_norm = _normalise_category(goal_object)
    prompt_count = max(len(prompt_list), 1)

    for prompt_index, prompt in enumerate(prompt_list):
        prompt_norm = _normalise_category(prompt)
        if not prompt_norm:
            continue

        # ms_deform_attn CUDA kernel only supports fp32; disable autocast for GDINO
        device_type = device.split(":")[0] if isinstance(device, str) else device.type
        with torch.autocast(device_type=device_type, enabled=False):
            boxes_cx, confidences, labels = predict(
                model=grounding_model,
                image=image_tensor.float(),
                caption=f"{prompt_norm}.",
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device,
            )
        if len(boxes_cx) == 0:
            continue

        boxes_px = box_convert(
            boxes_cx * torch.tensor([image_width, image_height, image_width, image_height], dtype=torch.float32),
            in_fmt="cxcywh",
            out_fmt="xyxy",
        ).numpy()
        conf_np = confidences.numpy()

        is_goal_prompt = goal_norm is not None and prompt_norm == goal_norm
        priority_bonus = 1.0 if is_goal_prompt else 0.01 * (prompt_count - prompt_index)

        for bbox, confidence, raw_label in zip(boxes_px.tolist(), conf_np.tolist(), labels):
            candidates.append({
                "prompt_index": prompt_index,
                "prompt": prompt_norm,
                "category": prompt_norm,
                "raw_grounding_label": raw_label.strip(),
                "bbox_xyxy": bbox,
                "confidence": float(confidence),
                "nms_score": float(confidence) + priority_bonus,
                "is_goal": is_goal_prompt,
            })

    return candidates


# ─── Core segmentation function ───────────────────────────────────────────────

def segment_bev_image(
    image_path: str | Path,
    goal_object: Optional[str],
    contextual_nouns: list[str],
    sam2_predictor: SAM2ImagePredictor,
    grounding_model,
    device: str,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
    nms_iou: float = NMS_IOU_THRESH,
    multimask_output: bool = False,
) -> dict:
    """
    Run Grounded-SAM2 on one BEV image.

    Returns a structured dict ready to be saved as JSON + .npz:
    {
        "image_path":    <str>,
        "image_width":   <int>,
        "image_height":  <int>,
        "goal_category": <str>,
        "goal_detected": <bool>,
        "num_detections":<int>,
        "detections": [
            {
                "id":         <int>,        # 0-indexed
                "category":   <str>,
                "bbox_xyxy":  [x1,y1,x2,y2],  # pixel coordinates
                "confidence": <float>,
                "is_goal":    <bool>,
                "mask_key":   <str>,        # key into the .npz file
                # Fields added by DINOv2 stage (placeholder):
                "dinov2_embedding": null,
            },
            ...
        ],
        "masks_npz_path": <str>,            # relative path to the .npz file
    }
    """
    image_path = Path(image_path)
    image_source, image_tensor = load_image(str(image_path))
    h, w = image_source.shape[:2]

    goal_object = _normalise_category(goal_object)
    contextual_nouns = [_normalise_category(noun) for noun in contextual_nouns]
    contextual_nouns = [noun for noun in contextual_nouns if noun]
    prompt_list = build_prompt_list(goal_object, contextual_nouns)
    if not prompt_list:
        return _empty_result(image_path, goal_object, w, h)

    candidates = _detect_per_prompt(
        image_tensor=image_tensor,
        image_width=w,
        image_height=h,
        prompt_list=prompt_list,
        goal_object=goal_object,
        grounding_model=grounding_model,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if not candidates:
        return _empty_result(image_path, goal_object, w, h, prompt_list=prompt_list)

    boxes_px = np.asarray([cand["bbox_xyxy"] for cand in candidates], dtype=np.float32)
    nms_scores = np.asarray([cand["nms_score"] for cand in candidates], dtype=np.float32)
    keep = _apply_nms(boxes_px, nms_scores, nms_iou)
    kept_candidates = [candidates[i] for i in keep.tolist()]

    kept_candidates.sort(key=lambda c: (-int(c["is_goal"]), c["prompt_index"], -c["confidence"]))
    boxes_px = np.asarray([cand["bbox_xyxy"] for cand in kept_candidates], dtype=np.float32)

    if len(kept_candidates) == 0:
        return _empty_result(image_path, goal_object, w, h)

    # ── SAM2 mask prediction ───────────────────────────────────────────────────
    # Enable bfloat16 if GPU supports it (Ampere+); otherwise leave default
    if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device == "cuda":
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        ctx = nullcontext()

    sam2_predictor.set_image(image_source)

    with ctx:
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_px,
            multimask_output=multimask_output,
        )

    # Shape: (N, 1, H, W) if multimask_output=False, or (N, 3, H, W)
    if masks.ndim == 4:
        if multimask_output:
            best = np.argmax(scores, axis=1)
            masks = masks[np.arange(masks.shape[0]), best]
        else:
            masks = masks[:, 0]   # (N, H, W)

    # ── Assemble output ────────────────────────────────────────────────────────
    goal_detected = False
    detections = []

    for idx, (candidate, bbox, mask) in enumerate(zip(kept_candidates, boxes_px.tolist(), masks)):
        is_goal = bool(goal_object and candidate["category"] == goal_object)
        if is_goal:
            goal_detected = True

        detections.append({
            "id":               idx,
            "category":         candidate["category"],
            "bbox_xyxy":        [round(v, 2) for v in bbox],
            "confidence":       round(float(candidate["confidence"]), 4),
            "is_goal":          is_goal,
            "source_prompt":    candidate["prompt"],
            "prompt_index":     candidate["prompt_index"],
            "grounding_phrase": candidate["raw_grounding_label"],
            "mask_key":         f"mask_{idx:04d}",
            "dinov2_embedding": None,   # filled by Stage 1a
        })

    result = {
        "image_path":    str(image_path),
        "image_width":   w,
        "image_height":  h,
        "goal_category": goal_object,
        "goal_detected": goal_detected,
        "prompt_list":   prompt_list,
        "num_detections": len(detections),
        "detections":    detections,
        "masks_npz_path": None,   # filled after saving
    }
    # Attach raw masks for saving (not serialised to JSON)
    result["_masks_array"] = masks

    return result


def _empty_result(image_path, goal_object, w, h, prompt_list: Optional[list[str]] = None):
    return {
        "image_path":     str(image_path),
        "image_width":    w,
        "image_height":   h,
        "goal_category":  goal_object,
        "goal_detected":  False,
        "prompt_list":    prompt_list or [],
        "num_detections": 0,
        "detections":     [],
        "masks_npz_path": None,
        "_masks_array":   np.zeros((0, h, w), dtype=bool),
    }


# ─── Saving ───────────────────────────────────────────────────────────────────

def save_segmentation(result: dict, output_dir: Path) -> tuple[Path, Path]:
    """
    Persist segmentation to disk.

    Returns (json_path, npz_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result["image_path"]).stem

    # Save masks as compressed numpy array
    masks = result.pop("_masks_array", np.zeros((0,), dtype=bool))
    npz_path = output_dir / f"{stem}_masks.npz"
    np.savez_compressed(str(npz_path), **{d["mask_key"]: masks[i] for i, d in enumerate(result["detections"])})
    result["masks_npz_path"] = npz_path.name

    # Save JSON (all fields are JSON-serialisable at this point)
    json_path = output_dir / f"{stem}_seg.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    return json_path, npz_path


def load_segmentation(json_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """
    Load a previously saved segmentation.

    Returns (result_dict, {mask_key: np.ndarray}).
    """
    with open(json_path) as f:
        result = json.load(f)
    npz_path = result.get("masks_npz_path")
    masks = {}
    if npz_path:
        npz_path = Path(npz_path)
        if not npz_path.is_absolute():
            npz_path = json_path.parent / npz_path
    if npz_path and Path(npz_path).exists():
        data = np.load(npz_path, allow_pickle=False)
        masks = {k: data[k] for k in data.files}
    return result, masks


# ─── Visualisation ────────────────────────────────────────────────────────────

def visualise_segmentation(result: dict, masks: dict[str, np.ndarray]) -> np.ndarray:
    """
    Overlay masks, bounding boxes, and labels on the original BEV image.

    Goal object: brighter colour + thicker box border.
    Returns an annotated BGR image (numpy array).
    """
    img = cv2.imread(result["image_path"])
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {result['image_path']}")
    overlay = img.copy()

    # Sort: draw contextual first, goal on top
    detections = sorted(result["detections"], key=lambda d: d["is_goal"])

    colour_idx = 1   # 0 is reserved for goal
    for det in detections:
        mask_arr = masks.get(det["mask_key"])
        if mask_arr is None:
            continue

        if det["is_goal"]:
            colour = _COLOURS[0]
            border_thickness = 3
            alpha = 0.55
        else:
            colour = _COLOURS[colour_idx % len(_COLOURS)]
            colour_idx += 1
            border_thickness = 2
            alpha = 0.40

        # Colour mask
        mask_bool = mask_arr.astype(bool)
        coloured   = np.zeros_like(overlay)
        coloured[mask_bool] = colour
        overlay    = cv2.addWeighted(overlay, 1.0, coloured, alpha, 0)

        # Bounding box
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, border_thickness)

        # Label
        label = f"{'[GOAL] ' if det['is_goal'] else ''}{det['category']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1 - 5, th + 4)
        cv2.rectangle(overlay, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), colour, -1)
        cv2.putText(overlay, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return overlay


def save_visualisation(vis_img: np.ndarray, output_dir: Path, stem: str) -> Path:
    vis_path = output_dir / f"{stem}_vis.png"
    cv2.imwrite(str(vis_path), vis_img)
    return vis_path


# ─── Summary printer ──────────────────────────────────────────────────────────

def print_summary(result: dict):
    print(f"\n{'─'*60}")
    print(f"Image          : {result['image_path']}")
    print(f"Goal category  : {result['goal_category']}  "
          f"({'DETECTED' if result['goal_detected'] else 'NOT FOUND'})")
    print(f"Total masks    : {result['num_detections']}")
    if result["detections"]:
        print("Detections:")
        for d in result["detections"]:
            tag = " [GOAL]" if d["is_goal"] else ""
            print(f"  [{d['id']:2d}] {d['category']:<25s}  conf={d['confidence']:.3f}{tag}")
    print(f"{'─'*60}")


# ─── Pipeline: single image ───────────────────────────────────────────────────

def run_pipeline(
    image_path: str | Path,
    instruction_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    verbose: bool = True,
) -> dict:
    """
    Full pipeline for one BEV image:
      1. Parse instruction → goal + contextual nouns
      2. Run Grounded-SAM2
      3. Save structured output + masks
      4. Save visualisation
      5. Return result dict

    The returned dict has masks_npz_path set (no _masks_array key).
    """
    image_path   = Path(image_path)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse instruction
    goal, contextual, desc = parse_instruction(instruction_path)
    if verbose:
        print(f"Goal: {goal!r}  |  Contextual: {contextual}")

    # Load models (cached)
    sam2_pred, gdino_model, dev = load_models(device)

    # Run segmentation
    result = segment_bev_image(
        image_path=image_path,
        goal_object=goal,
        contextual_nouns=contextual,
        sam2_predictor=sam2_pred,
        grounding_model=gdino_model,
        device=dev,
    )
    result["instruction_path"] = str(instruction_path)
    result["description_text"] = desc
    result["contextual_categories"] = contextual

    # Save masks + JSON
    masks_raw = result.get("_masks_array", np.zeros((0,), dtype=bool))
    masks_dict = {d["mask_key"]: masks_raw[i] for i, d in enumerate(result["detections"])}

    json_path, npz_path = save_segmentation(result, output_dir)

    # Visualise
    stem    = image_path.stem
    vis_img = visualise_segmentation(result, masks_dict)
    vis_path = save_visualisation(vis_img, output_dir, stem)

    if verbose:
        print_summary(result)
        print(f"Saved JSON : {json_path}")
        print(f"Saved NPZ  : {npz_path}")
        print(f"Saved VIS  : {vis_path}")

    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Run Grounded-SAM2 on one BEV image.")
    p.add_argument("image",       help="Path to BEV image (.png/.jpg)")
    p.add_argument("instruction", help="Path to object_description_with_help.json")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: same folder as image)")
    p.add_argument("--device", default="auto",
                   help="'cuda', 'cpu', or 'auto' (default: auto)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out_dir = args.output_dir or str(Path(args.image).parent / "seg_output")
    run_pipeline(
        image_path=args.image,
        instruction_path=args.instruction,
        output_dir=out_dir,
        device=args.device,
    )

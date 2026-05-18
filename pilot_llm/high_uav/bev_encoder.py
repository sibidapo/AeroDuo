"""
bev_encoder.py — Frozen GroundingDINO + SAM2 encoder for a window of T BEV frames.

Owns both frozen models internally so AeroDuoPolicy.forward() receives only the
raw dataset outputs (BEV PIL images + instruction) and nothing SAM2/GDINO-specific.

Batchification strategy
-----------------------
GroundingDINO has no multi-image batch API — detection runs per-frame sequentially.

SAM2 is batched within the T-frame window:
    set_image_batch([img_0, ..., img_{T-1}])
        → Hiera backbone runs ONCE for all T frames
        → _features["image_embed"]    [T, 256, 64, 64]
        → _features["high_res_feats"] [T, C_i, H_i, W_i] per scale

Mask decoding (cheap) runs per-frame by temporarily slicing the batch features
into single-image views that the existing SAM2 predict() API understands.

For gradient accumulation across N episodes, each episode is processed
independently with a fresh BEVEncoder.__call__; SAM2 state does not persist
between calls (reset_predictor is called inside set_image_batch).

Returned tensors
----------------
image_embeds  : Tensor [T, 256, 64, 64]  on ``device``, detached — no grad
masks_arrays  : List[np.ndarray [N, H, W] bool],  length T
detections_list: List[List[dict]],                length T
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image

# ── Path setup: make Grounded-SAM-2 importable ───────────────────────────────
_REPO_ROOT  = Path(__file__).resolve().parent.parent          # aeroduo/pilot_llm/
_GSAM2_DIR  = _REPO_ROOT / "Grounded-SAM-2"
if str(_GSAM2_DIR) not in sys.path:
    sys.path.insert(0, str(_GSAM2_DIR))

# Import shared utilities from the existing segmentation module rather than
# duplicating them.  The module-level side-effects (os.environ setdefault,
# sys.path insert) are harmless when imported repeatedly.
try:
    from .bev_segmentation import (
        _apply_nms,
        _normalise_category,
        BOX_THRESHOLD,
        TEXT_THRESHOLD,
        NMS_IOU_THRESH,
        _detect_per_prompt,
    )
    from .noun_extractor import parse_instruction, build_prompt_list
except ImportError:
    from bev_segmentation import (  # direct script execution
        _apply_nms,
        _normalise_category,
        BOX_THRESHOLD,
        TEXT_THRESHOLD,
        NMS_IOU_THRESH,
        _detect_per_prompt,
    )
    from noun_extractor import parse_instruction, build_prompt_list

# GroundingDINO predict() — imported here so bev_encoder is self-contained
from grounding_dino.groundingdino.util.inference import predict as gdino_predict  # noqa: E402
from torchvision.ops import box_convert                                           # noqa: E402


# ── PIL → GroundingDINO input tensor ─────────────────────────────────────────
# Replicates what groundingdino.util.inference.load_image does, but accepts a
# PIL image directly so we avoid round-tripping through disk.
#
# GroundingDINO transform: RandomResize([800], max_size=1333) → ToTensor → Normalize.
# When the size list contains a single value the resize is deterministic:
#   scale = 800 / min(H, W),  clamped so max(H, W) * scale ≤ 1333.
_GDINO_MEAN = [0.485, 0.456, 0.406]
_GDINO_STD  = [0.229, 0.224, 0.225]


def _pil_to_gdino(pil_img: Image.Image) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Convert a PIL RGB image to GroundingDINO's expected input format.

    Returns
    -------
    img_np   : np.ndarray [H, W, 3] uint8 — original resolution, used by SAM2
    img_t    : Tensor     [3, H', W'] float32 — resized+normalised, used by GDINO
    """
    pil_img = pil_img.convert("RGB")
    img_np = np.array(pil_img)                     # [H, W, 3] uint8, original size

    w, h = pil_img.size
    scale = 800.0 / min(h, w)
    if scale * max(h, w) > 1333:
        scale = 1333.0 / max(h, w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))

    pil_resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    img_t = TVF.to_tensor(pil_resized)                          # [3, H', W'] in [0,1]
    img_t = TVF.normalize(img_t, _GDINO_MEAN, _GDINO_STD)      # [3, H', W']

    return img_np, img_t


# ── Per-frame detection (GDINO + NMS) ────────────────────────────────────────

def _detect_frame(
    pil_img: Image.Image,
    prompt_list: List[str],
    goal_object: Optional[str],
    grounding_model,
    device: str,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
    nms_iou: float = NMS_IOU_THRESH,
) -> Tuple[np.ndarray, List[np.ndarray], List[dict]]:
    """
    Run GroundingDINO detection + NMS on one BEV frame.

    Parameters
    ----------
    pil_img         : PIL RGB image
    prompt_list     : ordered list of text prompts (goal first)
    goal_object     : normalised goal category string or None
    grounding_model : frozen GroundingDINO model
    device          : "cuda" or "cpu"

    Returns
    -------
    img_np      : np.ndarray [H, W, 3] uint8 — for SAM2 set_image_batch
    boxes_px    : np.ndarray [N, 4] float32 xyxy — pixel coords in original image
    detections  : List[dict] — N detection dicts with "category", "is_goal", etc.
    """
    img_np, img_t = _pil_to_gdino(pil_img)
    h, w = img_np.shape[:2]

    if not prompt_list:
        return img_np, np.zeros((0, 4), dtype=np.float32), []

    candidates = _detect_per_prompt(
        img_t, w, h, prompt_list, goal_object,
        grounding_model, device, box_threshold, text_threshold,
    )
    if not candidates:
        return img_np, np.zeros((0, 4), dtype=np.float32), []

    boxes   = np.asarray([c["bbox_xyxy"] for c in candidates], dtype=np.float32)
    scores  = np.asarray([c["nms_score"]  for c in candidates], dtype=np.float32)
    keep    = _apply_nms(boxes, scores, nms_iou)
    kept    = [candidates[i] for i in keep.tolist()]

    # Goal first, then by prompt index, then confidence descending
    kept.sort(key=lambda c: (-int(c["is_goal"]), c["prompt_index"], -c["confidence"]))

    # Re-extract boxes in the sorted order
    boxes_kept = np.asarray([c["bbox_xyxy"] for c in kept], dtype=np.float32)
    return img_np, boxes_kept, kept


# ── BEVEncoder ────────────────────────────────────────────────────────────────

class BEVEncoder:
    """
    Frozen GroundingDINO + SAM2 encoder for a window of T BEV frames.

    Not an nn.Module — holds no trainable parameters and is not registered
    with PyTorch's module system.  This keeps the frozen multi-GB models out
    of AeroDuoPolicy.state_dict() and away from the optimizer.

    Parameters
    ----------
    sam2_predictor  : SAM2ImagePredictor  — already loaded on the target device
    grounding_model : GroundingDINO model — already loaded on the target device

    Usage
    -----
    image_embeds, masks_arrays, detections_list = bev_encoder(
        bev_images, instruction, device
    )
    """

    def __init__(self, sam2_predictor, grounding_model) -> None:
        self.sam2_predictor  = sam2_predictor
        self.grounding_model = grounding_model

    # ── Autocast context helper ───────────────────────────────────────────────

    @staticmethod
    def _autocast_ctx(device):
        device = torch.device(device) if isinstance(device, str) else device
        if device.type == "cuda":
            major = torch.cuda.get_device_properties(device).major
            dtype  = torch.bfloat16 if major >= 8 else torch.float16
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True
            return torch.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()

    # ── Main entry point ──────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(
        self,
        bev_images:  List[Image.Image],   # PIL RGB, length T
        instruction: str,
        device:      torch.device,
    ) -> Tuple[torch.Tensor, List[np.ndarray], List[List[dict]]]:
        """
        Encode a window of T BEV frames.

        Returns
        -------
        image_embeds   : Tensor [T, 256, 64, 64] on ``device``, detached
        masks_arrays   : List[ndarray [N, H, W] bool], length T
        detections_list: List[List[dict]], length T
        """
        T = len(bev_images)
        device_str = str(device)

        # ── 1. Parse instruction ───────────────────────────────────────────────
        goal_object, contextual_nouns, _ = parse_instruction(instruction)
        goal_object      = _normalise_category(goal_object)
        contextual_nouns = [_normalise_category(n) for n in contextual_nouns if n]
        contextual_nouns = [n for n in contextual_nouns if n]
        prompt_list      = build_prompt_list(goal_object, contextual_nouns)

        # ── 2. GDINO detection: per-frame (no batch API) ───────────────────────
        raw_images:      List[np.ndarray]       = []   # uint8 [H, W, 3] for SAM2
        boxes_per_frame: List[np.ndarray]       = []   # float32 [N, 4] xyxy
        detections_list: List[List[dict]]       = []
        img_hw:          List[Tuple[int, int]]  = []

        for pil_img in bev_images:
            img_np, boxes_px, dets = _detect_frame(
                pil_img, prompt_list, goal_object,
                self.grounding_model, device_str,
            )
            raw_images.append(img_np)
            boxes_per_frame.append(boxes_px)
            detections_list.append(dets)
            img_hw.append(img_np.shape[:2])   # (H, W)

        # ── 3. SAM2 batch image encoding — ONE Hiera forward for all T frames ──
        # set_image_batch populates:
        #   _features["image_embed"]    [T, 256, 64, 64]
        #   _features["high_res_feats"] list of [T, C_i, H_i, W_i] tensors
        ctx = self._autocast_ctx(device)
        with ctx:
            self.sam2_predictor.set_image_batch(raw_images)

        # Clone batch features before mask-decode modifies the predictor's state
        embed_batch     = self.sam2_predictor._features["image_embed"].clone()        # [T, 256, 64, 64]
        hres_batch      = [f.clone() for f in self.sam2_predictor._features["high_res_feats"]]
        orig_hw_backup  = list(self.sam2_predictor._orig_hw)

        # ── 4. SAM2 mask decoding: per-frame using cached batch features ────────
        # The mask decoder is cheap; batch-encoding the backbone is the speedup.
        # We temporarily slice the batch tensors to a single-image view, call
        # the standard predict() API, then restore at the end.
        masks_arrays: List[np.ndarray] = []

        for t in range(T):
            H_t, W_t = img_hw[t]

            if len(boxes_per_frame[t]) == 0:
                masks_arrays.append(np.zeros((0, H_t, W_t), dtype=bool))
                continue

            # Temporarily configure predictor for frame t
            self.sam2_predictor._features["image_embed"]    = embed_batch[t:t+1]
            self.sam2_predictor._features["high_res_feats"] = [f[t:t+1] for f in hres_batch]
            self.sam2_predictor._orig_hw                    = [(H_t, W_t)]
            self.sam2_predictor._is_image_set               = True
            self.sam2_predictor._is_batch                   = False

            with ctx:
                masks_t, _, _ = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes_per_frame[t],
                    multimask_output=False,
                )

            # masks_t: [N, 1, H, W] (multimask_output=False) or [N, H, W]
            if masks_t.ndim == 4:
                masks_t = masks_t[:, 0]        # [N, H, W]
            masks_arrays.append(masks_t.astype(bool))

        # ── 5. Restore batch state ─────────────────────────────────────────────
        self.sam2_predictor._features["image_embed"]    = embed_batch
        self.sam2_predictor._features["high_res_feats"] = hres_batch
        self.sam2_predictor._orig_hw                    = orig_hw_backup
        self.sam2_predictor._is_batch                   = True

        return embed_batch.to(device), masks_arrays, detections_list

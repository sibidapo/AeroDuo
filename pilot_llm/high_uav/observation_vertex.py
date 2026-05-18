"""
observation_vertex.py — Build observation vertices o_k from SAM2 cached features.

Each detected object in a BEV frame becomes one observation vertex o_k.
Features are extracted by pooling the SAM2 image encoder's cached spatial
feature map within the object's binary mask — at zero additional compute cost
because set_image was already called inside segment_bev_image.

Why SAM2 features?
------------------
• GroundingDINO localises objects but its internal features are not spatially
  decomposed per object in an easily extractable way.
• SAM2 encodes the full BEV image into a rich, spatially-aligned feature map
  during set_image (called on line 275 of bev_segmentation.py).
• Pooling within SAM2 masks gives spatially precise features using the same
  encoder that defined those mask boundaries.

Feature map accessed
--------------------
After segment_bev_image returns (and set_image has been called internally):

    sam2_predictor._features = {
        "image_embed":  Tensor [1, 256, 64,  64],   # ← used
        "high_res_feats": [
            Tensor [1, 256, 256, 256],               # index 0 — highest res
            Tensor [1, 256, 128, 128],               # index 1
        ]
    }

We use ``image_embed`` because:
  • It is the output of the full Hiera trunk with global attention across the
    entire BEV image — the most semantically rich representation.
  • For navigation graph nodes the goal is to distinguish object identity
    ("umbrella table" vs "trees" vs "buildings"), which benefits more from
    deep semantic features than from spatial precision.
  • BEV objects (building clusters, tree groups, roads) occupy large image
    regions, so the 64×64 resolution still provides enough mask-pooling
    samples for a stable mean.

Mask pooling
------------
For each binary mask of shape [H_img, W_img]:
  1. Resize mask to [256, 256] via nearest-neighbour interpolation.
  2. Mean-pool the feature map at True positions:
        feats[:, mask_resized]  → mean over N_true pixels → [256]
  3. If no True pixels (empty mask), use the global mean of the feature map.

Projection
----------
Each 256-dim raw vector is passed through obs_projector:
    Linear(sam2_feature_dim, D_g) + LayerNorm(D_g)
→ o_k of shape [D_g].

obs_projector is learned jointly with the graph encoder in Stage 4.

Output
------
Returns a list of ObsVertex namedtuples:
    ObsVertex(feature: Tensor [D_g], category: str, is_goal: bool)

These become heterogeneous graph nodes in Stage 3 (PyTorch Geometric).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import AeroduoConfig
except ImportError:
    from config import AeroduoConfig  # direct script execution


# ─── Output type ──────────────────────────────────────────────────────────────

class ObsVertex(NamedTuple):
    """
    A single observation vertex for one detected object.

    Attributes
    ----------
    feature  : Tensor [D_g]  — projected SAM2 feature vector
    category : str           — normalised category name (e.g. "motorcycle")
    is_goal  : bool          — True if this object is the navigation goal
    """
    feature:  torch.Tensor
    category: str
    is_goal:  bool


# ─── Module ───────────────────────────────────────────────────────────────────

class ObservationVertexBuilder(nn.Module):
    """
    Extract and project per-object SAM2 feature vectors into graph node space.

    Trainable parameters
    --------------------
    obs_projector : Linear(sam2_feature_dim, D_g) + LayerNorm(D_g)
        Receives gradients in Stage 2 (joint graph-encoder + flow-matching
        training) and Stage 4 (end-to-end fine-tuning).

    Usage
    -----
    builder = ObservationVertexBuilder(cfg)

    # After calling segment_bev_image(... sam2_predictor=pred ...):
    obs_vertices = builder(
        sam2_predictor=pred,
        masks_array=result["_masks_array"],      # np.ndarray [N, H, W] bool
        detections=result["detections"],          # list of detection dicts
        device=torch.device("cuda"),
    )
    # obs_vertices: List[ObsVertex]
    """

    def __init__(self, cfg: AeroduoConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Learned projection: raw SAM2 pool vector → graph node embedding
        self.obs_projector = nn.Sequential(
            nn.Linear(cfg.sam2_feature_dim, cfg.D_g),
            nn.LayerNorm(cfg.D_g),
        )

    # ── SAM2 feature extraction ───────────────────────────────────────────────

    @staticmethod
    def _extract_feature_map(sam2_predictor) -> torch.Tensor:
        """
        Extract the globally-contextualized image_embed feature map from sam2_predictor.

        After segment_bev_image → set_image has been called, the predictor
        holds:

            _features = {
                "image_embed":    [1, 256, 64,  64],   ← used
                "high_res_feats": [
                    [1, 256, 256, 256],
                    [1, 256, 128, 128],
                ]
            }

        We extract ``image_embed`` and detach it from any existing
        computation graph (it was produced in a no_grad context inside
        set_image, but we detach defensively).

        Returns
        -------
        feat : Tensor [1, C, H_f, W_f]  with C=256, H_f=W_f=64
        """
        features = sam2_predictor._features
        if features is None:
            raise RuntimeError(
                "sam2_predictor._features is None — call segment_bev_image "
                "before ObservationVertexBuilder.forward()."
            )

        image_embed = features.get("image_embed")
        if image_embed is None:
            raise RuntimeError(
                "sam2_predictor._features['image_embed'] is missing."
            )

        # image_embed: globally contextualized trunk output, shape [1, 256, 64, 64]
        feat = image_embed.detach()   # [1, C, H_f, W_f]
        return feat

    # ── Mask pooling ──────────────────────────────────────────────────────────

    @staticmethod
    def _pool_mask(feat: torch.Tensor, mask_hw: "np.ndarray") -> torch.Tensor:
        """
        Mean-pool ``feat`` at positions where ``mask_hw`` is True.

        Parameters
        ----------
        feat     : Tensor [1, C, H_f, W_f]  — feature map (already on device)
        mask_hw  : np.ndarray [H_img, W_img] bool — binary mask at image resolution

        Returns
        -------
        pooled : Tensor [C]
        """
        import numpy as np

        C  = feat.shape[1]
        Hf = feat.shape[2]
        Wf = feat.shape[3]
        device = feat.device

        # Resize mask to feature-map resolution via nearest-neighbour
        # numpy → float32 → [1,1,H,W] → F.interpolate → [1,1,Hf,Wf]
        mask_f32 = torch.from_numpy(mask_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask_resized = F.interpolate(
            mask_f32, size=(Hf, Wf), mode="nearest"
        ).squeeze(0).squeeze(0).bool().to(device)   # [Hf, Wf]

        # feat: [1, C, Hf, Wf] → [C, Hf, Wf]
        feat_sq = feat.squeeze(0)  # [C, Hf, Wf]

        n_true = mask_resized.sum().item()
        if n_true == 0:
            # Empty mask — fall back to global mean of the feature map
            pooled = feat_sq.mean(dim=[1, 2])   # [C]
        else:
            # feat_sq[:, mask_resized] → [C, N_true]
            pooled = feat_sq[:, mask_resized].mean(dim=1)   # [C]

        return pooled   # [C]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        sam2_predictor,
        masks_array,    # np.ndarray [N, H_img, W_img] bool, N = num detections
        detections: list[dict],
        device: torch.device,
        image_embed: Optional[torch.Tensor] = None,  # [1, C, Hf, Wf] — skips predictor lookup
    ) -> list[ObsVertex]:
        """
        Build observation vertices for unique detected objects in one BEV frame.

        When the same category appears in multiple detections (e.g. "numerous
        trees" matched three times), the raw SAM2 pool vectors for all instances
        are averaged before projection, producing a single representative o_k.

        The result is capped at 5 vertices: at most 4 contextual-noun vertices
        plus 1 goal vertex, matching the graph node budget.

        Parameters
        ----------
        sam2_predictor : SAM2ImagePredictor or None
            Must have been used inside segment_bev_image — _features populated.
            Ignored when ``image_embed`` is supplied.
        masks_array : np.ndarray [N, H, W] bool
            Binary masks in detection order, as returned in result["_masks_array"].
        detections : list[dict]
            Detection dicts from result["detections"].  Each must contain
            "category" (str) and "is_goal" (bool).
        device : torch.device
        image_embed : Tensor [1, C, Hf, Wf] or None
            Pre-extracted SAM2 ``image_embed`` feature map.  When provided,
            ``sam2_predictor`` is not accessed.  This is the preferred path in
            ``AeroDuoPolicy`` where features are extracted once per frame in the
            training loop and passed in as plain tensors.

        Returns
        -------
        vertices : list[ObsVertex]
            One ObsVertex per unique detected category (goal first, then
            contextual in order of first detection).  Empty list if N == 0.
        """
        if len(detections) == 0 or masks_array.shape[0] == 0:
            return []

        # ── Extract the cached SAM2 feature map ───────────────────────────
        if image_embed is not None:
            feat = image_embed.detach().to(device)   # [1, C, Hf, Wf]
        else:
            feat = self._extract_feature_map(sam2_predictor).to(device)  # [1,C,Hf,Wf]

        # Keep the learned projector colocated with the pooled SAM2 features.
        first_param = next(self.obs_projector.parameters(), None)
        if first_param is not None and first_param.device != feat.device:
            self.obs_projector.to(feat.device)

        # ── Group raw pool vectors by category ────────────────────────────
        # category → list of raw [C] tensors
        category_vecs: dict[str, list[torch.Tensor]] = {}
        category_is_goal: dict[str, bool] = {}
        category_order: list[str] = []   # insertion order (goal first)

        for i, det in enumerate(detections):
            mask_hw = masks_array[i]   # [H_img, W_img] bool
            raw_vec = self._pool_mask(feat, mask_hw)   # [C] on device

            cat = det["category"]
            is_goal = bool(det["is_goal"])

            if cat not in category_vecs:
                category_vecs[cat] = []
                category_is_goal[cat] = is_goal
                # Insert goal category at the front so it is always included
                # if the 5-vertex cap is reached.
                if is_goal:
                    category_order.insert(0, cat)
                else:
                    category_order.append(cat)

            category_vecs[cat].append(raw_vec)

        vertices: list[ObsVertex] = []

        for cat in category_order[:self.cfg.max_obs_vertices]:
            # Average pool vectors from all instances of this category
            stacked = torch.stack(category_vecs[cat], dim=0)  # [K, C]
            mean_vec = stacked.mean(dim=0)                     # [C]

            # Project to D_g: Linear + LayerNorm
            ok_vec = self.obs_projector(mean_vec.unsqueeze(0)).squeeze(0)  # [D_g]

            vertex = ObsVertex(
                feature=ok_vec,
                category=cat,
                is_goal=category_is_goal[cat],
            )
            vertices.append(vertex)

        return vertices



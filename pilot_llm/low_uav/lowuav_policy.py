from __future__ import annotations

import gc
import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from .config.lowuavconfig import LowUAVConfig
    from .model.lowuav_action_head import LowUAVActionHead
except ImportError:
    from config.lowuavconfig import LowUAVConfig
    from model.lowuav_action_head import LowUAVActionHead


# ── Fuzzy egocentric goal verbalisation ────────────────────────────────────────
#
# Counterpart of high_uav.smolvlm2_encoder.fuzzy_goal_phrase, but expressed in
# the low UAV's body frame: the front camera looks along the heading, so a
# compass cue ("to the south-west") is useless — the cue must say where the
# target lies relative to where the drone is FACING.
#
# Hal-13k frame (validated against BEV flow and the dataset's own priors):
#   +x = north, +y = east; heading = yaw radians from north toward east
#   (verified: motion bearing atan2(Δeast, Δnorth) matches rel_state[:, 3]).
# Relative bearing = wrap(atan2(Δeast, Δnorth) − heading); positive → right.

_EGO_SECTORS = [
    "straight ahead of you",
    "ahead of you and to the right",
    "to your right",
    "behind you and to the right",
    "directly behind you",
    "behind you and to the left",
    "to your left",
    "ahead of you and to the left",
]


def fuzzy_ego_goal_phrase(d_north: float, d_east: float, heading: float) -> str:
    """
    Verbalise a goal offset (Δnorth, Δeast in metres) as a fuzzy cue relative
    to the low UAV's heading, e.g. "ahead of you and to the left, close by".
    Deliberately coarse: 8 direction sectors × 4 distance buckets, so the
    target location is never revealed exactly.
    """
    rel = math.atan2(d_east, d_north) - heading
    rel = (rel + math.pi) % (2.0 * math.pi) - math.pi  # wrap to (-π, π]
    sector = _EGO_SECTORS[int(((math.degrees(rel) + 22.5) % 360.0) // 45.0)]

    dist = math.hypot(d_north, d_east)
    if dist < 30.0:
        qual = "very close"
    elif dist < 80.0:
        qual = "close by"
    elif dist < 160.0:
        qual = "a moderate distance away"
    else:
        qual = "far away"
    return f"{sector}, {qual}"


class LowVLMEncoder(nn.Module):
    """
    Egocentric VLM encoder for the low UAV's front-facing camera.

    Uses a truncated SmolVLM2-2.2B backbone (layers[:vlm_layer_cutoff]) to extract
    hidden states from the front-camera image and navigation instruction.  The
    backbone is always frozen; no gradients flow through it.
    """

    def __init__(self, cfg: LowUAVConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.processor = AutoProcessor.from_pretrained(cfg.smolvlm2_model_name)
        self.processor.image_processor.do_image_splitting = False

        self.vlm = AutoModelForImageTextToText.from_pretrained(
            cfg.smolvlm2_model_name,
            torch_dtype=cfg.dtype,
        ).to(cfg.device)

        self.vlm.eval()
        for param in self.vlm.parameters():
            param.requires_grad_(False)

        # Keep the full layer list in a local only: storing it on self would
        # register all 24 layers as a submodule and pin the truncated upper
        # half (~half the decoder) in GPU memory for the whole run.
        full_layers = self.vlm.model.text_model.layers
        n = len(full_layers)
        cutoff = cfg.vlm_layer_cutoff if cfg.vlm_layer_cutoff is not None else n // 2
        self.vlm_layer_cutoff = cutoff
        self.vlm.model.text_model.layers = full_layers[:cutoff]
        del full_layers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.hidden_size = self.vlm.config.text_config.hidden_size

    def train(self, mode: bool = True):
        # VLM backbone is always frozen; keep it in eval mode regardless.
        return self

    def embed_image(
        self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(dtype=self.vlm.model.vision_model.dtype)
        return self.vlm.get_image_features(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
        )

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.vlm.model.text_model.get_input_embeddings()(token_ids)

    def build_processor_inputs(
        self,
        front_image,
        language_text: str,
        device: torch.device = None,
        goal_offset: Optional[Sequence[float]] = None,
        heading: Optional[float] = None,
    ) -> dict:
        """
        Build SmolVLM2 processor inputs for one front-camera frame.

        goal_offset : optional (Δnorth, Δeast) in metres from the low UAV's
            *current* position to the target.  When given together with
            ``heading`` (yaw radians, north-referenced), the prompt carries a
            fuzzy heading-relative direction cue; otherwise the prompt has no
            direction line (target description only).
        """
        device = device or torch.device(self.cfg.device)
        text_parts = language_text.split(
            "The description of the target and its surrounding is shown below."
        )
        target_text = text_parts[-1].strip()

        if goal_offset is not None and heading is not None:
            direction_line = (
                "Relative to where you are facing, the target is "
                f"{fuzzy_ego_goal_phrase(float(goal_offset[0]), float(goal_offset[1]), float(heading))}.\n"
            )
        else:
            direction_line = ""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": front_image},
                    {
                        "type": "text",
                        "text": (
                            "Front-facing camera from a low-altitude UAV navigating to a target.\n"
                            f"Target and surroundings: {target_text}\n"
                            f"{direction_line}"
                            "Is the target visible? Identify environment features ahead "
                            "that match the target's surroundings, and any clear path toward them."
                        ),
                    },
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(text=prompt, images=[front_image], return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}

    def forward(
        self,
        front_image,
        lang_description: str,
        device: torch.device = None,
        goal_offset: Optional[Sequence[float]] = None,
        heading: Optional[float] = None,
    ) -> torch.Tensor:
        device = device or torch.device(self.cfg.device)

        proc = self.build_processor_inputs(front_image, lang_description, device, goal_offset, heading)
        input_ids = proc["input_ids"]
        pixel_values = proc["pixel_values"]
        pixel_attention_mask = proc["pixel_attention_mask"]
        attn_mask = proc["attention_mask"]

        inputs_embeds = self.embed_tokens(input_ids)
        image_embeds = self.embed_image(pixel_values, pixel_attention_mask)

        inputs_embeds = self.vlm.model.inputs_merger(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_hidden_states=image_embeds,
        )

        lm_out = self.vlm.model.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        return lm_out.last_hidden_state  # [1, L, H]


class LowUAVPolicy(nn.Module):
    """
    Stage 2 policy: frozen high-UAV graph encoder + frozen egocentric VLM + trainable action head.

    Frozen (not registered as nn.Module submodules):
        high_uav_policy  — Stage 1 AeroDuoPolicy; called via get_graph() (no_grad inside).
                           May be None when cfg.use_zgraph=False (standalone low-UAV:
                           the action head conditions on the VLM embeddings alone).

    Frozen (registered nn.Module, zero trainable params):
        vlm_encoder      — LowVLMEncoder (SmolVLM2 backbone truncated at vlm_layer_cutoff)

    Trainable:
        action_head      — [FeatureMerger +] DiT + state/action encoders + decoder
    """

    def __init__(self, cfg: LowUAVConfig, high_uav_policy: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.use_zgraph and high_uav_policy is None:
            raise ValueError("cfg.use_zgraph=True requires a Stage 1 high_uav_policy")
        # Bypass nn.Module.__setattr__ so PyTorch never registers Stage 1 as a submodule.
        # Keeps its weights out of parameters(), state_dict(), and DDP sync.
        object.__setattr__(self, '_high_uav_policy', high_uav_policy)

        self.vlm_encoder = LowVLMEncoder(cfg)
        self.action_head = LowUAVActionHead(cfg)

    def _encode_vlm_batch(
        self,
        front_images: List,
        instructions: List[str],
        device: torch.device,
        low_goal_offsets: Optional[List] = None,   # [B] × ([2] or None) — (Δnorth, Δeast) m
        headings: Optional[torch.Tensor] = None,   # [B] — yaw radians (north-referenced)
    ) -> torch.Tensor:
        """Run LowVLMEncoder over a batch; returns [B, L, H]."""
        if low_goal_offsets is None:
            low_goal_offsets = [None] * len(front_images)
        embs = []
        for b, (img, instr, off) in enumerate(zip(front_images, instructions, low_goal_offsets)):
            heading = float(headings[b]) if (headings is not None and off is not None) else None
            with torch.no_grad():
                emb = self.vlm_encoder(img, instr, device, goal_offset=off, heading=heading)  # [1, L, H]
            embs.append(emb.squeeze(0))
        return torch.stack(embs)  # [B, L, H]

    def forward(
        self,
        bev_images,
        low_uav_front_image: List,
        high_uav_poses: torch.Tensor,
        low_uav_poses_window: torch.Tensor,
        low_uav_pose_current: torch.Tensor,
        instruction: List[str],
        device: torch.device,
        low_uav_traj_target: Optional[torch.Tensor] = None,
        goal_offsets: Optional[List] = None,   # [B] × ([T, 2] or None) — (Δnorth, Δeast) m, high-UAV frame
        low_goal_offset: Optional[List] = None,  # [B] × ([2] or None) — (Δnorth, Δeast) m, low-UAV frame at t_end
    ) -> Dict[str, torch.Tensor]:
        """
        Training (provide low_uav_traj_target):
            → {"loss": scalar, "z_graph": [B, T, D_g] (None when use_zgraph=False)}

        Inference (omit low_uav_traj_target):
            → {"actions": [B, H, 5], "z_graph": [B, T, D_g] (None when use_zgraph=False)}
        """
        if self.cfg.use_zgraph:
            z_graph = self._high_uav_policy.get_graph(
                bev_images=bev_images,
                high_uav_poses=high_uav_poses,
                low_uav_poses_window=low_uav_poses_window,
                instruction=instruction,
                device=device,
                goal_offsets=goal_offsets,
            )  # [B, T, D_g]
        else:
            z_graph = None

        vl_embs = self._encode_vlm_batch(
            low_uav_front_image, instruction, device,
            low_goal_offsets=low_goal_offset,
            headings=low_uav_pose_current[:, 3],  # raw yaw radians (never normalized)
        )  # [B, L, H]

        out: Dict[str, torch.Tensor] = {"z_graph": z_graph}

        if low_uav_traj_target is None:
            out["actions"] = self.action_head.predict_action(vl_embs, z_graph, low_uav_pose_current)
            return out

        out["loss"] = self.action_head(
            vl_embs=vl_embs,
            z_graph=z_graph,
            low_state=low_uav_pose_current,
            low_actions=low_uav_traj_target,
        )
        return out

    @torch.no_grad()
    def get_action(
        self,
        bev_images,
        low_uav_front_image: List,
        high_uav_poses: torch.Tensor,
        low_uav_poses_window: torch.Tensor,
        low_uav_pose_current: torch.Tensor,
        instruction: List[str],
        device: torch.device,
        goal_offsets: Optional[List] = None,   # [B] × ([T, 2] or None) — (Δnorth, Δeast) m, high-UAV frame
        low_goal_offset: Optional[List] = None,  # [B] × ([2] or None) — (Δnorth, Δeast) m, low-UAV frame
    ) -> torch.Tensor:
        """
        Inference-only action prediction.  Returns [B, H, 5] action tensor
        ([x, y, z, sin_h, cos_h]) without building a loss or returning auxiliary outputs.
        """
        if self.cfg.use_zgraph:
            z_graph = self._high_uav_policy.get_graph(
                bev_images=bev_images,
                high_uav_poses=high_uav_poses,
                low_uav_poses_window=low_uav_poses_window,
                instruction=instruction,
                device=device,
                goal_offsets=goal_offsets,
            )  # [B, T, D_g]
        else:
            z_graph = None
        vl_embs = self._encode_vlm_batch(
            low_uav_front_image, instruction, device,
            low_goal_offsets=low_goal_offset,
            headings=low_uav_pose_current[:, 3],
        )  # [B, L, H]
        return self.action_head.predict_action(vl_embs, z_graph, low_uav_pose_current)

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        sd: Dict[str, torch.Tensor] = {}
        for prefix, module in self._trainable_modules():
            for k, v in module.state_dict().items():
                sd[f"{prefix}.{k}"] = v
        return sd

    def load_trainable_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        for prefix, module in self._trainable_modules():
            prefix_dot = f"{prefix}."
            sub_sd = {
                k[len(prefix_dot):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix_dot)
            }
            if sub_sd:
                module.load_state_dict(sub_sd, strict=strict)

    def _trainable_modules(self):
        yield "action_head", self.action_head

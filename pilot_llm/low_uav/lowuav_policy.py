from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from .config.lowuavconfig import LowUAVConfig
    from .model.lowuav_action_head import LowUAVActionHead
except ImportError:
    from config.lowuavconfig import LowUAVConfig
    from model.lowuav_action_head import LowUAVActionHead


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

        self._lm_layers = self.vlm.model.text_model.layers
        n = len(self._lm_layers)
        cutoff = cfg.vlm_layer_cutoff if cfg.vlm_layer_cutoff is not None else n // 2
        self.vlm_layer_cutoff = cutoff
        self.vlm.model.text_model.layers = self._lm_layers[:cutoff]

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
    ) -> dict:
        device = device or torch.device(self.cfg.device)
        text_parts = language_text.split(
            "The description of the target and its surrounding is shown below."
        )
        target_text = text_parts[-1].strip()

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
    ) -> torch.Tensor:
        device = device or torch.device(self.cfg.device)

        proc = self.build_processor_inputs(front_image, lang_description, device)
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
        high_uav_policy  — Stage 1 AeroDuoPolicy; called via encode_graph() under no_grad

    Frozen (registered nn.Module, zero trainable params):
        vlm_encoder      — LowVLMEncoder (SmolVLM2 backbone truncated at vlm_layer_cutoff)

    Trainable:
        action_head      — FeatureMerger + DiT + state/action encoders + decoder
    """

    def __init__(self, cfg: LowUAVConfig, high_uav_policy: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
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
    ) -> torch.Tensor:
        """Run LowVLMEncoder over a batch; returns [B, L, H]."""
        embs = []
        for img, instr in zip(front_images, instructions):
            with torch.no_grad():
                emb = self.vlm_encoder(img, instr, device)  # [1, L, H]
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
    ) -> Dict[str, torch.Tensor]:
        """
        Training (provide low_uav_traj_target):
            → {"loss": scalar, "z_graph": [B, T, D_g]}

        Inference (omit low_uav_traj_target):
            → {"actions": [B, H, 5], "z_graph": [B, T, D_g]}
        """
        with torch.no_grad():
            z_graph = self._high_uav_policy.encode_graph(
                bev_images=bev_images,
                high_uav_poses=high_uav_poses,
                low_uav_poses_window=low_uav_poses_window,
                instruction=instruction,
                device=device,
            )  # [B, T, D_g]

        vl_embs = self._encode_vlm_batch(low_uav_front_image, instruction, device)  # [B, L, H]

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

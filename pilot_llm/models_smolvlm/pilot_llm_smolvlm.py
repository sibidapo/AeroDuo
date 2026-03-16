import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration


class MaskHead(nn.Module):
    def __init__(self, hidden_size, upsample_mode="bilinear"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.upsample_mode = upsample_mode

    def _tokens_to_blocks(self, mask_hidden_states: torch.Tensor, image_seq_len: int) -> torch.Tensor:
        num_tokens = mask_hidden_states.shape[0]
        if num_tokens % image_seq_len != 0:
            raise ValueError(
                f"Expected mask tokens to be divisible by image_seq_len, got {num_tokens} and {image_seq_len}."
            )

        block_side = math.isqrt(image_seq_len)
        if block_side * block_side != image_seq_len:
            raise ValueError(f"Expected a square image_seq_len, got {image_seq_len}.")

        num_blocks = num_tokens // image_seq_len
        token_logits = self.mlp(mask_hidden_states).squeeze(-1)
        return token_logits.view(num_blocks, block_side, block_side)

    def _stitch_blocks(self, block_maps: torch.Tensor) -> torch.Tensor:
        num_blocks, block_h, block_w = block_maps.shape
        if num_blocks == 1:
            return block_maps[0]

        local_blocks = num_blocks - 1
        local_side = math.isqrt(local_blocks)
        if local_side * local_side != local_blocks:
            raise ValueError(
                "SmolVLM split-image decoding expects `rows * cols + 1` blocks "
                f"(local crops + global crop), got {num_blocks}."
            )

        local_map = block_maps[:-1].view(local_side, local_side, block_h, block_w)
        local_map = local_map.permute(0, 2, 1, 3).reshape(local_side * block_h, local_side * block_w)

        # SmolVLM appends one global crop after the split tiles. Blend it back in.
        global_map = block_maps[-1][None, None]
        global_map = F.interpolate(
            global_map,
            size=local_map.shape,
            mode=self.upsample_mode,
            align_corners=False,
        )[0, 0]
        return 0.5 * (local_map + global_map)

    def forward(self, mask_hidden_states: torch.Tensor, image_seq_len: int, target_hw) -> torch.Tensor:
        block_maps = self._tokens_to_blocks(mask_hidden_states, image_seq_len)
        coarse_mask = self._stitch_blocks(block_maps)
        pred_mask = F.interpolate(
            coarse_mask[None, None],
            size=target_hw,
            mode=self.upsample_mode,
            align_corners=False,
        )
        return pred_mask[0]


class PilotLLMSmolVLM(SmolVLMForConditionalGeneration):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        hidden_size = config.text_config.hidden_size
        self.num_image_token = kwargs.get("num_image_token", 2048)
        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2) / (config.scale_factor**2)
        )
        self.mask_head = MaskHead(hidden_size)
        self.depth_head = MaskHead(hidden_size)
        self.mask_query = nn.Parameter(torch.zeros([self.num_image_token, hidden_size]))
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        masks_list: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        is_inference: bool = False,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("`input_ids` is required for PilotLLMSmolVLM.")
        if labels is None:
            raise ValueError("`labels` is required so assistant image tokens can be located.")

        bs = input_ids.shape[0]
        labels = labels.to(input_ids.device)
        special_mask = (input_ids == self.config.image_token_id) & (labels == self.config.image_token_id)
        mask_counts = special_mask.sum(dim=1)
        max_needed = int(mask_counts.max().item()) if mask_counts.numel() > 0 else 0
        if max_needed > self.mask_query.shape[0]:
            raise ValueError(
                f"Need {max_needed} mask query tokens, but the query bank only has {self.mask_query.shape[0]}. "
                "Increase `num_image_token` when loading the model."
            )

        if inputs_embeds is None:
            inputs_embeds = self.model.text_model.get_input_embeddings()(input_ids).to(input_ids.device)

            if pixel_values is not None and image_hidden_states is not None:
                raise ValueError("You cannot specify both `pixel_values` and `image_hidden_states` at the same time.")
            elif pixel_values is not None:
                image_hidden_states = self.model.get_image_features(pixel_values, pixel_attention_mask).to(input_ids.device)
            elif image_hidden_states is not None:
                image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

            if image_hidden_states is not None:
                inputs_embeds = self.model.inputs_merger(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    image_hidden_states=image_hidden_states,
                )

            for i in range(bs):
                count = int(mask_counts[i].item())
                if count == 0:
                    continue
                mask_indices = special_mask[i].nonzero(as_tuple=True)[0]
                inputs_embeds[i, mask_indices] = self.mask_query[:count].to(inputs_embeds.dtype)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]

        preds = []
        for i in range(bs):
            mask_indices = special_mask[i].nonzero(as_tuple=True)[0]
            if mask_indices.numel() == 0:
                raise ValueError("No assistant image tokens were found for one sample in the batch.")

            target_hw = masks_list[i].shape[-2:] if masks_list is not None else (784, 784)
            pred_mask = self.mask_head(hidden_states[i, mask_indices], self.image_seq_len, target_hw)
            preds.append(pred_mask)

        preds = torch.stack(preds, dim=0)

        if is_inference:
            return preds

        if masks_list is None:
            raise ValueError("`masks_list` is required to compute the PilotLLMSmolVLM training loss.")

        bce_loss = 0
        num_masks = 0
        for pred_mask, gt_mask in zip(preds, masks_list):
            gt_mask = gt_mask.to(pred_mask.device)
            pred_mask = pred_mask[:, : gt_mask.shape[1], : gt_mask.shape[2]]
            assert (
                gt_mask.shape == pred_mask.shape
            ), f"gt_mask.shape: {gt_mask.shape}, pred_mask.shape: {pred_mask.shape}"
            bce_loss += self.bce_loss(pred_mask, gt_mask)
            num_masks += gt_mask.shape[0]

        loss = bce_loss / (num_masks + 1e-8)
        return preds, loss

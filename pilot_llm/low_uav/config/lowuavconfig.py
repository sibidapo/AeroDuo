from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

@dataclass
class FeatureMergerConfig:

    attn_hidden_dim: int = 2048 

    n_head: int = 8
    
    kv_dim: int = 2048 # z_graph output proj dimension

    dropout: float = 0.0

@dataclass
class DITConfig:

    num_attention_heads: int = field(default=32, metadata={"help": "Number of attention heads"})

    attention_head_dim: int = field(default=48, metadata={"help": "Attention head dim"})

    num_layers: int = field(default=16, metadata={"help": "Number of interleavened self and cross attention layers"})

    max_num_positional_embeddings: int = field(default=512, metadata={"help":""})

    interleave_self_attention: bool = field(default=True, metadata={"help": "Whether or not use self attention interleavened with cross attention"})

    cross_attention_dim: int = field(default=2048, metadata={"help":"Cross attention dim"})

    dropout: float = field(default=0.2, metadata={"help":"Dropout probability"})

    activation_fn: str = field(default="gelu-approximate", metadata={"help":""})

    final_dropout: bool = field(default=True, metadata={"help":""})

    norm_type: str = field(default="ada_norm", metadata={"help":""})

    positional_embeddings: str = field(default="sinusoidal", metadata={"help":""}) ##TODO

    attention_bias: bool = field(default=True, metadata={"help":""})

    norm_eps: float = field(default=1e-5, metadata={"help":""})

    norm_elementwise_affine: bool = field(default=False, metadata={"help":""})

    upcast_attention: bool = field(default=False, metadata={"help":""})


@dataclass
class LowUAVConfig:
    feat_merger_cfg: FeatureMergerConfig = field(default_factory=FeatureMergerConfig)

    dit_cfg: DITConfig = field(default_factory=DITConfig)
    
    ## VLM READOUT
    smolvlm2_model_name: str = field(default="HuggingFaceTB/SmolVLM2-2.2B-Instruct", metadata={"help":""})

    dtype: torch.dtype = field(default=torch.bfloat16, metadata={"help": "dtype for loading SmolVLM2 weights"})

    device: str = field(default="cuda", metadata={"help": "Device to run on: cuda | cpu"})

    vlm_layer_cutoff: int = field(default=12, metadata={"help":"num_hidden_layers = 24 → cutoff = 24 // 2 = 12."})

    ## ACTION HEAD
    action_horizon: int = field(default=8, metadata={"help":"action horizon of low UAV"})

    vlm_hidden_dim: int = field(default=2048, metadata={"help":"Hidden size of the LLaMA text decoder inside SmolVLM2-2.2B-Instruct"})

    D_g: int = field(default=1024, metadata={"help":"Shared output dimensionality for all graph nodes"})

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})

    action_dim: int = field(default=5, metadata={"help": "Action Dim -> [x, y, z, sin_h, cos_h] -> Transformed from [x, y, z, heading] in the Actionhead"})

    state_dim: int = field(default=5, metadata={"help": "Action Dim -> [x, y, z, sin_h, cos_h]"})

    hidden_dim: int = field(default=1024, metadata={"help": "State and Action encoder hidden size"})

    output_dim: int = field(default=1024, metadata={"help": ""})

    input_emb_dim: int = field(default=1536, metadata={"help": "DIT input dim"})

    max_seq_len: int = field(default=1024, metadata={"help":""})

    num_inference_timesteps: Optional[int] = field(default=4, metadata={"help": "Number of inference steps for noise diffusion."})

    ## BETA DISRTIBUTION
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})

    noise_beta_beta : float = field(default=1.0, metadata={"help": ""})

    noise_s: float = field(default=0.99, metadata={"help": "Flow matching noise Beta distribution s."})
    
    num_timestep_buckets: int = field(default=1000, metadata={"help": "Number of timestep discretization buckets."})


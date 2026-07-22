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

    num_attention_heads: int = field(default=16, metadata={"help": "Number of attention heads"})

    attention_head_dim: int = field(default=48, metadata={"help": "Attention head dim"})

    num_layers: int = field(default=8, metadata={"help": "Number of interleavened self and cross attention layers"})

    max_num_positional_embeddings: int = field(default=512, metadata={"help":""})

    interleave_self_attention: bool = field(default=True, metadata={"help": "Whether or not use self attention interleavened with cross attention"})

    cross_attention_dim: int = field(default=2048, metadata={"help":"Cross attention dim"})

    dropout: float = field(default=0.1, metadata={"help":"Dropout probability"})

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
    use_zgraph: bool = field(default=True, metadata={"help": "Condition the action head on the high-UAV z_graph via the FeatureMerger. False → standalone low-UAV: DiT cross-attends to the (LayerNormed) VLM embeddings directly and Stage 1 is never called."})

    action_horizon: int = field(default=8, metadata={"help":"action horizon of low UAV"})

    vlm_hidden_dim: int = field(default=2048, metadata={"help":"Hidden size of the LLaMA text decoder inside SmolVLM2-2.2B-Instruct"})

    D_g: int = field(default=1024, metadata={"help":"Shared output dimensionality for all graph nodes"})

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})

    action_dim: int = field(default=5, metadata={"help": "Action Dim -> [x, y, z, sin_h, cos_h] -> Transformed from [x, y, z, heading] in the Actionhead"})

    state_dim: int = field(default=5, metadata={"help": "Action Dim -> [x, y, z, sin_h, cos_h]"})

    hidden_dim: int = field(default=512, metadata={"help": "State and Action encoder hidden size"})

    output_dim: int = field(default=512, metadata={"help": ""})

    input_emb_dim: int = field(default=768, metadata={"help": "DIT input dim; must equal num_attention_heads * attention_head_dim"})

    max_seq_len: int = field(default=1024, metadata={"help":""})

    num_inference_timesteps: Optional[int] = field(default=4, metadata={"help": "Number of inference steps for noise diffusion."})

    ## BETA DISRTIBUTION
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})

    noise_beta_beta : float = field(default=1.0, metadata={"help": ""})

    noise_s: float = field(default=0.99, metadata={"help": "Flow matching noise Beta distribution s."})
    
    num_timestep_buckets: int = field(default=1000, metadata={"help": "Number of timestep discretization buckets."})

    ## POSE NORMALIZATION
    # Applied by low_uav/dataset2.py at load time to each UAV's rel_state
    # (xyz relative to that UAV's own first frame in the episode; heading is
    # left as an absolute, wrapped world yaw and is never z-normalized — it's
    # circular and gets sin/cos-encoded inside the model instead).
    #
    # Each mean/std is a SINGLE GLOBAL statistic: relative [dx, dy, dz] rows
    # were pooled across every episode in data/train_data_new.json (8901
    # episodes; see data_preprocessing/generate_trajectories.py), not
    # computed per-episode. This is the same convention
    # eval/dualuavpilot.py._normalize_pose already uses from raw AirSim
    # poses at inference time, so training and eval normalization now agree.
    high_pose_mean: list = field(default_factory=lambda: [3.3610526945746626, -0.3325858628344677, 0.33880942396714486])
    high_pose_std: list = field(default_factory=lambda: [114.7756392836723, 109.91593135767647, 0.33489186206406946])
    low_pose_mean: list = field(default_factory=lambda: [3.94354055381394, -0.4868388438716282, -15.014240206727887])
    low_pose_std: list = field(default_factory=lambda: [119.14186092932334, 115.9592733580364, 24.65062999168105])

    ## ACTION NORMALIZATION
    # GLOBAL min/max for the low-UAV "action": the xyz displacement of each
    # of the H future low-UAV steps relative to the CURRENT pose (not the
    # episode start — that's the pose normalization above). Computed by
    # pooling every valid (episode, anchor t) chunk's [dx, dy, dz] rows
    # dataset-wide, separately per horizon (data_preprocessing/
    # compute_action_stats.py, over the same 8899 episodes as the pose
    # stats). Used by low_uav/dataset2.py and high_uav/dataset.py to build
    # low_uav_traj_target: normalized = 2*(x-min)/(max-min) - 1. Must have
    # an entry for whichever action_horizon is actually used.
    action_min_max: dict = field(default_factory=lambda: {
        2: {
            "min": [-57.82021141052246, -55.12806701660156, -20.334530651569366],
            "max": [57.591400146484375, 45.1695556640625, 37.870065689086914],
        },
        4: {
            "min": [-99.37514746189117, -76.58756959438324, -20.36567783355713],
            "max": [114.47732543945312, 72.4588623046875, 47.49151420593262],
        },
        8: {
            "min": [-126.96470642089844, -90.15975892543793, -39.8125],
            "max": [202.19479370117188, 113.83048248291016, 65.4615421295166],
        },
    })


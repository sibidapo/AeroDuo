#!/usr/bin/env python
"""Quick sanity test for Stage 2 modules (no heavy model loading required)."""
import sys
sys.path.insert(0, '/storage/project/r-cj124-0/sibidapo3/8750/aeroduo_ws/aeroduo/pilot_llm/high_uav')

failed = []

def check(name, cond, msg=""):
    if cond:
        print(f"  [OK]  {name}")
    else:
        print(f"  [FAIL] {name}  {msg}")
        failed.append(name)

print("\n=== AeroduoConfig ===")
from config import AeroduoConfig
cfg = AeroduoConfig()
check("D_g",               cfg.D_g == 256)
check("smolvlm2_hidden_dim", cfg.smolvlm2_hidden_dim == 2048)
check("vlm_layer_cutoff",  cfg.vlm_layer_cutoff == 12)
check("high_uav_pose_dim", cfg.high_uav_pose_dim == 5)
check("low_uav_state_dim", cfg.low_uav_state_dim == 7)
check("sam2_feature_dim",  cfg.sam2_feature_dim == 256)
check("sam2_spatial_res",  cfg.sam2_spatial_res == 256)
check("lora_rank is None", cfg.lora_rank is None)

print("\n=== State Projectors ===")
import torch
from state_projector import HighUAVPoseProjector, LowUAVStateProjector, encode_heading

pose_proj  = HighUAVPoseProjector(cfg)
state_proj = LowUAVStateProjector(cfg)
h   = torch.tensor([0.5, 3.14, -1.0])
enc = encode_heading(h)
check("encode_heading shape", enc.shape == (3, 2), f"got {enc.shape}")
pose = torch.tensor([[1.0, 2.0, 30.0, 0.785]])
out  = pose_proj(pose)
check("HighUAVPoseProjector shape", out.shape == (1, 1, cfg.smolvlm2_hidden_dim), f"got {out.shape}")
state = torch.randn(1, cfg.low_uav_state_dim)
out2  = state_proj(state)
check("LowUAVStateProjector shape", out2.shape == (1, 1, cfg.smolvlm2_hidden_dim), f"got {out2.shape}")
null_out = state_proj(None)
check("null_token shape", null_out.shape == (1, 1, cfg.smolvlm2_hidden_dim), f"got {null_out.shape}")
check("null_token is Parameter", isinstance(state_proj.null_token, torch.nn.Parameter))

print("\n=== ObservationVertexBuilder ===")
import numpy as np
from observation_vertex import ObservationVertexBuilder, ObsVertex
obs_builder = ObservationVertexBuilder(cfg)

class FakePredictor:
    pass
mock_pred = FakePredictor()
mock_pred._features = {
    'image_embed':   torch.zeros(1, 256, 64, 64),
    'high_res_feats': [torch.randn(1, 256, 256, 256), torch.randn(1, 256, 128, 128)]
}
N, H, W = 3, 480, 640
masks = (np.random.rand(N, H, W) > 0.7).astype(bool)
dets  = [
    {'category': 'motorcycle', 'is_goal': True},
    {'category': 'bridge',     'is_goal': False},
    {'category': 'tree',       'is_goal': False},
]
verts = obs_builder(mock_pred, masks, dets, torch.device('cpu'))
check("num vertices", len(verts) == N, f"got {len(verts)}")
check(
    "obs_projector device",
    next(obs_builder.obs_projector.parameters()).device.type == 'cpu',
    f"got {next(obs_builder.obs_projector.parameters()).device}",
)
for i, v in enumerate(verts):
    check(f"vertex[{i}] type",     isinstance(v, ObsVertex))
    check(f"vertex[{i}] feature",  v.feature.shape == (cfg.D_g,))
    check(f"vertex[{i}] category", v.category == dets[i]['category'])
    check(f"vertex[{i}] is_goal",  v.is_goal  == dets[i]['is_goal'])

empty = obs_builder(mock_pred, np.zeros((0, H, W), dtype=bool), [], torch.device('cpu'))
check("empty case", empty == [])

allF  = np.zeros((1, H, W), dtype=bool)
v_fb  = obs_builder(mock_pred, allF, [{'category':'road','is_goal':False}], torch.device('cpu'))
check("zero-pixel fallback", len(v_fb)==1 and v_fb[0].feature.shape==(cfg.D_g,))

if torch.cuda.is_available():
    obs_builder_cuda = ObservationVertexBuilder(cfg)
    mock_pred_cuda = FakePredictor()
    mock_pred_cuda._features = {
        'image_embed': torch.zeros(1, 256, 64, 64, device='cuda'),
        'high_res_feats': [torch.randn(1, 256, 32, 32, device='cuda')],
    }
    verts_cuda = obs_builder_cuda(mock_pred_cuda, masks, dets, torch.device('cuda'))
    check("cuda num vertices", len(verts_cuda) == N, f"got {len(verts_cuda)}")
    check(
        "obs_projector cuda",
        next(obs_builder_cuda.obs_projector.parameters()).device.type == 'cuda',
        f"got {next(obs_builder_cuda.obs_projector.parameters()).device}",
    )
    check(
        "cuda vertex features",
        all(v.feature.device.type == 'cuda' for v in verts_cuda),
        "expected all vertex features on cuda",
    )
else:
    print("  [SKIP] CUDA observation test (CUDA unavailable)")

print()
if failed:
    print(f"FAILED: {failed}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")

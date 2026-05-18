# high_uav — High-UAV Spatiotemporal Graph Node Construction

This directory implements **Stage 1** (Grounded-SAM2 segmentation) and
**Stage 2** (position vertex V_t and observation vertices o_k) of the aeroduo
collaborative UAV navigation system.

---

## Directory layout

```
high_uav/
├── config.py               # Centralised config dataclass — single source of truth
├── noun_extractor.py       # Stage 1a: parse navigation instruction → goal + context nouns
├── bev_segmentation.py     # Stage 1b: run Grounded-SAM2 on one BEV image
├── batch_segment.py        # Stage 1c: batch wrapper for the above
├── smolvlm2_encoder.py     # Stage 2: SmolVLM2 frozen encoder, mid-layer readout
├── state_projector.py      # Stage 2: HighUAVPoseProjector + LowUAVStateProjector
├── position_vertex.py      # Stage 2: PositionVertexBuilder → V_t ∈ R^{D_g}
├── observation_vertex.py   # Stage 2: ObservationVertexBuilder → o_k ∈ R^{D_g}
└── weights/                # (local model weight symlinks / small assets)
```

---

## Full on-the-fly pipeline per frame

For every BEV frame the following runs entirely in memory with no disk I/O
beyond reading the raw image:

```
raw BEV image + instruction JSON
        │
        ▼
noun_extractor.parse_instruction()
        │  goal_object, contextual_nouns, description
        ▼
bev_segmentation.segment_bev_image(sam2_predictor, grounding_model, ...)
        │  result["detections"], result["_masks_array"]
        │  sam2_predictor._features now cached (set_image called internally)
        │
        ├─────────────────────────────────────────────────────────────┐
        ▼                                                             ▼
ObservationVertexBuilder.forward(                    PositionVertexBuilder.forward(
    sam2_predictor,                                      bev_image,
    masks_array,                                         language_text,
    detections,                                          high_uav_pose,
    device)                                              low_uav_state,
        │                                                device)
        │  List[ObsVertex(feature[D_g], category, is_goal)]         │  V_t [D_g]
        │                                                             │
        └──────────────────── Stage 3: PyG graph assembly ───────────┘
```

---

## Stage 2 modules

### `config.py` — `AeroduoConfig`

Single source of truth for all hyperparameters.  Every stage imports from here.

| Field | Value | Source |
|---|---|---|
| `D_g` | 256 | graph embedding dim |
| `smolvlm2_model_name` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | — |
| `smolvlm2_hidden_dim` | 2048 | config.json → text_config → hidden_size |
| `vlm_layer_cutoff` | 12 | num_hidden_layers(24) // 2 |
| `high_uav_pose_dim` | 5 | (x, y, z, sin_h, cos_h) |
| `low_uav_state_dim` | 7 | configurable — match your dataset |
| `sam2_feature_dim` | 256 | FPN neck output channels (all scales) |
| `sam2_spatial_res` | 256 | spatial dim of high_res_feats[0] |
| `lora_rank` | None | set to int in Stage 4 |

---

### `state_projector.py`

**`HighUAVPoseProjector`**
- Input: `(x, y, z, heading_rad)` → heading encoded as `(sin, cos)` → 5-dim
- Output: `[1, smolvlm2_hidden_dim]` token embedding

**`LowUAVStateProjector`**
- Input: `(x, y, z, heading_rad, ...)` or `None`
- Output: `[1, smolvlm2_hidden_dim]` token, or `null_token` (learnable `nn.Parameter`) when input is `None`

---

### `smolvlm2_encoder.py` — `SmolVLM2Encoder`

Loads SmolVLM2-2.2B-Instruct completely frozen.  Exposes:
- `build_processor_inputs(bev_image, language_text, device)` — standard processor
- `forward_with_extra_tokens(bev_image, language_text, pose_token, state_token, device)`
  — runs the decoder with the two appended tokens, returns `hidden_states[vlm_layer_cutoff]`
- `decoder_layers` — reference to the 24 `LlamaDecoderLayer` modules for
  LoRA attachment (Stage 4) without refactoring

**Token layout** (confirmed by printing tokenised inputs):
```
[img_tok_0 … img_tok_M]  [lang_tok_0 … lang_tok_L]  [pose_tok]  [state_tok]
 ↑ BEV patch tokens          ↑ instruction tokens      ↑ pose      ↑ state/null
 (variable)                  (variable)                  (1)         (1)
```
Mean-pool at layer 12 spans **all** positions → V_t attends to every modality.

Named attention parameters accessible via `SmolVLM2Encoder.vlm.named_parameters()`:
```
language_model.model.layers.{i}.self_attn.q_proj.weight
language_model.model.layers.{i}.self_attn.k_proj.weight
language_model.model.layers.{i}.self_attn.v_proj.weight
language_model.model.layers.{i}.self_attn.o_proj.weight
```

---

### `position_vertex.py` — `PositionVertexBuilder`

```python
builder = PositionVertexBuilder(cfg)
V_t = builder(
    bev_image=pil_image,
    language_text=instruction_str,
    high_uav_pose=torch.tensor([x, y, z, heading]),  # float32
    low_uav_state=torch.tensor([...]),                # or None
    device=torch.device("cuda"),
)
# V_t: Tensor [D_g]
```

Trainable: `pose_proj`, `state_proj`, `vt_proj` (Linear + LayerNorm).  
Frozen: all SmolVLM2 parameters.

---

### `observation_vertex.py` — `ObservationVertexBuilder`

```python
builder = ObservationVertexBuilder(cfg)
obs_vertices = builder(
    sam2_predictor=pred,              # after segment_bev_image
    masks_array=result["_masks_array"],   # [N, H, W] bool
    detections=result["detections"],
    device=torch.device("cuda"),
)
# obs_vertices: List[ObsVertex(feature[D_g], category, is_goal)]
```

Extracts `predictor._features["high_res_feats"][0]` — shape `[1, 256, 256, 256]`.  
Resizes masks to 256×256 (nearest-neighbour), mean-pools, projects via `obs_projector`.

Trainable: `obs_projector` (Linear + LayerNorm).

---

## Future stages

| Stage | What is built |
|---|---|
| Stage 3 | PyTorch Geometric heterogeneous graph from `V_t` + `ObsVertex` lists; temporal edges with relative pose displacement; observation edges weighted by `is_goal` |
| Stage 4 | Graph transformer + flow-matching training end-to-end; LoRA adapters on SmolVLM2 attention projections |
| Stage 5 | Low-UAV policy conditioned on frozen `z_graph`, front-camera VLM embedding, and low-UAV state |

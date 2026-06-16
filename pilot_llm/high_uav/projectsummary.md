# Project Summary: Collaborative Dual-UAV Navigation System (AeroDuo)

## Project Goal
Two-UAV collaborative navigation system where a high-altitude UAV builds a spatiotemporal observation graph from BEV imagery, and a low-altitude UAV uses the graph embedding to navigate to a language-described goal object.

---

## System Architecture — Two-Stage Training

**Stage 1:** High UAV graph encoder trained jointly with flow matching trajectory predictor, supervised by low UAV ground truth trajectory from `low_uav_traj.json`. Graph encoder, SmolVLM2 (with eventual LoRA), and flow matching network trained end-to-end.

**Stage 2:** Graph encoder frozen. Low UAV trains navigation policy conditioned on frozen `z_graph`, front-camera SmolVLM2 embedding, and state.

---

## Dataset — Hal-13k

```
<dataset_root>/<town_name>/<episode_id>/
    bevcamera/                          ← BEV PNG frames (high UAV)
    object_description_with_help.json   ← language instruction (list with one string)
    high_uav_traj.json                  ← {"normalized_state": [[x,y,z,heading], ...]}  [T, 4]
    low_uav_traj.json                   ← {"normalized_state": [[x,y,z,heading], ...]}  [T, 4]
```

Both UAV trajectories are `[T, 4]` — `(x, y, z, heading_rad)`. The low UAV trajectory is the flow matching supervision target. There is no separate action file.

---

## Implemented Modules

### `SmolVLM2Encoder` (`smolvlm2_encoder.py`)

**Model:** `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (notebook uses 2.2B)

**Key decisions:**
- `do_image_splitting = False` — global image only, no tiling → `pixel_values` shape `[1, 1, 3, H, W]` → image embed `[1, 81, 2048]`
- Decoder physically truncated to N//2 layers at init: `self.vlm.model.text_model.layers = self._lm_layers[:cutoff]`
- All VLM parameters frozen: `requires_grad_(False)`
- `embed_image` calls `self.vlm.get_image_features(pixel_values, pixel_attention_mask)` which internally runs vision_model → `.last_hidden_state` → connector. Returns `[1, 81, 2048]` after pixel shuffle. No additional `.last_hidden_state` call needed — `get_image_features` on `SmolVLMModel` already does the full pipeline including connector.
- `embed_tokens` calls `self.vlm.model.text_model.embed_tokens(token_ids)` directly
- `inputs_merger` used to replace image placeholder tokens (49190) in input_ids with actual image embeddings — processor already inserts boundary tokens and placeholders, so manual boundary token prepending is WRONG
- State tokens appended AFTER merged sequence, attention mask extended by 2

**Token layout in input_ids (confirmed by inspection):**
```
[1, 11126, 42]  ← BOS, user turn tokens
[49189]         ← fake_image_token (boundary start)
[49152]         ← global_image_token
[49190 × 81]    ← image placeholder tokens → replaced by actual image embeds via inputs_merger
[49189]         ← fake_image_token (boundary end)
[lang tokens ~100]
[49279, 198]    ← end of turn, newline
```

**forward() signature:**
```python
def forward(self, bev_image, lang_description, low_state, high_state, device) -> Tensor[1, 191, 2048]
```

**forward() pipeline:**
1. `build_processor_inputs` → proc dict
2. `embed_tokens(input_ids)` → `[1, S, 2048]`
3. `embed_image(pixel_values, pixel_attention_mask)` → `[1, 81, 2048]`
4. `vlm.model.inputs_merger(input_ids, inputs_embeds, image_hidden_states)` → `[1, S, 2048]` with placeholders replaced
5. Cat `high_state_token [1,1,2048]` and `low_state_token [1,1,2048]`
6. Extend attention mask by 2
7. `vlm.model.text_model(inputs_embeds, attention_mask, output_hidden_states=False, use_cache=False)`
8. Return `lm_out.last_hidden_state` → `[1, 191, 2048]`

**Important:** LlamaModel uses causal attention. State tokens (appended last) attend to all prior tokens, but image/language tokens CANNOT attend back to state tokens. This is acceptable — states are informed by scene but not vice versa. Full bidirectionality would require prepending state tokens or a custom attention mask.

**Normalisation:** Handled by processor — do NOT add manual normalisation in `embed_image`.

### `UAVPoseProjector` (`smolvlm2_encoder.py`)
```python
# Input: [B, 4] pose (x, y, z, heading_rad)
# encode_heading: extracts xyz + sin/cos(heading) → [B, 5]
# proj: Linear(5, 2048) → [B, 2048]
# output: [B, 1, 2048] — unsqueeze(1) for token concatenation
```

### `ObservationVertexBuilder` (`observation_vertex.py`)

**Key decisions:**
- Uses SAM2 `image_embed` features — `[1, 256, 64, 64]` — NOT `high_res_feats`. Rationale: globally contextualized, semantically richest, spatially aligned with masks, sufficient resolution for BEV objects
- Pools within each detection mask by resizing mask to 64×64 (nearest-neighbour) then mean-pooling feature map at True positions
- Empty mask fallback: global mean of feature map
- Groups detections by category, running mean of raw [256] vectors across multiple detections of same category
- `ObsProjector`: `nn.Sequential(nn.Linear(256, D_g), nn.LayerNorm(D_g))` — applied AFTER category accumulation, NOT inside the builder
- Returns `List[ObsVertex(feature=[256], category, is_goal)]` with raw unprojected features
- Cap: 5 unique categories max (goal pinned first)

**SAM2 feature cache:** After `segment_bev_image` → `set_image`, features live in `sam2_predictor._features`:
```python
'image_embed':   [1, 256, 64, 64]   ← used
'high_res_feats': [[1, 32, 256, 256], [1, 64, 128, 128]]  ← not used
```

### `PositionVertexBuilder` (`position_vertex.py`)
- Calls `SmolVLM2Encoder.forward` → `[1, 191, 2048]`
- Mean pools across 191 tokens → `[1, 2048]` (noted limitation — loses spatial structure)
- Projects via `vt_proj: Linear(2048, D_g) + LayerNorm` → `V_t [D_g]`
- **NOTE:** `@torch.no_grad()` must NOT be on the outer forward — gradients must flow through `vt_proj`, `pose_proj`, `state_proj`

---

## Vertex Representations (Confirmed in Notebook)

```python
position_vertices.shape    = [5, 191, 2048]
# 5 = trajectory timesteps (sliding window)
# 191 = VLM token sequence length (image + language + state tokens)
# 2048 = SmolVLM2 hidden dimension

observation_vertices.shape = [5, 3, 256]
# 5 = timesteps
# 3 = max unique detections per timestep (padded with zeros if fewer detected)
# 256 = mean-pooled SAM2 image_embed feature per unique category
```

---

## Graph Structure

**Type:** Heterogeneous graph with two node types and two edge types.

**Node types:**
- `position` nodes: one per timestep, feature `[D_g]` (from SmolVLM2 readout)
- `observation` nodes: one per unique detected category (accumulated across timesteps by running mean), feature `[D_g]` (from ObsProjector)

**Edge types:**
- `(position, temporal, position)`: consecutive timestep edges, edge feature = relative pose displacement `Δpose [4]`
- `(position, observes, observation)`: position node to observation nodes detected at that timestep, edge feature = binary `is_goal` weight (1.0 goal, 0.5 contextual)

**Adjacency matrix `A [T, K]`:**
- `T` rows = timesteps, `K` columns = unique object categories
- `A[t, k] = 1` if category k was detected at timestep t, else 0
- NOT required to be square — `T` and `K` are independently capped (T=5 window, K=5 categories)
- Padded rows/columns are zeros when fewer timesteps processed or fewer categories detected
- No-detection timesteps: insert synthetic global-mean SAM2 feature node with category "scene" as fallback

**Graph operation (bilinear interaction):**
```
A: [T, K], P: [T, 2048], O: [K, 256]
(A' × P)' × O = [K,T] × [T,2048] then transpose × [K,256] = [2048, 256]
```
This cross-modal interaction matrix is then reduced — e.g. `result.mean(dim=-1)` → `[2048]` → `Linear(2048, D_g)` → `z_graph`.

---

## Graph Encoder — HGTConv

**Decision:** Use `HGTConv` from PyTorch Geometric — handles heterogeneous node types natively, pip-installable, maps directly to `HeteroData`.

**Why not Graphormer HuggingFace:** Maintenance-only mode, tightly coupled to molecular graph preprocessing pipeline, impractical to adapt.

**Key insight on position node representation:**
- Mean pooling `[191, 2048]` → `[2048]` loses spatial structure — image tokens each represent specific BEV spatial regions
- Better: Perceiver-style compression — 1 learned query cross-attending over 191 tokens → `[D_g]` per timestep
- This is exactly what SmolVLM2's own connector does for image patches

**Proposed `GraphEncoder` pipeline:**
```python
# 1. Encode position nodes
PositionNodeEncoder: [T, 191, 2048] → cross-attn with 1 learned query → [T, D_g]

# 2. Encode observation nodes  
ObservationNodeEncoder: valid rows of [T, K_max, 256] → [K_active, D_g]
obs_mask = observation_vertices.abs().sum(dim=-1) > 0  # [T, K_max]

# 3. Build edge indices
temporal_edge_index: [2, T-1]   # consecutive position nodes
obs_edge_index:      [2, K_active]  # position t → its observation nodes

# 4. HGTConv layers (L=3, heads=4)
metadata = (['position', 'observation'],
            [('position','temporal','position'), ('position','observes','observation')])
HGTConv(in_channels={'position': D_g, 'observation': D_g}, out_channels=D_g, ...)

# 5. Readout
z_graph = x_dict['position']   # [T, D_g] — keep full sequence, DO NOT mean pool
```

**Critical decision on z_graph shape:**
- `z_graph` should be `[T, D_g]` NOT `[D_g]` — mean pooling to a single 256-dim vector loses temporal ordering and is too small to carry episode-level information
- The flow matching transformer denoiser cross-attends to the full `[T, D_g]` sequence of conditioning tokens, exactly as SmolVLA's action expert cross-attends to the VLM prefix hidden states

---

## Flow Matching Network

**Architecture:** Transformer denoiser following SmolVLA action expert pattern — interleaved cross-attention and causal self-attention layers.

**Inputs:**
- `z_graph [T, D_g]` — graph embedding sequence (conditioning)
- `x_tau [H, action_dim]` — noisy trajectory at interpolation time τ
- `tau` — scalar interpolation time

**Output:** `v_theta [H, action_dim]` — predicted vector field

**Loss:**
```python
x_0 = gt_trajectory    # clean — from low_uav_traj.json
x_1 ~ N(0, I)          # noise
tau ~ U[0, 1]
x_tau = tau * x_1 + (1 - tau) * x_0
v_target = x_1 - x_0
L_flow = MSE(v_theta(x_tau, tau, z_graph), v_target)
```

---

## Config (`AeroduoConfig`)

```python
D_g = 256
smolvlm2_hidden_dim = 2048
vlm_layer_cutoff = 12          # N//2 for 24-layer model
high_uav_pose_dim = 5          # (x, y, z, sin_h, cos_h)
low_uav_state_dim = 4          # (x, y, z, heading_rad) from dataset inspection
sam2_feature_dim = 256
graph_encoder_layers = 3
graph_encoder_heads = 4
graph_edge_dim = 64
flow_matching_layers = 4
flow_matching_heads = 4
action_horizon = 50
max_obs_vertices = 5
learning_rate = 1e-4
warmup_steps = 1000
max_steps = 200000
checkpoint_every = 5000
dataset_root = ""
```

---

## Trainable Parameters (Stage 1)

| Module | Trainable |
|--------|-----------|
| SmolVLM2 | Frozen (LoRA in Stage 4) |
| SAM2 | Frozen |
| UAVPoseProjector | Yes |
| ObsProjector | Yes |
| PositionVertexBuilder.vt_proj | Yes |
| PositionNodeEncoder (cross-attn readout) | Yes |
| ObservationNodeEncoder.proj | Yes |
| GraphEncoder (HGTConv layers) | Yes |
| FlowMatchingNetwork | Yes |

---

## Key Literature

- **SmolVLA** (HuggingFace 2025): VLM + flow matching action expert, dual-stream attention, layer skipping, state projection pattern — primary architectural reference
- **Graphormer** (NeurIPS 2021): Industry standard graph transformer — spatial encoding, centrality encoding, edge encoding as attention biases
- **HGT** (WWW 2020): Heterogeneous Graph Transformer — implemented in PyG as `HGTConv`, handles multiple node/edge types with type-specific projections
- **TSGM** (CoRL 2022): Most direct navigation graph precedent — visited image nodes + object nodes, GCN encoder

---

## Core Training Submodules

1. **`graph_encoder.py`** — `GraphEncoder` with `PositionNodeEncoder` (Perceiver cross-attn), `ObservationNodeEncoder`, `HGTConv` layers, `[T, D_g]` readout
2. **`flow_matching.py`** — `FlowMatchingNetwork` with cross-attention to `z_graph [T, D_g]`
3. **`dataset.py`** — `AeroduoDataset` returning raw episode data, `collate_fn`
4. **`train.py`** — full training loop with online graph construction per episode

---

## Important Implementation Notes

- `num_workers=0` in dataloader — heavy models (SmolVLM2, SAM2, GroundingDINO) are not picklable
- Running mean accumulation of raw SAM2 features per category happens in training loop, NOT inside `ObservationVertexBuilder`
- `ObsProjector` applied ONCE per category AFTER accumulation
- Graph construction is differentiable through `obs_projector` and `vt_proj` — do NOT detach these tensors
- SmolVLA uses `resize_with_pad` to fixed 512×512 then calls `vision_model` directly — bypasses processor entirely. Your system uses the processor with `do_image_splitting=False` instead, which is cleaner and handles normalization automatically
- SmolVLA dual-stream attention (VLMWithExpert): prefix (image+lang+state) goes through VLM layers, suffix (noisy actions) goes through expert layers, joint self-attention during training (`forward_attn_layer`), cross-attention during inference (`forward_cross_attn_layer`). Your system separates graph encoding from flow matching rather than interleaving them

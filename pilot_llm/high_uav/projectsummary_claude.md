# AeroDuo Stage 1 — Implementation Reference

*Generated from the actual source code in `aeroduo/pilot_llm/high_uav/`.*

---

## Project Goal

Vanguard is a two-UAV collaborative navigation system. A **high-altitude UAV** observes the scene from above via BEV (bird's-eye-view) imagery and builds a spatiotemporal observation graph encoding what objects are visible and where, across a sliding window of T timesteps. A **low-altitude UAV** uses the graph embedding — together with a language-described navigation goal — to predict a trajectory toward the target object.

Stage 1 trains the high-UAV pipeline end-to-end: the graph encoder (BEV images → heterogeneous graph → `z_graph`) and the flow-matching trajectory predictor (conditioned on `z_graph`) are jointly supervised by the low-UAV ground-truth trajectory from the dataset. SmolVLM2 and SAM2 are frozen throughout.

---

## Overall System Architecture

```
Dataset: Hal-13k
  bevcamera/*.png          BEV frames (high UAV, T per episode)
  high_uav_traj.json       high UAV poses [N, 4]
  low_uav_traj.json        low UAV poses  [N, 4]  ← flow-matching target
  object_description.json  language instruction

                          ┌─────────────────────────────────────────────────┐
                          │  AeroDuoPolicy (nn.Module)                      │
                          │                                                 │
 BEV images [T PIL]  ───► │  BEVEncoder  (frozen, plain Python)             │
 instruction str     ───► │    GroundingDINO  — per-frame detection         │
                          │    SAM2 set_image_batch — one Hiera fwd (T frs) │
                          │    SAM2 predict — per-frame mask decode          │
                          │    → image_embeds [T,256,64,64]  (detached)     │
                          │    → masks_arrays  List[ndarray]                │
                          │    → detections_list List[List[dict]]           │
                          │                                                 │
 BEV images [T PIL]  ───► │  SmolVLM2Encoder  (frozen nn.Module)            │
 instruction str     ───► │    processor → token layout (image+lang+poses)  │
 high_poses [T,4]    ───► │    truncated LLaMA decoder (12/24 layers)       │
 low_poses  [T,4]    ───► │    pose_token_proj  ← TRAINABLE                 │
                          │    → hidden_states [T, S, 2048]                 │
                          │                                                 │
                          │  PositionVertexBuilder  (trainable)             │
                          │    PerceiverIO + output_query                   │
                          │    [T, S, 2048] → [T, D_g]  place nodes        │
                          │                                                 │
                          │  ObservationVertexBuilder  (trainable)          │
                          │    SAM2 mask-pool → per-category mean → project │
                          │    [T, K_max, D_g]  obs nodes                  │
                          │                                                 │
                          │  GraphEncoder  (trainable)                      │
                          │    HGTConv × 3  (heterogeneous graph)           │
                          │    position ←→ temporal ←→ position             │
                          │    position ←→ observes  ←→ observation         │
                          │    → z_graph [T, D_g]                          │
                          │                                                 │
 low_pose_current [4] ──► │  FlowMatchingNetwork  (trainable, Stage 1)      │
 x_tau [H,4]        ───► │    DiT denoiser (cross-attn to z_graph)         │
 tau scalar         ───► │    → v_pred [H, action_dim]                     │
                          │                                                 │
                          │  Loss: MSE(v_pred, clean − noise)               │
                          └─────────────────────────────────────────────────┘

Two-stage training plan
-----------------------
Stage 1 (this codebase):  Train BEV graph encoder + flow-matching network.
    Frozen:    SmolVLM2 body, SAM2, GroundingDINO.
    Trainable: pose_token_proj, PositionVertexBuilder, ObservationVertexBuilder,
               GraphEncoder, FlowMatchingNetwork.
    Target:    low_uav_traj_target from dataset (flow-matching MSE).

Stage 2:  Freeze graph encoder, train low-UAV navigation policy conditioned on
    the frozen z_graph, front-camera SmolVLM2 embedding, and low-UAV state.
```

---

## File Map

| File | Role | Trainable? |
|------|------|-----------|
| `train.py` | Training loop entry point | — |
| `config.py` | `AeroduoConfig` dataclass — single source of truth | — |
| `dataset.py` | `AeroduoDataset` + `collate_fn` | — |
| `aeroduo_policy.py` | `AeroDuoPolicy` — assembles all components | wrapper |
| `bev_encoder.py` | `BEVEncoder` — frozen GroundingDINO + SAM2 | No |
| `smolvlm2_encoder.py` | `SmolVLM2Encoder` + `UAVPoseProjector` | pose proj only |
| `position_vertex_builder.py` | `PositionVertexBuilder` — PerceiverIO | Yes |
| `observation_vertex.py` | `ObservationVertexBuilder` — SAM2 mask-pool | Yes |
| `graph_encoder.py` | `GraphEncoder` — HGTConv × 3 | Yes |
| `flow_matching.py` | `FlowMatchingNetwork` — DiT denoiser | Yes |

---

## Config (`config.py`)

`AeroduoConfig` is a `@dataclass` with no required arguments — all fields have defaults.

```
D_g                     = 1024       # shared graph node dim
smolvlm2_model_name     = HuggingFaceTB/SmolVLM2-2.2B-Instruct
smolvlm2_hidden_dim     = 2048       # LLaMA hidden size
vlm_layer_cutoff        = 12         # 24 layers → truncated to first 12
high_uav_pose_dim       = 5          # (x,y,z,sin_h,cos_h)
low_uav_state_dim       = 7          # placeholder in config; not used in Stage 1 — actual dataset has 4-dim (x,y,z,heading_rad)
sam2_feature_dim        = 256        # SAM2 FPN-neck channels
window_T                = 5          # BEV sliding window
action_dim              = 4          # (x,y,z,heading_rad)
action_horizon          = 8          # H future steps for flow matching
flow_matching_layers    = 4
flow_matching_heads     = 4
flow_matching_ffn_mult  = 4
num_timestep_buckets    = 1000
perceiver_M             = 64         # latent vectors
perceiver_D_latent      = 256
perceiver_depth         = 2
perceiver_n_heads       = 8
graph_encoder_layers    = 3
graph_encoder_heads     = 4
max_obs_vertices        = 5
mixed_precision         = bf16
smolvlm2_load_dtype     = bfloat16
lora_rank               = None       # set to int in Stage 4
```

Train overrides allowed via CLI: `--window_T`, `--action_horizon`, `--mixed_precision`.

---

## Dataset (`dataset.py`)

### `AeroduoDataset`

Scans `<dataset_root>/Carla_*/` directories at init time. For each episode directory:

1. Loads `object_description_with_help.json` → `instruction: str`
2. Loads `high_uav_traj.json["normalized_state"]` → `high_poses [N_high, 4]`
3. Loads `low_uav_traj.json["normalized_state"]` → `low_poses [N_low, 4]`
4. Sorts `bevcamera/*.png` → `bev_frame_paths [N_high]`
5. Validates `len(bev_frame_paths) == len(high_poses)` — skips mismatches
6. `n_overlap = min(N_high, N_low)`

**Sample index:** All `(episode_idx, t_end)` pairs where `t_end ∈ [T-1, n_overlap-H-1]` (inclusive). This is enumerated once at init — `__len__` reports the total across all episodes.

**`__getitem__(idx)` returns:**
```
bev_images           : List[PIL.Image]  length T  — loaded lazily
high_uav_poses       : np.ndarray [T, 4]
low_uav_poses_window : np.ndarray [T, 4]  — concurrent low-UAV pose per BEV frame
low_uav_pose_current : np.ndarray [4]     — == low_uav_poses_window[-1]
low_uav_traj_target  : np.ndarray [H, 4]  — low_poses[t_end+1 : t_end+1+H]
instruction          : str
episode_path         : str
window_start         : int
t_end                : int
```

### `collate_fn`

Supports `batch_size >= 1`. PIL images stay as nested Python lists `[B][T]`. All numpy arrays stacked into `[B, T, 4]` / `[B, 4]` / `[B, H, 4]` tensors via `torch.stack`.

**Constraint:** `num_workers=0` required — SAM2, SmolVLM2, GroundingDINO are not picklable across worker processes.

---

## Policy Assembly (`aeroduo_policy.py`)

### `AeroDuoPolicy(nn.Module)`

Owns all components. Init signature:
```python
AeroDuoPolicy(cfg: AeroduoConfig, sam2_predictor, grounding_model)
```

**Component registration:**
- `self.bev_encoder = BEVEncoder(sam2_predictor, grounding_model)` — **plain Python object, not `nn.Module`**. GroundingDINO and SAM2 never appear in `state_dict()` or `parameters()`.
- `self.vlm_encoder = SmolVLM2Encoder(cfg)` — frozen `nn.Module`. Registered so `.to(device)` works; optimizer never receives gradients through it.
- `self.place_node_builder = PositionVertexBuilder(...)` — trainable
- `self.obs_vertex_builder = ObservationVertexBuilder(cfg)` — trainable
- `self.graph_encoder = GraphEncoder(D_g, num_layers, heads)` — trainable
- `self.flow_net = FlowMatchingNetwork(cfg)` — trainable

### `encode_graph(bev_images, high_uav_poses, low_uav_poses_window, instruction, device) → [B, T, D_g]`

Accepts both single-episode (`str`, `[T,4]`) and batched (`List[str]`, `[B,T,4]`) — single inputs are promoted to `B=1`.

**Pipeline (per episode b, then batched):**
1. `bev_encoder(bev_images[b], instruction[b], device)` → `image_embeds_b [T,256,64,64]`, `masks_b`, `detections_b`
2. For each frame `t`: `vlm_encoder(bev_image, lang, low_state[b,t], high_state[b,t])` → `[1, S, 2048]`, squeezed to `[S, 2048]`
3. Stack: `all_hidden [B, T, S, 2048]` cast to trainable dtype
4. `place_node_builder(position_vertices)` → `place_nodes [B, T, D_g]`  (gradient tape starts here)
5. `obs_vertex_builder(sam2_predictor=None, masks, detections, device, image_embed=embed_b[t])` → `List[ObsVertex]` per frame → stacked to `obs_tensor [B, T, K_max, D_g]`
6. `graph_encoder(place_nodes, obs_tensor)` → `z_graph [B, T, D_g]`

### `forward(..., low_uav_traj_target=None) → Dict`

**Training** (with `low_uav_traj_target`):
```python
clean    = low_uav_traj_target  # [B, H, 4]
noise    = randn_like(clean)
tau      = rand(B).clamp(1e-4, 1-1e-4)  # [B]
x_tau    = (1 - tau[:,None,None]) * noise + tau[:,None,None] * clean
v_target = clean - noise
v_pred   = flow_net(z_graph, low_uav_pose_current, x_tau, tau)
loss     = F.mse_loss(v_pred, v_target)
return {"loss": loss, "z_graph": z_graph, "v_pred": v_pred, "tau": tau}
```

**Inference** (without `low_uav_traj_target`): returns `{"z_graph": z_graph}` only.

### Selective checkpointing

`trainable_state_dict()` / `load_trainable_state_dict()` operate over `_trainable_modules()`:
```python
("place_node_builder", self.place_node_builder)
("obs_vertex_builder", self.obs_vertex_builder)
("graph_encoder",      self.graph_encoder)
("flow_net",           self.flow_net)
("vlm_encoder.pose_token_proj", self.vlm_encoder.pose_token_proj)
```
Frozen models (SmolVLM2, SAM2, GroundingDINO) excluded — saves ~10–20 GB per checkpoint.

---

## BEV Encoder (`bev_encoder.py`)

**Not an `nn.Module`.** Holds references to frozen `sam2_predictor` and `grounding_model`.

### `BEVEncoder.__call__(bev_images [T PIL], instruction str, device) → (image_embeds, masks_arrays, detections_list)`

1. **Parse instruction** via `parse_instruction` + `build_prompt_list` → goal + contextual nouns → ordered `prompt_list`
2. **GroundingDINO per-frame:** For each of the T frames:
   - `_pil_to_gdino(pil_img)` — resize to 800px short-side (max 1333), normalize with ImageNet stats → `img_t [3, H', W']`
   - `_detect_per_prompt(img_t, ..., prompt_list)` → raw candidates
   - `_apply_nms(boxes, scores, NMS_IOU_THRESH)` → `kept` sorted: goal first, then prompt index, then confidence descending
3. **SAM2 batch encoding:** `sam2_predictor.set_image_batch(raw_images)` — one Hiera forward for all T frames → `_features["image_embed"] [T, 256, 64, 64]`; batch features cloned before mask decode
4. **SAM2 mask decoding per-frame:** Temporarily slice batch features to single-image views (`[1, 256, 64, 64]`), call `predict(box=boxes_px, multimask_output=False)` → `masks_t [N, H, W]` bool. Restore batch state after loop.
5. Returns: `embed_batch.to(device)` (detached, no grad), `masks_arrays`, `detections_list`

**Key detail:** Autocast context (`bfloat16` on Ampere+, `float16` on older) applied to both `set_image_batch` and `predict` calls. Entire `__call__` is `@torch.no_grad()`.

---

## SmolVLM2 Encoder (`smolvlm2_encoder.py`)

### `SmolVLM2Encoder(nn.Module)`

**Init:**
- Loads `HuggingFaceTB/SmolVLM2-2.2B-Instruct` with `torch_dtype=bfloat16` via `AutoModelForImageTextToText.from_pretrained`
- Sets `processor.image_processor.do_image_splitting = False` (global image only, no tiling)
- Freezes all VLM parameters: `requires_grad_(False)`
- **Physically truncates decoder:** `vlm.model.text_model.layers = layers[:12]` (N//2 = 12 out of 24)
- Creates `self.pose_token_proj = UAVPoseProjector(uav_pose_dim=5, vlm_hidden_dim=2048)` — **this is trainable**

### `build_processor_inputs(bev_image, language_text, device)`

Splits `language_text` on `"The description of the target..."` to extract direction prior and target description. Builds a structured prompt with `apply_chat_template`, calls `processor(text=prompt, images=[bev_image])` → returns `{input_ids, pixel_values, pixel_attention_mask, attention_mask}` on device.

### `forward(bev_image, lang_description, low_state [B,≥4], high_state [B,≥4], device) → [B, S+2, 2048]`

**Token layout fed to truncated decoder:**
```
[BOS, user-turn tokens]
[49189]        ← fake_image_token (boundary)
[49152]        ← global_image_token
[49190 × 81]   ← image placeholder tokens → replaced by actual image embeds
[49189]        ← fake_image_token (boundary)
[lang tokens ~100]
[49279, 198]   ← end-of-turn, newline
[high_state_token]  ← appended after merger
[low_state_token]   ← appended after merger
```

**Execution sequence:**
1. `pose_token_proj(low_state)` → `low_state_token [B, 1, 2048]`
2. `pose_token_proj(high_state)` → `high_state_token [B, 1, 2048]`
3. `build_processor_inputs` → `{input_ids, pixel_values, ...}`
4. `embed_tokens(input_ids)` → `[B, S, 2048]`
5. `embed_image(pixel_values, pixel_attention_mask)` → `[B, 81, 2048]` via `get_image_features`
6. `vlm.model.inputs_merger(input_ids, inputs_embeds, image_hidden_states)` → replace placeholder tokens with actual image embeds → `[B, S, 2048]`
7. `cat([merged, high_state_token, low_state_token], dim=1)` → `[B, S+2, 2048]`
8. Extend `attention_mask` by 2 ones
9. `vlm.model.text_model(inputs_embeds, attention_mask, use_cache=False)` → `lm_out.last_hidden_state`

**Note:** Executed under `@torch.no_grad()` indirectly (VLM params frozen). Gradients do flow through `pose_token_proj` (trainable). Causal attention: state tokens attend to all prior; image/language tokens do NOT attend back to state tokens.

### `UAVPoseProjector(nn.Module)`

```python
# Input: [B, >=4] — uses first 4 elements (x, y, z, heading_rad)
# _encode_heading: (x,y,z) + sin(h) + cos(h) → [B, 5]
# proj: Linear(5, 2048) → [B, 2048]
# output: unsqueeze(1) → [B, 1, 2048]
```

---

## Position Vertex Builder (`position_vertex_builder.py`)

### `PositionVertexBuilder(nn.Module)`

Trainable PerceiverIO that compresses `[T, S, 2048]` → `[T, D_g]` (or `[B, T, S, 2048]` → `[B, T, D_g]`).

**Components:**
- `self.perceiver = PerceiverIO(dim=2048, queries_dim=D_g, logits_dim=D_g, depth=2, num_latents=64, latent_dim=256, cross_heads=1, latent_heads=8)` from `perceiver-pytorch`
- `self.output_query = nn.Parameter(torch.randn(1, 1, D_g))` — single learned query per timestep

**Forward:**
- **Single `[T, S, dim]`:** expand query to `[T, 1, D_g]`, call `perceiver(position_vertices, queries)` → `[T, 1, D_g]` → squeeze → `[T, D_g]`
- **Batched `[B, T, S, dim]`:** reshape to `[B*T, S, dim]`, same single query expanded to `[B*T, 1, D_g]`, reshape output back → `[B, T, D_g]`

---

## Observation Vertex Builder (`observation_vertex.py`)

### `ObservationVertexBuilder(nn.Module)`

**Trainable component:** `self.obs_projector = nn.Sequential(nn.Linear(256, D_g), nn.LayerNorm(D_g))`

### `forward(sam2_predictor, masks_array [N,H,W], detections List[dict], device, image_embed [1,C,Hf,Wf]=None) → List[ObsVertex]`

1. Uses `image_embed` directly if provided (skips predictor lookup) — preferred path in `AeroDuoPolicy`
2. **Group by category:** For each detection `i`, call `_pool_mask(feat, masks_array[i])` → raw `[256]` vector. Accumulate into `category_vecs: dict[str, List[Tensor]]`. Goal category inserted first in `category_order`.
3. **Per unique category (up to `max_obs_vertices=5`):**
   - `stacked = torch.stack(category_vecs[cat])` → `[K_instances, 256]`
   - `mean_vec = stacked.mean(dim=0)` → `[256]`
   - `obs_projector(mean_vec.unsqueeze(0)).squeeze(0)` → `[D_g]`
   - Append `ObsVertex(feature=[D_g], category=str, is_goal=bool)`

### `_pool_mask(feat [1,C,Hf,Wf], mask_hw [H_img,W_img]) → [C]`

Resize mask to `[Hf, Wf]` via `F.interpolate(..., mode="nearest")`. If `n_true > 0`: `feat.squeeze(0)[:, mask_resized].mean(dim=1)`. Else: global mean fallback.

Feature map used: `image_embed [1, 256, 64, 64]` — the globally-contextualized Hiera trunk output, more semantic than `high_res_feats`.

---

## Graph Encoder (`graph_encoder.py`)

### `GraphEncoder(nn.Module)`

**Node types:** `position` (T nodes) and `observation` (K_active nodes).

**Edge types:**
- `(position, temporal, position)` — bidirectional between consecutive timesteps: both `t→t+1` and `t+1→t`
- `(position, observes, observation)` — position t → each observation node seen at t
- `(observation, observed_by, position)` — reverse: each obs node → the positions that saw it

**Components:**
- `self.convs = ModuleList([HGTConv(in_channels=D_g, out_channels=D_g, metadata=METADATA, heads=4) for _ in range(3)])`
- `self.norms_pos = ModuleList([LayerNorm(D_g) for _ in range(3)])`
- `self.norms_obs = ModuleList([LayerNorm(D_g) for _ in range(3)])`

### `_build_graph(place_nodes [T,D_g], observation_vertices [T,K_max,D_g]) → HeteroData`

1. `obs_mask = observation_vertices.abs().sum(dim=-1) > 0` → `[T, K_max]` bool
2. `obs_feats = observation_vertices[obs_mask]` → `[K_active, D_g]` (flattened active nodes)
3. **Temporal edges:** forward and backward `arange` concatenated → `edge_index [2, 2*(T-1)]`
4. **Observation edges:** iterate `(t, k)` in same order as `obs_mask.nonzero()`, assign global obs node index → `observes_ei [2, K_active]`, `observed_by_ei [2, K_active]`
5. If `K_active == 0`, insert dummy `[1, D_g]` zeros node so HGTConv sees consistent metadata

### Per-layer message passing:
```python
out = conv(x_dict, edge_index_dict)
x_dict['position']    = norms_pos[i](out['position']    + x_dict['position'])
x_dict['observation'] = norms_obs[i](out['observation'] + x_dict['observation'])
```
Returns `x_dict['position']` → `[T, D_g]`.

### Batched forward `[B, T, D_g]`

Builds B independent `HeteroData` graphs, fuses into one disconnected super-graph via `Batch.from_data_list(graphs)`, runs all HGTConv layers in a single kernel call, then reshapes position nodes back → `[B, T, D_g]`.

---

## Flow Matching Network (`flow_matching.py`)

### Architecture overview

GR00T-style DiT with joint `[state | action]` self-attention and cross-attention to `z_graph`.

**τ convention:** τ=0 → noise, τ=1 → clean. `x_τ = (1-τ)·noise + τ·clean`. `v_target = clean - noise`.

### Components

**`TimestepEncoder`:** `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=1)` → `TimestepEmbedding(256, D)`. Input: `LongTensor [B]` (τ bucketed into `num_timestep_buckets=1000`). Output: `[B, D]`.

**`AdaLayerNorm`:** `SiLU(temb)` → `Linear(D, 2D)` → `(scale, shift)`. Output: `LayerNorm(x) * (1 + scale[:,None]) + shift[:,None]`. No zero-init (stability from `proj_out_2` zero-init).

**`ActionEncoder`:** `W1: Linear(action_dim, D)` + `W2: Linear(2D, D)` + `W3: Linear(D, D)`. Fuses `x_tau` embedding with `temb` via `SiLU(W2(cat([W1(actions), temb_broadcast])))`. Handles both `[H, 4]` and `[B, H, 4]`.

**`FlowDenoiserBlock`:**
```
norm1 (AdaLN) → attn1 (bidirectional self-attn, cross_attn_dim=None) → residual
norm2 (AdaLN) → attn2 (cross-attn, K,V = context) → residual
norm3 (LayerNorm) → ff (FeedForward, gelu-approximate) → residual
```

**`_PoseProjector`:** `(x,y,z,sin_h,cos_h)` → `Linear(5, D)` → `[1, D]`. Used for low-UAV state token.

### `FlowMatchingNetwork(nn.Module)` — Init

```
timestep_encoder: TimestepEncoder(D)
action_encoder:   ActionEncoder(action_dim, D)
horizon_pos:      nn.Parameter [H, D]    — learned horizon positional embeddings
graph_pos:        nn.Parameter [T, D]    — learned graph positional embeddings
type_embed:       nn.Embedding(2, D)     — type 0 = state, type 1 = action
state_proj:       _PoseProjector(D)
blocks:           ModuleList of L=4 FlowDenoiserBlock
norm_out:         LayerNorm(D, elementwise_affine=False)
proj_out_1:       Linear(D, 2*D)         — DiT-style scale/shift
proj_out_2:       Linear(D, action_dim)  — zero-initialized
```

### `forward(z_graph, low_state, x_tau, tau) → [H, action_dim] or [B, H, action_dim]`

Normalises single-episode inputs to batched; remembers to squeeze on return.

1. `t_bucket = (tau * 1000).long()` → `temb = timestep_encoder(t_bucket)` → `[B, D]`
2. `context = z_graph + graph_pos[:T]` → `[B, T, D]`
3. `state_tok = (state_proj(low_state) + type_embed(zeros)).unsqueeze(1)` → `[B, 1, D]`
4. `action_toks = action_encoder(x_tau, temb) + horizon_pos[:H] + type_embed(ones)` → `[B, H, D]`
5. `sa = cat([state_tok, action_toks], dim=1)` → `[B, 1+H, D]`
6. For each block: `sa = block(sa, context, temb)`
7. `a = sa[:, 1:]` → `[B, H, D]`
8. DiT readout: `shift, scale = proj_out_1(SiLU(temb)).chunk(2)` → `v_pred = proj_out_2(norm_out(a) * (1+scale[:,None]) + shift[:,None])` → `[B, H, action_dim]`

---

## Training Loop (`train.py`)

### CLI arguments

| Arg | Default | Notes |
|-----|---------|-------|
| `--dataset_root` | required | path to Hal-13k |
| `--towns` | all Carla_* | restrict to list |
| `--window_T` | 5 | |
| `--action_horizon` | 8 | |
| `--batch_size` | 1 | num_workers always 0 |
| `--output_dir` | checkpoints/stage1 | |
| `--resume` | None | path to `trainable_state.pt` |
| `--checkpointing_steps` | 500 | |
| `--checkpoints_total_limit` | 3 | prune oldest |
| `--num_train_epochs` | 10 | |
| `--max_train_steps` | None | overrides epochs |
| `--gradient_accumulation_steps` | 4 | effective batch = 4 |
| `--mixed_precision` | bf16 | |
| `--learning_rate` | 3e-4 | |
| `--weight_decay` | 1e-4 | |
| `--lr_scheduler_type` | cosine | |
| `--num_warmup_steps` | 200 | |
| `--wandb_project` | None | disabled if omitted |

### Initialization sequence

```python
# 1. Accelerator (DDP + grad accumulation + mixed precision)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(gradient_accumulation_steps=N, mixed_precision="bf16", ...)

# 2. Load frozen vision models (module-level cache, safe to call once)
sam2_predictor, grounding_model, _ = load_models(device=device_str)

# 3. Config + Policy
cfg = AeroduoConfig(window_T=..., action_horizon=..., mixed_precision=...)
policy = AeroDuoPolicy(cfg, sam2_predictor, grounding_model)

# 4. Optimizer — trainable submodules only
trainable_params = [p for submodule in (
    policy.place_node_builder,
    policy.obs_vertex_builder,
    policy.graph_encoder,
    policy.flow_net,
    policy.vlm_encoder.pose_token_proj,
) for p in submodule.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-4)

# 5. Dataset + DataLoader (batch_size=1, num_workers=0)
train_dataset = AeroduoDataset(dataset_root, window_T, action_horizon, towns)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

# 6. LR scheduler
lr_scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=200, num_training_steps=max_steps)

# 7. accelerator.prepare(policy, optimizer, dataloader, lr_scheduler)
```

### Training loop

```python
for epoch in range(starting_epoch, num_train_epochs):
    policy.train()
    for batch in train_dataloader:
        with accelerator.accumulate(policy):
            out  = policy(**{k: v for k, v in batch.items() if k in _POLICY_KEYS},
                          device=accelerator.device)
            loss = out["loss"]
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:  # True only when optimizer step occurred
            completed_steps += 1
            # Log: loss, lr, tau.mean(), grad_norm → TensorBoard + wandb
            if completed_steps % checkpointing_steps == 0:
                _save_checkpoint(...)      # saves trainable_state_dict only
                _prune_checkpoints(...)    # keeps at most 3 step checkpoints
```

**`_POLICY_KEYS`** (keys forwarded to `AeroDuoPolicy.forward`):
```python
{"bev_images", "high_uav_poses", "low_uav_poses_window",
 "low_uav_pose_current", "low_uav_traj_target", "instruction"}
```

### Checkpoint format

`trainable_state.pt` contains:
```python
{
    "model":           policy.trainable_state_dict(),  # 5 submodules only
    "optimizer":       optimizer.state_dict(),
    "lr_scheduler":    lr_scheduler.state_dict(),
    "completed_steps": int,
    "epoch":           int,
}
```

**Resume:** `load_trainable_state_dict(ckpt["model"])` rebuilds only the trainable submodules. Frozen SmolVLM2 and SAM2 are re-loaded from HuggingFace/disk on each run.

### Logged metrics

| Metric | Description |
|--------|-------------|
| `Loss/train` | MSE flow-matching loss per optimizer step |
| `LR` | current learning rate |
| `tau` | mean τ value in the step |
| `grad_norm` | L2 norm of all trainable gradients |

---

## Trainable Parameter Summary

| Submodule | Parameters | Notes |
|-----------|-----------|-------|
| `vlm_encoder.pose_token_proj` | `Linear(5, 2048)` | UAVPoseProjector shared for both high+low state |
| `place_node_builder` | PerceiverIO + `output_query [1,1,D_g]` | Compresses VLM hidden states |
| `obs_vertex_builder.obs_projector` | `Linear(256, D_g)` + `LayerNorm` | Projects SAM2 pool vectors |
| `graph_encoder` | 3 × HGTConv + 6 × LayerNorm | Heterogeneous message passing |
| `flow_net` | TimestepEncoder, ActionEncoder, 4 × FlowDenoiserBlock, proj_out | DiT denoiser |

SmolVLM2 (all VLM layers), SAM2, and GroundingDINO are all frozen.

---

## Data Flow Shapes (single episode, B=1)

```
bev_images              List[PIL], T=5
high_uav_poses          [T=5, 4]
low_uav_poses_window    [T=5, 4]
low_uav_pose_current    [4]
low_uav_traj_target     [H=8, 4]

↓ BEVEncoder (frozen)
image_embeds            [T=5, 256, 64, 64]   detached
masks_arrays            List[ndarray [N,H,W] bool], T lists
detections_list         List[List[dict]], T lists

↓ SmolVLM2Encoder (frozen body, trainable pose_proj)
hidden_states           [T=5, S≈191, 2048]   (via loop per frame)

↓ PositionVertexBuilder (trainable)
place_nodes             [T=5, D_g=1024]

↓ ObservationVertexBuilder (trainable)
obs_tensor              [T=5, K_max=5, D_g=1024]   zero-padded

↓ GraphEncoder (trainable)
z_graph                 [T=5, D_g=1024]

↓ FlowMatchingNetwork (trainable) — during training only
v_pred                  [H=8, action_dim=4]
loss                    scalar MSE
```

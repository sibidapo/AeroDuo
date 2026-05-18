# BEV Image Segmentation Pipeline — Stage 1

Grounded-SAM2-based segmentation for the high-altitude UAV BEV frames.
This is the first stage of the spatiotemporal graph construction pipeline.

## Conda environment

```bash
conda activate /storage/project/r-cj124-0/sibidapo3/anxcnda/aeroduo
# or equivalently:
source activate /storage/project/r-cj124-0/sibidapo3/anxcnda/aeroduo
```

All scripts must be run from within this environment.

---

## Files

| File | Purpose |
|------|---------|
| `noun_extractor.py` | Parse instruction JSON → goal object + contextual noun list |
| `bev_segmentation.py` | Run Grounded-SAM2 on one BEV image, save structured output |
| `batch_segment.py` | Batch processor over an entire dataset directory |
| `final.ipynb` | Interactive verification notebook (in `aeroduo/pilot_llm/`) |

---

## Step 1 — Noun extraction

```bash
# Test on any instruction file
python noun_extractor.py /path/to/object_description_with_help.json

# Run self-tests against built-in samples
python noun_extractor.py
```

The extractor:
- Strips the compass/bearing preamble (splits on the fixed separator sentence)
- Extracts the **goal object** (first concrete noun in the description sentence)
- Extracts **contextual nouns** via spaCy noun chunks when available, with a rule-based fallback
- Filters against a blocklist of non-segmentable abstract terms
- Ranks contextual nouns by specificity and caps at 7
- Handles common Hal-13k openings such as `The ...`, `A ...`, and `In the image, a ...`

---

## Step 2 — Single-image segmentation

```bash
python bev_segmentation.py \
    /path/to/bevcamera/000000.png \
    /path/to/object_description_with_help.json \
    --output-dir /path/to/output \
    --device cuda
```

Saves three files per image:

| File | Content |
|------|---------|
| `<stem>_seg.json` | Structured segmentation metadata (category, bbox, confidence, goal flag) |
| `<stem>_masks.npz` | Compressed boolean masks; one key `mask_XXXX` per detection |
| `<stem>_vis.png` | Visualisation overlay with coloured masks, boxes, and labels |

Detection flow:
- GroundingDINO is run once per extracted noun prompt so each detection keeps a stable prompt-aligned category
- Cross-prompt duplicates are removed with NMS on the predicted boxes
- SAM2 then refines the surviving boxes into binary masks

### Output JSON schema

```json
{
  "image_path":    "<str>",
  "image_width":   512,
  "image_height":  512,
  "goal_category": "motorcycle",
  "goal_detected": true,
  "prompt_list": ["motorcycle", "bridge", "yellow lane markings"],
  "num_detections": 4,
  "detections": [
    {
      "id":         0,
      "category":   "motorcycle",
      "bbox_xyxy":  [x1, y1, x2, y2],
      "confidence": 0.82,
      "is_goal":    true,
      "source_prompt": "motorcycle",
      "prompt_index": 0,
      "grounding_phrase": "motorcycle",
      "mask_key":   "mask_0000",
      "dinov2_embedding": null
    }
  ],
  "masks_npz_path": "<stem>_masks.npz"
}
```

`dinov2_embedding` is `null` here; it will be populated in Stage 1a (DINOv2 feature extraction).
The mask arrays can be loaded with `np.load(masks_npz_path)[mask_key]`.

---

## Step 3 — Batch processing

```bash
# Process the full Hal-13k Town01 split
python batch_segment.py \
    /path/to/aeroduo/data/Hal-13k/Carla_Town01 \
    --output-dir /path/to/outputs/seg \
    --device cuda

# Quick test: first 3 episodes, 2 frames each
python batch_segment.py \
    /path/to/aeroduo/data/Hal-13k/Carla_Town01 \
    --max-episodes 3 \
    --frames-per-episode 2 \
    --device cuda
```

Outputs:
- One `<stem>_seg.json`, `<stem>_masks.npz`, and `<stem>_vis.png` per BEV frame
- A `batch_report.json` in the output root with dataset-level statistics
- The report includes overall goal-detection rate, average contextual detections per frame, and a list of images where detection failed or produced zero masks

## Notebook verification

Use [`final.ipynb`](/storage/project/r-cj124-0/sibidapo3/8750/aeroduo_ws/aeroduo/pilot_llm/final.ipynb) for the staged verification flow:
- environment and import checks in the provided conda env
- noun extraction sanity check on the canonical motorcycle/bridge instruction
- single-frame segmentation verification with saved JSON/NPZ/PNG artifacts
- a small batch run that emits the dataset summary fields needed for Stage 1

---

## Model paths

Models are loaded from fixed paths relative to the `Grounded-SAM-2` repo:

| Model | Path |
|-------|------|
| SAM2 checkpoint | `Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt` |
| SAM2 config | `configs/sam2.1/sam2.1_hiera_l.yaml` |
| GroundingDINO config | `Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py` |
| GroundingDINO checkpoint | `Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth` |

GroundingDINO also relies on the `bert-base-uncased` tokenizer from the local Hugging Face cache.
In this workspace the scripts run in offline mode, so that cache entry must already exist before inference.

---

## Downstream compatibility (Stage 1a — DINOv2)

Each `_seg.json` contains all the information needed by the DINOv2 feature
extraction stage:
- `image_path` — path to the BEV image on which to run DINOv2
- `masks_npz_path` — path to the binary masks for pooling features
- `detections[*].mask_key` — key to index into the NPZ file
- `detections[*].is_goal` — which detection gets the high goal-relevance score
- `detections[*].category` — category label used for MSG-style contrastive loss

The `dinov2_embedding` field in each detection is intentionally left `null` here
and will be filled in-place by the Stage 1a script.

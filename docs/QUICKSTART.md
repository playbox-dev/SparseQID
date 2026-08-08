# Quickstart: from nothing to a tracking video

This is the shortest path to seeing SparseQID work: one validation scene, 30
frames, rendered as MP4. It downloads about 2.6 GB — 2.3 GB of video for the
scene's 16 cameras plus a 328 MB checkpoint — and needs a CUDA GPU only for
step 5.

For the full paper workflow, see [REPRODUCE.md](REPRODUCE.md). For every option
of every command, see [CLI.md](CLI.md).

## 1. Install

```bash
uv sync --dev
uv run pytest
```

## 2. Download one scene

`sqid infer` fuses **all** cameras listed in `calibration.json`, so a scene has
to be downloaded whole — a partial camera set will fail when the loader looks
for the missing frames. Videos and calibration are enough; ground truth is not
needed to produce or visualize predictions.

```bash
export AICITY26_DATA=$PWD/data/MTMC_Tracking_2026

uvx --from huggingface-hub hf download \
  nvidia/PhysicalAI-SmartSpaces --repo-type dataset \
  --include "MTMC_Tracking_2026/val/Warehouse_020/videos/*" \
             "MTMC_Tracking_2026/val/Warehouse_020/calibration.json" \
  --local-dir "$PWD/data"
```

The dataset is public; no Hugging Face login is required.

## 3. Download a checkpoint

```bash
uvx --from huggingface-hub hf download playbox-dev/SparseQID \
  --include "checkpoints/paper/ngc-frozen_v128_pe30_it6k.pth" --local-dir .
```

This one file carries the detector *and* the identity model — the NGC base
checkpoint is not needed for inference.

## 4. Build a 30-frame cache

```bash
export AICITY26_CACHE=$PWD/data/aicity2026_frames_540

uv run sqid extract \
  --split val --scenes Warehouse_020 \
  --scale 0.5 --quality 90 --every 1 --num-frames 30
```

Decoding stops after 30 frames per camera, so this takes seconds. Dropping
`--num-frames` decodes all 9,000 frames of every camera and takes considerably
longer.

## 5. Run tracking

```bash
CUDA_VISIBLE_DEVICES=0 uv run sqid infer \
  --checkpoint checkpoints/paper/ngc-frozen_v128_pe30_it6k.pth \
  --scene Warehouse_020 --frames 30 --amp \
  --output outputs/quickstart/Warehouse_020.txt
```

This is the deterministic probe from REPRODUCE.md: it should report
**1,305 rows and 49 track IDs**. A different count means the frame cache was
built differently — most often by resizing already-encoded JPEGs instead of
resizing before encoding.

The result is an 11-column file, one row per box per frame:

```text
scene_id class_id object_id frame_id  x y z  w l h  yaw
```

## 6. Render it

```bash
uv run sqid visualize \
  --preds outputs/quickstart/Warehouse_020.txt \
  --scene Warehouse_020 --frames 0:30 \
  --cameras Camera_0000 Camera_0003 \
  --out outputs/quickstart/viz
```

That writes three clips into `outputs/quickstart/viz/`:

- `Warehouse_020_Camera_0000.mp4` and `..._Camera_0003.mp4` — the tracked boxes
  projected back into those camera views.
- `Warehouse_020_bev.mp4` — a top-down view of the whole scene, with a marker
  per camera and a short motion trail per track.

Every track gets its own colour, held for as long as the identity survives, so
an identity switch shows up as a box changing colour. Pass `--color-by class`
to colour by object class instead.

Add `--view camera` or `--view bev` to render only one of the two.

## What to look at

Camera views are the quickest check that detection and calibration are sane:
boxes should stand on the floor and sit on objects. Do not be alarmed by boxes
that appear to float over empty aisles — these warehouse scenes are heavily
occluded, and a box drawn on shelving is usually a real object that a *different*
camera can see. Rendering the ground truth through the same command shows the
same pattern.

The bird's-eye view is where tracking quality is visible: watch for a track
whose colour changes (an identity switch) or one that vanishes and reappears as
a new colour (a lost track).

## Next steps

- Full 9,000-frame scene: drop `--num-frames`/`--frames` from steps 4 and 5.
- Scoring: fetch `ground_truth.json` for the scene and use NVIDIA's official
  HOTA tool, per [REPRODUCE.md](REPRODUCE.md#8-hota-evaluation).
- The `test` split works identically — it has no ground truth, which inference
  does not need.

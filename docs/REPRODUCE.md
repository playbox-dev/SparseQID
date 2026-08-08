# Reproducing SparseQID

Run all commands from the repository root. Full training used eight A100 GPUs;
full-scene inference requires one CUDA-capable GPU.

This page is the end-to-end workflow. [CLI.md](CLI.md) documents every option
of `sqid extract`, `sqid train`, and `sqid infer` with its default.

## 1. Install

```bash
uv sync --dev
uv run pytest
```

No custom CUDA extension is required.

## 2. Download the dataset

SparseQID reads NVIDIA's
[PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces)
dataset (CC-BY-4.0). Access is not gated: no Hugging Face login or terms
acceptance is required. Two subsets of that repository are used:

| Subset | Splits used | Scenes |
|---|---|---|
| `MTMC_Tracking_2026` | `train`, `val`, `test` | `Warehouse_000`–`019` (train), `020`–`022` (val), 5 test scenes |
| `MTMC_Tracking_2025` | `train` | `Warehouse_000`–`014` |

The 2025 subset is only needed to reproduce training; inference and evaluation
use 2026 alone.

### What this repository actually reads

Each scene ships `videos/`, `calibration.json`, `ground_truth.json`, `map.png`,
and `depth_maps/`. SparseQID **never opens `map.png` or `depth_maps/`**, and
`depth_maps/` dominates the download size, so always filter it out. Download
only what your target workflow needs:

| Goal | Needs |
|---|---|
| Inference on `val`/`test` | `videos/`, `calibration.json` |
| HOTA evaluation of `val` predictions | the above + `ground_truth.json` |
| Training the identity model | `videos/`, `calibration.json`, `ground_truth.json` for every train scene |

Inference is ground-truth-free because it recenters the world frame on the mean
camera position (`recenter_mode="camera"`) rather than on ground-truth boxes.
Training needs `ground_truth.json` for the Hungarian query-to-identity matching.
`ground_truth.json` is large — roughly 256 MB for `Warehouse_020`.

### Download commands

Validation scenes (videos and calibration only):

```bash
uvx --from huggingface-hub hf download \
  nvidia/PhysicalAI-SmartSpaces --repo-type dataset \
  --include "MTMC_Tracking_2026/val/Warehouse_02[012]/videos/*" \
             "MTMC_Tracking_2026/val/Warehouse_02[012]/calibration.json" \
  --local-dir /path/to/datasets
```

Add `"MTMC_Tracking_2026/val/Warehouse_02[012]/ground_truth.json"` to that
`--include` list if you intend to score the predictions.

Full training data (2026 train, then 2025 train):

```bash
for subset in MTMC_Tracking_2026 MTMC_Tracking_2025; do
  uvx --from huggingface-hub hf download \
    nvidia/PhysicalAI-SmartSpaces --repo-type dataset \
    --include "$subset/train/*/videos/*" \
               "$subset/train/*/calibration.json" \
               "$subset/train/*/ground_truth.json" \
    --local-dir /path/to/datasets
done
```

`hf download` resumes, so an interrupted transfer can be re-run verbatim.

## 3. Data and model paths

Point the four environment variables at the subset roots you just downloaded
and at the two frame caches you are about to build. Every command below reads
these; there are no other implicit paths.

```bash
export AICITY26_DATA=/path/to/datasets/MTMC_Tracking_2026
export AICITY26_CACHE=/path/to/aicity2026_frames_540
export AICITY25_DATA=/path/to/datasets/MTMC_Tracking_2025
export AICITY25_CACHE=/path/to/aicity2025_frames_540
```

`$AICITY26_DATA` must be the directory that directly contains `train/`, `val/`,
and `test/`:

```text
$AICITY26_DATA/
└── val/
    └── Warehouse_020/
        ├── calibration.json          # required
        ├── ground_truth.json         # training and evaluation only
        └── videos/
            ├── Camera_0000.mp4
            ├── Camera_0001.mp4
            └── ...                   # numbering is not contiguous
```

Each `.mp4` stem must equal the matching camera `id` in `calibration.json`
(`Camera_0000.mp4` ↔ `"id": "Camera_0000"`). The published dataset already
satisfies this; it only matters if you rename files or supply your own scenes.

Obtain the public `nvidia/tao/sparse4d_rn101:trainable_v2.2` checkpoint from
NGC for training. SparseQID checkpoints contain the detector and do not need
that separate base checkpoint at inference time.

Download the paper weights with:

```bash
uvx --from huggingface-hub hf download playbox-dev/SparseQID \
  --include "checkpoints/**" --local-dir .
```

## 4. Frame caches

The models never read MP4s directly. `sqid extract` decodes each video once,
resizes by `--scale`, and writes JPEGs to the cache root; training and
inference then read only those JPEGs.

Inference uses every source frame, resized to 540×960 before JPEG encoding at
quality 90:

```bash
uv run sqid extract \
  --split val \
  --scenes Warehouse_020 Warehouse_021 Warehouse_022 \
  --scale 0.5 --quality 90 --every 1 --num-frames 9000 \
  --data-root "$AICITY26_DATA" --cache-root "$AICITY26_CACHE"
```

The submitted training recipe used the same resize/encoding order with
`--every 3` for the 2025 and 2026 train scenes. Resizing an already encoded
1080p JPEG is not numerically equivalent.

```bash
uv run sqid extract --split train \
  --scenes $(printf 'Warehouse_%03d ' $(seq 0 19)) \
  --scale 0.5 --quality 90 --every 3 \
  --data-root "$AICITY26_DATA" --cache-root "$AICITY26_CACHE"

uv run sqid extract --split train \
  --scenes $(printf 'Warehouse_%03d ' $(seq 0 14)) \
  --scale 0.5 --quality 90 --every 3 \
  --data-root "$AICITY25_DATA" --cache-root "$AICITY25_CACHE"
```

`extract` defaults to the 2026 roots, so the 2025 pass needs both `--data-root`
and `--cache-root` given explicitly.

The result mirrors the dataset layout, one directory per camera, named by the
source video stem, with files named by **source** frame index:

```text
$AICITY26_CACHE/
└── val/
    └── Warehouse_020/
        ├── Camera_0000/
        │   ├── 000000.jpg
        │   ├── 000001.jpg          # 000000, 000003, ... when --every 3
        │   └── ...
        └── Camera_0001/
```

Because filenames keep the source index, `--every 3` makes one cached step
equal three source frames. The training defaults `--interval-min 1
--interval-max 4` are measured in *cached* frames, so on an `--every 3` cache a
30-frame clip spans 3–12 source frames per step.

Three properties of the cache are worth knowing before you build it:

- **One resolution per cache root and split.** The dataset infers the source
  image size from the first JPEG it finds under `<cache-root>/<split>` and
  applies it to every scene's projection matrices. Mixing scales inside one
  cache root silently skews the projections. Use a separate cache root per
  `--scale`.
- **`extract` skips files that already exist**, so re-running it after an
  interruption is cheap — but changing `--scale` or `--quality` will not
  rewrite existing JPEGs. Delete the scene directory first.
- **Frame discovery reads the first camera directory only.** All cameras of a
  scene must be extracted with the same `--every` and `--num-frames`.

## 5. Training

The submitted identity model used:

- 2026 train scenes 000–019 and 2025 train scenes 000–014;
- 30-frame recurrent clips with a random cached-frame interval from 1 to 4;
- six random relative-ID target groups, 0.5 occlusion probability, and 0.5
  switch probability;
- a 128-entry identity vocabulary and Fourier 3D position tokens;
- a frozen public NGC recurrent detector;
- AdamW at `4e-4`, gradient clipping at `1.0`, and 6,000 iterations;
- eight data-parallel processes with gradients averaged after every clip.

Those are the command defaults:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc-per-node=8 --no-python \
  sqid train \
  --base-checkpoint /path/to/sparse4d_warehouse_v2.2_r101.pth \
  --output checkpoints/sparseqid.pth
```

Inspect the resolved recipe without data or a GPU:

```bash
uv run sqid train --base-checkpoint base.pth --dry-run
```

With one extracted training scene, a minimal training check is:

```bash
CUDA_VISIBLE_DEVICES=0 uv run sqid train \
  --base-checkpoint /path/to/sparse4d_warehouse_v2.2_r101.pth \
  --scenes Warehouse_000 --clip-length 2 --iterations 1 \
  --checkpoint-every 0 --output outputs/train-smoke.pth
```

The submitted checkpoint includes one detector warm-up update before freezing;
its detector weights differ from the NGC base by at most `9.54e-7`. The trainer
freezes the detector from the first step. To train the identity layer against
the submitted detector state, pass the paper checkpoint as `--base-checkpoint`.

## 6. Deterministic inference probe

```bash
CUDA_VISIBLE_DEVICES=0 uv run sqid infer \
  --checkpoint checkpoints/paper/ngc-frozen_v128_pe30_it6k.pth \
  --data-root "$AICITY26_DATA" --cache-root "$AICITY26_CACHE" \
  --scene Warehouse_020 --frames 30 --amp \
  --output outputs/smoke/Warehouse_020.txt
```

The native-540 reference produces 1,305 rows and 49 global IDs.

## 7. Full validation inference

```bash
for scene in Warehouse_020 Warehouse_021 Warehouse_022; do
  CUDA_VISIBLE_DEVICES=0 uv run sqid infer \
    --checkpoint checkpoints/paper/ngc-frozen_v128_pe30_it6k.pth \
    --data-root "$AICITY26_DATA" --cache-root "$AICITY26_CACHE" \
    --scene "$scene" --frames 9000 --amp \
    --output "outputs/paper_val/$scene.txt"
done
```

Defaults are the submitted settings: score threshold 0.4, 2.5 m spatial gate,
0.8 m newborn suppression, combined appearance/Fourier-position tokens, and
object-max assignment.

## 8. HOTA evaluation

SparseQID does not ship an evaluator. Use NVIDIA’s official offline 3D-box HOTA
tool, [`evaluate_aicity_mtmc.py`](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/tree/develop/libs/analytics/spatialai-data-utils/tools/evaluation),
following the current
[Physical AI Smart Spaces dataset documentation](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/blob/main/README.md).

Expected per-scene camera-ready results:

| Scene | HOTA | DetA | AssA | LocA |
|---|---:|---:|---:|---:|
| Warehouse_020 | 45.90 | 49.48 | 44.50 | 76.61 |
| Warehouse_021 | 7.15 | 10.21 | 5.48 | 70.15 |
| Warehouse_022 | 14.38 | 28.05 | 7.54 | 46.12 |

The five-scene test ground truth is private, so the official test score can
only be verified on the challenge evaluation service.

# Reproducing SparseQID

Run all commands from the repository root. Full training used eight A100 GPUs;
full-scene inference requires one CUDA-capable GPU.

## 1. Install

```bash
uv sync --dev
uv run pytest
```

No custom CUDA extension is required.

## 2. Data and model paths

```bash
export AICITY26_DATA=/path/to/MTMC_Tracking_2026
export AICITY26_CACHE=/path/to/aicity2026_frames_540
export AICITY25_DATA=/path/to/MTMC_Tracking_2025
export AICITY25_CACHE=/path/to/aicity2025_frames_540
```

The 2026 root must contain `train/`, `val/`, and `test/`. Obtain the public
`nvidia/tao/sparse4d_rn101:trainable_v2.2` checkpoint from NGC for training.
SparseQID checkpoints contain the detector and do not need that separate base
checkpoint at inference time.

Download the paper weights with:

```bash
uvx --from huggingface-hub hf download playbox-dev/SparseQID \
  --include "checkpoints/**" --local-dir .
```

## 3. Frame caches

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

## 4. Training

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

## 5. Deterministic inference probe

```bash
CUDA_VISIBLE_DEVICES=0 uv run sqid infer \
  --checkpoint checkpoints/paper/ngc-frozen_v128_pe30_it6k.pth \
  --data-root "$AICITY26_DATA" --cache-root "$AICITY26_CACHE" \
  --scene Warehouse_020 --frames 30 --amp \
  --output outputs/smoke/Warehouse_020.txt
```

The native-540 reference produces 1,305 rows and 49 global IDs.

## 6. Full validation inference

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

## 7. HOTA evaluation

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

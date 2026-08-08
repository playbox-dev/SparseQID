# Command reference

`sqid` has three subcommands. Every option is listed below with its default;
`sqid <command> --help` prints the same set.

```text
sqid extract     build the JPEG frame cache from the dataset videos
sqid train       train the identity model on a frozen recurrent detector
sqid infer       run detector + identity assignment and write a submission
sqid visualize   render a predictions file as MP4 clips
```

Paths default to the environment variables described in
[REPRODUCE.md](REPRODUCE.md#3-data-and-model-paths); the tables show the
fallback used when a variable is unset.

## `sqid extract`

Decodes each camera video once and writes JPEGs to
`<cache-root>/<split>/<scene>/<camera>/<source frame index, 6 digits>.jpg`.
Existing files are skipped, so an interrupted run can be resumed by repeating
the command.

| Option | Default | Meaning |
|---|---|---|
| `--scenes` | *required* | One or more scene names, e.g. `Warehouse_020 Warehouse_021`. |
| `--split` | `val` | Dataset split directory to read and to mirror in the cache. |
| `--every` | `1` | Keep every N-th source frame. The paper used `1` for inference, `3` for training. |
| `--scale` | `0.5` | Resize factor applied before JPEG encoding. `0.5` turns 1080p into the 540×960 the models expect. |
| `--quality` | `90` | JPEG quality. |
| `--num-frames`, `--n-frames` | `9000` | Upper bound on source frames read per camera. |
| `--data-root`, `--root` | `$AICITY26_DATA`, else `data/MTMC_Tracking_2026` | Dataset subset root containing `train/`, `val/`, `test/`. |
| `--cache-root`, `--cache` | `$AICITY26_CACHE`, else `data/aicity2026_frames_540` | Destination for the JPEG cache. |

Changing `--scale` or `--quality` does not rewrite existing JPEGs — delete the
scene directory first. Keep one cache root per resolution.

## `sqid train`

Trains the identity modules (trajectory memory, ID decoder, position encoder)
while the detector stays frozen and runs under `no_grad`. Launch under
`torchrun` for multi-GPU; gradients are averaged after every clip.

### Data and output

| Option | Default | Meaning |
|---|---|---|
| `--base-checkpoint` | *required* | NGC `sparse4d_rn101:trainable_v2.2` checkpoint, or a SparseQID checkpoint to train against that detector state. |
| `--output` | `checkpoints/sparseqid.pth` | Destination checkpoint. |
| `--scenes` | 35 paper scenes | Scene list. A `2025:` prefix routes a scene to the 2025 roots, e.g. `2025:Warehouse_003`. Unprefixed names use the 2026 roots. |
| `--split` | `train` | Split directory, applied to both years. |
| `--data-root` | `$AICITY26_DATA`, else `data/MTMC_Tracking_2026` | 2026 dataset root. |
| `--cache-root` | `$AICITY26_CACHE`, else `data/aicity2026_frames_540` | 2026 JPEG cache. |
| `--data-root-2025` | `$AICITY25_DATA`, else `data/MTMC_Tracking_2025` | 2025 dataset root, used by `2025:`-prefixed scenes. |
| `--cache-root-2025` | `$AICITY25_CACHE`, else `data/aicity2025_frames_540` | 2025 JPEG cache. |
| `--config-dir` | packaged `configs/sparse4d_rn101_v2.2` | Detector architecture config and anchor `.npy`. |

Every training scene needs `ground_truth.json`: the Hungarian matcher pairs
detector queries with ground-truth track IDs to build identity targets.

### Schedule

| Option | Default | Meaning |
|---|---|---|
| `--iterations` | `6000` | Total clips. One iteration is one clip. |
| `--checkpoint-every` | `500` | Periodic save interval; `0` disables intermediate saves. |
| `--learning-rate` | `4e-4` | AdamW learning rate (weight decay is fixed at `1e-3`). |
| `--gradient-clip` | `1.0` | Global gradient-norm clip. |
| `--seed` | `0` | Torch seed; the Python RNG is seeded `--seed + rank`. |
| `--device` | `cuda:0` if available, else `cpu` | Ignored under `torchrun`, which uses `LOCAL_RANK`. |
| `--no-amp` | off | Disable bfloat16 autocast in the detector forward pass. |
| `--dry-run` | off | Print the resolved arguments as JSON and exit — no data or GPU needed. |

### Clip sampling and identity targets

| Option | Default | Meaning |
|---|---|---|
| `--clip-length` | `30` | Frames per clip. Also sets the decoder's relative-position table length, and therefore the tracker's memory window at inference. |
| `--interval-min` | `1` | Minimum stride between clip frames, in **cached** frames. |
| `--interval-max` | `4` | Maximum stride, sampled uniformly per clip. |
| `--augmentation-groups` | `6` | Independent random ID-slot permutations per clip. This is what forces in-context copying instead of memorised identities. |
| `--vocabulary-size` | `128` | Number of relative ID slots; the decoder predicts these plus one newborn token. |
| `--occlusion-probability` | `0.5` | Per (group, track) chance of blanking a random contiguous span. |
| `--switch-probability` | `0.5` | Per-frame chance of swapping matched detections across tracks. |
| `--position-encoding` | `fourier` | `fourier` (parameterless multi-frequency sin/cos), `raw` (`Linear(3, 256)`), or `mlp` (two-layer). Recorded in the checkpoint as `pos_enc_mode`. |

## `sqid infer`

Runs the detector and tracker over one scene and writes the 11-column
submission format.

### Inputs and outputs

| Option | Default | Meaning |
|---|---|---|
| `--checkpoint`, `--ckpt` | *required* | SparseQID checkpoint; contains both detector and identity weights. |
| `--scene` | *required* | Single scene name. |
| `--split` | `val` | Split directory. |
| `--frames` | `9000` | Maximum cached frames to process. |
| `--data-root` | `$AICITY26_DATA`, else `data/MTMC_Tracking_2026` | Dataset root; only `calibration.json` is read. |
| `--cache-root` | `$AICITY26_CACHE`, else `data/aicity2026_frames_540` | JPEG cache root. |
| `--config-dir` | packaged config | Detector architecture config. |
| `--output` | `outputs/predictions/<scene>.txt` | Submission path. Two siblings are written next to it: `<output>.scores` (one detector confidence per row) and `<output stem>.json` (a run summary). |
| `--scene-id` | digits parsed from `--scene` | Value written in the submission's first column; `Warehouse_020` yields `20`. |
| `--device` | `cuda:0` | Pass `cpu` for a CPU smoke run. |
| `--amp` | off | bfloat16 CUDA autocast. Used for the published results. |

Inference needs no `ground_truth.json` — the world frame is recentred on the
mean camera position.

### Detection and identity thresholds

Defaults are the submitted settings; changing them changes the reported metrics.

| Option | Default | Meaning |
|---|---|---|
| `--score-threshold`, `--score` | `0.4` | Minimum detector confidence for a query to enter the tracker. |
| `--identity-threshold` | `0.2` | Minimum softmax score to accept a predicted ID instead of declaring a newborn. |
| `--newborn-threshold` | `0.6` | Detector confidence a newborn must clear to start a track. Detections below it that match no existing ID are dropped. |
| `--spatial-gate` | `2.5` | Metres. Zeroes the ID score for candidates farther than this from a track's last known position. The radius grows by 0.15 m per frame missed. |
| `--newborn-suppression` | `0.8` | Metres. Greedily re-attaches a would-be newborn to an unclaimed dormant track within this radius, which also grows with the gap. |
| `--assignment` | `object-max` | `object-max` gives each ID to its single highest-scoring detection; `hungarian` solves a global assignment over a padded score matrix. |
| `--token-mode` | `combined` | Which tokens feed the identity model: `combined` (detector query + 3D position), `appearance` (query only), or `position` (position only). |

`--token-mode` must match how the checkpoint was trained. The paper model and
all `ngc-*` checkpoints use `combined`; the `tok-appearance` checkpoint needs
`appearance`, and `tok-pos-fourier` / `tok-pos-raw` need `position`. See
[CHECKPOINTS.md](CHECKPOINTS.md).

To disable a geometric constraint for the ablation, pass a very large value
(for example `--spatial-gate 1e9`); the gate is always active with its default.

## `sqid visualize`

Renders a predictions file written by `sqid infer` as MP4 clips. It reads the
JPEG frame cache and `calibration.json` — no checkpoint, no GPU, and no ground
truth. [QUICKSTART.md](QUICKSTART.md) walks through it end to end.

| Option | Default | Meaning |
|---|---|---|
| `--preds` | *required* | 11-column predictions file. Any file in that format works, so ground truth converted to it renders the same way. |
| `--scene` | *required* | Scene name, used to locate calibration, cached frames, and to name the output clips. |
| `--split` | `val` | Split directory. |
| `--data-root` | `$AICITY26_DATA`, else `data/MTMC_Tracking_2026` | Dataset root; only `calibration.json` is read. |
| `--cache-root` | `$AICITY26_CACHE`, else `data/aicity2026_frames_540` | JPEG cache built by `sqid extract`. |
| `--out` | `outputs/viz` | Output directory. |
| `--frames` | `0:900` | Source frame range `START:END` or `START:END:STEP`; `END` is exclusive. Only frames present in the cache are rendered. |
| `--cameras` | first camera of the scene | Camera ids to render. One MP4 per camera. |
| `--view` | `both` | `camera` for projected overlays, `bev` for the top-down panel, `both` for each. |
| `--color-by` | `id` | `id` gives every track its own stable colour, so an identity switch appears as a colour change. `class` colours by object class instead. |
| `--fps` | real-time | Output frame rate. The default divides 30 by the cache stride, so an `--every 3` cache plays back at 10 fps rather than triple speed. |
| `--bev-size` | `900` | Longest side of the bird's-eye image, in pixels. |

Outputs are `<scene>_<camera>.mp4` per requested camera and `<scene>_bev.mp4`
for the top-down view.

Boxes whose corners all fall behind a camera, or entirely outside its frame,
are skipped for that view; the per-camera line printed on completion reports how
many were actually drawn. Because the cache holds resized frames, the projection
is scaled to the cached image size — so the clips match whatever `--scale` the
cache was built with.


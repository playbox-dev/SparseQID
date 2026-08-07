# Checkpoints

The paper uses seven checkpoints. Download them from
[playbox-dev/SparseQID on Hugging Face](https://huggingface.co/playbox-dev/SparseQID).

## Paper checkpoints

| File | Iteration | Detector / ID training | Paper use |
|---|---:|---|---|
| `ngc-frozen_v128_pe30_it6k.pth` | 6,000 | NGC detector frozen; query + Fourier 3D position ID tokens | Submitted model; NGC decoupled row; geometric-constraint ablations; deployed combined-token reference |
| `ngc-joint_v128_pe30_it5k.pth` | 5,000 | Continued joint detector--ID training from the NGC lineage | NGC joint row |
| `ngc-ft4k-frozen_v128_pe30_it8k.pth` | 8,000 | 4k-finetuned detector frozen during ID training | Finetuned-detector decoupled row |
| `ngc-ft4k-joint_v128_pe30_it2k.pth` | 2,000 | Continued joint detector--ID training from the finetuned lineage | Finetuned-detector joint row |
| `tok-appearance_v128_pe30_it2k.pth` | 2,000 | Frozen NGC detector; detector-query token only | Matched token ablation: detector query / no position encoding |
| `tok-pos-fourier_v128_pe30_it2k.pth` | 2,000 | Frozen NGC detector; 3D position token only | Matched token ablation: Fourier position |
| `tok-pos-raw_v128_pe30_it2k.pth` | 2,000 | Frozen NGC detector; 3D position token only | Matched token ablation: learned linear position |

All seven checkpoints contain `backbone_state`, `neck_state`,
`head_state`, `trajectory_modeling`, and `id_decoder` state dictionaries. The
token-ablation checkpoints also record `pos_enc` and `pos_enc_mode`.

## Verification

The following checks used `ngc-frozen_v128_pe30_it6k.pth` unless noted
otherwise.

| Check | Result |
|---|---|
| PyTorch loading | Pass, 7/7 |
| Strict detector-head loading | Pass, no missing or unexpected keys |
| Checkpoint metadata | Pass: iteration 6,000, vocabulary 128, six ID-decoder layers, relative-position window 30 |
| W020 two-frame inference | Pass: 90 detections, 48 track IDs, valid 11-column submission |
| W020 two-frame reference match | Pass: 90/90 rows match byte-for-byte |
| W020 30-frame inference | Pass: 1,305/1,305 rows match byte-for-byte |
| One-iteration training test | Pass: trained, saved, and loaded a checkpoint using a two-frame clip |
| Saved 9,000-frame validation predictions | W020 45.90/49.48/44.50, W021 7.15/10.21/5.48, W022 14.38/28.05/7.54 (HOTA/DetA/AssA) |
| Full validation inference | Not yet rerun from this release |
| Five-scene challenge test | Ground truth is private; use the challenge server |

The 30-frame test covers the decoder's complete relative-position window and
checks preprocessing, recurrent detection, identity assignment, and submission
writing at the beginning of W020.

### Frame preprocessing

The camera-ready validation predictions used a native-540 JPEG cache: decode
each source MP4 frame, resize it by 0.5 to 540x960, and encode it as JPEG at
quality 90. Resizing an already encoded 1080p JPEG produced 86 rather than 90
above-threshold detections in the two-frame test.

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

### Frame preprocessing

The camera-ready validation predictions used a native-540 JPEG cache: decode
each source MP4 frame, resize it by 0.5 to 540x960, and encode it as JPEG at
quality 90. 

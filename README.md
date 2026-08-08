# SparseQID

**Online multi-camera 3D tracking by ID prediction over recurrent sparse queries.**

SparseQID combines a recurrent outside-in Sparse4D detector with an online
identity-prediction module. The detector fuses synchronized, calibrated cameras
into world-frame 3D observations. `SparseQIDTracker` assigns persistent
scene-global IDs using the detector's query features and a finite trajectory
memory.

The system placed third in Track 1 of the 2026 AI City Challenge. Its identity
layer raised test HOTA from 29.63 to 38.01 over native detector identities,
primarily through an AssA increase from 20.83 to 31.10.

## Install and test

Python 3.11 or 3.12 and a CUDA-capable GPU are recommended.

```bash
uv sync --dev
uv run pytest
```

The package exposes one command with three subcommands:

```text
sqid extract     build the JPEG frame cache
sqid train       train the identity model with a frozen recurrent detector
sqid infer       run detector, identity assignment, and submission writing
```

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the complete data preparation,
training, inference, and submission workflow.

## Checkpoints

Paper weights are hosted at
[playbox-dev/SparseQID on Hugging Face](https://huggingface.co/playbox-dev/SparseQID).
Their experimental roles and test results are listed in
[docs/CHECKPOINTS.md](docs/CHECKPOINTS.md). Download them with:

```bash
uvx --from huggingface-hub hf download playbox-dev/SparseQID \
  --include "checkpoints/**" --local-dir .
```

Each checkpoint contains the backbone, neck, detector head, trajectory memory,
identity decoder, and position encoder needed for inference.

## Evaluation

SparseQID does not redistribute or wrap an evaluator. NVIDIA’s current
[Physical AI Smart Spaces dataset documentation](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/blob/main/README.md)
points to the official offline 3D-box HOTA implementation,
[`evaluate_aicity_mtmc.py`](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/tree/develop/libs/analytics/spatialai-data-utils/tools/evaluation).

## Citation

```bibtex
@inproceedings{shrestha2026sparseqid,
  title     = {Online Multi-Camera 3D Tracking via ID Prediction over Recurrent Sparse Queries},
  author    = {Shrestha, Pragyan and Nakayama, Haruto and Scott, Atom},
  booktitle = {ECCV Workshops},
  year      = {2026}
}
```

Please also cite the MOTIP and outside-in Sparse4D papers when using their
corresponding components.

## License

SparseQID includes code adapted from
[NVIDIA TAO Sparse4D](https://github.com/NVIDIA/tao_pytorch_backend) and
[MOTIP](https://github.com/MCG-NJU/MOTIP). Original copyright headers are
retained in adapted files.

See [LICENSE](LICENSE) and [NOTICE](NOTICE).

# Reproducibility

## Hardware

| Component | Used in paper | Minimum required |
|---|---|---|
| GPU | 2× NVIDIA H100 80 GB HBM3 PCIe | Same. Single-A100-40 GB has not been validated. |
| RAM | 256 GB | ≥64 GB (for SemanticKITTI dataset caching) |
| Disk | ~1 TB SSD | ~300 GB SSD (eval-only: 135 GB) |
| OS | Ubuntu 22.04 + CUDA 12.8 | Linux + CUDA 12.x |

## Exact software environment

```
Python:         3.10.14 or 3.11.x
PyTorch:        2.4.0
CUDA:           12.8
spconv:        2.3.8 (built for cu126, with our kernel-shape patches)
NumPy:          1.26.x (NOT 2.x — spconv v2.3 incompat)
```

The exact pin set is in `uv.lock`. Reproduce with:

```bash
uv venv --python 3.10
uv sync
uv pip install spconv-cu126==2.3.8
```

## Random seeds

The headline run uses **seed 42** for both data shuffling and parameter
initialization. The seed is passed via `--seed 42` (default in
`scripts/train.py`).

We use a single seed throughout the paper. **This matches the prevailing
SemanticKITTI SSC reporting convention** — every method in `tab:main_results`
(LMSCNet, SSA-SC, JS3C-Net, SCPNet, TALoS) likewise reports a single
test-server entry per configuration with no variance bars, because each full
training run at 256×256×32 voxel resolution costs roughly 37 GPU-hours on
2×H100.

## Per-table reproduction

| Table / Figure | Command | Expected |
|---|---|---|
| Tab. I (test mIoU) | `python scripts/infer.py infer/test_d4tta --checkpoint data/checkpoints/gssc_31k_mf_step40000.safetensors --output preds/test/` then submit to SemanticKITTI Codabench | 39.2 mIoU, 59.0 IoU_cmpl |
| Tab. II (val per-class) | `python scripts/eval.py eval/val_1step --checkpoint data/checkpoints/gssc_31k_mf_step40000.safetensors --metrics miou per_class` | 38.54 mIoU |
| Tab. III (safety metrics) | `python scripts/eval.py eval/val_1step --checkpoint ... --metrics safety` | SC-mIoU 35.2, VRU-IoU 19.6 |
| Tab. V (step reduction) | `python scripts/eval.py eval/step_sweep --checkpoint ...` | 38.54 (N=1), 38.59 (N=2), 38.65 (N=4), 38.16 (N=100) |
| Tab. VII (data scaling) | Per-row checkpoint, e.g. `python scripts/eval.py eval/val_1step --checkpoint data/checkpoints/gssc_31K_sf_step100000.safetensors` | See MODEL_ZOO.md |
| Tab. VIII (DW-IoU) | `python scripts/eval.py eval/val_1step --checkpoint ... --metrics dwiou` | per-T_w table |
| Tab. XII (training timesteps) | Per-row checkpoint | T=10: 37.83, T=50: 37.92, T=100-skewed: 38.18, T=100-uniform: 38.54 |
| Tab. XV (BEV) | `python scripts/eval.py eval/bev_secondary --checkpoint data/checkpoints/bev_perception_net.safetensors` | 36.09 BEV mIoU |
| Fig. 4 / Fig. 5 (qualitative) | See `examples/` notebooks | — |

All commands assume `data/checkpoints/` and `data/scpnet_predictions/` already exist (run `scripts/download_assets.py --checkpoints --predictions`).

## Determinism caveats

* PyTorch's CUDNN backend is stochastic by default. We disable nondeterminism
  in evaluation with `torch.backends.cudnn.deterministic = True`. Training
  retains nondeterministic CUDNN for speed; multiple training runs from the
  same seed will produce *very similar* but not byte-identical weights.
* The 256×256×32 sparse-conv kernel cache in spconv 2.3 is order-dependent.
  The released checkpoint is the canonical numerical reference.

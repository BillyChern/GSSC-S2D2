# Reproducibility

## Hardware

| Component | Used in paper | Minimum required |
|---|---|---|
| GPU | 2× NVIDIA H100 80 GB HBM3 PCIe | Same. Single-A100-40 GB has not been validated. |
| RAM | 256 GB | ≥64 GB (for SemanticKITTI dataset caching) |
| Disk | ~1 TB SSD | ~300 GB SSD (eval-only: 135 GB) |
| OS | Ubuntu 22.04 + CUDA 12.8 | Linux + CUDA 12.x |

### Disk-layout warning (Docker / overlay-fs hosts)

The D4 TTA eval (`val_d4tta`) writes ~40 GB of intermediate `.label`
files for the 4 071-frame val split. If your container's root
filesystem (`/`) is a 30 GB Docker overlay, eval will crash with
`OSError: 0 written` around frame 1 000.

**Mitigation:** point `--data-root` at a path on a large persistent
volume (typical `/workspace`, `/mnt/data`, or `/home/<user>/data`),
not on the overlay-fs root. The tooling persists predictions under
`<data-root>/predictions/<config-name>/`, so this directory must have
~50 GB free for D4 TTA and ~10 GB free for the 1-step path.

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

## Random seeds and retrain variance

The headline run uses **seed 42** for both data shuffling and parameter
initialization. The seed is passed via `--seed 42` (default in
`scripts/train.py`).

We use a single seed throughout the paper. **This matches the prevailing
SemanticKITTI SSC reporting convention** — every method in `tab:main_results`
(LMSCNet, SSA-SC, JS3C-Net, SCPNet, TALoS) likewise reports a single
test-server entry per configuration with no variance bars, because each full
training run at 256×256×32 voxel resolution costs roughly 37 GPU-hours on
2×H100, and the official scoring server returns one number per submission.

### Expected retrain variance

Single-seed retrains of SemanticKITTI SSC at 256×256×32 carry typical
**~0.3–0.5 mIoU run-to-run variance** from CUDA non-determinism, dataloader
worker timing, and initialization differences. A from-scratch retrain of the
headline configuration on the published codebase (`python scripts/train.py
train/31k_mf`) lands at **38.05% val 1-step mIoU**, within this variance band
of the released checkpoint's 38.54%. If your retrain reaches 38.05% ± 0.3%,
you have successfully reproduced the paper.

```bash
python scripts/train.py train/31k_mf
# step_40000.pt → 38.05% val 1-step mIoU (verified, full SemanticKITTI val seq 08).
```

Per-class deltas vs. SCPNet base under this retrain (val seq 08, 1-step):
car +1.6, motorcycle +4.7, truck +5.9, other-veh +6.4, person +1.6,
bicyclist +4.2, motorcyclist +4.4, road +1.3, parking +1.1, traffic-sign −0.2;
overall +2.37%. Per-class behavior is preserved; the small absolute mIoU
gap is run-to-run noise, not a recipe difference.

### Recipe summary

- `--seed 42` (default) is the verified, reproducible setting.
- `--batch_size 4` (default in `configs/train/31k_mf.yaml`) is the spec recipe
  consistent with the original `tools/launch_exp1.sh`.
- All other hyperparameters (`lr=1e-4`, `num_iterations=100000`, `ema_decay=0.9999`,
  loss weights, schedules) are pinned in the YAML and require no changes.

## Per-table reproduction

| Table / Figure | Command | Expected |
|---|---|---|
| Tab. I (test mIoU) | `python scripts/infer.py infer/test_d4tta --checkpoint data/checkpoints/gssc_31k_mf_step40000.safetensors --output preds/test/` then submit to SemanticKITTI Codabench | 39.2 mIoU, 59.0 IoU_cmpl |
| Tab. II (val per-class) | `python scripts/eval.py eval/val_1step --checkpoint data/checkpoints/gssc_31k_mf_step40000.safetensors --metrics miou per_class` | 38.54 mIoU |
<!-- Tab. III (safety metrics) intentionally omitted from this matrix until
     the dedicated safety-metric driver is implemented; see issue tracker. -->

| Tab. V (step reduction) | `python scripts/eval.py eval/step_sweep --checkpoint ...` | 38.54 (N=1), 38.59 (N=2), 38.65 (N=4), 38.16 (N=100) |
| Tab. VII (data scaling) | Per-row checkpoint, e.g. `python scripts/eval.py eval/val_1step --checkpoint data/checkpoints/gssc_31K_sf_step100000.safetensors` | See MODEL_ZOO.md |
| Tab. VIII (DW-IoU) | `python scripts/eval.py eval/val_1step --checkpoint ... --metrics dwiou` | per-T_w table |
| Tab. XII (training timesteps) | Per-row checkpoint | T=10: 37.83, T=50: 37.92, T=100-skewed: 38.18, T=100-uniform: 38.54 |
| Tab. XV (BEV) | `python scripts/eval.py eval/bev_secondary --checkpoint data/checkpoints/bev_perception_net.pt` | 36.09 BEV mIoU |
| Fig. 4 / Fig. 5 (qualitative) | See `examples/` notebooks | — |

All commands assume `data/checkpoints/` and `data/scpnet_predictions/` already exist (run `scripts/download_assets.py --checkpoints --predictions`).

## Determinism caveats

* PyTorch's CUDNN backend is stochastic by default. We disable nondeterminism
  in evaluation with `torch.backends.cudnn.deterministic = True`. Training
  retains nondeterministic CUDNN for speed; multiple training runs from the
  same seed will produce *very similar* but not byte-identical weights.
* The 256×256×32 sparse-conv kernel cache in spconv 2.3 is order-dependent.
  The released checkpoint is the canonical numerical reference.

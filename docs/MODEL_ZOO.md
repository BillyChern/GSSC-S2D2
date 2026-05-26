# Model Zoo

All checkpoints are released under Apache-2.0. Hosted on Hugging Face Hub at
`[CHECKPOINTS_URL]` (downloaded by `scripts/download_assets.py --checkpoints`).

## Layout (since v1.1.0)

Each release checkpoint lives in a per-checkpoint subdir matching the modern
Hugging Face / diffusers convention:

```
data/checkpoints/<group>/<name>/
├── model.safetensors        # training weights (resumable from this)
├── model_ema.safetensors    # deployment weights (paper convention, default for inference)
└── config.json              # train cfg + best_miou + global_step + source SHA256
```

Each safetensors file is a complete model state_dict — EMA-tracked parameters
overlaid onto trained BatchNorm running statistics (running_mean / running_var /
num_batches_tracked) — so `load_state_dict(strict=True)` works out of the box.

The legacy SCPNet base ships as a flat `scpnet_v2_port.pth` (third-party
convention, kept as-is).

## Headline scene-completion checkpoints

| Subdir | Paper section | Val mIoU | Test mIoU | Config | Size |
|---|---|---|---|---|---|
| `gssc_mf/gssc_31k_mf_step40000/` | **Headline** | 38.54 | 39.0 (N=4 plain) / 39.2 (+D4 TTA) / 38.8 (N=1 real-time) | `configs/train/31k_mf.yaml` | ~265 MB |
| `gssc_mf/gssc_57k_mf_step40000/` | Tab. V (negative result) | 37.76 (N=1) | — | `configs/train/57k_mf.yaml` | ~265 MB |

## Cross-base portability (paper Tab. III rows 90-91)

The same recipe and hyperparameters applied to three structurally different
frozen base models lifts every one of them. LMSCNet and JS3C-Net ship as
released checkpoints; SCPNet uses the same training recipe with
`configs/train/31k_mf.yaml` and lands the headline `gssc_mf/gssc_31k_mf_step40000`.

| Subdir | Base | Architecture family | Base mIoU | +S²D² mIoU | Δ | Config |
|---|---|---|---|---|---|---|
| `gssc_lmsc/gssc_lmsc_s2d2_real/` | LMSCNet | 2D CNN (dense)        | 12.10 | **16.59** | **+4.49** | `configs/train/lmscnet_real.yaml` |
| `gssc_js3c/gssc_js3c_s2d2_real/` | JS3C-Net | Point + voxel hybrid | 22.73 | **26.72** | **+3.99** | `configs/train/js3c_real.yaml`    |
| (uses `gssc_mf/gssc_31k_mf_step40000/`) | SCPNet | Sparse 3D CNN       | 36.17 | **38.54** | **+2.37** | `configs/train/31k_mf.yaml`       |

Each cross-base checkpoint is ~265 MB. Evaluators differ across rows: the
LMSCNet and JS3C-Net deltas are measured under the official
`semantic-kitti-api`; the SCPNet delta uses our internal `SSCMetrics`
evaluator (+0.31 pp internal/official gap, documented in
supp tab:supp_b6_val). The JS3C-Net row also has a derived-BEV
deploy-protocol number (**24.32**) under
`eval/js3c_val_realistic.yaml`:
- `eval/js3c_val_paper.yaml` reproduces the **26.72%** number by loading
  preprocessed GT BEV via `--bev_source gt`. The paper used the internal
  SSCMetrics evaluator; GSSC-S2D2 reports the same protocol via the official
  semantic-kitti-api, which differs by the +0.31 pp internal/official gap
  documented in supp tab:supp_b6_val (so the released number lands near the
  paper claim once that delta is applied).
- `eval/js3c_val_realistic.yaml` uses derived BEV (topmost-non-empty class
  from JS3C-Net's 3D prediction) — the honest deploy-time number. The
  released JS3C+S²D² model was trained with derived BEV, so this protocol
  matches its training distribution.

Reproduction requires `data/js3cnet_predictions/` (54 GB; download via
`scripts/download_assets.py --js3c-predictions` or dump locally via
`scripts/dump_js3c_predictions.py`; see `docs/REPRODUCIBILITY.md`).

## Single-frame retrains (Tab. VII data scaling)

| Subdir | Synthetic pool | Val mIoU (N=1 / peak) | Config |
|---|---|---|---|
| `gssc_sf/gssc_0K_sf_step100000/`  | None (real only)         | 38.18 / 38.46 (N=5)  | `configs/train/0K_sf.yaml`  |
| `gssc_sf/gssc_10K_sf_step100000/` | 10K synthetic            | 38.06 / 38.50 (N=10) | `configs/train/10K_sf.yaml` |
| `gssc_sf/gssc_20K_sf_step100000/` | 20K synthetic            | 38.14 / 38.49 (N=5)  | `configs/train/20K_sf.yaml` |
| `gssc_sf/gssc_31K_sf_step100000/` | 31K synthetic (headline) | 38.42 / 38.49 (N=2-5)| `configs/train/31K_sf.yaml` |
| `gssc_sf/gssc_57K_sf_step100000/` | 57K synthetic            | 37.66 / 38.05 (N=5)  | `configs/train/57K_sf.yaml` |

## Training-timestep ablations (Tab. XII)

| Subdir | Schedule | Val mIoU | Config |
|---|---|---|---|
| `gssc_timesteps/gssc_T10/`         | T=10 uniform            | 37.83 | `configs/train/T10.yaml`         |
| `gssc_timesteps/gssc_T50/`         | T=50 uniform            | 37.92 | `configs/train/T50.yaml`         |
| `gssc_timesteps/gssc_T100skewed/`  | T=100 skewed (t=T heavy)| 38.18 | `configs/train/T100skewed.yaml` |

## Pyramid diffusion (offline data augmentation)

| Subdir | Resolution | Purpose |
|---|---|---|
| `pyramid/pyramid_s1/` | 32×32×4    | Coarse scene generator |
| `pyramid/pyramid_s2/` | 64×64×8    | Mid-resolution refiner |
| `pyramid/pyramid_s3/` | 256×256×32 | Final-resolution generator (used to produce the 31K synthetic pool) |

Pyramid checkpoints do not use EMA; each subdir ships
`model.safetensors` + `config.json` only.

## BEV second task (Tab. XV)

| Subdir | Task | Pipeline mIoU | Config |
|---|---|---|---|
| `bev/bev_perception_net/` | LiDAR-only BEV refinement (S²D² applied to BEV) | **36.09** (34.27 base + 1.82 refinement) | `configs/train/bev_secondary.yaml` |
| `bev/bev_direct_l3_deeper/` | Supp BEV ablation (deeper 3D-direct baseline) | n/a (ablation only) | recipe in `docs/REPRODUCIBILITY.md` (no shipped config — was a one-off ablation run) |

## SCPNet base (frozen)

| File | What | Notes |
|---|---|---|
| `scpnet_v2_port.pth` | SCPNet pretrained weights, ported to spconv v2.3 with kernel-shape patches. | Loads via `gssc.models.scpnet_base`. Third-party flat `.pth` (not converted). |

## How to use

```python
from pathlib import Path
from safetensors.torch import load_file
from gssc.models.s2d2_unet import SceneCompletionUNetSparse

# Deployment uses model_ema.safetensors (EMA weights; paper convention)
ckpt_dir = Path("data/checkpoints/gssc_mf/gssc_31k_mf_step40000")
state = load_file(ckpt_dir / "model_ema.safetensors")
model = SceneCompletionUNetSparse.from_config("configs/model/s2d2_unet.yaml")
model.load_state_dict(state)
model.train(False)
```

Or via the eval entry point::

```bash
# SCPNet headline (38.54% val)
python scripts/eval.py eval/val_1step \
    --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors

# JS3C-Net cross-base (26.72% val, +3.99 pp)
python scripts/eval.py eval/js3c_val_1step \
    --checkpoint data/checkpoints/gssc_js3c/gssc_js3c_s2d2_real/model_ema.safetensors

# LMSCNet cross-base (16.59% val, +4.49 pp)
python scripts/eval.py eval/lmscnet_val_1step \
    --checkpoint data/checkpoints/gssc_lmsc/gssc_lmsc_s2d2_real/model_ema.safetensors
```

Or reproduce a specific paper table with the all-in-one driver::

```bash
python scripts/reproduce_table.py tab:perclass             # 38.54% val
python scripts/reproduce_table.py tab:cross_base_js3c      # 26.72% val
python scripts/reproduce_table.py tab:bev_results          # 36.09% BEV
```

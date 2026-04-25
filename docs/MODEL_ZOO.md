# Model Zoo

All checkpoints are released under Apache-2.0. Hosted on Hugging Face Hub at
`[CHECKPOINTS_URL]` (downloaded by
`scripts/download_assets.py --checkpoints`).

## Headline scene-completion checkpoints

| File | Paper section | Val mIoU | Test mIoU | Config | Size |
|---|---|---|---|---|---|
| `gssc_31k_mf_step40000.safetensors` | **Headline** | 38.54 | 39.0 (N=4 plain) / 39.2 (+D4 TTA) / 38.8 (N=1 real-time) | `configs/train/31k_mf.yaml` | 150 MB |
| `gssc_57k_mf_step40000.safetensors` | Tab. V (negative result) | 37.76 (1-step) | — | `configs/train/57k_mf.yaml` | 150 MB |

## Single-frame retrains (Tab. VII data scaling)

| File | Synthetic pool | Val mIoU (1-step / peak) | Config | Size |
|---|---|---|---|---|
| `gssc_0K_sf_step100000.safetensors`  | None (real only)        | 38.18 / 38.46 (N=5)  | `configs/train/0K_sf.yaml`  | 150 MB |
| `gssc_10K_sf_step100000.safetensors` | 10K synthetic           | 38.06 / 38.50 (N=10) | `configs/train/10K_sf.yaml` | 150 MB |
| `gssc_20K_sf_step100000.safetensors` | 20K synthetic           | 38.14 / 38.49 (N=5)  | `configs/train/20K_sf.yaml` | 150 MB |
| `gssc_31K_sf_step100000.safetensors` | 31K synthetic (headline)| 38.42 / 38.49 (N=2-5)| `configs/train/31K_sf.yaml` | 150 MB |
| `gssc_57K_sf_step100000.safetensors` | 57K synthetic           | 37.66 / 38.05 (N=5)  | `configs/train/57K_sf.yaml` | 150 MB |

## Training-timestep ablations (Tab. XII)

| File | Schedule | Val mIoU | Config | Size |
|---|---|---|---|---|
| `gssc_T10.safetensors`         | T=10 uniform           | 37.83 | `configs/train/T10.yaml`         | 150 MB |
| `gssc_T50.safetensors`         | T=50 uniform           | 37.92 | `configs/train/T50.yaml`         | 150 MB |
| `gssc_T100skewed.safetensors`  | T=100 skewed (t=T heavy)| 38.18 | `configs/train/T100skewed.yaml` | 150 MB |

## Pyramid diffusion (offline data augmentation)

| File | Resolution | Purpose | Size |
|---|---|---|---|
| `pyramid_s1.safetensors` | 32×32×4    | Coarse scene generator | 30 MB |
| `pyramid_s2.safetensors` | 64×64×8    | Mid-resolution refiner | 35 MB |
| `pyramid_s3.safetensors` | 256×256×32 | Final-resolution generator (used to produce the 31K synthetic pool) | 80 MB |

## BEV second task (Tab. XV)

| File | Task | mIoU | Config | Size |
|---|---|---|---|---|
| `bev_perception_net.safetensors` | LiDAR-only BEV semantic segmentation | 36.09 | `configs/train/bev_secondary.yaml` | 85 MB |

## SCPNet base (frozen)

| File | What | Notes |
|---|---|---|
| `scpnet_v2_port.pth` | SCPNet pretrained weights with spconv v2.3 kernel-shape patches applied. | Loads via `gssc.models.scpnet_base`. |

## How to use

```python
from safetensors.torch import load_file
from gssc.models.s2d2_unet import SceneCompletionUNetSparse

state = load_file("data/checkpoints/gssc_31k_mf_step40000.safetensors")
model = SceneCompletionUNetSparse.from_config("configs/model/s2d2_unet.yaml")
model.load_state_dict(state)
model.eval()
```

Or via the eval entry point::

```bash
python scripts/eval.py eval/val_1step --checkpoint data/checkpoints/gssc_31k_mf_step40000.safetensors
```

# GSSC-S2D2

**Official PyTorch implementation of S²D² (Structured Source Discrete Diffusion)**

> **Generative Semantic Scene Completion** &nbsp;·&nbsp; TPAMI 2026

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.4](https://img.shields.io/badge/pytorch-2.4-orange.svg)](https://pytorch.org/)

---

## Headline numbers (SemanticKITTI hidden test, single-frame single-sample)

| Method | Test mIoU | Δ over SCPNet |
|---|---|---|
| LMSCNet | 17.6 | — |
| JS3C-Net | 23.8 | — |
| SCPNet (base) | 36.7 | baseline |
| TALoS (NeurIPS 2024, prev. SOTA) | 37.9 | +1.2 |
| **S²D² (Ours, $N{=}1$ real-time)** | **38.8** | **+2.1** |
| **S²D² (Ours, $N{=}4$, no TTA)** | **39.0** | **+2.3** |
| **S²D² (Ours, $N{=}4$, $D_4$ TTA)** | **39.2** | **+2.5** |

On full SemanticKITTI val seq 08: **38.54%** mIoU (1-step) / **38.73%** (+$D_4$ TTA), $+2.37$ over our SCPNet baseline.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/BillyChern/GSSC-S2D2.git && cd GSSC-S2D2

# 2. Install (uv recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv pip install spconv-cu121==2.3.6   # spconv installed separately, see docs/REPRODUCIBILITY.md

# 3. Download checkpoint + reference data (~3 GB models, optional 230 GB synth pool)
gssc-download --checkpoints --predictions

# 4. Reproduce headline val mIoU
gssc-eval +config=eval/val_1step
# expects 38.54% on SemanticKITTI val seq 08

# 5. Reproduce hidden-test 39.2% predictions (D₄ TTA)
gssc-infer +config=infer/test_d4tta --output predictions/
# bundle predictions/ → submit to SemanticKITTI Codabench
```

See **[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)** for the exact env, hardware, and per-table number reproduction.

---

## What's in the repo

| Path | What |
|---|---|
| `src/gssc/` | Python package: models, diffusion core, data, losses, training, inference |
| `configs/` | Hydra configs for every train + eval + inference recipe in the paper |
| `scripts/` | One-command driver scripts (`train.py`, `eval.py`, `infer.py`, `download_assets.py`, `reproduce_table.py`) |
| `docs/` | [MODEL_ZOO](docs/MODEL_ZOO.md), [DATASET](docs/DATASET.md), [REPRODUCIBILITY](docs/REPRODUCIBILITY.md), [TRAIN](docs/TRAIN.md), [INFERENCE](docs/INFERENCE.md), [BASELINES](docs/BASELINES.md) |
| `tests/` | pytest smoke + unit tests |
| `examples/` | Jupyter notebooks: `00_quickstart.ipynb` → run on 1 frame |

---

## Reproducing the paper

| Table / Figure | Command | Expected |
|---|---|---|
| Tab. I (test mIoU + per-class) | `gssc-infer +config=infer/test_d4tta` then submit | 39.2% test mIoU |
| Tab. II (val per-class) | `gssc-eval +config=eval/val_1step` | 38.54% val mIoU, full per-class |
| Tab. III (safety metrics) | `gssc-eval +config=eval/val_1step --metrics safety` | SC-mIoU 35.2, VRU-IoU 19.6 |
| Tab. V (step reduction) | `gssc-eval +config=eval/step_sweep` | full sweep N=1..100 |
| Tab. VII (data scaling) | `gssc-eval +config=eval/data_scaling_sf` | 0K/10K/20K/31K/57K SF retrains |
| Tab. VIII (DW-IoU) | `gssc-eval +config=eval/val_1step --metrics dwiou` | per-T_w table |
| Fig. 4 (qualitative) | `python scripts/render_figures.py --fig 4` | bicyclist 003096 + motorcyclist 001417 panels |
| Fig. 5 (gallery) | `python scripts/render_figures.py --gallery` | 10 rare-class scene comparisons |
| Tab. XII (training timesteps) | `gssc-eval +config=eval/timestep_ablation` | T=10/50/100-skewed/100-uniform |
| Tab. XV (BEV second-task) | `gssc-eval +config=eval/bev_secondary` | 36.09% BEV mIoU |

The full mapping is in **[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)**.

---

## Hardware

The headline results were trained on **2× NVIDIA H100 80 GB HBM3 (PCIe)**, ~37 GPU-hours per 100K-iteration run. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the exact environment matrix.

---

## Citation

```bibtex
@article{gssc2026,
  title   = {Generative Semantic Scene Completion},
  author  = {TBD},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2026}
}
```

(also see [`CITATION.cff`](CITATION.cff))

---

## License

[Apache License 2.0](LICENSE) — code, configs, and documentation.
Released model weights are under the same license; SemanticKITTI raw data is governed by its own license terms (see [SemanticKITTI](http://www.semantic-kitti.org/)).

---

## Acknowledgements

This codebase builds on top of:
- **SCPNet** ([CVPR 2023](https://github.com/SCPNet/Codes-for-SCPNet)) — frozen base model.
- **SemanticKITTI** ([ICCV 2019](http://www.semantic-kitti.org/)) — dataset.
- **spconv** ([2.3](https://github.com/traveller59/spconv)) — sparse 3D conv backend.
- **D3PM / Multinomial Diffusion** ([NeurIPS 2021](https://arxiv.org/abs/2107.03006)) — discrete diffusion family.
- **SegRefiner** ([NeurIPS 2023](https://github.com/MengyuWang826/SegRefiner)) — closest 2D refinement predecessor.

<div align="center">

# GSSC-S2D2

**Official PyTorch implementation of S²D² (Structured Source Discrete Diffusion)**

📄 **[Generative Semantic Scene Completion](https://arxiv.org/abs/TBD)** &nbsp;·&nbsp; *TPAMI 2026*

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.4](https://img.shields.io/badge/pytorch-2.4-orange.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-46a2f1.svg)](https://github.com/astral-sh/ruff)
[![Type checks: mypy strict](https://img.shields.io/badge/types-mypy_strict-2A6DB2.svg)](https://mypy.readthedocs.io/)

State-of-the-art LiDAR semantic scene completion on SemanticKITTI: **39.2 % test mIoU**, the first leaderboard advance among LiDAR single-frame single-sample submissions since TALoS (NeurIPS 2024).

</div>

---

## TL;DR

S²D² refines the prediction of any frozen base SSC network through iterative correction sampling on the per-voxel probability simplex. Starting from a *structured source* (the base model's prediction) instead of pure noise, the method learns a velocity field that transports the source toward ground truth in a single Euler step at deployment.

> *One forward pass on top of the base model, no distillation, no test-time adaptation, runs at 9.33 FPS marginal throughput on a single H100, and beats the previous SOTA (TALoS, NeurIPS 2024) by +1.3 absolute mIoU on the SemanticKITTI hidden test leaderboard.*

---

## Headline numbers — SemanticKITTI hidden test (single-frame single-sample LiDAR)

| Method | Test mIoU | IoU<sub>cmpl</sub> | Δ over SCPNet | Notes |
|---|---:|---:|---:|---|
| LMSCNet | 17.6 | 56.7 | — | CVPRW 2020 |
| JS3C-Net | 23.8 | 56.6 | — | AAAI 2021 |
| SSA-SC | 23.5 | 58.8 | — | IROS 2021 |
| SCPNet (base) | 36.7 | 56.1 | baseline | CVPR 2023 (frozen base) |
| TALoS (prev. SOTA) | 37.9 | **60.2** | +1.2 | NeurIPS 2024, line-of-sight test-time adaptation |
| **S²D² (Ours, 1-step real-time)** | **38.8** | 58.9 | **+2.1** | 9.33 FPS marginal; cheapest deployable |
| **S²D² (Ours, *N* = 4, no TTA)** | **39.0** | 58.8 | **+2.3** | practical deployable variant |
| **S²D² (Ours, *N* = 4, *D*<sub>4</sub> TTA)** | 🏆 **39.2** | 59.0 | **+2.5** | leaderboard SOTA |

On full SemanticKITTI val seq 08:
* **38.54 %** mIoU (1-step Algo2)
* **38.73 %** mIoU (+ *D*<sub>4</sub> TTA)
* **+2.37** absolute over our SCPNet base (36.17 %)

---

## Quick start (reproduce 38.54 % val in three commands)

```bash
# 1. Clone + install (uv recommended)
git clone https://github.com/BillyChern/GSSC-S2D2.git && cd GSSC-S2D2
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10 && uv sync && uv pip install spconv-cu121==2.3.6

# 2. Pull pretrained checkpoint + SCPNet predictions
python scripts/download_assets.py --checkpoints --predictions
# → data/checkpoints/gssc_31k_mf_step40000.safetensors  (150 MB)
# → data/scpnet_predictions/                            (~50 GB, val + test)

# 3. Reproduce the headline 38.54 % val mIoU
python scripts/eval.py eval/val_1step \
    --checkpoint data/checkpoints/gssc_31k_mf_step40000.safetensors
```

That's it. Expected output:

```
[gssc.eval] Loading checkpoint: data/checkpoints/gssc_31k_mf_step40000.safetensors
[gssc.eval] SemanticKITTI val seq 08 (4071 frames)
[gssc.eval] Algo2 correction sampling, N=1
...
[gssc.eval] mIoU       38.54 %
[gssc.eval] IoU_cmpl   55.45 %
[gssc.eval] Per-class:
[gssc.eval]   motorcyclist 12.4   bicyclist 23.2   person 23.2   ...
```

For the full hidden-test leaderboard submission flow (39.2 % via *D*<sub>4</sub> TTA), see [docs/INFERENCE.md](docs/INFERENCE.md).

---

## Repository layout

```
GSSC-S2D2/
├── src/gssc/                       # the Python package
│   ├── models/                     # sparse 3D U-Net, SCPNet base, pyramid, BEV variant, FiLM
│   ├── diffusion/                  # forward process, Dirac posterior, Algo2 sampler, D4 TTA
│   ├── data/                       # SemanticKITTI loader, synthetic pool, object bank, HDL-64E ray-tracer
│   ├── losses/                     # KL posterior + Lovász + auxiliary + focal-CE
│   ├── training/                   # canonical trainer + EMA + logging
│   ├── inference/                  # eval (mIoU + safety metrics + DW-IoU), visualisation
│   └── utils/                      # config loader, seeding, registry
├── configs/                        # Hydra configs (one per recipe in the paper)
│   ├── train/{31k_mf,0K_sf,...,T100skewed}.yaml
│   ├── eval/{val_1step,val_d4tta,step_sweep}.yaml
│   └── infer/{test_d4tta,val_1step}.yaml
├── scripts/                        # one-command drivers
│   ├── train.py
│   ├── eval.py
│   ├── infer.py
│   ├── prepare_data.py
│   ├── download_assets.py
│   └── reproduce_table.py
├── docs/                           # full reproducibility documentation
│   ├── REPRODUCIBILITY.md
│   ├── DATASET.md
│   ├── MODEL_ZOO.md
│   ├── TRAIN.md
│   ├── INFERENCE.md
│   └── BASELINES.md
├── tests/                          # pytest unit + smoke tests
├── examples/                       # Jupyter notebooks for new users
├── external/                       # third-party (semantic-kitti-api, multinomial_diffusion)
├── CITATION.cff                    # citation metadata
├── CONTRIBUTING.md                 # code-quality standards
├── pyproject.toml                  # uv-managed Python project
└── LICENSE                         # Apache-2.0
```

---

## Reproducing every paper number

The repo ships the exact recipe + checkpoint for every reported number.

| Paper artefact | Command | Expected |
|---|---|---|
| **Tab. I** (test mIoU + per-class) | `python scripts/infer.py infer/test_d4tta --checkpoint data/checkpoints/gssc_31k_mf_step40000.safetensors --output preds/` then submit to [Codabench](https://codalab.lisn.upsaclay.fr/competitions/7170) | **39.2 %** test mIoU |
| **Tab. II** (val per-class) | `python scripts/eval.py eval/val_1step --checkpoint <headline> --metrics miou per_class completion_iou` | **38.54 %** val mIoU |
| **Tab. III** (safety metrics) | `python scripts/eval.py eval/val_1step --checkpoint <headline> --metrics safety` | SC-mIoU 35.2, VRU-IoU 19.6 |
| **Tab. V** (step reduction) | `python scripts/eval.py eval/step_sweep --checkpoint <headline>` | 38.54 (N=1) … 38.65 (N=4 peak) … 38.16 (N=100) |
| **Tab. VII** (data scaling) | `python scripts/reproduce_table.py tab:data_scaling` | 0K/10K/20K/31K/57K SF retrains |
| **Tab. VIII** (DW-IoU) | `python scripts/eval.py eval/val_1step --checkpoint <headline> --metrics dwiou` | per-T_w table |
| **Tab. XII** (training timesteps) | `python scripts/reproduce_table.py tab:train_timesteps_curriculum` | T=10/50/100-skewed/100-uniform |
| **Tab. XV** (BEV second task) | `python scripts/eval.py eval/bev_secondary --checkpoint data/checkpoints/bev_perception_net.safetensors` | **36.09 %** BEV mIoU |
| **Fig. 4** / **Fig. 5** (qualitative) | `examples/01_render_figures.ipynb` | bicyclist 003096 + motorcyclist 001417 (Fig. 4); 10-row gallery (Fig. 5) |

Full mapping with anticipated wall-clock and disk requirements: **[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)**.

---

## Retraining the headline (≈ 37 GPU-hours on 2 × H100 80 GB)

```bash
python scripts/train.py train/31k_mf --gpu 0,1 --seed 42
# 100K iterations, batch size 4, AdamW lr 1e-4
# Logs → outputs/train_31k_mf/{tensorboard,step_*.pt,best.pt}
```

Expected best-EMA val mIoU ∈ [38.3 %, 38.7 %] (within seed noise of the 38.54 % headline). See [docs/TRAIN.md](docs/TRAIN.md) for every other recipe (data scaling, timestep ablations, pyramid diffusion, BEV second task).

---

## Hardware + environment

| Component | Reference (used in paper) | Minimum tested |
|---|---|---|
| GPU | 2 × NVIDIA H100 80 GB HBM3 PCIe | Same — single-A100-40 GB **not** validated |
| RAM | 256 GB | 64 GB |
| Disk | 1 TB SSD | 300 GB SSD (135 GB for eval-only) |
| OS | Ubuntu 22.04 + CUDA 12.8 | Linux + CUDA 12.x |
| Python | 3.10 / 3.11 | 3.10+ |
| PyTorch | 2.4.0 | 2.4.x |
| spconv | 2.3.6 (cu121, with our patches) | required |

Pinned versions in `uv.lock`. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the exact environment matrix.

---

## Method at a glance

S²D² introduces three departures from standard discrete diffusion:

1. **Structured source.** Replace the noise endpoint with a learned base model's prediction `x_src`. The forward process becomes the Dirac mixture `x_t = ᾱ_t · x_0 + (1 − ᾱ_t) · x_src`, a deterministic interpolant between ground truth and `x_src`.
2. **Algo2 correction sampling.** A non-noise reverse process that routes the full residual `x̂_0 − x_src` directly per step. At `N = 1`, the iterate at `t = T` coincides with `x_src`, giving a Lipschitz-free single-step bound (App. A.5 in the paper).
3. **Pyramid diffusion data augmentation.** A coarse-to-fine pyramid (32³ → 64³ → 256² × 32) generates synthetic `(sparse, complete)` pairs. Combined with HDL-64E ray-tracing and a 57,789-instance object bank, this yields the 31K-scene synthetic pool used by the headline configuration.

The mathematical derivations are in App. A of the paper (`prop:forward`, `prop:posterior`, `prop:elbo`, `prop:fm`, `prop:meanflow`, `thm:error`, `cor:onestep`, `cor:lipprop`, `prop:proj`).

---

## Asset releases

| What | Where | Size |
|---|---|---|
| Pretrained checkpoints (~14 files) | [HF: gssc-s2d2/checkpoints](`[CHECKPOINTS_URL]`) | 3 GB |
| SCPNet val + test predictions | [HF: gssc-s2d2/scpnet_predictions](`[SCPNET_PREDICTIONS_URL]`) | 50 GB |
| Object bank (57,789 instances, 8 rare classes) | [HF: gssc-s2d2/object_bank](`[OBJECT_BANK_URL]`) | 448 MB |
| Synthetic pool (0K / 10K / 20K / 31K / 57K) | [IEEE DataPort](`[SYNTHETIC_POOL_URL]`) | 120 – 220 GB per variant |

All weights and synthetic data are released under Apache-2.0; SemanticKITTI raw data follows its own license (see [semantic-kitti.org](http://www.semantic-kitti.org/)).

---

## Code-quality + testing

This codebase targets Google/Apple production-grade standards. The toolchain is **CI-enforced**:

```bash
ruff check src/ tests/ scripts/      # style + import order
mypy --strict src/gssc/              # static types
pytest tests/ -v                     # unit + smoke tests
pytest --cov=gssc --cov-fail-under=80 # coverage
vulture src/                         # dead code
bandit -r src/                       # security
```

Style conventions, commit conventions, and hard requirements: **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## FAQ

**Q. Why use this over running SCPNet alone?**
A. We add **+2.5 absolute mIoU** on the hidden test set with a single extra forward pass (107 ms on H100), no extra training data beyond what SCPNet was trained on, and no distillation. The gains concentrate on safety-critical rare classes (motorcyclist +8.2, other-vehicle +6.4, truck +5.9, bicyclist +4.2 on val seq 08).

**Q. Can S²D² be applied to a different base SSC network?**
A. Yes — the framework is base-model-agnostic. We provide a working SCPNet integration; switching to JS3C-Net or any other base requires only providing per-voxel categorical predictions as `x_src`. See [docs/BASELINES.md](docs/BASELINES.md).

**Q. Do we need the synthetic pool to use the released checkpoint?**
A. **No.** Eval-only deployment uses the released weights + SCPNet predictions only (~135 GB total). The 230 GB synthetic pool is only needed for retraining from scratch.

**Q. Why does the train script use a YAML "config" rather than direct CLI args?**
A. Every paper artefact corresponds to a config file. `python scripts/train.py train/31k_mf` runs the exact headline recipe with no chance of accidentally diverging from the paper.

**Q. Single-seed numbers — why no error bars?**
A. Same convention as every method in the leaderboard table: a full SemanticKITTI SSC training run is expensive (~37 GPU-hours), and the official scoring server takes a single submission. We use a single seed (42) to match this convention. See §V.A of the paper for the variance-disclosure discussion.

**Q. The repo has no figures — where are they?**
A. Figures and paper-typesetting code live with the paper repo, not here. This repo focuses on **method reproduction**. The qualitative comparisons in Fig. 4 / Fig. 5 are reproducible via `examples/01_render_figures.ipynb`, which produces SSIM-matched outputs from the released checkpoint.

---

## Citation

```bibtex
@article{gssc2026,
  title   = {Generative Semantic Scene Completion},
  author  = {[AUTHOR_LIST_TBD]},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2026},
  doi     = {10.1109/TPAMI.2026.[DOI_TBD]}
}
```

(Machine-readable: [`CITATION.cff`](CITATION.cff))

---

## License

* **Code, configs, documentation:** [Apache License 2.0](LICENSE).
* **Released model weights:** Apache-2.0 (compatible with downstream commercial use).
* **SemanticKITTI raw data:** governed by its own license — see [semantic-kitti.org](http://www.semantic-kitti.org/).
* **Third-party code under `external/`:** retains its original license.

---

## Acknowledgements

This codebase builds on top of:

* **SCPNet** ([CVPR 2023](https://github.com/SCPNet/Codes-for-SCPNet)) — frozen base model whose predictions seed the structured source.
* **SemanticKITTI** ([ICCV 2019](http://www.semantic-kitti.org/)) — voxelised LiDAR scene completion benchmark.
* **spconv 2.3** ([traveller59/spconv](https://github.com/traveller59/spconv)) — sparse 3D convolution backend.
* **D3PM / Multinomial Diffusion** ([NeurIPS 2021](https://arxiv.org/abs/2107.03006)) — discrete diffusion family.
* **SegRefiner** ([NeurIPS 2023](https://github.com/MengyuWang826/SegRefiner)) — closest 2D refinement predecessor.
* **TALoS** ([NeurIPS 2024](https://arxiv.org/abs/2410.15674)) — previous SemanticKITTI SSC SOTA, included as the leaderboard reference baseline.
* **Cold Diffusion** ([Bansal et al., NeurIPS 2023](https://arxiv.org/abs/2208.09392)) — non-noise-degradation correction sampling.

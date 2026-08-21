<div align="center">

# Generative Semantic Scene Completion

### A three-pillar generative framework: PS³ · SGSC · S²D²

📄 **Paper** *(under review — link added on acceptance)* &nbsp;·&nbsp; 🌐 **Project page** *(public on acceptance)* &nbsp;·&nbsp; 🏆 **[Leaderboard](https://www.codabench.org/competitions/13814/#/results-tab)** &nbsp;·&nbsp; 📦 **[Model Zoo](docs/MODEL_ZOO.md)** &nbsp;·&nbsp; 📊 **[Reproducibility](docs/REPRODUCIBILITY.md)** &nbsp;·&nbsp; 📒 **[Colab](examples/quickstart.ipynb)** &nbsp;·&nbsp; 🐛 **[Issues](https://github.com/BillyChern/GSSC-S2D2/issues)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillyChern/GSSC-S2D2/blob/main/examples/quickstart.ipynb)

[![Test status](https://github.com/BillyChern/GSSC-S2D2/actions/workflows/test.yml/badge.svg)](https://github.com/BillyChern/GSSC-S2D2/actions/workflows/test.yml)
[![Lint status](https://github.com/BillyChern/GSSC-S2D2/actions/workflows/lint.yml/badge.svg)](https://github.com/BillyChern/GSSC-S2D2/actions/workflows/lint.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.4](https://img.shields.io/badge/pytorch-2.4-orange.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-46a2f1.svg)](https://github.com/astral-sh/ruff)
[![Type checks: mypy](https://img.shields.io/badge/types-mypy-2A6DB2.svg)](https://mypy.readthedocs.io/)
[![SemanticKITTI](https://img.shields.io/badge/SemanticKITTI%20test-38.8%20mIoU%20%28N%3D1%2C%20no%20TTA%29-orange.svg)](https://www.codabench.org/competitions/13814/#/results-tab)

**SemanticKITTI hidden test — 38.8 % mIoU** at the single-frame single-sample setting (*N* = 1, no TTA) — to our knowledge the best single-frame single-sample result on the leaderboard to date (+0.9 over TALoS, 37.9) — rising to **39.2 % mIoU** with four correction steps and *D*<sub>4</sub> TTA, a row the single-sample predicate excludes and which therefore carries no TALoS margin. The leaderboard entry is public on [Codabench](https://www.codabench.org/competitions/13814/#/results-tab).

</div>

The paper organises the method into three pillars that share one structured-source discrete-diffusion core:

* **PS³** — *Paired Sparse–Dense Scene Synthesis*: the offline data-augmentation pipeline (pyramid multinomial diffusion 𝒮₁→𝒮₂→𝒮₃ + Jensen–Shannon filter + rare-class object bank + HDL-64E ray-tracer) that builds a 32,039-frame synthetic (sparse, complete) pool; pooled with the 19,130 real frames this gives a 51,169-frame total training set (2.67× expansion).
* **SGSC** — *Semantic-guided Generative Scene Completion*: the from-noise regime that completes a scene by sampling from the prior (no frozen 3D base), reaching **30.5 % val mIoU** on SemanticKITTI seq 08, comparable to PaSCo's 30.1 % (within run-to-run variance, a 0.4 pp gap; not protocol-matched, since PaSCo's 30.1 % is a subsidiary panoptic-mode number). *The SGSC checkpoint is not part of this release; the released surface is the deployment-oriented S²D² pillar.*
* **S²D²** — *Structured Source Discrete Diffusion*: the one-step deployment regime that refines a frozen base SSC model's prediction — the headline 38.54 % val / 39.2 % test result and the focus of this repository.

> [!IMPORTANT]
> **One-pass refinement of a frozen base SSC model via discrete diffusion on the probability simplex.** No distillation and no test-time adaptation, at a **+0.9 absolute mIoU** hidden-test gain over the previous SOTA TALoS (37.9 → 38.8) — equivalently **+2.1 pp** over the frozen SCPNet base on hidden test (36.7 → 38.8) and **+2.36 pp** on val seq 08 (36.17 → 38.54, single correction step). Those margins are the *N* = 1, no-TTA row, which is the only one the paper's `online, causal, single-sweep, single-sample` predicate admits. The 8-view *D*<sub>4</sub> ensemble row reads 39.2, and the paper excludes it from exactly this comparison — do not quote a TALoS margin off it. The correction step costs 107.2 ms, i.e. 9.33 FPS for the added pass alone; the paper is explicit that this is "an incremental pass, not a deployable rate", and that neither it nor the 3.23 FPS end-to-end pipeline matches the sensor's 10 Hz cadence. Replaces the base argmax with one cheap correction step (validated on SCPNet) — and the same mechanism transfers to 2D BEV semantic segmentation (paper Sec. 4 secondary task, **+1.3 BEV mIoU** over the base-derived BEV: 34.8 % → 36.1 % on val seq 08, scored by the training-time 2D BEV evaluator on 100 fixed val frames (seed 42) — not the 4,071-frame `semantic-kitti-api` protocol every 3D number here is scored under).

> [!TIP]
> **In a hurry?** Skip to [Quick start](#quick-start-reproduce-3854--val-in-three-commands) for the 3-command reproduction recipe of the headline 38.54 % val mIoU. Total wall-clock: **~6 minutes** on a single H100 once the base predictions are local.

> [!NOTE]
> **Assets.** The pretrained checkpoints and base-model predictions are public; `scripts/download_assets.py` provisions them, and `docs/DATASET.md` documents how to regenerate every artefact locally instead. A fresh clone cannot reach 38.54 % without those assets. The commands below are the exact recipe — the "verified end-to-end" labels record what the maintainers measured locally, not what an external clone can run today.

### What's new

* **2026-08-13** — Releases **v2.3.1 – v2.3.8**: a run of correction-only patch releases — no API, config or measured-value change. They removed a JS3C-Net cross-base figure that appears nowhere in the paper and restored **24.3 % (+1.6 pp)** as that base's headline; retracted the margins that had been quoted off the excluded 8-view *D*<sub>4</sub> row in favour of the predicate-satisfying **+0.9** (over TALoS) and **+2.1 pp** (over the frozen base); withdrew the "real-time" and "cheapest deployable" wording the paper explicitly disclaims; stopped three docs asserting a hidden-test measurement for our SCPNet port, which was never submitted to the scoring server; corrected an auxiliary-loss weight that two comments quoted 100× too high; recorded that the PS³ Jensen–Shannon filter ships as screened *data* rather than as code (the supplementary specifies it completely); priced a full 100K-iteration launch at ~90 GPU-h, the ~37 GPU-h figure being what reaches the released step-40000 checkpoint; and brought all four version declarations back in step so `uv lock --check` is clean. Per-release detail in [CHANGELOG.md](CHANGELOG.md).
* **2026-08-12** — Release **v2.3.0**, the TPAMI submission snapshot. Adds `configs/infer/test_1step.yaml`, the hidden-test single-sample (*N* = 1) configuration — until then the **38.8 %** headline row had no runnable command while the *D*<sub>4</sub>-TTA row did. Also adds the per-frame VRU instrument (`scripts/perframe_vru.py`), DW-IoU (`src/gssc/utils/dw_iou.py`), a `--tau` flag so the paper's temperature-invariance claim can be checked rather than taken on trust, `configs/eval/round2_a.yaml`, and configs for ablation rows that previously had none (`train/{57k_mf,T10,T50,c1_lossmatched_t99}`). Fixes three recipes that could not run as advertised (an unbound `S3DSKDDataset`, `RareClassEnhancer` reading dropped paste-budget fields, and `eval/val_d4tta` set to `correction_steps: 1` where 4 is what reproduces the +*D*<sub>4</sub> val number), and corrects the retrain per-class deltas and the headline **+2.36 pp** val delta.
* **2026-06-10** — Release **v2.2.0**: zero-shot cross-dataset evaluation. The frozen headline checkpoint transfers with no fine-tuning and no target labels — **SSCBench-KITTI-360** (val seq 06) **5.8 → 6.2 mIoU (+0.4)** and **18.1 → 19.5 CompIoU (+1.4)**; **SemanticPOSS** (val seq 02, TALoS Tab. 4 map) **1.0 → 6.5 mIoU (+5.5)** and **31.8 → 54.9 CompIoU (+23.1)** — via `scripts/eval_kitti360.py` and `scripts/eval_semanticposs.py`. Also reconciles every JS3C-Net cross-base figure across the docs to one labelled scheme, and re-exports the LMSCNet EMA checkpoint with its 45 BatchNorm buffers so it reproduces **16.6 %** val mIoU directly.
* **2026-05-26** — Release **v2.1.0**: LMSCNet third-base support. Stacked on the lightweight dense-2D-CNN LMSCNet (Roldao et al., 3DV 2020), one-step S²D² lifts val mIoU **14.8 % → 16.6 % (+1.8 pp)** under the official `semantic-kitti-api` evaluator (paper tab:portable_s2d2; the LMSCNet base is re-scored from on-disk predictions, superseding the earlier 12.10 summary). Together with the v1.1.0 JS3C-Net row this gives three structurally different frozen bases (dense 2D CNN, point-voxel hybrid, sparse 3D CNN) all lifted by the same recipe and hyperparameters — base-agnostic by construction, not by tuning. Reproduce: `python scripts/eval.py eval/lmscnet_val_1step --checkpoint data/checkpoints/gssc_lmsc/gssc_lmsc_s2d2_real/model_ema.safetensors` (or `python scripts/reproduce_table.py tab:portable_s2d2`, which runs every cross-base row). We also trimmed the release surface: 22 unreferenced development modules were pruned, and the remaining V2/V3 FiLM denoiser variants are kept as clearly labeled research-reference prototypes (excluded from the public API and CI gating), so the dense-Conv3d headline path is the single supported surface.
* **2026-05-18** — Release **v2.0.0**: remove the legacy SCPNet-specific BEV-derivation flag (the pre-v1.1.1 name of `--bev_from_base`) in favour of `--bev_from_base`. `base_pred_dir` is now the preferred config key (used by the JS3C-Net and LMSCNet bases); `scpnet_pred_dir` is still accepted and remains in the SCPNet configs for backward compatibility. Headline numerical artefacts unchanged.
* **2026-05-14** — Release **v1.1.0**: JS3C-Net cross-base support. Stacked on the point-voxel hybrid JS3C-Net (Yan et al., AAAI 2021), one-step S²D² lifts JS3C-Net val mIoU **22.7 % → 24.3 % (+1.6 pp)**, the paper's headline for this base (official `semantic-kitti-api`, derived BEV). The GT-BEV diagnostic for the same run reads **26.05 % (+3.3 pp)**, which the paper does not print; the same model under the paper's internal training-time evaluator reads **26.7 % (+4.0 pp)** (a continuity row), and the reproducible at-deploy number under the official `semantic-kitti-api` with derived BEV (what `scripts/reproduce_table.py` yields) is **24.3 % (+1.6 pp)** — see `docs/REPRODUCIBILITY.md`. Release-asset layout migrated to per-checkpoint safetensors subdirs matching the modern HF Hub convention.
* **2026-04** — Public release **v1.0.0**. Headline checkpoint released under Apache 2.0; eval round-trip verified at 38.54 % val mIoU.
* **2026-04** — Secondary BEV-task reproduction path added (`eval/bev_secondary` config + driver, checkpoint `bev/bev_s2d2_scpnet`). LiDAR-only BEV refinement at 36.1 % mIoU, scored by the training-time 2D BEV evaluator on 100 fixed val frames (seed 42) — not the 4,071-frame `semantic-kitti-api` protocol.
* **2026-03** — **39.2 %** mIoU on SemanticKITTI hidden test leaderboard — paper under review.

---

## Method at a glance

<p align="center">
  <img src="assets/teaser.png" width="92%" alt="GSSC-S2D2 two-stage pipeline: offline data augmentation + S²D² one-step deployment" />
</p>

<sub><b>Stage A</b> (top) — offline data augmentation: pyramid multinomial diffusion (𝒮₁ → 𝒮₂ → 𝒮₃ at 32²×4 → 64²×8 → 256²×32) synthesises complete scenes; a 57,789-instance / 8-rare-class object bank (bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist, trunk) pastes rare classes on ground-level voxels; then an HDL-64E Bresenham3D ray-tracer (64×2048 rays) converts each into a matching sparse input. The 32,039 synthetic pairs are pooled with the 19,130-frame real SemanticKITTI training split for a 51,169-frame total training pool (2.67× expansion). The SemanticKITTI split follows the standard SSC convention: sequences 00–07, 09, 10 train, sequence 08 val, sequences 11–21 hidden test (full layout in <a href="docs/DATASET.md">docs/DATASET.md</a>). <b>Stage B</b> (bottom) — at deployment, a real sparse LiDAR scan is voxelized through a frozen base SSC model <i>g</i><sub>φ</sub> (e.g. SCPNet, LMSCNet) to produce <b>x</b><sub>src</sub>, then refined by <i>f</i><sub>θ</sub> — a dense 3D U-Net (Conv3d) with additive <b>L</b>/<b>B</b> conditioning + time-AdaGN at every level — into <b>x̂</b><sub>0</sub> in one forward pass with EMA weights. No distillation. Source: paper Fig. 2.</sub>

**Why it works.** Pure-noise diffusion wastes capacity learning to invert random corruption. By starting from the base model's *structured* prediction **x**<sub>src</sub> rather than *x*<sub>T</sub> ∼ π, the network only has to learn the **residual** between **x**<sub>src</sub> and ground truth — a much smaller chunk of probability mass to transport. One correction step suffices for the headline 38.54 % val mIoU; four steps + *D*<sub>4</sub> TTA push to **39.2 % test**.

<p align="center">
  <img src="assets/architecture.png" width="92%" alt="S²D² simplex transport + 4-level dense 3D U-Net (Conv3d) denoiser" />
</p>

<sub><b>Top:</b> the learned velocity field <b>v</b><sub>θ</sub> = <i>f</i><sub>θ</sub>(<b>x</b><sub><i>t</i></sub>, <i>t</i>, <b>c</b>) − <b>x</b><sub>src</sub> transports the base prediction (gray cluster) toward the ground truth (orange cluster) in a single Euler step on the per-voxel simplex. <b>Middle:</b> <i>f</i><sub>θ</sub> is a 4-level dense 3D U-Net (Conv3d) (stages 256²×32 → 32²×4, channels 32 → 256, 16²×2 bottleneck with two Residual Blocks, mirror decoder) with additive <b>L</b>/<b>B</b> conditioning + time-AdaGN modulation at every level and three conditioning buses (timestep <i>t</i>, multi-scale LiDAR <b>L</b>, base-derived BEV <b>B</b>). <b>Bottom:</b> a representative rare-class recovery on SemanticKITTI val seq 08 frame 001390 — motorcyclist IoU lifts from 27.3 % to <b>41.5 %</b> with a single Euler step (<i>N</i>=1), a +14.2 point per-sample jump. Source: paper Fig. 3 (qualitative panel, frame 001390); the full-val per-class motorcyclist gain (4.1 → 12.4, +8.3) is tabulated in paper <code>tab:perclass_delta</code>.</sub>

---

## Headline numbers — SemanticKITTI hidden test (single-frame single-sample LiDAR)

| Method | Test mIoU | IoU<sub>cmpl</sub> | Δ over SCPNet | Notes |
|---|---:|---:|---:|---|
| LMSCNet | 17.6 | 56.7 | — | 3DV 2020 |
| JS3C-Net | 23.8 | 56.6 | — | AAAI 2021 |
| SSA-SC | 23.5 | 58.8 | — | IROS 2021 |
| SCPNet (base) | 36.7 | 56.1 | baseline | CVPR 2023 (frozen base; 36.7 hidden test, 36.17 val seq 08) |
| TALoS (prev. SOTA) | 37.9 | **60.2** | +1.2 | NeurIPS 2024, line-of-sight test-time adaptation |
| **S²D² (Ours, *N* = 1, no TTA)** | **38.8** | 58.9 | **+2.1** | The headline. Best single-frame single-sample result on the leaderboard to date (+0.9 over TALoS 37.9). 107.2 ms for the added pass (9.33 FPS marginal), 3.23 FPS end-to-end — below the sensor's 10 Hz, so not real-time |
| **S²D² (Ours, *D*<sub>4</sub> TTA)** | **39.2** | 59.0 | **+2.5** | *N* = 4 with an 8-view ensemble. EXCLUDED from the paper's single-sample predicate, so it carries no TALoS margin; leaderboard row public upon release |

<sub><b>Source.</b> The mIoU and IoU<sub>cmpl</sub> values for the baseline rows, and the IoU<sub>cmpl</sub> values for the S²D² <i>N</i>=1 / <i>D</i><sub>4</sub>-TTA rows, are the corresponding entries on the public SemanticKITTI SSC test leaderboard; they are not all reported in our paper. Only the S²D² mIoU column and the headline 39.2 % / 59.0 % <i>D</i><sub>4</sub>-TTA row are paper-reported (supplementary Tab. of test results).</sub>

On full SemanticKITTI **val** seq 08 (note: val numbers below, distinct from the 39.2 % **hidden-test** figure above):
* **val: 38.54 %** mIoU (single correction step, $N{=}1$, no TTA) — verified end-to-end by the maintainers (requires the released assets); the *same* $N{=}1$ no-TTA setting scores **38.8 %** on the hidden test (the 38.54 val / 38.8 test pair in the test table above) — to our knowledge the best single-frame single-sample result on the leaderboard to date (+0.9 over TALoS 37.9)
* **val: 38.73 %** mIoU (N=1 correction step + *D*<sub>4</sub> TTA) → the same recipe scores **39.2 %** on the hidden test
* **+2.36** absolute over our SCPNet base (36.17 % val)

<p align="center">
  <img src="assets/qualitative.png" width="92%" alt="Qualitative comparison on SemanticKITTI val seq 08 vs SOTA baselines" />
</p>

<sub>Two seq-08 frames where S²D² recovers a rare class the base SOTA misses entirely. Left → right: JS3C-Net, SCPNet, TALoS, S²D² (ours, <i>N</i>=4), Ground Truth. Source: paper Fig. 4.</sub>

### Per-class IoU on val seq 08 (single correction step, verified)

<sub>From `python scripts/eval.py eval/val_1step --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors`. Numbers below are the paper's per-class val row for this checkpoint (SCPNet + S²D², *N* = 1), as printed in the supplementary's full per-class table (`tab:supp_portable_full`).</sub>

| | car | bicycle | motorcycle | truck | other-veh. | person | bicyclist | motorcyclist | road | parking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **IoU** | 51.4 | 24.3 | 35.5 | 60.1 | 44.5 | 23.2 | 23.2 | 12.4 | 74.6 | 61.6 |

| | sidewalk | other-grnd | building | fence | vegetation | trunk | terrain | pole | traffic-sign | **mIoU** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **IoU** | 53.9 | 13.9 | 34.7 | 30.2 | 40.7 | 32.4 | 54.6 | 37.8 | 23.0 | **38.54** |

---

## Quick start (reproduce 38.54 % val in three commands)

```bash
# 1. Clone + install (uv recommended)
git clone https://github.com/BillyChern/GSSC-S2D2.git && cd GSSC-S2D2
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10 && uv sync && uv pip install spconv-cu126==2.3.8
#    ^ CUDA 11.8 users: use spconv-cu118==2.3.8 instead (see the spconv note below)

# 2. Pull pretrained checkpoint + SCPNet predictions
python scripts/download_assets.py --checkpoints --predictions
# → data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors           (~140 MB; full subdir ~265 MB — see docs/MODEL_ZOO.md)
# → data/scpnet_predictions/   (~178 GB real + synth; ~135 GB total for eval-only — see docs/DATASET.md)

# 3. Reproduce the headline 38.54 % val mIoU
python scripts/eval.py eval/val_1step \
    --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors
```

> **Note on `spconv`.** `spconv` is deliberately *not* pinned in `uv.lock` because it ships a CUDA-specific wheel; `uv sync` will not install it. Install it explicitly with the pinned, CUDA-coherent line shown above (`uv pip install spconv-cu126==2.3.8`). Note that the pinned `torch==2.4.0` PyPI wheel bundles its *own* CUDA 12.1 runtime libraries (`nvidia-*-cu12`), so torch carries cu121 regardless of your local toolkit; only the separately-installed `spconv` wheel must be matched to the local CUDA. The validated combination is **cu121-torch (2.4.0 stock wheel) + cu126-spconv (2.3.8)** — match the spconv wheel to your local CUDA toolkit if you deviate (e.g. for CUDA 11.8 use `uv pip install spconv-cu118==2.3.8`). Both libraries load on a ≥12.6 driver via CUDA minor-version forward compatibility, so the install line stays correct on the reference CUDA 12.8 box in the hardware table below. The stock PyPI wheel is sufficient: the SCPNet v1→v2 kernel-shape "patches" are applied in code at weight-load time, so no custom-built spconv is required.

Once the released assets are in place, that is the whole pipeline. Expected output (truncated):

```
[gssc.inference.evaluate] Eval config: eval/val_1step
[gssc.inference.evaluate] Sequences=08 steps=1 tta=none
[gssc.inference.evaluate] Stage 1 done. Predictions written under <tmpdir>
[gssc.inference.evaluate] Stage 2 (score): ... evaluate_completion.py ...
[gssc.inference.evaluate] mIoU         38.54 %
[gssc.inference.evaluate] IoU_cmpl     52.66 %

============================================================
 GSSC-S2D2 evaluation: eval/val_1step
============================================================
  mIoU       : 38.54 %
  Completion : 52.66 %   (val seq 08)
------------------------------------------------------------
 Per-class IoU:
  bicycle               24.30 %
  bicyclist             23.20 %
  car                   51.40 %
  motorcyclist          12.40 %
  pole                  37.80 %
  ...
============================================================
```

<sub>The per-class lines above are an abbreviated, illustrative excerpt; the
**38.54 % val mIoU** is the anchor number to check against. The eval script
prints the full 19-class table at runtime, so the exact log format may evolve
across releases — match on the mIoU value, not the verbatim layout.</sub>

<sub>Note: the `IoU_cmpl` 52.66 % here is on **val** seq 08, distinct from the 58.8–59.0 % `IoU<sub>cmpl</sub>` reported on the **hidden test** in the headline table above — the same val-vs-test split flagged for the mIoU column.</sub>

For the full hidden-test leaderboard submission flow (39.2 % via *D*<sub>4</sub> TTA), see [docs/INFERENCE.md](docs/INFERENCE.md).

### Secondary task: LiDAR-only BEV refinement

S²D² is not specific to 3D scene completion. The same correction-sampling
mechanism, applied to a 2D BEV diffusion model, refines the SCPNet-derived
base BEV map and lifts BEV mIoU from **34.8 %** (the parameter-free
base-derived projection) to **36.1 %** (S²D²-refined) on val seq 08, a
**+1.3 pp** gain — paper tab:bev_results. Both figures are scored by the
training-time 2D BEV evaluator on 100 fixed val frames (seed 42), *not* the
4,071-frame `semantic-kitti-api` protocol every 3D number in this README uses,
so they are not comparable with the 3D mIoU figures above.

```bash
# After step 3 above (predictions already downloaded), eval the BEV pipeline:
python scripts/eval.py eval/bev_secondary \
    --checkpoint data/checkpoints/bev/bev_s2d2_scpnet/model.safetensors
```

To retrain the BEV-A checkpoint from scratch (~24 h on 1× H100):

```bash
python scripts/train.py train/bev_secondary
```

### Cross-dataset zero-shot (KITTI-360, SemanticPOSS)

The headline SemanticKITTI checkpoint transfers to other LiDAR domains with **no
fine-tuning and no target labels** — the frozen `gssc_31k_mf_step40000`
checkpoint at N=1, single-frame, no TTA. On SSCBench-KITTI-360 val (seq 06,
1,812 frames, 16 shared classes; same-sensor near-domain) S²D² lifts the SCPNet
base from **5.8 → 6.2 mIoU (+0.4)** and **18.1 → 19.5 CompIoU (+1.4)**. On the
harder cross-sensor SemanticPOSS val (seq 02, ≈500 frames, 11-class TALoS Tab. 4
map) it lifts **1.0 → 6.5 mIoU (+5.5)** and **31.8 → 54.9 CompIoU (+23.1)**.

```bash
# SSCBench-KITTI-360 zero-shot (after provisioning per docs/DATASET.md)
python scripts/eval_kitti360.py eval/kitti360_zeroshot_1step \
    --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors

# SemanticPOSS zero-shot
python scripts/eval_semanticposs.py eval/semanticposs_seq02 \
    --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors
```

Both runs are frozen-checkpoint zero-shot: the SemanticKITTI-trained weights are
applied as-is, with no adaptation to the target domain.

---

## Repository layout

```
GSSC-S2D2/
├── src/gssc/                       # the Python package
│   ├── models/                     # denoiser: dense 3D U-Net (Conv3d, this release ≈ 35M); "Sparse" in class names (e.g. SceneCompletionUNetSparse) refers to the aux LiDAR encoder, not the denoiser. Plus SCPNet base, pyramid, BEV variant
│   ├── diffusion/                  # forward process, Dirac posterior, correction sampler, D4 TTA
│   ├── data/                       # SemanticKITTI loader, synthetic pool, object bank, HDL-64E ray-tracer
│   ├── losses/                     # KL posterior + Lovász + auxiliary + focal-CE
│   ├── training/                   # canonical trainer + EMA + logging
│   ├── inference/                  # eval (3D SSC mIoU + Completion IoU, 2D BEV mIoU), D4 TTA, prediction generation
│   └── utils/                      # config loader, seeding, registry
├── configs/                        # Hydra configs (one per recipe in the paper)
│   ├── train/{0K_sf,10K_sf,20K_sf,31k_sf,31k_mf,57K_sf,57k_mf,T10,T50,T100skewed,bev_secondary,c1_lossmatched_t99,js3c_real,js3c_real_derived,js3c_real_gtbev,lmscnet_real}.yaml
│   ├── eval/{val_1step,val_d4tta,step_sweep,timestep_ablation,data_scaling_sf,dwiou_sweep,round2_a,bev_secondary,js3c_val_1step,js3c_val_d4tta,js3c_val_paper,js3c_val_realistic,lmscnet_val_1step,kitti360_zeroshot_1step,semanticposs_seq02}.yaml
│   └── infer/{test_1step,test_d4tta,val_1step,val_d4tta}.yaml   # test_1step is the N=1 no-TTA hidden-test row
├── scripts/                        # one-command drivers
│   ├── train.py
│   ├── eval.py
│   ├── infer.py
│   ├── prepare_data.py
│   ├── download_assets.py
│   ├── reproduce_table.py
│   ├── eval_kitti360.py            # cross-dataset zero-shot
│   ├── eval_semanticposs.py        # cross-dataset zero-shot
│   ├── dump_js3c_predictions.py    # base-prediction dumpers
│   ├── dump_lmscnet_predictions.py
│   └── ...                         # fps_measure_dense_vs_sparse, score_kitti360, writeback_scpnet_baseline_kitti360, check_no_ai_attribution
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
├── assets/                         # README figures (teaser, architecture, qualitative)
├── data/                           # runtime dataset root (gitignored): checkpoints + base predictions land here
├── .github/                        # CI workflows (test, lint, release) + issue/PR templates
├── CITATION.cff                    # citation metadata
├── CONTRIBUTING.md                 # code-quality standards
├── CHANGELOG.md                    # release history
├── SECURITY.md                     # security policy
├── .pre-commit-config.yaml         # pre-commit hooks (ruff, mypy)
├── pyproject.toml                  # uv-managed Python project
├── uv.lock                         # pinned dependency lockfile
└── LICENSE                         # Apache-2.0
```

---

## Reproducing every paper number

The repo ships the exact recipe + checkpoint for every reported number.

| Paper artefact | Command | Expected |
|---|---|---|
| `tab:main_results` (test mIoU + per-class) | `python scripts/infer.py infer/test_d4tta --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors --output preds/` then submit to [Codabench](https://www.codabench.org/competitions/13814/#/results-tab) | **39.2 %** test mIoU |
| `tab:perclass_delta` (val per-class) | `python scripts/eval.py eval/val_1step --checkpoint <headline>` | **38.54 %** val mIoU |
| `tab:step_reduction` (step reduction) | `python scripts/eval.py eval/step_sweep --checkpoint <headline>` | 38.54 (N=1) … 38.65 (N=4 peak) … 38.16 (N=100) |
| single-frame data scaling | `python scripts/reproduce_table.py data_scaling_sf` | 0K/10K/20K/31K/57K SF retrains. Tab. VII is the multi-frame sweep and does not ship its per-row checkpoints |
| Training-timestep sweep — **no paper table**; the paper reports it in prose in supplementary App. C-A (`suppE:train_t`) | `python scripts/reproduce_table.py train_timesteps_curriculum` — a driver CLI key, not a paper label. It runs `eval/timestep_ablation` once per `gssc_timesteps/` checkpoint; the T=100-uniform row is the headline checkpoint. Paths in [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) | **38.54 %** (T=100 uniform) vs **38.2 %** (T=100 skewed) — the only pair the paper prints. T=10 (37.83) and T=50 (37.92) also ship, but the paper deliberately omits them as degenerate (β is fixed regardless of T, so a short schedule never reaches the source) |
| `tab:portable_s2d2` (JS3C-Net cross-base) | `python scripts/eval.py eval/js3c_val_realistic --checkpoint data/checkpoints/gssc_js3c/gssc_js3c_s2d2_real/model_ema.safetensors` (or `python scripts/reproduce_table.py tab:portable_s2d2`, which runs both cross-base rows) | **24.3 %** val mIoU (paper headline for this base: derived BEV under the official `semantic-kitti-api`, +1.6 pp over the 22.7 % base, and what this command yields). Diagnostics, not the headline: **26.05 %** with GT BEV under the official api, and **26.7 %** under the paper's internal training-time evaluator — see `docs/REPRODUCIBILITY.md` |
| `tab:portable_s2d2` (LMSCNet cross-base) | `python scripts/eval.py eval/lmscnet_val_1step --checkpoint data/checkpoints/gssc_lmsc/gssc_lmsc_s2d2_real/model_ema.safetensors` (or `python scripts/reproduce_table.py tab:portable_s2d2`, which runs both cross-base rows) | **16.6 %** val mIoU (derived BEV, official `semantic-kitti-api`; +1.8 pp over the 14.8 % on-disk-rescored base). The released LMSCNet `model_ema.safetensors` ships complete (278 tensors, 45 BN buffers) and reproduces 16.6 directly — see [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) |
| `tab:bev_results` (BEV second task) | `python scripts/eval.py eval/bev_secondary --checkpoint data/checkpoints/bev/bev_s2d2_scpnet/model.safetensors` | **36.1 %** BEV mIoU (34.8 % projection + 1.3), scored by the training-time 2D BEV evaluator on 100 fixed val frames (seed 42) — not the 4,071-frame `semantic-kitti-api` protocol used by every other row here |

Full mapping with anticipated wall-clock and disk requirements: **[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)**.

---

## Retraining the headline (≈ 37 GPU-hours to step 40000 on 2 × H100 80 GB)

```bash
python scripts/train.py train/31k_mf --gpu 0,1 --seed 42
# 100K iterations, batch size 4, AdamW lr 1e-4
# The released checkpoint is step 40000 (~37 GPU-h); running all 100K costs ~90 GPU-h
# Logs → outputs/train_31k_mf/{tensorboard,step_*.pt,best.pt}
```

A from-scratch seeded retrain lands at ≈ **38.05 % val 1-step mIoU** (success criterion 38.05 % ± 0.3 %, i.e. [37.8 %, 38.4 %]), within seed noise of the 38.54 % released headline checkpoint — reproduce the per-class delta over the SCPNet base, not the absolute (see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)). See [docs/TRAIN.md](docs/TRAIN.md) for every other recipe (data scaling, timestep ablations, pyramid diffusion, BEV second task).

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
| spconv | `spconv-cu126==2.3.8` (stock PyPI wheel; SCPNet v1→v2 kernel-shape patches applied in code at load time) | required |

Pinned versions in `uv.lock`. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the exact environment matrix.

---

## Three ideas behind S²D²

1. **Structured source.** Replace the noise endpoint with a learned base model's prediction $x_{\text{src}}$. The forward process becomes the Dirac mixture $x_t = \bar\alpha_t \cdot x_0 + (1 - \bar\alpha_t) \cdot x_{\text{src}}$, a deterministic interpolant between ground truth and $x_{\text{src}}$.
2. **S²D² correction sampling.** A non-noise deterministic reverse process (we specialise the non-noise correction sampler of Cold Diffusion, Bansal et al. 2022, to our linear simplex interpolant) that routes the full residual $\hat{\mathbf{x}}_0 - \mathbf{x}_{\text{src}}$ directly per step. At $N = 1$, the iterate at $t = T$ coincides with $\mathbf{x}_{\text{src}}$, giving a Lipschitz-free single-step bound (`cor:supp_onestep`, with the Lipschitz term priced in `cor:supp_lipprop`; supplementary App. B-C).
3. **Pyramid diffusion data augmentation.** A coarse-to-fine pyramid ($32^2{\times}4$ → $64^2{\times}8$ → $256^2{\times}32$) generates synthetic $(\text{sparse}, \text{complete})$ pairs. Combined with a 57,789-instance / 8-rare-class object bank (bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist, trunk) and HDL-64E Bresenham3D ray-tracing, this yields the 32,039-scene synthetic pool used by the headline configuration (pooled with the 19,130 real frames for a 51,169-frame total, a 2.67× expansion).

The mathematical derivations are in the supplementary, App. B (`suppB:math`): `prop:supp_forward`, `prop:supp_posterior`, `prop:supp_elbo`, `prop:supp_meanflow`, `thm:supp_error`, `cor:supp_onestep`, `cor:supp_margin`, `cor:supp_lipprop`, `prop:supp_proj`. The flow-matching correspondence (Lipman et al.) is argued in prose under *Velocity / mean-flow formulation* (App. B-B, `suppB:velocity`) — the paper states no numbered flow-matching proposition, so earlier revisions of this file pointed at a label that does not exist.

---

## Asset releases

| What | Where | Size |
|---|---|---|
| Pretrained checkpoints (18 subdirs in `gssc_mf/`, `gssc_sf/`, `gssc_js3c/`, `gssc_lmsc/`, `gssc_timesteps/`, `pyramid/`, `bev/`) | Hugging Face [`BillyChern/GSSC-S2D2-checkpoints`](https://huggingface.co/BillyChern/GSSC-S2D2-checkpoints) — see [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) | ≈ 4.9 GB |
| Base-model predictions (SCPNet, JS3C-Net, LMSCNet) for val + test | Hugging Face [`BillyChern/GSSC-S2D2-datasets`](https://huggingface.co/datasets/BillyChern/GSSC-S2D2-datasets) — or reproduce locally via `scripts/dump_{js3c,lmscnet}_predictions.py`, see [docs/DATASET.md](docs/DATASET.md) | ≈ 414 GB total (SCPNet ~178 GB + JS3C-Net ~190 GB + LMSCNet ~46 GB; only ~135 GB needed for the SCPNet eval-only headline) |
| Object bank (57,789 instances, 8 rare classes) | Hugging Face [`BillyChern/GSSC-S2D2-datasets`](https://huggingface.co/datasets/BillyChern/GSSC-S2D2-datasets) — see [docs/DATASET.md](docs/DATASET.md) | 448 MB |
| Synthetic pool (0K / 10K / 20K / 31K / 57K variants) | IEEE DataPort *(URL pending — see [docs/DATASET.md](docs/DATASET.md))* | ~128 GB (31K) – ~230 GB (57K), approx. |

These artefacts are released under two different terms. GSSC-authored code and the GSSC-trained model weights are Apache-2.0. The synthetic pool and object bank are derived from SemanticKITTI, which is distributed under CC-BY-NC-SA 4.0, so they inherit its non-commercial, share-alike restriction; the model weights, although Apache-2.0 as our contribution, were also trained on SemanticKITTI, so downstream use of the weights still carries that non-commercial caveat. SemanticKITTI raw data follows its own license (see [semantic-kitti.org](http://www.semantic-kitti.org/)). `scripts/download_assets.py` populates every entry automatically; `docs/DATASET.md` documents manual provisioning as an alternative.

---

## Code-quality + testing

This codebase targets production-grade standards. CI enforces a subset (`ruff check` and `mypy` on the public API surface, plus a torch-free `pytest` smoke pass — see `.github/workflows/`); the remaining checks below are run **locally** before release:

```bash
ruff check src/ tests/ scripts/      # style + import order   (CI-enforced)
mypy src                             # static types           (CI-enforced)
pytest tests/ -v                     # unit + smoke tests      (light subset CI-enforced)
pytest --cov=gssc --cov-report=term  # coverage (local)
vulture src/                         # dead code (local)
bandit -r src/                       # security (local)
```

Style conventions, commit conventions, and hard requirements: **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## FAQ

**Q. Why use this over running SCPNet alone?**
A. We add **+2.1 absolute mIoU** on the hidden test set with a single extra forward pass (107.2 ms marginal added-step latency on H100 — the same figure as the 9.33 FPS marginal throughput quoted above, since 1000 / 107.2 ≈ 9.33), no extra training data beyond what SCPNet was trained on, and no distillation. (+2.5 is the four-step, 8-view-ensemble row, so it is not what one extra pass buys.) The gains concentrate on safety-critical rare classes. For the **released checkpoint** these are motorcyclist +8.3, bicyclist +5.3, truck +5.3, other-vehicle +2.5, person +1.2 on val seq 08 (paper `tab:perclass_delta`, Released column). A separate, independent spconv-v2 *from-scratch retrain* (documented in `docs/REPRODUCIBILITY.md`, not the shipped release weights) holds most of that structure on a different seed — truck +5.4, bicyclist +5.2, other-vehicle +2.3, person +1.9, each within 0.7 pp of the released checkpoint — but **the motorcyclist gain does not reproduce**: it collapses from +8.3 to **+0.3**, and that single class accounts for 87 % of the retrain's lower overall delta (+1.9 against +2.36). Treat the rare-class gains as reproducible in structure, with motorcyclist as the documented seed-sensitive exception, not as a per-class guarantee (paper `tab:perclass_delta`, Retrain column).

**Q. Can S²D² be applied to a different base SSC network?**
A. Yes — the framework is base-model-agnostic. We provide a working SCPNet integration; switching to JS3C-Net or any other base requires only providing per-voxel categorical predictions as `x_src`. See [docs/BASELINES.md](docs/BASELINES.md).

**Q. Do we need the synthetic pool to use the released checkpoint?**
A. **No.** Eval-only deployment uses the released weights + SCPNet predictions only (~135 GB total). The synthetic pool (~128 GB for the 31K headline variant, ~230 GB for the 57K variant) is only needed for retraining from scratch.

**Q. Why does the train script use a YAML "config" rather than direct CLI args?**
A. Every paper artefact corresponds to a config file. `python scripts/train.py train/31k_mf` runs the exact headline recipe with no chance of accidentally diverging from the paper.

**Q. Single-seed numbers — why no error bars?**
A. Same convention as every method in the leaderboard table: a SemanticKITTI SSC training run is expensive (ours: ~37 GPU-hours to the released step-40000 checkpoint, ~90 for the full 100K-iteration launch), and the official scoring server takes a single submission. We use a single seed (42) to match this convention. See §V.A of the paper for the variance-disclosure discussion.

**Q. Where do I find figures and paper-typesetting source?**
A. The three README figures (`assets/teaser.png`, `assets/architecture.png`, `assets/qualitative.png`) live under `assets/` and are embedded above. The full paper figure set and the LaTeX typesetting source live in the (private until publication) paper repo, not here. This repo focuses on **method reproduction**; the shipped notebook (`examples/quickstart.ipynb`) walks through the headline eval end-to-end.

---

## Citation

```bibtex
@article{chen2026gssc,
  title   = {Generative Semantic Scene Completion},
  author  = {Chen, Shi and Ge, Weifeng},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (under review)},
  year    = {2026}
}
```

(Machine-readable: [`CITATION.cff`](CITATION.cff))

The TPAMI submission snapshot referenced in the paper supplementary is the **v2.3.8** release; its Hydra configs hold the same hyperparameters listed in the paper's reproducibility appendix. Keep this tag in step with the one named in the paper's reproducibility appendix — the paper cites the tag by name, so bumping one without the other strands the reference.

---

## License

* **Code, configs, documentation:** [Apache License 2.0](LICENSE).
* **Released model weights:** the GSSC-authored code and weights are licensed Apache-2.0. The weights were trained on SemanticKITTI, which is distributed under CC-BY-NC-SA 4.0 (non-commercial); downstream use of the weights therefore inherits that dataset's non-commercial restriction, so the Apache-2.0 grant on our contribution does not by itself authorise commercial use of the trained weights.
* **Synthetic pool + object bank:** derived from SemanticKITTI and therefore distributed under the same CC-BY-NC-SA 4.0 (non-commercial, share-alike) terms; see [semantic-kitti.org/dataset.html](http://www.semantic-kitti.org/dataset.html).
* **SemanticKITTI raw data:** governed by its own license — see [semantic-kitti.org](http://www.semantic-kitti.org/).
* **SSCBench-KITTI360 (evaluation-only):** governed by its own terms; see the [SSCBench repository](https://github.com/ai4ce/SSCBench).
* **SemanticPOSS (evaluation-only):** governed by its own terms; see the [SemanticPOSS dataset page](http://www.poss.pku.edu.cn/semanticposs.html).
* **Third-party code under `external/`:** retains its original license.

---

## Acknowledgements

Supported by NSFC Grant No. 624B1006. Thanks to the [SemanticKITTI](http://www.semantic-kitti.org/) authors for the public benchmark and raw data. Open-source software this project builds on:

* **Pyramid Discrete Diffusion** ([Liu et al., 2023](https://arxiv.org/abs/2311.12085)) — multi-scale discrete diffusion for 3D scene synthesis; foundation of our offline data augmentation pipeline (𝒮₁/𝒮₂/𝒮₃ + rare-class object bank + HDL-64E ray-tracing).
* **improved-diffusion** ([OpenAI, MIT](https://github.com/openai/improved-diffusion)) — diffusion training and sampling scaffolding our diffusion code adapts.
* **SCPNet** ([SCPNet/Codes-for-SCPNet](https://github.com/SCPNet/Codes-for-SCPNet)) — sparse-3D-CNN base SSC model used as the frozen headline base.
* **JS3C-Net** ([yanx27/JS3C-Net](https://github.com/yanx27/JS3C-Net)) — point-voxel hybrid base used for the cross-base lift.
* **LMSCNet** ([astra-vision/LMSCNet, Apache-2.0](https://github.com/astra-vision/LMSCNet)) — dense 2D-CNN base used for the third cross-base lift.
* **D3PM / Multinomial Diffusion** ([Austin et al., NeurIPS 2021](https://arxiv.org/abs/2107.03006)) — discrete diffusion family that S²D² generalises with a structured source.
* **Cold Diffusion** ([Bansal et al., 2022](https://arxiv.org/abs/2208.09392)) — non-noise correction sampling whose linear-simplex specialisation underlies our $N=1$ deployment.
* **spconv 2.3** ([traveller59/spconv](https://github.com/traveller59/spconv)) — sparse 3D convolution backend.
* **TALoS** ([NeurIPS 2024](https://arxiv.org/abs/2410.15674)) — previous SemanticKITTI SSC SOTA, included as the leaderboard reference baseline.

---

## ⭐ Star history

<a href="https://www.star-history.com/#BillyChern/GSSC-S2D2&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=BillyChern/GSSC-S2D2&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=BillyChern/GSSC-S2D2&type=Date" />
    <img alt="Star history" src="https://api.star-history.com/svg?repos=BillyChern/GSSC-S2D2&type=Date" />
  </picture>
</a>

---

<div align="center">

If GSSC-S2D2 helped your research, please ⭐ the repo and cite the paper.

</div>

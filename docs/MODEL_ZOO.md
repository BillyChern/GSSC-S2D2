# Model Zoo

All checkpoints are released under Apache-2.0 and mirrored on the Hugging Face
Hub at [`BillyChern/GSSC-S2D2-checkpoints`](https://huggingface.co/BillyChern/GSSC-S2D2-checkpoints).
Fetch them with `python scripts/download_assets.py --checkpoints` (~4 GB total);
`docs/DATASET.md` documents manual provisioning as the alternative route.

## Layout (since v1.1.0)

Each release checkpoint lives in a per-checkpoint subdir matching the modern
Hugging Face / diffusers convention:

```
data/checkpoints/<group>/<name>/
├── model.safetensors        # training weights (resumable from this)
├── model_ema.safetensors    # deployment weights (paper convention, default for inference)
└── config.json              # train cfg + best_miou + global_step + source SHA256
```

`model.safetensors` is a complete model state_dict, so
`load_state_dict(strict=True)` works out of the box on it. `model_ema.safetensors`
holds the EMA-tracked parameters; the released EMA files ship the full state
including BatchNorm running buffers (278 tensors, the same key set as
`model.safetensors`), so they load under `strict=True`, but the usage snippet
below keeps `strict=False` as a forward-compatible default. For exact
reproduction prefer `scripts/eval.py`, which wires the EMA weights in the same
way the paper numbers were produced.

> **Note — LMSCNet `model_ema.safetensors` ships complete.** The LMSCNet
> cross-base checkpoint
> (`gssc_lmsc/gssc_lmsc_s2d2_real/model_ema.safetensors`) ships the full 278
> tensors, including all 45 BatchNorm `running_mean` / `running_var` /
> `num_batches_tracked` buffers, so it loads cleanly and reproduces **16.59 %
> val mIoU** (+1.8 over the 14.76 % LMSCNet base). No buffer-completion step is
> needed.

**Download/disk sizes.** A single `model_ema.safetensors` (the deployment file
the quickstart in `README.md` points at) is **~140 MB**. `download_assets.py`
provisions the **whole per-checkpoint subdir** — `model.safetensors` +
`model_ema.safetensors` + `config.json` — which is **~265 MB total**. The "Size"
column below and the "~265 MB" figures in this doc refer to the full subdir; the
"~140 MB" figure in `README.md`'s quickstart refers to the single
`model_ema.safetensors` it downloads for inference.

The legacy SCPNet base ships as a flat `scpnet_v2_port.pth` (third-party
convention, kept as-is).

## Headline scene-completion checkpoints

Each result is keyed by its stable paper `\label` (the rendered Roman table
numbers drift between revisions; the labels do not).

The `mf` / `sf` tags in the subdir names mark the training regime: **`mf`** =
multi-frame-trained with single-frame input at inference (the headline regime);
**`sf`** = the single-frame data-scaling retrains, which are a *companion* to
`tab:data_scaling` and not its source. `tab:data_scaling` itself reports the
headline multi-frame-trained regime — see the single-frame section below, and
`docs/REPRODUCIBILITY.md` ("Tab. VII is the MULTI-frame sweep").

| Subdir | Paper label | Val mIoU | Test mIoU | Config | Size (full subdir) |
|---|---|---|---|---|---|
| `gssc_mf/gssc_31k_mf_step40000/` | **Headline** (`tab:portable_s2d2`) | 38.54 | 38.8 (N=1, no TTA) / 39.2 (+D4 TTA) | `configs/train/31k_mf.yaml` | ~265 MB |
| `gssc_mf/gssc_57k_mf_step40000/` | internal / unreported (in no paper table; `tab:data_scaling`'s 57K row is the multi-frame **38.4**, measured on the development codebase, and this released checkpoint is a different run — do not read 37.76 as that cell) | 37.76 (N=1) | — | `configs/train/57k_mf.yaml` | ~265 MB |

## Cross-base portability (paper tab:portable_s2d2, three frozen-base rows)

The same recipe and hyperparameters applied to three structurally different
frozen base models lifts every one of them. LMSCNet and JS3C-Net ship as
released checkpoints; SCPNet uses the same training recipe with
`configs/train/31k_mf.yaml` and lands the headline `gssc_mf/gssc_31k_mf_step40000`.

| Subdir | Base | Architecture family | Base mIoU | +S²D² mIoU | Δ | Config |
|---|---|---|---|---|---|---|
| `gssc_lmsc/gssc_lmsc_s2d2_real/` | LMSCNet | 2D CNN (dense)        | 14.8 | **16.6** | **+1.8** | `configs/train/lmscnet_real.yaml` |
| `gssc_js3c/gssc_js3c_s2d2_real/` | JS3C-Net | Point + voxel hybrid | 22.7 | **24.3** | **+1.6** | `configs/train/js3c_real.yaml`    |
| (uses `gssc_mf/gssc_31k_mf_step40000/`) | SCPNet | Sparse 3D CNN       | 36.17 | **38.54** | **+2.36** | `configs/train/31k_mf.yaml`       |

> **JS3C-Net number convention.** The JS3C row leads with **24.3 % (+1.6 pp)**,
> derived BEV under the official `semantic-kitti-api`, because that is the row
> the paper cites as its JS3C-Net headline and it is what
> `scripts/reproduce_table.py` yields end-to-end. It also matches the released
> checkpoint's training distribution, which used derived BEV
> (`configs/train/js3c_real.yaml` sets `bev_from_base: true`).
>
> Earlier revisions of this file led with **26.1 % (+3.3 pp)** and called it
> "the paper headline". That was wrong on both counts: the paper's JS3C headline
> is 24.3 %, and the string 26.1 appears **nowhere** in the paper or its
> supplement, so it cannot be a rounding of anything the paper prints. The
> figures the paper does carry for this base, besides 24.3, are diagnostics its
> supplement labels row by row and explicitly says are not protocol-matched to
> the SCPNet and LMSCNet rows — read them off that table rather than from here,
> since this file has twice mis-stated which protocol produced which.

> **Note on the "Architecture family" column.** This column describes the
> **frozen base model** (the predictor S²D² corrects), not the S²D² denoiser
> itself. The S²D² denoiser is a **dense `Conv3d` U-Net for every row**
> regardless of the base's architecture (see the "Architecture note" section
> below). So "Sparse 3D CNN" / "2D CNN (dense)" / "Point + voxel hybrid" refer
> to LMSCNet / JS3C-Net / SCPNet, not to the released checkpoint's denoiser.

Each cross-base checkpoint subdir is ~265 MB total (~140 MB for the single
`model_ema.safetensors` alone). The SCPNet (36.17 → 38.54, +2.36) and LMSCNet
(14.8 → 16.6, +1.8; LMSCNet base re-scored from on-disk predictions, superseding
the earlier 12.10 → 16.59 / +4.49 summary) deltas in the table are measured
end-to-end under the official `semantic-kitti-api`. The released LMSCNet
`model_ema.safetensors` ships complete (278 tensors, including all 45 BatchNorm
running buffers), loads cleanly, and reproduces the 16.59 figure directly.

The JS3C-Net row carries three numbers; the table leads with the paper's:

- **24.32 % (+1.59 pp)** — the **paper headline** for this base, and the
  reproducible at-deploy figure: derived BEV under the official
  `semantic-kitti-api`, protocol-matched to the 22.7 % base and to the released
  checkpoint's own training distribution. This is what
  `scripts/reproduce_table.py` yields end-to-end.
- **26.05 % (+3.32 pp)** — a GT-BEV diagnostic under the official
  `semantic-kitti-api`. **Not** the headline.
- **26.72 % (+3.99 pp)** — scored with the paper's **internal training-time
  evaluator** (`SSCMetrics`). A continuity row, **not** the headline.

For which protocol produced 26.05 versus 26.72, read the paper's supplementary
validation-protocol table rather than this file: its rows carry an explicit
Evaluator column, and this file has twice described the pairing wrongly.

SCPNet's 38.54 is an official `semantic-kitti-api` number everywhere it
appears. The JS3C-Net derived-BEV deploy number (**24.32**) is produced by
`eval/js3c_val_realistic.yaml`:
- `eval/js3c_val_paper.yaml` reproduces the GT-BEV protocol by loading
  preprocessed GT BEV via the config key `bev_source: gt` (set in
  `configs/eval/js3c_val_paper.yaml`; it is a YAML key, not a CLI flag). Under
  the official `semantic-kitti-api` this lands at the **26.05 %** diagnostic; the
  paper's internal SSCMetrics on the identical protocol reads **26.72 %**.
- `eval/js3c_val_realistic.yaml` uses derived BEV (topmost-non-empty class
  from JS3C-Net's 3D prediction, selected via the config key
  `bev_source: derived`) — the honest deploy-time number (**24.32 %**). The
  released JS3C+S²D² model was trained with derived BEV, so this protocol
  matches its training distribution.

Reproduction requires `data/js3cnet_predictions/` (190 GB real + synth; download via
`scripts/download_assets.py --js3c-predictions` or dump locally via
`scripts/dump_js3c_predictions.py`; see `docs/REPRODUCIBILITY.md`).

## Single-frame data-scaling companion sweep

These are single-frame-**trained** retrains that sweep the synthetic-pool volume.
They are a companion to — not the source of — the paper's `tab:data_scaling`
(supplementary **Table VII**, App. C-B), which reports the **headline** configuration (multi-frame-trained,
`T=100`-uniform, `N=1` deployment): 0K 37.7 → 10K 38.1 → 20K 38.3 → **32K 38.54
(headline)** → 57K 38.4, monotonically increasing through 32K. The `N=1` column
below is each single-frame checkpoint's own measured value, distinct from the
multi-frame `tab:data_scaling` cells above. The paper carries this single-frame
sweep as prose in the same appendix subsection (App. C-B), not as a table: it
prints 0K 38.2, 10K/20K 38.1, 32K 38.4, 57K 37.7, which is what the rows below
round to.

| Subdir | Synthetic pool | Val mIoU (N=1 / peak) | Config |
|---|---|---|---|
| `gssc_sf/gssc_0K_sf_step93000/`  | None (real only)         | 38.18 / 38.46 (N=5)  | `configs/train/0K_sf.yaml`  |
| `gssc_sf/gssc_10K_sf_step87000/` | 10K synthetic            | 38.06 / 38.50 (N=10) | `configs/train/10K_sf.yaml` |
| `gssc_sf/gssc_20K_sf_step85000/` | 20K synthetic            | 38.14 / 38.49 (N=5)  | `configs/train/20K_sf.yaml` |
| `gssc_sf/gssc_31K_sf_step72000/` | 31K synthetic            | 38.42 / 38.49 (N=2-5)| `configs/train/31k_sf.yaml` |
| `gssc_sf/gssc_57K_sf_step69000/` | 57K synthetic            | 37.66 / 38.05 (N=5)  | `configs/train/57K_sf.yaml` |

> **Copy these names verbatim — the casing is intentionally mixed.** The
> checkpoint subdirs use an uppercase `K` (e.g. `gssc_31K_sf_step72000`),
> but the `31K` row's training config is lowercase, `configs/train/31k_sf.yaml`
> (the `0K`/`20K`/`57K` configs keep the uppercase `K`). Do not "normalize" the
> case by analogy or you will hit a missing-file error.

## Training-timestep ablations (internal runs; no paper table)

> **No paper table backs these three rows, and the supplementary table label
> earlier revisions of this file cited does not exist in the paper at all.**
> Supplementary App. C-A reports only the two
> `T=100` schedules — the uniform headline at 38.54 % against a `t=T`-skewed
> variant at 38.2 % — and says the shorter schedules were *trained and omitted
> rather than reported*, because the implementation fixes β linear on
> [1e-4, 0.1] irrespective of `T`, so a smaller `T` never drives the forward
> process to the source and the model is then queried from a state it never
> saw. Treat the `T=10` / `T=50` checkpoints below as internal ablation runs,
> not as paper values.

| Subdir | Schedule | Val mIoU (internal) | Config |
|---|---|---|---|
| `gssc_timesteps/gssc_T10/`         | T=10 uniform            | 37.83 | `configs/train/T10.yaml`         |
| `gssc_timesteps/gssc_T50/`         | T=50 uniform            | 37.92 | `configs/train/T50.yaml`         |
| `gssc_timesteps/gssc_T100skewed/`  | T=100 skewed (t=T heavy)| 38.18 | `configs/train/T100skewed.yaml` |

## Pyramid diffusion (offline data augmentation)

| Subdir | Resolution | Purpose |
|---|---|---|
| `pyramid/pyramid_s1/` | 32×32×4    | Coarse scene generator |
| `pyramid/pyramid_s2/` | 64×64×8    | Mid-resolution refiner |
| `pyramid/pyramid_s3/` | 256×256×32 | Final-resolution generator (used to produce the 32,039-frame synthetic pool; `31K` is the historical `synthetic_pool_31K` dir label) |

Pyramid checkpoints do not use EMA; each subdir ships
`model.safetensors` + `config.json` only.

## BEV second task (tab:bev_results)

| Subdir | Task | Pipeline mIoU (training-time 2D BEV evaluator, 100 fixed val samples, seed 42) | Config |
|---|---|---|---|
| `bev/bev_s2d2_scpnet/` | LiDAR-only BEV refinement (2D S²D² on the base-derived BEV) | **36.1** = 34.8 parameter-free projection + 1.3 refinement | `configs/train/bev_secondary.yaml` |
| `bev/bev_direct_l3_deeper/` | Supp BEV ablation (deeper 3D-direct baseline) | n/a (ablation only) | one-off internal ablation; recipe not released (no shipped config) |

**Read the protocol before comparing this row to anything.** The BEV numbers are
scored by the training-time 2D BEV evaluator on 100 fixed val samples (seed 42),
over the 19 evaluation classes with class 0 excluded — a different instrument
from the full-validation 3D scoring used everywhere else in this file, and the
paper says so in its own supplementary BEV appendix. The two are not
commensurable; do not read 36.1 as a 3D-protocol number. The checkpoint's
`config.json` records the same protocol string and the unrounded pair it
produced (internal measurement: 34.75 base projection, 36.09 refined).

> **The BEV row moved checkpoints in this release.** Earlier revisions pointed
> this row at `bev/bev_perception_net/` and quoted a "34.3 base + 1.8
> refinement" split. Both were wrong: `bev_perception_net` is a different model
> from an earlier BEV experiment, it did not produce the paper's row, and the
> BEV evaluator cannot even load it
> under this release's config (its own `config.json` describes it as a
> lightweight 938K-param 2D refinement net). The model behind the paper's row is
> `bev/bev_s2d2_scpnet/`, and the split the paper prints is 34.8 + 1.3.

Run it with the explicit checkpoint path:

```bash
python scripts/eval.py eval/bev_secondary \
    --checkpoint data/checkpoints/bev/bev_s2d2_scpnet/model.safetensors
```

## SCPNet base (frozen)

| File | What | Notes |
|---|---|---|
| `scpnet_v2_port.pth` | SCPNet pretrained weights, ported to spconv v2.3 with kernel-shape patches. | Loads via `gssc.inference.run_scpnet`. Third-party flat `.pth` (not converted). |

## Architecture note (released checkpoint = paper denoiser)

The released checkpoint **is** the paper's denoiser as specified in the paper
Method section and the supplementary hyperparameter table: a **4-level dense 3D
U-Net** built from `nn.Conv3d` / `nn.ConvTranspose3d` (~35M parameters) with
additive L/B conditioning and time-AdaGN at every level. This matches the
repo's own `README.md` architecture caption ("dense 3D U-Net (Conv3d) … this
release ≈ 35M") and the released notebook.

The denoiser class is named `SceneCompletionUNetSparse`, but the word "Sparse"
in the class name refers only to the **auxiliary LiDAR encoder**
(`SparseLiDAREncoder`, which uses spconv), **not** to the denoiser body. The
denoiser does not use sparse convolutions.

The released code matches the paper's Method section and Fig. 3 caption exactly:
a dense `Conv3d` denoiser (~35M parameters) with additive L/B conditioning and
AdaGN-style time conditioning at every level. No sparse-SubMConv3d denoiser
variant is shipped.

## How to use

There is no `from_config` classmethod and no `configs/model/` directory; the
model is instantiated directly with the same constructor arguments the eval
path uses (see `src/gssc/inference/generate_predictions.py`):

```python
from pathlib import Path
from safetensors.torch import load_file
from gssc.models.s2d2_unet import SceneCompletionUNetSparse

# Deployment uses model_ema.safetensors (EMA weights; paper convention).
ckpt_dir = Path("data/checkpoints/gssc_mf/gssc_31k_mf_step40000")
state = load_file(ckpt_dir / "model_ema.safetensors")

# Same constructor the inference pipeline uses for the released checkpoints.
# "Sparse" names the LiDAR encoder; the denoiser itself is dense Conv3d.
model = SceneCompletionUNetSparse(
    num_classes=20,
    base_channels=32,
    time_emb_dim=128,
    lidar_base_channels=16,
    lidar_out_channels=32,
    lidar_in_channels=1,
    ssc_cond_channels=20,
    # no_bev / ssc_multiscale left at their constructor defaults (False) here;
    # the real inference path (generate_predictions.py) wires no_bev=args.no_bev,
    # ssc_multiscale=args.ssc_multiscale, and scripts/eval.py passes the production
    # values for exact reproduction — see below.
)
model.load_state_dict(state, strict=False)  # EMA files omit some buffers; for exact reproduction prefer scripts/eval.py
model.train(False)
```

Or via the eval entry point::

```bash
# SCPNet headline (38.54% val)
python scripts/eval.py eval/val_1step \
    --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors

# JS3C-Net cross-base (24.3% val is the paper headline for this base, derived BEV;
# 26.05% is a GT-BEV diagnostic, NOT the headline, and 26.1 is in no paper table;
# 26.72% internal training-time evaluator continuity row; 24.32% at-deploy derived BEV)
python scripts/eval.py eval/js3c_val_1step \
    --checkpoint data/checkpoints/gssc_js3c/gssc_js3c_s2d2_real/model_ema.safetensors

# LMSCNet cross-base (16.59% val, paper rounds to 16.6; +1.8 pp over the 14.76% on-disk-rescored base)
# The released LMSCNet model_ema.safetensors ships complete (278 tensors, 45 BN
# buffers) and reproduces 16.59 directly.
python scripts/eval.py eval/lmscnet_val_1step \
    --checkpoint data/checkpoints/gssc_lmsc/gssc_lmsc_s2d2_real/model_ema.safetensors
```

Or reproduce a specific paper table with the all-in-one driver::

```bash
# The strings below are reproduce_table.py CLI KEYS, not paper label names.
# The driver's tab:perclass key resolves to the paper's per-class val table
# (labelled tab:perclass_delta, main Tab. II); its tab:cross_base_js3c and
# tab:cross_base_lmsc keys resolve to per-base rows of tab:portable_s2d2.
python scripts/reproduce_table.py tab:perclass             # 38.54% val (paper tab:perclass_delta)
python scripts/reproduce_table.py tab:cross_base_js3c      # 24.3% val, the paper headline (derived BEV, tab:portable_s2d2)
# BEV: 36.1, measured by the training-time 2D BEV evaluator on 100 fixed val
# samples (seed 42) — see the BEV section above before quoting it.
python scripts/reproduce_table.py tab:bev_results
```

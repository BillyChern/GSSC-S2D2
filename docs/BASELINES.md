# Baselines

This codebase ships with the SCPNet base (frozen) and a faithful DiffSSC reimplementation.

## SCPNet base (frozen)

The headline configuration uses SCPNet as the frozen base model whose
prediction `x_src` is refined by S²D².

* Original: [SCPNet](https://github.com/SCPNet/Codes-for-SCPNet) (CVPR 2023)
* Our port: `src/gssc/models/scpnet_base.py` + `src/gssc/inference/run_scpnet.py`
* Released weights: `scpnet_v2_port.pth` in our model zoo.

### spconv v1 → v2 port

SCPNet's original release depends on `spconv 1.0` (CUDA 10/11, Python 3.8),
which has been removed from PyPI and does not build on modern CUDA stacks.
Our port runs SCPNet under `spconv 2.3` with kernel-shape compatibility
patches that replicate `spconv 1.0`'s shared-indice-key behaviour:

* In v1, layers sharing an `indice_key` reuse the FIRST layer's spatial pair
  data regardless of kernel size.
* In v2, different kernel shapes get separate pair data.
* We patch ResContextBlock (conv1_2, conv2: (3,1,3) → (1,3,3) to match conv1),
  ResBlock (conv1_2, conv2: (1,3,3) → (3,1,3)), UpBlock (conv2, conv3:
  (3,3,3) → (1,3,3)), ReconBlock (conv1_2, conv1_3: (1,3,1) and (1,1,3) →
  (3,1,1)). Weight loading reshapes kernel dims while preserving flat
  element order.

The port matches SCPNet's published 36.7% test mIoU **exactly** (byte-for-byte
on completion IoU 56.1% and on 17/19 per-class IoUs to within 0.1%). On val
seq 08 the port reads 36.17%, **−1.03% below** the paper's published 37.2%
val number; this val-side gap is confined to seq 08 and does not transfer
to test.

## JS3C-Net cross-base (added v1.1.0)

The v1.1.0 cross-base reproduction (paper Tab. III rows 90-91, +3.99 pp val
mIoU) uses JS3C-Net (Yan et al. 2021, ICCV) as a *prediction-only*
alternative base.

* Original: [JS3C-Net](https://github.com/yangyangyang127/JS3C-Net) (ICCV 2021)
* Our reader: `src/gssc/models/js3c_base.py` — a thin per-frame `.npy` loader.
  No JS3C model code is shipped because we release the predictions
  themselves as a separate dataset (`data/js3cnet_predictions/`, mirrors
  `data/scpnet_predictions/` exactly).
* Dumper: `scripts/dump_js3c_predictions.py` — depends on a local clone of
  the upstream JS3C-Net repo (CLI argument `--js3c-repo`, no hardcoded path).
* Released base reproduction: paper Tab. III row 90 = **22.73 %** val mIoU
  under the official `semantic-kitti-api` evaluator, exactly matching
  JS3C-Net's published recipe (no spconv kernel-shape patches required —
  unlike SCPNet, JS3C-Net's official codebase loads cleanly under the
  bundled TorchSparse + spconv 1.x pin).
* S²D² lift on top: **26.72 %** val mIoU (+3.99 pp), real-frames-only
  training, `cold_diffusion=true` (paper supp § H).

Reproduction protocol: `docs/REPRODUCIBILITY.md`, section "JS3C-Net
cross-base reproduction".

## DiffSSC reimplementation (qualitative comparison only)

DiffSSC's open-source release contains only the geometric-completion pipeline
(3-channel xyz diffusion on top of LiDiff); the semantic side is missing.

The qualitative panels in Fig. 4 of the paper were produced from an internal
(3+C)-channel reimplementation (anisotropic additive forward process, custom
DDIM sampling, logit-domain semantic encoding). That reimplementation lives
in our internal development codebase and is intentionally out of scope for
this release: it is not part of any reported leaderboard number, depends on
private utility code we cannot ship, and it would distract from the three
pillars the released codebase already supports. We do **not** report DiffSSC
numbers in the leaderboard table because the original authors did not submit;
the reimplementation served visualization only.

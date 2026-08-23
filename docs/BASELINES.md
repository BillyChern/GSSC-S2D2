# Baselines

This codebase ships three frozen base predictors — SCPNet (weights ported to
spconv v2.3), JS3C-Net, and LMSCNet (the latter two as prediction readers) —
and the S²D² refinement on top of each. Each base has its own section below.
The DiffSSC reimplementation is **not** shipped (out of scope for this release;
see the section below). It is not the source of any panel of the paper's
qualitative comparison, Fig. 6, whose columns are JS3C-Net, SCPNet, TALoS,
S²D² and ground truth; DiffSSC appears in no figure of the paper; its only tabulated entry is a row of main Tab. I.

## SCPNet base (frozen)

The headline configuration uses SCPNet as the frozen base model whose
prediction `x_src` is refined by S²D².

* Original: [SCPNet](https://github.com/SCPNet/Codes-for-SCPNet) (CVPR 2023)
* Our port + spconv patches: `src/gssc/inference/run_scpnet.py`
* Released weights: `scpnet_v2_port.pth` in our model zoo — SCPNet's own
  pretrained weights carried through our spconv v2 port, redistributed with
  the SCPNet authors' permission. They ship under this repository's MIT
  licence, but the model is Xia et al.'s work: cite the CVPR 2023 paper if
  you use it. See `docs/MODEL_ZOO.md` ("SCPNet base") for the full
  attribution note.

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

The port has **never been scored on the hidden test set.** We did not submit it to
the evaluation server, so no test measurement of the bare port exists, and every
36.7% figure quoted for SCPNet here and in the paper is *SCPNet's own published
number*, not ours. On val seq 08 the port reads 36.17%, **−1.03% below** the
paper's published 37.2% val number. Whether that val shortfall would also appear
on test is **unknown**: it cannot be checked without a submission we never made.
Earlier revisions of this file claimed the port "matches 36.7% exactly" with
per-class agreement, and that the val gap "does not transfer to test". Both
asserted a measurement that does not exist, and contradicted the paper, which
states in five places that we hold no test-server score for the bare port.

## JS3C-Net cross-base (added v1.1.0)

The v1.1.0 cross-base reproduction (paper tab:portable_s2d2, whose headline for
this base is **+1.6 pp** val mIoU with derived BEV) uses JS3C-Net
(Yan et al. 2021, AAAI) as a *prediction-only* alternative base. The +3.32 pp
figure this section used to lead with is our own GT-BEV diagnostic delta, not a
value the paper prints, and it is a different protocol from the +4.0 pp
internal-evaluator continuity row; see the three-number box below.

* Original: [JS3C-Net](https://github.com/yanx27/JS3C-Net) (AAAI 2021)
* Our reader: `src/gssc/models/js3c_base.py` — a thin per-frame `.npy` loader.
  No JS3C model code is shipped because we release the predictions
  themselves as a separate dataset (`data/js3cnet_predictions/`, mirrors
  `data/scpnet_predictions/` exactly).
* Dumper: `scripts/dump_js3c_predictions.py` — depends on a local clone of
  the upstream JS3C-Net repo (CLI argument `--js3c-repo`, no hardcoded path).
* Released base reproduction: **22.73 %** val mIoU (paper tab:portable_s2d2,
  base row) under the official `semantic-kitti-api` evaluator, exactly matching
  JS3C-Net's published recipe (no spconv kernel-shape patches required —
  unlike SCPNet, JS3C-Net's own upstream codebase loads cleanly under its
  published spconv 1.x stack, so we run the dumper against an unmodified clone
  rather than bundling any base-model dependency in this release).
* S²D² lift on top (paper headline): **22.7 → 24.3 % (+1.6 pp)** val mIoU under the
  official `semantic-kitti-api` with derived BEV, real-frames-only training,
  `cold_diffusion=true`. Derived BEV is the protocol-matched one: it is what the
  paper cites for this base and what the released checkpoint was trained under
  (`configs/train/js3c_real.yaml` sets `bev_from_base: true`).

  > **Three JS3C numbers, and which protocol each belongs to.** All three come
  > from the same released checkpoint, `gssc_js3c/gssc_js3c_s2d2_real`. **What
  > separates 26.7 from 24.3 is the evaluator, not the BEV source** — both are
  > derived BEV.
  > - **22.7 → 24.3 % (+1.6 pp)** — the **paper headline** for this base
  >   (main Tab. III, `tab:portable_s2d2`): **derived BEV**, scored by the
  >   **official `semantic-kitti-api`**, the same evaluator that scores the
  >   22.7 % base, so the delta is protocol-consistent. Precise outputs
  >   **22.73 % → 24.32 % (+1.59 pp)**. Reproduced by `eval/js3c_val_realistic`.
  > - **26.7 % (+4.0 pp)** — the **same derived-BEV setting** scored with the
  >   paper's **internal training-time evaluator** (`SSCMetrics`); precise
  >   internal output **26.72 %**. This is the row the supplementary
  >   validation-protocol table (`tab:supp_b6_val`) labels *"real-only, derived
  >   BEV, internal eval"*, and the supplement retains it for continuity only.
  >   It is **not** a GT-BEV number.
  > - **26.05 %** — a **GT-BEV diagnostic** of ours: `eval/js3c_val_paper`
  >   (= `eval/js3c_val_1step`) sets `bev_source: gt`, conditioning on a
  >   ground-truth BEV oracle, and is scored by the official
  >   `semantic-kitti-api`. **The paper does not print it** — neither "26.05"
  >   nor "26.1" appears in the paper or its supplement — and it is **not** the
  >   protocol behind 26.7. Earlier revisions of this file called it the paper
  >   headline and said the paper "rounds it to 26.1"; that was doubly wrong.
  >
  > The paper's own GT-BEV entry is a different thing again: the 61.8 % row of
  > `tab:supp_b6_val` is a *separately trained* GT-BEV variant, labelled there
  > as an upper bound rather than a deployment number, and it is not the
  > released checkpoint.
  >
  > See `docs/MODEL_ZOO.md` and `docs/REPRODUCIBILITY.md` for the full
  > disclosure.

Reproduction protocol: `docs/REPRODUCIBILITY.md`, section "JS3C-Net
cross-base reproduction".

## LMSCNet cross-base (added v2.1.0)

LMSCNet (Roldão et al. 2020, 3DV) is the third structurally different frozen
base alongside SCPNet (sparse 3D CNN) and JS3C-Net (point-voxel hybrid): a
lightweight (~0.4M-param) dense 2D-CNN that treats the Z=32 axis as input
channels. The v2.1.0 cross-base reproduction (paper tab:portable_s2d2, third
base, +1.8 pp val mIoU) uses LMSCNet as a *prediction-only* alternative base.

* Original: [LMSCNet](https://github.com/astra-vision/LMSCNet) (3DV 2020)
* Our reader: `src/gssc/models/lmscnet_base.py` — a thin per-frame `.npy`
  loader. No LMSCNet model code is shipped because we release the predictions
  themselves as a separate dataset (`data/lmscnet_predictions/`, mirrors
  `data/scpnet_predictions/` exactly).
* Dumper: `scripts/dump_lmscnet_predictions.py` — depends on a local clone of
  the upstream LMSCNet repo (CLI argument `--lmscnet-repo`, no hardcoded path;
  `--weights` is accepted as an alias for `--checkpoint`).
* Released base reproduction: **14.76 %** val mIoU (paper tab:portable_s2d2,
  base row; **14.8 %** rounded), re-scored from on-disk predictions through the
  official `semantic-kitti-api` evaluator — this supersedes the earlier 12.10 %
  summary (on-disk artifacts are authoritative). No spconv kernel-shape patches
  are required (LMSCNet is a plain dense 2D CNN, so there is no spconv
  v1 → v2 weight-loading concern as with SCPNet).
* S²D² lift on top: **16.59 %** val mIoU (paper rounds to **16.6 %**; **+1.8 pp**
  over the 14.76 % on-disk-rescored base), real-frames-only training,
  `cold_diffusion=true`. NOTE: the released LMSCNet `model_ema.safetensors` ships
  complete (278 tensors, 45 BN buffers) and reproduces 16.59 directly; no
  full-state-checkpoint workaround is needed. 16.59 is a **step-65,000** figure —
  the released checkpoint is the best-mIoU selection at `global_step: 65000`,
  even though `configs/train/lmscnet_real.yaml` sets `num_iterations: 100000`.
* Unlike the JS3C-Net row, LMSCNet has no GT-BEV vs. derived-BEV split: the
  seed BEV is always height-pooled from LMSCNet's own 3D prediction
  (`bev_from_base: true`, never GT BEV), so 16.59 % is already the at-deploy
  number with no GT-BEV oracle caveat.

Reproduction protocol: `docs/REPRODUCIBILITY.md`, section "LMSCNet cross-base
reproduction".

## DiffSSC reimplementation (internal visualisation only)

DiffSSC's open-source release contains only the geometric-completion pipeline
(3-channel xyz diffusion on top of LiDiff); the semantic side is missing.

The shipped paper contains no DiffSSC panel: its qualitative comparison
(Fig. 6) shows JS3C-Net, SCPNet, TALoS, S²D² and ground truth, and no
supplementary figure adds a DiffSSC column, so DiffSSC is tabulated in the paper only
as a row of main Tab. I. Our internal (3+C)-channel reimplementation
(anisotropic additive forward process, custom DDIM sampling, logit-domain
semantic encoding) lives in our internal development codebase and is
intentionally out of scope for this release: it is not part of any reported
leaderboard number and depends on private utility code we cannot ship. We do **not** report DiffSSC
numbers in the leaderboard table because the original authors did not submit;
the reimplementation served visualization only.

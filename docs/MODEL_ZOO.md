# Model Zoo

All checkpoints are released under the MIT licence and mirrored on the Hugging Face
Hub at [`BillyChern/GSSC-S2D2-checkpoints`](https://huggingface.co/BillyChern/GSSC-S2D2-checkpoints).
Fetch them with `python scripts/download_assets.py --checkpoints` (~4.9 GB total, 4.58 GiB);
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

**Download/disk sizes** (measured on the staged payload). A single
`model_ema.safetensors` (the deployment file the quickstart in `README.md` points
at) is **138.9 MB / 132.5 MiB**. `download_assets.py` provisions the **whole
per-checkpoint subdir** — `model.safetensors` + `model_ema.safetensors` +
`config.json` — which is **277.8 MB / 264.9 MiB**. The "Size" column below and
the "~265 MiB" figures in this doc refer to the full subdir; the "~140 MB" figure
in `README.md`'s quickstart refers to the single `model_ema.safetensors` it
downloads for inference. Those two are the *scene-completion* subdirs; the BEV
subdir `bev/bev_s2d2_scpnet/` is larger (569.9 MB, because it ships both a
`.pt` and a `.safetensors`) and each pyramid subdir is 229.2 MB (one
`model.safetensors`, no EMA).

The legacy SCPNet base ships as a flat `scpnet_v2_port.pth` (third-party
convention, kept as-is).

## Checkpoint SHA256 digests

GSSC-S2D2 loads pickle checkpoints (`.pt` / `.pth`) with
`torch.load(..., weights_only=False)`, so a tampered pickle is arbitrary code
execution. The call sites are `src/gssc/inference/generate_predictions.py:174`
(reached by `scripts/eval.py` and `scripts/infer.py` when you hand them a `.pt`),
`src/gssc/inference/evaluate_bev.py:182` (the `eval/bev_secondary` path) and
`src/gssc/inference/run_scpnet.py:253` (regenerating the SCPNet base
predictions). Of the 51 released files, 30 are `.safetensors` — a format that
cannot carry an executable payload — and exactly **two are pickles**
(`grep -cE '\.(pt|pth)$' data/checkpoints/checksums.txt` → 2):

* `scpnet_v2_port.pth`, the third-party SCPNet base. Its loader is
  `gssc.inference.run_scpnet`, driven by `scripts/eval_semanticposs.py` — **not**
  by `scripts/eval.py`, which consumes the pre-dumped `scpnet_predictions/`
  directory and never opens this file.
* `bev/bev_s2d2_scpnet/model.pt`, the pre-conversion copy of the BEV
  checkpoint's weights. The `model.safetensors` beside it is what the documented
  BEV eval command loads, so nothing routine reads the pickle.

`SECURITY.md` sends you here; this is the table it means.
**Verify before loading.**

The same digests ship as `checksums.txt` at the root of the Hugging Face
checkpoints repo, so the whole set can be checked in one command that returns a
verdict rather than a digest you have to compare by eye:

```bash
# Downloads into data/checkpoints/, checksums.txt included.
python scripts/download_assets.py --checkpoints

# Paths inside checksums.txt are relative to the download root itself
# (`MANIFEST.txt`, `bev/...`, `gssc_mf/...` -- no `checkpoints/` prefix), so run
# the check from INSIDE data/checkpoints/.
cd data/checkpoints && sha256sum -c checksums.txt
```

Every line must print `OK` and `sha256sum` must exit 0. A `FAILED` line, or a
`FAILED open or read` line, means do not load that file. The table below is the
per-file form of the same data, for verifying a single checkpoint you fetched by
hand:

```bash
sha256sum data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors
```

Paths are relative to the download root (`data/checkpoints/`). This covers every
file the checkpoint payload ships; digests were generated from the assets
checksums file, not transcribed.

| File (relative to the download root) | SHA256 |
|---|---|
| `MANIFEST.txt` | `87a91895447e0c22eb382b6e61f48daea74a8beb76e51f0eb37e0500088963ab` |
| `bev/bev_direct_l3_deeper/config.json` | `83d34868e0169dece91feeff2c01d532ef9a00c2ce6e744154fb1de540981fc6` |
| `bev/bev_direct_l3_deeper/model.safetensors` | `6a6f1aa1a72da78e624b9575510d6aed8f44753be461db2281f2e2666b5a3d75` |
| `bev/bev_perception_net/config.json` | `29e002364c90be8c13cd0033acaa08e0f3b4f1173e1d0b52150b7707f04860b1` |
| `bev/bev_perception_net/model.safetensors` | `6be1db3c557e5504793c998da86b10c0be2f0878b2a81632f82b774d6936d03d` |
| `bev/bev_s2d2_scpnet/config.json` | `babf70ebba9a1d1997a7e0ee4a40b1923aa3d55fc92a72e87d73960533ff5ed5` |
| `bev/bev_s2d2_scpnet/model.pt` | `46fcdb6fbdd9d4087c987b80ec514179625200417e67c71687a93ba036235111` |
| `bev/bev_s2d2_scpnet/model.safetensors` | `bd4002ccc22833481662212530d2a1d072afdfc039386d129cbfc2522c1ad881` |
| `gssc_js3c/gssc_js3c_s2d2_real/config.json` | `1fa714e5eb0e2a0abd4fe9b675bcd395061b414d9cd4603303772a47ac73680d` |
| `gssc_js3c/gssc_js3c_s2d2_real/model.safetensors` | `877e04ffdc0578078648ee5e5f219d2bc1c4e290ad3c45ada4d6d7d6a3aceed2` |
| `gssc_js3c/gssc_js3c_s2d2_real/model_ema.safetensors` | `c8743c7f5aeb37e243778374ff8d49a049c9920a66db4aec86f50c15104539dd` |
| `gssc_lmsc/gssc_lmsc_s2d2_real/config.json` | `7a9dee38e52df18ab49799e28fbc3ab14c7c37ae7e5e7558b3654977c3616330` |
| `gssc_lmsc/gssc_lmsc_s2d2_real/model.safetensors` | `7228e3339b0e4b31a1f83cfef66d22871e59cd4c97fe8527487863e7931a4a73` |
| `gssc_lmsc/gssc_lmsc_s2d2_real/model_ema.safetensors` | `ed6a0aa825ef83e6b20ab0e1016757e28a0bd1781b88b3b97b2c59c8704c0cfa` |
| `gssc_mf/gssc_31k_mf_step40000/config.json` | `f4e7cb22e577964ed3790d4388d8689f7d630ef9280f237d7e86ebe9fec082dd` |
| `gssc_mf/gssc_31k_mf_step40000/model.safetensors` | `487bb5af5bfc0d50d008d0ad8c462f73b4428305ade33af100feaaafb2704c40` |
| `gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors` | `3f80852d7c2db72571df66dc79c6add7dac50292815af883a20b460535a5193c` |
| `gssc_mf/gssc_57k_mf_step40000/config.json` | `c982809c14efac14ac2b6797eb0ceba998bc6d0fa3f64efee22a6d50a1a25b03` |
| `gssc_mf/gssc_57k_mf_step40000/model.safetensors` | `84ca0e67fd95d753855243a04e54612d706edf49dc4f2cc2d62f24d16ce3b324` |
| `gssc_mf/gssc_57k_mf_step40000/model_ema.safetensors` | `7bd49c0b15667e7019a810b16dd80ae59760881ee077dbee5413e2bbb7f66382` |
| `gssc_sf/gssc_0K_sf_step93000/config.json` | `2494dd03c7e0b1a36d41086e3515ab6f08b2a53ce1ed07251637ac84ce7c63e7` |
| `gssc_sf/gssc_0K_sf_step93000/model.safetensors` | `a3468c5565c3822381b349531d43c2ea5659cc7ac65bc3378bf77696a8762917` |
| `gssc_sf/gssc_0K_sf_step93000/model_ema.safetensors` | `5c10ffeeebe4164e1ff5f1a771f2df998f165b2351421a90f5281a831b50c20a` |
| `gssc_sf/gssc_10K_sf_step87000/config.json` | `a614d11a74a5006542651239e223cf28171553f12e8dafc2e63100db4d189b56` |
| `gssc_sf/gssc_10K_sf_step87000/model.safetensors` | `e6d4a540a0fc248ea6fc0fa9f2aca17eb5c7936f8e2a072ef039da6099c12264` |
| `gssc_sf/gssc_10K_sf_step87000/model_ema.safetensors` | `792e146146885f4292e47cab63d0463c5f09aea1971508371f01cd01db895f94` |
| `gssc_sf/gssc_20K_sf_step85000/config.json` | `b05a3c2aee9aec9e85b33f455a99048d76ead565a8cf463eeb7877a22d7f8407` |
| `gssc_sf/gssc_20K_sf_step85000/model.safetensors` | `150cfb7e233e93692fc8524b6eaf3247885e7a1dbf14e49fb87c1ee5fe6e0019` |
| `gssc_sf/gssc_20K_sf_step85000/model_ema.safetensors` | `59c740da2106cdd5a3adb6862d2d51e946acf53268579b0bdb71b25f6a9558d9` |
| `gssc_sf/gssc_31K_sf_step72000/config.json` | `0e5b8a4b84f2981eda62b9cc8969b63f3456e24e6e1576c727251e989190b7a8` |
| `gssc_sf/gssc_31K_sf_step72000/model.safetensors` | `2629234a6af325ef89c56f0427e088d932d4c7573fa278f672e2fa13dca8daa1` |
| `gssc_sf/gssc_31K_sf_step72000/model_ema.safetensors` | `2f0459d784c4f437b9e52fb2bcfa603876af269c6651ae8762774929aeeb9b31` |
| `gssc_sf/gssc_57K_sf_step69000/config.json` | `49312471306715af626092d8e2a4b2a46b8946d579e5fac823c987a444acaf26` |
| `gssc_sf/gssc_57K_sf_step69000/model.safetensors` | `8dfb3a19c45015f85386f4fa7db6f238ee2f03184619da767362359732ab55f1` |
| `gssc_sf/gssc_57K_sf_step69000/model_ema.safetensors` | `36e3b7548ddc288230ee46c8536f652fa309fa55cdf2e386140b1cd6e0d9ab50` |
| `gssc_timesteps/gssc_T10/config.json` | `1433e1487bcf80c9b73fae47f8b7e0f6507a73a9d7237e56813a91e188c88ddb` |
| `gssc_timesteps/gssc_T10/model.safetensors` | `6e9e0f00e4f163636ebe7ad4e2d8955ed4225df42cd909251d17d530b3f597f8` |
| `gssc_timesteps/gssc_T10/model_ema.safetensors` | `5273dea905bf73fb6b46e0d0fe06bb0edc947dff7b37b357b087cdc62a280515` |
| `gssc_timesteps/gssc_T100skewed/config.json` | `7760a75e3375abfdfc296de844f0b207176281eeefa8e9fc44df5f030660023d` |
| `gssc_timesteps/gssc_T100skewed/model.safetensors` | `9fee2ebfb6cfc736f2d60bebb84f686b4d0b0adaeb4959744fabcf2b5122fe5e` |
| `gssc_timesteps/gssc_T100skewed/model_ema.safetensors` | `14f842d2015ff8e4b3941392776f156da92d84ddd0727c3391ce91b38682a604` |
| `gssc_timesteps/gssc_T50/config.json` | `5418c94f39ef3e9ab0167e457c18c123b886b29805c4d1e490acef0f3c223af4` |
| `gssc_timesteps/gssc_T50/model.safetensors` | `b187dcb74eb06fdb5d23a30d6299a9ecb193a8a4c6073da641cc919ca36a7d8c` |
| `gssc_timesteps/gssc_T50/model_ema.safetensors` | `3f9fa9b36cdba98be0d3407d25b921f7711a2981116dad0d5209088c51e7c11a` |
| `pyramid/pyramid_s1/config.json` | `912842656a3a54969dd9922c5cb502863dcf24843c894482453ccfdf8f2d34fe` |
| `pyramid/pyramid_s1/model.safetensors` | `faae021aa4be483ff2d099af698a3d548d78812b534b1c48581e958cc78f0584` |
| `pyramid/pyramid_s2/config.json` | `955198fb793bfd78691059a30cc074a71c23689e2111e570f2e0b6ff6f19bf9a` |
| `pyramid/pyramid_s2/model.safetensors` | `2b7755e500d409a8dd86de06d8eba9d62a8ef96609ecbbf10f7dd8f37ef01692` |
| `pyramid/pyramid_s3/config.json` | `f7b94967a3cf78259dd2e28c986f58b5b1c5f69cbb561c2d47e8a648e6b7e3cc` |
| `pyramid/pyramid_s3/model.safetensors` | `698afc2acb62c100878a3446e25c15e633263f0cf1c554bcf186a2974f8b2c14` |
| `scpnet_v2_port.pth` | `f2d1cb27f4285690b2f8322e6a87e6631cc1af26e25d006758f7ff65587ed106` |

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
| `gssc_mf/gssc_31k_mf_step40000/` | **Headline** (`tab:main_results`, `tab:perclass_delta`; also the SCPNet pair of `tab:portable_s2d2`) | 38.54 | 38.8 (N=1, no TTA) / 39.2 (+D4 TTA) | `configs/train/31k_mf.yaml` | 265 MiB / 278 MB |
| `gssc_mf/gssc_57k_mf_step40000/` | internal / unreported (in no paper table; `tab:data_scaling`'s 57K row is the multi-frame **38.4**, measured on the development codebase, and this released checkpoint is a different run — do not read 37.76 as that cell) | 37.76 (N=1) | — | `configs/train/57k_mf.yaml` | 265 MiB / 278 MB |

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
> supplement, so it cannot be a rounding of anything the paper prints. Besides
> 24.3, the supplement carries **26.7** for this base — the *same* derived-BEV
> setting read by our internal training-time evaluator, retained for continuity
> — and **61.8**, a *separately trained* GT-BEV variant it calls an upper bound.
> Its rows carry explicit Setting and Evaluator columns; read the protocols off
> that table rather than from here, since this file has twice mis-stated which
> protocol produced which.

> **Note on the "Architecture family" column.** This column describes the
> **frozen base model** (the predictor S²D² corrects), not the S²D² denoiser
> itself. The S²D² denoiser is a **dense `Conv3d` U-Net for every row**
> regardless of the base's architecture (see the "Architecture note" section
> below). So "Sparse 3D CNN" / "2D CNN (dense)" / "Point + voxel hybrid" refer
> to LMSCNet / JS3C-Net / SCPNet, not to the released checkpoint's denoiser.

Each cross-base checkpoint subdir is 265 MiB / 278 MB total (138.9 MB for the
single `model_ema.safetensors` alone). The SCPNet (36.17 → 38.54, +2.36) and LMSCNet
(14.8 → 16.6, +1.8; LMSCNet base re-scored from on-disk predictions, superseding
the earlier 12.10 → 16.59 / +4.49 summary) deltas in the table are measured
end-to-end under the official `semantic-kitti-api`. The released LMSCNet
`model_ema.safetensors` ships complete (278 tensors, including all 45 BatchNorm
running buffers), loads cleanly, and reproduces the 16.59 figure directly.
Note its step: `gssc_lmsc_s2d2_real` is the best-mIoU selection at
`global_step: 65000`, not a step-100000 reading, even though
`configs/train/lmscnet_real.yaml` sets `num_iterations: 100000`. The JS3C
cross-base checkpoint beside it *does* record `global_step: 100000`; if you
retrain either, compare against your run's `best_miou.pt`.

The JS3C-Net row carries three numbers; the table leads with the paper's.
**What separates 26.7 from 24.3 is the evaluator, not the BEV source** — both
are derived BEV:

- **24.32 % (+1.59 pp)** — the **paper headline** for this base, and the
  reproducible at-deploy figure: **derived BEV** under the **official
  `semantic-kitti-api`**, protocol-matched to the 22.7 % base and to the
  released checkpoint's own training distribution. This is what
  `scripts/reproduce_table.py` yields end-to-end.
- **26.72 % (+3.99 pp)** — the **same derived-BEV setting**, scored with the
  paper's **internal training-time evaluator** (`SSCMetrics`). This is the row
  the paper's supplementary validation-protocol table labels *"real-only,
  derived BEV, internal eval"* (it prints **26.7**), retained there for
  continuity only. **Not** a GT-BEV number and **not** the headline.
- **26.05 %** — a **GT-BEV diagnostic** of ours, measured under the official
  `semantic-kitti-api`. The paper prints no such row: neither "26.05" nor
  "26.1" appears in it. **Not** the headline, and **not** the protocol that
  produced 26.72.

The paper's own GT-BEV entry is a separate diagnostic again — a *separately
trained* GT-BEV variant at 61.8 % official, labelled there as an upper bound
rather than a deployment number, and not the released checkpoint.

For the exact protocol behind each row, read the paper's supplementary
validation-protocol table rather than this file: its rows carry explicit
Setting and Evaluator columns, and this file has twice described the pairing
wrongly.

SCPNet's 38.54 is an official `semantic-kitti-api` number everywhere it
appears. The JS3C-Net derived-BEV deploy number (**24.32**) is produced by
`eval/js3c_val_realistic.yaml`:
- `eval/js3c_val_paper.yaml` runs the **GT-BEV diagnostic** by loading
  preprocessed GT BEV via the config key `bev_source: gt` (set in
  `configs/eval/js3c_val_paper.yaml`; it is a YAML key, not a CLI flag).
  Pointed at the released derived-BEV checkpoint it is a train/eval mismatch and
  lands at the **26.05 %** diagnostic — a number the paper does not print.
  Despite the filename, this is not the protocol behind any *main-paper* row for
  this base; the name is a retained backwards-compatibility alias. What the
  config *does* target, and says so in its own `_paper_table`, is the
  supplement's **61.8 % GT-BEV oracle row** — and only when pointed at the
  separately trained `train/js3c_real_gtbev.yaml` checkpoint, which this release
  does not ship.
- `eval/js3c_val_realistic.yaml` uses derived BEV (topmost-non-empty class
  from JS3C-Net's 3D prediction, selected via the config key
  `bev_source: derived`) — the honest deploy-time number (**24.32 %** official,
  **26.72 %** under the internal training-time evaluator). The released
  JS3C+S²D² model was trained with derived BEV, so this protocol
  matches its training distribution.

Reproduction requires `data/js3cnet_predictions/` (189 GiB / 203 GB real + synth; download via
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
> `bev/bev_s2d2_scpnet/`, and the split the paper prints is 34.8 + 1.3 --
> both halves under the training-time 2D BEV evaluator on 100 fixed val samples
> (seed 42), not the 4,071-frame `semantic-kitti-api` protocol.

Run it with the explicit checkpoint path:

```bash
python scripts/eval.py eval/bev_secondary \
    --checkpoint data/checkpoints/bev/bev_s2d2_scpnet/model.safetensors
```

## SCPNet base (frozen)

| File | What | Notes |
|---|---|---|
| `scpnet_v2_port.pth` | SCPNet pretrained weights, ported to spconv v2.3 with kernel-shape patches. | Loads via `gssc.inference.run_scpnet`. Third-party flat `.pth` (not converted). |

> **Attribution — `scpnet_v2_port.pth` is third-party work, redistributed with
> permission.** The underlying weights are SCPNet's, from Xia et al., *SCPNet:
> Semantic Scene Completion on Point Cloud*, CVPR 2023
> ([Codes-for-SCPNet](https://github.com/SCPNet/Codes-for-SCPNet)). What we add
> is the spconv v1 → v2 port described in `docs/BASELINES.md`. The SCPNet
> authors gave us permission to redistribute the ported weights in this release;
> the file therefore ships under this repository's MIT licence, but the model
> itself is theirs — cite Xia et al. if you use it, and go to their repository
> for anything beyond this port. If you would rather not take our copy, run
> `gssc.inference.run_scpnet` against an upstream SCPNet checkout instead.

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

The released code matches the paper's Method section and Fig. 4 caption:
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
model.load_state_dict(state, strict=False)  # strict=False is a forward-compatible default; the released EMA files ship the full 278-tensor state (incl. BN buffers) and load under strict=True. For exact reproduction prefer scripts/eval.py
model.train(False)
```

Or via the eval entry point::

```bash
# SCPNet headline (38.54% val)
python scripts/eval.py eval/val_1step \
    --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors

# JS3C-Net cross-base. 24.32% (paper 24.3) is the headline for this base: derived
# BEV under the official semantic-kitti-api. 26.72% (paper 26.7) is the SAME
# derived-BEV setting under the paper's internal training-time evaluator, a
# continuity row -- the evaluator differs, not the BEV source. 26.05% is a separate
# GT-BEV diagnostic of ours that the paper does not print.
# NOTE: eval/js3c_val_1step is an alias of eval/js3c_val_paper and sets
# bev_source: gt, so the command below prints the 26.05 GT-BEV diagnostic. For the
# paper headline run eval/js3c_val_realistic instead.
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
# The driver accepts the paper's own table labels, which is the form used here.
python scripts/reproduce_table.py tab:perclass_delta       # 38.54% val (main Tab. II, per-class val)
# tab:portable_s2d2 is the cross-base table; the driver reproduces its LMSCNet
# row (16.59% val) and its JS3C-Net row (24.3% val, the paper headline for that
# base, derived BEV) in one run, so both prediction dumps must be on disk.
python scripts/reproduce_table.py tab:portable_s2d2
# BEV: 36.1, measured by the training-time 2D BEV evaluator on 100 fixed val
# samples (seed 42) — see the BEV section above before quoting it.
python scripts/reproduce_table.py tab:bev_results
```

# Changelog

All notable changes to GSSC-S2D2 are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
under the policy stated below.

## Versioning policy

- **MAJOR** version (`2.0.0`) — Breaking changes to:
  - the public Python API surfaced by `from gssc.inference import …`
    or `from gssc.diffusion import …`;
  - the CLI surface of `scripts/eval.py`, `scripts/train.py`,
    `scripts/infer.py`, `scripts/download_assets.py`;
  - the on-disk format of released checkpoints;
  - the headline number's reproduction recipe (e.g. seed change,
    different SCPNet base).
- **MINOR** version (`1.1.0`) — Backwards-compatible feature additions:
  - new configs, new ablation drivers, new evaluators (e.g. BEV);
  - new checkpoints in the model zoo;
  - new docs / new tests.
- **PATCH** version (`1.0.1`) — Backwards-compatible fixes:
  - bugfixes, doc typo fixes, dependency pin bumps that don't change
    numerical behaviour;
  - CI / lint / type changes;
  - cosmetic README and badge updates.

A version that *changes the headline 38.54 % val mIoU number* is
always a **MAJOR** bump, even if the API is identical.

## [Unreleased]

## [1.1.0] — 2026-05-14

### Added — JS3C-Net cross-base support

- **Cross-base headline** (paper Tab. III rows 90–91): stacking S²D² on
  the older point-voxel hybrid base JS3C-Net (Yan et al. 2021) lifts val
  mIoU **22.73 % → 26.72 % (+3.99 pp)** under the official
  `semantic-kitti-api` evaluator. Reproducible end-to-end:
  ```
  python scripts/dump_js3c_predictions.py --js3c-repo external/JS3C-Net …
  python scripts/eval.py eval/js3c_val_1step --checkpoint data/checkpoints/gssc_js3c/gssc_js3c_s2d2_real/model_ema.safetensors
  ```
- `src/gssc/models/js3c_base.py` — predictions reader (no model code,
  predictions are shipped as a separate dataset mirroring
  `scpnet_predictions/`).
- `src/gssc/utils/compat.py` — `resolve_base_pred_dir()` shim that
  resolves the new `base_pred_dir` plus the deprecated `scpnet_pred_dir`
  alias with a `DeprecationWarning`.
- `configs/train/js3c_real.yaml` — cross-base training config (real
  frames only, `cold_diffusion=true`, 100 K steps).
- `configs/eval/js3c_val_1step.yaml`, `configs/eval/js3c_val_d4tta.yaml`
  — cross-base eval configs.
- `scripts/dump_js3c_predictions.py` — ported from internal
  `tools/dump_alt_base/dump_js3c_xsrc.py`; `--js3c-repo PATH` is now a
  CLI argument (no hardcoded path).
- `tests/test_js3c_base.py` — four tests covering predictions reader,
  D4 TTA symmetry on JS3C inputs, cold-diffusion forward determinism,
  and the `scpnet_pred_dir` deprecation alias.
- `scripts/reproduce_table.py tab:cross_base_js3c` — single-command
  reproduction of the cross-base headline (with pre-flight check that
  prints the exact dumper command if `js3cnet_predictions/` is empty).
- New release checkpoint `gssc_js3c/gssc_js3c_s2d2_real/` (paper Tab.
  III row 91; ~265 MB safetensors subdir).
- New release checkpoint `gssc_mf/gssc_57k_mf_step40000/` (paper Tab. V
  negative-result row; 37.76 % val mIoU under N=1 eval).
- `data/js3cnet_predictions/` dataset entry, mirroring the existing
  `scpnet_predictions/` layout (`{00..21}/` + `synthetic_31k/` +
  `synthetic_filtered/`). 597-frame JS3C-synth gap documented; SCPNet
  synth predictions are the recommended pseudo-label source for
  synth-augmented experiments.

### Changed

- **Release asset format**: every release checkpoint now ships as a
  per-checkpoint subdir
  `gssc_mf/<name>/{model.safetensors,model_ema.safetensors,config.json}`,
  matching the modern Hugging Face Hub convention (Llama, Qwen, SDXL,
  Flux, …). Legacy SCPNet weights still ship as a flat
  `scpnet_v2_port.pth` because they are a third-party export. Optimizer
  / scheduler / RNG state are stripped from the release tree (a
  training-time `.pt` is preserved locally as a safety net).
- `S3DSKDDataset` (semantickitti.py) accepts `base_pred_dir` and
  `base_kind` in addition to the legacy `scpnet_pred_dir`; the
  deprecated kwarg still works for one release and emits a
  `DeprecationWarning` (slated for removal in v2.0.0).
- `src/gssc/inference/d4_tta.py`: local var/param names renamed from
  `scp_*` to `base_*` for clarity; **public symbols unchanged**
  (`apply_d4`, `invert_d4`, `derive_bev`, `run_algo2_softmax`,
  `D4_ELEMENTS`). The diffusion-side kwarg
  `sample_algo2(..., scpnet_pred=...)` keeps its historical name.
- `scripts/eval.py`, `scripts/infer.py`, `scripts/reproduce_table.py`,
  `scripts/download_assets.py` accept `--base_pred_dir` /
  `--base_kind`; the legacy `--scpnet_pred_dir` flag is kept as a
  deprecated alias.
- `scripts/reproduce_table.py` resolves checkpoint paths to the new
  per-subdir `model_ema.safetensors` layout.
- `scripts/download_assets.py` learns `--js3c-predictions` flag.

### Documentation

- `docs/REPRODUCIBILITY.md` — added JS3C-Net cross-base reproduction
  protocol (clone external/JS3C-Net, dump predictions, eval).
- `docs/MODEL_ZOO.md` — added JS3C row + reworked layout to the new
  per-subdir paths.
- `docs/DATASET.md` — added `js3cnet_predictions/` section, including
  the 597-frame synth gap and the recommendation to use SCPNet's synth
  predictions instead.
- `docs/BASELINES.md` — added JS3C-Net section (no spconv kernel-shape
  patches needed; predictions-only release).
- `docs/INFERENCE.md`, `docs/TRAIN.md` — added JS3C examples.
- `README.md` — release-news entry, updated quick-start.

### Migration notes

- **No code change required** for existing v1.0.0 users. YAML configs
  that still set `scpnet_pred_dir` continue to work and emit a single
  `DeprecationWarning` at startup. Slated for removal in **v2.0.0**.
- The headline 38.54 % val mIoU and 39.2 % hidden test mIoU are
  unchanged; the SCPNet path is exercised by `D.1`/`D.2` regression
  tests on every release.

## [Pre-1.1.0 unreleased — folded into 1.1.0]

### Added
- BEV second-task driver (`scripts/eval.py eval/bev_secondary`,
  `scripts/train.py train/bev_secondary`) reproducing 36.09 % val
  BEV mIoU, paper Sec. 4 secondary-task result.
- Pyramid Discrete Diffusion (S₁/S₂/S₃) + LiDAR ray-tracing +
  rare-class object bank now import cleanly from the public package
  layout. 19 capability smoke tests pin the public API of every
  paper-claimed module.
- `.pre-commit-config.yaml` with ruff, ruff-format, light pytest, and
  a custom guard (`scripts/check_no_ai_attribution.py`) that fails
  commits attributing work to Claude / Anthropic / ChatGPT / OpenAI.
- `.github/workflows/release.yml`: auto-publish a GitHub Release on
  every `v*.*.*` tag with a generated changelog.
- `.github/ISSUE_TEMPLATE/{bug_report,feature_request,reproducibility_question}.md`
  and `.github/PULL_REQUEST_TEMPLATE.md`.
- `SECURITY.md`: vulnerability triage, scope, and checkpoint trust
  model.
- `uv.lock` (99 packages) for byte-deterministic visitor installs.
- Validated per-class IoU table (V4 numbers) inlined in the README.
- README "Reproduction status" panel showing per-claim verification
  state (✅ verified / 🟡 in progress / ⏳ asset-blocked).
- Disk-layout warning in `docs/REPRODUCIBILITY.md` for visitors on
  Docker hosts with small overlay-fs root volumes.

### Changed
- `scripts/eval.py` now dispatches `eval/bev_*` configs to the BEV
  evaluator, leaving the 3D SSC headline path unchanged.
- `scripts/train.py` now dispatches `train/bev_*` and
  `train/pyramid_*` configs to the right trainers.
- `scripts/download_assets.py` exits cleanly with a pointer to
  `docs/DATASET.md` when asset URLs are still placeholders, instead
  of failing inside `huggingface_hub` with a confusing 404.
- README quick-start now references the released `.pt` checkpoint
  filename (was incorrectly `.safetensors`).
- Acknowledgements: dropped SegRefiner and Cold Diffusion entries
  (per author preference); added Pyramid Discrete Diffusion (Liu
  et al. 2023).
- `gssc.inference.__init__.py` now re-exports `run_evaluation` and
  `evaluate_bev` for ergonomic top-level imports.

### Fixed
- `spconv` dependency pin corrected from non-existent `cu121==2.3.6`
  to the actual `cu126==2.3.8` matching the source repo's CUDA
  version.
- `evaluate_completion.py` two-stage scoring now passes an absolute
  `--datacfg` path and runs the subprocess from the script's own
  directory, fixing the `FileNotFoundError: 'config/semantic-kitti.yaml'`
  visitors hit on first eval.
- `_parse_eval_completion_output` regex updated to match the
  semantic_kitti_api's actual stdout format (`mIoU SSC =\t38.54`
  instead of the previously-assumed `Mean IoU = 38.54%`).
- `lidar_simulation.py`: removed the broken
  `from utils.util import Bresenham3D` import (the module never
  existed in the public layout) and inlined a self-contained
  26-connectivity 3D rasteriser.
- `pyramid_pipeline.py`, `train_pyramid_s2.py`, `train_pyramid_s3.py`
  imports now resolve via the package layout instead of bare
  `from kitti_dataset import …`.
- `sparse_complete_pairs.py` imports renamed file
  `lidar_resampler_v2` (was incorrectly `lidar_simulator_v2`).
- BEV second-task vendored modules (`bev_conditioning`,
  `bev_unet_v2`, `bev_diffusion_model`, `bev_multinomial_diffusion_2d`)
  with rewritten relative imports.
- `scripts/train.py` switched from `argparse.REMAINDER` to
  `parse_known_args` so that named flags appearing after the
  positional `config` argument are honoured.
- `train_scene_completion.py` now wires `--seed` and seeds
  `random / numpy / torch / cuda / PYTHONHASHSEED` at startup.
- Configs with `:` in `_paper_table`/`_description` fields are
  now properly quoted (was a YAML scanner error).

## [1.0.0] — 2026-04-26

### Added
- Initial public release.
- Headline checkpoint `gssc_31k_mf_step40000.pt` (140 MB) reproducing
  38.54 % val mIoU on SemanticKITTI seq 08, single S²D² correction step (N=1).
- 3D SSC inference pipeline: `scripts/eval.py eval/val_1step`
  (verified end-to-end on a fresh visitor clone, matches paper Tab. I
  exactly).
- D4 TTA inference pipeline: `scripts/eval.py eval/val_d4tta`
  reproducing 38.73 % val mIoU.
- Hidden-test submission flow documented in `docs/INFERENCE.md`
  reaching 39.2 % test mIoU on the SemanticKITTI leaderboard.
- Apache 2.0 license, Python 3.10–3.12, PyTorch 2.4, spconv 2.3.8.
- ruff lint gate + 80 pytest cases (89.4 % coverage on the testable
  inference + utils subset).

[Unreleased]: https://github.com/BillyChern/GSSC-S2D2/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/BillyChern/GSSC-S2D2/releases/tag/v1.1.0
[1.0.0]: https://github.com/BillyChern/GSSC-S2D2/releases/tag/v1.0.0

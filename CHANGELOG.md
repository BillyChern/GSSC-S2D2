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
  38.54 % val mIoU on SemanticKITTI seq 08, 1-step Algo2.
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

[Unreleased]: https://github.com/BillyChern/GSSC-S2D2/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/BillyChern/GSSC-S2D2/releases/tag/v1.0.0

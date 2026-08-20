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

Four changes on `main` past v2.3.8, all in paths a visitor actually runs. Commits
`ee2fbd3`, `6324ead`, `07725af`, plus the release-hygiene pass below.

### Fixed — the BEV secondary-task evaluator could not load its own model, and said nothing
`evaluate_bev()` built the denoiser from the factory defaults (`input_resolution=64`,
`cond_channels=128`); the shipped BEV run is 256 / 64. `load_state_dict(strict=False)`
accepted that **silently**: 12 `cond_proj` tensors shape-mismatched and 48 attention tensors
stayed at initialisation, so the evaluator would have scored a half-built model and returned a
number. The reconstruction keys now come from the checkpoint's own `config.json`
(`input_resolution`, `model_size`, `conditioning_type`, `use_self_conditioning`,
`lidar_channels`), and `gssc.utils.checkpoint.assert_bound()` refuses to score when anything fails to
bind. A wrong
number that looks right is worse than a crash.

The same cycle corrected *which* checkpoint the paper's BEV row belongs to. Every doc named
`data/checkpoints/bev/bev_perception_net/model.safetensors`; that is a different model and it
crashes the evaluator. The run behind the number is
`data/checkpoints/bev/bev_s2d2_scpnet/model.safetensors`, whose `config.json` records
`measured_base_miou` 0.3475 and `measured_miou` 0.3609 — and, verbatim, the protocol they were
measured under: *"training-time 2D BEV evaluator, 100 fixed val samples (seed 42) -- NOT the
4071-frame semantic-kitti-api protocol"*. That sentence travels with the 36.1 % figure
wherever it is quoted; the figure alone is not a semantic-kitti-api result.

### Fixed — an evaluation run could fill the root filesystem instead of failing
Stage 1 of an evaluation writes one `.label` per frame into `tempfile`'s directory, so a
4071-frame val run puts ~15.4 GiB wherever `TMPDIR` points — on a container, the small overlay
backing `/`. A reader following the supplement's repro matrix did not get a failed command,
they got a wedged host. `_assert_scratch_space()` now estimates from a measured 4.06 MB/frame
and refuses up front, naming `TMPDIR` and `--keep-predictions` as the remedies.

Two defects in the guard's own first draft, both caught by replaying measured numbers rather
than reasoning about them: it checked only whether the write *fits* (and so passed the exact
configuration that motivated it — 21.4 GiB free against a 15.4 GiB requirement), and
`_count_frames()` assumed a `dataset/` level this checkout does not have, returning 0 and
multiplying out to a zero-byte requirement, i.e. a guard that always passes. It now reserves
headroom (the larger of 8 GiB and 10 % of the filesystem), tries both layouts, and warns
loudly on a zero count. `configs/eval/round2_a.yaml`, which could not execute at all — it
named a directory of flat uint16 `.label` files where `--scpnet_dir` reads `(256, 256, 32)`
uint8 `_pred.npy` in learning-map space — now documents the required conversion step.

### Added — `--max-frames` states why it does not reproduce the published BEV numbers
The published BEV figures come from `run_algo2_on_samples`, which evaluates
`RandomState(42).choice(len(val_dataset), 100, replace=False)` — a seeded sample, not the
first 100 frames `--max-frames` takes. The seed indexes a *list*, and the two lists differ in
root, glob (`*_bev.npy` vs `*.bin`) and filtering (the dataset drops frames whose
`_voxels.npy` or `_bev_top.npy` is missing). Seeding the evaluator's list would select a
different 100 frames and return a plausible number reproducing nothing. Documented instead:
what `--max-frames` does, what the published protocol is, and what reproducing it would take.

### Changed — the CI badges now mean what they say
`.github/workflows/test.yml` installed only `pytest pyyaml`, so on a clean runner the job died
**at collection** (`gssc.inference.evaluate_bev` imports `numpy` at module scope) while staying
green on a developer box that already had numpy. It now declares `numpy` and runs the whole
CPU-runnable suite — 43 cases pass, 33 skip through `pytest.importorskip` — instead of four
node ids. `CONTRIBUTING.md`'s "CI-enforced" table listed five standards where two were
enforced; the coverage row would have failed the build (the CPU-runnable suite reaches ~51 %
against `fail_under = 80`). The table now lists only what a workflow runs, with the rest moved
to a "run locally" table that says why each is not wired up. `SECURITY.md` sent readers to a
hash table that did not exist and printed a bare `sha256sum <path>`, which emits a digest and
no verdict; it now documents the `sha256sum -c` path against the `checksums.txt` that ships to
the Hugging Face checkpoints repo root.

## [2.3.8] — 2026-08-13

### Fixed — v2.3.7 bumped 2 of the 4 version declarations, so `uv lock --check` failed
v2.3.5 and v2.3.6 each bumped `pyproject.toml`, `CITATION.cff`, `src/gssc/__init__.py` and
`uv.lock` together. v2.3.7 bumped only the first two, leaving `__init__.py` and the `uv.lock`
`gssc` entry at 2.3.6 — which makes `uv lock --check` fail, i.e. a visitor's first command
errors out. This is verbatim the defect the paper harness's `check_release_snapshot` R4 was
written to catch after the same drift sat in five files at v2.1.0; its comparison silently
accepted any drift inside one minor series, so it reported the mismatch as a *note* and exited 0.
That filter is fixed in the same cycle as this release.

All four declarations now read 2.3.8 and `uv lock --check` is clean. No code, config, doc text,
or measured value changed relative to v2.3.7 beyond the version strings.

## [2.3.7] — 2026-08-13

### Fixed — every training-cost label priced a 100K-iteration launch at the 40K-step price
`configs/train/*.yaml` all set `num_iterations: 100000`, and the released headline checkpoint
is `gssc_31k_mf_step40000` — step 40000. The paper's compute table prices the S²D² headline at
~37 GPU-h and the alt-base runs at ~90 GPU-h each, consistent at one rate (~0.9 GPU-h per 1K
steps). Six sites in this repo labelled a 100K-iteration launch with the headline's ~37 GPU-h,
so a reproducer sizing any full run under-provisioned by ~2.4×:

- `docs/TRAIN.md:18` — headline block, whose own Output line lists `step_{...,100000}.pt`
- `docs/TRAIN.md:76`, `:104` — the JS3C-Net and LMSCNet alt-base runs, both "identical to the
  headline 31k_mf run" (the paper says ~90 GPU-h each)
- `docs/REPRODUCIBILITY.md:157` — "each full training run … costs roughly 37 GPU-hours"
- `docs/REPRODUCIBILITY.md:315`, `:403` — the two alt-base launch comments
- `README.md:281` heading and `:364` FAQ

Each now states both figures and which checkpoint each buys: ~37 GPU-h reaches step 40000,
which is what reproduces the paper; ~90 GPU-h finishes all 100K iterations. No config, code or
measured value changed.

### Fixed — README's tag justification was false, and false for v2.3.1 too
`README.md:384` said the paper points at v2.3.1 "because `configs/infer/test_1step.yaml` … was
only added in it". `git tag --contains` on that file's adding commit lists **v2.3.0** onward, so
the justification was untrue of v2.3.1 as well as of the current tag. v2.3.6 corrected the tag
number in the sentence's first half and left the false clause standing in its second — the same
half-landed shape v2.3.6 itself was written to repair. The clause is removed and replaced with
the standing instruction to keep the tag in step with the paper's.

## [2.3.6] — 2026-08-13

### Fixed — the v2.3.3 "26.1 is not a paper number" sweep HALF-LANDED
v2.3.3 claimed to remove every site calling 26.1 / 26.05 the paper's JS3C-Net headline. It
missed **eight lines in six files**, two of them in files v2.3.3 named as fixed. An
adversarial review of the release repo found them; the residue was:

- `docs/TRAIN.md:67` "**26.05 %** (paper rounds to 26.1; paper tab:portable_s2d2 headline)"
- `docs/DATASET.md:71` "official `semantic-kitti-api` headline 22.7 → 26.1, +3.3 pp"
- `scripts/reproduce_table.py:42` "Paper headline = 26.05"
- `src/gssc/models/js3c_base.py:14` "Reproducing paper Tab. III row 91 (26.05 % val mIoU…)"
- `docs/INFERENCE.md:54`, `:68` "the number the paper rounds to 26.1"
- `docs/REPRODUCIBILITY.md:238`, `:279`, `:320` "paper rounds to 26.1 / +3.3", "22.7 % → 26.1 %"
- `README.md:43` "**22.7 % → 26.1 % (+3.3 pp)** … (paper headline)"

The paper's JS3C-Net headline is **24.3 % (+1.6 pp)** (derived BEV, official
`semantic-kitti-api`), and **"26.1" appears zero times** in the paper or its supplement, so it
cannot be a rounding of anything the paper prints. 26.05 and 26.72 are kept everywhere as
labelled GT-BEV diagnostics; only the false attribution to the paper is gone. Measured values
in code (`expected_mIoU: 26.05`) were NOT changed — they are real measurements; only the
comments calling them the paper's headline were.

Why it recurred: the v2.3.3 sweep matched phrasings ("26.1" beside "headline") rather than the
CLAIM. This release swept every line pairing 26.1/26.05 with the word "paper" and re-checked
with a claim-shaped pattern until it returned clean.

### Fixed — the repo told reviewers the paper points at the wrong tag
`README.md:384` asserted "the TPAMI submission snapshot referenced in the paper supplementary is
the **v2.3.1** release" and `docs/DATASET.md:424` stamped "Version: v2.3.1", while the paper's
reproducibility appendix names a later tag and "2.3.1" appears nowhere in the submission. Both
now say v2.3.6, and DATASET.md carries the instruction to bump it WITH supplementary.tex's tag,
which is how it drifted four releases behind.


## [2.3.5] — 2026-08-13

### Fixed — the PS³ Jensen–Shannon filter is not shipped, and a lookalike module is not it
- `docs/REPRODUCIBILITY.md` listed the Jensen–Shannon filter as a PS³ component and explained
  how to regenerate the pool, without saying that component is absent. A reader following it
  would build an **unscreened** pool believing the pipeline was reproduced. Now stated, with a
  pointer to supplementary Appendix A, which specifies the filter completely (the [2 %, 12 %]
  occupancy band, the road-plus-one-structural-class rule, the gravity caps, `τ = 0.35`, and the
  keep-top-⌊0.5·|S|⌋ ranking) so it can be reimplemented.
- `src/gssc/data/cascade_postprocess.py` is an ORPHAN that reads as authoritative: nothing
  imports it, it is not exported, and nothing references it — but its header claims the
  S1→S2→S3 post-processing role that the paper's Algorithm 1 occupies, while implementing a
  different, superseded rule (an occupied-voxel lower bound rather than a band; no divergence
  test at all). Its docstring now says it is orphaned, that it is **not** the paper's filter,
  and what the paper's filter actually is.
- No behaviour change. Reproducing any published number needs no filter code, because the
  screened pools ship as data via `scripts/download_assets.py`.


## [2.3.4] — 2026-08-13

### Fixed — two comments misstated the paper's auxiliary-loss weight by 100x
- `multinomial.py` and the `--auxiliary_loss_weight` CLI help both read "(paper: 0.05)".
  The paper states **lambda_a = 5e-4**, and every `configs/train/*.yaml` sets `0.0005`.
  0.05 is a superseded value that degraded performance and survived only in these comments.
- Behaviour was never affected (both defaults are 0.0; configs always override), but one of
  the two is user-facing `--help`, and the paper's reproducibility appendix points reviewers
  at a specific tag — so the tag they are sent to should not contain a false claim about the
  paper. That is the whole reason for this release.


## [2.3.3] — 2026-08-13

### Fixed — the docs led with a JS3C number that appears nowhere in the paper
- Eleven sites across README, MODEL_ZOO, BASELINES, INFERENCE, REPRODUCIBILITY and
  DATASET led with **26.1 % (+3.3 pp)** for JS3C-Net + S²D² and called it "the paper
  headline", several adding that "the paper rounds this to 26.1".
- The paper's JS3C headline is **24.3 % (+1.6 pp)** (derived BEV, official
  `semantic-kitti-api`), and the string **26.1 appears zero times** in the paper or
  its supplement — so it cannot be a rounding of anything the paper prints. The
  v2.2.0 alignment pass locked the docs to a paper claim the paper later corrected.
- 24.3 now leads everywhere, protocol-matched to the 22.7 % base and to the released
  checkpoint's own training distribution (`js3c_real.yaml` sets `bev_from_base: true`).
  26.05 and 26.72 are kept as labelled diagnostics.
- For which protocol produced 26.05 versus 26.72, the docs now defer to the paper's
  supplementary table (which carries an explicit Evaluator column) instead of
  asserting a pairing this repo has twice described wrongly.

### Fixed — margins and a rate the paper disclaims (see also e5639a7)
- Six sites quoted the 8-view D₄ ensemble row (39.2, N=4) as the deployable result:
  "+1.3 over TALoS", "+2.5 pp over the frozen base", and "+2.5 ... with a single
  extra forward pass" (self-contradictory, since +2.5 needs four steps plus the
  ensemble). The predicate-satisfying margins are **+0.9** and **+2.1**.
- "real-time" and "cheapest deployable" are retracted: the paper says the marginal
  9.33 FPS "is an incremental pass, not a deployable rate" and that neither it nor
  the 3.23 FPS pipeline matches the sensor's 10 Hz cadence.


## [2.3.2] — 2026-08-13

### Fixed — three docs asserted a hidden-test measurement that does not exist
- `docs/BASELINES.md` and `docs/REPRODUCIBILITY.md` (two sites) claimed the spconv-v2
  SCPNet port "matches"/"reproduces" SCPNet's published **36.7% test mIoU exactly**,
  one of them adding per-class agreement ("byte-for-byte on completion IoU 56.1% and
  on 17/19 per-class IoUs to within 0.1%") and the inference that the port's 1.03%
  val shortfall "does not transfer to test".
- **The port was never submitted to the evaluation server**, so none of that is
  measured. The paper states this in five separate places ("we hold no test-server
  score for the bare port", "we did not re-submit the port"), and the 36.7% is
  SCPNet's own published figure throughout.
- Why it mattered: the val-gap-does-not-transfer inference is exactly the claim the
  paper refuses to make, and it sits on the provenance of the headline +2.1 pp test
  margin. A reviewer reading the repo would have concluded we measured the port on
  test and that the paper's hedging was unnecessary.
- All three sites now state the port's test score is unmeasured. `REPRODUCIBILITY.md`
  line 183 was already correct ("test: 36.7 published") and is unchanged.


## [2.3.1] — 2026-08-12

### Fixed — MODEL_ZOO contradicted itself on the data-scaling regime
- `docs/MODEL_ZOO.md` described `tab:data_scaling` as the SINGLE-frame sweep in
  two places while correctly calling it MULTI-frame in a third, and
  `docs/REPRODUCIBILITY.md` and `README.md` both say multi-frame. The two wrong
  sites are corrected. The 57K row also now states that the released
  `gssc_57k_mf` checkpoint (37.76) is a DIFFERENT run from the paper cell
  (38.4), so the two are not mistaken for each other.

## [2.3.0] — 2026-08-12

The TPAMI submission snapshot. MINOR, not MAJOR: the headline 38.54 % val mIoU
is unchanged — every number below is either a new instrument or a correction to
a *stated* number that was never the headline.

### Added — the missing headline command
- **`configs/infer/test_1step.yaml`** — the hidden-test single-sample (N=1)
  configuration. Until now the release shipped only `test_d4tta.yaml`, so the
  38.8 % headline row had no runnable command while the 39.2 % 8-fold-D4 row
  did. This is why the submission snapshot is the v2.3.x line and not an earlier tag.
- **`scripts/perframe_vru.py`** + **`tests/test_perframe_vru.py`** — per-frame
  VRU instrument, gated on the published cells, and it now warns when
  `--skip_existing` would silently reuse a dump produced by a different base.
- **`src/gssc/utils/dw_iou.py`** + **`tests/test_dw_iou.py`** — DW-IoU,
  validated against all 20 published cells.
- **`--tau`** on the inference driver, so the paper's temperature-invariance
  claim can be checked rather than taken on trust
  (**`tests/test_tau_invariance.py`**).
- **`configs/eval/round2_a.yaml`** — the round-2 iteration the paper reports.
- **`configs/train/{57k_mf,T10,T50,c1_lossmatched_t99}.yaml`** — configs for
  ablation rows that previously had none.
- `scripts/reproduce_table.py` now accepts the paper's own table labels.

### Fixed — recipes and instruments that could not run as advertised
- `S3DSKDDataset` was never bound, so the advertised DSKD training recipes
  could not start.
- `RareClassEnhancer` read paste-budget fields that had been dropped.
- `scripts/fps_measure_dense_vs_sparse.py` could not import `gssc`.
- `configs/eval/val_d4tta.yaml` had `correction_steps: 1`; 4 is what
  reproduces the 38.73 % val +D4 number. The test suite asserted N=1 for the
  same config and was red before this was corrected.

### Changed — claims corrected against the artifacts
- Two ablations whose released configs cannot reproduce their paper rows are
  now flagged as such instead of implying they can.
- `reproduce_table.py` no longer advertises a table the release cannot
  regenerate.
- Corrected the retrain per-class deltas and the +2.37 headline delta.
- PS³ component order aligned to **paste → resample** (object-bank paste
  before HDL-64E ray-trace resampling) across README, DATASET and
  REPRODUCIBILITY, matching the code and the paper's §III prose.
- Fixed two Tab. VII mislabels and rewrote the DW-IoU scope note.
- The quickstart notebook now reproduces the paper's two rare-class chip IoUs.

## [2.2.0] — 2026-06-10

### Docs — JS3C-Net cross-base number reconciliation
- Aligned every JS3C-Net cross-base figure across README, docs, MODEL_ZOO,
  REPRODUCIBILITY, BASELINES, TRAIN, INFERENCE, and the release-asset MANIFEST
  to the canonical three-number scheme:
  - **26.05 % (+3.32 pp)** — paper headline, GT BEV + official `semantic-kitti-api`.
  - **26.72 % (+3.99 pp)** — same GT-BEV protocol under the paper's internal
    SSCMetrics; demoted to a footnote / ship-both number (it was previously
    mislabelled as the official-evaluator headline in several places).
  - **24.32 % (+1.59 pp)** — reproducible at-deploy number, derived BEV +
    official `semantic-kitti-api` (what `scripts/reproduce_table.py` yields).

### Added — zero-shot cross-dataset evaluation (KITTI-360, SemanticPOSS)
- **`scripts/eval_kitti360.py`**, **`scripts/score_kitti360.py`**, **`scripts/eval_semanticposs.py`**, **`configs/eval/kitti360_zeroshot_1step.yaml`**, **`configs/eval/semanticposs_seq02.yaml`**, and **`src/gssc/data/{kitti360.py, kitti360_class_map.py, semanticposs.py}`** evaluate the frozen SemanticKITTI headline checkpoint (`gssc_31k_mf_step40000`) on two unseen domains, with no fine-tuning and no target labels.
- Results: **SSCBench-KITTI360** (val seq. 06) 5.8 → 6.2 mIoU (+0.4) / 18.1 → 19.5 CompIoU (+1.4); **SemanticPOSS** (val seq. 02, TALoS Tab. 4 map) 1.0 → 6.6 mIoU (+5.5) / 31.8 → 54.9 CompIoU (+23.1). Provisioning and on-disk layout are in `docs/DATASET.md`; runnable commands in `README.md`.

### Fixed — LMSCNet `model_ema.safetensors` BatchNorm buffers
- Re-exported `gssc_lmsc/gssc_lmsc_s2d2_real/model_ema.safetensors` so it ships the full **278 tensors**, including all **45 BatchNorm** running buffers. It now loads cleanly and reproduces the paper's **16.59 %** val mIoU (+1.8 over the 14.76 % LMSCNet base) directly — no full-state-checkpoint workaround needed. The SCPNet and JS3C-Net EMA files were always complete. Details in `docs/MODEL_ZOO.md`.

## [2.1.0] — 2026-05-26

Its Hydra configs hold the hyperparameters quoted in the paper. This entry
previously called v2.1.0 "the TPAMI submission snapshot" and said it carried a
tag `submission-ready-tpami-2026`; both were wrong. No such tag was ever
created, and v2.1.0 predates `configs/infer/test_1step.yaml`, the command
behind the headline single-sample hidden-test number. The submission snapshot
is **v2.3.1**.

### Added — LMSCNet third-base support
- **`scripts/dump_lmscnet_predictions.py`**, **`src/gssc/models/lmscnet_base.py`** (`.npy` reader), **`configs/train/lmscnet_real.yaml`**, **`configs/eval/lmscnet_val_1step.yaml`** — together they let any visitor reproduce the paper's third cross-base result, **LMSCNet → +S²D² = 16.6 % val mIoU (+1.8 pp over the 14.8 % LMSCNet base)**, under the official `semantic-kitti-api` evaluator (the LMSCNet base is re-scored from on-disk predictions, superseding the earlier 12.10 % summary).
- **`base_kind` Literal** in `src/gssc/data/semantickitti.py` now accepts `'lmscnet'` alongside `'scpnet'` and `'js3c'`.
- **`tests/test_lmscnet_base.py`** — 4 unit tests (shape/dtype loading, error paths for shape mismatch / out-of-range / missing-file, uint8 → int64 upcast, base_kind Literal regression guard).
- **`scripts/reproduce_table.py tab:cross_base_lmsc`** — one-command repro for the LMSCNet+S²D² row; generalises `_check_js3c_predictions` to `_check_base_predictions(dir, base_kind)` driven by a new `BASE_DUMPER_INFO` table so adding future cross-bases needs only a config + a dict row.
- **`docs/MODEL_ZOO.md`** reframes the *Cross-base headline* section as a 3-row table (LMSCNet | JS3C-Net | SCPNet) instead of just listing JS3C.

### Removed
- **Drop unreferenced `src/gssc/models/extras_*.py` (22 files)** — these were development-time exploration modules (alternative diffusion variants, MIMO experiments, DSKD probes, etc.) with zero callers in the public path. Several had broken imports (e.g. references to a private `diffssc_utils` module that was never released, since the corresponding research direction is intentionally out of scope for the released codebase). The release surface now stays focused on the three pillars actually documented in the paper.

## [2.0.0] — 2026-05-18

### Removed (BREAKING)
- **Drop the deprecated legacy SCPNet-specific BEV-derivation flag (the pre-v1.1.1 name of `--bev_from_base`) and `gssc.utils.compat.resolve_bev_from_base` shim.** Callers must use `--bev_from_base` (added v1.1.1). The legacy YAML alias no longer works; use `bev_from_base:`. Deprecation was introduced in v1.1.1 with a `DeprecationWarning`-emitting shim; this removal is the v2.0.0 BREAKING follow-through. The older `--scpnet_pred_dir` / `scpnet_pred_dir:` v1.0.0 alias is unaffected (separate shim, separate removal path).
- **Drop `tests/test_config_loader.py::test_bool_flags_legacy_alias`** — covered the now-removed legacy BEV-derivation YAML alias.

### Migration guide (v1.x → v2.0.0)
- Replace every occurrence of the legacy SCPNet-specific BEV-derivation flag (CLI and YAML, the pre-v1.1.1 name) with `--bev_from_base` / `bev_from_base:`. The semantic is identical; only the name changed (see v1.1.1 entry below for the rename rationale).
- The headline numerical artefacts are unaffected: this is a CLI/API surface cleanup, not a model or recipe change. `38.54 % val mIoU` (SCPNet headline) and the JS3C-Net cross-base result (`26.05 %` paper headline under the official `semantic-kitti-api` with GT BEV; `26.72 %` internal SSCMetrics footnote; `24.32 %` at-deploy derived BEV) reproduce byte-identically from the same checkpoints.

## [1.1.1] — 2026-05-18

### Changed
- **Flag rename**: the legacy SCPNet-specific BEV-derivation flag → `--bev_from_base` (training, eval, and inference CLIs; identical YAML key `bev_from_base:`). The semantic was always "derive BEV by height-pooling the base 3D prediction (whichever base is wired in via `--base_pred_dir` / `--base_kind`)"; the SCPNet-specific name predates the JS3C-Net cross-base support. The old flag still works via a `DeprecationWarning`-emitting shim (`gssc.utils.compat.resolve_bev_from_base`) and is slated for removal in v2.0.0. Mirrors the v1.1.0 `scpnet_pred_dir` → `base_pred_dir` migration.

## [1.1.0] — 2026-05-14

### Added — JS3C-Net cross-base support

- **Cross-base headline** (paper Tab. III rows 90–91): stacking S²D² on
  the older point-voxel hybrid base JS3C-Net (Yan et al. 2021) lifts val
  mIoU **22.73 % → 26.05 % (+3.32 pp)** under the paper headline protocol
  (GT BEV + official `semantic-kitti-api`). The same GT-BEV protocol scored
  with the paper's internal SSCMetrics reads **26.72 % (+3.99 pp)** (a
  footnote ship-both number, see supp tab:supp_b6_val), and the reproducible
  at-deploy number under the realistic-deployment protocol (derived BEV +
  official `semantic-kitti-api`) is **22.73 % → 24.32 % (+1.59 pp)**. All
  three paths reproduce end-to-end from the released checkpoint:
  ```
  python scripts/dump_js3c_predictions.py --js3c-repo external/JS3C-Net …
  python scripts/eval.py eval/js3c_val_paper     …  # paper protocol  → ~26.7 %
  python scripts/eval.py eval/js3c_val_realistic …  # realistic deploy → 24.32 %
  ```
- `src/gssc/models/js3c_base.py` — predictions reader (no model code,
  predictions are shipped as a separate dataset mirroring
  `scpnet_predictions/`).
- `src/gssc/utils/compat.py` — `resolve_base_pred_dir()` shim that
  resolves the new `base_pred_dir` plus the deprecated `scpnet_pred_dir`
  alias with a `DeprecationWarning`.
- `configs/train/js3c_real.yaml` — cross-base training config (real
  frames only, `cold_diffusion=true`, 100 K steps).
- `configs/eval/js3c_val_paper.yaml`, `configs/eval/js3c_val_realistic.yaml`,
  `configs/eval/js3c_val_1step.yaml` (alias for paper protocol),
  `configs/eval/js3c_val_d4tta.yaml` — cross-base eval configs spanning
  both the paper-protocol (GT BEV) and realistic-deployment (derived BEV)
  paths.
- `configs/train/js3c_real_gtbev.yaml`, `configs/train/js3c_real_derived.yaml`
  — sibling training configs at `batch_size: 2` so both BEV-source
  protocols can be retrained simultaneously on a shared GPU0 for
  end-to-end codebase validation.
- `scripts/dump_js3c_predictions.py` — `--js3c-repo PATH` is a CLI
  argument (no hardcoded path).
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
  `model_ema.safetensors` is a *deployment-ready* state_dict: EMA-tracked
  parameters overlaid onto the trained BatchNorm running statistics
  (running_mean / running_var / num_batches_tracked), so
  `load_state_dict(strict=True)` succeeds on its own. This reproduces the
  v1.0.0 inference loader (which first loaded `model_state_dict` for
  buffers, then overlaid `ema_shadow` via `named_parameters()`) and
  matches the paper-reported 38.54 % val mIoU exactly.
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

## Pre-1.1.0 unreleased (folded into 1.1.0)

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
- README reproduction-status disclosure: inline "verified end-to-end"
  labels on the reported numbers plus an "Assets are not yet public"
  NOTE clarifying that the labels record local maintainer measurements,
  not what a fresh asset-blocked clone can run today.
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

[Unreleased]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.3.8...HEAD
[2.3.8]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.3.7...v2.3.8
[2.3.7]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.3.6...v2.3.7
[2.3.6]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.3.5...v2.3.6
[2.3.5]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.3.4...v2.3.5
[2.3.4]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.3.3...v2.3.4
[2.3.3]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.3.2...v2.3.3
[2.3.2]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.3.1...v2.3.2
[2.3.1]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.3.0...v2.3.1
[2.3.0]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/BillyChern/GSSC-S2D2/compare/v1.1.1...v2.0.0
[1.1.1]: https://github.com/BillyChern/GSSC-S2D2/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/BillyChern/GSSC-S2D2/releases/tag/v1.1.0
[1.0.0]: https://github.com/BillyChern/GSSC-S2D2/releases/tag/v1.0.0

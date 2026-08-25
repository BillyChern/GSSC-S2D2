# Changelog

All notable changes to GSSC-S2D2 are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
under the policy stated below.

Two early entries — **1.0.0** and **1.1.0** — are headed *untagged historical release*.
They document what shipped on those dates, but this repository carries no matching git
tag, so `git checkout v1.0.0` / `git checkout v1.1.0` will fail. Every other entry has a
tag of the same name (`git tag` lists them).

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

### Changed — every released artefact is reachable from the front page

- **`README.md`'s nav row now carries every public resource.** It previously named the
  project page as *"public on acceptance"* — false: `https://shichen.world/GSSC-project-page/`
  answers 200 to an unauthenticated request — and linked none of the released artefacts, so
  the first link to the checkpoints sat at README.md:347, 71 % of the way down the file. The
  row now leads with the project page, the checkpoints, the PS³ pool (free mirror *and* the
  citable DataPort DOI, kept distinct) and the baseline predictions. **Paper** stays an
  unlinked `*(under review)*` label: no preprint is posted, so there is nothing to link, and
  the previous *"link added on acceptance"* named the wrong trigger — a preprint link would
  arrive before a DOI does.
- **The PS³ Hugging Face mirror is described in the present tense.** `README.md`,
  `docs/DATASET.md`, `examples/quickstart.ipynb` and `scripts/download_assets.py` all said it
  "goes public with the other release repos" or was "PRIVATE today"; the downloader's own
  docstring told a reader an anonymous fetch would fail. Measured logged out on 2026-08-25:
  the repo resolves 200, `private: false`, `gated: false`.
- **The pinned submission snapshot is v2.4.1, not v2.3.8.** `README.md` and
  `docs/DATASET.md`'s datasheet still named v2.3.8 while `supplementary.tex` pins
  `v2.4.1`. v2.3.8 predates the archive fix, so that pointer sent a reviewer to the
  downloader that silently returned 3,334 of 4,071 val frames.
- **`docs/DATASET.md` documents the checkpoints' manual provisioning route**, which
  `README.md` and `docs/MODEL_ZOO.md` both promised and which did not exist — the section was
  a bare one-line command while every other artefact had one.
- **Archive counts corrected.** `README.md` attributed 609,349 files to "the prediction
  corpus" and then said it ships as ten archives; the 609,349 covers the three prediction
  trees *and* the object bank, which ship as nine + one. `examples/quickstart.ipynb` said the
  SCPNet predictions ship as ten archives; they ship as five. No size figure changed — all
  were re-measured against the live Hub and were already correct.
- **`pyproject.toml`'s `[project.urls]` and `CITATION.cff` carry the same set.** The URL
  table went from 3 entries to 11; `CITATION.cff` gains `repository-code` and
  `repository-artifact`, and its `url` becomes the project page. Every URL in both was
  verified 200 unauthenticated on 2026-08-25.
- `README.md`'s *What's new* now reaches v2.4.1, and v2.3.8 is dated 2026-08-23 to match its
  tag and its CHANGELOG entry (it read 2026-08-22).

## [2.4.2] — 2026-08-25

### Changed

- **Every released artefact is now linked from the first screenful of `README.md`.** The nav
  row said the project page was "public on acceptance" after it was already public, and
  carried no link to the checkpoints, the dataset, the DataPort record or the baseline
  predictions: the first such link was line 347, 71.7 % down the file, and the project page
  was never linked at all. All seven public resources are in the row now. Paper remains an
  unlinked label — it is not on arXiv yet.

### Fixed

- **`docs/MODEL_ZOO.md` published 18 stale SHA256s.** Scrubbing the private research-repo URL
  out of those `config.json` files changed their hashes and the docs were never updated, so
  `check_security_hashes` failed: the docs were stale, not the assets. Republished against the
  live Hub copies, fetched anonymously.

## [2.4.1] — 2026-08-25

### Fixed

- **`uv.lock` and `SECURITY.md` still declared the old version**, so `uv lock --check`
  failed on a clean checkout of `v2.4.0`. `uv sync` — the command the docs actually give
  first — works either way, but the release gate compares all four version declarations
  and this is the third time a run of releases has been needed to bring them back in step
  (see the v2.3.1–v2.3.7 note in README.md). Lock regenerated; the supported-versions
  table moves to `2.4.x`.

## [2.4.0] — 2026-08-25

### Fixed — the published reproduction command was silently evaluating a subset

- **`scripts/download_assets.py` now fetches and unpacks the released `.tar.zst` archives**
  instead of reading a per-frame tree on the Hub. The prediction corpus is 609,349 files /
  558 GiB / 600 GB uncompressed, and several of its directories hold more than the **10,000
  files Hugging Face serves from one directory**, so the per-frame upload it used to read
  could never be complete: `scpnet_predictions/08/` truncated at 10,000 files covering
  **3,334 of sequence 08's 4,071 val frames**. The command `docs/DATASET.md`,
  `configs/eval/val_1step.yaml`, `README.md` and `examples/quickstart.ipynb` all give for the
  headline evaluation —
  `python scripts/download_assets.py --predictions --include 'scpnet_predictions/08/*'` —
  therefore returned an incomplete val set, exited 0 and reported no error. Measured after the
  fix, against the live repo: **12,213 files, 9,072,663,168 B, 4,071 distinct frame ids
  `000000`–`004070`**, byte-identical to the reference tree.
- The abandoned per-frame upload (219,590 files across four prefixes) has been deleted from
  `Stone-Chern/GSSC-S2D2-datasets`, after verifying that every one of those paths is present
  inside the ten archives. The repo now holds exactly the ten archives plus its dataset card.
- **`--include` keeps its meaning and gets cheaper.** Patterns are still written against the
  `<prefix>/<seq>/<frame>` layout the docs describe (and the unanchored `'08/*'` short form
  still works); they now select both the archives to transfer and the members to unpack, so
  the val-08 fetch moved from 9.07 GB of per-frame transfers to **one 703 MiB / 737 MB
  archive** unpacking to the same 9.07 GB. A pattern that can match nothing exits with the
  documented `docs/DATASET.md` pointer rather than downloading nothing and reporting success.
- Every failure path still ends in that pointer rather than a traceback, including the two new
  ones (an archive that did not arrive, and a missing zstd backend).
- **New dependency: `zstandard>=0.22`**, so `uv sync` alone is enough to run the documented
  download commands. The `zstd` binary is used as a fallback when the package is absent.
- Sizes throughout `README.md`, `docs/DATASET.md`, `docs/REPRODUCIBILITY.md`,
  `configs/eval/val_1step.yaml` and `examples/quickstart.ipynb` now separate **download** from
  **on-disk**, which the archive layout makes differ by roughly 100x. `README.md`'s quickstart
  fetches val 08 with `--include` rather than the whole SCPNet group, which unpacks to
  324 GiB / 348 GB because the release materialises the three symlinked `synthetic*` farms.

## [2.3.8] — 2026-08-23

- **Free Hugging Face mirror for the PS³ synthetic pool**, and `--synthetic-pool` now downloads
  from it instead of printing retrieval steps. Written up in full under
  *[2.3.8] → Added — a free Hugging Face mirror for the synthetic pool* below.


### Added — a free Hugging Face mirror for the synthetic pool, and `--synthetic-pool` downloads again

- The two pool archives are now mirrored, **byte-identical** and under the same CC-BY-NC-SA 4.0
  LICENSE the deposit ships, on the Hugging Face dataset repo
  **[`Stone-Chern/PS3-SemanticKITTI`](https://huggingface.co/datasets/Stone-Chern/PS3-SemanticKITTI)**
  (`synthetic_pool_31K.tar.gz`, 2,311,614,021 B, 32,039 scenes; `synthetic_pool_57K.tar.gz`,
  4,161,739,608 B, 57,650 scenes), so the pool finally has a route that needs no IEEE DataPort
  subscription — the licence's redistribution grant is what permits it. `scripts/download_assets.py
  --synthetic-pool {31K,57K}` fetches the requested archive from that mirror through the same
  `_fetch` / `snapshot_download` path as every other asset group and then prints
  `doi:10.21227/nqgf-9k39` as the citation: **cite the DOI, download from either host.** The DOI
  remains the citable identifier of record and DataPort remains the alternative for anyone who
  prefers the archival deposit — it is still marked *Subscription Required*, still serves files
  only to a signed-in session, and still publishes no direct file URL, so no script fetches from
  it. This **supersedes two statements made lower down in this same entry**, which are kept as the
  record of why the mode used to exit rather than rewritten: the mode no longer "genuinely cannot
  provision the asset", and it depends on `huggingface_hub` again exactly as the other groups do
  (the early answer before the hub import is gone; the shared `ImportError` message covers it).
  `README.md`, `docs/DATASET.md`, `examples/quickstart.ipynb` and `CITATION.cff` (whose dataset
  reference now carries the mirror as `repository-artifact` beside the DOI) name the mirror
  wherever they name the DOI. No check was relaxed to accommodate any of this:
  `.release_checks/check_download_guard.py` still passes 9/9 with `--synthetic-pool` now measured
  on the same unreachable-repo path as the other five modes, and only its narrative — which
  described the mode as unfetchable — was corrected. The mirror is private until the release
  repos are flipped public together; until then an anonymous fetch fails into the documented
  `docs/DATASET.md` pointer.

### Fixed — `--synthetic-pool` printed a `wget` command that could never have worked

The synthetic pool's IEEE DataPort DOI was minted on 2026-08-23
(**[10.21227/nqgf-9k39](https://dx.doi.org/10.21227/nqgf-9k39)**, *PS3-SemanticKITTI*), and
`DATAPORT_URL` in `scripts/download_assets.py` now holds it instead of the
`[SYNTHETIC_POOL_URL]` placeholder — the last `[..._URL]` token in the repository.

Filling it in was not a string swap. The branch behind `--synthetic-pool` printed

    Direct: wget -O <root>/synthetic_pool_31K.tar.gz <DATAPORT_URL>/synthetic_pool_31K.tar.gz && tar -xzf ...

which is wrong twice over with a real DOI substituted: IEEE DataPort publishes no
`<landing-page>/<filename>` URL — the deposit page renders each archive as a login modal, not
an href — and it serves dataset files only to an authenticated session, so no anonymous fetch
of any shape succeeds. That command is **removed**, not re-pointed. It is replaced by the DOI,
the two archive names with their measured sizes (`synthetic_pool_31K.tar.gz`, 2.15 GiB /
2.31 GB, 32,039 scenes; `synthetic_pool_57K.tar.gz`, 3.88 GiB / 4.16 GB, 57,650 scenes — 31K is
a strict subset of 57K), the access caveat, and `tar -xzf` as a separate local step.

Two behaviour defects that the live DOI would otherwise have introduced are fixed with it:

- **`--synthetic-pool` would have exited 0 having provisioned nothing.** Its graceful exit came
  from `_ensure_url_configured` recognising a `[PLACEHOLDER]`; with a real URL that guard stops
  firing and the mode fell through to a `logger.info` block and `return`. The branch now
  `sys.exit`s non-zero with the `docs/DATASET.md` pointer, which is also the honest verdict —
  the mode genuinely cannot provision the asset. `.release_checks/check_download_guard.py`
  catches this (five of its nine arms go red without the fix); no check was relaxed to
  accommodate the change, and its narrative and selftest comments were updated where they
  described the placeholder as the live mechanism.
- **The mode no longer requires `huggingface_hub`.** Nothing about the pool involves Hugging
  Face, but once the URL guard stopped firing the mode reached the hub import and told users
  in a bare, pre-`uv sync` environment to `uv pip install huggingface-hub` for an asset the Hub
  does not host. It is now answered before that import.

`_ensure_url_configured` is kept as a regression guard, with a docstring that says so rather
than implying it still protects `--synthetic-pool`.

### Changed — docs describe the deposit as live, and how to actually get it

`README.md`, `docs/DATASET.md` (both the *Synthetic pool* passage and its *Maintenance*
duplicate) and `examples/quickstart.ipynb` no longer say the DOI "has not been minted yet" or
that the URL is "pending". They give the DOI and the access caveat: DataPort releases files
only to a signed-in session and, as deposited, requires an IEEE DataPort subscription, so the
local PS³ rebuild is now presented as the route that needs no IEEE credentials rather than as
a stopgap. The standing rule that the pool is **not** embargoed until publication is unchanged
and still holds.

### Added — the dataset's own citation

The pool is a separately citable artefact now that it has a DOI. `README.md`'s *Citation*
section and `docs/DATASET.md` carry IEEE DataPort's rendered citation and a portable `@misc`
BibTeX entry (IEEE generates `@data{nqgf-9k39-26, ...}`; `@data` is not a standard BibTeX
type). The SemanticKITTI citation requirements still apply on top of it.


Re-cut on 2026-08-23 onto the rewritten history described below, so this entry now also
covers the work that had been sitting under [Unreleased]. The rewrite removed two
undisclosed author addresses from the commit metadata before the repository is made
public; it changed every commit id and left every tree hash identical.

### Fixed — the published Hugging Face repo ids named an account that does not exist

Every Hugging Face reference read `BillyChern/GSSC-S2D2-{checkpoints,datasets}`, which 404s on the
Hub; the account is **`Stone-Chern`**. `HF_REPO_MODELS` and `HF_REPO_DATA` in
`scripts/download_assets.py`, the asset table in `README.md`, and the download lines in
`docs/MODEL_ZOO.md` and `docs/DATASET.md` now name `Stone-Chern/GSSC-S2D2-checkpoints` and
`Stone-Chern/GSSC-S2D2-datasets`. Measured 2026-08-23: `huggingface.co/Stone-Chern` returns 200,
`huggingface.co/BillyChern` returns 404. The `github.com/BillyChern` references are untouched —
that namespace is correct and the paper cites it.

### Fixed — three documents said the multi-frame cache has no builder, and it does

`docs/DATASET.md`, `docs/TRAIN.md` and `docs/REPRODUCIBILITY.md` each carried a box asserting that
the `_mf` recipes need a cache "this release does not build" whose absence is "silent". Both halves
were false: `scripts/prepare_multi_frame_data.py` builds it (23,201 frames / 2.2 GiB / 2.4 GB for
the eleven annotated sequences; val seq 08 alone 4,071 frames / 0.42 GB), and the loader emits one
`WARNING` per dataset naming the missing-frame count and that exact command. `docs/TRAIN.md`'s box
also cited a class that is not in the file it named (`SemanticKITTIDataset`, which lives in
`kitti_dataset.py`; the class there is `S3DSKDDataset`) and four line pointers that had all rotted;
every pointer in the replacement was re-measured. `docs/DATASET.md`'s disk-space table gained the
multi-frame row it was the only locally built tree missing from.

### Fixed — `SECURITY.md` undercounted the pickle attack surface

It named one pickle in the release; there are two (`awk '{print $2}' checksums.txt | grep -cE
'\.(pt|pth)$'` -> 2). The second is `bev/bev_s2d2_scpnet/model.pt`, the pre-conversion copy of the
BEV weights. `docs/MODEL_ZOO.md` had already been corrected to "exactly **two are pickles**", so
the security document was contradicting the model zoo.

### Fixed — an attribution claim produced by a top-of-file grep

`THIRD_PARTY_NOTICES.md` and `external/README.md` both said `evaluate_completion.py` "imports
nothing from the rest of the directory". It imports `auxiliary/np_ioueval.py` — the IoU accumulator
itself — through a function-local import at its line 129. The conclusion survives and is now
stated with its evidence: that file is byte-identical to the pin (blob `d31b631e…`) and imports
only `sys` and `numpy`, so no modified file is on the scoring path. `docs/REPRODUCIBILITY.md`
gained the disclosure that the "official `semantic-kitti-api`" it names 18 times is the vendored
copy at pin `4398778`, which no `docs/` file had said.

### Fixed — the six dataset LICENSE files required one attribution of the two

The terms that travel with the data named only SemanticKITTI. All six now name both papers
semantic-kitti.org requires (Behley et al., ICCV 2019 and Geiger et al., CVPR 2012), as every
other surface already did.

### Fixed — "a port of the third-party SCPNet release" read as a modified weight file

`README.md`, `hf_cards/LICENSE`, `hf_cards/model_card.md` and `assets/README.md` now say what
`THIRD_PARTY_NOTICES.md` §8 measured: it is SCPNet's own released checkpoint carried unmodified,
and the "port" in the filename is spconv-2.3 kernel-shape patching applied at load time.

### Removed — a stray local-environment script from the vendored evaluator

`external/semantic_kitti_api/fix_pip_paths.sh` hardcoded one machine's conda prefix and ran a
destructive `sed` over the pip shebang of every environment under it. It was unreferenced by any
code and unrelated to the evaluator. The counts that cite it moved with it: 44 tracked files -> 43,
and "four files are added by this project" -> three. The vendored-upstream count is unchanged at
40, because the file removed was one of the four added.

### Fixed — claims printed by the gates themselves

`check_paper_numbers` printed "20 findings" for its `CHANGELOG.md` exclusion; the gate's own
instrument returns **17** (9 paper-values-in-pdf, 4 signed-deltas-in-pdf, 1 delta-arithmetic,
3 scope-localised), and its illustrative quote named a ratio that appears in no file.
`check_configs_constructible` told the next maintainer to read its own correct green as breakage.
`check_history_clean` quoted a command/number pair that has never reproduced — the three spellings
of that object walk answer 532, 366 and 365, none of them the 358 it claimed. `run_all.sh`'s new
interpreter probe introduced a `/path/to/python` placeholder that `check_history_clean` correctly
rejects as an absolute maintainer path. `check_asset_coverage`'s path regex had no start anchor, so
research-tree provenance paths matched as release payload paths; the tightened form yields an
identical result set on the live corpus (42 matches, 7 distinct, none lost or gained).
`check_paper_labels` now also reads `src/gssc/**.py`, which was unwatched — `tests/` is deliberately
excluded, because `test_config_loader.py` writes a synthetic table label into a temp YAML,
which the gate would read as an unresolvable paper pointer.

### Fixed — two test docstrings cited LaTeX line numbers as table rows

`tests/test_js3c_base.py` and `tests/test_lmscnet_base.py` cited "Tab. III rows 90-91" and "row 90".
Those were line numbers in `4_experiments.tex`; the table has six data rows. Both now name
`tab:portable_s2d2` and the base/+S2D2 pair they exercise.

### Fixed — arithmetic that does not close on the page

`24.32 − 22.7 = 1.62`, not the `+1.59` printed beside it; the delta is against the 22.73 % base
`docs/BASELINES.md` records. Corrected in `docs/MODEL_ZOO.md`, `docs/BASELINES.md`,
`src/gssc/models/js3c_base.py` and `scripts/reproduce_table.py`. Separately, `configs/train/31k_mf.yaml`
and `scripts/train.py` called the run "the headline" while naming only 39.2, the excluded D4-TTA row;
both now lead with the paper's 38.8 at N=1.

### Fixed — a warning that reassured the user of something untrue

`scripts/train.py` told anyone passing `--seed` to a pyramid recipe that "the pyramid stages seed
inside their own trainers". They do not: `grep -in seed` over the three pyramid trainers returns
zero hits. It now says the runs are unseeded. `train_scene_completion.py`'s
`MissingVoxelCacheError` also misdescribed the enumeration code it exists to diagnose — `student`
mode drops on a missing predicted BEV, never on a missing multi-frame file, and `teacher` mode
*does* drop when no base-prediction dir is set.

### Added — `.release_checks/run_all.sh` says when the interpreter is the problem

The runner now probes for `fitz`, `huggingface_hub`, `safetensors`, `torch` and `yaml` before any
gate runs and names the ones it cannot import. Measured 2026-08-23: a bare system `python` turned
a 16/16 board into "7 failing" with nothing on screen connecting that to the interpreter.

### Removed — a `.gitignore` rule naming a file that exists in no reachable tree

`.migration_audit*` ignored nothing and only kept the purged name visible in the file a cloner is
most likely to open.

### Changed — the project is now MIT-licensed

GSSC-authored code, configs, documentation and the GSSC-trained model weights ship under the
**MIT** licence in place of Apache-2.0. Every licence name in `README.md`, `CITATION.cff`,
`assets/README.md` and in this file now reads MIT; the third-party terms are unaffected —
LMSCNet keeps its own Apache-2.0, SemanticKITTI and the artefacts derived from it (the synthetic
pool, the object bank, the base-model prediction dumps, and by inheritance the trained weights)
keep CC-BY-NC-SA 4.0, and third-party code keeps its original licence.

Third-party code is **not** confined to `external/`, and `README.md` no longer says it is: a
substantial copy from Pyramid Discrete Diffusion sits at `src/gssc/models/pyramid_unet.py` and the
`src/gssc/_improved_diffusion/` fork sits beside it. The full inventory — upstream project, terms,
copyright holder, files affected — is the new root `THIRD_PARTY_NOTICES.md`, which `README.md`'s
License section now links 3 times (it referenced neither that file nor any notice before) and which
`pyproject.toml` force-includes into the wheel at `gssc/THIRD_PARTY_NOTICES.md`. There is
deliberately **no root `NOTICE`**: MIT imposes no NOTICE-file obligation (that is Apache-2.0
§4(d)), and one briefly added during this sweep was measured to make
`.release_checks/check_strict_load.py` treat the whole tree as vendored and enforce nothing while
still printing OK. That gate's `_vendored()` now stops at the repository root before testing, and
`_selftest_root_notice()` pins the behaviour. The collapse it prevents is recorded in that gate's
own source comment at `check_strict_load.py:101-107`: on 2026-08-22, the day a root `NOTICE` was
briefly added, enforced files went 23 → 0 of the 99 then in the corpus while the gate still
printed "OK: 0 failing check(s)". That corpus grows whenever a release script is added, so the
pair is a dated record, not a standing count.

`README.md` now also records the SCPNet base weights (`scpnet_v2_port.pth`) as redistributed
with the SCPNet authors' permission under this repository's MIT licence, the upstream project
remaining governed by its own terms, and adds the licence bullet the ~441 GB of base-model
prediction dumps never had: SemanticKITTI-derived, so CC-BY-NC-SA 4.0, each carrying its
producing model's attribution as well. Beside the SemanticKITTI licence line it now names the
two citations semantic-kitti.org requires (Behley et al., ICCV 2019; Geiger, Lenz and Urtasun,
CVPR 2012) — neither appeared in any file this project authors before; both were already in the
vendored `external/semantic_kitti_api/README.md`.

### Changed — the three README figures are the paper's current figures, and the captions describe them

`assets/teaser.png`, `assets/architecture.png` and `assets/qualitative.png` were replaced with
the live paper figures, in that order:
Fig. 2 (`fig:ps3_pipeline`), the PS³ offline pipeline;
Fig. 5 (`fig:s2d2`), the refinement figure;
Fig. 6 (`fig:qualitative`), the qualitative comparison. The three
captions described the *previous* images and are rewritten against what is now on screen. Three
things they had been asserting are gone with them: a "Stage B" deployment panel the PS³ figure
does not contain; a four-level-denoiser panel that belongs to paper Fig. 4, not to the
refinement figure; and "a rare class the base SOTA misses entirely" on a pair of scenes where
SCPNet in fact scores 31.3 % and 33.0 % and it is JS3C-Net that collapses. The per-sample chips
are now labelled as the *N* = 4, +*D*<sub>4</sub>-TTA figures they are, with the voxel counts
behind them pointed at supplementary Tab. XIV.

`assets/README.md`, which the earlier pass left untouched, described the *previous* images too and
is rewritten from the files themselves: each row now states what is actually rendered band by
band, names the paper figure and the source PDF, and puts `teaser.png` in "Method at a glance"
rather than "top of README.md, above the badges", where it has never been. Its regeneration
recipe named a TeX file and a Ghostscript command that produce none of the three; the recipe is
now the one the shipped files were checked against —
`pdftoppm -png -r 200 material/figures/{fig4_da_pipeline,fig3_s2d2_v3,fig4_qualitative}.pdf` —
re-run today and reproducing all three **pixel-identically** (max per-channel difference 0 at
1355 × 1890, 1356 × 1377 and 1355 × 882).

### Changed — the JS3C-Net cross-base numbers are labelled by evaluator, not by BEV source

Across `README.md` and the historical entries in this file: **24.3 % (+1.6 pp)** is the paper's
headline for this base — derived BEV, official `semantic-kitti-api`, seq 08 in full — and
**26.7 %** is that *same derived-BEV setting* read by the paper's internal training-time
evaluator, a continuity row that is not protocol-matched to the SCPNet and LMSCNet rows. The
evaluator is what separates the two; the BEV source does not. GT-BEV conditioning is a separate
diagnostic (the repo measures **26.05 %** for it under the official api) and is not the protocol
behind 26.7. Every line calling 26.05 a "paper headline", and every line pairing 26.05 with
26.72 as one protocol under two evaluators, is corrected in place **across `README.md`, `docs/*.md`
and this file**, and in the two code comments that had carried it —
`src/gssc/models/js3c_base.py` and `scripts/reproduce_table.py`. The scope is stated explicitly
because a previous sweep of exactly this defect reported "clean" over a source list that silently
excluded the files where the defect survived.

Measured, not asserted: running `eval/js3c_val_realistic` (derived BEV, *N* = 1, no TTA) on the
released `gssc_js3c_s2d2_real` checkpoint over all 4,071 val frames of seq 08 returns **24.32 %**
under the official `semantic-kitti-api` — the 24.3 headline, from the derived-BEV setting, not
from GT BEV.

### Fixed — the SemanticKITTI SOTA claim on TALoS carried no restriction

`README.md`'s acknowledgement called TALoS the "previous SemanticKITTI SSC SOTA". The paper makes
no unrestricted superlative for anyone: its own claim is the best *causal, single-sweep,
single-sample* result on the leaderboard, and TALoS's line-of-sight adaptation aggregates other
moments — future ones included — which is why Tab. I excludes it from that comparison and from
its bolding. The acknowledgement now states TALoS's method, points at the headline table above it
for TALoS's test row, and claims no superlative. Both of our own test scores stay stated
plainly: 39.2 % (*N* = 4 + *D*<sub>4</sub>
TTA), our entry on the Codabench results tab, and 38.8 % (*N* = 1, no TTA), the row the paper
indexes its superlative on.

### Fixed — SemanticPOSS zero-shot read 6.6 where the paper prints 6.5

The v2.2.0 entry below put the zero-shot SemanticPOSS endpoint at 6.6 mIoU. The paper prints
**6.5** (exact measured value 0.0654969), which is also what `README.md` has said all along, and
the delta printed beside it in that entry — +5.5, left unchanged — is the one that closes only
against 6.5. This is the defect `.release_checks/check_paper_numbers.py` was written for; it
survived because that gate's source list does not include `CHANGELOG.md`.

### Fixed — the documented checksum command ran from the wrong directory

`SECURITY.md` and `.github/ISSUE_TEMPLATE/reproducibility_question.md` both stated that the paths
inside `checksums.txt` are "rooted at `checkpoints/`" and told the reader to run the check from
`data/`. They are relative to `checkpoints/` — the first entry is `MANIFEST.txt`, not
`checkpoints/MANIFEST.txt` — so from `data/` all 51 lines print `FAILED open or read` and
`sha256sum` exits 1 on a byte-perfect download, while `SECURITY.md` tells the reader never to
dismiss such a line. Both the issue template's command and `SECURITY.md`'s single-file variant now
run from `data/checkpoints/`, matching `SECURITY.md`'s primary command, which was already correct;
measured 51/51 `OK`, exit 0. The explanatory sentence above each is reworded to match.

### Fixed — README claimed CI enforces `mypy src`, which fails

`lint.yml` runs bare `mypy`, which takes its scope from the `[tool.mypy] files` list in
`pyproject.toml` (9 source files, clean). `mypy src` is a wider scope, is run by no workflow, and
does not pass. README's code-quality block printed `mypy src` under a `(CI-enforced)` label; it now
prints the command CI actually runs under that label and keeps the wide scope on a separate line
marked local and advisory. `CONTRIBUTING.md`'s CI-enforced table stated a third scope
(`mypy src/gssc/inference src/gssc/utils`) against its own rule that a row states the scope the
workflow runs; that cell now reads `mypy`.

### Fixed — the quick start never activated the virtualenv it creates

`uv venv` creates `.venv` and prints how to activate it; it does not change the caller's `PATH`.
Steps 2 and 3 of the README quick start then called a bare `python`, which on a fresh machine is
the system interpreter and has neither the project nor its dependencies. `source .venv/bin/activate`
is now part of step 1, and the same line is added to the bug-report template's reproduction block
and to the reproduction snippet `release.yml` writes into every GitHub Release.

### Fixed — release-history navigation pointed at tags that do not exist

`## [1.1.1]` linked to `compare/v1.1.0...v1.1.1`, and no `v1.1.0` tag exists in this repository or
on the remote, so that heading became a clickable 404 the moment the repo went public. It now
compares against `v1.0.0-rc1`, the nearest real predecessor tag. The unused `[1.1.0]` and `[1.0.0]`
definitions, which named release URLs for two untagged historical entries, are removed. A
`[Unreleased]` definition is restored — it had been overwritten by `[2.3.1]` and the heading had
been rendering as literal bracketed text since.

### Fixed — `[Unreleased]` documented changes that were already inside the tag

The three bullets it carried (the split SemanticKITTI badges, the one-way +2.1 pp margin, and the
both-scores framing) all shipped in `v2.3.8`, which points at the same commit as `HEAD`. They are
folded into the `[2.3.8]` entry, whose own text says "nothing sits outside the tag".

### Fixed — the published asset sizes were remembered, not measured

`README.md`'s asset table, quickstart comment, hardware row and FAQ carried a **~135 GB**
"eval-only" figure that `docs/DATASET.md` now explicitly retracts as wrong in both of its
readings, plus a **~414 GB** prediction total whose parts (178 + 190 + 46) were each stale. Every
size in those four places is re-measured with `du -Lsb` on the staged payload and quoted as
`GiB / GB`, matching `docs/DATASET.md`'s table: SCPNet 177 GiB / 190 GB, JS3C-Net 189 GiB /
203 GB, LMSCNet 45 GiB / 49 GB — 411.0 GiB / 441.3 GB together, the whole-unit rows adding to 442
only because each is rounded on its own — checkpoints 4.58 GiB / 4.9 GB across the 51 files
`checksums.txt` covers, object bank 313 MiB / 328 MB of data, synthetic pools 127 GiB / 136 GB
(31K) and 229 GiB / 246 GB (57K). The headline eval reads SCPNet val seq 08 alone —
8.5 GiB / 9.1 GB — and the quickstart now shows the `--include 'scpnet_predictions/08/*'` fetch
that takes only that, instead of the whole prefix. The eval-only stack is ~96 GB, not 135.

The synthetic-pool row advertised **five** variants (0K / 10K / 20K / 31K / 57K); two are
released. 0K means real frames only, so no 0K tarball can exist, and the 10K / 20K subsets are not
staged, so `scripts/download_assets.py`'s `--synthetic-pool` choices are cut to `["31K", "57K"]`
and every size in its `--help` is the measured `GiB / GB` pair.

### Fixed — the retraining recipe promised two GPUs to a single-GPU trainer

`README.md`'s retraining heading read "≈ 37 GPU-hours to step 40000 on 2 × H100 80 GB" above
`--gpu 0,1`. `gssc.training.train_scene_completion` implements no `DataParallel`, no
`DistributedDataParallel` and no `torch.distributed`, and `scripts/train.py` launches one plain
subprocess: `--gpu 0,1` only sets `CUDA_VISIBLE_DEVICES=0,1`, making a second card visible and
leaving it idle. The heading, the command and the hardware row now say one GPU and point at
`docs/TRAIN.md`, which carries the full note. (The *pyramid* trainers are a separate entry point
and do implement DDP.)

### Fixed — three lines of the repository-layout tree described the wrong code

`losses/` was listed as "KL posterior + Lovász + auxiliary + focal-CE"; only Lovász-Softmax lives
there, the other three terms being inside `gssc/diffusion/multinomial.py`, as that package's own
docstring says. `utils/` was listed as "config loader, seeding, registry"; it holds the config
loader, checkpoint I/O and binding checks, the v1.x deprecation shims and DW-IoU. `configs/` was
labelled "Hydra configs", which invites Hydra override syntax that cannot work — Hydra is not a
dependency of this project (`grep -ci hydra uv.lock` → 0) and the configs are plain YAML read by
`gssc.utils.config_loader`. The tree also now lists `THIRD_PARTY_NOTICES.md`.

### Fixed — `CITATION.cff` carried the one unrestricted SOTA claim left in the repo

Its `abstract:` read "State-of-the-art LiDAR semantic scene completion on SemanticKITTI" — an
unrestricted superlative in the file written for machine reuse, and one no gate could see until
this release: the only gate that opened `CITATION.cff` for content was `check_docs_freshness`,
which reads the `version:` and `date-released:` lines alone. `check_paper_numbers.py`'s
`doc_sources()` was widened in the same sweep and now judges `CITATION.cff`, `CONTRIBUTING.md`,
`SECURITY.md` and `.github/**`. The paper's superlative is restricted to *causal, single-sweep,
single-sample*, and Tab. I carries rows above ours outside that predicate. The abstract now
states both of our own scores with the predicate attached: 38.8 % at one step with no TTA, best
under that restriction to our knowledge; 39.2 % with four steps and an eight-view
*D*<sub>4</sub> ensemble, which sits outside it.


Cut on 2026-08-13 for the version-bump fix below, then re-cut onto the release-hardening work
that followed it, so the tag now spans the whole of `git log v2.3.7..v2.3.8`, which is where the
scope of this entry comes from. The tag has been re-pointed several times since the first cut;
run `git log -1 --format=%ci v2.3.8` for the cut date that is current rather than trusting a date
frozen into prose. Run
`git diff --shortstat v2.3.7 v2.3.8` for the current count rather than trusting a number
frozen into prose. Nothing sits outside the tag, so a `git checkout v2.3.8` gets every
change listed here. `CITATION.cff` keeps the 2026-08-13 release date.

### Added — `.release_checks/`, sixteen self-testing release gates
The release ships the harness that measures it: one gate per subject, each reading the
artefact instead of a belief and failing with a `file:line`. Their subjects are asset
coverage, manifest, namespace and provenance; CI honesty; CLI surface; config
constructibility; docs freshness; the download guard; history cleanliness; paper labels
and paper numbers; protocol disclosure; security hashes; strict checkpoint loading; and
tag parity. Every one carries a `--selftest` that re-injects the defect it was written
for and asserts that the *named* check fails, so a gate that has quietly stopped
measuring is caught by the gate rather than by a reader.

Ten of the sixteen read artefacts that are **not** part of the public release — the asset
bundle, the manuscript, the experiments checkout — and fail with a named line saying so rather
than passing on their absence; their roots are the environment variables `GSSC_REPO`,
`GSSC_ASSETS`, `GSSC_PAPER` and `GSSC_EXPERIMENTS`, each with a repo-relative default, documented
in [`.release_checks/README.md`](.release_checks/README.md) along with the entry point
(`.release_checks/run_all.sh`) and the measured coverage in a relocated clone.

Two of them exist because fixing the worktree is not the same as fixing the release.
`check_tag_parity.py` diffs what the paper's pinned tag actually contains against the worktree
— and parses the tag out of the paper rather than hardcoding it, so a paper that moves to
a tag nobody cut fails instead of agreeing with itself. `check_docs_freshness.py` pins
equalities between artefacts that both move (README's newest version to `pyproject`, the
CHANGELOG version set to the git tag set, `CITATION.cff` to the newest CHANGELOG entry),
because a constant would need editing at every release and would rot into a false green.

### Added — provenance and integrity travel with the shipped checkpoints
Every shipped checkpoint's `config.json` records the digest of the file it was converted
from (`source_sha256`, 18/18), and all but one also name the originating run and code
revision (`source_run`, `code_revision`, 17/18 — the exception is `gssc_lmsc`, whose
source run is genuinely unrecoverable and says so). Where a published figure came from a
non-standard protocol, that protocol is recorded verbatim (`eval_protocol`). `checksums.txt` and `MANIFEST.txt` moved inside
`checkpoints/`, the directory the download populates, so the verification file now arrives
alongside the files it covers instead of one level above them.

### Added — `scripts/label_to_base_pred.py`, without which `eval/round2_a` could not run
`scripts/infer.py` writes SemanticKITTI submission format (flat `uint16` `.label` under
`sequences/<SEQ>/predictions/`, original label space); `scripts/eval.py` sources `x_src`
from a base-prediction tree (`<dir>/<SEQ>/<frame>_pred.npy`, `(256, 256, 32)` `uint8`,
learning-map space). Pointing `base_pred_dir` straight at an inference output made every
frame miss the existence test and get dropped, so the eval reported a metric over zero
frames instead of failing. The new script emits the tree shape the real base predictions
use, and `configs/eval/round2_a.yaml` documents the three commands in order.

### Added — `--max-frames` states why it does not reproduce the published BEV numbers
The published BEV figures come from `run_algo2_on_samples`, which evaluates
`RandomState(42).choice(len(val_dataset), 100, replace=False)` — a seeded sample, not the
first 100 frames `--max-frames` takes. The seed indexes a *list*, and the two lists differ in
root, glob (`*_bev.npy` vs `*.bin`) and filtering (the dataset drops frames whose
`_voxels.npy` or `_bev_top.npy` is missing). Seeding the evaluator's list would select a
different 100 frames and return a plausible number reproducing nothing. Documented instead:
what `--max-frames` does, what the published protocol is, and what reproducing it would take.

### Changed — both hidden-test scores, stated plainly

- Presentation only; no API, config or measured-value change. Both hidden-test
  scores are now stated with equal confidence: **39.2 %** (N=4 + D4 TTA) is our
  entry on the Codabench results tab, which displays each team's best score, and
  **38.8 %** (N=1, no TTA) is the row the paper indexes its *causal,
  single-sweep, single-sample* superlative on. The previous wording framed 38.8's
  absence from the results tab as a deficiency ("not a row that can be looked up
  there", "treat the link as our leaderboard presence, not as the evidence for
  38.8", "EXCLUDED ... the only row of ours that is"); that is ordinary platform
  behaviour, not an omission, and the apologetic framing is gone. Appendix G
  records that the server returned both scores.
- The hidden-test margin is now stated ONE way throughout, matching the paper:
  **+2.1 pp** over the previous best published score under the predicate
  (SCPNet, 36.7). `README.md` had been quoting it two ways in one document —
  +2.1 pp over SCPNet in the hero paragraph, but "+0.9 over TALoS 37.9" in the
  results table, the val bullet and `docs/INFERENCE.md`. The paper states no
  TALoS margin: Tab. I marks TALoS "excluded from bolding: test-time
  adaptation", and the TALoS row now carries that label.
- The SemanticKITTI badge is split in two, so each badge's label matches what its
  destination shows: the leaderboard badge reads 39.2 and links to the Codabench
  results tab; the test badge reads 38.8 and links to `docs/REPRODUCIBILITY.md`,
  which carries that row and its runnable command. The old single badge promised
  38.8 and landed on a board displaying 39.2. Also adds the `#/results-tab`
  fragment the badge link alone was missing.

### Changed — the CI badges now mean what they say
`.github/workflows/test.yml` installed only `pytest pyyaml`, so on a clean runner the job died
**at collection** (`gssc.inference.evaluate_bev` imports `numpy` at module scope) while staying
green on a developer box that already had numpy. It now declares `numpy` and runs the
CPU-runnable suite instead of four node ids, with the counts it covers recorded in the
workflow itself. `CONTRIBUTING.md`'s "CI-enforced" table listed five standards where two were
enforced; the coverage row would have failed the build (the CPU-runnable suite reaches ~51 %
against `fail_under = 80`). The table now lists only what a workflow runs, with the rest moved
to a "run locally" table that says why each is not wired up. `SECURITY.md` sent readers to a
hash table that did not exist and printed a bare `sha256sum <path>`, which emits a digest and
no verdict; it now documents the `sha256sum -c` path against the `checksums.txt` that ships
with the checkpoint bundle.

### Changed — every paper pointer in the release resolves against the paper
Config files carry a `_paper_table` label naming the table they reproduce, and
`scripts/reproduce_table.py` takes such a label as its argument. Several named labels the
paper does not define, and one reproduced a different table from the one it was named
after, so a reader following the pointer landed on the wrong row or on nothing. Every
paper label reachable from a config, script or doc now resolves; the configs whose result
the paper reports in prose rather than in a table declare `_paper_table: none` and name
the section instead of inventing a table for it.

### Fixed — v2.3.7 bumped 2 of the 4 version declarations, so `uv lock --check` failed
v2.3.5 and v2.3.6 each bumped `pyproject.toml`, `CITATION.cff`, `src/gssc/__init__.py` and
`uv.lock` together. v2.3.7 bumped only the first two, leaving `__init__.py` and the `uv.lock`
`gssc` entry at 2.3.6 — which makes `uv lock --check` fail, i.e. a visitor's first command
errors out. This is verbatim the defect the paper harness's `check_release_snapshot` R4 was
written to catch after the same drift sat in five files at v2.1.0; its comparison silently
accepted any drift inside one minor series, so it reported the mismatch as a *note* and exited 0.
That filter is fixed in the same cycle as this release.

All four declarations now read 2.3.8 and `uv lock --check` is clean.

### Fixed — the BEV secondary-task evaluator could not load its own model, and said nothing
`evaluate_bev()` built the denoiser from the factory defaults (`input_resolution=64`,
`cond_channels=128`); the shipped BEV run is 256 / 64. `load_state_dict(strict=False)`
accepted that **silently**: 12 `cond_proj` tensors shape-mismatched and 48 attention tensors
stayed at initialisation, so the evaluator would have scored a half-built model and returned a
number. The reconstruction keys now come from the checkpoint's own `config.json`
(`input_resolution`, `model_size`, `conditioning_type`, `use_self_conditioning`,
`lidar_channels`), and `gssc.utils.checkpoint.assert_bound()` refuses to score when anything
fails to bind. A wrong number that looks right is worse than a crash.

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
loudly on a zero count.

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
cannot be a rounding of anything the paper prints. Both values are kept everywhere as labelled
diagnostics — 26.05 the GT-BEV one, 26.72 the internal-evaluator reading of the *same derived-BEV
run* the 24.3 headline comes from — and only the false attribution to the paper is gone. Measured values
in code (`expected_mIoU: 26.05`) were NOT changed — they are real measurements; only the
comments calling them the paper's headline were.

Why it recurred: the v2.3.3 sweep matched phrasings ("26.1" beside "headline") rather than the
CLAIM. This release swept every line pairing 26.1/26.05 with the word "paper" and re-checked
with a claim-shaped pattern until it returned clean **over the sources the sweep read** — README,
`docs/`, `src/`, `scripts/` and the asset manifests.

> **Scope correction, recorded later.** That sweep did not read `CHANGELOG.md`, and neither does
> `.release_checks/check_paper_numbers.py`, which names the file as a deliberate, printed
> exclusion in its `DOC_SOURCE_EXCLUSIONS` (a changelog's job is to quote the value it removed). Four claim-shaped lines therefore survived inside the dated entries below
> (2.2.0, 2.0.0 and 1.1.0) calling 26.05 the paper headline. They are corrected in place, each
> under its own note. Read "returned clean" as scoped to the sources listed above, not to this file.

### Fixed — the repo told reviewers the paper points at the wrong tag
`README.md:384` asserted that the submission snapshot referenced in the paper supplementary was
the **v2.3.1** release, and `docs/DATASET.md:424` stamped "Version: v2.3.1", while the paper's
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
  ensemble). The margin the paper states under the predicate is **+2.1 pp** over the previous best published score (SCPNet, 36.7); TALoS is excluded from that comparison as test-time adaptation.
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

The submission snapshot. MINOR, not MAJOR: the headline 38.54 % val mIoU
is unchanged — every number below is either a new instrument or a correction to
a *stated* number that was never the headline.

### Added — the missing headline command
- **`configs/infer/test_1step.yaml`** — the hidden-test single-sample (N=1)
  configuration. Until now the release shipped only `test_d4tta.yaml`, so the
  38.8 % headline row had no runnable command while the 39.2 % 8-fold-D4 row
  did. This is why the submission snapshot is the v2.3.x line and not an earlier tag.
- **`scripts/perframe_vru.py`** + **`tests/test_perframe_vru.py`** — per-frame
  VRU instrument, gated on the published cells.
- **`src/gssc/inference/generate_predictions.py`** now warns when `--skip_existing`
  would silently reuse a dump produced by a different base (the warning was
  attributed to `scripts/perframe_vru.py` in an earlier revision of this entry;
  `grep -n 'skip_existing' scripts/perframe_vru.py` returns nothing).
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
  to one labelled scheme.

  > **Corrected after this release — the scheme shipped here mislabelled two of its
  > three numbers.** It called 26.05 the "paper headline" and 26.72 "the same GT-BEV
  > protocol under the paper's internal SSCMetrics". Neither number is printed in the
  > paper, and the GT-BEV attribution for the internal-evaluator row is wrong. What
  > stands, per supplementary Tab. XV:
  >
  > - **24.3 % (+1.6 pp over the 22.7 % base)** — the paper's headline for this base
  >   (main Tab. III, `tab:portable_s2d2`): derived BEV, scored by the official
  >   `semantic-kitti-api` on all 4,071 frames of seq 08, and what
  >   `scripts/reproduce_table.py` yields.
  > - **26.7 % (+4.0 pp against the official 22.7 % base, +4.3 against the internal
  >   22.4 %)** — the *same* derived-BEV setting read by the paper's internal
  >   training-time evaluator. The evaluator is what separates it from 24.3, not the
  >   BEV source. The paper keeps it only for continuity with earlier drafts and marks
  >   it not protocol-matched to the SCPNet and LMSCNet rows.
  > - **26.05 %** — a repo-measured GT-BEV diagnostic under the official api. GT-BEV
  >   is a separate setting, not the protocol behind 26.7.

### Added — zero-shot cross-dataset evaluation (KITTI-360, SemanticPOSS)
- **`scripts/eval_kitti360.py`**, **`scripts/score_kitti360.py`**, **`scripts/eval_semanticposs.py`**, **`configs/eval/kitti360_zeroshot_1step.yaml`**, **`configs/eval/semanticposs_seq02.yaml`**, and **`src/gssc/data/{kitti360.py, kitti360_class_map.py, semanticposs.py}`** evaluate the frozen SemanticKITTI headline checkpoint (`gssc_31k_mf_step40000`) on two unseen domains, with no fine-tuning and no target labels.
- Results: **SSCBench-KITTI360** (val seq. 06) 5.8 → 6.2 mIoU (+0.4) / 18.1 → 19.5 CompIoU (+1.4); **SemanticPOSS** (val seq. 02, TALoS Tab. 4 map) 1.0 → 6.5 mIoU (+5.5) / 31.8 → 54.9 CompIoU (+23.1). Provisioning and on-disk layout are in `docs/DATASET.md`; runnable commands in `README.md`.

### Fixed — LMSCNet `model_ema.safetensors` BatchNorm buffers
- Re-exported `gssc_lmsc/gssc_lmsc_s2d2_real/model_ema.safetensors` so it ships the full **278 tensors**, including all **45 BatchNorm** running buffers. It now loads cleanly and reproduces the paper's **16.59 %** val mIoU (+1.8 over the 14.76 % LMSCNet base) directly — no full-state-checkpoint workaround needed. The SCPNet and JS3C-Net EMA files were always complete. Details in `docs/MODEL_ZOO.md`.

## [2.1.0] — 2026-05-26

Its Hydra configs hold the hyperparameters quoted in the paper. This entry
previously called v2.1.0 the submission snapshot and said it carried a tag
whose name embedded the target venue; both were wrong. No such tag was ever
created, and v2.1.0 predates `configs/infer/test_1step.yaml`, the command
behind the headline single-sample hidden-test number. The submission snapshot
is **v2.3.1**.

### Added — LMSCNet third-base support
- **`scripts/dump_lmscnet_predictions.py`**, **`src/gssc/models/lmscnet_base.py`** (`.npy` reader), **`configs/train/lmscnet_real.yaml`**, **`configs/eval/lmscnet_val_1step.yaml`** — together they let any visitor reproduce the paper's third cross-base result, **LMSCNet → +S²D² = 16.6 % val mIoU (+1.8 pp over the 14.8 % LMSCNet base)**, under the official `semantic-kitti-api` evaluator (the LMSCNet base is re-scored from on-disk predictions, superseding the earlier 12.10 % summary).
- **`base_kind` Literal** in `src/gssc/data/semantickitti.py` now accepts `'lmscnet'` alongside `'scpnet'` and `'js3c'`.
- **`tests/test_lmscnet_base.py`** — 4 unit tests (shape/dtype loading, error paths for shape mismatch / out-of-range / missing-file, uint8 → int64 upcast, base_kind Literal regression guard).
- **`scripts/reproduce_table.py`, LMSCNet cross-base entry point** — one-command repro for the LMSCNet+S²D² row of the paper's `tab:portable_s2d2` (invoke it by that paper label, which dispatches every cross-base row; the per-base driver key is a CLI name, not a paper label); generalises `_check_js3c_predictions` to `_check_base_predictions(dir, base_kind)` driven by a new `BASE_DUMPER_INFO` table so adding future cross-bases needs only a config + a dict row.
- **`docs/MODEL_ZOO.md`** reframes the *Cross-base headline* section as a 3-row table (LMSCNet | JS3C-Net | SCPNet) instead of just listing JS3C.

### Removed
- **Drop unreferenced `src/gssc/models/extras_*.py` (22 files)** — these were development-time exploration modules (alternative diffusion variants, MIMO experiments, DSKD probes, etc.) with zero callers in the public path. Several had broken imports (e.g. references to a private `diffssc_utils` module that was never released, since the corresponding research direction is intentionally out of scope for the released codebase). The release surface now stays focused on the three pillars actually documented in the paper.

## [2.0.0] — 2026-05-18

### Removed (BREAKING)
- **Drop the deprecated legacy SCPNet-specific BEV-derivation flag (the pre-v1.1.1 name of `--bev_from_base`) and `gssc.utils.compat.resolve_bev_from_base` shim.** Callers must use `--bev_from_base` (added v1.1.1). The legacy YAML alias no longer works; use `bev_from_base:`. Deprecation was introduced in v1.1.1 with a `DeprecationWarning`-emitting shim; this removal is the v2.0.0 BREAKING follow-through. The older `--scpnet_pred_dir` / `scpnet_pred_dir:` v1.0.0 alias is unaffected (separate shim, separate removal path).
- **Drop `tests/test_config_loader.py::test_bool_flags_legacy_alias`** — covered the now-removed legacy BEV-derivation YAML alias.

### Migration guide (v1.x → v2.0.0)
- Replace every occurrence of the legacy SCPNet-specific BEV-derivation flag (CLI and YAML, the pre-v1.1.1 name) with `--bev_from_base` / `bev_from_base:`. The semantic is identical; only the name changed (see v1.1.1 entry below for the rename rationale).
- The headline numerical artefacts are unaffected: this is a CLI/API surface cleanup, not a model or recipe change. `38.54 % val mIoU` (SCPNet headline) and the JS3C-Net cross-base result (`24.3 %`, the paper's headline for this base: derived BEV under the official `semantic-kitti-api`; `26.7 %` for that same derived-BEV setting under the paper's internal training-time evaluator; `26.05 %` for the separate GT-BEV diagnostic) reproduce byte-identically from the same checkpoints.

## [1.1.1] — 2026-05-18

### Changed
- **Flag rename**: the legacy SCPNet-specific BEV-derivation flag → `--bev_from_base` (training, eval, and inference CLIs; identical YAML key `bev_from_base:`). The semantic was always "derive BEV by height-pooling the base 3D prediction (whichever base is wired in via `--base_pred_dir` / `--base_kind`)"; the SCPNet-specific name predates the JS3C-Net cross-base support. The old flag still works via a `DeprecationWarning`-emitting shim (`gssc.utils.compat.resolve_bev_from_base`) and is slated for removal in v2.0.0. Mirrors the v1.1.0 `scpnet_pred_dir` → `base_pred_dir` migration.

## [1.1.0 — untagged historical release] — 2026-05-14

> **No `v1.1.0` git tag exists in this repository, so this release cannot be checked out.**
> The entry is kept as the historical record of what shipped on that date. The tags that
> bracket it are `v1.0.0-rc1` (2026-04-25) and `v1.1.1` (2026-05-18); `v1.1.1` is the first
> checkout-able point that contains this work.

### Added — JS3C-Net cross-base support

- **Cross-base headline** (paper Tab. III, `tab:portable_s2d2`): stacking S²D² on
  the older point-voxel hybrid base JS3C-Net (Yan et al. 2021) lifts val
  mIoU **22.7 % → 24.3 % (+1.6 pp)** under the paper's headline protocol for this
  base — derived BEV, official `semantic-kitti-api`, all 4,071 frames of seq 08.
  The *same* derived-BEV setting read by the paper's internal training-time
  evaluator reads **26.7 % (+4.0 pp against the official base, +4.3 against the
  internal one)**, a continuity row (supplementary Tab. XV) that differs from 24.3
  by the evaluator, not by the BEV source. GT-BEV conditioning is a separate
  diagnostic and reads **26.05 %** under the official api, a repo-measured figure
  the paper does not print. Both reproducible paths run end-to-end from the released
  checkpoint (26.7 is the internal evaluator's reading of the first, not a separate run,
  and no shipped config produces it):
  ```
  python scripts/dump_js3c_predictions.py --js3c-repo external/JS3C-Net …
  python scripts/eval.py eval/js3c_val_realistic …  # derived BEV, official api → 24.3 %
  python scripts/eval.py eval/js3c_val_paper     …  # GT-BEV diagnostic         → 26.05 %
  ```
  > **Corrected after this release.** As shipped, this entry called 26.05 the paper
  > headline and 26.72 the same GT-BEV protocol under the internal evaluator. Neither
  > value is printed in the paper; the labels above are the ones that stand.
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
- `scripts/reproduce_table.py`, JS3C-Net cross-base entry point — single-command
  reproduction of the cross-base headline (with pre-flight check that
  prints the exact dumper command if `js3cnet_predictions/` is empty).
- New release checkpoint `gssc_js3c/gssc_js3c_s2d2_real/` (paper Tab.
  III row 91; ~265 MB safetensors subdir).
- New release checkpoint `gssc_mf/gssc_57k_mf_step40000/` (negative result; in
  no paper table -- supp Tab. VII's 57K row is the multi-frame 38.4 from a
  different run; 37.76 % val mIoU under N=1 eval).
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

## [1.0.0 — untagged historical release] — 2026-04-26

> **No `v1.0.0` git tag exists in this repository, so this release cannot be checked out.**
> The entry is kept as the historical record of the first public release. The only tag from
> that period is the pre-release `v1.0.0-rc1` (2026-04-25); the first final tag in the
> repository is `v1.1.1` (2026-05-18).

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
- MIT license, Python 3.10–3.12, PyTorch 2.4, spconv 2.3.8.
- ruff lint gate + 80 pytest cases (89.4 % coverage on the testable
  inference + utils subset).

[Unreleased]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.4.2...HEAD
[2.4.2]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.4.1...v2.4.2
[2.4.1]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/BillyChern/GSSC-S2D2/compare/v2.3.8...v2.4.0
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
[1.1.1]: https://github.com/BillyChern/GSSC-S2D2/compare/v1.0.0-rc1...v1.1.1

<!-- No [1.1.0] or [1.0.0] link definition: those two entries are headed "untagged
     historical release" and no v1.1.0 / v1.0.0 tag exists, so any release or compare
     URL naming them is a 404. `v1.0.0-rc1` is the nearest real predecessor tag and is
     what the [1.1.1] compare above is based against. Every vX.Y.Z named in a link
     definition on this page resolves via `git rev-parse`. -->

# Release gates

Sixteen self-testing checks that measure the release instead of asserting it. Each one reads
an artefact and fails with a `file:line`, and each one carries a `--selftest` that proves it
still fails when the thing it measures is broken — a gate that has quietly stopped measuring
reports green forever otherwise.

## Running them

```bash
.release_checks/run_all.sh              # every gate; exit 1 if any fails, 2 if any is broken
.release_checks/run_all.sh --selftest   # every gate's selftest instead
.release_checks/run_all.sh -v           # also echo each gate's own PASS/FAIL lines
.release_checks/run_all.sh check_asset  # only gates whose name matches a substring
python3 .release_checks/check_docs_freshness.py            # one gate, directly
python3 .release_checks/check_docs_freshness.py --selftest
```

`run_all.sh` is only a convenience wrapper — it finds the gates, resolves one interpreter and
one scratch root, and separates "found a defect" (exit 1) from "the gate itself is broken"
(exit 2). Every gate runs standalone with no arguments, so nothing here depends on the wrapper
being present.

They are deliberately not pytest cases. Several read artefacts that live **outside** this
repository and several take minutes; folding them into `pytest` would make CI claim to
enforce things it does not run. One of the defects this harness exists for was
`CONTRIBUTING.md` advertising five "CI-enforced" standards where CI ran two.

## Roots

Every root is an environment variable with a repo-relative default, so a clone audits
**itself**. Absolute paths were hardcoded here once; a relocated clone then measured a tree
it was not running in and reported green on a checkout it had never read.

| Variable | What it points at | Default |
|---|---|---|
| `GSSC_PY` | interpreter to run the gates with | `<repo>/.venv/bin/python`, else `python3` |
| `GSSC_REPO` | the release checkout under test | the repository these files sit in |
| `GSSC_ASSETS` | the asset staging bundle | `<repo>/../GSSC-S2D2-assets` |
| `GSSC_PAPER` | the manuscript checkout | `<repo>/../GSSC-paper` |
| `GSSC_EXPERIMENTS` | the internal experiments checkout | `<repo>/../Semantic_Scene_Completion_LiDAR` |
| `TMPDIR` | scratch root — never `/tmp` on the maintainer's box | `~/.cache/gssc-release-checks` |

`run_all.sh` prints the interpreter and `TMPDIR` it resolved in its header. Four gates import
optional extras (`PyMuPDF`, `safetensors`, `torch`, `tqdm`); run under an interpreter that
lacks them and they report `ModuleNotFoundError`, which looks like a broken gate rather than a
wrong interpreter — set `GSSC_PY` to the environment that has them.

**The asset bundle, the manuscript and the experiments checkout are not part of the public
release.** They are maintainer working trees; a clone does not contain them, and the released
artefacts are distributed separately (`docs/DATASET.md`, `docs/MODEL_ZOO.md`). A gate that
needs one and cannot find it **fails** rather than passing — "the artefact is not here" is not
evidence that it is correct. Point the variable at your own copy, or skip that gate.

## Measured portability

Every number below was measured on **2026-08-22** in a genuinely relocated clone — a fresh
`git clone` of this repository with the working tree's uncommitted files copied over it, no
`GSSC-S2D2-assets/`, no `GSSC-paper/`, no `Semantic_Scene_Completion_LiDAR/`, and `TMPDIR`
pointed at a scratch directory. **The counts depend on the interpreter, so both are given
rather than the flattering one.** Re-measure with:

```bash
for g in .release_checks/check_*.py; do "$PY" "$g";            done   # plain
for g in .release_checks/check_*.py; do "$PY" "$g" --selftest;  done   # selftests
```

| | plain run | `--selftest` |
|---|---|---|
| an interpreter carrying the optional extras (`PyMuPDF`, `safetensors`, `torch`, `tqdm`, plus `ruff`/`pytest` on PATH) | **6 green / 10 fail**, 0 tracebacks | **8 green / 8 fail**, 5 tracebacks |
| a bare system `python3` | **3 green / 13 fail**, 3 tracebacks | **5 green / 11 fail**, 7 tracebacks |

The extras interpreter is the one the numbers below describe; a bare `python3` turns four
gates' `ModuleNotFoundError` into what looks like a broken gate, which is why `GSSC_PY` exists
and why `run_all.sh` prints the interpreter it resolved.

**Plain run, extras interpreter — the six that are green** need nothing but this checkout:
`check_cli_surface`, `check_configs_constructible`, `check_docs_freshness`,
`check_download_guard`, `check_history_clean`, and `check_ci_honesty` (which replays the
workflow commands for real, so it needs `ruff` and `pytest` visible). The other ten **fail
loudly, each naming exactly what it could not read** — none of them passes vacuously.
`check_strict_load` joins the green set once `scripts/download_assets.py --checkpoints` has
populated `data/checkpoints/`; the rest need the asset bundle, the paper, or the experiments
checkout.

**`--selftest`, extras interpreter — the eight that are green** are self-contained:
`check_ci_honesty`, `check_cli_surface`, `check_configs_constructible`, `check_docs_freshness`,
`check_download_guard`, `check_history_clean`, `check_strict_load`, `check_tag_parity`. Under a
bare `python3` only five of those survive — `check_configs_constructible` and
`check_download_guard` need `tqdm` and friends — so a "seven selftests are portable" line is
true under neither interpreter and is not stated here.

The remaining eight derive their fixture from the real artefact, deliberately, so a hand-typed
fixture cannot drift away from what ships; without it, **five of them raise a traceback rather
than printing a named line** (`check_asset_coverage`, `check_asset_manifest`,
`check_asset_provenance`, `check_paper_labels`, `check_protocol_disclosure`). That is a rough
edge, not a silent pass — the exit code is non-zero either way — but it is worth knowing before
quoting this harness's portability anywhere public. The "no tracebacks" claim above is scoped
to the **plain** run under the extras interpreter, where it holds at 0/16.

## Opt-in probes

`check_asset_namespace.py --probe-hf` (or `GSSC_PROBE_HF=1`) is the only check that touches
the network. It requires an anonymous `200` for each Hugging Face repo id the downloader
actually fetches from. Off by default: a gate that reaches the network is flaky by
construction, and the answer only matters on publication day. Note that a `401` from the HF
API is returned for private **and** for absent repos and therefore settles nothing on its own.

# Vendored third-party code

Each subdirectory here is an in-tree pin of an external project, kept in the
tree so that scoring and training stay reproducible without a network fetch.
The provenance of each vendored copy:

| Directory | Upstream | Pin | Notes |
|---|---|---|---|
| `semantic_kitti_api/` | [PRBonn/semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api) | upstream `4398778`, locally modified | Official evaluator used for all `semantic-kitti-api` mIoU/CompIoU numbers. MIT (© 2019 University of Bonn), upstream `LICENSE` reproduced at `semantic_kitti_api/LICENSE`. |
| `multinomial_diffusion/` | [ehoogeboom/multinomial_diffusion](https://github.com/ehoogeboom/multinomial_diffusion) (upstream D3PM / Multinomial Diffusion) | upstream `9d907a6`, one line changed | Reference categorical-diffusion implementation. **No upstream `LICENSE` file exists**; MIT is asserted only by a trove classifier in the upstream `setup.py`, and there is no upstream copyright notice. See its `NOTICE` and [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) §2 — this project claims no license on the upstream authors' behalf. |

Pinning the evaluator in-tree means a reported score reflects one fixed scoring
implementation rather than whatever version a user happens to clone. Each copy
is redistributed on its own upstream terms, not under this repository's MIT
license; what is reproduced under each subdirectory
(`semantic_kitti_api/LICENSE`, `multinomial_diffusion/NOTICE`) is the upstream
material we were able to verify, which for `multinomial_diffusion/` is a
classifier rather than a license text.

Both copies were re-checked against fresh upstream clones on 2026-08-22, and
**neither is an untouched snapshot** — here is exactly what differs.

`semantic_kitti_api/` holds 44 tracked files. Blob-matching them against all 50
upstream commits puts the pin at `4398778` (2022-11-24): 37 of the 40
upstream-derived files are byte-identical to it, `evaluate_completion.py` differs
by one line (`np.bool` → `bool`, an alias NumPy removed in 1.24),
`auxiliary/laserscan.py` adds a spatial crop before range projection, and
`visualize_voxels.py` adds commented-out debug lines. Three further files are ours
rather than upstream's: `config/semantic-kitti_my.yaml`, `config/kitti360.yaml`
and `config/semantic-poss.yaml`. The `LICENSE` is
byte-identical both to that commit and to today's upstream HEAD. The scoring
path is `evaluate_completion.py` plus the one vendored file it imports —
`auxiliary/np_ioueval.py`, through a function-local import at its line 129 —
and both are byte-identical to the pin, so none of the three local
modifications affects a score.

`multinomial_diffusion/` holds 46 tracked files — 45 from upstream (HEAD
`9d907a6`) plus the `NOTICE` this project added; the 9 upstream files not
vendored are its dataset loaders, and exactly one vendored file was changed
(`segmentation_diffusion/README.md`, a hard-coded scratch path replaced with
`./logs`). Everything else is byte-identical to upstream.

[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) §7 and §2 carry the
same record with the commands behind each figure.

**This directory is not the whole third-party story.** Code copied from external
projects also lives *outside* `external/` — most importantly
`src/gssc/models/pyramid_unet.py` (a substantial copy of Pyramid Discrete
Diffusion, MIT © 2023 Yuheng Liu) and the `src/gssc/_improved_diffusion/` fork.
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) is the complete record
and covers those in-src copies as well as the two directories here.

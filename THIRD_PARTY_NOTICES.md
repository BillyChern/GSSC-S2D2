# Third-Party Notices

GSSC-S2D2 itself is released under the MIT License, **Copyright (c) 2026 Shi
Chen, Weifeng Ge** (see [`LICENSE`](LICENSE)). That grant covers only the code,
configs, and documentation authored by this project.

This file records every third-party component that this repository — or the
artifact bundles released alongside it (checkpoints, prediction dumps, object
bank, synthetic pool) — redistributes or derives from: the upstream project,
the terms it carries, the copyright holder, and the exact files affected. It
must travel with any copy or substantial portion of this software; the
distributed wheel ships it at `gssc/THIRD_PARTY_NOTICES.md`.

Every upstream license statement below was checked against the upstream source
itself (a clone or the raw file on the project's own host), not against a
summary, on 2026-08-22; each section names the commit it was read at. Where an
upstream publishes **no** license, this file says so plainly and claims nothing
on that project's behalf.

This repository ships **no root `NOTICE` file**, deliberately. A NOTICE file is
an Apache-2.0 §4(d) obligation and this project is MIT, so nothing requires one:
this file is the attribution record, and `pyproject.toml` force-includes it into
the wheel. Each vendored subtree carries its own notice *beside the code it
covers* (`external/multinomial_diffusion/NOTICE`,
`src/gssc/_improved_diffusion/NOTICE`), which is also how
`.release_checks/check_strict_load.py` tells vendored code from ours. A root
NOTICE once broke that: it sat above every file and marked the whole tree
vendored, silently taking the gate from 23 enforced files to 0 while it still
printed OK. `_vendored()` now stops at the repository root before testing and
`_selftest_root_notice()` pins that — but MIT carries no NOTICE obligation, so
do not add one.

**Contents**

1. [Pyramid Discrete Diffusion — MIT, © 2023 Yuheng Liu](#1-pyramid-discrete-diffusion)
2. [multinomial_diffusion (Hoogeboom & Nielsen) — no upstream license](#2-multinomial_diffusion-hoogeboom--nielsen)
3. [denoising-diffusion-pytorch — MIT, © 2020 Phil Wang](#3-denoising-diffusion-pytorch-lucidrains)
4. [improved-diffusion — MIT, © 2021 OpenAI](#4-improved-diffusion-openai)
5. [point-e — MIT, © 2022 OpenAI](#5-point-e-openai)
6. [2DPASS — MIT, © 2022 Benny](#6-2dpass)
7. [semantic-kitti-api — MIT, © 2019 University of Bonn](#7-semantic-kitti-api)
8. [SCPNet — no upstream license; weights redistributed with permission](#8-scpnet)
9. [JS3C-Net — MIT, © 2020 Xu Yan](#9-js3c-net)
10. [LMSCNet — Apache-2.0, © 2020 Inria and AKKA Technologies](#10-lmscnet)
11. [Dataset terms and what they bind](#11-dataset-terms-and-what-they-bind)

---

## 1. Pyramid Discrete Diffusion

* **Upstream:** <https://github.com/yuhengliu02/pyramid-discrete-diffusion>
* **Paper:** Liu et al., *Pyramid Diffusion for Fine 3D Large Scene Generation*,
  ECCV 2024 (arXiv:2311.12085); cited in the manuscript.
* **License:** MIT. **Copyright (c) 2023 Yuheng Liu** — read from the upstream
  `LICENSE` (commit `964fca4`) on 2026-08-22.

Files in this repository that copy from or derive from it:

| File | Relation to upstream | Measured overlap |
|---|---|---|
| `src/gssc/models/pyramid_unet.py` | near-verbatim copy of `models/conditional_diffusion/con_denoise.py`, as its own module docstring states | 345 matched lines — **96.6 % of the upstream file, 84.8 % of ours** (87.8 % of ours before the attribution header was added to it) |
| `src/gssc/models/pyramid_diffusion.py` | derived from `models/conditional_diffusion/con_diffusion.py` | 185 matched lines (62.9 % of the upstream file, 58.9 % of ours) |
| `src/gssc/_improved_diffusion/multinomial_diffusion.py` | derived from the same `con_diffusion.py` | 252 matched lines (85.7 % of the upstream file, 59.9 % of ours) |

Overlap measured 2026-08-22 against a clone of the upstream repository at
commit `964fca4`, by this exact procedure — every overlap figure in this
document was produced with it, and the numbers reproduce only with it (leaving
`autojunk` at its `True` default gives smaller counts):

```python
import difflib
lines = lambda p: [ln.strip() for ln in open(p, encoding="utf-8").read().splitlines()]
sm = difflib.SequenceMatcher(None, lines(OURS), lines(UPSTREAM), autojunk=False)
matched = sum(b.size for b in sm.get_matching_blocks())
```

Percentages are `matched` divided by each file's total `splitlines()` count.
These counts are properties of two moving files: re-derive them with the
snippet above whenever either side changes.

Scope note, so the table is not read as claiming more than it shows: for the two
`*_diffusion.py` files, most of the shared content is the **common
multinomial-diffusion core** that upstream itself inherited — 111 of the 131
PDD-shared non-blank lines in `pyramid_diffusion.py`, and 138 of the 171 in
`_improved_diffusion/multinomial_diffusion.py`, appear verbatim in
`external/multinomial_diffusion/diffusion_utils/diffusion_multinomial.py`
(§2), and PDD's own `con_diffusion.py` header credits lucidrains (§3). The
PDD-specific residue is 20 lines in `pyramid_diffusion.py` and 33 in
`_improved_diffusion/multinomial_diffusion.py` — in both, chiefly the
auxiliary-loss weighting, plus PDD's intermediate-returning sample loop in the
second. `pyramid_unet.py` is a different matter: it is a substantial
reproduction of an upstream file that carries no third-party attribution of its
own, and it is what makes the MIT notice below mandatory rather than
courteous.

```
MIT License

Copyright (c) 2023 Yuheng Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The released pyramid checkpoints (`pyramid_s1.pt`, `pyramid_s2.pt`,
`pyramid_s3.pt` and the `pyramid/` subdirectories of the checkpoint bundle) were
produced by training this derived code; they are GSSC-trained weights, but the
architecture they instantiate is the one above.

---

## 2. multinomial_diffusion (Hoogeboom & Nielsen)

* **Upstream:** <https://github.com/ehoogeboom/multinomial_diffusion>
* **Paper:** Hoogeboom, Nielsen, Jaini, Forré & Welling, *Argmax Flows and
  Multinomial Diffusion: Learning Categorical Distributions*, NeurIPS 2021.
* **License: none is published.**

**There is no upstream license grant, and this project does not assert one on
the authors' behalf.** A clone of the upstream repository (checked 2026-08-22,
HEAD `9d907a6`) contains **no `LICENSE`, `LICENCE`, `COPYING`, `NOTICE` or
`COPYRIGHT` file of any kind**, and its `README.md` states no terms. The only
license signal anywhere in that repository is a single PyPI trove classifier
inside `setup.py`, quoted here in full:

```
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
```

That classifier is the sole basis on which this repository redistributes the
code. It is a genuine statement by the upstream author, but it is not a license
text, it names no copyright holder, and it grants no rights explicitly. The
names "Emiel Hoogeboom, Didrik Nielsen" that appear in
`external/multinomial_diffusion/NOTICE` come from `setup.py`'s `author=` field —
**there is no upstream copyright notice to retain**. An earlier revision of that
vendored NOTICE reproduced the MIT text under a synthesised copyright line and
said the original notice was retained; it was corrected on 2026-08-22 and now
states what this paragraph states.

What we redistribute or derive:

* `external/multinomial_diffusion/` — 46 tracked files: 45 from the upstream
  tree plus the `NOTICE` this project added. Upstream has 54 tracked files; the
  9 not vendored are its dataset loaders, and exactly one vendored file was
  changed (`segmentation_diffusion/README.md`, a hard-coded scratch path). No
  module outside that directory imports it; it is kept as the provenance record
  for the derivatives below.
* `src/gssc/_improved_diffusion/multinomial_diffusion.py` — a derivative that
  ships inside the installed package and the wheel (imported by `script_util.py`
  in the same fork directory; that fork is not imported by the public GSSC-S2D2
  API, see its `__init__.py`). 222 of its 421 lines match
  `diffusion_utils/diffusion_multinomial.py` in the vendored snapshot (52.7 % of
  our file, 54.0 % of that 411-line upstream file), so Hoogeboom and Nielsen's
  work is a direct ancestor of code this project distributes, not only of code
  it vendors for reference.
* `src/gssc/models/pyramid_diffusion.py` — 170 matched lines (54.1 % of our
  file) against the same upstream file, reaching us through Pyramid Discrete
  Diffusion (§1).
  This one **is** live: it is imported by the pyramid training entry points
  (`src/gssc/training/train_pyramid_s2.py`, `train_pyramid_s3.py`,
  `pyramid_pipeline.py`) and exercised by `tests/test_smoke.py`.

If you intend to redistribute this material yourself, seek an explicit grant
from the upstream authors rather than relying on the classifier.

---

## 3. denoising-diffusion-pytorch (lucidrains)

* **Upstream:** <https://github.com/lucidrains/denoising-diffusion-pytorch>
* **License:** MIT. **Copyright (c) 2020 Phil Wang** — read from the upstream
  `LICENSE` on `main` (commit `faed4db`) on 2026-08-22.

`src/gssc/_improved_diffusion/multinomial_diffusion.py` credits this project in
its own module docstring ("Based in part on …"), and Pyramid Discrete Diffusion's
`con_diffusion.py` — the ancestor of both of our `*_diffusion.py` files (§1) —
carries the same credit line upstream. The MIT permission notice below therefore
travels with those files.

```
MIT License

Copyright (c) 2020 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 4. improved-diffusion (OpenAI)

* **Upstream:** <https://github.com/openai/improved-diffusion>
* **Paper:** Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic
  Models*, ICML 2021 (arXiv:2102.09672).
* **License:** MIT. **Copyright (c) 2021 OpenAI** — read from the upstream
  `LICENSE` (commit `1bc7bbb`) on 2026-08-22.

`src/gssc/_improved_diffusion/` is a fork of this project. The full MIT text and
the copyright line are reproduced in that directory's own
[`NOTICE`](src/gssc/_improved_diffusion/NOTICE), which ships in the wheel. The
four UNet variants in that directory are derivatives of upstream `unet.py`
(re-measured 2026-08-22 with the §1 procedure, against a clone at commit
`1bc7bbb`: 85.6 %, 92.1 %, 92.7 % and 93.2 % of the 547-line upstream file
reproduced in `unet_sparse.py`, `unet_multinomial.py`, `unet_factorized.py` and
`unet_old_fullres_baseline.py` respectively).

---

## 5. point-e (OpenAI)

* **Upstream:** <https://github.com/openai/point-e>
* **License:** MIT. **Copyright (c) 2022 OpenAI** — read from the upstream
  `LICENSE` (commit `fc8a607`) on 2026-08-22.

`src/gssc/_improved_diffusion/transformer.py` is derived from
`point_e/models/transformer.py`: 262 matched lines, 88.5 % of our file and
53.0 % of the upstream file (re-measured 2026-08-22, clone at commit
`fc8a607`). This is a **different OpenAI project** from improved-diffusion
(§4), so the directory NOTICE's "Code authors: OpenAI" is accurate for it only
by coincidence; it is recorded here explicitly.

---

## 6. 2DPASS

* **Upstream:** <https://github.com/yanx27/2DPASS>
* **Paper:** Yan et al., *2DPASS: 2D Priors Assisted Semantic Segmentation on
  LiDAR Point Clouds*, ECCV 2022.
* **License:** MIT. **Copyright (c) 2022 Benny** — the name in the upstream
  `LICENSE` (commit `80b8646`), read 2026-08-22; `yanx27` is the repository
  owner, not the name in the notice.

`src/gssc/_improved_diffusion/point_encoder.py` is adapted from
`network/voxel_fea_generator.py`: 41 matched lines, 27.0 % of our file and
45.1 % of the upstream file, including the voxel-range and grid-index
computation (re-measured 2026-08-22, clone at commit `80b8646`). A bare upstream
filename does not identify a project, so that file's own header names 2DPASS,
the licence and the holder, and points here; this entry is the full notice.

```
MIT License

Copyright (c) 2022 Benny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 7. semantic-kitti-api

* **Upstream:** <https://github.com/PRBonn/semantic-kitti-api>
* **License:** MIT. **Copyright (c) 2019, University of Bonn.** The upstream
  `LICENSE` file is reproduced verbatim at
  `external/semantic_kitti_api/LICENSE` — byte-identical to upstream, verified
  2026-08-22 by diff against a fresh clone (commit `a9c749e`).

Vendored at `external/semantic_kitti_api/` (43 tracked files) and used as the
official evaluator for every score this project reports under the "official
`semantic-kitti-api`" protocol. It is pinned in-tree so that a reported number
reflects one fixed scoring implementation.

**The vendored copy is modified, and MIT permits that provided the notice above
travels with it. Exactly where it differs** (measured 2026-08-22 by matching the
blob hash of each of our 44 files against every one of the upstream repository's
50 commits; the closest is `4398778`, 2022-11-24, which is the pin):

* 37 of the 40 vendored upstream files are byte-identical to that commit, and
  no upstream file at that commit is missing;
* `evaluate_completion.py`, the scorer itself, differs by **one line**:
  `np.ones_like(labels, dtype=np.bool)` became `dtype=bool`, `np.bool` being an
  alias NumPy removed in 1.24. Nothing else in it changed. Its module-level
  imports are `argparse`, `numpy`, `scipy.io`, `yaml`, `os` and `time`, and it
  pulls in exactly one other vendored file: `auxiliary/np_ioueval.py`, via a
  function-local `from auxiliary.np_ioueval import iouEval` at its line 129 —
  the IoU accumulator itself. That file is byte-identical to the pin (blob
  `d31b631e5eb1d3e5577a91ae8626566b7e66e36d`) and imports only `sys` and
  `numpy`, so no modified file is on the scoring path;
* `auxiliary/laserscan.py` adds a spatial crop before range projection and
  `visualize_voxels.py` adds commented-out debug lines; neither is imported by
  `evaluate_completion.py` or by any scoring path in this repository;
* three files are **added by this project**, not upstream:
  `config/semantic-kitti_my.yaml` and `config/kitti360.yaml`, both derived from
  upstream's own `config/semantic-kitti.yaml` (95.8 % and 60.1 % of their lines
  match it); and `config/semantic-poss.yaml`, a completion-scorer datacfg whose
  label maps mirror JS3C-Net's `SemanticPOSS.yaml` (§9). The three configs
  carry upstream's own "covered by the LICENSE file in the root of this project"
  header, which here resolves to `external/semantic_kitti_api/LICENSE`.

---

## 8. SCPNet

* **Upstream:** <https://github.com/SCPNet/Codes-for-SCPNet>
* **Paper:** Xia et al., *SCPNet: Semantic Scene Completion on Point Cloud*,
  CVPR 2023.
* **License: none is published.** A clone of the upstream repository (checked
  2026-08-22, HEAD `c0f55fa`) contains no `LICENSE`, `LICENCE`, `COPYING`,
  `NOTICE` or `COPYRIGHT` file anywhere in the tree, and its `README.md` states
  no terms.

What we redistribute: **`scpnet_v2_port.pth`**, in the released checkpoint
bundle. It is SCPNet's own released pretrained checkpoint
(`model_load_dir/pretrained.pth`), carried unmodified. Measured 2026-08-22 on
the released file: 233,916,679 bytes, SHA-256
`f2d1cb27f4285690b2f8322e6a87e6631cc1af26e25d006758f7ff65587ed106`, 285
state-dict entries, every one of them under SCPNet's own
`cylinder_3d_generator` (30) / `cylinder_3d_spconv_seg` (255) prefixes —
nothing in it is GSSC-authored. This project keeps no copy of the upstream
download to diff against, so "unmodified" is a statement of origin, not of a
byte comparison run here. The "port" in its name refers to spconv-2.3
kernel-shape patches applied **at load time** by
`src/gssc/inference/run_scpnet.py` (see `load_scpnet_checkpoint`), not to a
modification of the file contents. It is the frozen base model behind the
paper's headline results, so it is released for reproducibility.

**Because upstream publishes no license, this redistribution rests on explicit
permission: this project's authors state that the SCPNet authors were contacted
and granted permission for this release.** The file is distributed under this
project's MIT license with this attribution recording that permission; no
written grant travels with the release, so this notice is the record of it. No
implicit grant is claimed, and the permission covers this redistribution — if
you redistribute the weights further, obtain your own permission from the SCPNet
authors.

No SCPNet network code is vendored in this repository; `run_scpnet.py` expects a
user-provided upstream checkout (see `docs/BASELINES.md`).

---

## 9. JS3C-Net

* **Upstream:** <https://github.com/yanx27/JS3C-Net>
* **Paper:** Yan et al., *Sparse Single Sweep LiDAR Point Cloud Segmentation via
  Learning Contextual Shape Priors from Scene Completion*, AAAI 2021.
* **License:** MIT. **Copyright (c) 2020 Xu Yan** — read from the upstream
  `LICENSE` (commit `d505d0b`) on 2026-08-22.

No JS3C-Net model code is shipped in this repository:
`src/gssc/models/js3c_base.py` is our own reader for pre-dumped predictions, and
`scripts/dump_js3c_predictions.py` expects a user-provided upstream checkout.
What we publish is `js3cnet_predictions/` in the released dataset bundle —
per-frame voxel-grid predictions produced by running upstream JS3C-Net on
SemanticKITTI. As model output over SemanticKITTI, that data carries the dataset
terms in §11 as well as this attribution.

---

## 10. LMSCNet

* **Upstream:** <https://github.com/astra-vision/LMSCNet>
* **Paper:** Roldão, de Charette & Verroust-Blondet, *LMSCNet: Lightweight
  Multiscale 3D Semantic Completion*, 3DV 2020.
* **License:** Apache License 2.0. **Copyright 2020 Inria and AKKA
  Technologies** — read from the upstream `LICENSE` (commit `ea1b42d`) on
  2026-08-22. Full text: <https://www.apache.org/licenses/LICENSE-2.0>.

No LMSCNet model code is shipped in this repository:
`src/gssc/models/lmscnet_base.py` is our own reader for pre-dumped predictions.
What we publish is `lmscnet_predictions/` in the released dataset bundle —
per-frame voxel-grid predictions produced by running upstream LMSCNet on
SemanticKITTI, subject to the dataset terms in §11 as well as this
attribution.

---

## 11. Dataset terms and what they bind

This project trains and evaluates on datasets that are **non-commercial**, and
publishes artifacts derived from them. Those artifacts inherit the upstream
restrictions; the MIT grant on GSSC-authored code does **not** override them.

| Dataset | Terms (verified 2026-08-22 on the dataset's own page) | Required attribution |
|---|---|---|
| **SemanticKITTI** | CC BY-NC-SA 4.0 — <http://www.semantic-kitti.org/dataset.html> links <https://creativecommons.org/licenses/by-nc-sa/4.0/> | The two BibTeX entries that page prints under its licence terms: J. Behley et al., *SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences*, ICCV 2019; and A. Geiger, P. Lenz & R. Urtasun, *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite*, CVPR 2012, pp. 3354–3361 |
| **SemanticPOSS** | CC BY-NC-SA 3.0 — "This dataset follow Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License", <http://www.poss.pku.edu.cn/semanticposs.html> | Pan et al., *SemanticPOSS: A Point Cloud Dataset with Large Quantity of Dynamic Instances*, IEEE Intelligent Vehicles Symposium (IV) 2020 |
| **SSCBench-KITTI-360** | The SSCBench repository (`ai4ce/SSCBench`) publishes **no LICENSE file** — no licence file in its root listing and `license: null` from the GitHub API, checked 2026-08-22; the underlying KITTI-360 data is CC BY-NC-SA 3.0 — "published under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License", <https://www.cvlibs.net/datasets/kitti-360/> | Li et al., *SSCBench: A Large-Scale 3D Semantic Scene Completion Benchmark for Autonomous Driving*, IROS 2024; and Liao, Xie & Geiger, *KITTI-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D*, TPAMI 2022 |

**Which of our published artifacts inherit these terms:**

* **Derived from SemanticKITTI — CC BY-NC-SA 4.0 (non-commercial, share-alike,
  attribution) applies in full:**
  * the synthetic pool (`synthetic_pool_31K`, `synthetic_pool_57K`) — complete
    scenes generated by the pyramid diffusion models trained on SemanticKITTI,
    pasted with object-bank instances and ray-traced into sparse scans;
  * the object bank (57,789 rare-class instances) — cut directly from
    SemanticKITTI scenes;
  * the base-model prediction dumps `scpnet_predictions/`,
    `js3cnet_predictions/`, `lmscnet_predictions/` — model output computed over
    SemanticKITTI sequences, and additionally subject to §8, §9 and §10
    respectively.
* **Trained on SemanticKITTI — MIT as our contribution, but downstream use still
  carries the dataset's non-commercial restriction:** every released GSSC
  checkpoint, including the pyramid generators and the BEV models. An MIT grant
  on our weights cannot authorise commercial use of a model trained on
  CC BY-NC-SA data.
* **Not derived, and therefore not redistributed at all:** SemanticPOSS and
  SSCBench-KITTI-360 are used **evaluation-only, zero-shot**. This project
  publishes no data derived from either; obtain them from their own
  distributors under the terms above. No GSSC checkpoint is trained on them.

Note on the KITTI citation: semantic-kitti.org asks for the **CVPR 2012**
benchmark paper, which is what the table above names, while the manuscript's own
KITTI reference is the IJRR 2013 dataset paper (*Vision meets Robotics: The
KITTI Dataset*). Both are the same authors' work on the same data; the entry
above follows the data provider's stated requirement.

Raw SemanticKITTI, SemanticPOSS and KITTI-360 data is never redistributed by
this project; `scripts/download_assets.py` fetches only GSSC-released artifacts.

---

*Corrections to this file are welcome — open an issue. If you are an upstream
author and believe an attribution here is wrong or incomplete, please contact
the maintainers and it will be corrected.*

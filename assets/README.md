# Visual assets for the GSSC-S2D2 README + docs

This directory holds the figures referenced from the top-level README. All
three are the **paper's current figures**, rendered straight from the
manuscript's figure PDFs — no repo-authored composites, no panels stitched
together for the README.

Currently present and embedded in `README.md`:

| Filename | What is actually in it | Paper source | Where it appears |
|---|---|---|---|
| `teaser.png` | The PS³ offline data-augmentation pipeline, four stacked bands: **INPUT** (the real SemanticKITTI training split, 10 sequences / 19,130 frames / 256²×32 / 20 classes); **PYRAMID** (𝒮₁ 32²×4 → 𝒮₂ 64²×8 → 𝒮₃ 256²×32 at left, a zoom-in of the denoiser ε<sub>θ</sub> — x<sub>t</sub> → E1–E3 → B → D3–D1 → x̂₀, with *t* and **c** entering at the bottleneck — at right, then the JS-divergence screen); **HALO** (a virtual Velodyne HDL-64E ray-tracing the enriched dense scene back into a sparse sweep, with the 1st/2nd return callouts); **OBJECT BANK → OUTPUT** (eight rare-class thumbnails and the 19,130 + 32,039 = 51,169 / 2.67× arithmetic) | Fig. 2, `fig:ps3_pipeline` (`material/figures/fig4_da_pipeline.pdf`) | `README.md` "Method at a glance" |
| `architecture.png` | The S²D² refinement figure, three stacked bands: **Intuition** (the velocity field **v**<sub>θ</sub> dragging the base cluster onto the ground-truth cluster in one Euler step on the per-voxel simplex); **why one Euler step is viable** (DDPM's curved *T*-step path vs the straight chord from x<sub>src</sub>, with Prop. 3 / Thm. 1 / Cor. 1 boxed underneath); **a qualitative example** (val seq 08 frame 001390, motorcyclist IoU 27.3 % → 41.5 %, +14.2, against 322 GT voxels). It does **not** contain the four-level dense 3D U-Net panel — that is paper Fig. 4 (`fig:sgsc`) and is not shipped here | Fig. 5, `fig:s2d2` (`material/figures/fig3_s2d2_v3.pdf`) | `README.md` "Method at a glance" |
| `qualitative.png` | A 5-column × 2-row comparison grid — JS3C-Net, SCPNet, TALoS, S²D² (ours), Ground truth — over two val seq 08 frames: **top** bicyclist (frame 003096; per-sample IoU 0.7 / 31.3 / 33.0 % vs our 56.9 %, TP/FP chips, 2,833 GT voxels) and **bottom** motorcyclist (frame 001417; 0.0 / 33.0 / 30.0 % vs our 62.3 %, 315 GT voxels). Chips are the *N* = 4, +*D*<sub>4</sub>-TTA configuration | Fig. 6, `fig:qualitative` (`material/figures/fig4_qualitative.pdf`) | `README.md` "Headline numbers" |

Planned additions (not yet present in this directory and not yet referenced
from `README.md` — the convention below reserves their names so the README
can embed them without churn once they are dropped in):

| Filename | What | Intended location |
|---|---|---|
| `bev_qualitative.png` | BEV second-task qualitative — base BEV vs S²D² BEV vs GT | `README.md` "Secondary task" |
| `step_sweep.png` | Supplementary Tab. VI step reduction plot (mIoU vs N) | `README.md` "Reproducing every paper number" |
| `data_scaling.png` | Supplementary Tab. VII data-scaling plot (mIoU vs synthetic-pool size) | `README.md` "Reproducing every paper number" |
| `demo.gif` | Optional animation: S2D2 correction sampling over t=99 → t=0 | `README.md` hero |

## How figures get here

Each PNG is a 200-dpi rasterisation of the paper's figure PDF, produced with
`pdftoppm` (poppler). From a checkout of the paper repo:

```bash
pdftoppm -png -r 200 material/figures/fig4_da_pipeline.pdf teaser   && mv teaser-1.png   <repo>/assets/teaser.png
pdftoppm -png -r 200 material/figures/fig3_s2d2_v3.pdf    architecture && mv architecture-1.png <repo>/assets/architecture.png
pdftoppm -png -r 200 material/figures/fig4_qualitative.pdf qualitative  && mv qualitative-1.png  <repo>/assets/qualitative.png
```

That is the command the shipped files were checked against, not a
reconstruction: re-running it on 2026-08-22 reproduced all three
**pixel-identically** (max per-channel difference 0 over every pixel), at
1355 × 1890, 1356 × 1377 and 1355 × 882 respectively. If a figure changes in
the paper, re-run the line above rather than editing the PNG.

(Note: the paper repo is private until publication; the steps are documented
for completeness once it goes public.)

## Missing-figure behaviour

The core figures (`teaser.png`, `architecture.png`, `qualitative.png`)
are present and embedded in `README.md`, so the README renders fully on
GitHub today. The "Planned additions" above are not yet referenced from
`README.md`; until they are dropped in, their absence is a no-op (no
image tag points at them), so a missing file never breaks the README.

## License

All figures in this directory are released under the same **MIT** licence as
the rest of the codebase (see [`../LICENSE`](../LICENSE)). The *scenes*
rendered inside them are SemanticKITTI-derived, so they also carry that
dataset's CC-BY-NC-SA 4.0 non-commercial, share-alike terms; the dataset
terms and their required attributions are recorded in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) §11. Cite the paper
(`../CITATION.cff`) if you reuse them.

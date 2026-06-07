# Visual assets for the GSSC-S2D2 README + docs

This directory holds the figures and animations referenced from the
top-level README. Drop-in convention so the README never breaks:

Currently present and embedded in `README.md`:

| Filename | What | Where it appears |
|---|---|---|
| `teaser.png` | Hero teaser — paper Fig. 2 (two-stage pipeline rendered for the README) | top of `README.md`, above the badges |
| `architecture.png` | Method architecture — paper Fig. 3 (denoiser UNet + S2D2 correction sampler) | `README.md` "Method at a glance" |
| `qualitative.png` | Five-column qualitative comparison — paper Fig. 4 (left → right: JS3C-Net, SCPNet, TALoS, S²D² ours N=4, Ground Truth) over two seq-08 frames | `README.md` "Headline numbers" |

Planned additions (not yet present in this directory and not yet referenced
from `README.md` — the convention below reserves their names so the README
can embed them without churn once they are dropped in):

| Filename | What | Intended location |
|---|---|---|
| `bev_qualitative.png` | BEV second-task qualitative — base BEV vs S²D² BEV vs GT | `README.md` "Secondary task" |
| `step_sweep.png` | Tab. V step reduction plot (mIoU vs N) | `README.md` "Reproducing every paper number" |
| `data_scaling.png` | Tab. VII data-scaling plot (mIoU vs synthetic-pool size) | `README.md` "Reproducing every paper number" |
| `demo.gif` | Optional animation: S2D2 correction sampling over t=99 → t=0 | `README.md` hero |

## How figures get here

The figures are authored in TikZ + Blender Cycles in the paper repo
(`paper_writing/.../material/figures/`). To regenerate the PNGs:

```bash
# from the paper repo
cd material/figures && latexmk -pdf fig2_pipeline_v2.tex
gs -sDEVICE=pngalpha -r300 -o teaser.png fig2_pipeline_v2.pdf
cp teaser.png ../../GSSC-S2D2/assets/
```

(Note: the paper repo is private until publication; the steps are
documented for completeness once it goes public.)

## Missing-figure behaviour

The core figures (`teaser.png`, `architecture.png`, `qualitative.png`)
are present and embedded in `README.md`, so the README renders fully on
GitHub today. The "Planned additions" above are not yet referenced from
`README.md`; until they are dropped in, their absence is a no-op (no
image tag points at them), so a missing file never breaks the README.

## License

All figures in this directory are released under the same Apache 2.0
license as the rest of the codebase. Cite the paper (`CITATION.cff`)
if you reuse them.

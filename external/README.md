# Vendored third-party code

Each subdirectory here is an in-tree pin of an external project, kept in the
tree so that scoring and training stay reproducible without a network fetch.
The provenance of each vendored copy:

| Directory | Upstream | Pin | Notes |
|---|---|---|---|
| `semantic_kitti_api/` | [PRBonn/semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api) | pinned snapshot | Official evaluator used for all `semantic-kitti-api` mIoU/CompIoU numbers. MIT (© 2019 University of Bonn). |
| `multinomial_diffusion/` | [ehoogeboom/multinomial_diffusion](https://github.com/ehoogeboom/multinomial_diffusion) (upstream D3PM / Multinomial Diffusion) | pinned snapshot | Reference categorical-diffusion implementation. MIT; see its `NOTICE`. |

Pinning the evaluator in-tree means a reported score reflects one fixed scoring
implementation rather than whatever version a user happens to clone. Each copy
keeps its own upstream license; the licenses are reproduced under each
subdirectory (`semantic_kitti_api/LICENSE`, `multinomial_diffusion/NOTICE`).

# Security Policy

## Supported versions

Only the latest tagged release of GSSC-S2D2 receives security updates.
The `main` branch tracks the bleeding edge and may be in flux.

| Version | Supported          |
|---------|--------------------|
| `2.4.x` | :white_check_mark: |
| `< 2.4` | :x:                |

## Reporting a vulnerability

If you discover a security vulnerability in GSSC-S2D2, please:

1. **Do not** open a public GitHub issue.
2. **Do not** post the vulnerability on social media or other public channels.
3. Email **shichen22@m.fudan.edu.cn** with:
   - A description of the vulnerability
   - Steps to reproduce
   - The version / commit you tested against
   - (Optional) a suggested fix

You should receive an acknowledgment within 5 working days. We aim to:

- Triage and confirm within 10 working days
- Issue a fix within 30 working days for high-severity issues
- Coordinate a public disclosure date with you

## Scope

Security issues we treat as in-scope:

- Code-execution vulnerabilities in the inference / training pipelines
- Deserialization attack vectors via `torch.load` on hostile checkpoints
- Path-traversal in dataset / checkpoint loaders
- Hardcoded credentials or tokens (we do not ship any)

Out of scope (do not report as security issues):

- Bugs in GPU kernels or upstream dependencies (`spconv`, `torch`,
  `huggingface_hub`) -- file those upstream
- Reproducibility issues (use the [issues tracker](https://github.com/BillyChern/GSSC-S2D2/issues))
- Performance regressions

## Trust model for checkpoints

GSSC-S2D2 loads checkpoints with `torch.load(..., weights_only=False)`
because the saved state carries optimizer + EMA buffers that are not
representable in `weights_only=True` mode. **Only load checkpoints from
sources you trust** -- running an attacker-supplied `.pt` / `.pth` file is
equivalent to running attacker-supplied code. The released weights are
`model.safetensors`, a format that cannot carry executable payloads; the two
pickles in the release are the third-party `scpnet_v2_port.pth` base and
`bev/bev_s2d2_scpnet/model.pt` (the pre-conversion copy of the BEV weights --
the `.safetensors` beside it is what the documented BEV eval command loads),
and both are covered by the digests below.

### Verifying what you downloaded

Every released checkpoint has a published SHA256 digest. The per-file table is
[docs/MODEL_ZOO.md](docs/MODEL_ZOO.md); the same digests ship as one
`checksums.txt` at the root of the Hugging Face checkpoints repo, which the download
unpacks into `data/checkpoints/`, so the check is a single command with a verdict
rather than a digest you have to eyeball:

```bash
# Downloads into data/checkpoints/, checksums.txt included.
python scripts/download_assets.py --checkpoints

# The paths inside checksums.txt are relative to `checkpoints/` (the first entry is
# `MANIFEST.txt`, not `checkpoints/MANIFEST.txt`), so run the check from INSIDE
# data/checkpoints/. Running it from data/ makes every line FAIL open-or-read.
cd data/checkpoints && sha256sum -c checksums.txt
```

Every line must print `OK`, and `sha256sum` must exit 0. Do not load a file whose
line prints `FAILED`, and do not dismiss a `FAILED open or read` line either --
a file the manifest lists and your download does not have is an incomplete
transfer, not a harmless difference.

To check one file rather than all of them:

```bash
cd data/checkpoints && grep 'gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors' \
    checksums.txt | sha256sum -c -
```

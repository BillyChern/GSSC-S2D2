# Security Policy

## Supported versions

Only the latest tagged release of GSSC-S2D2 receives security updates.
The `main` branch tracks the bleeding edge and may be in flux.

| Version | Supported          |
|---------|--------------------|
| `2.1.x` | :white_check_mark: |
| `< 2.1` | :x:                |

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
sources you trust** -- running an attacker-supplied `.pt` file is
equivalent to running attacker-supplied code.

Published checkpoints will ship with SHA256 hashes documented in
[docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) on release. Verify before loading:

```bash
sha256sum data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors
```

# Contributing to GSSC-S2D2

By participating you agree to the [Code of Conduct](.github/CODE_OF_CONDUCT.md)
(Contributor Covenant 2.1). Security issues go to [SECURITY.md](SECURITY.md), not
to the issue tracker.

## Code-quality standards

This repository targets **Google/Apple production-grade standards**.

### Hard requirements (CI-enforced)

Every row below is run by a job under `.github/workflows/`. A row is listed here only
if a workflow runs the command at the scope the row states.

| Standard | Tool | Command | Workflow |
|---|---|---|---|
| Style + import order | `ruff` | `ruff check src/ tests/ scripts/` | lint.yml |
| Tests | `pytest` | `pytest tests/ --ignore=tests/test_tau_invariance.py -v` | test.yml |
| Static types | `mypy` | `mypy src/gssc/inference src/gssc/utils` | lint.yml |

The test job runs on a torch-free CPU runner: 43 cases execute and 33 skip themselves
through `pytest.importorskip`. `tests/test_tau_invariance.py` is ignored because it
imports torch at module scope without that guard, so it cannot even be collected there.

### Run locally (no workflow runs these)

Useful, but deliberately not wired into a workflow -- each one either needs hardware the
runner does not have, or would fail today. Do not move a row up without first making the
workflow green.

| Standard | Tool | Command | Why it is not in a workflow |
|---|---|---|---|
| Full suite | `pytest` | `pytest tests/ -v` | 116 cases, needs torch 2.4 + spconv on the runner |
| Coverage | `pytest-cov` | `pytest --cov` | pyproject pins `fail_under = 80`; the CPU-runnable suite reaches ~51 %, so the gate would fail |
| Pre-commit | `pre-commit` | `pre-commit run --all-files` | its `pytest-light` hook is `language: system` and assumes a prepared local interpreter |
| Dead code | `vulture` | `vulture src/` | advisory -- no workflow runs it |
| Unused imports | `pyflakes` | `pyflakes src/` | advisory -- no workflow runs it |
| Security | `bandit` | `bandit -r src/` | advisory -- no workflow runs it |
| Strict types | `mypy --strict` | `mypy --strict src/gssc/inference` | advisory -- no workflow runs it |

> **Scope note.** Type-checking is enforced on the public-API surface
> (`gssc.inference`, `gssc.utils`) where annotations are complete.
> Legacy modules under `src/gssc/_improved_diffusion/`
> and `src/gssc/training/train_pyramid_*.py`
> are deliberately excluded from style + type gating until they're
> refactored or removed. The exclusion lists are not prose: they are
> `[tool.ruff] extend-exclude` and `[tool.mypy] files` in `pyproject.toml`,
> which is also where the gate reads them from.

### Style conventions

* **Type hints on every public function** (Python 3.10+ syntax: `list[int]`, `int | None`).
* **Google-style docstrings** for every public class and function:
  ```python
  def f(x: int) -> int:
      """One-line summary.

      Args:
          x: What x is.

      Returns:
          What is returned.

      Raises:
          ValueError: When x is negative.
      """
  ```
* **No `print()`**: use `logging.getLogger(__name__)` and explicit log levels.
* **No magic numbers**: every numeric constant goes into a config or named module-level constant.
* **No silent failures**: never `except: pass` or `except Exception:`. Catch the specific exception class.
* **Pure functions where possible**: side-effecting functions should make the side-effect explicit in the name (`save_*`, `load_*`, `update_*`).
* **Small functions**: target ≤50 lines per function; refactor longer ones unless they are mathematically dense (e.g. a single derivation).
* **Deterministic seeding**: every entry point seeds Python, NumPy, and PyTorch.
* **No global state** beyond module-level constants and registries.

### Module-level boilerplate

Every Python module starts with:

```python
"""<one-line summary>.

<longer description if needed>.
"""
from __future__ import annotations

import logging
from typing import ...

logger = logging.getLogger(__name__)
```

### Tests

* New code requires unit tests in `tests/` mirroring the source layout.
* Tests must pass on a CPU-only machine for everything outside `tests/gpu/`.
* Mark GPU-required tests with `@pytest.mark.gpu` and slow tests with `@pytest.mark.slow`.

### Commits + PRs

* Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
* One logical change per commit. Squash on merge if the branch has cleanup commits.
* Run `ruff check src/ tests/ scripts/`, `mypy`, and `pytest tests/ -v` locally before
  pushing -- the same three commands the workflows run.

### Things that will fail review

* `print()` for debugging (use `logger.debug`).
* `except Exception:` without re-raising.
* Hardcoded paths (`"/data/foo"`) — must come from config.
* Mutable default arguments (`def f(x: list = []):`).
* Module-level side effects (e.g. file I/O at import time).
* `from x import *`.
* Untyped public APIs.

# Contributing to GSSC-S2D2

## Code-quality standards

This repository targets **Google/Apple production-grade standards**.

### Hard requirements (CI-enforced)

| Standard | Tool | Command | Scope |
|---|---|---|---|
| Style + import order | `ruff` | `ruff check src/ tests/ scripts/` | All non-vendored code |
| Tests | `pytest` | `pytest tests/ -v` | 80 cases, all CPU-runnable |
| Coverage | `pytest-cov` | `pytest --cov --cov-fail-under=80` | `src/gssc/{utils,inference}` (the testable subset) |
| Static types | `mypy` | `mypy src/gssc/inference src/gssc/utils` | Public-API modules only |
| Pre-commit | `pre-commit run --all-files` | local + CI | every commit |

### Aspirational (run manually until CI gates are added)

| Standard | Tool | Command |
|---|---|---|
| Dead code | `vulture` | `vulture src/` |
| Unused imports | `pyflakes` | `pyflakes src/` |
| Security | `bandit` | `bandit -r src/` |
| Strict types | `mypy --strict` | `mypy --strict src/gssc/inference` |

> **Scope note.** Type-checking is enforced on the public-API surface
> (`gssc.inference`, `gssc.utils`) where annotations are complete.
> Legacy modules under `src/gssc/_improved_diffusion/`,
> `src/gssc/models/extras_*.py`, and `src/gssc/training/train_pyramid_*.py`
> are deliberately excluded from style + type gating until they're
> refactored or removed (tracked in `CHANGELOG.md` Unreleased).

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
* Run `ruff check && mypy --strict src/gssc && pytest` locally before pushing.

### Things that will fail review

* `print()` for debugging (use `logger.debug`).
* `except Exception:` without re-raising.
* Hardcoded paths (`"/data/foo"`) — must come from config.
* Mutable default arguments (`def f(x: list = []):`).
* Module-level side effects (e.g. file I/O at import time).
* `from x import *`.
* Untyped public APIs.

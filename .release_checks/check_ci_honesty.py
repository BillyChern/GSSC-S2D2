#!/usr/bin/env python3
"""GATE: CI must actually run what it claims to run, and what it runs must pass.

THE THREE DEFECTS THIS GATE EXISTS FOR (all measured 2026-08-20 at HEAD 07725af)
-------------------------------------------------------------------------------
D1  `.github/workflows/lint.yml` runs, verbatim, `ruff check src/ tests/ scripts/`.
    That command FAILS at HEAD:
        F401 `os` imported but unused --> tests/test_scratch_space.py:18:8
        Found 1 error.
    The lint badge is therefore red-by-construction for anybody who pushes.

D2  `.github/workflows/test.yml` installs exactly `pip install pytest pyyaml`, then runs
    four selected node ids.  On a CLEAN runner that command dies AT COLLECTION:
        tests/test_evaluate_parser.py:14 -> gssc.inference.__init__ ->
        gssc/inference/evaluate_bev.py:34 -> `import numpy as np`  --> ImportError
    It is green on the author's box only because torch/numpy are installed there.  This is
    the exact failure mode a "CI is green" claim cannot see from inside the box that has
    the dependencies.  So the check does not trust the local environment: it re-runs the
    workflow's own command with every third-party import ORIGINATING IN REPO CODE blocked
    unless the workflow declared it.  Measured: 1 error during collection, exit 2.

D3  `CONTRIBUTING.md:7` heads a table "Hard requirements (CI-enforced)" with FIVE rows.
    CI enforces TWO of them (ruff, mypy).  Unenforced: `pytest tests/ -v` ("80 cases" --
    CI runs 4 node ids from 3 of the 13 test files), `pytest --cov --cov-fail-under=80`
    (no workflow runs coverage at all; the local coverage number is ~57%, i.e. the gate
    would fail if it were wired up), and `pre-commit run --all-files` (labelled
    "local + CI"; no workflow mentions pre-commit).  A contributor reads that table as a
    contract.

WHY EACH CHECK IS SHAPED THE WAY IT IS
--------------------------------------
*Run the command, do not pattern-match it.*  A gate that greps a workflow for "ruff" would
be green today.  Only executing `ruff check src/ tests/ scripts/` finds D1, and only
executing pytest under a restricted import set finds D2.

*Execute in a COPY.*  Every command runs against a copy of the tracked + untracked-not-
ignored worktree under $TMPDIR (265 files, 6.6 MB measured), never against the repo:
`pytest` writes `.pytest_cache`, `ruff` writes `.ruff_cache`, and this harness must not
mutate the artefact it measures.  The copy is rebuilt per run, so uncommitted edits the
author is making right now ARE measured -- a `git archive HEAD` copy would silently test
the wrong tree during a fix cycle.

*The clean-runner probe must prove it was ACTIVE.*  A blocker that fails to load turns the
check into a vacuous pass -- the classic way this class of instrument dies.  The probe
therefore drops a sentinel file on import, and a run whose sentinel is missing is reported
as UNMEASURABLE (a FAIL), never as a pass.

*The probe blocks only REPO-ORIGIN imports.*  First cut blocked every non-declared module
and died inside pytest's own `_pytest._io.terminalwriter -> import pygments`.  pytest's
optional dependencies are part of the RUNNER, not of the project's dependency claim; the
blocker walks the stack and fires only when a frame under the repo asked for the module.

*The CI-enforced table is checked by SCOPE, not by tool name.*  `pytest` appears in a
workflow, so a name-only test would call the "Tests" row enforced.  The row promises the
directory `tests/`; the workflow selects four node ids.  Scope resolution expands both
sides to FILE SETS and requires workflow >= doc.  `mypy` invoked bare resolves its scope
from `[tool.mypy] files` in pyproject.toml, which is the only honest way to compare a
config-driven invocation with a documented path list.

*Unrunnable commands are named, not skipped.*  EXEMPT is an allowlist of EXEMPTIONS with
reasons, never an allowlist of targets: anything not listed and not runnable FAILS check
C4 rather than vanishing.  An allowlist of targets fails silent; this one fails loud.

DIRECTION OF THE FIX
--------------------
C1 -> fix the code (or the workflow command), not this gate.  C2 -> add the missing
packages to the workflow's install step (torch/numpy/spconv are heavy; the alternative is
to stop advertising those test files in a torch-free job).  C3 -> either wire the missing
gates into a workflow or move those rows to the "Aspirational" table below them.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml

REPO = Path("/workspace/GSSC-S2D2")
WORKFLOW_DIR = REPO / ".github" / "workflows"
CONTRIBUTING = REPO / "CONTRIBUTING.md"
PYPROJECT = REPO / "pyproject.toml"

#: Where the tools a workflow names actually live on this box.  A workflow says `ruff`;
#: GitHub's runner gets it from `uv pip install`.  We cannot install, so we point PATH at
#: the project venv.  A tool that is missing makes the command UNMEASURABLE -> FAIL, never
#: a pass: "the tool isn't here" is not evidence that CI is green.
VENV_BIN = Path("/workspace/Semantic_Scene_Completion_LiDAR/.venv/bin")

CMD_TIMEOUT = 900

#: Distribution name -> import name, where they differ.  Needed because the clean-runner
#: probe allows IMPORTS and the workflow declares DISTRIBUTIONS.
IMPORT_NAME: Dict[str, Tuple[str, ...]] = {
    "pyyaml": ("yaml", "_yaml"),
    "types-pyyaml": (),          # stub-only: contributes no runtime import
    "pytest-cov": ("pytest_cov", "coverage"),
    "pre-commit": ("pre_commit",),
}

#: Always importable on a runner regardless of the install step.
ALWAYS_ALLOWED: Tuple[str, ...] = ("pytest", "_pytest", "py", "pluggy", "iniconfig",
                                  "packaging", "setuptools", "pkg_resources", "gssc")

#: EXEMPT: (regex over the command, reason).  Measured, one entry per real cause.
EXEMPT_COMMANDS: Sequence[Tuple[str, str]] = (
    (r"\b(?:uv\s+)?pip\s+install\b",
     "network install; on this box it cannot run, and its CONTENT is what check C2 "
     "measures instead of its exit code"),
    (r"\bapt-get\b|\bbrew\b",
     "system package install; network + root"),
    (r"\$\{\{",
     "consumes a GitHub Actions context expression, which has no value outside Actions"),
    (r"\bGITHUB_OUTPUT\b|\bGITHUB_ENV\b",
     "writes to an Actions-provided file that does not exist outside Actions"),
    (r"\bgit\s+describe\b[\s\S]*RELEASE_NOTES\.md",
     "release.yml's changelog script runs only on a pushed v* tag: it consumes ${TAG} "
     "from a step env fed by ${{ github.ref_name }} and needs the tag's own checkout"),
)

#: EXEMPT scope narrowings for the CI-enforced table.  A row is honest if CI runs the
#: documented scope OR the doc itself declares the narrowing.
EXEMPT_SCOPE: Sequence[Tuple[str, str]] = (
    ("mypy",
     "CONTRIBUTING.md:26-31 states the type-gating scope note verbatim (public-API "
     "surface only; `src/gssc/_improved_diffusion/` and `training/train_pyramid_*` "
     "deliberately excluded), and pyproject [tool.mypy] files= encodes exactly that. "
     "The narrowing is DECLARED, so the row is not a false claim"),
)

COSMETIC_ARGS = {"-v", "-q", "-ra", "--verbose", "--quiet", "-vv"}


# --------------------------------------------------------------------------------------
class Gate:
    def __init__(self) -> None:
        self.failures: List[Tuple[str, str]] = []
        self.results: List[Tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, ok, detail))
        if not ok:
            self.failures.append((name, detail))
        return ok

    def report(self) -> int:
        for name, ok, detail in self.results:
            print(f"  PASS  {name}" if ok else f"  FAIL  {name}   ({detail})")
        n = len(self.failures)
        print("OK: 0 failing check(s)" if n == 0 else f"FAILED: {n} failing check(s)")
        return 0 if n == 0 else 1


class Cmd:
    """One `run:` block of one step, with enough provenance to be actionable."""

    def __init__(self, wf: Path, job: str, step: int, name: str, script: str,
                 env_has_ctx: bool, installs: Tuple[str, ...], line: int) -> None:
        self.wf, self.job, self.step, self.name = wf, job, step, name
        self.script = script
        self.env_has_ctx = env_has_ctx
        self.installs = installs
        self.line = line

    @property
    def where(self) -> str:
        return f"{self.wf.name}:{self.line} ({self.job}/step{self.step} {self.name!r})"

    def __repr__(self) -> str:
        return f"<Cmd {self.where} {self.script.splitlines()[0][:60]!r}>"


# --------------------------------------------------------------------------------------
# Workflow parsing
# --------------------------------------------------------------------------------------
def _line_of(text: str, needle: str) -> int:
    first = next((l for l in needle.splitlines() if l.strip()), "")
    for i, line in enumerate(text.splitlines(), 1):
        if first.strip() and first.strip() in line:
            return i
    return 0


def _install_pkgs(script: str) -> Tuple[str, ...]:
    """Distributions a `pip install` line declares.  Version specs and quotes stripped."""
    pkgs: List[str] = []
    for m in re.finditer(r"(?:uv\s+)?pip\s+install\s+([^\n;&|]+)", script):
        for tok in m.group(1).split():
            tok = tok.strip("'\"")
            if tok.startswith("-"):
                continue
            name = re.split(r"[<>=!~\[]", tok)[0].strip()
            if name:
                pkgs.append(name.lower())
    return tuple(pkgs)


def parse_workflows(wf_dir: Path) -> List[Cmd]:
    cmds: List[Cmd] = []
    for wf in sorted(list(wf_dir.glob("*.yml")) + list(wf_dir.glob("*.yaml"))):
        text = wf.read_text()
        # `on:` parses to the boolean True under YAML 1.1; harmless here, we only read jobs.
        doc = yaml.safe_load(text) or {}
        for job_name, job in (doc.get("jobs") or {}).items():
            job_env_ctx = "${{" in yaml.safe_dump(job.get("env") or {})
            installs: List[str] = []
            steps = job.get("steps") or []
            for st in steps:
                installs.extend(_install_pkgs(str(st.get("run") or "")))
            for i, st in enumerate(steps, 1):
                script = st.get("run")
                if not script:
                    continue
                env_ctx = job_env_ctx or "${{" in yaml.safe_dump(st.get("env") or {})
                cmds.append(Cmd(wf, str(job_name), i, str(st.get("name") or f"run #{i}"),
                                str(script), env_ctx, tuple(installs),
                                _line_of(text, str(script))))
    return cmds


def exemption_for(cmd: Cmd) -> Optional[str]:
    for pat, reason in EXEMPT_COMMANDS:
        if re.search(pat, cmd.script):
            return reason
    if cmd.env_has_ctx:
        return ("the step's env: block is fed by a GitHub Actions context expression, so "
                "its inputs do not exist outside Actions")
    return None


# --------------------------------------------------------------------------------------
# Sandbox: a copy of the worktree the commands may write into
# --------------------------------------------------------------------------------------
def snapshot(root: Path, dest: Path) -> int:
    """Tracked + untracked-not-ignored files.  NOT `git archive HEAD`: during a fix cycle
    the author's uncommitted edits are exactly what must be measured."""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-co", "--exclude-standard",
                          "-z"], capture_output=True, text=True).stdout
    rels = [r for r in out.split("\0") if r]
    n = 0
    for rel in rels:
        src = root / rel
        if not src.is_file():
            continue
        tgt = dest / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, tgt)
        n += 1
    return n


def _probe_files(sandbox: Path, sentinel: Path) -> None:
    """Install the clean-runner import blocker where the interpreter will find it.

    `sitecustomize` is imported during interpreter start-up, so it must sit on a PYTHONPATH
    entry.  test.yml's own command sets `PYTHONPATH=src` INLINE, which replaces anything we
    export -- so the probe is written into both the sandbox root and `src/`.  (Discovering
    that inline override is why this is not simply an exported PYTHONPATH.)
    """
    blocker = '''
import os, sys, importlib.abc, importlib.machinery, traceback, pathlib
_repo = os.environ.get("CLEAN_RUNNER_REPO")
_allow = set(os.environ.get("CLEAN_RUNNER_ALLOW", "").split(",")) | set(sys.stdlib_module_names)
_sent = os.environ.get("CLEAN_RUNNER_SENTINEL")
if _sent:
    pathlib.Path(_sent).write_text("loaded")


def _needs_install(name, path):
    """True only when `name` resolves to an INSTALLED DISTRIBUTION.

    A name-only allowlist is not enough. `sys.stdlib_module_names` omits the generated
    per-platform data modules -- `_sysconfigdata__x86_64-linux-gnu`, which sysconfig
    imports lazily -- so it blocked a module that ships WITH the interpreter and that no
    `pip install` line could ever declare. That cost one spurious test failure and a
    false FAIL for this entire gate on 2026-08-20; a real clean venv holding only the
    three declared distributions ran the same command green.

    What `pip install` actually governs is site-packages, so that is what gets tested.
    PathFinder is used rather than importlib.util.find_spec because it does not consult
    sys.meta_path and so cannot recurse back into this finder.
    """
    try:
        spec = importlib.machinery.PathFinder.find_spec(name, path)
    except (ImportError, AttributeError, ValueError):
        return False
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin or origin in ("built-in", "frozen", "namespace"):
        return False
    return "site-packages" in origin or "dist-packages" in origin


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in _allow:
            return None
        if not _needs_install(name, path):
            return None      # stdlib, builtin, or repo-local: no install declares it
        # Innermost repo frame first: the OUTERMOST one is the test file, but the module
        # that actually does the undeclared import is usually several frames deeper
        # (tests/test_evaluate_parser.py:14 -> gssc/inference/evaluate_bev.py:34 ->
        # `import numpy`). Reporting the outer frame sends the author to the wrong file.
        for fr in reversed(traceback.extract_stack()[:-1]):
            fn = fr.filename
            if fn.startswith("<"):
                continue
            try:
                real = os.path.realpath(fn)
            except OSError:
                continue
            if not os.path.isfile(real):
                continue
            if real.startswith(_repo) and "site-packages" not in real:
                rel = os.path.relpath(real, _repo)
                raise ImportError(
                    "[clean-runner] repo code imports %r at %s:%d, but the workflow's "
                    "install step never declares it" % (top, rel, fr.lineno))
        return None


if _repo:
    sys.meta_path.insert(0, _Blocker())
'''
    site = 'import os\nif os.environ.get("CLEAN_RUNNER_REPO"):\n    import _clean_runner\n'
    for d in (sandbox, sandbox / "src"):
        d.mkdir(parents=True, exist_ok=True)
        (d / "_clean_runner.py").write_text(blocker)
        (d / "sitecustomize.py").write_text(site)


def run_script(script: str, cwd: Path, extra_env: Optional[Dict[str, str]] = None
               ) -> Tuple[int, str]:
    env = {**os.environ, "PATH": f"{VENV_BIN}:{os.environ.get('PATH', '')}"}
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    env.update(extra_env or {})
    try:
        p = subprocess.run(["bash", "-eo", "pipefail", "-c", script], cwd=str(cwd),
                           capture_output=True, text=True, env=env, timeout=CMD_TIMEOUT)
    except subprocess.TimeoutExpired:
        return (124, f"timed out after {CMD_TIMEOUT}s")
    lines = [l.strip() for l in (p.stdout + p.stderr).splitlines() if l.strip()]
    # The probe's own ImportError names the file:line that imports the undeclared module.
    # It is the actionable line and it is NOT near the end of pytest output, so hoist it:
    # a detail string that only says "1 error during collection" sends the author hunting.
    marked = [l for l in lines if "[clean-runner]" in l]
    # An import the suite HANDLES (pytest.importorskip -> SKIPPED) is the probe working as
    # designed, not a defect; only an UNHANDLED one is. Ranking a benign SKIPPED line first
    # pushed the real failure past the 400-char cut, so the detail string showed pytest's
    # syntax-highlighted echo of this module's own source -- `%r` placeholders and all --
    # and read like the gate was broken rather than like a test was failing.
    hot = [l for l in marked if "FAILED" in l or "ERROR" in l or l.startswith("E ")]
    lead = (hot or [l for l in marked if "SKIPPED" not in l] or marked)[:1]
    tail = lead + lines[-4:]
    return (p.returncode, " | ".join(tail)[:400])


def missing_tools(script: str) -> List[str]:
    """Executables the script invokes that PATH cannot provide.

    Backslash continuations are JOINED first.  Without that, test.yml's four-line pytest
    invocation parsed as four separate commands and the gate reported that PATH could not
    provide `tests` and `-v` -- a false UNMEASURABLE that ALSO removed the one command the
    clean-runner check exists to run, leaving C2 with nothing to measure and a detail
    string blaming the wrong thing. Measured 2026-08-20, first run of this gate.
    """
    flat = re.sub(r"\\\s*\n\s*", " ", script)
    miss: List[str] = []
    for stmt in re.split(r"[;\n]|&&|\|\||\|", flat):
        stmt = stmt.strip()
        if not stmt or stmt.startswith("#"):
            continue
        m = re.match(r"(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*(\S+)", stmt)
        if not m:
            continue
        tool = m.group(1)
        if tool.startswith("-") or "/" in tool or "=" in tool or "::" in tool:
            continue
        if tool in {"set", "echo", "if", "fi", "then", "else", "elif", "for", "while",
                    "do", "done", "{", "}", "cd", "export", "true", "false", "exit",
                    "printf", "source", "."}:
            continue
        if shutil.which(tool, path=f"{VENV_BIN}:{os.environ.get('PATH', '')}") is None:
            miss.append(tool)
    return miss


# --------------------------------------------------------------------------------------
# CONTRIBUTING "CI-enforced" table
# --------------------------------------------------------------------------------------
class Row:
    def __init__(self, standard: str, tool: str, command: str, line: int) -> None:
        self.standard, self.tool, self.command, self.line = standard, tool, command, line


def parse_ci_table(path: Path) -> Tuple[List[Row], int]:
    """Rows under the heading that claims CI enforcement.

    The heading is matched structurally ("CI" + an enforcement word), not by the literal
    "(CI-enforced)": a rename to "CI enforced" or "enforced in CI" must not silence the
    gate.  Row 5 of the live table is column-shifted (`| Pre-commit |
    `pre-commit run --all-files` | local + CI | every commit |`), so the command is taken
    as the LONGEST backticked cell rather than "column 3".
    """
    lines = path.read_text().splitlines()
    start = None
    for i, l in enumerate(lines):
        if l.lstrip().startswith("#") and re.search(r"\bCI\b", l) and \
                re.search(r"enforc", l, re.I):
            start = i
            break
    if start is None:
        return ([], 0)
    rows: List[Row] = []
    for i in range(start + 1, len(lines)):
        l = lines[i]
        if l.lstrip().startswith("#"):
            break
        if not l.strip().startswith("|"):
            continue
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if not cells or set("".join(cells)) <= set("-: "):
            continue
        if cells[0].lower() in {"standard"}:
            continue
        codes = [c.strip("` ") for c in re.findall(r"`[^`]+`", l)]
        if not codes:
            continue
        command = max(codes, key=lambda c: (len(c.split()), len(c)))
        tool = codes[0].split()[0]
        rows.append(Row(cells[0], tool, command, i + 1))
    return (rows, start + 1)


# --------------------------------------------------------------------------------------
# Scope resolution: expand a command to the set of files it actually acts on
# --------------------------------------------------------------------------------------
def _expand(root: Path, target: str, pattern: str) -> Set[str]:
    p = (root / target).resolve()
    if p.is_dir():
        return {str(q.relative_to(root)) for q in p.rglob(pattern)
                if "__pycache__" not in q.parts}
    if p.is_file():
        return {str(p.relative_to(root))}
    return set()


def mypy_config_files(root: Path) -> Set[str]:
    """`mypy` with no path arguments takes its scope from pyproject `[tool.mypy] files`."""
    txt = (root / "pyproject.toml").read_text() if (root / "pyproject.toml").is_file() else ""
    block = re.search(r"\[tool\.mypy\](.*?)(?:\n\[|\Z)", txt, re.S)
    if not block:
        return set()
    files_m = re.search(r"^files\s*=\s*\[(.*?)\]", block.group(1), re.S | re.M)
    if not files_m:
        return set()
    excl = set(re.findall(r'"([^"]+)"',
                          (re.search(r"^exclude\s*=\s*\[(.*?)\]", block.group(1),
                                     re.S | re.M) or re.match("", "")).group(1)
                          if re.search(r"^exclude\s*=\s*\[", block.group(1), re.M) else ""))
    out: Set[str] = set()
    for t in re.findall(r'"([^"]+)"', files_m.group(1)):
        out |= _expand(root, t, "*.py")
    return {f for f in out if f not in excl}


def resolve_scope(root: Path, command: str) -> Tuple[str, Set[str], Set[str]]:
    """(tool, files acted on, significant flags)."""
    toks = command.split()
    toks = [t for t in toks if not re.match(r"^[A-Z_]+=", t)]
    if toks and toks[0] in {"python", "python3"}:
        toks = toks[1:]
        if toks and toks[0] == "-m":
            toks = toks[1:]
    tool = toks[0] if toks else ""
    args = toks[1:]
    flags = {a for a in args if a.startswith("-") and a not in COSMETIC_ARGS}
    paths = [a for a in args if not a.startswith("-")]
    files: Set[str] = set()
    if tool == "pytest":
        for p in paths:
            files |= _expand(root, p.split("::")[0], "test_*.py")
        if not paths:
            files |= _expand(root, "tests", "test_*.py")
    elif tool in {"ruff", "mypy", "vulture", "pyflakes", "bandit"}:
        sub = {p for p in paths if p in {"check", "format", "-r"}}
        for p in paths:
            if p in sub:
                continue
            files |= _expand(root, p, "*.py")
        if tool == "mypy" and not files:
            files = mypy_config_files(root)
        flags |= sub
    return (tool, files, flags)


def scope_exemption(tool: str) -> Optional[str]:
    for t, reason in EXEMPT_SCOPE:
        if t == tool:
            return reason
    return None


# --------------------------------------------------------------------------------------
def run(root: Path, gate: Optional[Gate] = None, tmp: Optional[Path] = None) -> Gate:
    g = gate or Gate()
    wf_dir = root / ".github" / "workflows"
    tmp = tmp or Path(tempfile.mkdtemp(prefix="ci_honesty_"))
    cmds = parse_workflows(wf_dir) if wf_dir.is_dir() else []

    # ---- C4 first: classification honesty.  If a command is neither runnable nor
    # exempted-with-a-reason it must FAIL here rather than be quietly dropped from C1.
    unclassified: List[str] = []
    runnable: List[Cmd] = []
    for c in cmds:
        if exemption_for(c):
            continue
        miss = missing_tools(c.script)
        if miss:
            unclassified.append(f"{c.where}: needs {miss} which PATH cannot provide "
                                f"-- UNMEASURABLE, not a pass")
        else:
            runnable.append(c)
    g.check("every_workflow_command_is_verifiable",
            bool(cmds) and not unclassified,
            (f"{wf_dir} parsed {len(cmds)} run-step(s); " if cmds else
             f"NO workflow run-steps parsed under {wf_dir} -- the gate read nothing; ")
            + "; ".join(unclassified[:5]))

    # ---- C1: run every runnable workflow command in a sandbox copy.
    sandbox = tmp / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    n_files = snapshot(root, sandbox)
    bad: List[str] = []
    for c in runnable:
        rc, tail = run_script(c.script, sandbox)
        if rc != 0:
            bad.append(f"{c.where} exit {rc}: `{c.script.strip().splitlines()[0][:70]}` "
                       f"-> {tail}")
    g.check("workflow_commands_exit_zero",
            bool(runnable) and not bad,
            (f"no runnable workflow command found (parsed {len(cmds)}) -- nothing was "
             f"executed, so a green here would be vacuous; " if not runnable else
             f"{len(bad)}/{len(runnable)} workflow command(s) fail on a {n_files}-file "
             f"copy of the worktree: ") + "; ".join(bad[:4]))

    # ---- C2: the same commands under the install set the workflow actually declares.
    clean = tmp / "cleanroom"
    clean.mkdir(parents=True, exist_ok=True)
    snapshot(root, clean)
    _probe_files(clean, clean / ".sentinel")
    py_cmds = [c for c in runnable
               if re.search(r"\b(?:pytest|python3?)\b", c.script)]
    dep_bad: List[str] = []
    unmeasured: List[str] = []
    for c in py_cmds:
        allow: Set[str] = set(ALWAYS_ALLOWED)
        for dist in c.installs:
            allow |= set(IMPORT_NAME.get(dist, (dist.replace("-", "_"),)))
        sentinel = clean / f".sentinel_{c.job}_{c.step}"
        rc, tail = run_script(c.script, clean, {
            "CLEAN_RUNNER_REPO": str(clean.resolve()),
            "CLEAN_RUNNER_ALLOW": ",".join(sorted(allow)),
            "CLEAN_RUNNER_SENTINEL": str(sentinel),
        })
        if not sentinel.exists():
            # The probe never loaded -> this run proves NOTHING.  Reported, never passed.
            unmeasured.append(f"{c.where}: the import probe did not load "
                              f"(sitecustomize not on the interpreter's path) -> "
                              f"UNMEASURABLE")
            continue
        if rc != 0:
            dep_bad.append(f"{c.where} exit {rc} with declared installs "
                           f"{list(c.installs) or '[]'} -> {tail}")
    g.check("workflow_commands_survive_declared_deps",
            bool(py_cmds) and not dep_bad and not unmeasured,
            (f"no python/pytest workflow command found -- nothing measured; "
             if not py_cmds else "") + "; ".join(dep_bad[:3] + unmeasured[:3]))

    # ---- C3: the CI-enforced table vs what workflows run.
    rows, tbl_line = parse_ci_table(root / "CONTRIBUTING.md")
    wf_scopes = [resolve_scope(root, c.script.replace("\n", " ")) for c in cmds]
    unbacked: List[str] = []
    for r in rows:
        tool, want_files, want_flags = resolve_scope(root, r.command)
        hit = None
        for wtool, wfiles, wflags in wf_scopes:
            if wtool != tool:
                continue
            if not want_flags <= wflags:
                continue
            if want_files and not want_files <= wfiles:
                hit = ("narrow", wfiles)
                continue
            hit = ("ok", wfiles)
            break
        if hit and hit[0] == "ok":
            continue
        ex = scope_exemption(tool)
        if hit and hit[0] == "narrow" and ex:
            continue
        if hit and hit[0] == "narrow":
            miss = sorted(want_files - hit[1])
            unbacked.append(
                f"CONTRIBUTING.md:{r.line} '{r.standard}' claims CI runs "
                f"`{r.command}` but the only workflow invocation of {tool} covers "
                f"{len(hit[1])}/{len(want_files)} of its scope; unenforced: "
                f"{miss[:4]}{'...' if len(miss) > 4 else ''}")
        else:
            unbacked.append(
                f"CONTRIBUTING.md:{r.line} '{r.standard}' claims CI runs "
                f"`{r.command}` but no workflow under .github/workflows runs "
                f"{tool}{' with ' + str(sorted(want_flags)) if want_flags else ''}")
    g.check("contributing_ci_enforced_rows_are_backed",
            bool(rows) and not unbacked,
            (f"no 'CI-enforced' table parsed from CONTRIBUTING.md -- the gate read "
             f"nothing; " if not rows else
             f"{len(unbacked)}/{len(rows)} rows under CONTRIBUTING.md:{tbl_line} are not "
             f"enforced by any workflow: ") + "; ".join(unbacked[:4]))
    return g


# --------------------------------------------------------------------------------------
# Selftest
# --------------------------------------------------------------------------------------
def _fixture(tmp: Path, root: Path) -> Path:
    """A writable clone of the repo's tracked files -- mutations go here, never in REPO."""
    dst = tmp / "fixture"
    dst.mkdir(parents=True, exist_ok=True)
    snapshot(root, dst)
    shutil.copytree(root / ".github", dst / ".github", dirs_exist_ok=True)
    # A tiny .git so snapshot() works inside the fixture too.
    subprocess.run(["git", "init", "-q", "-b", "main", str(dst)], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(dst), "add", "-A"], check=True, capture_output=True)
    return dst


def selftest() -> int:
    missed = 0
    tmp = Path(tempfile.mkdtemp(prefix="ci_selftest_"))

    def expect(label: str, gate: Gate, check: str, want_fail: bool) -> None:
        nonlocal missed
        got = dict((n, ok) for n, ok, _ in gate.results).get(check)
        if got is None:
            print(f"  MISSED   {check}   ({label}: check never ran)")
            missed += 1
        elif (not got) == want_fail:
            print(f"  TRIPPED  {check}")
        else:
            print(f"  MISSED   {check}   ({label}: expected "
                  f"{'FAIL' if want_fail else 'PASS'})")
            missed += 1

    base = _fixture(tmp, REPO)

    # --- FAULT 1 (C1): make one workflow command exit non-zero.  The mutation is asserted
    # to have changed the file: a no-op replace against a drifted pattern is exactly how a
    # selftest goes vacuous.
    f1 = tmp / "f1"
    shutil.copytree(base, f1)
    lint = f1 / ".github/workflows/lint.yml"
    before = lint.read_text()
    after = re.sub(r"(?m)^(\s*)- run: ruff check .*$", r"\1- run: exit 3", before)
    assert after != before, "FIXTURE DRIFT: no `- run: ruff check ...` line to mutate"
    lint.write_text(after)
    assert "exit 3" in lint.read_text()
    expect("command forced to exit 3", run(f1, tmp=tmp / "t1"),
           "workflow_commands_exit_zero", True)

    # --- FAULT 2 (C2): NARROW the declared install set -- drop numpy, which
    # gssc/inference/evaluate_bev.py imports at module scope -- and the cleanroom run must
    # go RED.
    #
    # POLARITY. This arm used to assert the LIVE workflow was red, because at the time it
    # was: test.yml declared `pytest pyyaml` and the suite died at collection on a runner
    # without numpy. That made the "fault" arm a restatement of the defect, so the moment
    # the defect was fixed the arm inverted and the selftest reported FIXTURE DRIFT. A
    # selftest has to inject a fault into a HEALTHY fixture; asserting that today's bug is
    # still present is a regression test wearing a selftest's clothes.
    f2 = tmp / "f2"
    shutil.copytree(base, f2)
    test_wf = f2 / ".github/workflows/test.yml"
    b2 = test_wf.read_text()
    a2 = b2.replace('pip install pytest "pyyaml>=6.0" "numpy>=1.26,<2"',
                    'pip install pytest "pyyaml>=6.0"')
    assert a2 != b2, ("FIXTURE DRIFT: test.yml's install step no longer reads "
                      '`pip install pytest "pyyaml>=6.0" "numpy>=1.26,<2"`')
    test_wf.write_text(a2)
    expect("numpy dropped from the declared installs", run(f2, tmp=tmp / "t2"),
           "workflow_commands_survive_declared_deps", True)
    # ...and the control: UNCHANGED, the same check must be GREEN. Without it, a probe
    # broken in a way that reddens everything would satisfy the fault arm above and look
    # like a working selftest.
    g2 = run(base, tmp=tmp / "t2b")
    got2 = dict((n, ok) for n, ok, _ in g2.results)["workflow_commands_survive_declared_deps"]
    if got2:
        print("  TRIPPED  control_declared_deps_green_when_deps_declared")
    else:
        det = dict((n, d) for n, ok, d in g2.results if not ok).get(
            "workflow_commands_survive_declared_deps", "")
        print(f"  MISSED   control_declared_deps_green_when_deps_declared   ({det[:200]})")
        missed += 1

    # --- FAULT 3 (C3): add a CI-enforced row no workflow backs.
    f3 = tmp / "f3"
    shutil.copytree(base, f3)
    contrib = f3 / "CONTRIBUTING.md"
    b3 = contrib.read_text()
    a3 = b3.replace("| Style + import order |",
                    "| Dead code | `vulture` | `vulture src/` | all |\n"
                    "| Style + import order |", 1)
    assert a3 != b3, "FIXTURE DRIFT: the CI-enforced table's first row moved"
    contrib.write_text(a3)
    g3 = run(f3, tmp=tmp / "t3")
    det3 = dict((n, d) for n, ok, d in g3.results
                if not ok).get("contributing_ci_enforced_rows_are_backed", "")
    expect("unbacked row injected", g3, "contributing_ci_enforced_rows_are_backed", True)
    if "vulture" in det3:
        print("  TRIPPED  ci_table_detail_names_the_injected_row")
    else:
        print(f"  MISSED   ci_table_detail_names_the_injected_row   ({det3[:160]})")
        missed += 1

    # --- FAULT 4 (C4): a workflow command whose tool does not exist must be reported as
    # UNMEASURABLE, never executed-and-passed and never silently skipped.
    f4 = tmp / "f4"
    shutil.copytree(base, f4)
    l4 = f4 / ".github/workflows/lint.yml"
    b4 = l4.read_text()
    a4 = re.sub(r"(?m)^(\s*)- run: ruff check .*$",
                r"\1- run: definitely_not_installed_xyz check src/", b4)
    assert a4 != b4, "FIXTURE DRIFT: no ruff run-line to replace"
    l4.write_text(a4)
    expect("unknown tool", run(f4, tmp=tmp / "t4"),
           "every_workflow_command_is_verifiable", True)

    # --- FAULT 5: an empty workflow directory must FAIL, not pass vacuously.  Two gates
    # in this project's history shipped green because their input parsed to nothing.
    f5 = tmp / "f5"
    shutil.copytree(base, f5)
    shutil.rmtree(f5 / ".github/workflows")
    (f5 / ".github/workflows").mkdir(parents=True)
    g5 = run(f5, tmp=tmp / "t5")
    for chk in ("every_workflow_command_is_verifiable", "workflow_commands_exit_zero",
                "workflow_commands_survive_declared_deps"):
        expect("no workflows at all", g5, chk, True)

    # --- FAULT 6: scope exemptions must be SCOPED.  The mypy narrowing must not excuse a
    # narrowed pytest, which is defect D3's central row.
    if scope_exemption("mypy") and not scope_exemption("pytest"):
        print("  TRIPPED  scope_exemption_does_not_leak_to_pytest")
    else:
        print("  MISSED   scope_exemption_does_not_leak_to_pytest")
        missed += 1

    shutil.rmtree(tmp, ignore_errors=True)
    total = 10
    print(f"SELFTEST OK: {total - missed}/{total} checks provably fail when broken"
          if not missed else
          f"SELFTEST FAILED: {total - missed}/{total} checks provably fail when broken")
    return 1 if missed else 0


def main() -> int:
    print(f"[check_ci_honesty] repo={REPO}")
    tmp = Path(tempfile.mkdtemp(prefix="ci_honesty_"))
    try:
        return run(REPO, tmp=tmp).report()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(selftest() if "--selftest" in sys.argv else main())

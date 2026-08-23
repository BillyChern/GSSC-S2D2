#!/usr/bin/env python3
"""GATE: every `download_assets.py` mode must fail into the DOCUMENTED pointer, not a traceback.

THE DEFECT THIS GATE EXISTS FOR
-------------------------------
LINE NUMBERS IN THIS BLOCK ARE A DATED SNAPSHOT of the checkout named above, not
navigation. Several have already moved: follow the SYMBOL, the heading or the quoted
text, and re-derive the location with `grep -n`. Every check below RE-MEASURES the
live artefacts, so nothing here is load-bearing for a verdict.

`scripts/download_assets.py:_ensure_url_configured` guards on ONE shape of unavailability:

    if url.startswith("[") and url.endswith("]"):        # a `[PLACEHOLDER]` token
        sys.exit("... - Manual instructions: docs/DATASET.md ...")

Its own docstring states the intent: "Direct visitors at the manual-download docs rather than
failing inside huggingface_hub with a confusing 'Repository not found'." But only
`DATAPORT_URL` is a placeholder. `HF_REPO_MODELS` and `HF_REPO_DATA` are real-LOOKING repo ids
(`Stone-Chern/GSSC-S2D2-checkpoints`), so the guard passes them through and the five Hugging
Face modes reach `snapshot_download`, where a repo that does not exist yet raises
`RepositoryNotFoundError` and Python prints a bare traceback. Measured on 2026-08-20 with the
network boundary neutralised: `--checkpoints` exits 1 with

    Traceback (most recent call last): ... TypeError/RepositoryNotFoundError

and the string `docs/DATASET.md` NOWHERE in the output. So the guard's stated purpose fails for
exactly the modes a first-time user runs first, and the ONE mode it does cover
(`--synthetic-pool`) is the one nobody starts with.

Three documents promise the behaviour the code does not have:

  README.md:37                  "`scripts/download_assets.py` provisions them, and
                                 `docs/DATASET.md` documents how to regenerate ..."
  docs/MODEL_ZOO.md:4-6         "... is released upon paper publication; until then the download
                                 command EXITS WITH the manual-download instructions in
                                 `docs/DATASET.md`."
  examples/quickstart.ipynb     "`scripts/download_assets.py` FAILS LOUDLY with that message
                                 until then"; "the cell below will print a `docs/DATASET.md`
                                 pointer"

A reviewer who clones and runs the first command in the README gets a stack trace, and the doc
that told them what would happen is wrong.

HOW THIS IS MEASURED, AND THE THREE THINGS THAT COULD HAVE GONE WRONG
--------------------------------------------------------------------
1. NEVER RUN THE REAL SCRIPT IN PLACE. The script's default `--root` is `<repo>/data`, and it
   calls `root.mkdir(parents=True)` before downloading. The probe runs a COPY in $TMPDIR with
   `--root` pointing at a $TMPDIR sandbox, and `repo_untouched` compares a before/after
   snapshot of the repository so a future edit to the default cannot quietly make this gate
   write into the artefact it is auditing.

2. NEVER TOUCH THE NETWORK. `snapshot_download` is monkeypatched in the child process. Two
   patch modes exist: `raise` (the sentinel `RepositoryNotFoundError`, which is what a
   not-yet-created repo does) and `record` (log the call and return), used only by the positive
   control below.

3. PROVE THE PATCH IS ON THE PATH. A neutralised probe that never reaches the patched function
   would "measure" whatever the script does for an unrelated reason -- a missing import, a
   typo -- and a gate keyed on the exit code alone would report the right verdict for the wrong
   reason, then invert silently once that unrelated fault was fixed. `patch_reaches_hub` is a
   separate `record`-mode run asserting the patched `snapshot_download` was actually CALLED.
   It is deliberately not keyed on the sentinel appearing in the output, because a correct fix
   will swallow that message -- a control that a successful fix breaks is not a control.

THE MODE LIST IS PARSED, NOT TYPED. Modes come from the script's own `argparse` calls via
`ast`, so a mode added during the release fixes is probed automatically. `modes_parsed` fails
loud if that parse finds nothing.

DIRECTION OF THE FIX. Broaden the guard (or wrap the `snapshot_download` calls) so that ANY
unreachable repo exits with the docs/DATASET.md pointer. Do NOT "fix" this by softening the
three documents: the documents describe the behaviour a user should get, and
`docs/MODEL_ZOO.md:6` is the specification.

STATUS. The fix above LANDED: `scripts/download_assets.py:_fetch` now wraps every
`snapshot_download` in a broad handler that `sys.exit`s with `_MANUAL_ROUTES`, so all 9
checks pass on the shipped artefacts (re-measured 2026-08-20 after the fix). The gate is
kept as a regression guard: it fails again the moment any mode loses that pointer.

WHERE THE SELFTEST ROTTED. Until 2026-08-20 the selftest built its healthy fixture by
APPENDING a simulated release guard (`_GUARD`) to the still-broken shipped script, and
injected its faults into that appended block -- and, for the three promise checks, used
"measure the shipped script" AS the fault. When the real fix landed inside `_fetch`, the
appended block became dead code and the shipped script became healthy, so five of nine
faults stopped reaching anything the gate measures while still reading like injections.
Every fault now disables BOTH guards -- the one that ships in `_fetch` and the appended
one -- and is asserted to actually perturb the probe before its check is graded.

ROOTS, AND WHAT IS NOT PART OF THE PUBLIC RELEASE
-------------------------------------------------
Every root below is an environment variable with a repo-relative default, so this gate
measures the checkout it ships in rather than one particular machine.  Absolute paths
were hardcoded here once; a relocated clone then audited a tree it was not running in,
and the paths themselves disclosed the maintainer's local layout to every visitor.

    GSSC_REPO        the release checkout under test        default: this file's repository
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

REPO = Path(os.environ.get("GSSC_REPO") or Path(__file__).resolve().parents[1])
DOWNLOADER = REPO / "scripts" / "download_assets.py"
POINTER = "docs/DATASET.md"

#: Documents that promise how the downloader behaves when the assets are not reachable.
PROMISE_DOCS = ("README.md", "docs/MODEL_ZOO.md", "examples/quickstart.ipynb")

#: A sentence promises the GRACEFUL-POINTER behaviour when it names the downloader (or the
#: asset URLs) AND a failure/print verb AND the pointer document. Keyed on the STRUCTURE rather
#: than on any one phrase: "exits with", "fails loudly" and "will print" are three different
#: wordings of one promise across the three documents, and a phrase list would have caught one.
SUBJECT = re.compile(r"download_assets|asset URLs|download command", re.I)
FAIL_VERB = re.compile(r"\bexits?\s+with\b|\bfails?\s+loudly\b|\bwill\s+print\b|\bprints?\b|"
                       r"\bexits?\s+with\s+the\b|\berror\b", re.I)

#: Scratch root. Deliberately NOT falling back to /tmp: a full /tmp has deadlocked the
#: maintainer's box five times, and a gate that hangs the machine is worse than a gate that
#: refuses to run. TMPDIR overrides it. The default is a NAMED CACHE DIR rather than nothing,
#: so a visitor who has exported no variables still gets a runnable gate that is not pointed
#: at /tmp; the refusal below survives only for a host with no home directory at all.
def _scratch_root() -> Optional[Path]:
    override = os.environ.get("TMPDIR")
    if override:
        return Path(override)
    try:
        return Path.home() / ".cache" / "gssc-release-checks"
    except RuntimeError:                      # no resolvable home on this host
        return None


TMP = _scratch_root()

_RUNNER = r'''
import json, os, runpy, sys

MODE = os.environ["PROBE_PATCH_MODE"]
CALLS = os.environ["PROBE_CALL_LOG"]
SENTINEL = "PROBE_SENTINEL_REPO_NOT_FOUND"

import huggingface_hub


def _sentinel_error():
    """Build the exception a not-yet-created HF repo really raises.

    The constructor is version-dependent (huggingface_hub 1.x requires a keyword-only httpx
    response). Falling back through the ladder keeps the probe honest across versions; the last
    rung is still a RepositoryNotFoundError INSTANCE, never a stand-in Exception, because the
    script may one day catch the specific class.
    """
    from huggingface_hub.utils import RepositoryNotFoundError
    msg = "404 Client Error. Repository Not Found for url: " + SENTINEL
    try:
        import httpx
        resp = httpx.Response(404, headers={}, text="Repository Not Found",
                              request=httpx.Request("GET", "https://huggingface.co/api/x"))
        return RepositoryNotFoundError(msg, response=resp)
    except Exception:
        pass
    try:
        err = RepositoryNotFoundError.__new__(RepositoryNotFoundError)
        BaseException.__init__(err, msg)
        return err
    except Exception:
        return OSError(msg)


def _patch(*args, **kwargs):
    with open(CALLS, "a") as fh:
        fh.write(json.dumps({"repo_id": kwargs.get("repo_id"),
                             "allow_patterns": kwargs.get("allow_patterns")}) + "\n")
    if MODE == "raise":
        raise _sentinel_error()
    return str(kwargs.get("local_dir") or ".")


huggingface_hub.snapshot_download = _patch
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
'''


class Run(NamedTuple):
    mode: Tuple[str, ...]
    rc: int
    out: str
    calls: Tuple[str, ...]

    @property
    def label(self) -> str:
        return " ".join(self.mode)


# --------------------------------------------------------------------------- probing


def parse_modes(src: str) -> List[Tuple[str, ...]]:
    """Every asset-selecting CLI mode, read off the script's own argparse calls.

    `--root` is not a mode (it has a default and no store_true); the `choices=` argument becomes
    one mode using its first choice.
    """
    modes: List[Tuple[str, ...]] = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "add_argument"):
            continue
        flags = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)
                 and a.value.startswith("--")]
        if not flags:
            continue
        kw = {k.arg: k.value for k in node.keywords}
        act = kw.get("action")
        if isinstance(act, ast.Constant) and act.value == "store_true":
            modes.append((flags[0],))
        elif isinstance(kw.get("choices"), (ast.List, ast.Tuple)):
            elts = [e.value for e in kw["choices"].elts if isinstance(e, ast.Constant)]
            if elts:
                modes.append((flags[0], str(elts[0])))
    return modes


def probe(script: Path, modes: Sequence[Tuple[str, ...]], patch_mode: str,
          sandbox: Path) -> List[Run]:
    """Run each mode in a child process with the hub neutralised. Never writes outside sandbox."""
    runner = sandbox / "_runner.py"
    runner.write_text(_RUNNER, encoding="utf-8")
    out: List[Run] = []
    for i, mode in enumerate(modes):
        log = sandbox / f"calls_{i}.jsonl"
        env = dict(os.environ,
                   PROBE_PATCH_MODE=patch_mode,
                   PROBE_CALL_LOG=str(log),
                   HF_HUB_OFFLINE="1",          # belt and braces: no egress even if unpatched
                   HF_HUB_DISABLE_TELEMETRY="1")
        root = sandbox / f"root_{i}"
        proc = subprocess.run(
            [sys.executable, str(runner), str(script), *mode, "--root", str(root)],
            capture_output=True, text=True, env=env, timeout=180)
        calls = tuple(log.read_text().splitlines()) if log.exists() else ()
        out.append(Run(tuple(mode), proc.returncode, proc.stdout + proc.stderr, calls))
    return out


def repo_snapshot(root: Path) -> Tuple[str, Tuple[str, ...]]:
    """Enough state to prove the probe did not write into the repository."""
    git = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                         capture_output=True, text=True)
    data = root / "data"
    listing = tuple(sorted(str(p.relative_to(root)) for p in data.rglob("*"))) \
        if data.is_dir() else ()
    return git.stdout, listing


# --------------------------------------------------------------------------- pure checks

Verdict = Tuple[bool, str]


def graceful(r: Run) -> bool:
    return r.rc != 0 and POINTER in r.out and "Traceback (most recent call last)" not in r.out


def sentences(text: str, join: bool = True) -> List[Tuple[int, str]]:
    """(first line of the paragraph, sentence) over a hard-wrapped markdown / notebook doc.

    PARAGRAPH-JOINING IS LOAD-BEARING. A per-LINE scan missed docs/MODEL_ZOO.md:4-6 entirely --
    the promise there is hard-wrapped across three lines, so no single line carried the subject,
    the verb and the pointer at once, and the gate reported the doc as making no promise. That
    false null is the reason this helper exists; the first version of this gate shipped with it.

    `.ipynb` cells are single JSON strings carrying escaped newlines, so those are split too or
    one cell would match a subject and a verb drawn from different paragraphs.
    """
    out: List[Tuple[int, str]] = []

    def emit(lines: "List[Tuple[int, str]]") -> None:
        """Sentence-split a paragraph and attribute each sentence to the line it STARTS on.

        Attributing every sentence to the paragraph's first line pointed a reader at
        `README.md:36`, which is the `> [!NOTE]` marker, not the claim. A gate's detail string
        is an instruction to go and look; sending someone one line off is a small lie.
        """
        start = lines[0][0]
        blob = " ".join(t for _n, t in lines)
        for chunk in blob.split("\\n"):          # escaped newlines in a notebook cell
            for piece in re.split(r"(?<=[.;:])\s+", chunk):
                piece = " ".join(piece.split())
                if not piece:
                    continue
                head = piece[:25]
                where = next((n for n, t in lines if head in " ".join(t.split())), start)
                out.append((where, piece))

    if not join:
        # JSON (.ipynb): one physical line per cell, so joining would merge unrelated cells and
        # let a subject from one match a verb from another. Line numbers stay exact this way.
        for n, line in enumerate(text.splitlines(), 1):
            emit([(n, line)])
        return out

    para: List[Tuple[int, str]] = []
    for n, line in enumerate(text.splitlines() + [""], 1):
        if line.strip():
            para.append((n, line))
        else:
            if para:
                emit(para)
            para = []
    return out


def promise_sites(text: str, join: bool = True) -> List[Tuple[int, str]]:
    """(line, sentence) for every GRACEFUL-POINTER promise: subject + failure verb + pointer."""
    return [(n, s[:200]) for n, s in sentences(text, join)
            if SUBJECT.search(s) and FAIL_VERB.search(s) and POINTER in s]


def downloader_sites(text: str, join: bool = True) -> List[Tuple[int, str]]:
    """(line, sentence) for every sentence that tells the reader what the downloader does."""
    return [(n, s[:200]) for n, s in sentences(text, join) if SUBJECT.search(s)]


def _join(doc: str) -> bool:
    """Hard-wrapped prose is joined into paragraphs; JSON notebooks are not."""
    return not doc.endswith((".ipynb", ".json"))


def evaluate(modes: Sequence[Tuple[str, ...]], runs: Sequence[Run],
             record_runs: Sequence[Run], before, after,
             docs: Dict[str, str]) -> "Dict[str, Verdict]":
    res: Dict[str, Verdict] = {}
    dlf = DOWNLOADER.relative_to(REPO)

    res["modes_parsed"] = (
        len(modes) >= 2 and len(runs) == len(modes),
        f"{dlf}: parsed {len(modes)} CLI mode(s) from argparse -- the probe has nothing to run, "
        f"so nothing was measured",
    )
    reached = [r.label for r in record_runs if r.calls]
    res["patch_reaches_hub"] = (
        bool(reached),
        f"the patched snapshot_download was never called by any mode "
        f"({[r.label for r in record_runs]}), so the raise-mode verdicts below describe some "
        f"OTHER failure and cannot be trusted",
    )

    bad_rc = [r.label for r in runs if r.rc == 0]
    res["every_mode_exits_nonzero"] = (
        not bad_rc,
        f"{dlf}: mode(s) {bad_rc} exit 0 with no asset provisioned -- a silent success is worse "
        f"than the traceback",
    )
    no_ptr = [r.label for r in runs if POINTER not in r.out]
    res["every_mode_names_dataset_md"] = (
        not no_ptr,
        f"{dlf}:37 _ensure_url_configured only bails on `[PLACEHOLDER]` values, so mode(s) "
        f"{no_ptr} die without ever naming {POINTER}",
    )
    tb = [r.label for r in runs if "Traceback (most recent call last)" in r.out]
    res["no_mode_emits_bare_traceback"] = (
        not tb,
        f"{dlf}: mode(s) {tb} print a Python traceback at the user "
        f"({[l for l in runs[0].out.splitlines() if 'Error' in l][:1]})",
    )
    res["repo_untouched"] = (
        before == after,
        f"the probe modified {REPO}: git status and/or data/ listing changed "
        f"({len(set(after[1]) - set(before[1]))} new path(s) under data/) -- fix the probe, not "
        f"the repo",
    )

    # A document is judged against the SPECIFICATION its siblings state. Two shapes of defect:
    #   (a) it promises the graceful pointer and the code does not deliver it;
    #   (b) it describes running the downloader but is SILENT about the failure path that a
    #       sibling document specifies, while the code delivers no such path either -- which is
    #       README.md:37 exactly ("provisions them", no caveat, traceback in practice).
    # Shape (b) is scoped to "while the code is broken": once every mode is graceful, a doc that
    # only documents the happy path is no longer misleading, so both shapes clear together and
    # neither can rot into a permanent failure.
    offenders = [r.label for r in runs if not graceful(r)]
    spec = [(d, n, s) for d in PROMISE_DOCS
            for n, s in promise_sites(docs.get(d, ""), _join(d))]
    for doc in PROMISE_DOCS:
        key = "promise_" + re.sub(r"\W+", "_", Path(doc).stem).lower() + "_matches_code"
        sites = promise_sites(docs.get(doc, ""), _join(doc))
        mentions = downloader_sites(docs.get(doc, ""), _join(doc))
        if sites:
            res[key] = (
                not offenders,
                "; ".join(f'{doc}:{n} promises "{s}" but mode(s) {offenders} do not deliver it'
                          for n, s in sites[:2]),
            )
        else:
            elsewhere = [(d, n) for d, n, _ in spec if d != doc]
            res[key] = (
                not (mentions and offenders and elsewhere),
                (f'{doc}:{mentions[0][0]} describes only the success path ("{mentions[0][1]}") '
                 f'while {elsewhere[0][0]}:{elsewhere[0][1]} specifies a {POINTER} pointer, and '
                 f'mode(s) {offenders} deliver neither') if mentions and elsewhere else "",
            )
    return res


ORDER = ("modes_parsed", "patch_reaches_hub", "every_mode_exits_nonzero",
         "every_mode_names_dataset_md", "no_mode_emits_bare_traceback", "repo_untouched",
         "promise_readme_matches_code", "promise_model_zoo_matches_code",
         "promise_quickstart_matches_code")


def report(res: "Dict[str, Verdict]") -> int:
    bad = 0
    for name in ORDER:
        ok, detail = res[name]
        if ok:
            print(f"  PASS  {name}")
        else:
            bad += 1
            print(f"  FAIL  {name}   ({detail})")
    print("OK: 0 failing check(s)" if not bad else f"FAILED: {bad} failing check(s)")
    return 1 if bad else 0


def read_docs() -> Dict[str, str]:
    return {d: (REPO / d).read_text(encoding="utf-8", errors="replace")
            for d in PROMISE_DOCS if (REPO / d).is_file()}


def measure(script: Path, sandbox: Path, docs: Dict[str, str],
            modes_override=None) -> "Dict[str, Verdict]":
    src = script.read_text(encoding="utf-8")
    modes = modes_override if modes_override is not None else parse_modes(src)
    before = repo_snapshot(REPO)
    runs = probe(script, modes, "raise", sandbox / "raise")
    rec = probe(script, modes, "record", sandbox / "record")
    after = repo_snapshot(REPO)
    return evaluate(modes, runs, rec, before, after, docs)


def _sandbox(name: str) -> Path:
    if TMP is None:
        raise SystemExit("FATAL: no scratch root. This probe runs a copy of the downloader "
                         "and must not scratch in /tmp (a full /tmp has deadlocked this box "
                         "repeatedly). Export TMPDIR to a writable directory and re-run.")
    d = TMP / "check_download_guard" / name
    if d.exists():
        shutil.rmtree(d)
    for sub in ("raise", "record"):
        (d / sub).mkdir(parents=True)
    return d


# --------------------------------------------------------------------------- selftest


def _mutate(text: str, pattern: str, repl: str, label: str, count: int = 1) -> str:
    new, n = re.subn(pattern, repl, text, count=count, flags=re.M)
    assert n >= 1 and new != text, f"fault '{label}' did not perturb the input (pattern drifted)"
    return new


#: The guard the release fix is expected to install, appended to the REAL script so the
#: fixture cannot drift away from the shipped source.
# NO BACKSLASHES BELOW, DELIBERATELY. This block is injected with `re.sub`, whose replacement
# string interprets escapes: a `\\n` here became a REAL newline inside the generated string
# literal and every selftest case died on `SyntaxError: unterminated string literal` while the
# checks all reported MISSED -- an instrument failure that reads exactly like a gate that cannot
# detect anything. Keep the message a triple-quoted literal.
_GUARD = '''

_GUARD_MSG = """
Asset repository is not reachable yet.
  - Manual instructions:    docs/DATASET.md
"""

_unguarded_main = main


def main() -> None:              # release guard under test by check_download_guard.py
    try:
        _unguarded_main()
    except SystemExit:
        raise
    except BaseException:
        raise SystemExit(_GUARD_MSG)


'''


def _untrapped(src: str) -> str:
    """Fixture whose Hugging Face modes die inside the hub, exactly as the pre-fix script did.

    TWO mutations because there are now TWO guards on that path, and faulting either one alone
    changes nothing observable:

      * the SHIPPED guard -- `_fetch`'s `try:` around `snapshot_download` -- is hoisted off the
        call, so the sentinel escapes `_fetch` instead of becoming a `sys.exit` pointer;
      * the APPENDED `_GUARD` wrapper is made to re-raise instead of exiting.

    `--synthetic-pool` is untouched and stays graceful (it never reaches the hub), which is
    faithful to the original defect: the modes a first-time user runs are the broken ones.
    """
    src = _mutate(src,
                  r"^    try:\n        snapshot_download\(repo_id=repo_id, \*\*kwargs\)$",
                  "    snapshot_download(repo_id=repo_id, **kwargs)\n    try:\n        pass",
                  "tb-inner")
    return _mutate(src, r"raise SystemExit\(_GUARD_MSG\)", "raise", "tb-outer")


def selftest() -> int:
    real = DOWNLOADER.read_text(encoding="utf-8")
    docs = read_docs()

    # Probe only two modes in the selftest: one that reaches the hub and one short-circuited by
    # the placeholder guard. Enough to exercise every check, ~10x cheaper than all seven.
    all_modes = parse_modes(real)
    modes = [m for m in all_modes if m[0] == "--checkpoints"] + \
            [m for m in all_modes if len(m) > 1][:1]
    assert len(modes) == 2, f"selftest mode selection drifted: {all_modes}"

    fixed_src = _mutate(real, r'^if __name__ == "__main__":$',
                        _GUARD.strip("\n") + '\n\nif __name__ == "__main__":', "guard")
    base_dir = _sandbox("base")
    fixed = base_dir / "download_assets.py"
    fixed.write_text(fixed_src, encoding="utf-8")

    base = measure(fixed, base_dir, docs, modes)
    missed: List[str] = []
    pre_bad = [n for n in ORDER if not base[n][0]]
    for n in pre_bad:
        print(f"  MISSED   {n}   (fails on the REPAIRED fixture: {base[n][1]})")
    missed.extend(pre_bad)

    # ONE non-graceful fixture, shared by the three promise checks. Probed here and re-fed to
    # `evaluate` per document below: one probe pair instead of three, and it lets each document
    # be ISOLATED (see the promise branch). The assert is the point of the rewrite -- the old
    # promise fault was "measure the shipped script", which silently became a no-op the day the
    # shipped script was fixed. A fixture that is still graceful cannot fault anything, and this
    # says so loudly instead of grading three vacuous arms.
    prom_dir = _sandbox("promise")
    prom = prom_dir / "download_assets.py"
    prom.write_text(_untrapped(fixed_src), encoding="utf-8")
    p_before = repo_snapshot(REPO)
    p_runs = probe(prom, modes, "raise", prom_dir / "raise")
    p_rec = probe(prom, modes, "record", prom_dir / "record")
    p_after = repo_snapshot(REPO)
    assert [r.label for r in p_runs if not graceful(r)], (
        "promise fixture is still graceful on every mode, so the promise checks would be asked "
        "to detect a defect that is not there -- the fault stopped reaching the code path")

    # Each fault perturbs a real input: the script source, the probe's patch mode, or the
    # snapshot pair. Never a check's own return value.
    #
    # EVERY SOURCE FAULT MUST DISABLE BOTH GUARDS. The graceful path is two-deep now: the
    # `try/except BaseException` that ships inside `_fetch`, and the `_GUARD` wrapper this
    # selftest appends. `every_mode_exits_nonzero` and `no_mode_emits_bare_traceback` used to
    # mutate the wrapper ALONE; once the real fix landed in `_fetch` the wrapper was dead code,
    # the mutated fixture behaved identically to the healthy one (measured: rc=1, pointer
    # present, no traceback -- on both), and both arms reported MISSED forever.
    src_faults = {
        "every_mode_exits_nonzero":
            # Warn-and-continue: the ordinary way a failure path decays. Both the placeholder
            # bail and the `_fetch` handler log instead of exiting, and the wrapper exits 0, so
            # the pointer is still printed and nothing is raised -- ONLY the exit code moves,
            # which keeps this arm from passing on some neighbouring check's fault.
            lambda s: _mutate(
                _mutate(s, r"sys\.exit\((?!0\))", "logger.error(", "rc-inner", count=0),
                r"raise SystemExit\(_GUARD_MSG\)", "raise SystemExit(0)", "rc-outer"),
        "every_mode_names_dataset_md":
            lambda s: _mutate(s, re.escape(POINTER), "docs/ELSEWHERE.md", "ptr", count=0),
        "no_mode_emits_bare_traceback":
            _untrapped,
        "modes_parsed":
            lambda s: _mutate(s, r'action="store_true"', 'action="count"', "modes", count=0),
    }
    for name in ORDER:
        if name in pre_bad:
            continue
        if name in src_faults:
            d = _sandbox(name)
            broken = d / "download_assets.py"
            broken.write_text(src_faults[name](fixed_src), encoding="utf-8")
            got = measure(broken, d, docs, modes if name != "modes_parsed" else parse_modes(
                src_faults[name](fixed_src)))
        elif name == "patch_reaches_hub":
            # Fault the CONTROL itself: a record run that never calls the patch.
            d = _sandbox(name)
            src = _mutate(fixed_src, r"^\s+snapshot_download\(repo_id=.*$",
                          "        pass  # patched out", "nocall", count=0)
            p = d / "download_assets.py"
            p.write_text(src, encoding="utf-8")
            got = measure(p, d, docs, modes)
        elif name == "repo_untouched":
            # Pure function of two snapshots; perturb the snapshot, never the real repo.
            d = _sandbox(name)
            src2 = fixed_src
            (d / "download_assets.py").write_text(src2, encoding="utf-8")
            before = repo_snapshot(REPO)
            runs = probe(d / "download_assets.py", modes, "raise", d / "raise")
            rec = probe(d / "download_assets.py", modes, "record", d / "record")
            dirty = (before[0], before[1] + ("data/checkpoints/leaked.safetensors",))
            assert dirty != before, "snapshot fault did not perturb the input"
            got = evaluate(modes, runs, rec, before, dirty, docs)
        else:                                   # the three promise checks
            doc = {"promise_readme_matches_code": "README.md",
                   "promise_model_zoo_matches_code": "docs/MODEL_ZOO.md",
                   "promise_quickstart_matches_code": "examples/quickstart.ipynb"}[name]
            # Fault = the non-graceful fixture above (code that does not deliver the pointer)
            # judged against the REAL documents. Both halves are real inputs; nothing is
            # doctored. This replaces "measure the shipped script", which stopped being a fault
            # the moment the shipped script started delivering the pointer.
            #
            # And the documents are ISOLATED, because an arm named for one document must be
            # shown to READ that document: every OTHER doc is blanked except the one sibling
            # that supplies the specification a shape-(b) arm needs (`keep_spec`). Without
            # isolation all three arms would trip off `offenders` alone and a cross-wired arm
            # -- README's check keyed on MODEL_ZOO's text -- would pass unnoticed.
            keep_spec = next((o for o in PROMISE_DOCS
                              if o != doc and promise_sites(docs.get(o, ""), _join(o))), None)
            iso = {o: (t if o in (doc, keep_spec) else "") for o, t in docs.items()}
            # A promise check is LIVE if the doc either promises the pointer itself (shape a)
            # or merely describes the downloader while a sibling doc specifies the pointer
            # (shape b -- README.md). Requiring shape (a) here reported README as vacuous on
            # first run, which was the selftest mis-modelling the check, not the check failing.
            # Evaluated on `iso`, i.e. on exactly what `evaluate` is about to see.
            text = iso.get(doc, "")
            sib = any(promise_sites(iso.get(o, ""), _join(o))
                      for o in PROMISE_DOCS if o != doc)
            live = bool(promise_sites(text, _join(doc))) or \
                (bool(downloader_sites(text, _join(doc))) and sib)
            if not live:
                print(f"  MISSED   {name}   (no promise and no sibling spec for {doc}: the "
                      f"check is vacuous -- re-point it or delete it)")
                missed.append(name)
                continue
            got = evaluate(modes, p_runs, p_rec, p_before, p_after, iso)

        if got[name][0]:
            missed.append(name)
            print(f"  MISSED   {name}")
        else:
            print(f"  TRIPPED  {name}")

    # The label must agree with the exit code. This line used to print "SELFTEST OK" and then
    # return 1, so a run with five dead arms announced OK -- the one word a reader skims for.
    n = len(ORDER)
    if missed:
        print(f"SELFTEST FAILED: {n - len(missed)}/{n} checks provably fail when broken; "
              f"missed: {', '.join(missed)}")
        return 1
    print(f"SELFTEST OK: {n}/{n} checks provably fail when broken")
    return 0


def main() -> int:
    if not DOWNLOADER.is_file():
        print(f"  FAIL  artefact_present   ({DOWNLOADER} missing)")
        print("FAILED: 1 failing check(s)")
        return 1
    if "--selftest" in sys.argv:
        return selftest()
    sandbox = _sandbox("live")
    # A COPY, always: the script mkdir()s its --root, and its default root is <repo>/data.
    copy = sandbox / "download_assets.py"
    shutil.copyfile(DOWNLOADER, copy)
    return report(measure(copy, sandbox, read_docs()))


if __name__ == "__main__":
    raise SystemExit(main())

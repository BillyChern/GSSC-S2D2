#!/usr/bin/env python3
"""GATE: what the paper's pinned tag ACTUALLY contains, versus what HEAD contains.

THE LESSON THIS GATE ENCODES -- THE ONE THAT COST THE MOST
-----------------------------------------------------------
The paper pins a release tag.  The sentence is in the supplement's reproducibility appendix
-- find it with `grep -n 'machine-readable Hydra configs' supplementary.tex`, not by line
number: it read `:1525` when this was written and the paper has reflowed since.  It reads:

    "The code repository tagged \\texttt{v2.3.8} contains the same values in
     machine-readable Hydra configs."

A reviewer follows that pointer and gets `git checkout v2.3.8` -- a FROZEN TREE.  They do
not get main, they do not get the worktree, and they never see an edit made after the tag
was cut.  Every other gate in this project reads the worktree.  So every fix applied to the
worktree made the MEASUREMENT green while leaving the ARTEFACT a reviewer receives exactly
as broken as it was.  Fixing the worktree fixed the instrument, not the release.

Measured 2026-08-20 at HEAD 07725af: `git diff --name-only v2.3.8..HEAD` = 4 files, of
which ONE is release-critical -- `configs/eval/round2_a.yaml`.  A reviewer checking out
v2.3.8 gets the OLD round2_a.yaml.  Three more release fixes land in the worktree every
hour of a fix cycle; each one widens this gap silently.

WHAT THIS GATE DOES *NOT* CLAIM
-------------------------------
It does not say the tag is wrong or that HEAD is wrong.  It says they DIFFER, and names
every file, so the author decides consciously between the two only real remedies:
  (a) move the pointer -- cut a new tag and update the paper's \\texttt{v...}; or
  (b) move the tag -- re-tag v2.3.8 onto the fixed tree and force-push.
Editing this gate is not a third remedy.

THE POINTER IS READ FROM THE PAPER, NEVER HARDCODED
---------------------------------------------------
Hardcoding "v2.3.8" would make this gate agree with itself instead of with the paper: if
the author bumps the paper to v2.3.9 and forgets to cut the tag, a hardcoded gate stays
green on the wrong tag.  So the tag is PARSED out of `supplementary.tex`, and:
  * ZERO pointers  -> FAIL (unmeasurable, not a pass).
  * TWO OR MORE DISTINCT pointers -> FAIL: the paper cannot pin two trees, and a gate that
    silently picks the first would launder a real contradiction into a green.
  * a pointer naming a tag git does not have -> FAIL: the reviewer's checkout fails.
`main.tex` is scanned as well; a version pointer there that disagrees with the supplement
is the same contradiction and fails the same check.  The paper is READ-ONLY here -- this
gate never writes to the paper checkout.

RELEASE-CRITICAL SET, AND WHY IT IS GLOB-CHECKED
------------------------------------------------
The set is the prose+config surface a reviewer reads or runs: README, every docs/*.md,
CITATION.cff, SECURITY.md, CONTRIBUTING.md, scripts/download_assets.py, and every
configs/eval/*.yaml (the reproduction matrix's entry points).  Source under src/ is
deliberately EXCLUDED: a code change between tag and HEAD is a normal development fact,
while a doc or an eval config that differs means the reviewer runs a different experiment
than the one the paper describes.

Each pattern is expanded on BOTH sides -- the tag's tree and the worktree -- and their
UNION is compared.  Expanding only the worktree would miss a file DELETED since the tag;
expanding only the tag would miss one ADDED since.  A pattern that matches nothing on
either side FAILS the instrument-control check, because a glob that has drifted (docs/ got
renamed) is how this gate would go quietly vacuous.

WHY WORKTREE AND NOT `git show HEAD:`
--------------------------------------
The comparison is tag-vs-WORKTREE, because during a fix cycle the author's changes are
uncommitted and those are precisely the changes that must eventually reach the tag.  Files
whose difference is still uncommitted are annotated as such in the failure detail, so the
author can tell "already committed, needs a re-tag" from "not even committed yet".

ROOTS, AND WHAT IS NOT PART OF THE PUBLIC RELEASE
-------------------------------------------------
Every root below is an environment variable with a repo-relative default, so this gate
measures the checkout it ships in rather than one particular machine.  Absolute paths
were hardcoded here once; a relocated clone then audited a tree it was not running in,
and the paths themselves disclosed the maintainer's local layout to every visitor.

    GSSC_REPO        the release checkout under test        default: this file's repository
    GSSC_PAPER       the manuscript checkout                default: <repo>/../GSSC-paper

THE MANUSCRIPT CHECKOUT IS NOT PART OF THE PUBLIC RELEASE.
It is a maintainer working tree; a clone of this repository does not contain it, and the
released artefacts are distributed separately (docs/DATASET.md, docs/MODEL_ZOO.md).
A gate that needs one and cannot find it FAILS rather than passing: "the artefact is
not here" is not evidence that it is correct.  Point the variable at your own copy,
or skip the gate.
"""

from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO = Path(os.environ.get("GSSC_REPO") or Path(__file__).resolve().parents[1])
PAPER = Path(os.environ.get("GSSC_PAPER") or REPO.parent / "GSSC-paper")

#: READ-ONLY.  Parsed for the pinned tag; never written.
PAPER_SUPP = PAPER / "supplementary.tex"
PAPER_MAIN = PAPER / "main.tex"

#: The surface a reviewer following the paper's pointer reads or runs.
RELEASE_CRITICAL: Tuple[str, ...] = (
    "README.md",
    "docs/*.md",
    "CITATION.cff",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "scripts/download_assets.py",
    "configs/eval/*.yaml",
)

#: `\texttt{v2.3.8}` and friends.  The braces are part of the pattern so a bare "v2.3.8"
#: appearing in ordinary prose (a sentence about an older release) is not mistaken for the
#: pointer; a second pass without them runs only to DETECT contradictions.
TEXTTT_TAG = re.compile(r"\\texttt\{(v\d+\.\d+\.\d+(?:-[0-9A-Za-z.\-]+)?)\}")
BARE_TAG = re.compile(r"(?<![\w.])(v\d+\.\d+\.\d+(?:-[0-9A-Za-z.\-]+)?)(?![\w.])")


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


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


# --------------------------------------------------------------------------------------
def paper_pointers(*tex: Path) -> Tuple[Dict[str, List[str]], List[str]]:
    """{tag: ['file:line', ...]} from \\texttt{v...}, plus notes about what was read."""
    found: Dict[str, List[str]] = {}
    notes: List[str] = []
    for path in tex:
        if not path.is_file():
            notes.append(f"{path} MISSING")
            continue
        lines = path.read_text(errors="replace").splitlines()
        n = 0
        for i, line in enumerate(lines, 1):
            for tag in TEXTTT_TAG.findall(line):
                found.setdefault(tag, []).append(f"{path.name}:{i}")
                n += 1
        notes.append(f"{path.name}: {n} \\texttt{{v...}} pointer(s) in {len(lines)} lines")
    return (found, notes)


def tag_files(root: Path, tag: str) -> Set[str]:
    r = git(root, "ls-tree", "-r", "--name-only", tag)
    return {l.strip() for l in r.stdout.splitlines() if l.strip()}


def worktree_files(root: Path) -> Set[str]:
    r = git(root, "ls-files", "-co", "--exclude-standard")
    return {l.strip() for l in r.stdout.splitlines() if l.strip()}


def match_patterns(paths: Set[str], patterns: Sequence[str]) -> Dict[str, Set[str]]:
    import fnmatch
    out: Dict[str, Set[str]] = {}
    for pat in patterns:
        out[pat] = {p for p in paths if fnmatch.fnmatch(p, pat)}
    return out


def blob_at(root: Path, tag: str, rel: str) -> Optional[str]:
    r = git(root, "show", f"{tag}:{rel}")
    if r.returncode != 0:
        return None
    return r.stdout


def head_blob(root: Path, rel: str) -> Optional[str]:
    r = git(root, "show", f"HEAD:{rel}")
    return r.stdout if r.returncode == 0 else None


def worktree_blob(root: Path, rel: str) -> Optional[str]:
    p = root / rel
    if not p.is_file():
        return None
    try:
        return p.read_text()
    except (UnicodeDecodeError, OSError):
        return None


def first_diff(a: str, b: str) -> Tuple[int, int, int]:
    """(first differing line number, lines added, lines removed)."""
    al, bl = a.splitlines(), b.splitlines()
    sm = difflib.SequenceMatcher(None, al, bl)
    first = 0
    added = removed = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if not first:
            first = i1 + 1
        removed += i2 - i1
        added += j2 - j1
    return (first, added, removed)


# --------------------------------------------------------------------------------------
def run(root: Path, supp: Path = PAPER_SUPP, main_tex: Path = PAPER_MAIN,
        gate: Optional[Gate] = None) -> Gate:
    g = gate or Gate()

    # ---- CHECK 1: the pointer itself.
    pointers, notes = paper_pointers(supp, main_tex)
    tag: Optional[str] = None
    if len(pointers) == 1:
        tag = next(iter(pointers))
        exists = git(root, "rev-parse", "--verify", f"{tag}^{{commit}}").returncode == 0
    else:
        exists = False
    detail = ""
    if not pointers:
        detail = (f"no \\texttt{{v...}} tag pointer found in {supp} or {main_tex} "
                  f"({'; '.join(notes)}) -- the gate cannot know which tree a reviewer "
                  f"gets, so this is UNMEASURABLE, not a pass")
    elif len(pointers) > 1:
        detail = ("the paper pins MORE THAN ONE tag, so a reviewer's checkout is "
                  "undetermined: "
                  + "; ".join(f"{t} at {', '.join(loc)}" for t, loc in sorted(pointers.items())))
    elif not exists:
        detail = (f"the paper pins {tag} ({', '.join(pointers[tag])}) but "
                  f"`git -C {root} rev-parse {tag}` fails -- a reviewer following the "
                  f"paper's pointer cannot check anything out")
    g.check("paper_pins_exactly_one_existing_tag", bool(tag) and exists, detail)
    if not tag or not exists:
        return g

    # ---- CHECK 2: instrument control -- every release-critical pattern must match
    # something on at least one side.  A glob that has drifted matches nothing and turns
    # this gate into a silent green.
    tf, wf = tag_files(root, tag), worktree_files(root)
    m_tag, m_wt = match_patterns(tf, RELEASE_CRITICAL), match_patterns(wf, RELEASE_CRITICAL)
    empty = [p for p in RELEASE_CRITICAL if not m_tag[p] and not m_wt[p]]
    covered = sorted(set().union(*m_tag.values(), *m_wt.values()))
    g.check("release_critical_patterns_match_files", not empty,
            f"{len(empty)} pattern(s) in RELEASE_CRITICAL match NOTHING at {tag} or in the "
            f"worktree: {empty} -- the file set was renamed and the parity check below "
            f"would be measuring a smaller surface than it claims "
            f"({len(covered)} files currently covered)")

    # ---- CHECK 3: parity.
    diffs: List[str] = []
    for rel in covered:
        at_tag = blob_at(root, tag, rel)
        at_wt = worktree_blob(root, rel)
        if at_tag is None and at_wt is not None:
            diffs.append(f"{rel}: ABSENT at {tag}, present in the worktree "
                         f"({len(at_wt.splitlines())} lines) -- a reviewer never sees it")
            continue
        if at_tag is not None and at_wt is None:
            diffs.append(f"{rel}: present at {tag} "
                         f"({len(at_tag.splitlines())} lines), DELETED in the worktree")
            continue
        if at_tag is None and at_wt is None:
            continue
        if at_tag == at_wt:
            continue
        line, added, removed = first_diff(at_tag, at_wt)
        committed = head_blob(root, rel) == at_wt
        diffs.append(f"{rel}:{line} differs (+{added}/-{removed} lines; "
                     f"{'committed past the tag -- needs a re-tag or a new tag' if committed else 'NOT YET COMMITTED'})")
    g.check("release_critical_files_identical_at_pinned_tag", not diffs,
            f"a reviewer following the paper's pointer to {tag} does NOT get "
            f"{len(diffs)}/{len(covered)} release-critical file(s): "
            + "; ".join(diffs[:8])
            + (f" [+{len(diffs) - 8} more]" if len(diffs) > 8 else "")
            + f" -- remedy is to move the pointer (new tag + edit the paper's "
              f"\\texttt{{v...}}) or move the tag ({tag} re-cut on the fixed tree), never "
              f"to edit this gate")
    return g


# --------------------------------------------------------------------------------------
# Selftest: a fixture repo + fixture .tex in $TMPDIR.  The real repo and the real paper
# are never touched.
# --------------------------------------------------------------------------------------
def _fixture(tmp: Path, name: str, tag: str = "v1.2.3") -> Tuple[Path, Path, Path]:
    root = tmp / name
    (root / "docs").mkdir(parents=True)
    (root / "configs" / "eval").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    (root / "README.md").write_text("# fixture\nline two\n")
    (root / "docs" / "TRAIN.md").write_text("train\n")
    (root / "CITATION.cff").write_text("version: 1.2.3\n")
    (root / "SECURITY.md").write_text("sec\n")
    (root / "CONTRIBUTING.md").write_text("contrib\n")
    (root / "scripts" / "download_assets.py").write_text("print('x')\n")
    (root / "configs" / "eval" / "val_1step.yaml").write_text("lr: 1\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"}
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True,
                   capture_output=True, env=env)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, capture_output=True,
                   env=env)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "init"], check=True,
                   capture_output=True, env=env)
    subprocess.run(["git", "-C", str(root), "tag", tag], check=True, capture_output=True,
                   env=env)
    supp = tmp / f"{name}_supp.tex"
    supp.write_text("Prose about the repo tagged \\texttt{" + tag + "} with configs.\n")
    main_tex = tmp / f"{name}_main.tex"
    main_tex.write_text("No pointer here.\n")
    return (root, supp, main_tex)


def selftest() -> int:
    missed = 0
    tmp = Path(tempfile.mkdtemp(prefix="tagparity_selftest_"))

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

    # --- CONTROL: tag == worktree -> all three checks green.
    root, supp, main_tex = _fixture(tmp, "ctrl")
    bad = [n for n, ok, d in run(root, supp, main_tex).results if not ok]
    if bad:
        print(f"  MISSED   control_tag_equals_worktree_is_green   (failing: {bad})")
        missed += 1
    else:
        print("  TRIPPED  control_tag_equals_worktree_is_green")

    # --- FAULT 1: THE defect.  Edit a release-critical file AFTER the tag.  The mutation
    # is asserted, and the file is left UNCOMMITTED -- which is exactly the state a fix
    # cycle is in, and the state every worktree-reading gate calls "fixed".
    root1, supp1, main1 = _fixture(tmp, "f1")
    readme = root1 / "README.md"
    before = readme.read_text()
    readme.write_text(before + "a fix that a reviewer at the tag will never see\n")
    assert readme.read_text() != before, "the fault did not perturb the file"
    g1 = run(root1, supp1, main1)
    expect("worktree edited past the tag", g1,
           "release_critical_files_identical_at_pinned_tag", True)
    det = dict((n, d) for n, ok, d in g1.results
               if not ok).get("release_critical_files_identical_at_pinned_tag", "")
    if "README.md" in det and "NOT YET COMMITTED" in det:
        print("  TRIPPED  parity_detail_names_file_and_commit_state")
    else:
        print(f"  MISSED   parity_detail_names_file_and_commit_state   ({det[:180]})")
        missed += 1

    # --- FAULT 1b: a file ADDED after the tag must also be reported -- expanding only the
    # tag's tree would miss it.
    root1b, supp1b, main1b = _fixture(tmp, "f1b")
    (root1b / "docs" / "NEWDOC.md").write_text("added after the tag\n")
    expect("doc added after the tag", run(root1b, supp1b, main1b),
           "release_critical_files_identical_at_pinned_tag", True)

    # --- FAULT 1c: a file DELETED after the tag -- expanding only the worktree would miss
    # it.
    root1c, supp1c, main1c = _fixture(tmp, "f1c")
    (root1c / "docs" / "TRAIN.md").unlink()
    expect("doc deleted after the tag", run(root1c, supp1c, main1c),
           "release_critical_files_identical_at_pinned_tag", True)

    # --- FAULT 2: the paper pins a tag git does not have.
    root2, supp2, main2 = _fixture(tmp, "f2")
    b2 = supp2.read_text()
    supp2.write_text(b2.replace("v1.2.3", "v9.9.9"))
    assert supp2.read_text() != b2, "the pointer mutation was a no-op"
    expect("paper pins a nonexistent tag", run(root2, supp2, main2),
           "paper_pins_exactly_one_existing_tag", True)

    # --- FAULT 2b: two DIFFERENT pointers across the two .tex files.
    root2b, supp2b, main2b = _fixture(tmp, "f2b")
    main2b.write_text("Also see \\texttt{v1.0.0} for the older release.\n")
    expect("contradictory pointers", run(root2b, supp2b, main2b),
           "paper_pins_exactly_one_existing_tag", True)

    # --- FAULT 2c: NO pointer at all is UNMEASURABLE, i.e. red -- never a quiet pass.
    root2c, supp2c, main2c = _fixture(tmp, "f2c")
    supp2c.write_text("Prose with no tag pointer at all.\n")
    expect("no pointer in the paper", run(root2c, supp2c, main2c),
           "paper_pins_exactly_one_existing_tag", True)

    # --- FAULT 3: a drifted glob.  RELEASE_CRITICAL must not silently cover less than it
    # claims; this is the anti-vacuity arm and it must be REACHABLE.
    global RELEASE_CRITICAL
    keep = RELEASE_CRITICAL
    try:
        RELEASE_CRITICAL = keep + ("documentation/*.md",)  # renamed-away directory
        root3, supp3, main3 = _fixture(tmp, "f3")
        expect("pattern matching nothing", run(root3, supp3, main3),
               "release_critical_patterns_match_files", True)
    finally:
        RELEASE_CRITICAL = keep

    # --- FAULT 4: the pointer must come FROM THE PAPER.  If the tag were hardcoded, a
    # paper edit could not change the gate's answer; assert that it can.
    root4, supp4, main4 = _fixture(tmp, "f4", tag="v7.7.7")
    g4 = run(root4, supp4, main4)
    ok4 = all(ok for _, ok, _ in g4.results)
    if ok4:
        print("  TRIPPED  pointer_is_read_from_the_paper_not_hardcoded")
    else:
        print("  MISSED   pointer_is_read_from_the_paper_not_hardcoded   "
              f"({[(n, d[:80]) for n, ok, d in g4.results if not ok]})")
        missed += 1

    shutil.rmtree(tmp, ignore_errors=True)
    total = 10
    print(f"SELFTEST OK: {total - missed}/{total} checks provably fail when broken"
          if not missed else
          f"SELFTEST FAILED: {total - missed}/{total} checks provably fail when broken")
    return 1 if missed else 0


def main() -> int:
    print(f"[check_tag_parity] repo={REPO} paper={PAPER_SUPP}")
    return run(REPO).report()


if __name__ == "__main__":
    raise SystemExit(selftest() if "--selftest" in sys.argv else main())

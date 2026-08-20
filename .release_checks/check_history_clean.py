#!/usr/bin/env python3
"""GATE: no release-stop marker survives anywhere in git history.

THE DEFECT THIS GATE EXISTS FOR
-------------------------------
Blob ``995e1af``, path ``.migration_audit.local.md``, whose FIRST LINE reads

    "# GSSC-S2D2 Migration Audit (working file — strip before public release)"

and which carries a section headed "## Privacy / strip-list (must grep before public
push)", was added in ``2112d7a`` ("P4-P6: assets staging + privacy sweep + final polish")
and deleted in ``1b24856`` ("chore: gitignore migration audit + remove leaked working
file").  Measured 2026-08-20:

  * the blob is still reachable: ``git cat-file -p 995e1af`` prints it in full;
  * ``git tag --contains 2112d7a`` = ALL FOURTEEN tags, v1.0.0-rc1 .. v2.3.8 -- including
    v2.3.8, the tag the paper pins;
  * ``git branch -r --contains 2112d7a`` = ``origin/main``, i.e. it is already PUSHED.

The file names the 230 GB / 178 GB private asset paths, the internal launcher scripts, and
a strip-list of files that were never meant to ship.  `git rm` moved it out of the WORKTREE
and out of nothing else.  This is the whole point: a deletion commit is a change to the tip,
not to history, and every gate that reads the worktree reports the repository as clean.

WHY THE DETECTOR IS TWO-CHANNEL (PATH *AND* CONTENT)
----------------------------------------------------
The path ``.migration_audit.local.md`` is itself the tell -- a ``.local.`` infix is the
convention this repo uses for "never ship this" (six ``.gitignore`` revisions say so in
prose).  A content-only detector would have missed a working file that happened to carry no
marker sentence; a path-only detector misses a leak inside an ordinary-looking filename.
Both channels run, and a hit on either fails.

CALIBRATION -- EVERY MARKER BELOW WAS MEASURED OVER ALL 978 TEXT BLOBS IN THIS HISTORY
--------------------------------------------------------------------------------------
Markers were not guessed.  Candidates were run over the whole object graph and the noisy
ones were REMOVED, with the measurement recorded here so nobody re-adds them:

  KEPT (1 hit each, all of them blob 995e1af):
      "strip before public release", "strip-list", "working file", "before public push"
  KEPT (0 hits today -- they guard classes this repo has not yet leaked):
      "do not commit", "internal only", "private note", "TODO ... before release",
      "delete/remove before release", "for author eyes only", "WIP ... do not ..."
  REJECTED, "not (yet) public": 47 hits, every one of them the LEGITIMATE README/CHANGELOG
      sentence "Assets are not yet public."  A gate that fails on an accurate availability
      statement is noise, and this harness has already had to remove that class of noise
      once (see the paper harness's check_availability_parity docstring).
  EXEMPTED, "local-only" / "never publish": 6 hits, all six ``.gitignore`` revisions, all
      six of them COMMENTS DESCRIBING WHAT IS EXCLUDED ("# Local-only retrain / debug
      scratch (never publish)").  An ignore file is the MECHANISM for not leaking; scanning
      it as if it were a leak inverts the gate.  Exempted by path, with the reason stated.

The exemption table is an allowlist of EXEMPTIONS, never of TARGETS.  An allowlist of
targets fails silent -- whatever is not listed is never examined.  This one fails loud:
every marker hit is reported unless a listed path claims it, so a new working file is
flagged the day it appears.

SCOPE, STATED SO THE GREEN IS HONEST
------------------------------------
  * Blobs only.  COMMIT MESSAGES are not scanned: ``1b24856``'s own subject contains the
    word "leaked" and rewriting history is not what this gate asks for.
  * Binary blobs (undecodable as UTF-8) are counted and reported, never silently skipped.
  * A blob over ``MAX_BLOB_BYTES`` is reported as UNSCANNED rather than passed.

DIRECTION OF THE FIX -- READ BEFORE "FIXING"
--------------------------------------------
There is no worktree edit that makes CHECK A pass.  Passing it requires REWRITING HISTORY
(``git filter-repo --invert-paths --path .migration_audit.local.md``), re-tagging all
fourteen tags, and a force-push -- and the paper pins v2.3.8, so the retag must preserve
that name.  Anything less leaves the blob fetchable by anyone who clones the repo.
THIS GATE IS THE ONE THAT GUARDS FLIPPING THE REPOSITORY TO PUBLIC.  It is expected to be
RED until that rewrite happens; a green here obtained by editing the worktree is a lie.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO = Path("/workspace/GSSC-S2D2")

#: Blobs larger than this are reported as UNSCANNED, not passed. 3 MB covers every text
#: file in this history (largest text blob measured: uv.lock, ~300 KB).
MAX_BLOB_BYTES = 3_000_000

# --------------------------------------------------------------------------------------
# Markers.  Each is a RELATION between words, not a fixed sentence: the paper harness
# lost a whole finding to a literal probe for "upon acceptance" that could not see
# "upon **paper** acceptance".  One inserted word must not blind a marker here either,
# so every pattern allows filler where filler is plausible.
# --------------------------------------------------------------------------------------
CONTENT_MARKERS: Dict[str, str] = {
    # The live defect, verbatim from 995e1af line 1.
    "strip-before-public-release": r"\b(?:strip|delete|remove|purge|drop)\b[^\n]{0,40}?"
                                   r"\bbefore\b[^\n]{0,40}?"
                                   r"\b(?:public|release|releasing|publish|publishing|push|"
                                   r"submission|submitting|open[- ]?sourc\w*)\b",
    # The section heading in the same blob.
    "strip-list": r"\bstrip[-\s]?list\b",
    # "(working file — strip before public release)"; "scratch notes"; "notes to self".
    "working-scratch-file": r"\b(?:working|scratch|throwaway|temp(?:orary)?)\s+"
                            r"(?:file|files|notes?|copy|doc(?:ument)?|pad)\b"
                            r"|\bnotes?[-\s]to[-\s]self\b",
    # "must grep before public push" -- deferral of a privacy action.
    "before-public-push": r"\bbefore\b[^\n]{0,30}?\bpublic\s+(?:push|release|launch)\b",
    # "ship" was MEASURED as a false positive: SECURITY.md:38 reads "Hardcoded
    # credentials or tokens (we do not ship any)" -- the object is credentials, not
    # this file. The marker means "do not commit THIS", so the verb list is restricted
    # to acts performed on a file. Re-adding "ship" re-breaks SECURITY.md.
    "do-not-commit": r"\bdo\s+not\s+(?:commit|check\s+in|push|publish|distribute|share)\b",
    "internal-only": r"\binternal[-\s]only\b|\bfor\s+internal\s+use\b"
                     r"|\binternal\s+(?:note|memo|doc(?:ument)?)s?\b",
    "private-note": r"\bprivate\s+(?:note|memo|doc(?:ument)?|comment)s?\b",
    "todo-before-release": r"\b(?:TODO|FIXME|XXX|HACK)\b[^\n]{0,60}?\b(?:before|prior\s+to)\b"
                           r"[^\n]{0,40}?\b(?:release|publish|public|submission)\b",
    "author-eyes-only": r"\bfor\s+(?:my|our|the\s+author'?s?|author)\s+eyes\s+only\b",
    "wip-do-not": r"\bWIP\b[^\n]{0,40}?\bdo\s+not\b",
    "confidential": r"\bconfidential\b|\bnot\s+for\s+(?:distribution|circulation)\b",
}

#: PATH channel. The leaked file announced itself in its own name. Patterns are anchored on
#: path SEGMENTS so an innocent substring cannot trip them (``docs/PRIVATE_API.md`` would,
#: deliberately -- a doc named PRIVATE in a public release is worth one look).
PATH_MARKERS: Dict[str, str] = {
    "dot-local-infix": r"(?:^|/)[^/]*\.local\.[^/]*$",
    "dot-private-infix": r"(?:^|/)[^/]*\.(?:private|secret|internal)\.[^/]*$",
    "agent-working-file": r"(?:^|/)(?:CLAUDE|AGENTS?|GEMINI|CURSOR)\.md$",
    "migration-audit": r"(?:^|/)\.?migration[-_]audit",
    "notes-to-self": r"(?:^|/)(?:notes?[-_]to[-_]self|scratch|todo\.local)",
    "draft-doc": r"(?:^|/)[^/]*\.draft\.(?:md|txt|tex)$",
}

#: EXEMPTIONS: (path regex, marker name or None for all, reason).  Measured false
#: positives only.  Anything not listed here is reported.
EXEMPTIONS: Sequence[Tuple[str, Optional[str], str]] = (
    (r"(?:^|/)\.(?:git|docker|npm|prettier)ignore$", None,
     "an ignore file is the mechanism for NOT leaking; its comments describe what is "
     "excluded ('# Local-only retrain / debug scratch (never publish)') -- 6 revisions "
     "measured 2026-08-20, all six comments"),
    (r"^\.release_checks/", None,
     "this gate harness NAMES the markers it hunts; the detector must not detect itself"),
)


# --------------------------------------------------------------------------------------
# Tiny gate harness (duplicated per gate on purpose: each gate must run standalone).
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
            if ok:
                print(f"  PASS  {name}")
            else:
                print(f"  FAIL  {name}   ({detail})")
        n = len(self.failures)
        print(f"OK: 0 failing check(s)" if n == 0 else f"FAILED: {n} failing check(s)")
        return 0 if n == 0 else 1


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True).stdout


# --------------------------------------------------------------------------------------
# Detector -- pure functions over (path, text).  Kept pure so --selftest can perturb the
# INPUT rather than a pattern string: a selftest that edits the regex proves nothing.
# --------------------------------------------------------------------------------------
def exemption_for(path: str, marker: str) -> Optional[str]:
    for pat, only, reason in EXEMPTIONS:
        if re.search(pat, path) and (only is None or only == marker):
            return reason
    return None


def markers_in(path: str, text: Optional[str]) -> List[Tuple[str, str, str]]:
    """Return [(channel, marker, evidence)] for one blob, exemptions already applied."""
    hits: List[Tuple[str, str, str]] = []
    for name, pat in PATH_MARKERS.items():
        if re.search(pat, path, re.I) and not exemption_for(path, name):
            hits.append(("path", name, path))
    if text is not None:
        for name, pat in CONTENT_MARKERS.items():
            m = re.search(pat, text, re.I)
            if m and not exemption_for(path, name):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(("content", name, f"line {line}: {m.group(0)[:70]!r}"))
    return hits


# --------------------------------------------------------------------------------------
# History walk
# --------------------------------------------------------------------------------------
class Blob:
    __slots__ = ("sha", "paths", "size", "text", "binary", "oversize")

    def __init__(self, sha: str, paths: List[str], size: int) -> None:
        self.sha, self.paths, self.size = sha, paths, size
        self.text: Optional[str] = None
        self.binary = False
        self.oversize = False


def walk_history(root: Path) -> List[Blob]:
    """Every blob reachable from ANY ref (branches, tags, notes), with its recorded paths.

    ``--objects --all`` is the load-bearing part: ``git log`` on a path shows nothing for a
    file deleted at the tip, and ``git grep HEAD`` shows nothing at all. This is the only
    view in which a deleted-but-reachable blob is visible.
    """
    paths_by_sha: Dict[str, List[str]] = {}
    for line in git(root, "rev-list", "--objects", "--all").splitlines():
        parts = line.split(" ", 1)
        sha = parts[0]
        paths_by_sha.setdefault(sha, [])
        if len(parts) > 1 and parts[1] not in paths_by_sha[sha]:
            paths_by_sha[sha].append(parts[1])
    if not paths_by_sha:
        return []
    payload = ("\n".join(paths_by_sha) + "\n").encode()
    chk = subprocess.run(["git", "-C", str(root), "cat-file", "--batch-check"],
                         input=payload, capture_output=True).stdout.decode(errors="replace")
    blobs: List[Blob] = []
    for line in chk.splitlines():
        f = line.split()
        if len(f) == 3 and f[1] == "blob":
            blobs.append(Blob(f[0], paths_by_sha.get(f[0], []), int(f[2])))
    small = [b for b in blobs if b.size <= MAX_BLOB_BYTES]
    # An oversize blob is only a COVERAGE HOLE if it is text. This repo's five oversize
    # blobs are all figure PDFs (assets/fig2_pipeline_v2.pdf, 3.6 MB, four revisions;
    # assets/fig3_s2d2_v2.pdf, 5.4 MB) -- binary, and no more scannable than the 18 small
    # binaries. Deciding by SNIFFING the bytes rather than by file extension: a working
    # file saved as `notes.pdf` would be waved through by an extension allowlist.
    for b in blobs:
        if b.size > MAX_BLOB_BYTES:
            head = subprocess.run(["git", "-C", str(root), "cat-file", "blob", b.sha],
                                  capture_output=True).stdout[:8192]
            if b"\x00" in head:
                b.binary = True
            else:
                b.oversize = True
    if small:
        raw = subprocess.run(["git", "-C", str(root), "cat-file", "--batch"],
                             input=("\n".join(b.sha for b in small) + "\n").encode(),
                             capture_output=True).stdout
        by_sha = {b.sha: b for b in small}
        i = 0
        while i < len(raw):
            j = raw.index(b"\n", i)
            sha, _typ, size_s = raw[i:j].decode().split()
            size = int(size_s)
            i = j + 1
            data = raw[i:i + size]
            i += size + 1
            b = by_sha.get(sha)
            if b is None:
                continue
            try:
                b.text = data.decode("utf-8")
            except UnicodeDecodeError:
                b.binary = True
    return blobs


def introducing_commit(root: Path, sha: str) -> Tuple[str, str]:
    """Oldest commit that added this blob, plus its subject.  ``--find-object`` reports the
    add AND the delete; the oldest of the two is the introduction."""
    out = git(root, "log", "--all", "--reverse", "--format=%H\t%s", f"--find-object={sha}")
    lines = [l for l in out.splitlines() if l.strip()]
    if not lines:
        return ("", "")
    h, _, subj = lines[0].partition("\t")
    return (h, subj)


def containing_tags(root: Path, commit: str) -> List[str]:
    if not commit:
        return []
    return [t for t in git(root, "tag", "--contains", commit).split() if t]


def pushed_refs(root: Path, commit: str) -> List[str]:
    if not commit:
        return []
    return [r.strip() for r in git(root, "branch", "-r", "--contains", commit).splitlines()
            if r.strip()]


def head_files(root: Path) -> List[Tuple[str, Optional[str]]]:
    """HEAD tree PLUS the working tree (tracked + untracked-not-ignored).

    Both, because the two answer different questions: HEAD is what a clone gets, the
    worktree is what the author is about to commit.  A re-leak must be caught before it
    becomes another immortal blob.
    """
    seen: Dict[str, Optional[str]] = {}
    for rel in git(root, "ls-tree", "-r", "--name-only", "HEAD").splitlines():
        rel = rel.strip()
        if not rel:
            continue
        blob = subprocess.run(["git", "-C", str(root), "show", f"HEAD:{rel}"],
                              capture_output=True).stdout
        try:
            seen[rel] = blob.decode("utf-8")
        except UnicodeDecodeError:
            seen[rel] = None
    for rel in git(root, "ls-files", "-co", "--exclude-standard").splitlines():
        rel = rel.strip()
        p = root / rel
        if not rel or not p.is_file():
            continue
        try:
            seen[rel] = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            seen.setdefault(rel, None)
    return sorted(seen.items())


# --------------------------------------------------------------------------------------
def run(root: Path, g: Optional[Gate] = None) -> Gate:
    g = g or Gate()

    blobs = walk_history(root)

    # ---- CHECK 0: instrument controls.  A green from a broken walk is the worst outcome
    # this gate can produce, so BOTH controls run: the walk read something real, AND the
    # query fires on a positive fixture.  (Paper-harness lesson: a positive control on the
    # INSTRUMENT is not a positive control on the QUERY -- only one of the two was run
    # there, and a whole finding was nearly reported as fixed.)
    probe = "# X (working file - strip before public release)\n"
    walk_ok = len(blobs) > 0 and any(b.text is not None for b in blobs)
    query_ok = bool(markers_in("some/file.md", probe))
    path_ok = bool(markers_in(".migration_audit.local.md", ""))
    g.check("instrument_controls",
            walk_ok and query_ok and path_ok,
            f"{root}: walk saw {len(blobs)} blob(s), decodable="
            f"{sum(1 for b in blobs if b.text is not None)}; content-probe fired={query_ok}; "
            f"path-probe fired={path_ok} -- the scan cannot be trusted")

    # ---- CHECK A: history
    findings: List[str] = []
    for b in sorted(blobs, key=lambda x: x.sha):
        for path in (b.paths or [f"<unnamed blob {b.sha[:9]}>"]):
            for channel, marker, evidence in markers_in(path, b.text):
                commit, subj = introducing_commit(root, b.sha)
                tags = containing_tags(root, commit)
                remotes = pushed_refs(root, commit)
                findings.append(
                    f"blob {b.sha[:9]} path {path!r} [{channel}:{marker}] {evidence}; "
                    f"introduced by {commit[:7] or '?'} ({subj[:48]!r}); "
                    f"reachable from {len(tags)} tag(s) "
                    f"[{','.join(tags) if tags else 'none'}]"
                    + (f"; ALREADY PUSHED to {','.join(remotes)}" if remotes else ""))
    g.check("history_free_of_release_stop_markers", not findings,
            "; ".join(findings[:6]) + (f" [+{len(findings) - 6} more]" if len(findings) > 6
                                       else "")
            + " -- no worktree edit fixes this: the blob is reachable from every listed "
              "tag. Requires git filter-repo + retag (v2.3.8 must keep its name) + "
              "force-push BEFORE the repository is made public")

    # ---- CHECK B: HEAD tree + worktree
    head_hits: List[str] = []
    for rel, text in head_files(root):
        for channel, marker, evidence in markers_in(rel, text):
            head_hits.append(f"{rel} [{channel}:{marker}] {evidence}")
    g.check("head_free_of_release_stop_markers", not head_hits,
            "; ".join(head_hits[:6]) + " -- a NEW working file is on its way into history; "
            "delete it before it is committed")

    # ---- CHECK C: coverage honesty.  Anything the scan could not read is named, not
    # rolled into the green.
    unscanned = [f"{b.sha[:9]} ({b.paths[:1]}, {b.size}B)" for b in blobs if b.oversize]
    binaries = sum(1 for b in blobs if b.binary)
    g.check("scan_coverage_complete", not unscanned,
            f"{len(unscanned)} blob(s) exceed MAX_BLOB_BYTES and were NOT scanned: "
            f"{', '.join(unscanned[:5])} (binary blobs, correctly unscannable: {binaries})")
    return g


# --------------------------------------------------------------------------------------
# Selftest.  The repos are read-only to this gate, so the fault is injected into a
# THROWAWAY git repo built in $TMPDIR and the SAME walk is pointed at it.  That proves the
# history walk itself finds a deleted-but-reachable blob -- not merely that a regex
# matches a string, which is what a fixture-only selftest would show.
# --------------------------------------------------------------------------------------
def _scratch_repo(tmp: Path, leak: bool, name: str = "") -> Path:
    root = tmp / (name or ("leaky" if leak else "clean"))
    root.mkdir(parents=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}

    def run_git(*a: str) -> None:
        subprocess.run(["git", "-C", str(root), *a], check=True,
                       capture_output=True, env=env)

    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True,
                   capture_output=True, env=env)
    (root / "README.md").write_text("# scratch\nAssets are not yet public.\n")
    (root / ".gitignore").write_text("# Local-only debug scratch (never publish)\n*.log\n")
    run_git("add", "-A")
    run_git("commit", "-qm", "init")
    if leak:
        (root / ".migration_audit.local.md").write_text(
            "# Audit (working file - strip before public release)\n"
            "## Privacy / strip-list (must grep before public push)\n")
        run_git("add", "-A")
        run_git("commit", "-qm", "P4-P6: privacy sweep")
        run_git("tag", "v9.9.9")
        run_git("rm", "-q", ".migration_audit.local.md")
        run_git("commit", "-qm", "chore: remove leaked working file")
    return root


def selftest() -> int:
    missed = 0

    def expect(name: str, gate: Gate, want_fail: bool) -> None:
        nonlocal missed
        got = dict((n, ok) for n, ok, _ in gate.results).get(name)
        if got is None:
            print(f"  MISSED   {name}   (check never ran)")
            missed += 1
        elif (not got) == want_fail:
            print(f"  TRIPPED  {name}")
        else:
            print(f"  MISSED   {name}   (expected "
                  f"{'FAIL' if want_fail else 'PASS'}, got {'PASS' if got else 'FAIL'})")
            missed += 1

    tmp = Path(tempfile.mkdtemp(prefix="hist_selftest_"))

    # --- CONTROL: a clean history must be green on all three checks.  Without this the
    # gate could be failing for an unrelated reason and the "TRIPPED" below would be
    # meaningless.
    clean = run(_scratch_repo(tmp, leak=False))
    ctrl_bad = [n for n, ok, _ in clean.results if not ok]
    if ctrl_bad:
        print(f"  MISSED   control_clean_repo_is_green   (failing: {ctrl_bad})")
        missed += 1
    else:
        print("  TRIPPED  control_clean_repo_is_green")

    # --- FAULT 1: the real defect, replayed.  Blob added, tagged, then DELETED at the tip
    # -- so the worktree is clean and only the history walk can see it.
    leaky_root = _scratch_repo(tmp, leak=True)
    assert not (leaky_root / ".migration_audit.local.md").exists(), \
        "fixture is wrong: the leak must be absent from the worktree, or CHECK A is being " \
        "proved by CHECK B's evidence"
    leaky = run(leaky_root)
    expect("history_free_of_release_stop_markers", leaky, want_fail=True)
    # ...and the detail string must actually carry the actionable facts.
    det = dict((n, d) for n, ok, d in leaky.results
               if not ok).get("history_free_of_release_stop_markers", "")
    if "v9.9.9" in det and ".migration_audit.local.md" in det:
        print("  TRIPPED  history_detail_names_path_and_tag")
    else:
        print(f"  MISSED   history_detail_names_path_and_tag   (detail={det[:160]!r})")
        missed += 1

    # --- FAULT 2: a live leak in the worktree (CHECK B).  Perturbation asserted.
    live = tmp / "live"
    live.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(live)], check=True,
                   capture_output=True)
    (live / "README.md").write_text("# scratch\n")
    (live / "notes.md").write_text("Internal only: do not commit this to the public repo.\n")
    before = markers_in("notes.md", "harmless\n")
    after = markers_in("notes.md", (live / "notes.md").read_text())
    assert not before and after, "the injected content did not perturb the detector"
    expect("head_free_of_release_stop_markers", run(live), want_fail=True)

    # --- FAULT 3: coverage honesty.  A blob past the size cap must be REPORTED, never
    # folded into a green.  Mutating the constant is the only way to reach this arm, and
    # the arm is asserted to be reachable -- a gate arm that never fires hides its own bug.
    global MAX_BLOB_BYTES
    keep = MAX_BLOB_BYTES
    try:
        MAX_BLOB_BYTES = 1
        oversized = run(_scratch_repo(tmp, leak=False, name="capped"))
        expect("scan_coverage_complete", oversized, want_fail=True)
        # And the walk-control must ALSO go red, because nothing was decodable: proof the
        # cap was really applied rather than the run silently succeeding.
        expect("instrument_controls", oversized, want_fail=True)
    finally:
        MAX_BLOB_BYTES = keep

    # --- FAULT 4: the REJECTED marker must stay rejected.  "not yet public" produced 47
    # false positives on this repo's own README/CHANGELOG; if someone re-adds it, this
    # fails and the noise is caught at the gate instead of in the author's inbox.
    if markers_in("README.md", "> **Assets are not yet public.** The checkpoints"):
        print("  MISSED   rejected_marker_stays_rejected   ('not yet public' is back: "
              "47 measured false positives")
        missed += 1
    else:
        print("  TRIPPED  rejected_marker_stays_rejected")

    # --- FAULT 5: exemptions are SCOPED.  The .gitignore exemption must not excuse the
    # same marker in an ordinary file.
    if exemption_for(".gitignore", "working-scratch-file") and \
            not exemption_for("docs/DATASET.md", "working-scratch-file"):
        print("  TRIPPED  exemption_is_scoped_to_ignore_files")
    else:
        print("  MISSED   exemption_is_scoped_to_ignore_files   (exemption leaks to "
              "ordinary docs, or the ignore-file exemption is gone)")
        missed += 1

    shutil.rmtree(tmp, ignore_errors=True)
    total = 8
    print(f"SELFTEST OK: {total - missed}/{total} checks provably fail when broken"
          if not missed else
          f"SELFTEST FAILED: {total - missed}/{total} checks provably fail when broken")
    return 1 if missed else 0


def main() -> int:
    print(f"[check_history_clean] repo={REPO}")
    return run(REPO).report()


if __name__ == "__main__":
    raise SystemExit(selftest() if "--selftest" in sys.argv else main())

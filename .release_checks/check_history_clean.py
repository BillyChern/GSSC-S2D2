#!/usr/bin/env python3
"""GATE: no release-stop marker survives anywhere in git history.

THE DEFECT CLASS THIS GATE EXISTS FOR
-------------------------------------
An internal working file -- the kind that carries a "strip before public release" header and
a strip-list of things that must never ship -- was committed, then removed with ``git rm``.
That moved it out of the WORKTREE and out of nothing else. A deletion commit is a change to
the tip, not to history: the blob stayed reachable from every tag cut before it, and it had
already been pushed. Every gate that reads the worktree reported the repository as clean.

RESOLVED 2026-08-20 by a history rewrite (``git filter-repo --invert-paths --path <file>``),
a re-cut of all affected tags, and a force-push. The gate stays because the defect class
recurs: the next working file will look just as harmless at the tip.

Deliberately NOT recorded here: the offending blob's hash, its commit ids, ITS NAME, its
first line, or its section headings. This file ships in the public repository, and a
force-push does not delete objects from the remote -- an orphaned commit stays fetchable by
anyone holding its sha until the host garbage-collects. Publishing the forensics of a purge
alongside the purge hands out the key to it. Provenance lives in the release runbook,
outside the repo.

That promise covers the SELFTEST FIXTURE below as well, and it did not always: an earlier
pass cleaned this docstring and left the fixture reconstructing the purged file's path and
two of its header lines, because the verification sweep was scoped to the sha and the commit
ids and never to the filename or the header text. A scoped sweep is guaranteed to come back
green. The fixture now uses ``SELFTEST_LEAK_PATH`` and PLACEHOLDER text assembled from the
MARKER NAMES this file already publishes; it only has to trip CONTENT_MARKERS, and it never
has to resemble a real file. Any future sweep must include the filename and the header text
among its search terms, not just the hashes.

WHY THE DETECTOR IS TWO-CHANNEL (PATH *AND* CONTENT)
----------------------------------------------------
A ``.local.`` infix is the convention this repo uses for "never ship this" (six ``.gitignore``
revisions say so in prose), so the path alone can be the tell. A content-only detector would
miss a working file carrying no marker sentence; a path-only detector misses a leak inside an
ordinary-looking filename. Both channels run, and a hit on either fails.

CALIBRATION -- EVERY MARKER BELOW WAS MEASURED OVER THE WHOLE OBJECT GRAPH
-------------------------------------------------------------------------
Markers were not guessed. Candidates were run over every text blob in the history and the
noisy ones REMOVED, with the measurement recorded here so nobody re-adds them:

  KEPT: "strip before public release", "strip-list", "working file", "before public push",
      "do not commit", "internal only", "private note", "TODO ... before release",
      "delete/remove before release", "for author eyes only", "WIP ... do not ...".
      All now score 0 hits -- they guard classes this repo is currently free of.
  REJECTED, "not (yet) public": 47 hits, every one the LEGITIMATE README/CHANGELOG sentence
      "Assets are not yet public." A gate that fails on an accurate availability statement is
      noise, and this harness has already had to remove that class of noise once (see the
      paper harness's check_availability_parity docstring).
  EXEMPTED, "local-only" / "never publish": 6 hits, all six ``.gitignore`` revisions, all six
      COMMENTS DESCRIBING WHAT IS EXCLUDED ("# Local-only retrain / debug scratch (never
      publish)"). An ignore file is the MECHANISM for not leaking; scanning it as if it were a
      leak inverts the gate. Exempted by path, with the reason stated.

The exemption table is an allowlist of EXEMPTIONS, never of TARGETS. An allowlist of targets
fails silent -- whatever is not listed is never examined. This one fails loud: every marker
hit is reported unless a listed path claims it, so a new working file is flagged the day it
appears.

SCOPE, STATED SO THE GREEN IS HONEST
------------------------------------
  * Blobs only. COMMIT MESSAGES are not scanned; rewriting them is not what this gate asks for.
  * Binary blobs (undecodable as UTF-8) are counted and reported, never silently skipped.
  * A blob over ``MAX_BLOB_BYTES`` is reported as UNSCANNED rather than passed.
  * Reachability only. This gate reads the LOCAL object graph, so it cannot see an object that
    survives on the remote with no ref pointing at it. After any purge, test the remote
    directly -- fetch the old sha into a throwaway clone -- before flipping to public.
  * EXEMPTIONS waves ``.release_checks/`` past both marker channels, and that exemption
    applies to the directory's OLD BLOBS TOO. Cleaning the fixture at the tip therefore does
    NOT make CHECK A's green cover the earlier revisions of this file, which are still
    reachable from every tag cut before the cleanup. Same rule as the top of this docstring:
    only a rewrite + retag + force-push removes them, and only a direct test against the
    remote shows what survived. A green here is not evidence that it did.
  * Commit MESSAGES and commit DIFFS are outside the walk -- it reads blob CONTENT. A commit
    body that narrates a purge, naming the file and quoting its lines, is invisible to every
    check below.
  * SIZE OF THAT BLIND SPOT, MEASURED rather than estimated (2026-08-22, local object graph,
    reachable from HEAD). The identifiers of the working document this gate was built for --
    its lower-case stem, the upper-case spelling that survives in ``.gitignore``, its blob sha,
    both of its original commit ids, and its first-line heading -- appear in the DIFFS of
    FOUR reachable commits and in the MESSAGES of TWO -- FIVE distinct commits, since one
    carries both. All five are contained in tag ``v2.3.8``, the tag the paper pins, and three
    of the five in all fourteen tags this repository has cut. The counts are
    reproduced without republishing the identifiers by pulling them from ``.gitignore`` and
    running ``git log --format=%h -G"<id>" HEAD`` and ``... --grep="<id>" -i HEAD`` per id, then
    taking the union -- NOT by ``-S`` on the lower-case stem alone, which answers 3: it misses
    the commit that added the upper-case ``.gitignore`` entry, which spells the name a fourth way.
    The purge itself did land: the blob sha and both original commit ids no longer resolve
    (``git cat-file -t`` fails on all three) and no reachable tree still carries the path.
    What survives is the FORENSICS, in diffs and messages this walk cannot reach -- so CHECK A
    stays green while the identifiers remain fetchable. Closing it needs a second
    ``filter-repo`` pass over those commits, a re-cut of the tags under the same NAMES, a
    force-push, and -- per the reachability bullet above -- a direct fetch against the REMOTE,
    which a force-push does not clean.

DIRECTION OF THE FIX -- READ BEFORE "FIXING"
--------------------------------------------
There is no worktree edit that makes CHECK A pass. Passing it requires REWRITING HISTORY,
re-tagging every affected tag, and a force-push -- and the paper pins a tag by NAME, so the
retag must preserve that name. Anything less leaves the blob fetchable by anyone who clones.
THIS GATE IS THE ONE THAT GUARDS FLIPPING THE REPOSITORY TO PUBLIC. A green obtained by
editing the worktree is a lie.

ROOTS, AND WHAT IS NOT PART OF THE PUBLIC RELEASE
-------------------------------------------------
The one root this gate reads is an environment variable with a repo-relative default, so it
measures the checkout it ships in rather than one particular machine.  It was an absolute
path once; a relocated clone then audited a tree it was not running in, and the path itself
disclosed the maintainer's local layout to every visitor.

    GSSC_REPO        the release checkout under test        default: this file's repository

The sibling gates read two more trees, ``GSSC_ASSETS`` (the asset staging bundle) and
``GSSC_PAPER`` (the manuscript checkout).  NEITHER IS PART OF THE PUBLIC RELEASE: both are
maintainer working trees, a clone does not contain them, and the released artefacts are
distributed separately (docs/DATASET.md, docs/MODEL_ZOO.md).
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

#: The checkout under test. Defaults to the repository this file ships in, so a clone
#: audits ITSELF; ``GSSC_REPO`` points it somewhere else. It used to be an absolute
#: path on the maintainer's box, which meant a relocated clone silently measured a
#: tree it was not running in -- a green that proved nothing about the checkout.
REPO = Path(os.environ.get("GSSC_REPO") or Path(__file__).resolve().parents[1])

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
    # Derived from the header line of the working file this gate was written for.
    "strip-before-public-release": r"\b(?:strip|delete|remove|purge|drop)\b[^\n]{0,40}?"
                                   r"\bbefore\b[^\n]{0,40}?"
                                   r"\b(?:public|release|releasing|publish|publishing|push|"
                                   r"submission|submitting|open[- ]?sourc\w*)\b",
    # A section heading that file carried.
    "strip-list": r"\bstrip[-\s]?list\b",
    # "(working file — strip before public release)"; "scratch notes"; "notes to self".
    "working-scratch-file": r"\b(?:working|scratch|throwaway|temp(?:orary)?)\s+"
                            r"(?:file|files|notes?|copy|doc(?:ument)?|pad)\b"
                            r"|\bnotes?[-\s]to[-\s]self\b",
    # "must grep before public push" -- deferral of a privacy action.
    "before-public-push": r"\bbefore\b[^\n]{0,30}?\bpublic\s+(?:push|release|launch)\b",
    # "ship" was MEASURED as a false positive: SECURITY.md's in-scope list carries
    # "Hardcoded credentials or tokens (we do not ship any)" -- the object is credentials, not
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
    # A working DOCUMENT whose NAME announces what it is (audit / triage / handover notes,
    # a strip-list). Stated as a CLASS: an earlier revision spelled out one specific stem
    # here, which republished the identifier a history rewrite had just been run to remove.
    # The class is also the stronger detector -- the next working file will not reuse the
    # last one's name. The leading dot is OPTIONAL: an uppercase, undotted variant of the
    # same document is the same leak.
    # CALIBRATED like the content markers, and calibrated with THIS GATE'S OWN INSTRUMENT so
    # the number can be re-derived instead of taken on trust: over the distinct blob paths
    # walk_history() returns -- 306 of them on 2026-08-22 -- this pattern has 0 hits. Re-measure
    # with:  python3 -c "import importlib.util,pathlib;s=importlib.util.spec_from_file_location(
    #   'h','.release_checks/check_history_clean.py');m=importlib.util.module_from_spec(s);
    #   s.loader.exec_module(m);b=m.walk_history(pathlib.Path('.'));
    #   ps=sorted({p for x in b for p in x.paths});print(len(ps))"
    # (A raw `git rev-list --objects --all | cut -d' ' -f2- | sort -u` answers 358 instead: it
    # counts TREE paths too, which this marker never sees. An earlier revision of this comment
    # quoted 359, a figure NEITHER instrument returns -- the frozen self-measurement this file's
    # own docstring warns about.) "sweep" was REJECTED from the word list -- also 0 hits on the
    # same walk, but "parameter_sweep.md" is an ordinary research doc and this gate must not
    # train people to ignore it.
    "working-doc-by-name": r"(?:^|/)\.?[^/]*(?:audit|triage|handover|handoff|strip[-_]list)"
                           r"[^/]*\.(?:md|txt|org|rst)$",
    "notes-to-self": r"(?:^|/)(?:notes?[-_]to[-_]self|scratch|todo\.local)",
    "draft-doc": r"(?:^|/)[^/]*\.draft\.(?:md|txt|tex)$",
}

#: ABSOLUTE-PATH channel, and why it is separate from the two above.  EXEMPTIONS waves the
#: whole of ``.release_checks/`` past the marker channels ("the detector must not detect
#: itself"), which is correct for marker WORDS and wrong for ABSOLUTE PATHS: a gate that
#: hardcodes one machine's directory layout discloses that layout to every visitor AND
#: measures the maintainer's tree instead of the clone it ships in -- a green that says
#: nothing about the checkout under test.  Thirteen gates did exactly that.  Absolute paths
#: are not marker words, so this channel runs INSIDE the exempted directory.
#:
#: Matches an absolute path whose FIRST SEGMENT is not a standard system directory, and
#: only when a second segment follows.  A relative path ("docs/DATASET.md"), a URL
#: ("https://host/x"), a home-relative path ("~/.cache/x") and a regex alternation
#: ("(?:^|/)") all lack one of those properties and cannot trip it.
#:
#: The lookbehind also excludes the closers ``} ) ] > %``.  MEASURED, not guessed: without
#: them the first run flagged check_asset_coverage.py's ``f"{REPO}/scripts/..."`` -- an
#: INTERPOLATED root, which is the fix, not the defect.  A detector that fails on the
#: correct form teaches people to delete it.
MAINTAINER_ABS_PATH = re.compile(
    r"(?<![\w./~}\)\]>%-])/([A-Za-z][A-Za-z0-9._+-]*)(?=/[A-Za-z0-9._+-])")
SYSTEM_ROOTS = frozenset(
    "bin boot dev etc lib lib64 media mnt opt proc run sbin srv sys tmp usr var".split())

#: Path the --selftest fixture writes.  A NEUTRAL PLACEHOLDER, deliberately: it only has
#: to match one PATH_MARKER (``dot-local-infix``) for the fixture to do its job, and a
#: fixture that reproduces a real purged file's name republishes that name in the one file
#: whose whole purpose is to keep it out of the public tree.
SELFTEST_LEAK_PATH = ".scratch_notes.local.md"

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


def absolute_paths_in(text: str) -> List[Tuple[int, str]]:
    """Return [(line, path)] for every absolute non-system path literal in one file."""
    out: List[Tuple[int, str]] = []
    for m in MAINTAINER_ABS_PATH.finditer(text):
        if m.group(1) in SYSTEM_ROOTS:
            continue
        line = text.count("\n", 0, m.start()) + 1
        tail = text[m.start():m.start() + 80].split()[0].rstrip('",\')`')
        out.append((line, tail))
    return out


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
    path_ok = bool(markers_in(SELFTEST_LEAK_PATH, ""))
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

    # ---- CHECK D: the harness's own blind spot.  Runs inside the directory EXEMPTIONS
    # exempts, because what it looks for is not a marker word.  See MAINTAINER_ABS_PATH.
    abs_hits: List[str] = []
    gate_dir = root / ".release_checks"
    for f in (sorted(gate_dir.glob("*.py")) + sorted(gate_dir.glob("*.sh"))
              + sorted(gate_dir.glob("*.md"))):
        try:
            body = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line, hit in absolute_paths_in(body):
            abs_hits.append(f"{f.name}:{line}: {hit}")
    g.check("gate_harness_free_of_absolute_maintainer_paths", not abs_hits,
            "; ".join(abs_hits[:8])
            + (f" [+{len(abs_hits) - 8} more]" if len(abs_hits) > 8 else "")
            + " -- replace with an env var over a repo-relative default "
              "(GSSC_REPO / GSSC_ASSETS / GSSC_PAPER / GSSC_EXPERIMENTS); a hardcoded root "
              "publishes the maintainer's layout and makes a relocated clone audit a tree "
              "it is not running in")
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
        # SYNTHETIC content, assembled from the MARKER NAMES declared at the top of this
        # file -- not quoted from anything. Two lines are enough: the first trips
        # ``strip-before-public-release`` (and ``working-scratch-file``), the second trips
        # ``strip-list``. The fixture's job is to be SEEN by CONTENT_MARKERS, not to
        # resemble any particular file.
        (root / SELFTEST_LEAK_PATH).write_text(
            "# PLACEHOLDER working file (delete before public release)\n"
            "## PLACEHOLDER strip-list\n")
        run_git("add", "-A")
        run_git("commit", "-qm", "chore: add placeholder working notes")
        run_git("tag", "v9.9.9")
        run_git("rm", "-q", SELFTEST_LEAK_PATH)
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
    assert not (leaky_root / SELFTEST_LEAK_PATH).exists(), \
        "fixture is wrong: the leak must be absent from the worktree, or CHECK A is being " \
        "proved by CHECK B's evidence"
    leaky = run(leaky_root)
    expect("history_free_of_release_stop_markers", leaky, want_fail=True)
    # ...and the detail string must actually carry the actionable facts.
    det = dict((n, d) for n, ok, d in leaky.results
               if not ok).get("history_free_of_release_stop_markers", "")
    if "v9.9.9" in det and SELFTEST_LEAK_PATH in det:
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

    # --- FAULT 6: CHECK D.  Injected into a HEALTHY fixture, in that order: a gate dir
    # that uses the portable idiom must be GREEN first, otherwise a red "TRIPPED" below
    # would only prove the check is broken.  (A selftest arm that asserts today's defect
    # inverts the day the defect is fixed -- so the assertion is on the perturbation.)
    absroot = _scratch_repo(tmp, leak=False, name="abspath")
    gate_dir = absroot / ".release_checks"
    gate_dir.mkdir()
    fake = gate_dir / "check_fake.py"
    fake.write_text("REPO = Path(__file__).resolve().parents[1]\n")
    portable = dict((n, ok) for n, ok, _ in run(absroot).results).get(
        "gate_harness_free_of_absolute_maintainer_paths")
    if portable:
        print("  TRIPPED  abspath_control_portable_gate_dir_is_green")
    else:
        print("  MISSED   abspath_control_portable_gate_dir_is_green   (a gate using no "
              "absolute path is already red -- CHECK D is measuring something else)")
        missed += 1
    # Assembled from parts so this source file does not itself carry an absolute path
    # literal -- CHECK D scans the gate directory, and that includes this file.
    fake.write_text('REPO = Path("' + "/" + "example-root" + '/Some-Repo")\n')
    expect("gate_harness_free_of_absolute_maintainer_paths", run(absroot), want_fail=True)

    shutil.rmtree(tmp, ignore_errors=True)
    total = 10
    print(f"SELFTEST OK: {total - missed}/{total} checks provably fail when broken"
          if not missed else
          f"SELFTEST FAILED: {total - missed}/{total} checks provably fail when broken")
    return 1 if missed else 0


def main() -> int:
    print(f"[check_history_clean] repo={REPO}")
    return run(REPO).report()


if __name__ == "__main__":
    raise SystemExit(selftest() if "--selftest" in sys.argv else main())

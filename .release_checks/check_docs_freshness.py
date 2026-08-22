#!/usr/bin/env python3
"""GATE: the three version-bearing documents must agree with each other and with git.

THE DEFECTS THIS GATE EXISTS FOR (all measured 2026-08-20 at HEAD 07725af)
--------------------------------------------------------------------------
LINE NUMBERS IN THIS BLOCK ARE A DATED SNAPSHOT of the checkout named above, not
navigation. Several have already moved: follow the SYMBOL, the heading or the quoted
text, and re-derive the location with `grep -n`. Every check below RE-MEASURES the
live artefacts, so nothing here is load-bearing for a verdict.

F1  README.md:39 heads "### What's new".  Its newest bullet is
        "* **2026-05-26** -- Release **v2.1.0**: LMSCNet third-base support. ..."
    while `pyproject.toml:7` reads `version = "2.3.8"`.  TEN CHANGELOG releases
    (2.2.0, 2.3.0 .. 2.3.8) and ~79 days landed after the newest entry the front page
    admits to.  The README is the first thing a reviewer following the paper's link reads,
    and it describes a repository two minor versions behind the one they get.

F2  `CHANGELOG.md:32` reads "## [Unreleased]" with an EMPTY body, while
        `git diff --stat v2.3.8..HEAD` = 4 files changed, 298 insertions(+), 15 deletions(-)
    -- including a new public behaviour (eval refuses to start when scratch space cannot
    hold the predictions) and a new test file.  An empty [Unreleased] is a claim that
    nothing has happened since the tag.  Something has.

F3  `CITATION.cff:13-14` reads `version: 2.3.8` / `date-released: 2026-08-12`, but the
    CHANGELOG dates 2.3.8 to 2026-08-13.  Whoever cites this software cites the wrong day,
    and the two files disagree about the same release.

F4  (found while writing this gate, and the reason the tag rule is bidirectional)
    CHANGELOG documents `[1.1.0] - 2026-05-14` and `[1.0.0] - 2026-04-26`; NEITHER has a
    git tag.  The tags are v1.0.0-rc1 and v1.1.1.  So the CHANGELOG advertises two releases
    that cannot be checked out.  A one-directional rule ("every tag has an entry") would
    have been green on this.

WHAT IS PINNED, AND WHY IT IS A RELATIONSHIP AND NOT A CONSTANT
---------------------------------------------------------------
Nothing here hardcodes 2.3.8, 2026-08-13, or a release count.  Each check pins an
EQUALITY BETWEEN TWO ARTEFACTS that both move:
    README newest version  ==  pyproject version
    CHANGELOG version set  ==  git tag set
    CITATION version/date  ==  CHANGELOG newest entry's version/date
    worktree differs from newest tag  <->  [Unreleased] non-empty
    every vX.Y.Z in a CHANGELOG link definition  ==  a ref `git rev-parse` resolves
A constant would need editing at every release and would rot into a false green; the paper
harness has already had a gate rot exactly that way (a frozen self-measurement with no live
counterpart).

INSTRUMENT NOTES -- WHERE A NAIVE PARSER GOES WRONG HERE
--------------------------------------------------------
* README's "What's new" bullets NAME OLD VERSIONS IN PROSE ("Together with the v1.1.0
  JS3C-Net row ..."), so "newest" is a SEMVER MAX over the section, never the first match.
* CHANGELOG contains the heading "## [Pre-1.1.0 unreleased -- folded into 1.1.0]", which is
  not a release.  Version headings are matched as a bare semver inside the brackets, so
  that heading and "[Unreleased]" are both excluded by construction, not by a blocklist.
* Both em dash and hyphen appear as the heading separator across entries; the date is read
  with a separator-agnostic pattern.
* Pre-release tags (`v1.0.0-rc1`) are EXEMPT from needing a CHANGELOG entry, with the
  reason recorded in PRERELEASE_EXEMPT below -- an rc is not a release.  The exemption is
  scoped to the pre-release suffix; it does not excuse v1.1.1 or any final tag.
* "HEAD is ahead of the newest tag" is measured with `git diff --stat`, not `git log`: a
  tag moved onto HEAD, or commits that revert to the tagged tree, both mean the DOCUMENT
  has nothing to record.  Diffing the trees answers the question the [Unreleased] section
  is actually about.
* ...and it diffs the tag against the WORKTREE, not against HEAD.  During a fix cycle the
  work is uncommitted, so `tag..HEAD` reports zero drift while the author is midway through
  writing the very entry this check reads.  Same doctrine, and the same reason, as
  check_tag_parity's tag-vs-worktree comparison.
* THE [Unreleased] ARM IS BIDIRECTIONAL, and was not until 2026-08-22.  `not ahead or
  bool(unrel_body)` is structurally incapable of seeing a NON-EMPTY [Unreleased] over an
  UNCHANGED tree -- a release note describing work that exists nowhere.  That is the same
  defect class as a changelog entry asserting an edit that was never made, which this
  session found live in this repository.  Both directions now fail.
* LINK DEFINITIONS ARE READ.  The gate looked only at `## [x.y.z]` headings, so a
  `compare/...v1.1.0` URL against a tag that does not exist survived every run.  Every
  `vX.Y.Z` inside a `[label]: <url>` line must resolve via `git rev-parse`; `HEAD` is not
  matched, because it always resolves and pins nothing.  CHANGELOG.md's own trailing
  comment already PROMISED this -- an unenforced promise is what this harness exists for.

ROOTS, AND WHAT IS NOT PART OF THE PUBLIC RELEASE
-------------------------------------------------
Every root below is an environment variable with a repo-relative default, so this gate
measures the checkout it ships in rather than one particular machine.  Absolute paths
were hardcoded here once; a relocated clone then audited a tree it was not running in,
and the paths themselves disclosed the maintainer's local layout to every visitor.

    GSSC_REPO        the release checkout under test        default: this file's repository
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO = Path(os.environ.get("GSSC_REPO") or Path(__file__).resolve().parents[1])

#: Tags with a pre-release suffix are not releases and need no CHANGELOG entry.
# NOTE the missing \b: "v1.0.0-rc1" has no word boundary between "rc" and "1", so a
# trailing \b made this exemption a no-op and the gate reported v1.0.0-rc1 as an
# untagged release. Measured on the first run of this gate, 2026-08-20.
PRERELEASE_EXEMPT = re.compile(r"-(?:rc|alpha|beta|dev|pre)", re.I)

SEMVER = r"(\d+\.\d+\.\d+(?:-[0-9A-Za-z.\-]+)?)"


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


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True).stdout


def git_ok(root: Path, *args: str) -> bool:
    """True when the command exits 0. Used to ask git whether a ref exists, not what it says."""
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True).returncode == 0


#: A markdown link definition line: `[2.3.8]: https://...`. Matched structurally so the block
#: is found wherever it sits, rather than by assuming it is the last N lines of the file.
LINK_DEF = re.compile(r"^\[[^\]]+\]:\s*\S+")
#: Every `vX.Y.Z[-pre]` inside such a line -- both endpoints of a compare URL, and the tag in a
#: release URL. `HEAD` is deliberately not matched: it always resolves and pins nothing.
#: The pre-release suffix is DOT-SEPARATED IDENTIFIERS, not "any run of dots and alphanumerics".
#: Measured: the loose form swallowed a whole compare URL -- `v1.0.0-rc1...v1.1.1` matched as ONE
#: token and `git rev-parse` reported it unresolvable, a finding about the regex, not the file.
LINK_DEF_VERSION = re.compile(
    r"\bv\d+\.\d+\.\d+(?:-[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)?(?![0-9A-Za-z-])")


def semver_key(v: str) -> Tuple:
    base, _, pre = v.partition("-")
    nums = tuple(int(x) for x in base.split("."))
    # A pre-release sorts BEFORE its own final release (2.0.0-rc1 < 2.0.0).
    return nums + ((0, pre) if pre else (1, ""))


# --------------------------------------------------------------------------------------
# Parsers.  Each returns (value, "file:line") so every failure names a source location.
# --------------------------------------------------------------------------------------
def pyproject_version(root: Path) -> Tuple[Optional[str], str]:
    p = root / "pyproject.toml"
    if not p.is_file():
        return (None, "pyproject.toml (missing)")
    in_project = False
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if line.strip().startswith("["):
            in_project = line.strip() == "[project]"
        m = re.match(r'\s*version\s*=\s*"' + SEMVER + '"', line)
        if in_project and m:
            return (m.group(1), f"pyproject.toml:{i}")
    return (None, "pyproject.toml (no [project] version)")


def readme_whats_new(root: Path) -> Tuple[List[str], List[str], str, int]:
    """(versions, dates, 'file:line' of the heading, number of bullets)."""
    p = root / "README.md"
    if not p.is_file():
        return ([], [], "README.md (missing)", 0)
    lines = p.read_text().splitlines()
    start = next((i for i, l in enumerate(lines)
                  if re.match(r"\s*#{2,4}\s.*what'?s new", l, re.I)), None)
    if start is None:
        return ([], [], "README.md (no \"What's new\" heading)", 0)
    body: List[str] = []
    for l in lines[start + 1:]:
        if re.match(r"\s*#{1,4}\s", l) or l.strip() == "---":
            break
        body.append(l)
    text = "\n".join(body)
    # SEMVER MAX, not first-match: old versions are named in the prose of newer bullets.
    versions = sorted(set(re.findall(r"\bv?" + SEMVER, text)), key=semver_key)
    dates = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    bullets = sum(1 for l in body if l.strip().startswith("*"))
    return (versions, dates, f"README.md:{start + 1}", bullets)


def changelog_entries(root: Path) -> Tuple[List[Tuple[str, Optional[str], int]], str, int]:
    """[(version, date, line)], the Unreleased body, and its line number."""
    p = root / "CHANGELOG.md"
    if not p.is_file():
        return ([], "", 0)
    lines = p.read_text().splitlines()
    entries: List[Tuple[str, Optional[str], int]] = []
    unrel_line, unrel_body = 0, []
    collecting = False
    for i, l in enumerate(lines, 1):
        # A version heading is a BARE semver in brackets. "[Pre-1.1.0 unreleased -- folded
        # into 1.1.0]" and "[Unreleased]" fail that shape, so no blocklist is needed.
        m = re.match(r"##\s*\[" + SEMVER + r"\]\s*(.*)$", l)
        if m:
            collecting = False
            d = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(2))
            entries.append((m.group(1), d.group(1) if d else None, i))
            continue
        if re.match(r"##\s*\[Unreleased\]", l, re.I):
            collecting, unrel_line = True, i
            continue
        if l.startswith("## "):
            collecting = False
            continue
        if collecting:
            unrel_body.append(l)
    return (entries, "\n".join(unrel_body).strip(), unrel_line)


def citation_fields(root: Path) -> Tuple[Optional[str], Optional[str], str, str]:
    p = root / "CITATION.cff"
    if not p.is_file():
        return (None, None, "CITATION.cff (missing)", "CITATION.cff (missing)")
    ver = rel = None
    vloc = rloc = "CITATION.cff"
    for i, l in enumerate(p.read_text().splitlines(), 1):
        m = re.match(r"\s*version\s*:\s*['\"]?" + SEMVER, l)
        if m and ver is None:
            ver, vloc = m.group(1), f"CITATION.cff:{i}"
        m = re.match(r"\s*date-released\s*:\s*['\"]?(\d{4}-\d{2}-\d{2})", l)
        if m and rel is None:
            rel, rloc = m.group(1), f"CITATION.cff:{i}"
    return (ver, rel, vloc, rloc)


def tags(root: Path) -> List[str]:
    return [t for t in git(root, "tag").split() if t]


# --------------------------------------------------------------------------------------
def run(root: Path, gate: Optional[Gate] = None) -> Gate:
    g = gate or Gate()

    proj_ver, proj_loc = pyproject_version(root)
    rd_versions, rd_dates, rd_loc, rd_bullets = readme_whats_new(root)
    entries, unrel_body, unrel_line = changelog_entries(root)
    cff_ver, cff_date, cff_vloc, cff_rloc = citation_fields(root)
    all_tags = tags(root)

    # ---- CONTROL: every parser read something.  A gate whose input parsed to nothing must
    # be RED; this project has already shipped a green whose input was empty.
    empty = [n for n, v in (("pyproject version", proj_ver),
                            ("README What's new versions", rd_versions),
                            ("CHANGELOG version entries", entries),
                            ("CITATION.cff version", cff_ver),
                            ("git tags", all_tags)) if not v]
    if not g.check("parsers_read_something", not empty,
                   f"read nothing for: {empty} (locations tried: {proj_loc}, {rd_loc}, "
                   f"CHANGELOG.md, {cff_vloc}) -- every check below would be vacuous"):
        return g

    newest_readme = max(rd_versions, key=semver_key)
    newest_entry = max(entries, key=lambda e: semver_key(e[0]))

    # ---- F1: README's front-page release note vs the shipped version.
    stale_between = [e for e in entries
                     if semver_key(e[0]) > semver_key(newest_readme)]
    days = ""
    newest_readme_date = max(rd_dates) if rd_dates else None
    if newest_readme_date and newest_entry[1]:
        try:
            days = (f", {(date.fromisoformat(newest_entry[1]) - date.fromisoformat(newest_readme_date)).days}"
                    f" days")
        except ValueError:
            days = ""
    g.check("readme_whats_new_names_current_version",
            newest_readme == proj_ver,
            f"{rd_loc} \"What's new\" ({rd_bullets} bullets) stops at v{newest_readme}"
            f"{' (' + newest_readme_date + ')' if newest_readme_date else ''} while "
            f"{proj_loc} ships {proj_ver}: {len(stale_between)} CHANGELOG release(s) "
            f"missing from the front page{days} -- "
            f"{', '.join(v for v, _, _ in sorted(stale_between, key=lambda e: semver_key(e[0])))}")

    # ---- F4/F2: CHANGELOG <-> git tags, BOTH directions.
    tag_versions = {t.lstrip("v"): t for t in all_tags}
    entry_versions = {v: ln for v, _, ln in entries}
    no_tag = sorted((v for v in entry_versions if v not in tag_versions), key=semver_key)
    no_entry = sorted((tv for v, tv in tag_versions.items()
                       if v not in entry_versions and not PRERELEASE_EXEMPT.search(tv)),
                      key=lambda t: semver_key(t.lstrip("v")))
    g.check("changelog_and_tags_are_in_bijection",
            not no_tag and not no_entry,
            "; ".join(filter(None, [
                (f"CHANGELOG documents {len(no_tag)} release(s) with NO git tag: "
                 + ", ".join(f"{v} (CHANGELOG.md:{entry_versions[v]})" for v in no_tag))
                if no_tag else "",
                (f"git has {len(no_entry)} tag(s) with NO CHANGELOG entry: "
                 + ", ".join(no_entry)) if no_entry else ""])))

    # ---- F3: CITATION.cff vs the CHANGELOG's newest entry (and the shipped version).
    cff_bad: List[str] = []
    if cff_ver != newest_entry[0] or cff_ver != proj_ver:
        cff_bad.append(f"{cff_vloc} says version {cff_ver}, CHANGELOG.md:{newest_entry[2]} "
                       f"says {newest_entry[0]}, {proj_loc} says {proj_ver}")
    if newest_entry[1] and cff_date != newest_entry[1]:
        cff_bad.append(f"{cff_rloc} says date-released {cff_date}, but "
                       f"CHANGELOG.md:{newest_entry[2]} dates {newest_entry[0]} to "
                       f"{newest_entry[1]} -- a citation of this software cites the "
                       f"wrong day")
    g.check("citation_agrees_with_changelog", not cff_bad, "; ".join(cff_bad))

    # ---- F2: unreleased work must be recorded, and recorded work must exist.
    newest_tag = max(all_tags, key=lambda t: semver_key(t.lstrip("v")))
    # WORKTREE, not `tag..HEAD`. During a fix cycle the work is in the worktree and not yet
    # committed, so `tag..HEAD` reports zero drift while the author is midway through writing
    # the very entry this check reads. Diffing the tag against the WORKING TREE answers the
    # question [Unreleased] is actually about -- the same doctrine check_tag_parity states for
    # the same reason. `git diff --stat <tag>` with no second ref does exactly that.
    stat = git(root, "diff", "--stat", newest_tag).strip().splitlines()
    ahead = bool(stat)
    names = [l for l in git(root, "diff", "--name-only", newest_tag).splitlines() if l.strip()]
    # BIDIRECTIONAL, and it was not. `not ahead or bool(unrel_body)` is structurally incapable
    # of seeing a NON-EMPTY [Unreleased] with ZERO drift -- a section describing work that
    # exists nowhere in the tree. The docstring's own line "HEAD ahead of newest tag ->
    # [Unreleased] non-empty" reads as an implication in one direction; the defect class it
    # guards runs both ways, and a release note asserting a repo state that does not exist is
    # exactly the failure this whole harness was built after.
    if ahead and not unrel_body:
        g.check("unreleased_section_records_head_drift", False,
                f"the worktree is ahead of {newest_tag} ({stat[-1].strip() if stat else '?'}; "
                f"{', '.join(names[:5])}) but CHANGELOG.md:{unrel_line} [Unreleased] is EMPTY "
                f"-- the document claims nothing has happened since the tag")
    elif unrel_body and not ahead:
        g.check("unreleased_section_records_head_drift", False,
                f"CHANGELOG.md:{unrel_line} [Unreleased] describes {len(unrel_body.split())} "
                f"words of unreleased work, but the worktree is byte-identical to {newest_tag} "
                f"-- the section records changes that exist in no tree, so a reader diffing the "
                f"tag against this checkout finds nothing the entry describes")
    else:
        g.check("unreleased_section_records_head_drift", True, "")

    # ---- F5: every version named in a CHANGELOG link definition must be checkoutable.
    # The gate never read the link-definition block at all, which is how a compare URL against
    # a non-existent `v1.1.0` survived. The file's own trailing comment now PROMISES that
    # "every vX.Y.Z named in a link definition on this page resolves via `git rev-parse`" --
    # an unenforced promise is the class this harness exists for, so it is enforced here.
    unresolvable: List[str] = []
    defs = 0
    cl_path = root / "CHANGELOG.md"
    cl_text = cl_path.read_text(encoding="utf-8", errors="replace") if cl_path.is_file() else ""
    for n, line in enumerate(cl_text.splitlines(), 1):
        if not LINK_DEF.match(line):
            continue
        defs += 1
        for ver in set(LINK_DEF_VERSION.findall(line)):
            if not git_ok(root, "rev-parse", "--verify", "--quiet", f"{ver}^{{commit}}"):
                unresolvable.append(f"CHANGELOG.md:{n} links {ver}, which `git rev-parse` "
                                    f"cannot resolve -- that compare URL is a 404")
    g.check("changelog_link_definitions_resolve",
            bool(defs) and not unresolvable,
            "; ".join(unresolvable[:4]) if unresolvable else
            "no `[x.y.z]: <url>` link definition found in CHANGELOG.md -- this check measured "
            "nothing, which is not the same as passing")
    return g


# --------------------------------------------------------------------------------------
# Selftest: a full fixture repo in $TMPDIR (git tags included), one fault per check.
# --------------------------------------------------------------------------------------
FIX_README = """# X

### What's new

* **2026-08-13** -- Release **v1.2.0**: things. Supersedes the v1.0.0 row.
* **2026-04-01** -- Release **v1.0.0**: first.

---

## Method
"""

FIX_CHANGELOG = """# Changelog

## [Unreleased]

- something landed

## [1.2.0] - 2026-08-13

### Fixed
- stuff

## [Pre-1.2.0 unreleased -- folded into 1.2.0]

- noise that is not a release

## [1.0.0] - 2026-04-01

- first

[Unreleased]: https://example.invalid/compare/v1.2.0...HEAD
[1.2.0]: https://example.invalid/compare/v1.0.0...v1.2.0
[1.0.0]: https://example.invalid/releases/tag/v1.0.0
"""

FIX_CFF = """cff-version: 1.2.0
title: "X"
version: 1.2.0
date-released: 2026-08-13
"""

FIX_PYPROJECT = """[build-system]
requires = ["hatchling"]

[project]
name = "x"
version = "1.2.0"
"""


def _fixture(tmp: Path, name: str = "fx") -> Path:
    root = tmp / name
    root.mkdir(parents=True)
    (root / "README.md").write_text(FIX_README)
    (root / "CHANGELOG.md").write_text(FIX_CHANGELOG)
    (root / "CITATION.cff").write_text(FIX_CFF)
    (root / "pyproject.toml").write_text(FIX_PYPROJECT)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"}
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True,
                   capture_output=True, env=env)
    for tag, extra in (("v1.0.0", "a"), ("v1.2.0", "b")):
        (root / "file.txt").write_text(extra)
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True,
                       capture_output=True, env=env)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", tag], check=True,
                       capture_output=True, env=env)
        subprocess.run(["git", "-C", str(root), "tag", tag], check=True,
                       capture_output=True, env=env)
    # The fixture's [Unreleased] section has a body, so the fixture must ALSO have drift past
    # the newest tag or the bidirectional arm rightly fails the baseline. An UNCOMMITTED file
    # is used deliberately: it is the state a maintainer is in while writing that very entry,
    # and it pins that the drift measurement reads the WORKTREE and not just `tag..HEAD`.
    (root / "unstaged_work.txt").write_text("work in progress\n")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True,
                   capture_output=True, env=env)
    return root


def selftest() -> int:
    missed = 0
    tmp = Path(tempfile.mkdtemp(prefix="docs_selftest_"))

    def expect(label: str, root: Path, check: str, want_fail: bool) -> None:
        nonlocal missed
        got = dict((n, ok) for n, ok, _ in run(root).results).get(check)
        if got is None:
            print(f"  MISSED   {check}   ({label}: check never ran)")
            missed += 1
        elif (not got) == want_fail:
            print(f"  TRIPPED  {check}")
        else:
            print(f"  MISSED   {check}   ({label}: expected "
                  f"{'FAIL' if want_fail else 'PASS'})")
            missed += 1

    def mutate(name: str, rel: str, old: str, new: str) -> Path:
        root = _fixture(tmp, name)
        p = root / rel
        before = p.read_text()
        after = before.replace(old, new, 1)
        # The classic vacuous selftest: a replace against a pattern that has drifted is a
        # silent no-op and the "fault" is never injected.  Assert the perturbation.
        assert after != before, f"FIXTURE DRIFT: {rel} does not contain {old!r}"
        p.write_text(after)
        return root

    # --- CONTROL: the untouched fixture is green everywhere, so each TRIPPED below is
    # attributable to its own fault and not to a broken fixture.
    base = _fixture(tmp, "base")
    bad = [n for n, ok, d in run(base).results if not ok]
    if bad:
        print(f"  MISSED   control_fixture_is_green   (failing: {bad})")
        missed += 1
    else:
        print("  TRIPPED  control_fixture_is_green")

    # --- FAULT 1: README's newest bullet falls behind the shipped version.
    expect("README stops at 1.0.0",
           mutate("f1", "README.md",
                  "* **2026-08-13** -- Release **v1.2.0**: things. Supersedes the "
                  "v1.0.0 row.\n", ""),
           "readme_whats_new_names_current_version", True)

    # --- FAULT 1b: the SEMVER-MAX rule.  Reordering the bullets so an OLD version is
    # first must NOT trip the check -- a first-match parser would fail here, and this
    # fixture is why the parser takes a max.
    reordered = _fixture(tmp, "f1b")
    p = reordered / "README.md"
    lines = p.read_text().splitlines(keepends=True)
    i = next(k for k, l in enumerate(lines) if "v1.2.0" in l)
    j = next(k for k, l in enumerate(lines) if "v1.0.0" in l and "Supersedes" not in l)
    lines[i], lines[j] = lines[j], lines[i]
    p.write_text("".join(lines))
    expect("bullets reordered, newest not first", reordered,
           "readme_whats_new_names_current_version", False)

    # --- FAULT 2: a CHANGELOG release with no tag (defect F4's shape).
    expect("changelog release 1.1.0 has no tag",
           mutate("f2", "CHANGELOG.md", "## [1.0.0] - 2026-04-01",
                  "## [1.1.0] - 2026-05-01\n\n- untagged\n\n## [1.0.0] - 2026-04-01"),
           "changelog_and_tags_are_in_bijection", True)

    # --- FAULT 2b: the other direction -- a tag with no CHANGELOG entry.
    f2b = _fixture(tmp, "f2b")
    subprocess.run(["git", "-C", str(f2b), "tag", "v1.3.0"], check=True,
                   capture_output=True)
    expect("tag v1.3.0 has no entry", f2b, "changelog_and_tags_are_in_bijection", True)

    # --- FAULT 2c: a PRE-RELEASE tag must stay exempt (scope of the exemption).
    f2c = _fixture(tmp, "f2c")
    subprocess.run(["git", "-C", str(f2c), "tag", "v1.4.0-rc1"], check=True,
                   capture_output=True)
    expect("rc tag is exempt", f2c, "changelog_and_tags_are_in_bijection", False)

    # --- FAULT 3: CITATION date drifts by one day (defect F3, exactly).
    expect("CITATION date-released off by one day",
           mutate("f3", "CITATION.cff", "date-released: 2026-08-13",
                  "date-released: 2026-08-12"),
           "citation_agrees_with_changelog", True)

    # --- FAULT 3b: CITATION version drifts.
    # The needle is "\nversion:", not "version:": the fixture's FIRST line is
    # "cff-version: 1.2.0", which contains "version: 1.2.0" as a substring, so the naive
    # needle mutated the CFF SCHEMA VERSION and left the release version untouched -- the
    # fault was never injected and the selftest reported a MISSED that was its own bug.
    expect("CITATION version behind",
           mutate("f3b", "CITATION.cff", "\nversion: 1.2.0", "\nversion: 1.1.0"),
           "citation_agrees_with_changelog", True)

    # --- FAULT 4: work past the newest tag with an empty [Unreleased].
    f4 = _fixture(tmp, "f4")
    (f4 / "new_file.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(f4), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(f4), "commit", "-qm", "past the tag"], check=True,
                   capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                        "PATH": "/usr/bin:/bin"})
    cl = f4 / "CHANGELOG.md"
    b = cl.read_text()
    a = b.replace("## [Unreleased]\n\n- something landed\n", "## [Unreleased]\n")
    assert a != b, "FIXTURE DRIFT: [Unreleased] body not found"
    cl.write_text(a)
    expect("commits past the tag, empty [Unreleased]", f4,
           "unreleased_section_records_head_drift", True)

    # --- FAULT 4b: THE OTHER DIRECTION -- a non-empty [Unreleased] with NO drift at all.
    # `not ahead or bool(unrel_body)` could not express this: the section describes work that
    # exists in no tree, and every arm of the old test was satisfied. Removing the fixture's
    # only drift (its uncommitted file) leaves the worktree byte-identical to v1.2.0 while
    # [Unreleased] still says "- something landed".
    f4b = _fixture(tmp, "f4b")
    (f4b / "unstaged_work.txt").unlink()
    subprocess.run(["git", "-C", str(f4b), "add", "-A"], check=True, capture_output=True)
    assert not git(f4b, "diff", "--stat", "v1.2.0").strip(), \
        "FIXTURE DRIFT: removing the scratch file did not make the worktree match the tag"
    expect("non-empty [Unreleased] with an unchanged tree", f4b,
           "unreleased_section_records_head_drift", True)

    # --- FAULT 6: a link definition naming a tag that does not exist (the v1.1.0 compare URL
    # shape). The gate never looked at the link-definition block at all.
    expect("link definition names a non-existent tag",
           mutate("f6", "CHANGELOG.md",
                  "[1.2.0]: https://example.invalid/compare/v1.0.0...v1.2.0",
                  "[1.1.0]: https://example.invalid/compare/v1.0.0...v1.1.0"),
           "changelog_link_definitions_resolve", True)

    # --- FAULT 6b: ...and anti-vacuity for it. No link definitions at all must be RED, not a
    # silent pass -- the failure mode two gates in this project have already shipped.
    expect("no link definitions at all",
           mutate("f6b", "CHANGELOG.md",
                  "[Unreleased]: https://example.invalid/compare/v1.2.0...HEAD\n"
                  "[1.2.0]: https://example.invalid/compare/v1.0.0...v1.2.0\n"
                  "[1.0.0]: https://example.invalid/releases/tag/v1.0.0\n",
                  "(the link-definition block deleted entirely)\n"),
           "changelog_link_definitions_resolve", True)

    # --- FAULT 5: anti-vacuity.  A README with no "What's new" section at all must be RED,
    # not silently skipped.
    expect("no What's new heading",
           mutate("f5", "README.md", "### What's new", "### Nothing here"),
           "parsers_read_something", True)

    shutil.rmtree(tmp, ignore_errors=True)
    total = 13
    print(f"SELFTEST OK: {total - missed}/{total} checks provably fail when broken"
          if not missed else
          f"SELFTEST FAILED: {total - missed}/{total} checks provably fail when broken")
    return 1 if missed else 0


def main() -> int:
    print(f"[check_docs_freshness] repo={REPO}")
    return run(REPO).report()


if __name__ == "__main__":
    raise SystemExit(selftest() if "--selftest" in sys.argv else main())

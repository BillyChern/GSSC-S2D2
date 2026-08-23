#!/usr/bin/env python3
"""GATE: SECURITY.md's checkpoint verification instruction must be executable.

THE DEFECT THIS GATE EXISTS FOR
-------------------------------
SECURITY.md's trust model is the repository's answer to the fact that it loads checkpoints with
`torch.load(..., weights_only=False)` -- i.e. that a hostile checkpoint is arbitrary code
execution. Its mitigation is one sentence:

    SECURITY.md, under its trust model (grep "Verify before loading", not a line number --
    this pointer read ":63" / ":66" and the file has been rewritten since):

        "Published checkpoints will ship with SHA256 hashes documented in
         [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) on release. Verify before loading:"
           sha256sum data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors

Measured on 2026-08-20:

  * `docs/MODEL_ZOO.md` contains ZERO 64-hex strings.
  * A repo-wide grep for `[0-9a-f]{64}` finds them in exactly one file, `uv.lock` (Python wheel
    hashes). Not one checkpoint hash is published anywhere in the repository.
  * The command as printed has nothing to compare against: `sha256sum <file>` PRINTS a digest,
    it does not CHECK one. A user who runs it gets 64 characters and no verdict.

So the only defence offered against the deserialization attack vector the same file lists as
in-scope is a pointer to a table that does not exist. That is worse than silence: a reader who
follows it concludes the verification exists and that they merely failed to find it.

TWO ARMS ADDED 2026-08-22, EACH FOR A DEFECT THAT SHIPPED GREEN
---------------------------------------------------------------
  published_metadata_hashes_match_assets
      `released_leaves()` drops `*.txt` as "MANIFEST / checksums metadata, not weights", which
      is right for the who-has-a-hash checks and wrong for the does-it-match check: nothing
      keyed docs/MODEL_ZOO.md's `MANIFEST.txt` row to anything, so when MANIFEST.txt was
      regenerated and its digest moved from `4d05afd9dc10...` to `87a91895447e...`, the doc
      kept the old value and this gate printed PASS. SECURITY.md tells readers a FAILED line
      means "do not load that file"; a silently wrong row in that table is worse than none.

  verification_command_exits_zero
      The old arm asserted `bool(cmds)` -- that some doc contained the STRING
      `sha256sum -c <file>`. Three byte-identical broken commands shipped green under that
      rule, because the string was right and the CWD was wrong: the paths inside checksums.txt
      are relative to the download root, so the same command run one directory up FAILS every
      line. Each documented command is now EXECUTED, in a miniature payload laid out where the
      DOWNLOADER puts it, and must exit 0. Measured while writing it: rewriting
      `cd data/checkpoints &&` to `cd data &&` leaves the string arm green and fails this one.

AND FOR A WHILE THE FILE THAT SHIPPED WAS THE WRONG ONE
-------------------------------------------------------
The defect this section was written for: the assets tree once carried FOUR manifest/checksum
files, and the one that actually reached a user covered nothing they downloaded:

  GSSC-S2D2-assets/checksums.txt              62 lines, covering every released leaf -- but it
                                              sat at the assets ROOT, and the upload procedure
                                              uploads `checkpoints/` only, so it never shipped.
  GSSC-S2D2-assets/checkpoints/checksums.txt  14 lines, every one a flat legacy `.pt` the assets
                                              README calls "not part of the public release".
                                              THIS one was inside the payload, so `sha256sum -c`
                                              at the download root failed 14/14 and verified
                                              nothing.

Two further facts measured at the time, each of which became its own check:

  * `checkpoints/bev/bev_s2d2_scpnet/{config.json,model.pt,model.safetensors}` was in the payload
    and in NEITHER checksums file, NEITHER MANIFEST, and no doc. An unlisted `.pt` inside a
    release payload is precisely the artefact the trust model is about.
    THE ASSETS TREE IS LIVE: `checkpoints/pyramid/_superseded_20260820/` and three rewritten
    pyramid `config.json` files appeared DURING the writing of this gate, and the uncovered
    count moved from 2 to 9 between two runs an hour apart. That is the argument for deriving
    the released set from the tree instead of listing it.
  * the root `checksums.txt` entry for `checkpoints/MANIFEST.txt` was STALE: recorded
    de582887..., actual 9b7d57ba.... A checksums file that is wrong about a file it does list is
    a worse instrument than one that omits it, because `-c` failures get dismissed as noise.

RE-MEASURED 2026-08-23, all four resolved: the two root files are retired under
`_superseded_20260820/` (as `*.root-copy`) and no longer exist at the assets root;
`checkpoints/checksums.txt` is the single authority at 51 rows with ZERO flat `.pt` entries; and
`bev/bev_s2d2_scpnet/` now has all three of its files listed there. This block is kept in the
past tense rather than deleted because the checks below are what hold that state.

HOW THIS IS MEASURED
--------------------
Nothing is pinned to a literal digest. The relationship is: for every released leaf, the digest
recorded in the assets checksums file must (a) also appear in a repository doc and (b) equal the
digest recomputed from the bytes on disk. Rewriting a checkpoint and re-recording its hash keeps
this gate green; forgetting either half does not.

THE RELEASED SET IS DERIVED, NOT LISTED. It is every file in the upload payload except the
metadata `*.txt` and the depth-1 flat `*.pt` legacy aliases the assets README excludes. A hand
listed set would silently stop covering a checkpoint subdir added next week -- which is exactly
how `bev/bev_s2d2_scpnet/` came to exist in the payload and in no manifest.

THE RECOMPUTE CAP. Verifying 10 GB of safetensors on every run is not affordable in a gate that
runs after every edit, so `assets_checksums_are_current` recomputes only files under
`HASH_CAP_BYTES` (1 MB: all 22 config.json / manifest entries) unless
`RELEASE_CHECK_FULL_HASH=1` is set, in which case it hashes everything. THE CAP IS A HOLE AND IS
NAMED AS ONE: a stale safetensors digest is invisible by default. Run the full form once before
publishing; the check's failure detail always reports how many files it actually hashed.

STATUS ON 2026-08-20: FAILS, by design. 7 of 9 checks fail on the shipped artefacts.

ROOTS, AND WHAT IS NOT PART OF THE PUBLIC RELEASE
-------------------------------------------------
Every root below is an environment variable with a repo-relative default, so this gate
measures the checkout it ships in rather than one particular machine.  Absolute paths
were hardcoded here once; a relocated clone then audited a tree it was not running in,
and the paths themselves disclosed the maintainer's local layout to every visitor.

    GSSC_REPO        the release checkout under test        default: this file's repository
    GSSC_ASSETS      the asset staging bundle               default: <repo>/../GSSC-S2D2-assets

THE ASSET STAGING BUNDLE IS NOT PART OF THE PUBLIC RELEASE.
It is a maintainer working tree; a clone of this repository does not contain it, and the
released artefacts are distributed separately (docs/DATASET.md, docs/MODEL_ZOO.md).
A gate that needs one and cannot find it FAILS rather than passing: "the artefact is
not here" is not evidence that it is correct.  Point the variable at your own copy,
or skip the gate.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

REPO = Path(os.environ.get("GSSC_REPO") or Path(__file__).resolve().parents[1])
ASSETS = Path(os.environ.get("GSSC_ASSETS") or REPO.parent / "GSSC-S2D2-assets")
SECURITY = REPO / "SECURITY.md"
ASSETS_CHECKSUMS = None  # resolved by _locate() below, after ASSETS is defined
PAYLOAD = ASSETS / "checkpoints"          # what `huggingface-cli upload <repo> checkpoints/` sends

HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
#: `sha256sum -c <file>` / `--check`: the only form that yields a VERDICT rather than a digest.
VERIFY_CMD = re.compile(r"sha256sum\s+(?:-c\b|--check\b)[^\n]*", re.I)
#: A SECURITY.md sentence that delegates the hashes to another document. Run over
#: WHITESPACE-COLLAPSED text: the shipped sentence is hard-wrapped between "documented in" and
#: "[docs/MODEL_ZOO.md](...)", and a line-oriented pattern reported "the trust model states no
#: verification path at all" -- a much graver finding than the true one, from an instrument bug.
HASH_DELEGATION = re.compile(
    r"SHA-?256.{0,120}?(?:\[[^\]]*\]\(([^)\s]+\.md)\)|`([^`\s]+\.md)`)", re.I | re.S)

HASH_CAP_BYTES = 1 << 20
FULL_HASH = os.environ.get("RELEASE_CHECK_FULL_HASH") == "1"

#: Files scanned for published digests. `uv.lock` is excluded on purpose: it is full of 64-hex
#: wheel hashes and would make every "a hash is published" check pass for the wrong reason --
#: the single most likely way this gate could have been fooled. `external/` is vendored.
#: `.github/ISSUE_TEMPLATE/*.md` is listed separately because `.github/*.md` does not reach
#: it, and the reproducibility template prints the same `sha256sum -c` command a reader is
#: told to run -- an instruction that was outside every check here until 2026-08-22.
DOC_GLOBS = ("*.md", "docs/*.md", "examples/*.ipynb", "assets/*.md", ".github/*.md",
             ".github/ISSUE_TEMPLATE/*.md")


class Leaf(NamedTuple):
    rel: str                  # path relative to the ASSETS root, e.g. "checkpoints/pyramid/..."
    size: int


# --------------------------------------------------------------------------- readers


def _locate(name: str) -> Path:
    """Find `name` anywhere in the asset bundle, preferring the upload payload.

    It used to be pinned to the bundle root. It now lives inside checkpoints/ so that
    it actually ships; hardcoding either location makes the gate report "missing" for
    a file that is present and correct, which is a gate defect, not a release defect.
    """
    inside = ASSETS / "checkpoints" / name
    if inside.exists():
        return inside
    root = ASSETS / name
    if root.exists():
        return root
    hits = sorted(ASSETS.rglob(name))
    hits = [h for h in hits if "_superseded_" not in h.as_posix()]
    return hits[0] if hits else inside


ASSETS_CHECKSUMS = _locate("checksums.txt")
CHECKSUM_BASE = ASSETS_CHECKSUMS.parent   # sha256sum -c resolves relative to this


def _payload_rel(rel: str) -> str:
    """A leaf path expressed relative to the checksums file's own directory.

    Leaves are tracked ASSETS-relative ("checkpoints/bev/x"), but a checksums file
    that ships INSIDE the payload lists them payload-relative ("bev/x") -- which is
    what `sha256sum -c` needs at the download root. Comparing the two spellings
    directly made every entry look absent.
    """
    pre = CHECKSUM_BASE.relative_to(ASSETS).as_posix()
    if pre in (".", "") or not rel.startswith(pre + "/"):
        return rel
    return rel[len(pre) + 1:]


def read_docs(repo: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pat in DOC_GLOBS:
        for p in sorted(repo.glob(pat)):
            if "external" in p.parts:
                continue
            out[str(p.relative_to(repo))] = p.read_text(encoding="utf-8", errors="replace")
    return out


def parse_checksums(text: str, prefix: str = "") -> Dict[str, str]:
    """`<hex>  <path>` lines -> {path: hex}, paths optionally re-rooted with `prefix`.

    The two checksum files use DIFFERENT roots (the assets root vs. the checkpoints dir), and
    comparing them without re-rooting would report every entry as missing from the other. That
    normalisation is the reason this takes a prefix instead of being inlined twice.
    """
    out: Dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^([0-9a-f]{64})\s+\*?(\S.*)$", line.strip())
        if m:
            out[prefix + m.group(2).strip()] = m.group(1)
    return out


def released_leaves(payload: Path, assets_root: Path) -> List[Leaf]:
    """Every file the upload payload delivers that is a CHECKPOINT.

    Excluded, with the reason each exclusion is legitimate:
      * `*.txt`  -- MANIFEST / checksums metadata, not weights.
      * depth-1 `*.pt` -- the flat legacy v1.0 aliases the assets README declares "not part of
        the public release". Note the depth qualifier: `bev/bev_s2d2_scpnet/model.pt` is at
        depth 3 and IS a released file, and a blanket `*.pt` exclusion would have hidden the
        unlisted-checkpoint finding this gate exists to report.
    """
    out: List[Leaf] = []
    for p in sorted(payload.rglob("*")):
        if not p.is_file():
            continue
        rel_in_payload = p.relative_to(payload)
        if p.suffix == ".txt":
            continue
        if p.suffix == ".pt" and len(rel_in_payload.parts) == 1:
            continue
        out.append(Leaf(str(p.relative_to(assets_root)), p.stat().st_size))
    return out


def recompute(leaves: Sequence[Leaf], recorded: Dict[str, str],
              assets_root: Path) -> Dict[str, Optional[str]]:
    """{rel: digest or None-if-skipped} for every path the checksums file records.

    Everything recorded is a candidate, not just the released leaves -- the stale entry actually
    present in the shipped file is `checkpoints/MANIFEST.txt`, which is metadata and would have
    been filtered out by a released-only scan.
    """
    out: Dict[str, Optional[str]] = {}
    for rel in recorded:   # keys are relative to CHECKSUM_BASE
        p = CHECKSUM_BASE / rel        # recorded keys are payload-relative
        if not p.is_file():
            out[rel] = None
            continue
        if not FULL_HASH and p.stat().st_size > HASH_CAP_BYTES:
            out[rel] = None
            continue
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[rel] = h.hexdigest()
    return out


def published_hashes(docs: Dict[str, str]) -> List[Tuple[str, int, str, str]]:
    """(doc, line, hex, the rest of the line) for every 64-hex digest published in the docs."""
    out: List[Tuple[str, int, str, str]] = []
    for doc, text in sorted(docs.items()):
        for n, line in enumerate(text.splitlines(), 1):
            for m in HEX64.finditer(line):
                out.append((doc, n, m.group(0), line.replace(m.group(0), "").strip()))
    return out


#: Files whose verification commands are EXECUTED, and the size ceiling for a fixture file.
#: Only small recorded entries are materialised: the point under test is CWD RESOLUTION, and
#: copying 4.9 GB to prove a relative path would make the gate unrunnable.
VERIFY_FIXTURE_MAX_BYTES = 1 << 20
VERIFY_FIXTURE_SAMPLE = 6

#: A ```fence``` opener/closer. Only a command INSIDE one is an instruction; the same string in
#: prose is a mention. Measured need: CHANGELOG.md narrates "it now documents the `sha256sum -c`
#: path against the `checksums.txt` that ships", which VERIFY_CMD matches and which is not a
#: command anybody can run.
FENCE = re.compile(r"^\s*(?:```|~~~)")


def fenced_verify_commands(docs: Dict[str, str]) -> List[Tuple[str, int, str]]:
    """(doc, first-line, command) for every `sha256sum -c` command in a fenced code block.

    BACKSLASH CONTINUATIONS ARE JOINED FIRST. The shipped one-file form is written over two
    lines; reading them separately yielded the fragment `checksums.txt | sha256sum -c -`, which
    bash reports as "checksums.txt: command not found" -- a finding about the parser, not about
    the doc. Same class of bug as `_pattern_strings` in check_asset_namespace.
    """
    out: List[Tuple[str, int, str]] = []
    for doc, text in sorted(docs.items()):
        inside = False
        pending: List[str] = []
        start = 0
        for n, line in enumerate(text.splitlines(), 1):
            if FENCE.match(line):
                inside = not inside
                pending = []
                continue
            if not inside:
                continue
            stripped = line.strip().lstrip("$ ").strip()
            if not pending:
                start = n
            if stripped.endswith("\\"):
                pending.append(stripped[:-1].strip())
                continue
            full = " ".join(pending + [stripped]).strip()
            pending = []
            if VERIFY_CMD.search(full):
                out.append((doc, start, full))
    return out


def _fixture_for(command: str) -> Optional[Tuple[Path, str]]:
    """Build a miniature download tree for one documented command; -> (cwd_root, shell_cmd).

    The tree reproduces the REPO-RELATIVE path the command's own `cd` names, so a command that
    cds to the wrong directory fails here exactly as it fails for a reader. The checksums file
    is the SHIPPED one, trimmed to the entries actually materialised -- trimming keeps the
    recorded PATHS untouched, which is the thing under test, while keeping the fixture small.
    """
    # THE DOWNLOAD ROOT IS NOT TAKEN FROM THE COMMAND. Reading it from the command's own `cd`
    # made the fixture move with the fault and the check could not fail: rewriting
    # `cd data/checkpoints` to `cd data` also built the payload under `data/`, so the wrong
    # command still exited 0. The tree is laid out where the DOWNLOADER puts it -- the repo's
    # own `data/checkpoints` -- and the command's `cd` is then the thing under test.
    root_rel = (REPO / "data" / "checkpoints")
    if not root_rel.exists():
        return None
    dl_rel = root_rel.relative_to(REPO).as_posix()
    ck_file = PAYLOAD / "checksums.txt"
    if not ck_file.is_file():
        return None
    recorded = parse_checksums(ck_file.read_text(encoding="utf-8"))
    # Any path literal the command names must be in the sample, or a `grep <path> | sha256sum -c -`
    # form would run against an empty stream and prove nothing about that path.
    named = [k for k in recorded if k in command]
    small = sorted((k for k in recorded
                    if (PAYLOAD / k).is_file()
                    and (PAYLOAD / k).stat().st_size <= VERIFY_FIXTURE_MAX_BYTES),
                   key=lambda k: (PAYLOAD / k).stat().st_size)
    sample = [k for k in dict.fromkeys(named + small[:VERIFY_FIXTURE_SAMPLE])
              if (PAYLOAD / k).is_file()]
    if not sample:
        return None
    root = Path(tempfile.mkdtemp(prefix="gssc-verify-"))     # honours TMPDIR; never /tmp here
    dl = root / dl_rel
    dl.mkdir(parents=True, exist_ok=True)
    digests: Dict[str, str] = {}
    for k in sample:
        dst = dl / k
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (PAYLOAD / k).stat().st_size <= VERIFY_FIXTURE_MAX_BYTES:
            shutil.copyfile(PAYLOAD / k, dst)
            digests[k] = recorded[k]
        else:
            # A path the command NAMES explicitly but that is too large to copy (the shipped
            # one-file example names a 133 MiB safetensors). A STAND-IN carrying its own
            # recomputed digest is written at the same relative path. What that proves is what
            # this check is for: the command, run from the cwd the doc names, RESOLVES the
            # recorded path and prints OK. Whether the REAL digest matches is a different
            # question and is owned by assets_checksums_are_current, which hashes real bytes.
            body = f"stand-in for {k}\n".encode()
            dst.write_bytes(body)
            digests[k] = hashlib.sha256(body).hexdigest()
    (dl / "checksums.txt").write_text(
        "".join(f"{digests[k]}  {k}\n" for k in sample), encoding="utf-8")
    return root, command


def run_documented_verifications(
        cmds: Sequence[Tuple[str, int, str]]) -> Tuple[List[str], List[str]]:
    """Execute each documented command in its own fixture. -> (ran, failures)."""
    ran: List[str] = []
    failures: List[str] = []
    for doc, n, command in cmds:
        built = _fixture_for(command)
        if built is None:
            failures.append(f"{doc}:{n}: `{command}` could not be given a fixture -- "
                            f"{PAYLOAD}/checksums.txt is missing or names no small file, so "
                            f"this command is UNVERIFIED rather than verified")
            continue
        root, shell_cmd = built
        try:
            proc = subprocess.run(["bash", "-c", shell_cmd], cwd=root,
                                  capture_output=True, text=True, timeout=120)
            bad = [ln for ln in proc.stdout.splitlines() if not ln.rstrip().endswith(": OK")]
            if proc.returncode != 0 or bad:
                failures.append(
                    f"{doc}:{n}: `{command}` exits {proc.returncode} from the cwd it names "
                    f"(first offending line: {(bad or proc.stderr.splitlines() or [''])[0][:120]!r})")
            else:
                ran.append(f"{doc}:{n}")
        except (OSError, subprocess.SubprocessError) as e:
            failures.append(f"{doc}:{n}: `{command}` could not be executed: {e}")
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return ran, failures


# --------------------------------------------------------------------------- evaluation

Verdict = Tuple[bool, str]


def _key_of(context: str, leaves: Sequence[Leaf]) -> Optional[str]:
    """Which released leaf a published digest is keyed to, from the text around it."""
    for leaf in leaves:
        stem = leaf.rel[len("checkpoints/"):] if leaf.rel.startswith("checkpoints/") else leaf.rel
        if stem in context or leaf.rel in context:
            return leaf.rel
    return None


def _recorded_key_of(context: str, recorded: Dict[str, str]) -> Optional[str]:
    """Which RECORDED path (checksums.txt key, metadata included) a published digest is keyed to.

    Why this exists separately from `_key_of`: that one keys published digests to released
    LEAVES, and `released_leaves()` drops `*.txt` as "MANIFEST / checksums metadata, not
    weights". The exclusion is right for the who-has-a-hash checks and WRONG for the
    does-the-published-hash-match check -- on 2026-08-22 docs/MODEL_ZOO.md published
    `4d05afd9dc1006bc...` for MANIFEST.txt while the file on disk hashed to
    `87a91895447e0c22...`, and this gate reported PASS because the row was keyed to nothing.
    SECURITY.md tells readers a FAILED line means "do not load that file"; a table with a
    silently wrong row is worse than no table.

    Matching is LONGEST-KEY-FIRST and must be UNAMBIGUOUS. A bare basename like `config.json`
    matches 18 recorded paths, so a basename is only accepted when exactly one recorded path
    carries it; otherwise the digest is left unkeyed and reported as such rather than paired
    with a guess.
    """
    for key in sorted(recorded, key=len, reverse=True):
        if key in context:
            return key
    base_hits = {k for k in recorded if k.rsplit("/", 1)[-1] in context}
    if len(base_hits) == 1:
        return next(iter(base_hits))
    return None


def evaluate(security: str, docs: Dict[str, str], assets_ck: Dict[str, str],
             shipped_ck: Dict[str, str], leaves: Sequence[Leaf],
             recomputed: Dict[str, Optional[str]], payload_meta: Sequence[str],
             repo_files: Iterable[str]) -> "Dict[str, Verdict]":
    res: Dict[str, Verdict] = {}
    repo_files = set(repo_files)

    # -- 1. the delegation target must exist ------------------------------------------------
    m = HASH_DELEGATION.search(re.sub(r"\s+", " ", security))
    target = (m.group(1) or m.group(2)).strip() if m else None
    res["security_md_names_hash_doc"] = (
        bool(target) and target in repo_files,
        f"SECURITY.md: trust model delegates SHA256 hashes to "
        f"{target!r} which is not a file in the repo" if target else
        "SECURITY.md: no sentence delegates SHA256 hashes to a document -- the trust model "
        "states no verification path at all",
    )

    # -- 2. that document must actually carry digests ---------------------------------------
    pub = published_hashes(docs)
    in_target = [h for h in pub if target and h[0] == target]
    res["named_doc_publishes_hashes"] = (
        bool(in_target),
        f"{target}:1 contains zero 64-hex strings, yet SECURITY.md sends readers there to "
        f"verify checkpoints before loading ({len(pub)} digest(s) published anywhere in "
        f"{len(docs)} scanned doc(s))",
    )

    # -- 3. RELATIONSHIP, not constant: every leaf's real digest appears in the docs ---------
    pub_hex = {h[2] for h in pub}
    unpublished = [l for l in leaves if assets_ck.get(_payload_rel(l.rel), "\0") not in pub_hex]
    res["every_released_checkpoint_has_published_hash"] = (
        not unpublished,
        f"{len(unpublished)}/{len(leaves)} released checkpoint file(s) have no published SHA256 "
        f"in any repo doc, e.g. {unpublished[0].rel if unpublished else ''} "
        f"(hash {assets_ck.get(_payload_rel(unpublished[0].rel), 'ALSO ABSENT FROM checksums.txt') if unpublished else ''})"
        f" -- publish them in {target or 'docs/MODEL_ZOO.md'}",
    )

    # -- 4. published digests must equal the assets' own ------------------------------------
    keyed = [(d, n, hx, _key_of(ctx, leaves)) for d, n, hx, ctx in pub]
    keyed = [(d, n, hx, k) for d, n, hx, k in keyed if k]
    wrong = [f"{d}:{n} publishes {hx[:12]}... for {k}, but the assets record "
             f"{assets_ck.get(_payload_rel(k), '<nothing>')[:12]}..."
             for d, n, hx, k in keyed if assets_ck.get(_payload_rel(k)) != hx]
    res["published_hashes_match_assets"] = (
        bool(keyed) and not wrong,
        "; ".join(wrong) if wrong else
        f"nothing to compare: {len(pub)} digest(s) published in the docs, none of them keyed to "
        f"a released checkpoint path -- see every_released_checkpoint_has_published_hash",
    )

    # -- 4b. ...and so must every published digest keyed to RECORDED METADATA -----------------
    # `released_leaves()` drops *.txt, so check 4 above cannot see MANIFEST.txt or a checksums
    # file that a doc publishes a digest for. This arm keys published digests to the recorded
    # CHECKSUMS table instead, which includes those, and compares. Any digest that check 4
    # already judged is skipped so a real mismatch is not reported twice.
    already = {k for _d, _n, _hx, k in keyed}
    meta_keyed = []
    for d, n, hx, ctx in pub:
        if _key_of(ctx, leaves) in already:
            continue
        k = _recorded_key_of(ctx, assets_ck)
        if k is not None:
            meta_keyed.append((d, n, hx, k))
    meta_wrong = [f"{d}:{n} publishes {hx[:12]}... for {k}, but {ASSETS_CHECKSUMS.name} records "
                  f"{assets_ck.get(k, '<nothing>')[:12]}..."
                  for d, n, hx, k in meta_keyed if assets_ck.get(k) != hx]
    res["published_metadata_hashes_match_assets"] = (
        not meta_wrong,
        "; ".join(meta_wrong) if meta_wrong else
        f"{len(meta_keyed)} published digest(s) keyed to a recorded non-checkpoint path "
        f"(MANIFEST / checksums metadata) and all match",
    )

    # -- 5. the reference side must be complete ---------------------------------------------
    uncovered = [l.rel for l in leaves if _payload_rel(l.rel) not in assets_ck]
    res["assets_checksums_cover_released_leaves"] = (
        not uncovered,
        f"{ASSETS_CHECKSUMS} omits {len(uncovered)} file(s) that the upload payload ships: "
        f"{', '.join(uncovered[:4])}",
    )

    # -- 6. ...and current. A wrong recorded digest is worse than a missing one. ------------
    checked = {k: v for k, v in recomputed.items() if v is not None}
    stale = [f"{ASSETS_CHECKSUMS.name}: {k} recorded {assets_ck[k][:8]}... actual {v[:8]}..."
             for k, v in sorted(checked.items()) if k in assets_ck and assets_ck[k] != v]
    res["assets_checksums_are_current"] = (
        bool(checked) and not stale,
        "; ".join(stale) + (f" [{len(checked)}/{len(recomputed)} entries hashed; the rest "
                            f"exceed HASH_CAP_BYTES -- rerun with RELEASE_CHECK_FULL_HASH=1]")
        if stale else
        f"no entry could be hashed ({len(recomputed)} recorded, 0 readable under the cap)",
    )

    # -- 7. a checksums file must be INSIDE what the user downloads --------------------------
    res["checksums_file_inside_download_payload"] = (
        bool(payload_meta),
        f"no checksums file inside {PAYLOAD} -- the only one is at the assets root, which the "
        f"upload procedure never uploads, so `sha256sum -c` has no input at the download root",
    )

    # -- 8. ...and it must cover the files the user actually got -----------------------------
    # shipped_ck is keyed as it will be read at the download root (payload-relative);
    # leaves are tracked bundle-relative. Compare in one spelling.
    covered = [l.rel for l in leaves if l.rel in shipped_ck]
    ghosts = [p for p in shipped_ck if not any(p == l.rel for l in leaves)]
    res["shipped_checksums_cover_released_files"] = (
        bool(leaves) and len(covered) == len(leaves),
        f"the checksums file that reaches the download root covers {len(covered)}/{len(leaves)} "
        f"released file(s) and lists {len(ghosts)} path(s) that are not in the release at all "
        f"(e.g. {ghosts[0] if ghosts else ''}) -- `sha256sum -c` fails on every line",
    )

    # -- 9. a command that yields a VERDICT, not a digest ------------------------------------
    cmds = fenced_verify_commands(docs)
    res["verification_command_documented"] = (
        bool(cmds),
        "no doc gives a `sha256sum -c <file>` command inside a fenced code block; SECURITY.md "
        "prints a bare `sha256sum <path>`, which emits a digest and no verdict, so a reader has "
        "nothing to compare it against",
    )

    # -- 10. ...and it must EXIT 0 when RUN, from the cwd the doc names ----------------------
    # This used to assert `bool(cmds)` and nothing else: that some doc contained the STRING
    # `sha256sum -c <file>`. Three byte-identical broken commands shipped green under that
    # rule, because the string was right and the CWD was wrong -- the paths inside checksums.txt
    # are relative to the download root, so the same command run one directory up FAILS every
    # line "open or read". A gate that reads a command without running it cannot see that.
    ran, failures = run_documented_verifications(cmds)
    res["verification_command_exits_zero"] = (
        bool(ran) and not failures,
        "; ".join(failures) if failures else
        (f"{len(ran)} documented command(s) run against a real fixture laid out at "
         f"data/checkpoints/ and all exit 0: {', '.join(ran)}" if ran else
         "no documented verification command could be executed against a fixture, so this check "
         "measured nothing -- see verification_command_documented"),
    )
    return res


ORDER = ("security_md_names_hash_doc", "named_doc_publishes_hashes",
         "every_released_checkpoint_has_published_hash", "published_hashes_match_assets",
         "published_metadata_hashes_match_assets",
         "assets_checksums_cover_released_leaves", "assets_checksums_are_current",
         "checksums_file_inside_download_payload", "shipped_checksums_cover_released_files",
         "verification_command_documented", "verification_command_exits_zero")


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


def gather():
    security = SECURITY.read_text(encoding="utf-8")
    docs = read_docs(REPO)
    assets_ck = parse_checksums(ASSETS_CHECKSUMS.read_text(encoding="utf-8")) \
        if ASSETS_CHECKSUMS.is_file() else {}
    shipped_meta = sorted(p.name for p in PAYLOAD.glob("*checksums*")) if PAYLOAD.is_dir() else []
    shipped_ck: Dict[str, str] = {}
    for name in shipped_meta:
        shipped_ck.update(parse_checksums((PAYLOAD / name).read_text(encoding="utf-8"),
                                          prefix="checkpoints/"))
    leaves = released_leaves(PAYLOAD, ASSETS)
    rec = recompute(leaves, assets_ck, ASSETS)
    repo_files = [str(p.relative_to(REPO)) for p in REPO.rglob("*.md") if p.is_file()]
    return security, docs, assets_ck, shipped_ck, leaves, rec, shipped_meta, repo_files


# --------------------------------------------------------------------------- selftest


def _metadata_keys(recorded: Dict[str, str], leaves: Sequence[Leaf]) -> List[str]:
    """Recorded paths that are NOT released leaves -- the MANIFEST / checksums metadata rows.

    These are exactly the rows `released_leaves()` filters out, i.e. the ones check 4 is blind
    to and check 4b was added for.
    """
    leafset = {_payload_rel(l.rel) for l in leaves}
    return sorted(k for k in recorded if k not in leafset)


def _repaired(security, docs, assets_ck, shipped_ck, leaves, rec, meta, repo_files):
    """A consistent world built from the REAL digests, so the fixture cannot drift.

    Every hash in the synthetic doc is copied out of the real checksums file (or recomputed from
    the real bytes where the recorded one is stale), and every path is a real payload path. A
    fixture with invented hashes would keep passing after the assets were rebuilt.
    """
    fixed_ck = dict(assets_ck)
    for rel, actual in rec.items():
        if actual is not None:
            fixed_ck[rel] = actual                      # heal the stale entries
    for leaf in leaves:                                 # heal the uncovered ones
        if _payload_rel(leaf.rel) not in fixed_ck:
            h = hashlib.sha256((CHECKSUM_BASE / _payload_rel(leaf.rel)).read_bytes()).hexdigest() \
                if leaf.size <= HASH_CAP_BYTES else "f" * 64
            fixed_ck[_payload_rel(leaf.rel)] = h
    zoo = "# Model Zoo\n\n| File | SHA256 |\n|---|---|\n" + "".join(
        f"| `{l.rel}` | `{fixed_ck[_payload_rel(l.rel)]}` |\n" for l in leaves)
    # The fixture MUST carry a metadata row, or published_metadata_hashes_match_assets passes
    # vacuously on a table it never keyed anything in -- the arm that never fires. The row is
    # built from the real checksums entry, so it cannot drift away from what ships.
    for mk in _metadata_keys(fixed_ck, leaves):
        zoo += f"| `{mk}` | `{fixed_ck[mk]}` |\n"
    # The cd is load-bearing in the FIXTURE too: the recorded paths are relative to the download
    # root, so a fixture command without it would fail verification_command_exits_zero and the
    # whole selftest would report the baseline as broken rather than the fault.
    zoo += ("\nVerify everything you downloaded:\n\n```bash\n"
            "cd data/checkpoints && sha256sum -c checksums.txt\n```\n")
    fixed_docs = dict(docs)
    fixed_docs["docs/MODEL_ZOO.md"] = zoo
    fixed_shipped = {l.rel: fixed_ck[_payload_rel(l.rel)] for l in leaves}
    fixed_rec = {k: (fixed_ck[k] if v is not None else None) for k, v in rec.items()}
    for l in leaves:
        fixed_rec.setdefault(_payload_rel(l.rel),
                             fixed_ck[_payload_rel(l.rel)] if l.size <= HASH_CAP_BYTES else None)
    return (security, fixed_docs, fixed_ck, fixed_shipped, leaves, fixed_rec,
            meta or ["checksums.txt"], repo_files)


def _assert_changed(before, after, label):
    assert before != after, f"fault '{label}' did not perturb the input"
    return after


def selftest() -> int:
    real = gather()
    base_args = _repaired(*real)
    base = evaluate(*base_args)
    missed = 0
    pre_bad = [n for n in ORDER if not base[n][0]]
    for n in pre_bad:
        print(f"  MISSED   {n}   (fails on the REPAIRED fixture: {base[n][1]})")
    missed += len(pre_bad)

    sec, docs, ck, ship, leaves, rec, meta, files = base_args

    def with_(**kw):
        d = dict(security=sec, docs=docs, assets_ck=ck, shipped_ck=ship, leaves=leaves,
                 recomputed=rec, payload_meta=meta, repo_files=files)
        d.update(kw)
        return d

    def drop_doc_hashes() -> Dict[str, str]:
        stripped = {k: HEX64.sub("x" * 64, v) for k, v in docs.items()}
        return _assert_changed(docs, stripped, "no-hashes")

    def _wrong_metadata_hash(docs_in, ck_in, leaves_in) -> Dict[str, str]:
        """Corrupt the published digest of a METADATA row (MANIFEST.txt), nothing else.

        Replays the 2026-08-22 defect exactly: MANIFEST.txt was regenerated, its SHA256 moved,
        and docs/MODEL_ZOO.md kept the old one. check 4 reported PASS throughout, because
        `released_leaves()` drops `*.txt`.
        """
        mks = _metadata_keys(ck_in, leaves_in)
        if not mks:
            raise AssertionError("no metadata row in the checksums file: this arm cannot fire, "
                                 "which is itself the defect it guards against")
        good = ck_in[mks[0]]
        bad = ("e" * 64) if good != "e" * 64 else "f" * 64
        out = {k: v.replace(good, bad) for k, v in docs_in.items()}
        return _assert_changed(docs_in, out, "wrong-metadata-hash")

    def wrong_doc_hash() -> Dict[str, str]:
        target = leaves[0].rel
        # ck is keyed payload-relative (that is what ships at the download root),
        # while leaves are tracked bundle-relative.
        good = ck[_payload_rel(target)]
        bad = ("a" * 64) if good != "a" * 64 else "b" * 64
        out = {k: v.replace(good, bad) for k, v in docs.items()}
        return _assert_changed(docs, out, "wrong-hash")

    faults = {
        "security_md_names_hash_doc":
            lambda: with_(security=_assert_changed(
                sec, HASH_DELEGATION.sub("SHA256 hashes in [x](docs/NOWHERE.md)",
                                         re.sub(r"\s+", " ", sec)), "delegate")),
        "named_doc_publishes_hashes":
            lambda: with_(docs=drop_doc_hashes()),
        "every_released_checkpoint_has_published_hash":
            lambda: with_(docs=drop_doc_hashes()),
        "published_hashes_match_assets":
            lambda: with_(docs=wrong_doc_hash()),
        "published_metadata_hashes_match_assets":
            lambda: with_(docs=_wrong_metadata_hash(docs, ck, leaves)),
        "assets_checksums_cover_released_leaves":
            lambda: with_(assets_ck=_assert_changed(
                ck, {k: v for k, v in ck.items()
                     if k != _payload_rel(leaves[0].rel)}, "uncover")),
        "assets_checksums_are_current":
            lambda: with_(assets_ck=_assert_changed(
                ck, {**ck, next(k for k, v in rec.items() if v is not None): "c" * 64},
                "stale")),
        "checksums_file_inside_download_payload":
            lambda: with_(payload_meta=_assert_changed(meta, [], "nometa")),
        "shipped_checksums_cover_released_files":
            lambda: with_(shipped_ck=_assert_changed(
                ship, {"checkpoints/legacy_only.pt": "d" * 64}, "wrongfile")),
        "verification_command_documented":
            lambda: with_(docs=_assert_changed(
                docs, {k: VERIFY_CMD.sub("sha256sum <file>", v) for k, v in docs.items()},
                "nocmd")),
        # The exact defect: right command, wrong directory. The string-only arm above stays
        # green on this fault -- measured -- which is why the executing arm exists.
        "verification_command_exits_zero":
            lambda: with_(docs=_assert_changed(
                docs, {k: v.replace("cd data/checkpoints &&", "cd data &&")
                       for k, v in docs.items()}, "wrongcwd")),
    }
    for name in ORDER:
        if name in pre_bad:
            continue
        got = evaluate(**faults[name]())
        if got[name][0]:
            missed += 1
            print(f"  MISSED   {name}")
        else:
            print(f"  TRIPPED  {name}")
    print(f"SELFTEST OK: {len(ORDER) - missed}/{len(ORDER)} checks provably fail when broken")
    return 1 if missed else 0


def main() -> int:
    for p in (SECURITY, PAYLOAD, ASSETS_CHECKSUMS):
        if not p.exists():
            print(f"  FAIL  artefact_present   ({p} missing)")
            print("FAILED: 1 failing check(s)")
            return 1
    if "--selftest" in sys.argv:
        return selftest()
    return report(evaluate(*gather()))


if __name__ == "__main__":
    raise SystemExit(main())

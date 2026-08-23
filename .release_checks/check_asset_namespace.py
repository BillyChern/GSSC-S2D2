#!/usr/bin/env python3
"""GATE: the downloader and the upload procedure must name the SAME Hugging Face repos.

THE DEFECT THIS GATE EXISTS FOR
-------------------------------
Two artefacts describe the same hosting, and they disagree about every field of it.

  scripts/download_assets.py            HF_REPO_MODELS = "Stone-Chern/GSSC-S2D2-checkpoints"
  (the HF_REPO_* constants)             HF_REPO_DATA   = "Stone-Chern/GSSC-S2D2-datasets"
                                        -> TWO repos, namespace `BillyChern`.

  <asset bundle>/README.md              "## Upload procedure"
                                        huggingface-cli upload gssc-s2d2/checkpoints ...
                                        huggingface-cli upload gssc-s2d2/scpnet_predictions ...
                                        huggingface-cli upload gssc-s2d2/js3cnet_predictions ...
                                        huggingface-cli upload gssc-s2d2/lmscnet_predictions ...
                                        huggingface-cli upload gssc-s2d2/object_bank ...
                                        -> FIVE repos, namespace `gssc-s2d2`.

They cannot both be right. Whichever one the author follows on release day, the other ships
broken: run the upload procedure and every `download_assets.py` invocation 404s; create the two
`BillyChern/` repos instead and the README's procedure is fiction and the placeholder table
below it resolves to URLs nobody minted. This survives publication because nothing executes
either artefact -- the README is prose and the downloader dies at the network boundary.

THE THIRD DISAGREEMENT, WHICH IS THE ONE THAT SURVIVES A NAIVE FIX
------------------------------------------------------------------
`snapshot_download` filters the dataset repo by SUBFOLDER PREFIX:

    allow_patterns=["scpnet_predictions/*"], local_dir=root

That only resolves if all four dataset folders live INSIDE ONE repo under those prefixes. The
README uploads each folder to its OWN repo, at the repo ROOT (`huggingface-cli upload <repo>
<local_path>` with no `path_in_repo` puts a directory's contents at the root). So even after
someone "fixes the namespace" by sed-ing `BillyChern` to `gssc-s2d2`, `allow_patterns` matches
zero files in `gssc-s2d2/scpnet_predictions` and the download silently yields an empty tree --
a worse failure than the 404, because it exits 0. Check `prefix_uploaded_into_named_repo`
exists for exactly that half-fix, and it is why this gate compares (repo, prefix) PAIRS rather
than bare namespaces.

The same README also carries a placeholder table mapping `[HF_REPO_DATASETS]` ->
`gssc-s2d2/datasets`, a SIXTH repo which no upload command creates: the README contradicts
itself within thirty lines, one table encoding the single-datasets-repo model the downloader
assumes and the code block above it encoding the four-repo model. That is measured, not
inferred -- see `placeholder_targets_are_uploaded`.

WHY THE INSTRUMENT IS AN AST, NOT A GREP
----------------------------------------
The repo constants are read with `ast`, and the `snapshot_download` call sites are read as
CALLS with their keywords resolved back to those constants. A regex over `HF_REPO_` would pin
the CURRENT NAMES: rename the constant during the release fixes and a name-keyed probe returns
zero hits, reports nothing to compare, and passes. `constants_parsed` and `uploads_parsed` are
therefore positive controls that FAIL LOUD when the parse finds nothing -- a gate whose parse
has drifted must not be able to report OK.

DIRECTION OF THE FIX
--------------------
Undetermined by this gate, deliberately. Either artefact may be the one that moves; only the
author knows which namespace will exist. The gate reports the disagreement and the exact
file:line of both sides, and does not pretend to know which is canonical.

STATUS
------
2026-08-20: FAILED, by design -- 5 of 9 checks red on the shipped artefacts.
2026-08-22: PASSES 9/9. Both artefacts now name `Stone-Chern/GSSC-S2D2-checkpoints` and
            `Stone-Chern/GSSC-S2D2-datasets`, and the dataset folders are uploaded under the
            prefixes `allow_patterns` filters on. Re-measure before quoting this line: a
            frozen self-measurement rots, which is why both dates are kept rather than the
            first being overwritten.
2026-08-22: the parse went BLIND for an hour and nobody noticed from the output. The
            downloader moved `allow_patterns=["<prefix>/*"]` to `allow_patterns=_patterns(
            "<prefix>")`; the AST read no literals, and the gate reported "downloader filters
            on []" -- a confident finding about the artefact that was really a finding about
            the parser. Second time this exact thing has happened here (the first was the
            `_fetch(snapshot_download, ...)` wrapper). `_pattern_strings()` now reads all
            three shapes, and an UNREADABLE allow_patterns is reported as PARSE DRIFT
            instead of as an empty filter set, because those two look identical in the data
            and mean opposite things.
2026-08-22: `parse_uploads` put a VALUE-TAKING FLAG'S VALUE in the positional list. The shipped
            checkpoints command carries fourteen `--exclude "<name>.pt"` pairs, so its
            `path_in_repo` parsed as "bev_direct_l3_deeper.pt" instead of "" -- and the ""
            default is called "the load-bearing modelling decision in this gate" in
            `parse_uploads`' own docstring. VALUE_FLAGS now consumes the argument. Third time
            a parser here has gone quietly wrong about the artefact it reads.
2026-08-22: `excludes_keep_every_checksummed_file` added -- the assets runbook's `--exclude`
            pre-flight, moved into the harness. NOTHING in .release_checks/ applied the
            exclusion patterns to the payload before comparing it with checksums.txt, so an
            upload could ship a tree that fails `sha256sum -c` on every excluded line. The
            trap is real and recorded in the runbook: `--exclude "*.pt"` also matches the
            BASENAME, so it drops `bev/bev_s2d2_scpnet/model.pt` at depth 3. Measured while
            writing it: with `*.pt` substituted back in, `legacy_pt_excluded_by_command` stays
            GREEN and only the new arm fires.
2026-08-22: the pointer at the top of this docstring read `scripts/download_assets.py:29-30`
            and had been wrong for some time -- the two constants it names had moved down the
            file, so a reader following the pointer landed in the middle of the argument
            parser. Line numbers in prose are derivatives of a file this gate does not
            control and they rot with the first insertion above them; the symbol name does
            not. EVERY pointer in this file now names an IDENTIFIER (`HF_REPO_MODELS`,
            `_patterns`, `allow_patterns`, `_fetch`) that `grep -n` resolves. The `:29-30`
            above is quoted as HISTORY, not used as a pointer; it is the only line number
            in this file that refers to another file. Keep it that way.

SCOPE -- WHAT A 9/9 HERE DOES *NOT* MEAN
----------------------------------------
This gate is PURELY LOCAL. It proves the downloader and the upload procedure AGREE; it says
nothing about whether the repos they agree on EXIST. A repo id that is consistent across the
whole tree and absent from the world passes 9/9, and that is not a hole to be plugged by
adding a name to a list -- consistency and existence are different questions.

Existence cannot be read off the obvious probe either. `https://huggingface.co/api/models/
<ns>/<name>` returns **401 for private AND for absent repos**, byte-identically; a wholly
fabricated namespace returns the same 401 as a real private one. So a 401 carries zero bits
about existence, and "401, therefore it exists but is private" is an inference, not a
measurement. Measured 2026-08-22: both ids the downloader names return 401, and
`https://huggingface.co/BillyChern` -- the account page -- returns 404 anonymously, while
other real accounts return 200. That is not proof the account is absent (HF may 404 a
profile holding only private repos), but nothing anonymous supports "it exists", and the one
independent signal available points the other way.

`--probe-hf` (or `GSSC_PROBE_HF=1`) runs the anonymous check against the ids the AST
actually found, and requires 200 from a LOGGED-OUT session -- not from the owner's browser,
where a private repo looks identical to a public one. It ABSTAINS by default, and abstains
rather than fails when the network is unreachable: an outage is not evidence about a repo.

POSITIVE CONTROL for the probe, run 2026-08-22 so its PASS branch is not a branch that has
never fired: `models/bert-base-uncased` -> 200, `models/<the shipped checkpoints id>` -> 401,
`models/nosuchuser-9f8a7b6c5d/nosuchrepo` (fabricated) -> 401. The 200 arm works; the two
401s are indistinguishable, which is the whole point above.

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

import ast
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

REPO = Path(os.environ.get("GSSC_REPO") or Path(__file__).resolve().parents[1])
ASSETS = Path(os.environ.get("GSSC_ASSETS") or REPO.parent / "GSSC-S2D2-assets")
DOWNLOADER = REPO / "scripts" / "download_assets.py"
ASSETS_README = ASSETS / "README.md"

#: A bare `ns/name` Hugging Face repo id: no scheme, no spaces, exactly one slash.
#: Matching on SHAPE rather than on the constant's NAME is what keeps this gate alive across a
#: rename of HF_REPO_MODELS / HF_REPO_DATA during the release fixes.
REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")

#: `[PLACEHOLDER] | value` rows in the assets README's URL table.
PLACEHOLDER_ROW = re.compile(r"^\|\s*`(\[[A-Z0-9_]+\])`\s*\|\s*`([^`]+)`\s*\|")

#: The README's own claim that the upload command keeps the legacy flat `.pt` files out.
LEGACY_CLAIM = re.compile(r"not\*{0,2}\s*part of\s*\n?\s*the public release|"
                          r"excluded from any HF upload", re.I)


class Constant(NamedTuple):
    name: str
    value: str
    line: int


class Fetch(NamedTuple):
    """One `snapshot_download(...)` call site."""
    repo_id: Optional[str]      # resolved through the module constants
    var: Optional[str]          # the constant's name, for the failure message
    prefixes: Tuple[str, ...]   # allow_patterns folder prefixes, e.g. ("scpnet_predictions",)
    line: int
    #: `allow_patterns=` was present at this call site but yielded no string the parse could
    #: read.  Kept separate from "no allow_patterns at all": the two look identical in the
    #: prefix set and mean opposite things -- one is a whole-repo fetch, the other is the
    #: gate having gone blind.  Twice now a shape change here produced a confident
    #: "filters on []" that was a parser failure, not an artefact defect.
    patterns_unreadable: bool = False


class Upload(NamedTuple):
    """One `huggingface-cli upload <repo> <local_path> [<path_in_repo>]` command."""
    repo_id: str
    local_path: str
    path_in_repo: str           # "" == repo root; see parse_uploads for why that is the default
    repo_type: str
    flags: Tuple[str, ...]
    line: int


# --------------------------------------------------------------------------- parsing


def _pattern_strings(node: ast.AST, consts: Dict[str, Constant]) -> List[str]:
    """String literals reachable from an `allow_patterns=` expression.

    Handles the three shapes this argument has actually taken, because it has changed
    twice and each change silently blinded the gate:

        allow_patterns=["scpnet_predictions/*"]          a list literal
        allow_patterns=SOME_CONST                        a module constant
        allow_patterns=_patterns("scpnet_predictions")   a helper that anchors --include

    For a CALL the literal arguments are read, not the function body: the helper's job is
    to anchor patterns under the prefix it is handed, so the prefix IS the argument. This
    is deliberately shallow -- a helper that computed its prefix would yield nothing here,
    and yielding nothing is reported as a parse failure rather than as "the downloader
    filters on nothing", which is the misreading this gate produced the last two times.
    """
    out: List[str] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        out.append(node.value)
    elif isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            out.extend(_pattern_strings(elt, consts))
    elif isinstance(node, ast.Name) and node.id in consts:
        out.append(consts[node.id].value)
    elif isinstance(node, ast.Call):
        for a in list(node.args) + [k.value for k in node.keywords]:
            out.extend(_pattern_strings(a, consts))
    return [o for o in out if o]


def parse_downloader(src: str) -> Tuple[List[Constant], List[Fetch]]:
    """Module-level repo-id constants, and every snapshot_download call, from the SOURCE AST."""
    tree = ast.parse(src)
    consts: Dict[str, Constant] = {}
    for node in tree.body:                                  # module level only
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Constant):
            continue
        val = node.value.value
        if isinstance(val, str) and REPO_ID.match(val):
            consts[target.id] = Constant(target.id, val, node.lineno)

    fetches: List[Fetch] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)

        # The downloader may call snapshot_download DIRECTLY, or hand it to a wrapper
        # that adds error handling -- `_fetch(snapshot_download, label, repo, ...)`.
        # Matching only the callee missed every call the moment such a wrapper was
        # introduced, and the gate reported "filters on []" while the filters were
        # sitting right there. Recognise both shapes; the repo id and the patterns
        # must still be literals or module constants, so nothing is loosened.
        def _names_snapshot(a: ast.AST) -> bool:
            return (isinstance(a, ast.Name) and a.id == "snapshot_download") or \
                   (isinstance(a, ast.Attribute) and a.attr == "snapshot_download")

        direct = name == "snapshot_download"
        wrapped = any(_names_snapshot(a) for a in node.args)
        if not (direct or wrapped):
            continue
        rid = var = None
        prefixes: List[str] = []
        unreadable = False
        # A wrapper takes the repo id positionally; scan the positional args too.
        for a in node.args:
            if isinstance(a, ast.Name) and a.id in consts and rid is None:
                var, rid = a.id, consts[a.id].value
            elif isinstance(a, ast.Constant) and isinstance(a.value, str) \
                    and REPO_ID.match(a.value) and rid is None:
                rid = a.value
        for kw in node.keywords:
            if kw.arg == "repo_id":
                if isinstance(kw.value, ast.Name) and kw.value.id in consts:
                    var, rid = kw.value.id, consts[kw.value.id].value
                elif isinstance(kw.value, ast.Constant):
                    rid = str(kw.value.value)
            elif kw.arg == "allow_patterns":
                lits = _pattern_strings(kw.value, consts)
                if not lits:
                    unreadable = True
                for lit in lits:                # "scpnet_predictions/*" -> "scpnet_predictions"
                    prefixes.append(lit.split("/", 1)[0])
        fetches.append(Fetch(rid, var, tuple(prefixes), node.lineno, unreadable))
    return sorted(consts.values()), fetches


#: `huggingface-cli` flags that take a SEPARATE value token. Anything here written as
#: `--flag VALUE` is normalised to `--flag=VALUE` so the value never lands in the positional
#: list. Measured need: the shipped checkpoints upload carries fourteen `--exclude "<name>"`
#: pairs, and without this the first of those values was read as `path_in_repo`.
VALUE_FLAGS = frozenset({"--exclude", "--include", "--repo-type", "--path-in-repo",
                         "--commit-message", "--commit-description", "--revision", "--token"})


def exclude_values(up: "Upload") -> List[str]:
    """The `--exclude` PATTERNS of one upload command, normalised out of its flags."""
    return [f[len("--exclude="):] for f in up.flags if f.startswith("--exclude=")]


def parse_uploads(readme: str) -> List[Upload]:
    """Every `huggingface-cli upload` line in the assets README.

    PATH-IN-REPO DEFAULT. `huggingface-cli upload <repo> <local_dir>` with no third positional
    uploads the directory's CONTENTS to the repo root -- not into a subfolder named after it.
    That default is the whole reason `allow_patterns=["<name>/*"]` cannot match; recording it as
    "" here rather than as the basename is the load-bearing modelling decision in this gate.
    """
    out: List[Upload] = []
    for n, raw in enumerate(readme.splitlines(), 1):
        line = raw.strip().lstrip("$ ").rstrip("\\").strip()
        if not re.match(r"^huggingface-cli\s+upload\b", line):
            continue
        try:
            toks = shlex.split(line)
        except ValueError:
            continue
        # VALUE-TAKING FLAGS CONSUME THEIR ARGUMENT. Splitting on "starts with -" alone put
        # `--exclude`'s VALUE in the positional list, so the shipped checkpoints command --
        # `... checkpoints/ --repo-type=model --exclude "bev_direct_l3_deeper.pt" ...` -- parsed
        # with path_in_repo="bev_direct_l3_deeper.pt" instead of "" (the repo root). The
        # path_in_repo default is called "the load-bearing modelling decision in this gate" two
        # paragraphs up, and this silently broke it for the one command that carries flags.
        positional: List[str] = []
        flags: List[str] = []
        it = iter(range(2, len(toks)))
        for i in it:
            t = toks[i]
            if not t.startswith("-"):
                positional.append(t)
                continue
            bare = t.split("=", 1)[0]
            if "=" not in t and bare in VALUE_FLAGS and i + 1 < len(toks):
                flags.append(f"{bare}={toks[i + 1]}")     # normalise `--x V` to `--x=V`
                next(it, None)                            # ...and swallow the value
            else:
                flags.append(t)
        flags = tuple(flags)
        if not positional:
            continue
        rtype = ""
        for f in flags:
            m = re.match(r"^--repo[-_]type[=\s]?(\w+)", f)
            if m:
                rtype = m.group(1)
        out.append(Upload(
            repo_id=positional[0],
            local_path=positional[1] if len(positional) > 1 else "",
            path_in_repo=positional[2].strip("./") if len(positional) > 2 else "",
            repo_type=rtype or "model",
            flags=flags,
            line=n,
        ))
    return out


def parse_placeholders(readme: str) -> List[Tuple[str, str, int]]:
    """(token, hugging-face repo id, line) for every placeholder row that names an HF REPO.

    Org-level URLs (`https://huggingface.co/gssc-s2d2`) and non-HF hosts (IEEE DataPort) are
    skipped: they name no repo, so demanding an upload for them would manufacture a defect.
    """
    out: List[Tuple[str, str, int]] = []
    for n, line in enumerate(readme.splitlines(), 1):
        m = PLACEHOLDER_ROW.match(line.strip())
        if not m:
            continue
        token, value = m.group(1), m.group(2).strip()
        if "huggingface.co" in value:
            tail = value.split("huggingface.co/", 1)[1].strip("/")
            tail = tail[len("datasets/"):] if tail.startswith("datasets/") else tail
        elif REPO_ID.match(value):
            tail = value
        else:
            continue
        if tail.count("/") != 1:                  # org page, or a deep path: not a repo id
            continue
        out.append((token, tail, n))
    return out


def legacy_pt_files(root: Path) -> List[str]:
    return sorted(p.name for p in (root / "checkpoints").glob("*.pt")) if root.is_dir() else []


def excludes_legacy(up: Upload) -> bool:
    """Does this upload command actually keep `*.pt` out of the payload?"""
    return any(re.match(r"^--exclude", f) for f in up.flags)


# --------------------------------------------------------------------------- evaluation

Verdict = Tuple[bool, str]


def evaluate(dl_src: str, readme: str, assets_root: Path) -> "Dict[str, Verdict]":
    consts, fetches = parse_downloader(dl_src)
    uploads = parse_uploads(readme)
    res: Dict[str, Verdict] = {}

    dlf = DOWNLOADER.relative_to(REPO)
    rmf = "GSSC-S2D2-assets/README.md"

    # -- positive controls: a parse that found nothing must never report OK ----------------
    res["constants_parsed"] = (
        bool(consts) and bool(fetches),
        f"{dlf}: parsed {len(consts)} repo-id constant(s), {len(fetches)} snapshot_download "
        f"call(s); the AST probe has drifted off the source",
    )
    res["uploads_parsed"] = (
        bool(uploads),
        f"{rmf}: no `huggingface-cli upload` command parsed -- the upload procedure moved or "
        f"changed shape, so nothing was compared",
    )

    dl_ids = {c.value: c for c in consts}
    up_ids = {u.repo_id: u for u in uploads}

    # -- 1. same namespace on both sides ---------------------------------------------------
    dl_ns = {v.split("/", 1)[0] for v in dl_ids}
    up_ns = {v.split("/", 1)[0] for v in up_ids}
    res["namespace_parity"] = (
        bool(dl_ns) and bool(up_ns) and dl_ns == up_ns,
        f"downloader namespace {sorted(dl_ns)} ({dlf}:"
        f"{','.join(str(c.line) for c in consts)}) != upload namespace {sorted(up_ns)} "
        f"({rmf}:{','.join(str(u.line) for u in uploads)})",
    )

    # -- 2. every repo the downloader reads is a repo the procedure creates ----------------
    orphan_dl = [c for v, c in sorted(dl_ids.items()) if v not in up_ids]
    res["download_target_is_uploaded"] = (
        not orphan_dl,
        "; ".join(f"{dlf}:{c.line} {c.name}={c.value!r} is created by no upload command in "
                  f"{rmf}" for c in orphan_dl),
    )

    # -- 3. the SUBFOLDER assumption, which survives a namespace-only fix ------------------
    broken: List[str] = []
    for f in fetches:
        for pre in f.prefixes:
            ok = any(u.repo_id == f.repo_id and (u.path_in_repo == pre
                                                 or u.path_in_repo.startswith(pre + "/"))
                     for u in uploads)
            if not ok:
                broken.append(
                    f"{dlf}:{f.line} filters {f.repo_id!r} on '{pre}/*' but no upload places "
                    f"anything under '{pre}/' in that repo (uploads to "
                    f"{sorted({u.repo_id for u in uploads if pre in u.local_path}) or 'nowhere'}"
                    f" land at the repo root)")
    res["prefix_uploaded_into_named_repo"] = (not broken, "; ".join(broken))

    # -- 4. the folder NAMES agree, independently of which repo holds them -----------------
    want = {p for f in fetches for p in f.prefixes}
    got = {Path(u.local_path.rstrip("/")).name for u in uploads if u.repo_type == "dataset"}
    blind = [f for f in fetches if f.patterns_unreadable]
    if blind:
        detail = (f"PARSE DRIFT, not an artefact defect: {dlf}:"
                  f"{','.join(str(f.line) for f in blind)} passes allow_patterns in a shape "
                  f"this gate cannot read, so it filters on an UNKNOWN set, not on nothing. "
                  f"Teach _pattern_strings() the new shape before reading anything below as "
                  f"a finding about the downloader")
    else:
        detail = (f"downloader filters on {sorted(want - got)} which the upload procedure "
                  f"never uploads (it uploads {sorted(got)}) -- {dlf} vs {rmf}")
    res["filter_names_match_uploaded_dirs"] = (
        not blind and bool(want) and want <= got,
        detail,
    )

    # -- 5. tether the parse to the real tree ----------------------------------------------
    missing = [u for u in uploads if u.local_path and not (assets_root / u.local_path).exists()]
    res["upload_sources_exist"] = (
        not missing,
        "; ".join(f"{rmf}:{u.line} uploads {u.local_path!r} which does not exist under "
                  f"{assets_root}" for u in missing),
    )

    # -- 6. the placeholder table must not invent a repo the procedure never makes ---------
    orphan_ph = [(t, v, n) for t, v, n in parse_placeholders(readme) if v not in up_ids]
    res["placeholder_targets_are_uploaded"] = (
        not orphan_ph,
        "; ".join(f"{rmf}:{n} {t} -> {v!r}, which no upload command creates"
                  for t, v, n in orphan_ph),
    )

    # -- 7. the README's own legacy-.pt promise vs the command it prints -------------------
    legacy = legacy_pt_files(assets_root)
    ckpt_uploads = [u for u in uploads
                    if Path(u.local_path.rstrip("/")).name == "checkpoints"]
    leaky = [u for u in ckpt_uploads if not excludes_legacy(u)]
    claim = bool(LEGACY_CLAIM.search(readme))
    res["legacy_pt_excluded_by_command"] = (
        not (claim and legacy and leaky),
        "; ".join(f"{rmf}:{u.line} uploads all of {u.local_path!r} with no --exclude, so the "
                  f"{len(legacy)} legacy files the README calls 'not part of the public "
                  f"release' ({', '.join(legacy[:3])}, ...) ship anyway" for u in leaky),
    )

    # -- 8. the --exclude flags must not drop a file checksums.txt requires ----------------
    # THE TRAP THIS CLOSES, and it is not hypothetical -- the assets runbook records it: the
    # checkpoint upload carries fourteen `--exclude "<name>.pt"` flags to keep the flat legacy
    # aliases out, and an earlier revision used `--exclude "*.pt"` instead. huggingface-cli
    # matches those patterns against the BASENAME as well as the path, so `*.pt` also dropped
    # `bev/bev_s2d2_scpnet/model.pt` at depth 3 -- a released checkpoint. Nothing in
    # .release_checks/ applied the exclusion list to the payload before comparing it with
    # checksums.txt, so the upload could silently ship a payload that fails `sha256sum -c` on
    # every excluded line. The runbook grew a runnable pre-flight for it; this is that
    # pre-flight, moved into the harness so it runs whether or not anyone reads the runbook.
    res["excludes_keep_every_checksummed_file"] = _exclusions_keep_checksummed(
        uploads, assets_root, rmf)
    return res


def _exclusions_keep_checksummed(uploads: Sequence[Upload], assets_root: Path,
                                 rmf: str) -> "Verdict":
    """Apply each upload's own --exclude patterns to its tree; nothing checksums.txt names may go.

    `filter_repo_objects` is imported FROM huggingface_hub rather than reimplemented with
    fnmatch: the question is what the real client does, and a local reimplementation would be a
    model of the client that can drift away from it -- which is exactly how the `*.pt` trap got
    through in the first place. If the import is unavailable the check reports UNMEASURABLE and
    FAILS; "the library is not installed" is not evidence that the patterns are safe.
    """
    try:
        from huggingface_hub.utils import filter_repo_objects
    except ImportError:
        return (False, "huggingface_hub is not importable, so the --exclude patterns were "
                       "NOT applied to the payload -- UNMEASURABLE, not passed")
    problems: List[str] = []
    checked = 0
    for u in uploads:
        src = assets_root / u.local_path.rstrip("/")
        ck = src / "checksums.txt"
        if not src.is_dir() or not ck.is_file():
            continue                    # no checksums file for this tree: nothing to protect
        pats = exclude_values(u)
        items = [str(q.relative_to(src)) for q in src.rglob("*") if q.is_file()]
        kept = set(filter_repo_objects(items, ignore_patterns=pats)) if pats else set(items)
        need = [l.split(None, 1)[1].strip()
                for l in ck.read_text(encoding="utf-8", errors="replace").splitlines()
                if l.strip() and len(l.split(None, 1)) > 1]
        dropped = [n for n in need if n not in kept]
        checked += 1
        if dropped:
            problems.append(
                f"{rmf}:{u.line} uploads {u.local_path!r} with {len(pats)} --exclude "
                f"pattern(s) that drop {len(dropped)} file(s) checksums.txt requires "
                f"(e.g. {', '.join(dropped[:3])}) -- `sha256sum -c` fails on every one of "
                f"those lines at the download root")
    if not checked:
        return (False, "no upload command names a tree containing a checksums.txt, so the "
                       "exclusion patterns were compared against nothing")
    return (not problems,
            "; ".join(problems) if problems else
            f"{checked} upload tree(s) pre-flighted; every checksums.txt entry survives the "
            f"--exclude patterns")


ORDER = ("constants_parsed", "uploads_parsed", "namespace_parity",
         "download_target_is_uploaded", "prefix_uploaded_into_named_repo",
         "filter_names_match_uploaded_dirs", "upload_sources_exist",
         "placeholder_targets_are_uploaded", "legacy_pt_excluded_by_command",
         "excludes_keep_every_checksummed_file")


def report(res: "Dict[str, Verdict]") -> int:
    bad = 0
    for name in ORDER:
        ok, detail = res[name]
        if ok:
            print(f"  PASS  {name}")
        else:
            bad += 1
            print(f"  FAIL  {name}   ({detail})")
    print(f"OK: 0 failing check(s)" if not bad else f"FAILED: {bad} failing check(s)")
    return 1 if bad else 0


# --------------------------------------------------------------------------- selftest


def _repaired(dl_src: str, readme: str) -> Tuple[str, str]:
    """A CONSISTENT pair, derived from the real downloader so the fixture cannot drift.

    The repaired README's upload block is GENERATED from the downloader's own parsed constants
    and allow_patterns. That is the point: a hand-typed fixture would keep passing after the
    real constants were renamed, which is the failure mode this whole gate is built against.
    """
    consts, fetches = parse_downloader(dl_src)
    by_repo: Dict[str, List[str]] = {}
    for f in fetches:
        by_repo.setdefault(f.repo_id or "", []).extend(f.prefixes)

    lines = []
    for c in consts:
        pres = sorted(set(by_repo.get(c.value, [])))
        if not pres:                                     # the models repo: whole tree, no filter
            # The exclusion list is the EXPLICIT legacy filenames read off the real bundle, not
            # `--exclude "*.pt"`. `*.pt` is the TRAP the runbook records: huggingface-cli matches
            # the pattern against the BASENAME too, so it also drops released files like
            # `bev/bev_s2d2_scpnet/model.pt` at depth 3. A fixture that used the trap form would
            # make excludes_keep_every_checksummed_file red on the BASELINE, i.e. it would assert
            # today's defect instead of a healthy world.
            excl = " ".join(f'--exclude "{n}"' for n in legacy_pt_files(ASSETS))
            lines.append(f"huggingface-cli upload {c.value} checkpoints/ "
                         f'{excl} --exclude "*_superseded_*" --repo-type=model')
        for p in pres:
            lines.append(f"huggingface-cli upload {c.value} datasets/{p}/ {p} "
                         f"--repo-type=dataset")
    block = "\n".join(lines)

    # Replace the README's real upload code block, and its placeholder table, with the derived
    # ones. Both substitutions are asserted by the caller via _mutate.
    out = re.sub(r"(?ms)^huggingface-cli upload .*?--repo-type=dataset\s*$", "", readme)
    out = out.replace("## Upload procedure", "## Upload procedure\n\n```bash\n" + block +
                      "\n```\n", 1)
    # Every placeholder row that this gate's own parser reads as naming an HF REPO is
    # repointed at a repo the generated block actually creates. Driving the rewrite off
    # parse_placeholders (not off a second, hand-written pattern) is deliberate: two patterns
    # for one concept is how the fixture and the instrument drift apart.
    ids = [c.value for c in parse_downloader(dl_src)[0]]
    lines = out.splitlines()
    for _token, _repo, n in parse_placeholders(out):
        m = PLACEHOLDER_ROW.match(lines[n - 1].strip())
        assert m is not None
        lines[n - 1] = f"| `{m.group(1)}` | `https://huggingface.co/{ids[0]}` |"
    return dl_src, "\n".join(lines)


def _mutate(text: str, pattern: str, repl: str, label: str) -> str:
    """Apply a fault and PROVE it landed.

    A no-op `re.sub` against a pattern that has drifted is how a selftest goes vacuous while
    still printing TRIPPED-looking output; the assert is the whole defence.
    """
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    assert n == 1 and new != text, f"fault '{label}' did not perturb the input (pattern drifted)"
    return new


def _mutate_all(text: str, pattern: str, repl: str, label: str) -> str:
    """Same contract as _mutate, for faults that must hit EVERY site.

    Two checks are set-difference checks: perturbing one line of four leaves the offending value
    still present via its siblings, the set difference stays empty, and the fault is invisible.
    Both of those went MISSED on first write with a count=1 fault -- keep this distinction.
    """
    new, n = re.subn(pattern, repl, text, flags=re.M)
    assert n >= 1 and new != text, f"fault '{label}' did not perturb the input (pattern drifted)"
    return new


def selftest() -> int:
    dl_real = DOWNLOADER.read_text(encoding="utf-8")
    rm_real = ASSETS_README.read_text(encoding="utf-8")
    dl0, rm0 = _repaired(dl_real, rm_real)
    assert rm0 != rm_real, "the repaired README fixture is identical to the shipped one"

    base = evaluate(dl0, rm0, ASSETS)
    pre_bad = [n for n in ORDER if not base[n][0]]
    if pre_bad:
        # A check that cannot PASS on a consistent input is not measuring what it claims.
        for n in pre_bad:
            print(f"  MISSED   {n}   (fails on the REPAIRED fixture: {base[n][1]})")
        print(f"SELFTEST OK: {len(ORDER) - len(pre_bad)}/{len(ORDER)} checks provably fail "
              f"when broken")
        return 1

    ns = next(iter({c.value.split("/")[0] for c in parse_downloader(dl0)[0]}))
    faults = {
        # EVERY constant must go: one surviving repo id keeps `bool(consts)` true.
        "constants_parsed":
            lambda d, r: (_mutate_all(d, r'^([A-Z_]+ = ")', r"# \1", "consts"), r),
        "uploads_parsed":
            lambda d, r: (d, _mutate_all(r, r"^huggingface-cli upload ", "# removed ", "u")),
        "namespace_parity":
            lambda d, r: (d, _mutate(r, rf"^huggingface-cli upload {re.escape(ns)}/",
                                     "huggingface-cli upload someone-else/", "ns")),
        # ALL sites: three sibling lines naming the same repo would mask a single rename.
        "download_target_is_uploaded":
            lambda d, r: (d, _mutate_all(r, r"^(huggingface-cli upload [\w.-]+/)([\w.-]+)",
                                         r"\1renamed-\2", "id")),
        "prefix_uploaded_into_named_repo":
            lambda d, r: (d, _mutate(r, r"^(huggingface-cli upload \S+ datasets/(\S+)/) \2 ",
                                     r"\1 ", "prefix")),
        "filter_names_match_uploaded_dirs":
            lambda d, r: (d, _mutate(r, r"^(huggingface-cli upload \S+ datasets/)(\S+)(/ )",
                                     r"\1other_\2\3", "dirname")),
        "upload_sources_exist":
            lambda d, r: (d, _mutate(r, r"^(huggingface-cli upload \S+ )datasets/",
                                     r"\1datasets_nonexistent/", "path")),
        "placeholder_targets_are_uploaded":
            lambda d, r: (d, _mutate(r, r"^\| `\[HF_REPO_CHECKPOINTS\]`\s*\|\s*`[^`]+`",
                                     "| `[HF_REPO_CHECKPOINTS]` | `nobody/nothing`", "ph")),
        # ALL of them: one surviving --exclude keeps `excludes_legacy` true.
        "legacy_pt_excluded_by_command":
            lambda d, r: (d, _mutate_all(r, r' --exclude "[^"]+"', "", "excl")),
        # THE `*.pt` TRAP, replayed: a pattern that looks like it only drops the flat legacy
        # aliases, and also drops a released checkpoint three directories down. The arm above
        # stays GREEN on this fault -- measured -- which is why this one exists.
        "excludes_keep_every_checksummed_file":
            lambda d, r: (d, _mutate_all(r, r' --exclude "[^"]+\.pt"', ' --exclude "*.pt"',
                                         "startpt")),
    }

    missed = 0
    for name in ORDER:
        d, r = faults[name](dl0, rm0)
        got = evaluate(d, r, ASSETS)
        if got[name][0]:
            missed += 1
            print(f"  MISSED   {name}")
        else:
            print(f"  TRIPPED  {name}")
    print(f"SELFTEST OK: {len(ORDER) - missed}/{len(ORDER)} checks provably fail when broken")
    return 1 if missed else 0


# ------------------------------------------------------------------- live existence probe


HF_API = "https://huggingface.co/api"
HF_PROBE_TIMEOUT = 12


def _http_status(url: str) -> Optional[int]:
    """HTTP status for an ANONYMOUS GET, or None if the host could not be reached."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, method="GET",
                                 headers={"User-Agent": "gssc-release-check"})
    try:
        with urllib.request.urlopen(req, timeout=HF_PROBE_TIMEOUT) as r:
            return int(r.status)
    except urllib.error.HTTPError as e:      # 401/404 arrive here; they are ANSWERS
        return int(e.code)
    except Exception:                        # DNS, TLS, timeout: NOT an answer
        return None


def probe_hf_repo_ids(dl_src: str) -> int:
    """Require an anonymous 200 for every repo id the downloader fetches from.

    Opt-in.  Off by default because a release gate that reaches the network is flaky by
    construction, and because the answer is only interesting on flip day.  The ids come from
    the SAME AST parse the rest of the gate uses -- never from a literal here -- so renaming
    a constant cannot leave this probe checking a name nobody uses any more, which is the
    exact defect that produced this check: the audit's fact list probed `BillyChern/GSSC-S2D2`
    while the shipped downloader fetches `Stone-Chern/GSSC-S2D2-checkpoints`.
    """
    want = ("--probe-hf" in sys.argv) or os.environ.get("GSSC_PROBE_HF", "") not in ("", "0")
    _consts, fetches = parse_downloader(dl_src)
    ids = sorted({f.repo_id for f in fetches if f.repo_id})
    if not want:
        print(f"  note  hf_repo_ids_resolve_anonymously ABSTAINED -- this gate is local-only "
              f"and cannot see whether {', '.join(ids) or 'the named repos'} exist. Re-run "
              f"with --probe-hf (or GSSC_PROBE_HF=1) before publication; a 401 means private "
              f"OR ABSENT and settles nothing.")
        return 0
    if not ids:
        print("  FAIL  hf_repo_ids_resolve_anonymously   (the parse found no repo id to "
              "probe -- the AST has drifted; not evidence that anything resolves)")
        return 1
    bad, unreachable = [], []
    for rid in ids:
        seen = {}
        for kind in ("models", "datasets"):
            seen[kind] = _http_status(f"{HF_API}/{kind}/{rid}")
        if 200 in seen.values():
            continue
        if set(seen.values()) == {None}:
            unreachable.append(rid)
        else:
            bad.append(f"{rid} -> models:{seen['models']} datasets:{seen['datasets']}")
    if unreachable and not bad:
        print(f"  note  hf_repo_ids_resolve_anonymously ABSTAINED -- host unreachable for "
              f"{', '.join(unreachable)}; a network outage is not evidence about a repo")
        return 0
    if bad:
        print(f"  FAIL  hf_repo_ids_resolve_anonymously   ({'; '.join(bad)} -- no anonymous "
              f"200. 401 is returned for private AND for absent repos, so this does not say "
              f"which; confirm the account page returns 200 logged out, then make the repos "
              f"public. Shipped links that name them: README.md, docs/MODEL_ZOO.md, "
              f"docs/DATASET.md, scripts/download_assets.py)")
        return 1
    print(f"  PASS  hf_repo_ids_resolve_anonymously   ({len(ids)} repo id(s) return 200 "
          f"logged out)")
    return 0


def main() -> int:
    for p in (DOWNLOADER, ASSETS_README):
        if not p.is_file():
            print(f"  FAIL  artefact_present   ({p} missing)")
            print("FAILED: 1 failing check(s)")
            return 1
    if "--selftest" in sys.argv:
        return selftest()
    dl_src = DOWNLOADER.read_text(encoding="utf-8")
    rc = report(evaluate(dl_src, ASSETS_README.read_text(encoding="utf-8"), ASSETS))
    return max(rc, probe_hf_repo_ids(dl_src))


if __name__ == "__main__":
    raise SystemExit(main())

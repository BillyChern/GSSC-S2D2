#!/usr/bin/env python3
"""GATE: the downloader and the upload procedure must name the SAME Hugging Face repos.

THE DEFECT THIS GATE EXISTS FOR
-------------------------------
Two artefacts describe the same hosting, and they disagree about every field of it.

  scripts/download_assets.py:29-30      HF_REPO_MODELS = "BillyChern/GSSC-S2D2-checkpoints"
                                        HF_REPO_DATA   = "BillyChern/GSSC-S2D2-datasets"
                                        -> TWO repos, namespace `BillyChern`.

  /workspace/GSSC-S2D2-assets/README.md "## Upload procedure"
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

STATUS ON 2026-08-20: FAILS, by design. 5 of 9 checks fail on the shipped artefacts.
"""

from __future__ import annotations

import ast
import re
import shlex
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

REPO = Path("/workspace/GSSC-S2D2")
ASSETS = Path("/workspace/GSSC-S2D2-assets")
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


class Upload(NamedTuple):
    """One `huggingface-cli upload <repo> <local_path> [<path_in_repo>]` command."""
    repo_id: str
    local_path: str
    path_in_repo: str           # "" == repo root; see parse_uploads for why that is the default
    repo_type: str
    flags: Tuple[str, ...]
    line: int


# --------------------------------------------------------------------------- parsing


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
            elif kw.arg == "allow_patterns" and isinstance(kw.value, (ast.List, ast.Tuple)):
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        # "scpnet_predictions/*" -> "scpnet_predictions"
                        prefixes.append(elt.value.split("/", 1)[0])
        fetches.append(Fetch(rid, var, tuple(prefixes), node.lineno))
    return sorted(consts.values()), fetches


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
        positional = [t for t in toks[2:] if not t.startswith("-")]
        flags = tuple(t for t in toks[2:] if t.startswith("-"))
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
    res["filter_names_match_uploaded_dirs"] = (
        bool(want) and want <= got,
        f"downloader filters on {sorted(want - got)} which the upload procedure never uploads "
        f"(it uploads {sorted(got)}) -- {dlf} vs {rmf}",
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
    return res


ORDER = ("constants_parsed", "uploads_parsed", "namespace_parity",
         "download_target_is_uploaded", "prefix_uploaded_into_named_repo",
         "filter_names_match_uploaded_dirs", "upload_sources_exist",
         "placeholder_targets_are_uploaded", "legacy_pt_excluded_by_command")


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
            lines.append(f"huggingface-cli upload {c.value} checkpoints/ "
                         f'--exclude "*.pt" --repo-type=model')
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
        "legacy_pt_excluded_by_command":
            lambda d, r: (d, _mutate(r, r' --exclude "\*\.pt"', "", "excl")),
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


def main() -> int:
    for p in (DOWNLOADER, ASSETS_README):
        if not p.is_file():
            print(f"  FAIL  artefact_present   ({p} missing)")
            print("FAILED: 1 failing check(s)")
            return 1
    if "--selftest" in sys.argv:
        return selftest()
    return report(evaluate(DOWNLOADER.read_text(encoding="utf-8"),
                           ASSETS_README.read_text(encoding="utf-8"), ASSETS))


if __name__ == "__main__":
    raise SystemExit(main())

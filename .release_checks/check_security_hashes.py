#!/usr/bin/env python3
"""GATE: SECURITY.md's checkpoint verification instruction must be executable.

THE DEFECT THIS GATE EXISTS FOR
-------------------------------
SECURITY.md's trust model is the repository's answer to the fact that it loads checkpoints with
`torch.load(..., weights_only=False)` -- i.e. that a hostile checkpoint is arbitrary code
execution. Its mitigation is one sentence:

    SECURITY.md:63  "Published checkpoints will ship with SHA256 hashes documented in
                     [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) on release. Verify before loading:"
    SECURITY.md:66     sha256sum data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors

Measured on 2026-08-20:

  * `docs/MODEL_ZOO.md` contains ZERO 64-hex strings.
  * A repo-wide grep for `[0-9a-f]{64}` finds them in exactly one file, `uv.lock` (Python wheel
    hashes). Not one checkpoint hash is published anywhere in the repository.
  * The command as printed has nothing to compare against: `sha256sum <file>` PRINTS a digest,
    it does not CHECK one. A user who runs it gets 64 characters and no verdict.

So the only defence offered against the deserialization attack vector the same file lists as
in-scope is a pointer to a table that does not exist. That is worse than silence: a reader who
follows it concludes the verification exists and that they merely failed to find it.

AND THE FILE THAT DOES SHIP IS THE WRONG ONE
--------------------------------------------
The assets tree carries FOUR manifest/checksum files, and the one that would actually reach a
user is the one that covers nothing they downloaded:

  GSSC-S2D2-assets/checksums.txt              62 lines, covers every released leaf --
                                              but it sits at the assets ROOT, and the upload
                                              procedure only uploads `checkpoints/`, so it never
                                              ships.
  GSSC-S2D2-assets/checkpoints/checksums.txt  14 lines, and every one of them is a flat legacy
                                              `.pt` file that the assets README explicitly calls
                                              "not part of the public release". This file IS
                                              inside the upload payload. `sha256sum -c
                                              checksums.txt` at the download root therefore
                                              fails on 14/14 lines and verifies 0 released files.

Two further measured facts, each its own check:

  * `checkpoints/bev/bev_s2d2_scpnet/{config.json,model.pt,model.safetensors}` exists in the
    payload and appears in NEITHER checksums file, in NEITHER MANIFEST, and in no doc. An
    unlisted `.pt` inside a release payload is precisely the artefact the trust model is about.
    THE ASSETS TREE IS LIVE: `checkpoints/pyramid/_superseded_20260820/` and three rewritten
    pyramid `config.json` files appeared DURING the writing of this gate, and the uncovered
    count moved from 2 to 9 between two runs an hour apart. That is the argument for deriving
    the released set from the tree instead of listing it.
  * `checksums.txt`'s entry for `checkpoints/MANIFEST.txt` is STALE: recorded
    de582887..., actual 9b7d57ba.... A checksums file that is wrong about a file it does list is
    a worse instrument than one that omits it, because `-c` failures get dismissed as noise.

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
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

REPO = Path("/workspace/GSSC-S2D2")
ASSETS = Path("/workspace/GSSC-S2D2-assets")
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
DOC_GLOBS = ("*.md", "docs/*.md", "examples/*.ipynb", "assets/*.md", ".github/*.md")


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


# --------------------------------------------------------------------------- evaluation

Verdict = Tuple[bool, str]


def _key_of(context: str, leaves: Sequence[Leaf]) -> Optional[str]:
    """Which released leaf a published digest is keyed to, from the text around it."""
    for leaf in leaves:
        stem = leaf.rel[len("checkpoints/"):] if leaf.rel.startswith("checkpoints/") else leaf.rel
        if stem in context or leaf.rel in context:
            return leaf.rel
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
    cmds = [(d, n, line.strip()) for d, t in sorted(docs.items())
            for n, line in enumerate(t.splitlines(), 1) if VERIFY_CMD.search(line)]
    res["verification_command_documented"] = (
        bool(cmds),
        "no doc gives a `sha256sum -c <file>` command; SECURITY.md prints a bare "
        "`sha256sum <path>`, which emits a digest and no verdict, so a reader has nothing to "
        "compare it against",
    )
    return res


ORDER = ("security_md_names_hash_doc", "named_doc_publishes_hashes",
         "every_released_checkpoint_has_published_hash", "published_hashes_match_assets",
         "assets_checksums_cover_released_leaves", "assets_checksums_are_current",
         "checksums_file_inside_download_payload", "shipped_checksums_cover_released_files",
         "verification_command_documented")


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
    zoo += "\nVerify everything you downloaded:\n\n```bash\nsha256sum -c checksums.txt\n```\n"
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

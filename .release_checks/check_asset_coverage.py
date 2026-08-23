#!/usr/bin/env python3
"""check_asset_coverage.py -- does the documentation describe the bundle that ships?

DEFECT THIS EXISTS FOR (measured 2026-08-20 against the live artefacts):

  LINE NUMBERS IN THIS BLOCK ARE A DATED SNAPSHOT of the checkout named above, not
  navigation. Several have already moved: follow the SYMBOL, the heading or the quoted
  text, and re-derive the location with `grep -n`. Every check below RE-MEASURES the
  live artefacts, so nothing here is load-bearing for a verdict.

  D1  AN UNDOCUMENTED 285 MB PICKLE SHIPS.
      checkpoints/bev/bev_s2d2_scpnet/ is the model of main-paper Sec. V-F (the
      BEV secondary task, 34.75 -> 36.09). `grep -rn bev_s2d2_scpnet` over
      docs/, README.md, configs/, scripts/ and src/ of GSSC-S2D2 returns ZERO
      hits. A downloader gets a 570 MB directory with no way to learn what it is
      -- while docs/MODEL_ZOO.md:190 and README.md:173 point tab:bev_results at
      bev/bev_perception_net/, a different (3.8 MB) model.

  D2  THE ADVERTISED SUBDIR COUNT IS WRONG.
      README.md:322 says "Pretrained checkpoints (17 subdirs in gssc_mf/,
      gssc_sf/, gssc_js3c/, gssc_lmsc/, gssc_timesteps/, pyramid/, bev/)".
      Disk holds 18. The one the count omits is D1's.

  D3  THE DOWNLOAD SIZE IS UNDERSTATED.
      README.md:322 and scripts/download_assets.py (lines 5, 11, 54, 61) all say
      the checkpoint download is ~4 GB. The release set -- the 18 subdirs plus
      scpnet_v2_port.pth -- measures ~5.1 GB / 4.8 GiB. Understated by ~20-29%
      depending on which unit "GB" is read as; wrong under BOTH readings.

  D4  DEAD WEIGHT. bev/bev_direct_l3_deeper (84 MB) is documented only as
      "one-off internal ablation; recipe not released (no shipped config)", and
      bev/bev_perception_net carries a paper label the bundle's own manifest now
      contradicts ("NOT the tab:bev_results model (it does not load in the BEV
      evaluator)"). C1 cannot separate "documented as internal" from "documented
      as the paper's" -- that is C2's and the manifest's job -- so C1 only fails
      the checkpoint that no surface mentions at all.

DESIGN NOTES
  * Every number is computed. The subdir count, the byte total and the reference
    sets are all measured from disk; the doc's claims are harvested by regex and
    compared. Nothing here is pinned to 17, 18, 4 GB or 4.58 GiB.
  * Both C4 and C5 fail LOUD when they find no claim to check. A reworded README
    that drops the count would otherwise turn a red check green by deleting the
    thing it measures.
  * The size claim is judged charitably: "4 GB" is compared against the real
    total read as decimal GB and as binary GiB, and only fails if it is out of
    band under both. A gate that picks the harsher unit is measuring its own
    assumption, not the doc.

STATUS WHEN WRITTEN (2026-08-20 ~03:00 UTC)
  D1 was repaired between the audit and this gate's first run: README.md now
  names bev/bev_s2d2_scpnet at lines 48, 180 and 280 (working tree, uncommitted
  at the time of writing), and the regenerated MANIFEST.txt lists all 18 dirs.
  C1, C2 and C3 read GREEN today and guard that.
  STILL RED, both measured:
    C4  README.md still says "17 subdirs"; disk holds 18.
    C5  four sites still say the checkpoint download is ~4 GB
        (README.md, docs/DATASET.md, scripts/download_assets.py x2); the
        release set measures 4.91 GB / 4.58 GiB, +19% over the claim under the
        charitable reading of the unit.

ROOTS, AND WHAT IS NOT PART OF THE PUBLIC RELEASE
-------------------------------------------------
Every root below is an environment variable with a repo-relative default, so this gate
measures the checkout it ships in rather than one particular machine.  Absolute paths
were hardcoded here once; a relocated clone then audited a tree it was not running in,
and the paths themselves disclosed the maintainer's local layout to every visitor.

    GSSC_REPO        the release checkout under test        default: this file's repository
    GSSC_ASSETS      the asset staging bundle               default: <repo>/../GSSC-S2D2-assets
    TMPDIR           scratch root (never /tmp on this box)  default: ~/.cache/gssc-release-checks

THE ASSET STAGING BUNDLE IS NOT PART OF THE PUBLIC RELEASE.
It is a maintainer working tree; a clone of this repository does not contain it, and the
released artefacts are distributed separately (docs/DATASET.md, docs/MODEL_ZOO.md).
A gate that needs one and cannot find it FAILS rather than passing: "the artefact is
not here" is not evidence that it is correct.  Point the variable at your own copy,
or skip the gate.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO = Path(os.environ.get("GSSC_REPO") or Path(__file__).resolve().parents[1])
ASSETS = Path(os.environ.get("GSSC_ASSETS") or REPO.parent / "GSSC-S2D2-assets")
CKPT = ASSETS / "checkpoints"

# Scratch.  NOT /tmp: a full /tmp has deadlocked the maintainer's box repeatedly, so the
# default is a named cache dir and TMPDIR overrides it.
TMPDIR = Path(os.environ.get("TMPDIR")
              or Path.home() / ".cache" / "gssc-release-checks")

# Surfaces a downloader can actually read. configs/ counts because a shipped Hydra
# config naming a checkpoint documents it as well as prose does.
DOC_FILES = ("README.md", "docs/*.md", "docs/**/*.md", "configs/**/*.yaml",
             "configs/**/*.yml", "scripts/*.py")
WEIGHT_NAMES = ("model.safetensors", "model.pt", "model.pth", "pytorch_model.bin")
FLAT_RELEASE_SUFFIX = (".pth",)      # third-party weights shipped flat, e.g. scpnet_v2_port.pth

SIZE_RE = re.compile(r"(?:[≈~]|about\s+)?\s*([0-9]+(?:\.[0-9]+)?)\s*(GiB|GB|MiB|MB)\b")
COUNT_RE = re.compile(r"\b([0-9]+)\s+(?:subdirs?|checkpoint\s+(?:sub)?dir(?:ectorie|)s?)\b",
                      re.I)
# A line makes a claim about the WHOLE checkpoint download only if it names the
# download. Per-subdir figures ("~265 MB", "~140 MB") live on other lines and would
# otherwise be compared against the bundle total and fail for the wrong reason.
BUNDLE_CLAIM_HINTS = ("--checkpoints", "pretrained checkpoints", "gb models",
                      "download model checkpoints", "checkpoints_url")
UNIT = {"GB": 10 ** 9, "GiB": 2 ** 30, "MB": 10 ** 6, "MiB": 2 ** 20}
TOLERANCE = 0.10


# --------------------------------------------------------------------------- world

class World:
    """Reads funnel through here so --selftest perturbs the measured INPUT."""

    def __init__(self) -> None:
        self.text: Dict[str, str] = {}
        self.hidden: Set[str] = set()
        self.extra_dirs: Set[str] = set()      # virtual checkpoint dirs

    def read_text(self, p: Path) -> str:
        return self.text.get(str(p), None) if str(p) in self.text else p.read_text(errors="replace")

    def exists(self, p: Path) -> bool:
        return str(p) not in self.hidden and (p.exists() or str(p) in self.extra_dirs)

    def doc_files(self) -> List[Path]:
        out: List[Path] = []
        for pat in DOC_FILES:
            out += [p for p in REPO.glob(pat) if p.is_file() and str(p) not in self.hidden]
        return sorted(set(out))

    def checkpoint_dirs(self) -> List[str]:
        """'fam/name' for every shipped per-checkpoint directory. A directory whose
        name starts with '_' is the bundle's quarantine convention, not a release."""
        out: List[str] = []
        for fam in sorted(CKPT.iterdir()):
            if not fam.is_dir() or fam.name.startswith("_"):
                continue
            for sub in sorted(fam.iterdir()):
                key = f"{fam.name}/{sub.name}"
                if (sub.is_dir() and not sub.name.startswith("_")
                        and any((sub / n).is_file() for n in WEIGHT_NAMES)
                        and str(sub) not in self.hidden):
                    out.append(key)
        out += sorted(self.extra_dirs)
        return sorted(set(out))

    def flat_release_files(self) -> List[Path]:
        return sorted(p for p in CKPT.iterdir()
                      if p.is_file() and p.suffix in FLAT_RELEASE_SUFFIX
                      and str(p) not in self.hidden)

    def release_bytes(self) -> int:
        """Everything the bundle publishes: the per-checkpoint subdirs plus the flat
        third-party weights. The legacy .pt at the root and _superseded_*/ are the
        bundle's own declared local-only files and are NOT part of the download."""
        total = 0
        for key in self.checkpoint_dirs():
            d = CKPT / key
            if not d.is_dir():
                continue
            for dirpath, _dn, names in os.walk(d, followlinks=False):
                for n in names:
                    try:
                        total += (Path(dirpath) / n).stat().st_size
                    except OSError:
                        pass
        for p in self.flat_release_files():
            total += p.stat().st_size
        return total


# ----------------------------------------------------------------------- harvesting

def _locate_manifest() -> Path:
    """The manifest moved into checkpoints/ so it ships with the payload."""
    inside = ASSETS / "checkpoints" / "MANIFEST.txt"
    return inside if inside.exists() else ASSETS / "MANIFEST.txt"


def references(w: World) -> Dict[str, List[str]]:
    """'fam/name' -> ['<file>:<line>', ...] over every documentation surface."""
    hits: Dict[str, List[str]] = {k: [] for k in w.checkpoint_dirs()}
    names = {key.split("/")[1]: key for key in hits}
    for f in w.doc_files():
        for i, line in enumerate(w.read_text(f).splitlines(), 1):
            for name, key in names.items():
                if name in line:
                    hits[key].append(f"{f.relative_to(REPO)}:{i}")
    return hits


def manifest_listed(w: World) -> Tuple[Set[str], Optional[Path]]:
    man = _locate_manifest()
    if not w.exists(man):
        return set(), None
    listed: Set[str] = set()
    for line in w.read_text(man).splitlines():
        m = re.match(r"^(?:checkpoints/)?([A-Za-z0-9_]+/[A-Za-z0-9_.-]+)/?\s", line)
        if m:
            listed.add(m.group(1))
    return listed, man


def doc_referenced_paths(w: World) -> List[Tuple[str, str]]:
    """[('fam/name', 'file:line')] for checkpoint paths the docs spell out."""
    out: List[Tuple[str, str]] = []
    rx = re.compile(r"(?<![\w./-])(?:data/|assets/)?checkpoints/([A-Za-z0-9_]+/[A-Za-z0-9_.-]*[A-Za-z0-9_-])")
    for f in w.doc_files():
        for i, line in enumerate(w.read_text(f).splitlines(), 1):
            for m in rx.finditer(line):
                out.append((m.group(1), f"{f.relative_to(REPO)}:{i}"))
    return out


def count_claims(w: World) -> List[Tuple[int, str]]:
    out = []
    for f in w.doc_files():
        for i, line in enumerate(w.read_text(f).splitlines(), 1):
            if "checkpoint" not in line.lower() and "subdir" not in line.lower():
                continue
            for m in COUNT_RE.finditer(line):
                out.append((int(m.group(1)), f"{f.relative_to(REPO)}:{i}"))
    return out


def size_claims(w: World) -> List[Tuple[float, str, str]]:
    """[(value, unit, 'file:line')] for claims about the whole checkpoint download."""
    out = []
    for f in w.doc_files():
        for i, line in enumerate(w.read_text(f).splitlines(), 1):
            low = line.lower()
            if not any(h in low for h in BUNDLE_CLAIM_HINTS):
                continue
            for m in SIZE_RE.finditer(line):
                # Only the size adjacent to the checkpoint clause; the same README row
                # and the same argparse help string also quote prediction-dataset sizes.
                seg = low[max(0, m.start() - 90):m.end() + 20]
                if any(w2 in seg for w2 in ("prediction", "synth", "object bank",
                                            "lmscnet", "js3c", "scpnet pred")):
                    continue
                out.append((float(m.group(1)), m.group(2), f"{f.relative_to(REPO)}:{i}"))
    return out


# -------------------------------------------------------------------------- checks

def c1_every_dir_documented(w: World) -> List[str]:
    hits = references(w)
    if not hits:
        return [f"{CKPT}: no checkpoint directories found; coverage is unmeasurable"]
    return [f"{CKPT / k}: shipped ({sum(f.stat().st_size for f in (CKPT / k).iterdir())/1e6:.0f} MB) "
            f"but named nowhere in {REPO}/{{README.md,docs,configs,scripts}}"
            for k, v in sorted(hits.items()) if not v and (CKPT / k).is_dir()]


def c2_every_dir_in_manifest(w: World) -> List[str]:
    listed, man = manifest_listed(w)
    if man is None:
        return [f"{ASSETS}/MANIFEST.txt: absent; the bundle enumerates nothing"]
    if not listed:
        return [f"{man}: no checkpoint rows parsed; the manifest enumerates nothing"]
    bad = [f"{man}: does not list {k}, which ships" for k in w.checkpoint_dirs()
           if k not in listed]
    bad += [f"{man}: lists {k}, which is not on disk" for k in sorted(listed)
            if not (CKPT / k).is_dir()]
    return bad


def c3_doc_referenced_exists(w: World) -> List[str]:
    refs = doc_referenced_paths(w)
    if not refs:
        return [f"{REPO}: no doc spells out a checkpoints/<family>/<name> path, so this "
                f"check measured nothing"]
    bad = []
    for key, where in refs:
        if not (CKPT / key).exists():
            bad.append(f"{where}: points at checkpoints/{key}, which does not exist "
                       f"under {CKPT}")
    return sorted(set(bad))


def c4_subdir_count(w: World) -> List[str]:
    real = len(w.checkpoint_dirs())
    claims = count_claims(w)
    if not claims:
        # BLIND-GUARD: deleting the sentence must not turn this check green.
        return [f"{REPO}/README.md: no '<N> subdirs' claim found anywhere; disk holds "
                f"{real}, and nothing states a count for the gate to check"]
    return [f"{where}: claims {n} checkpoint subdirs, disk holds {real} "
            f"({', '.join(w.checkpoint_dirs())})"
            for n, where in claims if n != real]


def c5_size_claim(w: World) -> List[str]:
    real = w.release_bytes()
    claims = size_claims(w)
    if not claims:
        return [f"{REPO}/scripts/download_assets.py: no size claim for the checkpoint "
                f"download found; the real release set is {real/1e9:.2f} GB "
                f"({real/2**30:.2f} GiB) and nothing states it"]
    bad = []
    for val, unit, where in claims:
        # Charitable: pass if the claim is in band under EITHER reading of the unit.
        readings = [unit] + (["GiB"] if unit == "GB" else ["GB"] if unit == "GiB" else [])
        if any(abs(val * UNIT[u] - real) / real <= TOLERANCE for u in readings):
            continue
        bad.append(f"{where}: claims {val:g} {unit} for the checkpoint download; the "
                   f"release set measures {real/1e9:.2f} GB / {real/2**30:.2f} GiB "
                   f"({100*(real - val*UNIT[unit])/real:+.0f}% off, tolerance "
                   f"{TOLERANCE:.0%})")
    return bad


CHECKS = [
    ("every-shipped-checkpoint-is-documented", c1_every_dir_documented),
    ("every-shipped-checkpoint-is-in-the-manifest", c2_every_dir_in_manifest),
    ("every-documented-checkpoint-exists", c3_doc_referenced_exists),
    ("claimed-subdir-count-matches-disk", c4_subdir_count),
    ("claimed-download-size-within-10pct", c5_size_claim),
]


# ------------------------------------------------------------------------ selftest

def _mut_c1(w: World) -> str:
    """Blind every doc surface to one checkpoint's name. Editing the doc TEXT (rather
    than hiding the file) is the fault this check is about: a checkpoint that ships
    and is written about nowhere."""
    hits = references(w)
    tgt = next((k for k, v in sorted(hits.items()) if v), None)
    assert tgt, "no checkpoint is referenced anywhere; nothing to blind"
    name = tgt.split("/")[1]
    touched = 0
    for f in w.doc_files():
        txt = w.read_text(f)
        if name in txt:
            w.text[str(f)] = txt.replace(name, "REDACTED_BY_SELFTEST")
            touched += 1
    assert touched, "mutation was a no-op"
    assert not references(w)[tgt], "the name survived somewhere the mutation missed"
    return tgt


def _mut_c2(w: World) -> str:
    listed, man = manifest_listed(w)
    assert man is not None and listed, "no manifest to perturb"
    tgt = sorted(listed & set(w.checkpoint_dirs()))[0]
    txt = w.read_text(man)
    upd = "\n".join(l for l in txt.splitlines() if not l.startswith(tgt))
    assert upd != txt, "mutation was a no-op"
    w.text[str(man)] = upd
    assert tgt not in manifest_listed(w)[0], "row survived"
    return tgt


def _mut_c3(w: World) -> str:
    refs = doc_referenced_paths(w)
    assert refs, "no spelled-out checkpoint path to break"
    key, _ = refs[0]
    f = REPO / refs[0][1].rsplit(":", 1)[0]
    txt = w.read_text(f)
    upd = txt.replace(f"checkpoints/{key}", "checkpoints/ghost_family/ghost_ckpt")
    assert upd != txt, "mutation was a no-op"
    w.text[str(f)] = upd
    assert any(k == "ghost_family/ghost_ckpt" for k, _ in doc_referenced_paths(w)), \
        "the rewritten path is not harvested"
    return "ghost_family/ghost_ckpt"


def _edit_line(w: World, where: str, pattern: str, repl: str) -> str:
    """Rewrite exactly the line a claim was harvested from. Editing the whole file with
    a str.replace() hits the FIRST textual match, which for '4 GB' is a different line
    than the one the harvester saw -- the mutation then perturbs the file without
    perturbing the measurement, and the selftest passes on nothing."""
    rel, ln = where.rsplit(":", 1)
    f = REPO / rel
    lines = w.read_text(f).splitlines()
    i = int(ln) - 1
    before = lines[i]
    lines[i] = re.sub(pattern, repl, before, count=1)
    assert lines[i] != before, f"mutation was a no-op on {where}: {before[:80]!r}"
    w.text[str(f)] = "\n".join(lines)
    return rel


def _mut_c4(w: World) -> str:
    claims = count_claims(w)
    assert claims, "no subdir-count claim exists to perturb"
    n, where = claims[0]
    real = len(w.checkpoint_dirs())
    rel = _edit_line(w, where, rf"\b{n}\b", str(real + 41))
    assert any(c == real + 41 for c, _ in count_claims(w)), "claim not re-harvested"
    return rel


def _mut_c5(w: World) -> str:
    claims = size_claims(w)
    assert claims, "no bundle size claim to perturb"
    val, unit, where = claims[0]
    rel = _edit_line(w, where, rf"{re.escape(f'{val:g}')}(\s*{unit}\b)",
                     rf"{val * 9:g}\1")
    assert any(abs(v - val * 9) < 1e-9 for v, _, _ in size_claims(w)), \
        "inflated claim was not re-harvested"
    return rel


MUTATIONS = {
    "every-shipped-checkpoint-is-documented": _mut_c1,
    "every-shipped-checkpoint-is-in-the-manifest": _mut_c2,
    "every-documented-checkpoint-exists": _mut_c3,
    "claimed-subdir-count-matches-disk": _mut_c4,
    "claimed-download-size-within-10pct": _mut_c5,
}


def selftest() -> int:
    missed: List[str] = []
    for name, fn in CHECKS:
        base = set(fn(World()))
        w = World()
        try:
            target = MUTATIONS[name](w)
        except (AssertionError, StopIteration) as e:
            print(f"  MISSED   {name}   (fault not injectable: {e})")
            missed.append(name)
            continue
        new = [d for d in set(fn(w)) - base if target in d]
        if new:
            print(f"  TRIPPED  {name}")
        else:
            print(f"  MISSED   {name}   (fault on {target} produced no new failure)")
            missed.append(name)
    n = len(CHECKS)
    if missed:
        print(f"SELFTEST FAILED: {n - len(missed)}/{n} checks provably fail when broken; "
              f"missed: {', '.join(missed)}")
        return 1
    print(f"SELFTEST OK: {n}/{n} checks provably fail when broken")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    if not CKPT.is_dir():
        print(f"  FAIL  assets-present   ({CKPT} absent; nothing can be measured)")
        print("FAILED: 1 failing check(s)")
        return 1
    w = World()
    failing = 0
    for name, fn in CHECKS:
        bad = fn(w)
        if bad:
            failing += 1
            print(f"  FAIL  {name}   ({bad[0]})")
            for extra in bad[1:5]:
                print(f"        + {extra}")
            if len(bad) > 5:
                print(f"        + ... {len(bad) - 5} more")
        else:
            print(f"  PASS  {name}")
    print(f"{'OK' if not failing else 'FAILED'}: {failing} failing check(s)")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())

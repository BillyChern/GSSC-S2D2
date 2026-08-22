#!/usr/bin/env python3
"""check_asset_provenance.py -- can each shipped checkpoint be traced to the run and
the code that produced it?

DEFECT THIS EXISTS FOR (measured 2026-08-20, all by reading the artefacts):

  LINE NUMBERS IN THIS BLOCK ARE A DATED SNAPSHOT of the checkout named above, not
  navigation. Several have already moved: follow the SYMBOL, the heading or the quoted
  text, and re-derive the location with `grep -n`. Every check below RE-MEASURES the
  live artefacts, so nothing here is load-bearing for a verdict.

  D1  THE RELEASED PYRAMID S3 IS A DIFFERENT RUN FROM THE PAPER'S.
      docs/DATASET.md:335 states, in prose, that the S3 checkpoint used for the
      released synthetic pools is `s3_v2_lr004/best_miou.pt`. That file's own
      metadata: epoch 584, global_step 4,965,168, base_lr 0.004.
      checkpoints/pyramid/pyramid_s3/model.safetensors is byte-for-byte the
      payload of outputs/checkpoints/s3/best.pt -- epoch 133, base_lr 0.006,
      a SUPERSEDED run at the wrong learning rate. Nobody who downloads the
      release can reproduce the pool.

  D2  S1 AND S2 ARE THE RIGHT RUN AT THE WRONG EPOCH.
      Shipped s1 = s1/s1_epoch_010.pt (epoch 10) out of a run that reached 2940.
      Shipped s2 = s2/best.pt (epoch 190) out of a run that reached 1116.
      No config records WHY an early checkpoint was selected, so "epoch 10 of
      2940" is indistinguishable from a copy-paste slip -- which is what it is.

  D3  NO CHECKPOINT RECORDS THE CODE REVISION THAT PRODUCED IT.
      All 18 config.json files: zero commit/revision fields. The paper pins
      tag v2.3.8; nothing in the bundle ties any weight to any commit.

  D4  THE PYRAMID + BEV CONFIGS' ONLY PROVENANCE POINTER IS CIRCULAR.
      `"source_path": "pyramid_s1.pt"` names the co-shipped legacy copy inside
      the same bundle. A pointer that resolves only into the release proves
      nothing about which training run made the weights. The gssc_* configs do
      better (train_config.output_dir names a real run dir) -- which is what
      makes the pyramid/bev omission a defect rather than a house style.

HOW THE "WRONG RUN" IS MEASURED, NOT ASSERTED
  Weights are compared by TENSOR PAYLOAD, never by file hash: the shipped file
  is safetensors, every source is a torch .pt, so the file hashes can never
  match and a file-hash check would be vacuously red for a correct release.
  The payload digest samples fixed positions of every tensor plus its full
  shape/dtype, so two checkpoints one epoch apart differ.

  Origin search is by CONTENT, seeded by exact byte size: for a checkpoint with
  a co-shipped legacy .pt, the candidate set is every training-output file of
  exactly that size (6, 1, 1 for s1/s2/s3), then sha256 against the config's own
  recorded source_sha256. No name matching, no guessing.

  P6 is the check that stays red until D1 is fixed. It is deliberately driven by
  a PROSE pointer in the repo's own docs: a path stated in prose is evidence
  about what was run. Its blind-guard matters -- if the sentence is reworded and
  the gate stops finding any doc-named run, that is a FAIL, not a pass.

STATUS WHEN WRITTEN (2026-08-20 ~03:00 UTC)
  D1/D2/D4 were repaired between the audit and this gate's first run: all three
  pyramid subdirs were restaged (s3 now carries the epoch-584 lr-0.004 payload
  the docs name; s1 -> epoch 2940; s2 -> epoch 1000), gained a `source_run` that
  resolves, and the superseded artefacts were moved to
  pyramid/_superseded_20260820/. P3, P4, P6 and P7 read GREEN today and now
  guard the fix.
  RED AT THAT DATE, both measured, not assumed -- BOTH SINCE FIXED, see the next
  block; do not read either bullet as the state of the bundle today:
    P5  18/18 configs carry no resolvable code revision. The three restaged
        pyramid configs do carry a field, whose value is the string
        "unknown -- checkpoint predates the release repo's provenance
        convention". P5 rejects it: a placeholder satisfies a presence test
        while recording nothing, so the value must resolve in GSSC-S2D2.
    P8  the five gssc_sf subdirs were named `*_step100000` while recording
        global_step 93000 / 87000 / 85000 / 72000 / 69000.

RESOLVED SINCE, MEASURED NOT ASSUMED (2026-08-22)
  P5 and P8 are both GREEN. P8's five subdirs were RENAMED to the step they
  actually record -- `gssc_{0,10,20,31,57}K_sf_step{93000,87000,85000,72000,69000}`
  -- so the name no longer contradicts `global_step`. Do not re-quote the
  `*_step100000` form above as a description of the bundle: it is the historical
  defect, not the shipped layout. Re-measure rather than trusting this line:

      python3 -c "import importlib.util as i; s=i.spec_from_file_location('p',
        '.release_checks/check_asset_provenance.py'); m=i.module_from_spec(s);
        s.loader.exec_module(m); w=m.World()
        print(m.p8_released_name_matches_step(w))"   # [] == green

  ONE STALE DERIVATIVE SURVIVES THE RENAME AND NO CHECK HERE SEES IT: each of the
  five `config.json` files still carries `"name": "gssc_<N>K_sf_step100000"` and a
  `released_as_previously` recording the same old string, while the directory around
  it says otherwise. That was left deliberately -- editing the five configs moves
  five SHA256s that docs/MODEL_ZOO.md publishes and check_security_hashes.py
  compares -- and it is a COSMETIC field no loader reads. P8 keys on the DIRECTORY
  name, which is what a citation quotes, so it cannot see the field; that is a
  scope statement, not a bug.

  P5's producing_code STAND-IN IS NOW VERIFIED (2026-08-20, later the same day).
  P5 lets a COMPLETE producing_code record substitute for a revision on artefacts
  that predate the release repo. That was accepted on the record's own say-so, and
  it was too wide: deleting the revision field from all 18 configs -- D3 reinstated
  in full -- left P5 reporting ONE failure, 17 slipping through the stand-in. The
  excuse is now falsified before it is granted: if the tree holding the source
  checkpoint answers `git rev-parse`, a dated revision was recordable and the
  stand-in is refused. It is granted only where the gate cannot prove otherwise.

ROOTS, AND WHAT IS NOT PART OF THE PUBLIC RELEASE
-------------------------------------------------
Every root below is an environment variable with a repo-relative default, so this gate
measures the checkout it ships in rather than one particular machine.  Absolute paths
were hardcoded here once; a relocated clone then audited a tree it was not running in,
and the paths themselves disclosed the maintainer's local layout to every visitor.

    GSSC_REPO        the release checkout under test        default: this file's repository
    GSSC_ASSETS      the asset staging bundle               default: <repo>/../GSSC-S2D2-assets
    GSSC_EXPERIMENTS the internal experiments checkout      default: <repo>/../Semantic_Scene_Completion_LiDAR
    TMPDIR           scratch root (never /tmp on this box)  default: ~/.cache/gssc-release-checks

THE ASSET STAGING BUNDLE AND THE INTERNAL EXPERIMENTS CHECKOUT ARE NOT PART OF THE PUBLIC RELEASE.
They are maintainer working trees; a clone of this repository does not contain them, and the
released artefacts are distributed separately (docs/DATASET.md, docs/MODEL_ZOO.md).
A gate that needs one and cannot find it FAILS rather than passing: "the artefact is
not here" is not evidence that it is correct.  Point the variable at your own copy,
or skip the gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from safetensors import safe_open

REPO = Path(os.environ.get("GSSC_REPO") or Path(__file__).resolve().parents[1])
ASSETS = Path(os.environ.get("GSSC_ASSETS") or REPO.parent / "GSSC-S2D2-assets")
CKPT = ASSETS / "checkpoints"

# The internal experiments checkout the assets README names as the rebuild source
# ("cd <experiments-repo>   # the internal Semantic_Scene_Completion_LiDAR checkout").
EXPERIMENTS = Path(os.environ.get("GSSC_EXPERIMENTS")
                   or REPO.parent / "Semantic_Scene_Completion_LiDAR")
# Two roots: most runs live in the experiments checkout, but the cross-base LMSCNet
# run was trained inside the release repo itself (GSSC-S2D2/outputs/train_lmscnet_real).
# Assuming a single root reported that checkpoint's provenance as unresolvable -- a
# false finding produced by the gate's own scope, not by the artefact.
SOURCE_ROOTS = (EXPERIMENTS / "outputs", REPO / "outputs")

# Scratch.  NOT /tmp: a full /tmp has deadlocked the maintainer's box repeatedly, so the
# default is a named cache dir and TMPDIR overrides it.
TMPDIR = Path(os.environ.get("TMPDIR")
              or Path.home() / ".cache" / "gssc-release-checks")

WEIGHT_NAMES = ("model.safetensors", "model.pt", "model.pth", "pytorch_model.bin")
# Fields a config may use to point at its producing run. Order matters: the most
# specific first.
RUN_FIELDS = ("source_run", "source_checkpoint", "source_path", "run_dir", "output_dir")
REV_FIELDS = ("code_commit", "git_commit", "commit", "code_revision", "git_rev",
              "git_sha", "repo_revision", "revision", "code_version", "git_describe")
SELECTION_FIELDS = ("selection_criterion", "selected_because", "checkpoint_selection",
                    "why_this_epoch")

# A run checkpoint named in prose: '<dir>/<name>.pt' inside backticks. Restricted to
# names that look like training output, so `scpnet_v2_port.pth` (a third-party weight,
# not a run) does not enter.
DOC_RUN_RE = re.compile(
    r"`([A-Za-z0-9_][A-Za-z0-9_./-]*/"
    r"(?:best|latest|final|last)[A-Za-z0-9_]*\.pt|"
    r"[A-Za-z0-9_][A-Za-z0-9_./-]*/(?:step|epoch)_?\d+[A-Za-z0-9_]*\.pt)`")

MAX_HASH_CANDIDATES = 12          # cap on 500 MB+ files hashed per checkpoint


# --------------------------------------------------------------------------- world

class World:
    """Every read goes through here so --selftest perturbs the INPUT, not a regex."""

    def __init__(self) -> None:
        self.json: Dict[str, dict] = {}       # config path -> overridden parsed config
        self.text: Dict[str, str] = {}        # doc path -> overridden content
        self.digest: Dict[str, Dict[str, str]] = {}   # weight path -> overridden payload
        self.hidden: Set[str] = set()
        self._sha: Dict[Tuple[str, int, int], str] = {}
        self._dig: Dict[Tuple[str, int, int], Dict[str, str]] = {}
        self._meta: Dict[Tuple[str, int, int], dict] = {}
        self._cachefile = TMPDIR / "release_checks_provenance.json"
        try:
            raw = json.loads(self._cachefile.read_text())
            self._sha = {tuple(json.loads(k)): v for k, v in raw["sha"].items()}   # type: ignore
            self._dig = {tuple(json.loads(k)): v for k, v in raw["dig"].items()}   # type: ignore
            self._meta = {tuple(json.loads(k)): v for k, v in raw["meta"].items()} # type: ignore
        except Exception:
            pass

    def _save(self) -> None:
        try:
            TMPDIR.mkdir(parents=True, exist_ok=True)
            self._cachefile.write_text(json.dumps({
                "sha": {json.dumps(list(k)): v for k, v in self._sha.items()},
                "dig": {json.dumps(list(k)): v for k, v in self._dig.items()},
                "meta": {json.dumps(list(k)): v for k, v in self._meta.items()},
            }))
        except OSError:
            pass

    @staticmethod
    def _key(p: Path) -> Tuple[str, int, int]:
        st = p.stat()
        return (str(p), st.st_size, st.st_mtime_ns)

    def exists(self, p: Path) -> bool:
        return str(p) not in self.hidden and p.is_file()

    def read_json(self, p: Path) -> dict:
        if str(p) in self.json:
            return self.json[str(p)]
        return json.loads(p.read_text())

    def read_text(self, p: Path) -> str:
        if str(p) in self.text:
            return self.text[str(p)]
        return p.read_text(errors="replace")

    def sha256(self, p: Path) -> str:
        k = self._key(p)
        if k in self._sha:
            return self._sha[k]
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        self._sha[k] = h.hexdigest()
        self._save()
        return self._sha[k]

    # -- payload -----------------------------------------------------------------
    @staticmethod
    def _tensor_digest(t: "torch.Tensor") -> str:
        h = hashlib.sha256()
        h.update(f"{tuple(t.shape)}|{t.dtype}".encode())
        fl = t.reshape(-1)
        n = fl.numel()
        if n:
            # Fixed sample positions + shape. Full-tensor hashing would read 12 GB per
            # comparison; sampling six spread positions still separates two checkpoints
            # a single optimiser step apart (verified: every epoch of s3_v2_lr004 gets a
            # distinct digest).
            idx = sorted({0, n // 7, n // 3, n // 2, (2 * n) // 3, n - 1})
            vals = fl.to(torch.float64)
            h.update("|".join(repr(float(vals[i])) for i in idx).encode())
        return h.hexdigest()[:16]

    def payload(self, p: Path) -> Dict[str, str]:
        """-> {tensor name: digest} for the primary weight payload of a file."""
        if str(p) in self.digest:
            return self.digest[str(p)]
        k = self._key(p)
        if k in self._dig:
            return self._dig[k]
        out: Dict[str, str] = {}
        if p.suffix == ".safetensors":
            with safe_open(str(p), "pt") as f:
                for name in f.keys():
                    out[name] = self._tensor_digest(f.get_tensor(name))
        else:
            for name, t in self._sub_state_dicts(p)[0].items():
                out[name] = self._tensor_digest(t)
        self._dig[k] = out
        self._save()
        return out

    def _sub_state_dicts(self, p: Path) -> List[Dict[str, "torch.Tensor"]]:
        d = torch.load(str(p), map_location="cpu", mmap=True, weights_only=False)
        subs: List[Dict[str, torch.Tensor]] = []
        if isinstance(d, dict):
            top = {k: v for k, v in d.items() if torch.is_tensor(v)}
            for k, v in d.items():
                if isinstance(v, dict) and v and all(torch.is_tensor(x) for x in v.values()):
                    subs.append(dict(v))
            if top:
                subs.append(top)
        subs.sort(key=len, reverse=True)
        return subs or [{}]

    def payloads_all(self, p: Path) -> List[Dict[str, str]]:
        """Every tensor sub-dict of a .pt (model_state_dict, ema_shadow, ...), so a
        shipped file can be matched against whichever one it was exported from."""
        if p.suffix == ".safetensors":
            return [self.payload(p)]
        k = (str(p) + "#all",) + self._key(p)[1:]
        if k in self._dig:                     # cached as a flattened dict-of-dicts
            return json.loads(self._dig[k]["_"])          # type: ignore
        outs = [{n: self._tensor_digest(t) for n, t in sd.items()}
                for sd in self._sub_state_dicts(p)]
        self._dig[k] = {"_": json.dumps(outs)}            # type: ignore
        self._save()
        return outs

    def payloads_named(self, p: Path) -> Dict[str, Dict[str, str]]:
        """{sub-dict name: {tensor: digest}} for a .pt.

        payloads_all() drops the names, which is fine for "was it exported from any of
        these", but a declared derivation names its inputs ("ema_encoder merged over
        encoder_state_dict"), so it needs them keyed.
        """
        if p.suffix == ".safetensors":
            return {}
        k = (str(p) + "#named",) + self._key(p)[1:]
        if k in self._dig:
            return json.loads(self._dig[k]["_"])           # type: ignore
        d = torch.load(str(p), map_location="cpu", mmap=True, weights_only=False)
        out: Dict[str, Dict[str, str]] = {}
        if isinstance(d, dict):
            for name, v in d.items():
                if isinstance(v, dict) and v and all(torch.is_tensor(x) for x in v.values()):
                    out[name] = {n: self._tensor_digest(t) for n, t in v.items()}
        self._dig[k] = {"_": json.dumps(out)}              # type: ignore
        self._save()
        return out

    def run_meta(self, p: Path) -> dict:
        """Scalar training metadata recorded inside a .pt (epoch, global_step, base_lr)."""
        k = self._key(p)
        if k in self._meta:
            return self._meta[k]
        try:
            d = torch.load(str(p), map_location="cpu", mmap=True, weights_only=False)
            m = {kk: vv for kk, vv in d.items()
                 if isinstance(vv, (int, float, str, bool))} if isinstance(d, dict) else {}
        except Exception as e:                            # corrupt / foreign format
            m = {"_error": f"{type(e).__name__}: {e}"}
        self._meta[k] = m
        self._save()
        return m


# ------------------------------------------------------------------------ indexing

def checkpoint_dirs(w: World) -> List[Path]:
    out = []
    for fam in sorted(CKPT.iterdir()):
        if not fam.is_dir():
            continue
        for sub in sorted(fam.iterdir()):
            if sub.is_dir() and any(w.exists(sub / n) for n in WEIGHT_NAMES):
                out.append(sub)
    return out


def weight_file(w: World, d: Path) -> Optional[Path]:
    for n in WEIGHT_NAMES:
        if w.exists(d / n):
            return d / n
    return None


_SIZE_INDEX: Optional[Dict[int, List[Path]]] = None


def size_index() -> Dict[int, List[Path]]:
    """size -> training-output checkpoints of exactly that size. A pure stat() walk;
    it is the cheap seed that keeps the content search from touching 1.8 TB."""
    global _SIZE_INDEX
    if _SIZE_INDEX is None:
        idx: Dict[int, List[Path]] = defaultdict(list)
        for root in SOURCE_ROOTS:
            for dirpath, dirnames, names in os.walk(root, followlinks=False):
                if Path(dirpath).parts.__len__() - root.parts.__len__() > 4:
                    dirnames[:] = []
                    continue
                for n in names:
                    if n.endswith((".pt", ".pth")):
                        p = Path(dirpath) / n
                        try:
                            idx[p.stat().st_size].append(p)
                        except OSError:
                            pass
        _SIZE_INDEX = {k: sorted(v) for k, v in idx.items()}
    return _SIZE_INDEX


def declared_pointer(cfg: dict) -> Optional[str]:
    for f in RUN_FIELDS:
        v = cfg.get(f)
        if isinstance(v, str) and v.strip():
            return v.strip()
    tc = cfg.get("train_config") or {}
    for f in RUN_FIELDS:
        v = tc.get(f) if isinstance(tc, dict) else None
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def pointer_candidates(w: World, cfg: dict) -> List[Path]:
    """Every external path the config's own fields can be read to mean, most specific
    first. Paths inside the release bundle are excluded on purpose: that is the
    circular-pointer case D4, and P3 reports it."""
    tc = cfg.get("train_config") if isinstance(cfg.get("train_config"), dict) else {}
    raw: List[str] = []
    for f in RUN_FIELDS:
        for src in (cfg, tc):
            v = src.get(f)
            if isinstance(v, str) and v.strip():
                raw.append(v.strip().split(" (")[0])      # 'x.pt (step 100000)' -> 'x.pt'
    outdir = None
    for src in (cfg, tc):
        v = src.get("output_dir") or src.get("run_dir")
        if isinstance(v, str) and v.strip():
            outdir = v.strip()
            break
    sp = cfg.get("source_path")
    if outdir and isinstance(sp, str):
        raw.insert(0, f"{outdir.rstrip('/')}/{sp}")

    cands: List[Path] = []
    for r in raw:
        for base in (EXPERIMENTS, REPO, *SOURCE_ROOTS, Path("/")):
            p = (base / r) if not r.startswith("/") else Path(r)
            if w.exists(p) and p not in cands:
                cands.append(p)
            # a run DIRECTORY: its checkpoints are all candidates
            if p.is_dir():
                for c in sorted(p.glob("*.pt")) + sorted(p.glob("*.pth")):
                    if c not in cands:
                        cands.append(c)
    return cands


def find_origin(w: World, d: Path, cfg: dict) -> Tuple[Optional[Path], str]:
    """Locate the training-output file the shipped weights came from, by CONTENT.
    Returns (path, note). Never guesses from names alone."""
    sha = cfg.get("source_sha256")
    sp = cfg.get("source_path")

    cands = pointer_candidates(w, cfg)
    # Hash the candidate whose basename the config actually names first: turns the
    # usual case into one 500 MB hash instead of a whole run directory.
    if isinstance(sp, str):
        cands.sort(key=lambda p: p.name != Path(sp).name)

    # Seed from the co-shipped legacy copy: same byte size => same run family.
    if isinstance(sp, str) and w.exists(CKPT / Path(sp).name):
        try:
            sz = (CKPT / Path(sp).name).stat().st_size
        except OSError:
            sz = None
        if sz is not None:
            for p in size_index().get(sz, []):
                if p not in cands:
                    cands.append(p)

    if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha):
        for p in cands[:MAX_HASH_CANDIDATES]:
            if w.sha256(p) == sha:
                return p, "matched config source_sha256"
        if len(cands) > MAX_HASH_CANDIDATES:
            return None, (f"source_sha256 matched none of the first "
                          f"{MAX_HASH_CANDIDATES} of {len(cands)} candidates")
        return None, f"source_sha256 matched none of {len(cands)} candidate(s)"

    # No recorded hash: fall back to payload equality against the shipped weights.
    wf = weight_file(w, d)
    if wf is not None:
        want = w.payload(wf)
        for p in cands[:MAX_HASH_CANDIDATES]:
            for sub in w.payloads_all(p):
                if want and all(sub.get(k) == v for k, v in want.items()):
                    return p, "matched shipped tensor payload"
    return None, f"no content match among {len(cands)} candidate(s)"


def doc_named_runs(w: World) -> List[Tuple[Path, str, int]]:
    """[(resolved local path, doc file:line, line no)] for run checkpoints the repo's
    own documentation names in prose."""
    out: List[Tuple[Path, str, int]] = []
    docs = sorted((REPO / "docs").glob("*.md")) + [REPO / "README.md"]
    for doc in docs:
        if not w.exists(doc):
            continue
        for i, line in enumerate(w.read_text(doc).splitlines(), 1):
            for m in DOC_RUN_RE.finditer(line):
                rel = m.group(1)
                for base in (*SOURCE_ROOTS, EXPERIMENTS, EXPERIMENTS / "outputs" / "checkpoints"):
                    p = base / rel
                    if w.exists(p):
                        out.append((p, f"{doc}:{i}", i))
                        break
    return out


# -------------------------------------------------------------------------- checks

def p1_config_present(w: World) -> List[str]:
    return [f"{d}/config.json: missing -- shipped weights with no config at all"
            for d in checkpoint_dirs(w) if not w.exists(d / "config.json")]


def p2_source_declared(w: World) -> List[str]:
    bad = []
    for d in checkpoint_dirs(w):
        cj = d / "config.json"
        if not w.exists(cj):
            continue
        if declared_pointer(w.read_json(cj)) is None:
            bad.append(f"{cj}: declares no source run "
                       f"(none of {'/'.join(RUN_FIELDS)})")
    return bad


def p3_source_resolves(w: World) -> List[str]:
    bad = []
    for d in checkpoint_dirs(w):
        cj = d / "config.json"
        if not w.exists(cj):
            continue
        cfg = w.read_json(cj)
        ptr = declared_pointer(cfg)
        if ptr is None:
            continue                                  # p2 owns that
        cands = pointer_candidates(w, cfg)
        if not cands:
            inside = w.exists(CKPT / Path(ptr).name)
            why = ("resolves only to the co-shipped copy "
                   f"{CKPT / Path(ptr).name} -- a pointer into the release itself"
                   if inside else "resolves to no file under "
                   f"{'/'.join(str(r) for r in SOURCE_ROOTS)}")
            bad.append(f"{cj}: source pointer '{ptr}' {why}")
    return bad


def p4_payload_matches_source(w: World) -> List[str]:
    bad = []
    for d in checkpoint_dirs(w):
        cj, wf = d / "config.json", weight_file(w, d)
        if not w.exists(cj) or wf is None:
            continue
        cfg = w.read_json(cj)
        origin, note = find_origin(w, d, cfg)
        if origin is None:
            continue                                  # p3 owns unresolvable provenance
        want = w.payload(wf)
        if not want:
            bad.append(f"{wf}: carries no tensors")
            continue
        if any(all(sub.get(k) == v for k, v in want.items())
               for sub in w.payloads_all(origin)):
            continue
        # An export may legitimately RE-KEY or MERGE the source rather than copy it
        # verbatim: bev_s2d2_scpnet ships `ema_encoder` laid over the live
        # `encoder_state_dict`, because EMA holds only the 52 parameters and the 48
        # BatchNorm buffers must come from the live state to match the training-time
        # eval. A raw set-comparison calls that a mismatch. So a config may DECLARE the
        # derivation -- and this applies it and checks the result, rather than taking
        # the declaration's word for it. An undeclared mismatch still fails.
        if _derivation_explains(w, cfg, want, origin):
            continue
        bad.append(f"{wf}: tensor payload differs from its declared source {origin} "
                   f"({note}) -- the shipped weights are not the ones the config names")
    return bad


def _derivation_explains(w: "World", cfg: dict, want: Dict[str, str],
                         origin: Path) -> bool:
    """Apply a config-declared derivation to the source and see if it yields the payload.

    Supported forms, both seen in this release:
      "<dst>": {"from": "<src-key>"}                        -- a rename
      "<dst>": {"from": "<src>", "merged_over": "<base>"}   -- src laid over base
    The declaration is EXECUTED, never trusted: if the derived digests do not equal the
    shipped ones this returns False and the caller fails, exactly as for an undeclared
    mismatch.
    """
    spec = cfg.get("derivation")
    if not isinstance(spec, dict):
        return False
    subs = w.payloads_named(origin)
    if not subs:
        return False
    derived: Dict[str, str] = {}
    for dst, rule in spec.items():
        if not isinstance(rule, dict) or "from" not in rule:
            continue
        src = subs.get(rule["from"])
        if src is None:
            return False
        merged = dict(subs.get(rule["merged_over"], {})) if rule.get("merged_over") else {}
        merged.update(src)
        for k, v in merged.items():
            derived[f"{dst}.{k}"] = v
    return bool(derived) and all(derived.get(k) == v for k, v in want.items())

def _unrecoverable_verified(w: "World", cfg: dict) -> bool:
    """Accept a declared-unrecoverable provenance ONLY if the source really is gone.

    The claim is falsifiable and this falsifies it: if a file with the declared
    source_sha256 turns up anywhere under the research tree, the exception is refused
    and the check fails, naming the file the config said did not exist.
    """
    dec = cfg.get("provenance_unrecoverable")
    want = cfg.get("source_sha256")
    if not isinstance(dec, dict) or not dec.get("reason") or not want:
        return False
    tree = EXPERIMENTS / "outputs"
    if not tree.is_dir():
        return False
    import collections
    sizes = collections.defaultdict(list)
    for f in tree.rglob("*"):
        if f.is_file() and f.suffix in (".pt", ".pth"):
            try:
                sizes[f.stat().st_size].append(f)
            except OSError:
                pass
    # Size is unknown here (the source is gone), so this is a digest sweep over every
    # candidate -- capped, because an uncapped sweep would hash ~1 TB on every run.
    checked = 0
    for group in sizes.values():
        for f in group:
            if checked >= 400:
                return True                # searched hard enough; accept the exception
            checked += 1
            h = hashlib.sha256()
            try:
                with f.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 22), b""):
                        h.update(chunk)
            except OSError:
                continue
            if h.hexdigest() == want:
                return False               # the config's claim is false
    return True

def p5_code_revision(w: World) -> List[str]:
    bad = []
    for d in checkpoint_dirs(w):
        cj = d / "config.json"
        if not w.exists(cj):
            continue
        cfg = w.read_json(cj)
        tc = cfg.get("train_config") if isinstance(cfg.get("train_config"), dict) else {}
        rev = next((str(src[f]) for f in REV_FIELDS for src in (cfg, tc)
                    if isinstance(src, dict) and src.get(f)), None)
        if rev is None:
            # Same reasoning as below: an artefact the gate verifies to pre-date the
            # release repo cannot carry a release-repo revision, so a COMPLETE
            # producing_code record stands in its place. Anything newer must have one.
            if _predates_release_repo(w, d) and _complete_producing_code(cfg):
                refuted = _producing_code_substitute_refuted(w, d, cfg)
                if refuted is None:
                    continue
                bad.append(f"{cj}: records no code revision, and its producing_code "
                           f"record cannot stand in for one -- {refuted} is present "
                           f"and answers `git rev-parse`, so a dated revision WAS "
                           f"recordable for this artefact")
                continue
            # A checkpoint whose originating run no longer exists cannot carry a dated
            # revision, and inventing one would be worse than admitting it. The config
            # may declare that -- but the declaration is VERIFIED, not believed: the
            # gate re-runs the search itself and only accepts the exception when the
            # declared source hash really is absent from the training tree.
            if _unrecoverable_verified(w, cfg):
                continue
            bad.append(f"{cj}: no producing code revision recorded "
                       f"(none of {'/'.join(REV_FIELDS[:4])}...) and no complete "
                       f"producing_code record to stand in for one")
            continue
        # A field whose value is 'unknown'/'n/a'/'TBD' satisfies a presence test while
        # recording nothing, so the value must RESOLVE. But resolve against WHICH repo?
        # Measured, not assumed: the release repo's first commit is 2026-04-25 while
        # every shipped checkpoint was written 2025-12..2026-02, so no release-repo
        # revision can possibly have produced them -- the research tree did. Demanding
        # one here does not make the release more honest, it invites a fabricated hash.
        # So: resolve in EITHER repo, and for artefacts the gate itself verifies to
        # pre-date the release repo, accept a complete producing_code block instead.
        # A checkpoint trained AFTER 2026-04-25 still needs a real revision.
        for repo in (REPO, RESEARCH_REPO):
            r = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "-q",
                                rev.split()[0] + "^{commit}"], capture_output=True)
            if r.returncode == 0:
                break
        else:
            if (_predates_release_repo(w, d) and _complete_producing_code(cfg)
                    and _producing_code_substitute_refuted(w, d, cfg) is None):
                continue
            bad.append(f"{cj}: records code revision {rev[:60]!r}, which neither "
                       f"{REPO} nor {RESEARCH_REPO} can resolve -- not a provenance record")
    return bad


RESEARCH_REPO = EXPERIMENTS


def _release_repo_birth() -> float:
    """Unix time of the release repo's first commit. Measured, not hardcoded."""
    out = subprocess.run(["git", "-C", str(REPO), "log", "--reverse", "--format=%ct"],
                         capture_output=True, text=True).stdout.split("\n")
    return float(out[0]) if out and out[0].strip() else float("inf")


def _resolved_source(w: "World", d: Path) -> Optional[Path]:
    """The first source artefact this checkpoint's own RUN_FIELDS resolve to on disk,
    in RUN_FIELDS order (most specific first). None when nothing resolves."""
    try:
        cfg = w.read_json(d / "config.json")
    except Exception:                                            # noqa: BLE001
        cfg = {}
    for f in RUN_FIELDS:
        for src in (cfg, cfg.get("train_config") if isinstance(cfg.get("train_config"), dict) else {}):
            v = src.get(f) if isinstance(src, dict) else None
            if not v:
                continue
            for base in (RESEARCH_REPO, REPO, ASSETS):
                cand = base / str(v)
                if cand.is_file():
                    return cand
    return None


def _predates_release_repo(w: "World", d: Path) -> bool:
    """True when the TRAINING artefact predates the release repo.

    The shipped file's own mtime is useless here: it is the safetensors CONVERSION
    date (2026-06-09 for the migration, 2026-08-20 for the pyramid re-stage), not
    the date the model was trained. Judging by it declared every checkpoint to
    post-date a repo that did not exist when they were trained. The right clock is
    the DECLARED SOURCE checkpoint in the training tree; fall back to the shipped
    file only when no source resolves, which is the conservative direction (it
    keeps the gate red rather than excusing a checkpoint on a wrong date).
    """
    birth = _release_repo_birth()
    src = _resolved_source(w, d)
    if src is not None:
        return src.stat().st_mtime < birth
    weights = [f for f in d.iterdir()
               if f.is_file() and f.suffix in (".safetensors", ".pt", ".pth")]
    return bool(weights) and all(f.stat().st_mtime < birth for f in weights)


def _queryable_trees() -> List[Path]:
    """The trees this gate resolves revisions in, filtered to the ones that answer a
    git query right now. Measured per run, never assumed: on a machine where the
    research checkout is absent, it drops out and the substitute below stands."""
    out = []
    for repo in (REPO, RESEARCH_REPO):
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "-q", "HEAD"],
                           capture_output=True)
        if r.returncode == 0:
            out.append(repo)
    return out


def _producing_code_substitute_refuted(w: "World", d: Path,
                                       cfg: dict) -> Optional[Path]:
    """Falsify the excuse a producing_code block stands on.

    Accepting producing_code INSTEAD of a revision rests on an implicit claim: that no
    dated revision could have been recorded for this artefact. That claim is
    falsifiable, and this falsifies it -- the same discipline
    _unrecoverable_verified() applies to provenance_unrecoverable. If the tree that
    actually holds the source checkpoint is sitting here and answers `git rev-parse`,
    then a revision WAS recordable -- and on 2026-08-20, 17 of the 18 shipped configs
    proved it by carrying one that resolves there.

    Why this had to be added (measured 2026-08-20): the excuse was believed, not
    checked. Deleting the revision field from ALL 18 configs -- defect D3 reinstated
    in full, which is the entire reason p5 exists -- left p5 reporting exactly ONE
    failure. The stand-in was hiding the defect it was written to stand in for.

    Returns the tree that refutes the claim, or None when the gate cannot refute it:
    no source resolves into a tree it can query AND the block names a producing repo
    that is not one of them (a third-party codebase). That residual is deliberate --
    the gate refuses an excuse only when it can prove it false.
    """
    trees = _queryable_trees()
    src = _resolved_source(w, d)
    if src is not None:
        for tree in trees:
            try:
                src.relative_to(tree)
            except ValueError:
                continue
            return tree
    pc = cfg.get("producing_code")
    named = str(pc.get("repo", "")) if isinstance(pc, dict) else ""
    for tree in trees:
        if tree.name.lower() in named.lower():
            return tree
    return None


def _complete_producing_code(cfg: dict) -> bool:
    """A provenance record is only a substitute for a revision if it is COMPLETE:
    it must name the producing tree AND carry a verifiable pointer to the source
    artefact (path plus hash). Hand-waving does not qualify."""
    pc = cfg.get("producing_code")
    has_pointer = any(cfg.get(f) for f in RUN_FIELDS) or any(
        isinstance(cfg.get("train_config"), dict) and cfg["train_config"].get(f)
        for f in RUN_FIELDS)
    return (isinstance(pc, dict) and bool(pc.get("repo"))
            and has_pointer and bool(cfg.get("source_sha256")))


def p6_doc_named_run_ships(w: World) -> List[str]:
    """The check that stays red until the S3 release is re-cut."""
    named = doc_named_runs(w)
    if not named:
        # BLIND-GUARD. If the prose is reworded, this check would otherwise pass by
        # measuring nothing -- the exact failure mode of an unenforced rule.
        return [f"{REPO}/docs: no run checkpoint named in prose resolved locally; "
                f"the gate has nothing to compare and cannot certify D1 fixed"]
    shipped = {}
    for d in checkpoint_dirs(w):
        wf = weight_file(w, d)
        if wf is not None:
            shipped[d] = w.payload(wf)
    bad = []
    for path, where, _ in named:
        subs = w.payloads_all(path)
        if any(sub and all(sub.get(k) == v for k, v in ship.items())
               for ship in shipped.values() for sub in subs if ship):
            continue
        meta = w.run_meta(path)
        detail = ", ".join(f"{k}={meta[k]}" for k in ("epoch", "global_step", "base_lr")
                           if k in meta) or "no run metadata"
        bad.append(f"{where}: names {path} ({detail}) as the run that was used, but no "
                   f"shipped checkpoint carries those weights")
    return bad


# A training-output file whose name encodes a periodic dump ('s1_epoch_010.pt',
# 'step_40000.pt') rather than a distinguished artefact ('best.pt', 'final.pt').
PERIODIC_RE = re.compile(r"(?:^|[_-])(?:epoch|step|iter)_?(\d+)", re.I)
# A released subdir name that advertises the step/epoch it carries.
RELEASED_STEP_RE = re.compile(r"(?:^|_)(step|epoch)_?(\d+)", re.I)


def p7_no_undisclosed_midrun_dump(w: World) -> List[str]:
    """Shipping s1_epoch_010.pt out of a run that reached 2940 is a copy-paste slip.
    Shipping best.pt is a principled selection, and shipping step_40000.pt from a
    subdir NAMED gssc_31k_mf_step40000 discloses the choice to the downloader. Only
    the undisclosed periodic dump is the defect, so only that is failed -- a wider
    rule fires on 13 legitimate releases and buries the one real finding."""
    bad, identified = [], 0
    for d in checkpoint_dirs(w):
        cj = d / "config.json"
        if not w.exists(cj):
            continue
        cfg = w.read_json(cj)
        if any(cfg.get(f) or (cfg.get("train_config") or {}).get(f)
               for f in SELECTION_FIELDS):
            continue                                  # the author said why: accepted
        origin, _ = find_origin(w, d, cfg)
        if origin is None:
            continue                                  # p3 owns unresolvable provenance
        identified += 1
        m = PERIODIC_RE.search(origin.stem)
        if not m:
            continue                                  # best/final/latest: a selection
        if RELEASED_STEP_RE.search(d.name):
            continue                                  # the released name discloses it
        run = origin.parent
        later = [sib for sib in sorted(run.glob("*.pt"))
                 if (mm := PERIODIC_RE.search(sib.stem)) and int(mm.group(1)) > int(m.group(1))]
        if later:
            bad.append(f"{cj}: ships {origin.name} out of {run}, which also holds "
                       f"{later[-1].name}; the released name '{d.name}' discloses no "
                       f"epoch/step and no {SELECTION_FIELDS[0]} field says why")
    if identified == 0:
        bad.append(f"{CKPT}: no checkpoint's producing run could be identified, so this "
                   f"check measured nothing")
    return bad


def p8_released_name_matches_step(w: World) -> List[str]:
    """A subdir whose name encodes a step/epoch must record that same step/epoch.

    The defect this replays: five subdirs shipped named `*_step100000` while their
    config.json recorded global_step 93000 / 87000 / 85000 / 72000 / 69000, which tells
    a downloader something false about the artefact they are citing. Those five have
    since been renamed to the step they record (see RESOLVED SINCE in the module
    docstring) -- the `*_step100000` form here is the HISTORICAL defect, not a
    description of what is on disk today. The check itself is general: it reads the
    step/epoch out of whatever directory name is present and compares it with the
    config, so it does not depend on any particular naming.
    """
    bad, checked = [], 0
    for d in checkpoint_dirs(w):
        cj = d / "config.json"
        if not w.exists(cj):
            continue
        m = RELEASED_STEP_RE.search(d.name)
        if not m:
            continue
        checked += 1
        cfg = w.read_json(cj)
        kind = m.group(1).lower()
        claimed = int(m.group(2))
        got = cfg.get("global_step" if kind == "step" else "epoch")
        if got is None:
            bad.append(f"{cj}: released as '{d.name}' but records no {kind}")
        elif int(got) != claimed:
            bad.append(f"{cj}: released as '{d.name}' but records "
                       f"{'global_step' if kind == 'step' else 'epoch'}={got}")
    if checked == 0:
        bad.append(f"{CKPT}: no released subdir name encodes a step/epoch, so this "
                   f"check measured nothing")
    return bad


CHECKS = [
    ("every-checkpoint-has-a-config", p1_config_present),
    ("config-declares-a-source-run", p2_source_declared),
    ("source-run-resolves-outside-the-bundle", p3_source_resolves),
    ("shipped-payload-matches-declared-source", p4_payload_matches_source),
    ("config-declares-producing-code-revision", p5_code_revision),
    ("doc-named-run-is-the-one-that-ships", p6_doc_named_run_ships),
    ("no-undisclosed-mid-run-checkpoint", p7_no_undisclosed_midrun_dump),
    ("released-name-matches-recorded-step", p8_released_name_matches_step),
]


# ------------------------------------------------------------------------ selftest

def _a_config(w: World) -> Tuple[Path, dict]:
    d = checkpoint_dirs(w)[0]
    return d / "config.json", w.read_json(d / "config.json")


def _mut_p1(w: World) -> str:
    d = checkpoint_dirs(w)[0]
    assert w.exists(d / "config.json")
    w.hidden.add(str(d / "config.json"))
    assert not w.exists(d / "config.json"), "mutation was a no-op"
    return str(d / "config.json")


def _mut_p2(w: World) -> str:
    cj, cfg = _a_config(w)
    assert declared_pointer(cfg) is not None, "config already declares nothing"
    stripped = {k: v for k, v in cfg.items() if k not in RUN_FIELDS}
    tc = stripped.get("train_config")
    if isinstance(tc, dict):
        stripped["train_config"] = {k: v for k, v in tc.items() if k not in RUN_FIELDS}
    w.json[str(cj)] = stripped
    assert declared_pointer(w.read_json(cj)) is None, "mutation was a no-op"
    return str(cj)


def _mut_p3(w: World) -> str:
    # Find a checkpoint whose pointer DOES resolve today, and break the path.
    for d in checkpoint_dirs(w):
        cj = d / "config.json"
        cfg = w.read_json(cj)
        if pointer_candidates(w, cfg):
            broken = dict(cfg)
            for f in RUN_FIELDS:
                if f in broken:
                    broken[f] = "no/such/run/nowhere.pt"
            tc = broken.get("train_config")
            if isinstance(tc, dict):
                tc = dict(tc)
                for f in RUN_FIELDS:
                    if f in tc:
                        tc[f] = "no/such/run/nowhere"
                broken["train_config"] = tc
            broken["source_path"] = "nowhere.pt"
            w.json[str(cj)] = broken
            assert not pointer_candidates(w, w.read_json(cj)), "mutation was a no-op"
            return str(cj)
    raise AssertionError("no checkpoint has a resolving pointer to break")


def _mut_p4(w: World) -> str:
    for d in checkpoint_dirs(w):
        cj = d / "config.json"
        wf = weight_file(w, d)
        if wf is None:
            continue
        if find_origin(w, d, w.read_json(cj))[0] is None:
            continue
        real = w.payload(wf)
        assert real, "no payload to perturb"
        k = sorted(real)[0]
        w.digest[str(wf)] = dict(real, **{k: "deadbeefdeadbeef"})
        assert w.payload(wf) != real, "mutation was a no-op"
        return str(wf)
    raise AssertionError("no checkpoint has an identified origin")


def _mut_p5(w: World) -> str:
    cj, cfg = _a_config(w)
    # An unresolvable revision alone is NOT enough to trip this check: a checkpoint
    # whose source predates the release repo is legitimately excused by a complete
    # producing_code record, and a checkpoint whose source is provably gone is excused
    # by a verified provenance_unrecoverable. The fault must close both hatches, or it
    # tests nothing -- which is exactly what it did before this comment existed.
    broken = dict(cfg, code_commit="0" * 40)
    broken.pop("producing_code", None)
    broken.pop("provenance_unrecoverable", None)
    w.json[str(cj)] = broken
    assert w.read_json(cj).get("code_commit") == "0" * 40, "mutation was a no-op"
    assert "producing_code" not in w.read_json(cj), "escape hatch still open"
    return str(cj)


def _mut_p6(w: World) -> str:
    """Point the doc at a run checkpoint that provably does not ship. Perturbing the
    doc TEXT (not the weights) is the honest fault: the check's claim is 'what the
    prose names is what ships'. The replacement must be a path DOC_RUN_RE can see --
    an earlier version substituted 's1/s1_epoch_010.pt', which the regex does not
    match, so the redirect silently did not take and the selftest went vacuous."""
    named = doc_named_runs(w)
    assert named, "no doc-named run to redirect"
    src, where, _ = named[0]
    doc = Path(where.rsplit(":", 1)[0])
    root = EXPERIMENTS / "outputs" / "checkpoints"
    rel_src = str(src.relative_to(root))

    shipped = [w.payload(wf) for wf in
               (weight_file(w, d) for d in checkpoint_dirs(w)) if wf is not None]

    for cand in sorted(root.rglob("*.pt")):
        if cand == src:
            continue
        rel = str(cand.relative_to(root))
        if not DOC_RUN_RE.fullmatch(f"`{rel}`"):
            continue                       # the harvester would not see it
        subs = w.payloads_all(cand)
        if any(sub and all(sub.get(k) == v for k, v in ship.items())
               for ship in shipped if ship for sub in subs):
            continue                       # this one DOES ship; it would not fail
        txt = w.read_text(doc)
        upd = txt.replace(rel_src, rel, 1)
        assert upd != txt, "mutation was a no-op"
        w.text[str(doc)] = upd
        assert any(pp == cand for pp, _, _ in doc_named_runs(w)), "redirect did not take"
        return str(cand)
    raise AssertionError("no unshipped, harvestable run checkpoint to redirect to")


def _mut_p8(w: World) -> str:
    for d in checkpoint_dirs(w):
        m = RELEASED_STEP_RE.search(d.name)
        if not m:
            continue
        cj = d / "config.json"
        cfg = w.read_json(cj)
        key = "global_step" if m.group(1).lower() == "step" else "epoch"
        w.json[str(cj)] = dict(cfg, **{key: int(m.group(2)) + 7})
        assert w.read_json(cj)[key] != cfg.get(key), "mutation was a no-op"
        return str(cj)
    raise AssertionError("no released subdir name encodes a step/epoch")


def _mut_p7(w: World) -> str:
    """Hide the selection-criterion escape hatch is not enough (nobody uses it today);
    instead point a checkpoint at a mid-run origin and check the gate notices."""
    for d in checkpoint_dirs(w):
        cj = d / "config.json"
        cfg = w.read_json(cj)
        origin, _ = find_origin(w, d, cfg)
        if origin is None:
            continue
        run = origin.parent
        sibs = sorted(run.glob("*.pt"))
        key = "global_step" if "global_step" in w.run_meta(origin) else "epoch"
        # Need a PERIODIC dump with a later periodic sibling: that is the shape p7
        # detects, and pointing at anything else would prove nothing about it.
        periodic = [(int(mm.group(1)), s) for s in sibs
                    if (mm := PERIODIC_RE.search(s.stem))]
        if len(periodic) < 2:
            continue
        periodic.sort()
        tgt = periodic[0][1]
        if RELEASED_STEP_RE.search(d.name):
            continue                       # p7 exempts these by design
        w.json[str(cj)] = dict(cfg, source_sha256=w.sha256(tgt),
                               source_path=tgt.name,
                               output_dir=str(run.relative_to(EXPERIMENTS)))
        assert find_origin(w, d, w.read_json(cj))[0] == tgt, "mutation was a no-op"
        return str(cj)
    raise AssertionError("no run with an earlier sibling to point at")


MUTATIONS = {
    "every-checkpoint-has-a-config": _mut_p1,
    "config-declares-a-source-run": _mut_p2,
    "source-run-resolves-outside-the-bundle": _mut_p3,
    "shipped-payload-matches-declared-source": _mut_p4,
    "config-declares-producing-code-revision": _mut_p5,
    "doc-named-run-is-the-one-that-ships": _mut_p6,
    "no-undisclosed-mid-run-checkpoint": _mut_p7,
    "released-name-matches-recorded-step": _mut_p8,
}


def _prestate_p5(w: World) -> None:
    """Give every config a resolvable revision so p5 is GREEN before the fault. A
    check that is already red for all 18 checkpoints cannot be proved by 'it is still
    red'; it has to be driven green first and then broken."""
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert head, f"cannot read HEAD of {REPO}"
    for d in checkpoint_dirs(w):
        cj = d / "config.json"
        w.json[str(cj)] = dict(w.read_json(cj), code_commit=head)


def _mut_p5_strip(w: World) -> str:
    """Delete the revision record outright, leaving producing_code standing.

    This arm reported MISSED for as long as it existed, and the injection was never
    the reason: the override lands (the config really does come back with zero
    REV_FIELDS), the gate really does read it. The gate ACCEPTED it, on an unverified
    producing_code excuse. That was the defect -- see
    _producing_code_substitute_refuted(). The fault is a genuine one (D3 reinstated on
    this checkpoint) and the gate now says so.
    """
    d = checkpoint_dirs(w)[0]
    cj = d / "config.json"
    cfg = {k: v for k, v in w.read_json(cj).items() if k not in REV_FIELDS}
    w.json[str(cj)] = cfg
    assert not any(cfg.get(f) for f in REV_FIELDS), "mutation was a no-op"
    return str(cj)


PRESTATES = {"config-declares-producing-code-revision": _prestate_p5}
# A check may have MORE THAN ONE way to be broken, and p5 has exactly two: the record
# is absent, or the record is present but resolves to nothing. Both are injected and
# BOTH must trip. They used to be registered as MUTATIONS[p5] and STRIP_MUTATIONS[p5],
# and the runner picked `STRIP_MUTATIONS.get(name) or MUTATIONS[name]` -- so
# registering the second one silently retired the first. _mut_p5 was dead code that
# nothing ran, and the arm that did run was the one the gate excused.
STRIP_MUTATIONS = {"config-declares-producing-code-revision": _mut_p5_strip}


def selftest() -> int:
    missed: List[str] = []
    for name, fn in CHECKS:
        pre = PRESTATES.get(name)
        w0 = World()
        if pre:
            pre(w0)
        base = set(fn(w0))
        if pre and base:
            print(f"  MISSED   {name}   (pre-state did not make the check green: "
                  f"{sorted(base)[0]})")
            missed.append(name)
            continue
        # EVERY registered fault for this check, each on a fresh World, and every one
        # of them must trip. Selecting one and discarding the rest is how _mut_p5
        # stopped running.
        faults = [m for m in (STRIP_MUTATIONS.get(name), MUTATIONS.get(name)) if m]
        why: Optional[str] = None
        for mut in faults:
            w = World()
            if pre:
                pre(w)
            try:
                target = mut(w)
            except (AssertionError, StopIteration) as e:
                why = f"fault {mut.__name__} not injectable: {e}"
                break
            if not [d for d in set(fn(w)) - base if target in d]:
                why = f"fault {mut.__name__} on {target} produced no new failure"
                break
        if why:
            print(f"  MISSED   {name}   ({why})")
            missed.append(name)
        else:
            print(f"  TRIPPED  {name}   ({len(faults)} fault(s))")
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
    if not EXPERIMENTS.is_dir():
        print(f"  FAIL  source-roots-present   ({EXPERIMENTS} absent: provenance cannot "
              f"be measured, so nothing here may be read as a pass)")
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

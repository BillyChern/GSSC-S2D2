#!/usr/bin/env python3
r"""GATE: a number measured under a NON-OFFICIAL evaluation protocol must never be quoted bare.

READ THIS BEFORE "FIXING" ANYTHING IT REPORTS
--------------------------------------------------------------------------------------------
The checkpoint behind the paper's BEV secondary-task row records, in its own config.json
(``GSSC-S2D2-assets/checkpoints/bev/bev_s2d2_scpnet/config.json``):

    "measured_miou": 0.3609,
    "measured_base_miou": 0.3475,
    "eval_protocol": "training-time 2D BEV evaluator, 100 fixed val samples (seed 42)
                      -- NOT the 4071-frame semantic-kitti-api protocol"

So 36.1 % is a 100-frame training-time number, not a 4071-frame semantic-kitti-api number. The
release quotes 36.1 in at least six places (README.md:31,45,167-168,273; docs/MODEL_ZOO.md:190,279;
docs/REPRODUCIBILITY.md:247; the assets README and MANIFEST) and, before the release fixes, NONE of
them said so. Worse, several of them point the reader at
``data/checkpoints/bev/bev_perception_net/model.safetensors``, whose own config.json declares
``best_miou = 0.2422`` and is a 938K-parameter refinement net -- it is not the model that produced
36.09 at all, and the BEV evaluator could not even load it.

THE TRAP, and why this gate exists rather than a one-line doc edit: repointing those commands at
``bev/bev_s2d2_scpnet/`` makes the docs LOOK right and makes them MORE wrong -- a reader would
then run the named checkpoint, score it with the official 4071-frame api, get something else, and
have no way to know why. Naming the right checkpoint without naming the protocol converts a
broken pointer into a reproducibility claim that cannot hold. Hence: disclosure, not repointing,
is the primitive this gate enforces, and R4 fails a block that asserts the OFFICIAL protocol for
one of these numbers even though every other rule here would pass it.

WHAT IT CHECKS
--------------------------------------------------------------------------------------------
 R1 protocol-inventory     every config.json under the assets checkpoint tree is read; those
                           declaring an ``eval_protocol`` that is not the official one are
                           enumerated LIVE (no hardcoded list), and the affirmative frame count
                           must be parseable out of each such string. If the parse fails, every
                           later rule is blind and this fires instead of passing.
 R2 nonofficial-disclosed  every doc block quoting a value that a non-official-protocol
                           checkpoint declares must state, in that same block, both a protocol
                           and a frame/sample count.
 R3 bev-miou-disclosed     the same rule for any BEV mIoU quoted anywhere in the release, whether
                           or not a config declares that exact value. The BEV pipeline has no
                           official-protocol number at all, so a bare BEV mIoU is unqualified by
                           construction.
 R4 no-official-claim      no text WITHIN 200 CHARS of such a value may attach the official
                           semantic-kitti-api / 4071-frame protocol to it. This is the rule that
                           makes a naive fix fail. It ships GREEN: it is a trap for the edit that
                           repoints the docs at bev_s2d2_scpnet and calls the number official.
                           Scope is deliberately local, not block: README.md:43-45 is a single
                           changelog block whose first bullet says "official semantic-kitti-api"
                           about the JS3C row, and a block-scoped rule accused the wrong bullet.
 R5 frame-count-agrees     where a frame count sits beside such a value (same 200-char window),
                           it must be the count in the checkpoint's own eval_protocol string.
                           Also ships green; block scope was tried and read the ASCII dataset
                           tree's "23,201 frames" as a disclosure about a BEV mIoU.
 R6 pointer-declares-value where a doc line pairs a checkpoint path with an expected mIoU and
                           the value is declared by a DIFFERENT shipped checkpoint, the command
                           names the wrong model. Live: three sites tell the reader to run
                           bev/bev_perception_net and expect 36.1 %, which only
                           bev/bev_s2d2_scpnet declares. The rule ABSTAINS when no shipped config
                           declares the value at all -- test scores, TTA rows and protocol
                           variants live in no config.json, and demanding otherwise produced 14
                           findings that were not defects.

WHAT COUNTS AS DISCLOSURE
--------------------------------------------------------------------------------------------
Both halves, in the same block: a PROTOCOL NAME (semantic-kitti-api, "training-time evaluator",
"internal evaluator", SSCMetrics) and a FRAME COUNT (an integer immediately qualifying
frames/samples/scans). Naming one without the other is what the release already did -- "36.1 %
BEV mIoU on val seq 08" names a split, which is not a protocol, and no count at all.

SCOPE
--------------------------------------------------------------------------------------------
 * A checkpoint with no ``eval_protocol`` key is NOT assumed official; it is UNDECLARED, counted,
   and printed on every run. The absence of a claim is not a claim -- but it is also not
   evidence, and this gate says which checkpoints it could say nothing about.
 * ``_superseded_*`` directories are skipped: they are kept locally, not shipped, and their
   configs would double-count every value.
 * The paper is READ-ONLY here; this gate does not open it. Whether a number is IN the paper is
   check_paper_numbers.py's question, not this one.

USAGE
    python .release_checks/check_protocol_disclosure.py
    python .release_checks/check_protocol_disclosure.py --selftest
"""
from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence, Tuple

REPO = Path("/workspace/GSSC-S2D2")
ASSETS = Path("/workspace/GSSC-S2D2-assets")
CKPT_ROOT = ASSETS / "checkpoints"

#: Keys that carry a measured percentage. Fractions (<= 1.0) are scaled; percents are kept.
MIOU_KEYS = ("measured_miou", "measured_base_miou", "best_miou", "expected_miou", "miou")

OFFICIAL = re.compile(r"semantic[-_ ]kitti[-_ ]api|official\s+(?:api|scorer|evaluator|protocol)", re.I)
PROTOCOL_WORDS = re.compile(
    r"semantic[-_ ]kitti[-_ ]api|training[- ]time\s+(?:2D\s+)?(?:BEV\s+)?evaluator|"
    r"internal\s+(?:training[- ]time\s+)?evaluator|SSCMetrics|eval_protocol", re.I)
#: A count only counts when it QUALIFIES frames/samples/scans -- "seq 08" is a split, not a count.
FRAME_COUNT = re.compile(r"(\d[\d,]*)[- ](?:fixed[- ])?(?:val[- ])?(?:frame|sample|scan)s?\b", re.I)
BEV_MENTION = re.compile(r"\bBEV\b|bev_", re.I)
NUMBER = re.compile(r"(?<![\w.])([+\-−]?)\s?(\d{1,3}\.\d{1,2})(?![\d.])")
METRIC_MARK = re.compile(r"%|\bpp\b|\bmIoU\b|\bIoU\b", re.I)
CKPT_PATH = re.compile(r"(?:data/)?checkpoints/([A-Za-z0-9_]+/[A-Za-z0-9_]+)/")


def doc_sources() -> List[Path]:
    out = [REPO / "README.md"]
    out += sorted((REPO / "docs").glob("*.md"))
    out += sorted(REPO.glob("configs/*/*.yaml"))
    out += [ASSETS / "MANIFEST.txt", ASSETS / "README.md", ASSETS / "checkpoints" / "MANIFEST.txt"]
    return [p for p in out if p.is_file()]


class Ckpt(NamedTuple):
    rel: str                       # e.g. "bev/bev_s2d2_scpnet"
    path: Path
    protocol: str                  # "" when the config declares none
    frames: str                    # affirmative frame count from the protocol string, "" if none
    values: Dict[str, str]         # metric key -> percent, 2 dp

    @property
    def official(self) -> bool:
        """Official only when it says so AND does not disown it ("NOT the ... api protocol")."""
        return bool(OFFICIAL.search(self.affirmative)) if self.protocol else False

    @property
    def affirmative(self) -> str:
        """The part of the protocol string before it starts saying what it is NOT.

        Load-bearing: the live string ends '-- NOT the 4071-frame semantic-kitti-api protocol'.
        Reading the whole string would find both 'semantic-kitti-api' and '4071 frames' in it and
        conclude the checkpoint was scored officially on 4071 frames -- the exact inversion this
        gate exists to prevent, produced by the gate itself.
        """
        return re.split(r"--|\bNOT\b|\bnot\b", self.protocol)[0]


class Result:
    def __init__(self, name: str) -> None:
        self.name, self.findings, self.notes = name, [], []

    def fail(self, detail: str) -> None:
        self.findings.append(detail)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    @property
    def ok(self) -> bool:
        return not self.findings


def pct(v) -> str:
    """A metric value as a percent string with 2 dp. Fractions <= 1.0 are scaled by 100."""
    d = Decimal(str(v))
    if d <= 1:
        d *= 100
    return str(d.quantize(Decimal("1.00"), rounding=ROUND_HALF_UP))


def renderings(percent: str) -> List[str]:
    """The 2-dp value and its ROUND_HALF_UP renderings at 1 dp -- how a doc may legitimately print it."""
    out = [percent, str(Decimal(percent).quantize(Decimal("1.0"), rounding=ROUND_HALF_UP))]
    return sorted(set(out))


def load_ckpts(root: Path = CKPT_ROOT) -> Tuple[List[Ckpt], List[str]]:
    out, broken = [], []
    for cfg in sorted(root.rglob("config.json")):
        if any(part.startswith("_superseded") for part in cfg.parts):
            continue                      # kept locally, not shipped
        try:
            d = json.loads(cfg.read_text())
        except Exception as exc:          # noqa: BLE001 -- the finding IS that it did not parse
            broken.append(f"{cfg}: {exc}")
            continue
        vals = {}
        for k in MIOU_KEYS:
            if isinstance(d.get(k), (int, float)):
                vals[k] = pct(d[k])
        proto = str(d.get("eval_protocol", "") or "")
        c = Ckpt(str(cfg.parent.relative_to(root)), cfg, proto, "", vals)
        m = FRAME_COUNT.search(c.affirmative)
        out.append(c._replace(frames=m.group(1).replace(",", "") if m else ""))
    return out, broken


class Block(NamedTuple):
    path: Path
    first: int
    lines: List[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def line_of(self, idx: int) -> int:
        return self.first + idx


def blocks(paths: Sequence[Path]) -> List[Block]:
    out = []
    for p in paths:
        lines = p.read_text(errors="replace").splitlines()
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            j = i
            while j < len(lines) and lines[j].strip():
                j += 1
            out.append(Block(p, i + 1, lines[i:j]))
            i = j
    return out


NEAR = 200        # chars; how close a protocol claim must sit to be a claim ABOUT this value


def near(text: str, value: str) -> str:
    """The text within NEAR chars of any occurrence of `value` in `text`."""
    pat = re.compile(r"(?<![\d.])" + re.escape(value) + r"(?![\d.])")
    return " ".join(text[max(0, m.start() - NEAR):m.end() + NEAR] for m in pat.finditer(text))


def discloses(text: str) -> Tuple[bool, bool]:
    return bool(PROTOCOL_WORDS.search(text)), bool(FRAME_COUNT.search(text))


def value_sites(blk: Block, wanted: Sequence[str]) -> List[Tuple[int, str, str]]:
    """[(line, printed value, line text)] for any rendering of any wanted percent."""
    hits = []
    for off, line in enumerate(blk.lines):
        for m in NUMBER.finditer(line):
            if m.group(1):
                continue                                    # a delta is not a level
            if not METRIC_MARK.search(line[max(0, m.start() - 30):m.end() + 40]):
                continue
            if m.group(2) in wanted:
                hits.append((blk.line_of(off), m.group(2), line.strip()))
    return hits


# --------------------------------------------------------------------------------------------
def run(paths: Sequence[Path], ckpts: Sequence[Ckpt], broken: Sequence[str]) -> List[Result]:
    r1 = Result("protocol-inventory")
    r2 = Result("nonofficial-disclosed")
    r3 = Result("bev-miou-disclosed")
    r4 = Result("no-official-claim")
    r5 = Result("frame-count-agrees")
    r6 = Result("pointer-declares-value")

    for b in broken:
        r1.fail(f"{b} -- a checkpoint config that does not parse is a checkpoint whose protocol "
                f"nobody can check")
    declared = [c for c in ckpts if c.protocol]
    nonofficial = [c for c in declared if not c.official]
    r1.note(f"{len(ckpts)} shipped checkpoint config(s); {len(declared)} declare an eval_protocol, "
            f"{len(nonofficial)} of those are NOT the official protocol; "
            f"{len(ckpts) - len(declared)} declare none (UNDECLARED, not assumed official)")
    for c in nonofficial:
        r1.note(f"  non-official: {c.rel} frames={c.frames or '??'} values="
                f"{sorted(set(c.values.values()))} :: {c.protocol[:90]}")
        if not c.frames:
            r1.fail(f"{c.path}: eval_protocol declares a non-official protocol but no frame or "
                    f"sample count can be parsed from '{c.affirmative[:60]}' -- R2/R5 cannot "
                    f"judge disclosure against it, so this gate is blind here")
    if not ckpts:
        r1.fail(f"{CKPT_ROOT}: no config.json found; the asset tree moved and this gate is "
                f"judging nothing")

    # value -> the non-official checkpoints that declare it
    owners: Dict[str, List[Ckpt]] = {}
    for c in nonofficial:
        for v in c.values.values():
            for r in renderings(v):
                owners.setdefault(r, []).append(c)

    bev_ckpts = [c for c in ckpts if c.rel.startswith("bev/")]
    bev_values = {r for c in bev_ckpts for v in c.values.values() for r in renderings(v)}

    n2 = n3 = 0
    for blk in blocks(paths):
        has_proto, has_frames = discloses(blk.text)

        # R2 -- values a non-official checkpoint declares.
        for line, val, text in value_sites(blk, list(owners)):
            n2 += 1
            cs = owners[val]
            who = ", ".join(c.rel for c in cs)
            if not (has_proto and has_frames):
                miss = ("no protocol and no frame count" if not (has_proto or has_frames)
                        else "no frame count" if has_proto else "no protocol name")
                r2.fail(f"{blk.path}:{line}: quotes {val} %, which {who} measured under "
                        f"'{cs[0].affirmative.strip()[:60]}', and the block gives {miss} | "
                        f"{text[:110]}")
            # R4/R5 read a WINDOW around the value, not the whole block. Block scope was tried
            # first and misattributes: README.md:43-45 is one changelog block whose FIRST bullet
            # says "official semantic-kitti-api" about the JS3C row and whose THIRD bullet quotes
            # the BEV 36.1; and the assets README's ASCII tree puts the dataset frame counts
            # (23,201 / 32,039 / 57,650) in the same block as a BEV line. Both produced findings
            # that named the wrong sentence. Disclosure (R2) stays block-scoped -- a reader takes
            # the protocol from the surrounding block -- but an ACCUSATION must be local.
            win = near(blk.text, val)
            if OFFICIAL.search(win) and not re.search(r"NOT\s+the\s+\d", win, re.I):
                seg = OFFICIAL.search(win)
                r4.fail(f"{blk.path}:{line}: quotes {val} % and attaches the OFFICIAL protocol "
                        f"('{seg.group(0)}') to it, but {who} records "
                        f"'{cs[0].protocol.strip()[:80]}'. Naming the right checkpoint without "
                        f"the right protocol is a worse claim, not a fix | {text[:90]}")
            # R5 -- a disclosed count must be the checkpoint's own.
            got = {m.group(1).replace(",", "") for m in FRAME_COUNT.finditer(win)}
            want = {c.frames for c in cs if c.frames}
            if got and want and not (got & want):
                r5.fail(f"{blk.path}:{line}: quotes {val} % beside frame count(s) {sorted(got)}, "
                        f"but {who} was measured on {sorted(want)} | {text[:100]}")

        # R3 -- any BEV mIoU, declared or not.
        if not BEV_MENTION.search(blk.text):
            continue
        for off, line_text in enumerate(blk.lines):
            for m in NUMBER.finditer(line_text):
                if m.group(1):
                    continue
                win = line_text[max(0, m.start() - 60):m.end() + 60]
                if not (METRIC_MARK.search(win) and BEV_MENTION.search(win)):
                    continue
                if m.group(2) not in bev_values and m.group(2) not in owners:
                    continue
                n3 += 1
                if not (has_proto and has_frames):
                    r3.fail(f"{blk.path}:{blk.line_of(off)}: quotes BEV mIoU {m.group(2)} % with "
                            f"{'no frame count' if has_proto else 'no protocol statement'}; the "
                            f"BEV pipeline has no official-protocol number, so a bare value here "
                            f"cannot be reproduced | {line_text.strip()[:110]}")
    r2.note(f"{n2} site(s) quote a value declared by a non-official-protocol checkpoint")
    r3.note(f"{n3} BEV mIoU quotation(s) judged; BEV checkpoint values seen: {sorted(bev_values)}")

    # R6 -- a command that names a checkpoint and an expected number.
    by_rel = {c.rel: c for c in ckpts}
    n6, abst6 = 0, [0]
    for blk in blocks(paths):
        for off, line_text in enumerate(blk.lines):
            paths_on_line = CKPT_PATH.findall(line_text)
            if not paths_on_line:
                continue
            # The expectation may sit in the same table row as the command, which is how the
            # assets README and the repro matrix are written.
            vals = [m.group(2) for m in NUMBER.finditer(line_text)
                    if not m.group(1) and METRIC_MARK.search(
                        line_text[max(0, m.start() - 30):m.end() + 40])]
            for rel in paths_on_line:
                c = by_rel.get(rel)
                if c is None or not c.values:
                    continue
                declared_r = {r for v in c.values.values() for r in renderings(v)}
                for v in vals:
                    n6 += 1
                    if v in declared_r:
                        continue
                    # A checkpoint's config declares ONE measurement (its val number). Docs
                    # legitimately quote test scores, TTA rows and protocol variants beside the
                    # same checkpoint, and demanding the config declare those produced 14
                    # findings that were not defects. The precise signal is: this value is
                    # declared by a DIFFERENT shipped checkpoint -- i.e. the doc named the wrong
                    # model. That is the bev_perception_net / bev_s2d2_scpnet swap exactly.
                    elsewhere = [o.rel for o in ckpts if o.rel != rel
                                 and v in {r for x in o.values.values() for r in renderings(x)}]
                    if not elsewhere:
                        n6 -= 1
                        abst6[0] += 1
                        continue
                    r6.fail(f"{blk.path}:{blk.line_of(off)}: names checkpoint '{rel}' and expects "
                            f"{v} %, but '{rel}' declares only {sorted(declared_r)} while "
                            f"{elsewhere} declares {v} -- the command points at the wrong model | "
                            f"{line_text.strip()[:110]}")
    r6.note(f"{n6} (checkpoint, expected value) pairing(s) judged against config.json; "
            f"{abst6[0]} ABSTAINED because no shipped checkpoint declares that value at all "
            f"(test scores, TTA rows and protocol variants live in no config.json)")
    return [r1, r2, r3, r4, r5, r6]


def report(results: Sequence[Result]) -> int:
    nfail = 0
    for r in results:
        for n in r.notes:
            print(f"        note  {n}")
        if r.ok:
            print(f"  PASS  {r.name}")
        else:
            nfail += 1
            for d in r.findings:
                print(f"  FAIL  {r.name}   ({d})")
    print("OK: 0 failing check(s)" if not nfail else f"FAILED: {nfail} failing check(s)")
    return 0 if not nfail else 1


# --------------------------------------------------------------------------------------------
def selftest() -> int:
    import os
    import shutil
    import tempfile

    base = os.environ.get("TMPDIR") or tempfile.gettempdir()
    tmp = Path(tempfile.mkdtemp(prefix="protocol_", dir=base))
    missed: List[str] = []
    live, broken = load_ckpts()

    def write(rel: str, text: str) -> Path:
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def trip(name: str, paths, ckpts=live, brk=(), label=None) -> None:
        res = {r.name: r for r in run(paths, ckpts, brk)}
        shown = label or name
        if name in res and not res[name].ok:
            print(f"  TRIPPED  {shown}")
        else:
            print(f"  MISSED   {shown}")
            missed.append(shown)

    fake = Ckpt("bev/fake", tmp / "config.json",
                "training-time 2D BEV evaluator, 100 fixed val samples (seed 42) -- NOT the "
                "4071-frame semantic-kitti-api protocol", "100", {"measured_miou": "36.09"})
    assert not fake.official, "affirmative-only reading broke: the fixture read as official"
    assert fake.frames == "100"

    # R1 -- a config.json that does not parse.
    trip("protocol-inventory", [], live, [f"{tmp}/config.json: Expecting value"])

    # R1b -- a non-official protocol whose frame count cannot be parsed blinds R2/R5.
    blind = fake._replace(protocol="training-time evaluator on a fixed subset", frames="")
    assert blind.frames != fake.frames, "R1b fault did not perturb the checkpoint record"
    trip("protocol-inventory", [], [blind], label="protocol-inventory (unparseable count arm)")

    # R2 -- a bare quotation.
    f2 = write("r2.md", "BEV\n\nThe pipeline reaches **36.09 %** BEV mIoU on val seq 08.\n")
    trip("nonofficial-disclosed", [f2], [fake])
    # ... and the disclosed version must NOT trip it.
    ok2 = write("r2ok.md", "BEV\n\nThe pipeline reaches **36.09 %** BEV mIoU under the "
                           "training-time 2D BEV evaluator on 100 fixed val samples (seed 42).\n")
    res = {r.name: r for r in run([ok2], [fake], [])}
    if res["nonofficial-disclosed"].ok:
        print("  TRIPPED  nonofficial-disclosed/negative-control")
    else:
        print("  MISSED   nonofficial-disclosed/negative-control")
        missed.append("nonofficial-disclosed/negative-control")

    # R3 -- a BEV mIoU with a protocol word but no count.
    f3 = write("r3.md", "BEV\n\nUnder the training-time evaluator the BEV mIoU is 36.09 %.\n")
    trip("bev-miou-disclosed", [f3], [fake])

    # R4 -- the naive fix: right checkpoint, wrong protocol.
    f4 = write("r4.md", "BEV\n\nRun bev_s2d2_scpnet and expect 36.09 % BEV mIoU under the "
                        "official semantic-kitti-api on 4071 frames.\n")
    trip("no-official-claim", [f4], [fake])

    # R5 -- disclosed count that is not the checkpoint's.
    f5 = write("r5.md", "BEV\n\nThe 36.09 % BEV mIoU comes from the training-time evaluator over "
                        "500 val frames.\n")
    trip("frame-count-agrees", [f5], [fake])

    # R6 -- a command naming a checkpoint that declares a different number. Uses the LIVE tree so
    # the fixture cannot drift from the shipped layout.
    # The fault must be a value ANOTHER checkpoint declares. 99.87 was tried first and is not
    # usable: R6 abstains on a value no shipped config declares at all (that is the test-score
    # case), so the fixture proved nothing -- a selftest that goes vacuous exactly where the rule
    # is most permissive.
    with_vals = [c for c in live if c.values]
    assert len(with_vals) >= 2, "R6 fixture: need two checkpoints that declare values"
    c0 = with_vals[0]
    own = {r for v in c0.values.values() for r in renderings(v)}
    swap = next((r for c in with_vals[1:] for v in c.values.values()
                 for r in renderings(v) if r not in own), None)
    assert swap, "R6 fixture: no other checkpoint declares a value c0 does not"
    f6 = write("r6.md", f"Zoo\n\n| run | `python scripts/eval.py --checkpoint "
                        f"data/checkpoints/{c0.rel}/model.safetensors` | **{swap} %** mIoU |\n")
    trip("pointer-declares-value", [f6], live)

    shutil.rmtree(tmp, ignore_errors=True)
    total = 8
    print(f"SELFTEST OK: {total - len(missed)}/{total} checks provably fail when broken"
          if not missed else
          f"SELFTEST FAILED: {len(missed)} check(s) did not fail when broken: {missed}")
    return 1 if missed else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    _c, _b = load_ckpts()
    raise SystemExit(report(run(doc_sources(), _c, _b)))

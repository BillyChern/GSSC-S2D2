#!/usr/bin/env python3
r"""GATE: a release doc must not present, as a PAPER value, a number the paper never prints.

WHY THIS EXISTS -- the defects it was built from, all confirmed by measurement on 2026-08-20
--------------------------------------------------------------------------------------------
LINE NUMBERS IN THIS BLOCK ARE A DATED SNAPSHOT of the checkout named above, not
navigation. Several have already moved: follow the SYMBOL, the heading or the quoted
text, and re-derive the location with `grep -n`. Every check below RE-MEASURES the
live artefacts, so nothing here is load-bearing for a verdict.

 1. The asset bundle's ``checkpoints/MANIFEST.txt`` stated "the 26.1% / +3.3 pp value is the paper headline
    ... It is the value cited in the abstract and conclusion".  ``26.1``, ``26.05`` and ``+3.3``
    occur ZERO times in main.pdf and supplementary.pdf.  The paper's JS3C-Net cross-base row is
    24.3 / +1.6.  The claim is not a rounding of anything the paper prints.
 2. ``README.md:351`` reports the retrain motorcyclist delta as ``+4.4`` and cites
    ``tab:supp_retrain_deltas`` for it; ``+4.4`` occurs ZERO times in either PDF.
 3. ``README.md:190`` reports SemanticPOSS ``1.0 -> 6.6 mIoU (+5.5)``.  The paper prints
    ``1.0 -> 6.5``.  ``6.6`` DOES occur in both PDFs -- as a per-class cell (JS3C-Net building
    36.6, person 36.6->0.7) and as a phantom-voxel ratio (6.6x).  A membership test alone is
    therefore VACUOUS on this defect, which is why R3 (scope localisation) exists: 6.6 never
    occurs within 1200 characters of any "SemanticPOSS"/"POSS" mention in either PDF, while 6.5,
    1.0, 31.8 and 54.9 each occur near 8-12 of them.  The doc's own arithmetic is the second
    witness: 6.6 - 1.0 = +5.6, but the doc says +5.5, which is 6.5 - 1.0 (R4).
 4. The superseded BEV pair ``34.3`` / ``+1.8`` survives in ``README.md:167`` and
    ``docs/MODEL_ZOO.md:190``.  The paper's BEV secondary task reads 34.8 -> 36.1 (+1.3);
    ``34.3`` occurs ZERO times in either PDF.  (The checkpoint's own config.json says the base is
    34.75, i.e. 34.8 -- see check_protocol_disclosure.py, which owns the protocol half of this.)

SOURCE SET WIDENED 2026-08-22, because a gate only judges what it reads. It read README.md,
docs/*.md and the two assets manifests. CITATION.cff's abstract, CONTRIBUTING.md, SECURITY.md,
the issue templates, the workflow files, the four per-dataset asset READMEs and the two Hugging
Face cards -- 18 more public surfaces -- were invisible to all sixteen gates. Each was measured
ALONE against this gate before being added; 16 came back green and are now judged, and the two
that did not are in DOC_SOURCE_EXCLUSIONS with their finding counts and are PRINTED on every
run. `_selftest`'s `widened-source:*` arms plant a fabricated paper value in each new family
and require it to be caught, so a widening that lists paths without judging them cannot pass.

PROGRESS SIGNAL, observed while this gate was being written: the assets ``MANIFEST.txt`` was
regenerated mid-session by ``make_manifest.py`` and the false 26.1 / +3.3 headline claim
disappeared from it, along with several of the sites listed above. The gate's live finding count
dropped in step. That is the intended behaviour -- the evidence above is kept because it is why
the rules are shaped this way, not because the sites are all still live.

WHAT IT CHECKS
--------------------------------------------------------------------------------------------
 R1 sources-readable        every doc source exists and is non-empty (a gate that silently stops
                            reading a file it was written for is worse than no gate).
 R2 pdf-instrument-live     the PDFs parse, yield thousands of numeric tokens, and CONTROL_VALUES
                            -- values a human has confirmed the paper prints -- are all found.
                            Without this, a broken extractor reports "all values matched" on an
                            empty haystack. It is the positive control on the instrument.
 R3 paper-values-in-pdf     every paper-anchored, metric-shaped value must appear in a PDF, at its
                            own precision or as a coarser ROUND_HALF_UP of it (24.32 -> 24.3 is a
                            match; 26.05 -> 26.1/26.0 is not, because neither is printed).
 R4 signed-deltas-in-pdf    a signed delta must appear signed in a PDF, or reconcile with an
                            "a -> b" pair in its own clause whose BOTH endpoints already matched.
                            Derivation from a rejected endpoint grants nothing.
 R5 delta-arithmetic        where a doc prints both "a -> b" and a delta, b - a must equal the
                            delta to 0.05 pp. This is a check on the doc against ITSELF and needs
                            no paper at all; it is what catches a single mistyped endpoint.
 R6 scope-localised         a value whose block names a scope key (a distinctive proper noun that
                            also occurs in the PDFs, e.g. SemanticPOSS, KITTI-360, LMSCNet) must
                            occur in a PDF WINDOW around some occurrence of one of those keys.
                            This is the rule that sees defect 3. It ABSTAINS when a block has no
                            distinctive key, and says so on every run.
 R7 headline-claims         a clause that asserts paper-headline status for a value ("is the paper
                            headline", "cited in the abstract", "the paper's headline is") is NOT
                            eligible for the disclaimer exemption of R3. A hedge sitting two
                            clauses away must not launder an assertion.

THE FILTERS, and why they are printed on every run
--------------------------------------------------------------------------------------------
A gate's filters are where defects hide, so both filters are reported as counts and lists:

 * TRIGGER (block scope): a value is judged only if its BLOCK -- the run of contiguous non-blank
   lines around it -- carries a paper anchor ("paper", "Tab.", "Fig.", "tab:", "\label", "supp",
   "SS") AND a metric mark (%, pp, mIoU, IoU). Block scope, not line scope, is deliberate:
   docs/MODEL_ZOO.md:190 is a bare markdown row, "| **36.1** (34.3 base + 1.8 refinement) |",
   whose paper anchor lives in the table's header two lines up. A line-scoped trigger cannot see
   defect 4 at all.
 * EXEMPTION (clause scope + 1): a value is skipped when its own clause, or the next one, says
   the number is NOT a paper value ("internal", "diagnostic", "not tabulated", "appears nowhere",
   "superseded", "companion", "earlier revisions ..."). Clause scope, not block scope, is
   deliberate in the other direction: the MANIFEST paragraph that makes the false 26.1 claim ALSO
   contains the word "internal" 230 characters later, describing a DIFFERENT number, and a
   block-scoped exemption would swallow the defect this gate exists for. The +1 clause of slack
   is what lets docs/REPRODUCIBILITY.md:324-325 ("expect 26.05 % ... This is the GT-BEV
   DIAGNOSTIC, not the paper's headline") pass honestly. R7 is the backstop for the leak that
   slack opens.

WHAT IT DOES NOT DO
--------------------------------------------------------------------------------------------
 * It does not judge whether a number that IS in the paper is the RIGHT number for the claim,
   except through R6's necessary condition. "Near a scope key" is not "in the correct row".
 * It judges only metric-shaped values (a decimal, with a metric mark within +-40 chars).
   Integers, sizes, versions, dates, step counts and hyperparameters are out of scope BY
   CONSTRUCTION, and the run prints how many tokens each filter discarded.
 * The paper is READ-ONLY here. This gate opens the PDFs and nothing else.

USAGE
    python .release_checks/check_paper_numbers.py
    python .release_checks/check_paper_numbers.py --selftest

ROOTS, AND WHAT IS NOT PART OF THE PUBLIC RELEASE
-------------------------------------------------
Every root below is an environment variable with a repo-relative default, so this gate
measures the checkout it ships in rather than one particular machine.  Absolute paths
were hardcoded here once; a relocated clone then audited a tree it was not running in,
and the paths themselves disclosed the maintainer's local layout to every visitor.

    GSSC_REPO        the release checkout under test        default: this file's repository
    GSSC_ASSETS      the asset staging bundle               default: <repo>/../GSSC-S2D2-assets
    GSSC_PAPER       the manuscript checkout                default: <repo>/../GSSC-paper

THE ASSET STAGING BUNDLE AND THE MANUSCRIPT CHECKOUT ARE NOT PART OF THE PUBLIC RELEASE.
They are maintainer working trees; a clone of this repository does not contain them, and the
released artefacts are distributed separately (docs/DATASET.md, docs/MODEL_ZOO.md).
A gate that needs one and cannot find it FAILS rather than passing: "the artefact is
not here" is not evidence that it is correct.  Point the variable at your own copy,
or skip the gate.
"""
from __future__ import annotations

import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence, Tuple

REPO = Path(os.environ.get("GSSC_REPO") or Path(__file__).resolve().parents[1])
ASSETS = Path(os.environ.get("GSSC_ASSETS") or REPO.parent / "GSSC-S2D2-assets")
PAPER = Path(os.environ.get("GSSC_PAPER") or REPO.parent / "GSSC-paper")
PAPER_PDFS = (
    PAPER / "pdf" / "main.pdf",
    PAPER / "pdf" / "supplementary.pdf",
)

#: Doc sources, in the order a reader meets them. Globs are expanded live: a new docs/*.md must
#: come under judgement without editing this gate.
#: Surfaces DELIBERATELY not judged, each with the measurement that put it here. A silent
#: omission is how "the SemanticPOSS 6.6 line survived several sweeps" happens; a NAMED one is
#: printed by R1 on every run, so the hole is visible to whoever reads the output.
#: MEASURED 2026-08-22 by pointing doc_sources() at each file alone and running main():
DOC_SOURCE_EXCLUSIONS: Tuple[Tuple[str, str], ...] = (
    ("CHANGELOG.md",
     "20 findings, and every one sampled is the release note QUOTING the string it removed -- "
     "e.g. a '### Fixed -- SemanticPOSS zero-shot read 6.6 where the paper prints 6.5' section "
     "whose body lists `docs/TRAIN.md:67 \"**26.05 %** (paper ... headline)\"` as the residue "
     "that was deleted. A changelog's job is to quote the wrong value; a gate that fails on "
     "that trains people to ignore it. Judging this file needs a QUOTATION rule (a value inside "
     "a quoted fragment under a Fixed/Removed heading is a record, not a claim), which does not "
     "exist yet. Until it does, the file is unjudged and this line says so."),
    ("hf_cards/THIRD_PARTY_NOTICES.md",
     "8+ findings, all of them SOURCE-OVERLAP RATIOS ('165 matched lines (52.5 % of our file)') "
     "anchored only by that document's own internal section-sign cross-references, which "
     "PAPER_ANCHOR reads as paper pointers. Measured before excluding: `grep -nE "
     "'mIoU|IoU|paper (prints|reports|headline)'` over the file returns ONE line, and it makes "
     "no numeric claim -- the file publishes no metric about the paper, so excluding it loses "
     "no coverage. check_protocol_disclosure and check_paper_labels still read it."),
)


def doc_sources() -> List[Path]:
    """Every PUBLIC prose surface that can present a paper value.

    WIDENED 2026-08-22. It read README.md, docs/*.md and the two assets manifests, which is why
    a retracted number could sit unjudged in CITATION.cff, CONTRIBUTING.md, SECURITY.md, the
    issue templates, the workflow files, the per-dataset asset READMEs and the Hugging Face
    cards -- all of them read by strangers. Every file added below was measured ALONE against
    this gate first and came back green; the two that did not are in DOC_SOURCE_EXCLUSIONS with
    their finding counts, not dropped in silence.
    """
    out = [REPO / "README.md", REPO / "CITATION.cff", REPO / "CONTRIBUTING.md",
           REPO / "SECURITY.md"]
    out += sorted((REPO / "docs").glob("*.md"))
    out += sorted((REPO / ".github").rglob("*.md"))
    out += sorted((REPO / ".github").rglob("*.yml"))
    out += [ASSETS / "MANIFEST.txt", ASSETS / "README.md", ASSETS / "checkpoints" / "MANIFEST.txt"]
    out += sorted(ASSETS.glob("datasets/*/README.md"))
    out += sorted(ASSETS.glob("hf_cards/*.md"))
    excluded = {name for name, _why in DOC_SOURCE_EXCLUSIONS}
    keep = []
    for q in out:
        rel = str(q)
        if any(rel.endswith("/" + name) or q.name == name for name, _w in DOC_SOURCE_EXCLUSIONS):
            continue
        keep.append(q)
    assert excluded, "the exclusion table must never be silently emptied"
    return [q for q in keep if q.is_file()]


#: Values a human opened the PDFs and confirmed. R2 fails if the extractor cannot find them --
#: this is the positive control that keeps R3 from passing against an empty haystack.
CONTROL_VALUES: Tuple[str, ...] = ("38.54", "38.8", "36.17", "24.3", "16.6", "34.8", "36.1", "6.5")

PAPER_ANCHOR = re.compile(r"\bpapers?\b|\bTab\.|\bFig\.|\b(?:tab|fig|sec|eq|alg):|\\label|\bsupp\w*\b|§", re.I)
METRIC_MARK = re.compile(r"%|\bpp\b|\bmIoU\b|\bIoU\b", re.I)

#: A metric-shaped literal: 1-3 integer digits, 1-2 decimals, optional sign. The trailing
#: (?![\d.]) is load-bearing -- without it "v2.3.8" yields "2.3" and every version pin in the
#: docs becomes a fake paper claim.
NUMBER = re.compile(r"(?<![\w.])([+\-−]?)\s?(\d{1,3}\.\d{1,2})(?![\d.])")

#: Units that make a literal a size/latency/count rather than a metric. Checked on the 8
#: characters that FOLLOW the literal.
NON_METRIC_UNIT = re.compile(r"^\s*(?:MB|GB|KB|TB|[GM]iB|ms\b|s\b|h\b|Hz|FPS|M\b|K\b|B\b|[x×]\b|GPU|GHz|W\b)", re.I)
VERSION_CONTEXT = re.compile(r"(?:\bv|version|spconv|torch|cuda|python|CUDA|Ubuntu)\s*$", re.I)

#: Phrases that mark a number as explicitly NOT a paper value. Matched on the value's clause and
#: the next one. Every hit is listed on stdout: an exemption nobody can see is a hole.
DISCLAIMER = re.compile(
    r"not tabulat|not in any table|\binternal\b|\bcompanion\b|diagnostic|unreported|"
    r"does not print|does NOT print|appears \*{0,2}nowhere|supersed|previously said|"
    r"earlier revision|one-off|not protocol-matched|\bno table\b|continuity",
    re.I,
)

#: Values ATTRIBUTED TO ANOTHER WORK are out of scope: this gate judges our docs against OUR
#: paper. Measured need: docs/REPRODUCIBILITY.md:123-125 quotes a third-party reproduction study
#: ("in their Semantic KITTI val Table 1 ... 22.44% ... 27.89% ... a 5.45% gap"). 27.89 rounds to
#: 27.9, which our paper happens to print elsewhere, so R3 passed it and R6 then reported it as
#: printed-about-something-else -- a true statement about the wrong question.
THIRD_PARTY = re.compile(
    r"they report|their (?:Table|reimplementation|footnote|paper)|arXiv:|\bet al\b|"
    r"third-party|published (?:number|value|score)|\breads\b.{0,40}\bin their\b", re.I)

#: Clauses that ASSERT paper-headline status. R7 refuses the disclaimer exemption for these.
HEADLINE_CLAIM = re.compile(
    r"is the paper headline|is the paper's headline|paper headline is|cited in the abstract|"
    r"the paper cites|paper's \w+ headline|as its \w+ headline",
    re.I,
)

#: Clause boundaries. Sentence splitting on "." alone is wrong here: "semantic-kitti-api." and
#: "Tab. III" both fire it. Table pipes and em-dashes are real boundaries in these docs.
CLAUSE_SPLIT = re.compile(r"(?:\.\s|;|\||--|—|–\s|\n)")

ARROW_PAIR = re.compile(
    r"(?<![\w.])(\d{1,3}\.\d{1,2})\s*(?:->|→|–>|to)\s*\*{0,2}(\d{1,3}\.\d{1,2})(?![\d.])"
)

#: Scope keys (R6) are DERIVED, not listed. A token is a key when it is shaped like a PROPER
#: NOUN -- initial capital, at least one further capital, at least one lowercase letter -- is >=4
#: chars, occurs in the PDFs, and occurs there at most SCOPE_KEY_MAX times. Each clause of that
#: shape rule was forced by a false positive measured on the live docs:
#:   * "at least one lowercase" drops SHOUTED EMPHASIS ("NOTE", "HEADLINE", "DATASET", "GT-BEV")
#:     and acronyms ("AAAI"), which localise nothing and produced 30+ false failures.
#:   * "initial capital" drops "no-TTA".
#:   * the frequency cap drops "SemanticKITTI" and "SCPNet", which appear on nearly every page.
#: Surviving keys on the live docs are the ones that matter: SemanticPOSS, LMSCNet, JS3C-Net,
#: TALoS, CarlaSC, KITTI-360.
SCOPE_KEY_SHAPE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-‑][A-Za-z0-9]+)*\b")
SCOPE_KEY_MAX = 40

#: Tokens that pass the shape and frequency rules but localise NOTHING, because they name the
#: implementation rather than a dataset, a benchmark or a method. This is a stoplist and
#: therefore rots, so the rule for adding to it is strict: a token goes here ONLY with the
#: finding that forced it, never pre-emptively.
#:   * "PyTorch" -- CITATION.cff's abstract opens "Official PyTorch implementation ... 38.8%
#:     mIoU on the SemanticKITTI hidden test". 38.8 is the paper headline and is in
#:     CONTROL_VALUES, but R6 took "PyTorch" as the block's scope key and failed the value for
#:     not sitting within SCOPE_WINDOW of it in the PDFs. The framework a repo is written in
#:     says nothing about which measurement a number belongs to.
#: A MECHANICAL criterion was tried first and REJECTED, so nobody re-derives it: "a scope key
#: must occur in the PDFs within 120 chars of a metric-shaped number" keeps PyTorch, PyPI and
#: NumPy (all 2/2, 2/2, 1/1) while dropping LiDiff and SegRefiner (0/1, 0/2), which are real
#: method names. It separates nothing.
SCOPE_KEY_STOP = frozenset({"PyTorch"})
SCOPE_WINDOW = 1200          # characters either side of a key occurrence
DELTA_TOL = Decimal("0.05")  # pp; R5 compares one-decimal doc arithmetic
MIN_CLAUSE = 20              # chars; below this a "clause" is a wrap fragment
DOC_LOCALITY = 200           # chars; how close a scope key must sit to its value


# --------------------------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------------------------
class Finding(NamedTuple):
    site: str
    detail: str


class Result:
    """One named check. `fail` is what makes the gate able to exit 1; `note` never does."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.findings: List[Finding] = []
        self.notes: List[str] = []

    def fail(self, site: str, detail: str) -> None:
        self.findings.append(Finding(site, detail))

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    @property
    def ok(self) -> bool:
        return not self.findings


def norm(text: str) -> str:
    """Unicode minus/dashes -> ASCII, NBSP -> space. PDFs render U+2212 for a minus sign."""
    return (text.replace("−", "-").replace("–", "-").replace("—", "-")
                .replace(" ", " ").replace(" ", " ").replace(" ", " "))


class PaperMissing(RuntimeError):
    """The manuscript PDFs are not on this machine.  Raised instead of letting PyMuPDF
    throw, so a visitor without the (unpublished) paper checkout gets one named FAIL line
    rather than a traceback quoting a path that means nothing to them."""


def read_pdfs() -> Dict[str, str]:
    import fitz  # PyMuPDF

    absent = [str(p) for p in PAPER_PDFS if not p.is_file()]
    if absent:
        raise PaperMissing(
            f"{', '.join(absent)} not found. The manuscript checkout is NOT part of the "
            f"public release; point GSSC_PAPER at your own copy to run this gate. Absent "
            f"is not evidence the numbers agree, so this gate fails rather than skipping.")
    out: Dict[str, str] = {}
    for p in PAPER_PDFS:
        doc = fitz.open(p)
        try:
            out[p.name] = norm("\n".join(doc[i].get_text() for i in range(doc.page_count)))
        finally:
            doc.close()
    return out


def coarser(value: str) -> List[str]:
    """The value itself plus its ROUND_HALF_UP renderings at fewer decimals, down to 1 dp.

    Never down to 0 dp: a bare integer like "26" occurs all over a per-class table and would
    make 26.1 look printed. That is exactly the false pass this gate exists to prevent.
    """
    out = [value]
    dp = len(value.split(".")[1])
    for k in range(dp - 1, 0, -1):
        out.append(str(Decimal(value).quantize(Decimal("1." + "0" * k), rounding=ROUND_HALF_UP)))
    return out


def in_pdfs(pdfs: Dict[str, str], value: str, signed: str = "") -> str:
    """Return the matched rendering ("" if none). `signed` is "+"/"-" for a delta."""
    for cand in coarser(value):
        pat = re.compile((r"(?<![\d.])" + re.escape(signed) + r"\s?" if signed
                          else r"(?<![\d.])") + re.escape(cand) + r"(?![\d.])")
        if any(pat.search(t) for t in pdfs.values()):
            return signed + cand
    return ""


class Value(NamedTuple):
    path: Path
    line: int
    sign: str
    value: str
    clause: str
    block: str
    line_text: str

    @property
    def site(self) -> str:
        return f"{self.path}:{self.line}"

    @property
    def shown(self) -> str:
        return f"{self.sign}{self.value}"


def blocks(lines: Sequence[str]) -> List[Tuple[int, List[str]]]:
    """[(1-based first line, block lines)] -- a block is a run of contiguous non-blank lines."""
    out, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].strip():
            j += 1
        out.append((i + 1, list(lines[i:j])))
        i = j
    return out


def clause_window(block_text: str, pos: int) -> str:
    """The clause containing `pos`, plus the following clause.

    The +1 clause of slack is measured, not arbitrary: REPRODUCIBILITY.md:324 states an expected
    value and disclaims it in the NEXT sentence. Without the slack that honest doc fails; with
    more slack the MANIFEST's false 26.1 claim would be laundered by the word "internal" that
    describes a different number further down the paragraph. R7 backstops the leak.
    """
    bounds = [0] + [m.end() for m in CLAUSE_SPLIT.finditer(block_text)] + [len(block_text)]
    for k in range(len(bounds) - 1):
        if bounds[k] <= pos < bounds[k + 1]:
            j = k + 1
            end = bounds[min(j + 1, len(bounds) - 1)]      # own clause + the next one
            # Advance over FRAGMENTS. Measured: docs/REPRODUCIBILITY.md:324 ends its clause at
            # "semantic-kitti-api." and the next clause is the 11-character wrap "This is the";
            # a fixed +1 clause landed on that fragment and the honest disclaimer one clause
            # further ("GT-BEV DIAGNOSTIC, not the paper's headline") was invisible. Advancing
            # only over fragments, never over a real clause, keeps the MANIFEST's false 26.1
            # claim unexempted -- its neighbouring clauses are all full-length.
            while j + 1 < len(bounds) and (bounds[j + 1] - bounds[j]) < MIN_CLAUSE:
                j += 1
                end = bounds[min(j + 1, len(bounds) - 1)]
            return block_text[bounds[k]:end]
    return block_text


def harvest(paths: Sequence[Path]) -> Tuple[List[Value], Dict[str, int]]:
    """Every paper-anchored, metric-shaped value, with its clause and block. Plus filter counts."""
    vals: List[Value] = []
    stats = {"tokens": 0, "no_anchor": 0, "unit": 0, "no_metric": 0, "version": 0}
    for p in paths:
        if not p.is_file():
            continue
        lines = norm(p.read_text(errors="replace")).splitlines()
        for start, blk in blocks(lines):
            btxt = "\n".join(blk)
            anchored = bool(PAPER_ANCHOR.search(btxt)) and bool(METRIC_MARK.search(btxt))
            for off, line in enumerate(blk):
                for m in NUMBER.finditer(line):
                    stats["tokens"] += 1
                    if not anchored:
                        stats["no_anchor"] += 1
                        continue
                    if NON_METRIC_UNIT.match(line[m.end():m.end() + 8]):
                        stats["unit"] += 1
                        continue
                    if VERSION_CONTEXT.search(line[max(0, m.start() - 12):m.start()]):
                        stats["version"] += 1
                        continue
                    if not METRIC_MARK.search(line[max(0, m.start() - 30):m.end() + 40]):
                        stats["no_metric"] += 1
                        continue
                    pos = sum(len(x) + 1 for x in blk[:off]) + m.start()
                    vals.append(Value(p, start + off, m.group(1).replace("−", "-"),
                                      m.group(2), clause_window(btxt, pos), btxt, line.strip()))
    return vals, stats


def scope_keys(block: str, pdfs: Dict[str, str], freq: Dict[str, int]) -> List[str]:
    """Distinctive proper nouns shared by the block and the PDFs (see SCOPE_KEY_MAX)."""
    keys = []
    for tok in set(SCOPE_KEY_SHAPE.findall(block)):
        if tok in SCOPE_KEY_STOP:
            continue
        if len(tok) < 4 or sum(c.isupper() for c in tok) < 2:
            continue
        if not any(c.islower() for c in tok):
            continue                      # SHOUTED emphasis / acronym, not a scope
        n = freq.get(tok)
        if n is None:
            n = sum(len(re.findall(r"\b" + re.escape(tok) + r"\b", t)) for t in pdfs.values())
            freq[tok] = n
        if 0 < n <= SCOPE_KEY_MAX:
            keys.append(tok)
    return sorted(keys)


def near_in_doc(v: "Value", key: str) -> bool:
    """Is `key` within DOC_LOCALITY chars of the value's own occurrence in its block?"""
    pat = re.compile(r"(?<![\d.])" + re.escape(v.value) + r"(?![\d.])")
    for m in pat.finditer(v.block):
        w = v.block[max(0, m.start() - DOC_LOCALITY):m.end() + DOC_LOCALITY]
        if re.search(r"\b" + re.escape(key) + r"\b", w):
            return True
    return False


def near_key(pdfs: Dict[str, str], key: str, value: str) -> bool:
    pat = re.compile(r"(?<![\d.])" + re.escape(value) + r"(?![\d.])")
    for t in pdfs.values():
        for m in re.finditer(r"\b" + re.escape(key) + r"\b", t):
            if pat.search(t[max(0, m.start() - SCOPE_WINDOW):m.end() + SCOPE_WINDOW]):
                return True
    return False


# --------------------------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------------------------
def run(paths: Sequence[Path], pdfs: Dict[str, str]) -> List[Result]:
    r1 = Result("sources-readable")
    r2 = Result("pdf-instrument-live")
    r3 = Result("paper-values-in-pdf")
    r4 = Result("signed-deltas-in-pdf")
    r5 = Result("delta-arithmetic")
    r6 = Result("scope-localised")
    r7 = Result("headline-claims-backed")

    # A missing source is not automatically a defect: the assets manifests are GENERATED, and
    # ``checkpoints/MANIFEST.txt`` was deleted by a manifest rebuild while this gate was being
    # written. What must never happen is a WHOLE FAMILY going unjudged in silence -- that is how
    # a gate keeps passing after the thing it watches has moved.
    for p in paths:
        if p.is_file() and not p.read_text(errors="replace").strip():
            r1.fail(str(p), "source is empty")
    for label, group in (("repo entry point", [REPO / "README.md"]),
                         ("repo metadata", [REPO / "CITATION.cff", REPO / "CONTRIBUTING.md",
                                            REPO / "SECURITY.md"]),
                         ("repo docs", sorted((REPO / "docs").glob("*.md"))),
                         ("github surface", sorted((REPO / ".github").rglob("*.md"))
                                           + sorted((REPO / ".github").rglob("*.yml"))),
                         ("assets docs", [ASSETS / "MANIFEST.txt", ASSETS / "README.md",
                                          ASSETS / "checkpoints" / "MANIFEST.txt"]),
                         ("assets dataset docs", sorted(ASSETS.glob("datasets/*/README.md"))),
                         ("hugging face cards", [ASSETS / "hf_cards" / "dataset_card.md",
                                                 ASSETS / "hf_cards" / "model_card.md"])):
        alive = [p for p in group if p.is_file()]
        if not alive:
            r1.fail(str(group[0].parent), f"no '{label}' source exists any more; this gate is "
                                          f"judging none of that family")
        elif len(alive) < len(group):
            r1.note(f"{label}: {len(alive)}/{len(group)} present "
                    f"({', '.join(str(p) for p in group if not p.is_file())} absent)")
    r1.note(f"{sum(1 for p in paths if p.is_file())} doc source(s) read")
    # A named exclusion is printed on EVERY run. An unjudged surface that nobody can see in the
    # output is indistinguishable from a judged one, which is the failure this table exists for.
    for name, why in DOC_SOURCE_EXCLUSIONS:
        r1.note(f"NOT JUDGED: {name} -- {why}")

    total = sum(len(re.findall(r"\d", t)) for t in pdfs.values())
    if total < 5000:
        r2.fail(",".join(pdfs), f"only {total} digit chars extracted -- PDF text layer suspect")
    for cv in CONTROL_VALUES:
        if not in_pdfs(pdfs, cv):
            r2.fail("CONTROL_VALUES", f"{cv} is known to be in the paper but the extractor "
                                      f"cannot find it; every 'matched' verdict is now suspect")
    r2.note(f"{len(pdfs)} PDF(s), {total} digit chars, {len(CONTROL_VALUES)} control value(s) found")

    vals, stats = harvest(paths)
    matched: Dict[Tuple[str, str], str] = {}
    exempt: List[Value] = []
    freq: Dict[str, int] = {}

    thirdparty: List[Value] = []
    for v in vals:
        disclaimed = bool(DISCLAIMER.search(v.clause)) or bool(THIRD_PARTY.search(v.clause))
        headline = bool(HEADLINE_CLAIM.search(v.clause))
        hit = in_pdfs(pdfs, v.value)
        matched[(str(v.path), v.value)] = hit

        if v.sign:
            continue                     # R4 owns signed values
        if disclaimed and not headline:
            (thirdparty if THIRD_PARTY.search(v.clause) else exempt).append(v)
            continue
        if not hit:
            r3.fail(v.site, f"{v.shown} is presented as a paper value but neither it nor any "
                            f"coarser rounding of it occurs in either PDF | {v.line_text[:130]}")
        if headline and not hit:
            r7.fail(v.site, f"{v.shown} is asserted to be a PAPER HEADLINE value and the paper "
                            f"does not print it | {v.line_text[:130]}")
        elif headline and disclaimed:
            r7.note(f"{v.site}: {v.shown} carries both a headline assertion and a hedge; "
                    f"judged as a headline claim (the hedge does not exempt it)")

    for v in (x for x in vals if x.sign):
        signed_hit = in_pdfs(pdfs, v.value, "+" if v.sign == "+" else "-")
        if signed_hit:
            continue
        if (DISCLAIMER.search(v.clause) or THIRD_PARTY.search(v.clause)) \
                and not HEADLINE_CLAIM.search(v.clause):
            (thirdparty if THIRD_PARTY.search(v.clause) else exempt).append(v)
            continue
        # A delta may legitimately be the doc's own arithmetic over two values the paper DOES
        # print ("31.8 -> 54.9 (+23.1)"), so an unprinted delta is not automatically wrong. The
        # derivation is only worth something if BOTH endpoints are themselves printed: deriving a
        # delta from a rejected endpoint would launder the endpoint's defect into a pass.
        derived = ""
        pool = [x[1] for x in NUMBER.findall(v.clause) if not x[0]]
        for a in pool:
            for b in pool:
                if a == b:
                    continue
                if abs(Decimal(b) - Decimal(a) - Decimal(v.value)) > DELTA_TOL:
                    continue
                if in_pdfs(pdfs, a) and in_pdfs(pdfs, b):
                    derived = f"{a} -> {b}"
                    break
            if derived:
                break
        if derived:
            r4.note(f"{v.site}: {v.shown} is not printed but reconciles with the printed pair "
                    f"{derived}")
            continue
        r4.fail(v.site, f"{v.shown} is presented as a paper delta; the paper prints no such "
                        f"signed value and no printed pair in its clause derives it | "
                        f"{v.line_text[:130]}")

    # R5 -- the doc against ITSELF; needs no paper.
    seen_pairs = set()
    for v in vals:
        for a, b in ARROW_PAIR.findall(v.clause):
            key = (v.site, a, b)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            # x[0] is "" for an unsigned literal, and "" in "+-" is TRUE in Python -- that
            # bug made every value in the clause count as a "stated delta" and printed nonsense.
            deltas = [x for x in NUMBER.findall(v.clause) if x[0] and x[0] in "+-−"]
            for sign, d in deltas:
                want = Decimal(b) - Decimal(a)
                got = Decimal(d) if sign == "+" else -Decimal(d)
                if abs(want - got) <= DELTA_TOL:
                    break
            else:
                if deltas:
                    r5.fail(v.site, f"the clause states {a} -> {b} (= {Decimal(b) - Decimal(a):+}) "
                                    f"but its stated delta(s) are "
                                    f"{', '.join(s + d for s, d in deltas)} | {v.line_text[:110]}")
    r5.note(f"{len(seen_pairs)} 'a -> b' pair(s) checked against their own stated delta")

    # R6 -- scope localisation. Only values that ALREADY matched can be wrong-row; a value that
    # failed R3 is reported once, not twice.
    abstained = 0
    skipped = {(x.site, x.shown) for x in exempt} | {(x.site, x.shown) for x in thirdparty}
    for v in vals:
        if (v.site, v.shown) in skipped:
            continue
        # R6 judges the rendering R3 ACCEPTED (16.59 is looked up as the printed 16.6), otherwise
        # the rule fires on every doc that carries more precision than the paper prints.
        rendered = matched.get((str(v.path), v.value)) or in_pdfs(pdfs, v.value, v.sign)
        if not rendered:
            continue
        # DOC LOCALITY. Without it R6 fires on a value that shares a big markdown table with an
        # unrelated row's proper noun (measured: the repro matrix's per-frame VRU cell judged
        # against "LMSCNet" from another row) and on long figure captions where an architecture
        # term sits 200+ chars away. The key must be near the value HERE before the gate demands
        # they be near each other in the paper.
        keys = [k for k in scope_keys(v.block, pdfs, freq) if near_in_doc(v, k)]
        if not keys:
            abstained += 1
            continue
        # EVERY doc-local key must be satisfied, not merely one. Measured on README.md:190: the
        # SemanticPOSS mIoU is quoted as 6.6, which the paper never prints near "SemanticPOSS" --
        # but "TALoS" also sits in that sentence, and 6.6 DOES occur within 1200 chars of a TALoS
        # mention elsewhere (the main results table). Under an any-key rule the defect this rule
        # was written for passed. Both keys describe the same claim here, so both must hold.
        unmet = [k for k in keys if not near_key(pdfs, k, rendered.lstrip("+-"))]
        if unmet:
            r6.fail(v.site, f"{v.shown} (printed as {rendered}) does occur in the paper, but "
                            f"never within {SCOPE_WINDOW} "
                            f"chars of scope key(s) {unmet} -- it is printed "
                            f"somewhere else, about something else | {v.line_text[:110]}")
    r6.note(f"{abstained} value(s) ABSTAINED: no proper noun that is distinctive in the paper "
            f"(<= {SCOPE_KEY_MAX} occurrences) sits within {DOC_LOCALITY} chars of them, so the "
            f"rule places no constraint on them")

    r3.note(f"{len(vals)} value(s) judged; {len(exempt)} exempted by an adjacent disclaimer, "
            f"{len(thirdparty)} attributed to another work; "
            f"filters discarded {stats['no_anchor']} unanchored, {stats['no_metric']} "
            f"non-metric, {stats['unit']} unit-suffixed, {stats['version']} version-context "
            f"token(s) of {stats['tokens']} seen")
    for v in exempt[:60]:
        r3.note(f"  exempt {v.site}: {v.shown} -- clause says it is not a paper value")
    for v in thirdparty[:20]:
        r3.note(f"  third-party {v.site}: {v.shown} -- clause attributes it to another work")
    return [r1, r2, r3, r4, r5, r6, r7]


def report(results: Sequence[Result]) -> int:
    nfail = 0
    for r in results:
        for n in r.notes:
            print(f"        note  {n}")
        if r.ok:
            print(f"  PASS  {r.name}")
        else:
            nfail += 1
            for f in r.findings:
                print(f"  FAIL  {r.name}   ({f.site}: {f.detail})")
    print(f"OK: 0 failing check(s)" if not nfail else f"FAILED: {nfail} failing check(s)")
    return 0 if not nfail else 1


# --------------------------------------------------------------------------------------------
# selftest -- every check must be provably able to fail
# --------------------------------------------------------------------------------------------
def _tmpdir() -> Path:
    import os
    import tempfile
    base = os.environ.get("TMPDIR") or tempfile.gettempdir()
    d = Path(tempfile.mkdtemp(prefix="paper_numbers_", dir=base))
    return d


def selftest() -> int:
    pdfs = read_pdfs()
    tmp = _tmpdir()
    missed: List[str] = []

    def trip(name: str, paths: Sequence[Path], pdf_override: Dict[str, str] | None = None) -> None:
        res = {r.name: r for r in run(paths, pdf_override if pdf_override is not None else pdfs)}
        if name in res and not res[name].ok:
            print(f"  TRIPPED  {name}")
        else:
            print(f"  MISSED   {name}")
            missed.append(name)

    def write(rel: str, text: str) -> Path:
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    # R1 -- an EMPTY source file. (A merely absent optional file is deliberately not a failure;
    # see the rule. The empty-file arm is what remains provable per-file.)
    trip("sources-readable", [write("empty.md", "\n")])

    # R2 -- blind the instrument. Replacing ONE control literal is not enough and the first
    # version of this fixture was vacuous because of it: in_pdfs accepts a coarser rounding, so
    # blanking "38.54" still matched the printed "38.5". Strip every digit instead, and assert
    # the text really changed.
    blinded = {k: re.sub(r"\d", "#", v) for k, v in pdfs.items()}
    assert any(blinded[k] != pdfs[k] for k in pdfs), "R2 fault did not perturb the PDF text"
    assert not in_pdfs(blinded, CONTROL_VALUES[0]), "R2 fault left the control value findable"
    trip("pdf-instrument-live", [], blinded)

    # R3 -- a fabricated paper value. 99.87 is asserted absent from the real PDFs first.
    assert not in_pdfs(pdfs, "99.87"), "R3 fixture value is present in the paper; pick another"
    f3 = write("r3.md", "Headline\n\nThe paper tab:main_results row reads **99.87 %** mIoU.\n")
    trip("paper-values-in-pdf", [f3])

    # ... and the same file with a real value must NOT trip it (the rule is not "everything fails").
    ok3 = write("r3ok.md", "Headline\n\nThe paper tab:main_results row reads **38.54 %** mIoU.\n")
    res = {r.name: r for r in run([ok3], pdfs)}
    if res["paper-values-in-pdf"].ok:
        print("  TRIPPED  paper-values-in-pdf/negative-control")
    else:
        print("  MISSED   paper-values-in-pdf/negative-control")
        missed.append("paper-values-in-pdf/negative-control")

    # R4 -- a delta the paper never prints, with no derivation available.
    # 7.77 was tried first and is NOT usable: it rounds to 7.8, which the paper prints as +7.8.
    assert not in_pdfs(pdfs, "97.61", "+"), "R4 fixture delta is present in the paper"
    f4 = write("r4.md", "Deltas\n\nPer paper tab:perclass_delta, motorcyclist gains **+97.61 pp** mIoU.\n")
    trip("signed-deltas-in-pdf", [f4])

    # R5 -- doc-internal arithmetic. Needs no paper at all.
    f5 = write("r5.md", "POSS\n\nOn SemanticPOSS the paper reports 1.0 -> 6.6 mIoU (+5.5 pp).\n")
    trip("delta-arithmetic", [f5])

    # R6 -- a value that IS printed in the paper, but nowhere near the block's scope key.
    # 36.6 is a real per-class cell; SemanticPOSS is a real, distinctive key; the two never meet.
    assert in_pdfs(pdfs, "36.6"), "R6 fixture value is absent from the paper; pick another"
    f6 = write("r6.md", "POSS\n\nOn SemanticPOSS val seq 02 the paper tab:portable_s2d2 row "
                        "reports **36.6 %** mIoU.\n")
    trip("scope-localised", [f6])

    # R7 -- a headline assertion the hedge must not launder. Same clause carries "internal",
    # which WOULD exempt it under R3; R7 must still fire.
    f7 = write("r7.md", "Cross-base\n\nThe 99.87% value is the paper headline for this base, "
                        "an internal continuity number.\n")
    res = {r.name: r for r in run([f7], pdfs)}
    if not res["headline-claims-backed"].ok:
        print("  TRIPPED  headline-claims-backed")
    else:
        print("  MISSED   headline-claims-backed")
        missed.append("headline-claims-backed")

    # WIDENED-SOURCE ARM. doc_sources() gained CITATION.cff, CONTRIBUTING.md, SECURITY.md,
    # .github/**, the per-dataset asset READMEs and the HF cards on 2026-08-22. Listing a path
    # is not judging it: a widening that returned the paths and then dropped them in `harvest`
    # would look identical in the output. So this asserts the FAMILIES are present AND that a
    # fabricated paper value planted in each new family's file shape is actually caught.
    for fam, rel in (("repo metadata", "CITATION.cff"),
                     ("github surface", ".github/ISSUE_TEMPLATE/x.md"),
                     ("assets dataset docs", "datasets/scpnet_predictions/README.md"),
                     ("hugging face cards", "hf_cards/model_card.md")):
        planted = write(rel, "Card\n\nThe paper tab:main_results row reads **99.87 %** mIoU.\n")
        res = {r.name: r for r in run([planted], pdfs)}
        if res["paper-values-in-pdf"].ok:
            print(f"  MISSED   widened-source:{fam}")
            missed.append(f"widened-source:{fam}")
        else:
            print(f"  TRIPPED  widened-source:{fam}")
    live = {str(q) for q in doc_sources()}
    for must in ("CITATION.cff", "CONTRIBUTING.md", "SECURITY.md"):
        if not any(q.endswith("/" + must) for q in live):
            print(f"  MISSED   widened-source:doc_sources-contains-{must}")
            missed.append(f"widened-source:doc_sources-contains-{must}")
    if not any("/.github/" in q for q in live):
        print("  MISSED   widened-source:doc_sources-contains-github")
        missed.append("widened-source:doc_sources-contains-github")
    if not any("/hf_cards/" in q for q in live):
        print("  MISSED   widened-source:doc_sources-contains-hf_cards")
        missed.append("widened-source:doc_sources-contains-hf_cards")
    # ...and an exclusion must stay EXCLUDED-AND-NAMED, never quietly dropped from the table.
    if any(q.endswith("/CHANGELOG.md") for q in live) or not DOC_SOURCE_EXCLUSIONS:
        print("  MISSED   widened-source:exclusions-are-named")
        missed.append("widened-source:exclusions-are-named")
    else:
        print("  TRIPPED  widened-source:exclusions-are-named")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)          # scratch only, never inside a repo
    n = 18
    print(f"SELFTEST OK: {n - len(missed)}/{n} checks provably fail when broken"
          if not missed else
          f"SELFTEST FAILED: {len(missed)} check(s) did not fail when broken: {missed}")
    return 1 if missed else 0


def main() -> int:
    try:
        pdfs = read_pdfs()
    except PaperMissing as e:
        print(f"  FAIL  paper_pdfs_present   ({e})")
        print("FAILED: 1 failing check(s)")
        return 1
    return report(run(doc_sources(), pdfs))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        try:
            raise SystemExit(selftest())
        except PaperMissing as e:
            print(f"  FAIL  paper_pdfs_present   ({e})")
            print("SELFTEST FAILED: the fixture needs the manuscript PDFs")
            raise SystemExit(1)
    raise SystemExit(main())

#!/usr/bin/env python3
r"""GATE: every paper cross-reference shipped in the release must resolve against the paper.

WHY THIS EXISTS -- measured on 2026-08-20, before any release fix landed
--------------------------------------------------------------------------------------------
LINE NUMBERS IN THIS BLOCK ARE A DATED SNAPSHOT of the checkout named above, not
navigation. Several have already moved: follow the SYMBOL, the heading or the quoted
text, and re-derive the location with `grep -n`. Every check below RE-MEASURES the
live artefacts, so nothing here is load-bearing for a verdict.

54 reference sites in the release name 8 distinct labels that DO NOT EXIST in
``<paper checkout>/*.tex``:

    tab:perclass                  11 sites (README.md:64,95,267,351, ...)  -- the paper's label
                                  is tab:perclass_delta
    tab:cross_base_js3c           10 sites (README.md:271, docs/INFERENCE.md:76,80, ...)
    tab:cross_base_lmsc           10 sites (README.md:41,272, docs/INFERENCE.md:103,107, ...)
    tab:train_timesteps_ablation  10 sites (docs/MODEL_ZOO.md:167, assets MANIFEST, ...)
    fig:da_pipeline                6 sites (assets MANIFEST.txt, checkpoints/MANIFEST.txt)
    tab:train_timesteps_curriculum 4 sites (README.md:270, scripts/reproduce_table.py:35)
    tab:supp_retrain_deltas        2 sites (README.md:351, docs/REPRODUCIBILITY.md:185)
    tab:step_reduction_sf          1 site  (configs/train/31k_sf.yaml:1)

Four ``_paper_table:`` declarations additionally cite ROW NUMBERS that cannot exist:
``configs/train/js3c_real.yaml`` says "tab:main_results rows 90-91" and three eval configs say
"row 91", while the tabular behind ``\label{tab:main_results}`` carries 19 ``\\`` row breaks in
total. A reader following that pointer finds nothing and cannot tell whether the config or the
paper moved.

And the rendered numbers drift, which is the reason the manifests tell readers to use labels:
``configs/eval/bev_secondary.yaml:2`` says "paper Tab. XV (secondary task)", but
``\label{tab:bev_results}`` renders as **supplementary Tab. XXI** (main.aux/supplementary.aux),
and the supplement's Tab. XV is a different table.

WHAT IT CHECKS
--------------------------------------------------------------------------------------------
 R1 paper-index-live       the label index is built from main.tex and supplementary.tex by
                           FOLLOWING their \input/\include, not by globbing a directory -- a glob
                           silently picks up _archive/ and thesis/ copies and would resolve labels
                           that the shipped paper does not define. The index must be non-trivial
                           (>=100 labels, >=25 of them tab:) and the rendered-number index read
                           from the .aux files must agree with the built PDFs: every table number
                           the .aux assigns must occur as "TABLE <N>" in that document's PDF.
                           Without that last clause a stale .aux would silently redefine R4.
 R2 labels-resolve         every tab:/fig:/sec:/eq:/alg:/app: label referenced anywhere in the
                           release must be defined in the paper. Reported per label with every
                           file:line that cites it.
 R3 table-rows-exist       a reference of the form "<label> row N" / "rows N-M" must name a row
                           the table actually has; the row count is MEASURED from the tabular
                           body, never assumed.
 R4 rendered-number-agrees where a line pairs a label with a rendered "Tab. N" / "Fig. N", the
                           .aux number for that label must match, including which document it
                           lives in. Lines that give a rendered number with NO label are not
                           judged (they are usually third-party: "TALoS Tab. 4"); the count of
                           those abstentions is printed.
 R5 reference-census       the harvest must actually find references, in several files. A path
                           that stops matching is the classic way this class of gate goes green
                           by seeing nothing.

WHAT IT DOES NOT DO
--------------------------------------------------------------------------------------------
 * It does not check that a resolving label is the RIGHT label for the claim beside it (a doc may
   cite tab:main_results for a number that lives in tab:portable_s2d2). check_paper_numbers.py
   attacks that from the value side.
 * The paper is READ-ONLY. This gate only reads .tex, .aux and .pdf.

USAGE
    python .release_checks/check_paper_labels.py
    python .release_checks/check_paper_labels.py --selftest

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
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence, Set, Tuple

REPO = Path(os.environ.get("GSSC_REPO") or Path(__file__).resolve().parents[1])
ASSETS = Path(os.environ.get("GSSC_ASSETS") or REPO.parent / "GSSC-S2D2-assets")
PAPER = Path(os.environ.get("GSSC_PAPER") or REPO.parent / "GSSC-paper")
PAPER_SEEDS = (PAPER / "main.tex", PAPER / "supplementary.tex")
AUX = {"main": PAPER / "main.aux", "supplementary": PAPER / "supplementary.aux"}
PDF = {"main": PAPER / "pdf" / "main.pdf", "supplementary": PAPER / "pdf" / "supplementary.pdf"}

LABEL_REF = re.compile(r"(?<![\w:])((?:tab|fig|sec|eq|alg|app|thm|lem|prop|cor):[A-Za-z0-9_\-]+)")
LABEL_DEF = re.compile(r"\\label\{([^}]+)\}")
INPUT_CMD = re.compile(r"\\(?:input|include)\{([^}]+)\}")
ROW_REF = re.compile(r"rows?\s+(\d+)(?:\s*[-–]\s*(\d+))?", re.I)
RENDERED = re.compile(r"\b(Tab|Table|Fig|Figure)\.?\s*([IVXLC]+|\d{1,2})\b")
ROMAN = re.compile(r"^[IVXLC]+$")
LOOKBACK = 2      # lines; how far a rendered number may sit from its label


def ref_sources() -> List[Path]:
    """Everything in the release that can carry a paper pointer. Globs expand live."""
    out: List[Path] = [REPO / "README.md", REPO / "CHANGELOG.md", REPO / "CONTRIBUTING.md"]
    out += sorted((REPO / "docs").glob("*.md"))
    out += sorted(REPO.glob("configs/*/*.yaml"))
    out += sorted((REPO / "scripts").glob("*.py"))
    # src/ carries paper pointers too and was unwatched. tests/ is deliberately NOT added:
    # test_config_loader.py writes a synthetic `_paper_table: tab:foo` into a temp YAML, so
    # the fixture would read as an unresolvable pointer. (Two test docstrings did carry a real
    # defect -- "Tab. III rows 90-91", which were LaTeX LINE numbers, not row indices -- and
    # they were corrected by hand; the gate still cannot see that directory.)
    out += sorted((REPO / "src" / "gssc").rglob("*.py"))
    out += sorted((REPO / "examples").glob("*.ipynb"))
    out += [ASSETS / "MANIFEST.txt", ASSETS / "README.md", ASSETS / "checkpoints" / "MANIFEST.txt"]
    return [p for p in out if p.is_file()]


class Ref(NamedTuple):
    path: Path
    line: int
    label: str
    text: str

    @property
    def site(self) -> str:
        return f"{self.path}:{self.line}"


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


# --------------------------------------------------------------------------------------------
# the paper side
# --------------------------------------------------------------------------------------------
def tex_sources(seeds: Sequence[Path] = PAPER_SEEDS) -> List[Path]:
    """Seeds plus everything they \\input, transitively.

    NOT a glob over PAPER/**/*.tex: the paper repo carries _archive/ and thesis/ trees whose
    labels are NOT in the shipped documents. Resolving a release pointer against an archived
    copy would report a dead reference as healthy -- the exact way this gate could lie.
    """
    seen: Set[Path] = set()
    queue = [p for p in seeds if p.is_file()]
    while queue:
        p = queue.pop()
        if p in seen:
            continue
        seen.add(p)
        for rel in INPUT_CMD.findall(p.read_text(errors="replace")):
            cand = (p.parent / rel)
            cand = cand if cand.suffix else cand.with_suffix(".tex")
            if cand.is_file():
                queue.append(cand.resolve())
    return sorted(seen)


def paper_labels(files: Sequence[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in files:
        for lab in LABEL_DEF.findall(p.read_text(errors="replace")):
            out.setdefault(lab, p)
    return out


def rendered_index() -> Dict[str, Tuple[str, str]]:
    """label -> (document, rendered number), from the .aux files LaTeX wrote."""
    out: Dict[str, Tuple[str, str]] = {}
    for doc, aux in AUX.items():
        if not aux.is_file():
            continue
        for m in re.finditer(r"\\newlabel\{([^}@]+)\}\{\{([^{}]*)\}\{(\d+)\}", aux.read_text(errors="replace")):
            out.setdefault(m.group(1), (doc, m.group(2)))
    return out


def table_rows(label: str, files: Sequence[Path]) -> int:
    r"""Row breaks in the tabular that carries \label{label}; -1 if not locatable.

    An over-count is deliberate: header and rule rows are counted too, so the rule fires only on
    a row index that is impossible under the most generous reading.
    """
    for p in files:
        t = p.read_text(errors="replace")
        i = t.find("\\label{" + label + "}")
        if i < 0:
            continue
        s = t.rfind("\\begin{tab", 0, i)
        e = t.find("\\end{table", i)
        if s < 0 or e < 0:
            return -1
        return t.count("\\\\", s, e)
    return -1


def harvest(paths: Sequence[Path]) -> List[Ref]:
    out: List[Ref] = []
    for p in paths:
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            for lab in LABEL_REF.findall(line):
                out.append(Ref(p, i, lab, line.strip()))
    return out


def pdf_text(doc: str) -> str:
    import fitz

    d = fitz.open(PDF[doc])
    try:
        return "\n".join(d[i].get_text() for i in range(d.page_count))
    finally:
        d.close()


# --------------------------------------------------------------------------------------------
def run(paths: Sequence[Path], seeds: Sequence[Path] = PAPER_SEEDS,
        rendered: Dict[str, Tuple[str, str]] | None = None,
        pdfs: Dict[str, str] | None = None) -> List[Result]:
    r1 = Result("paper-index-live")
    r2 = Result("labels-resolve")
    r3 = Result("table-rows-exist")
    r4 = Result("rendered-number-agrees")
    r5 = Result("reference-census")

    tex = tex_sources(seeds)
    labels = paper_labels(tex)
    rend = rendered_index() if rendered is None else rendered
    ntab = sum(1 for k in labels if k.startswith("tab:"))
    r1.note(f"{len(tex)} tex file(s) reached from {len(seeds)} seed(s); {len(labels)} label(s), "
            f"{ntab} of them tab:; {len(rend)} rendered number(s) from .aux")
    if len(labels) < 100 or ntab < 25:
        r1.fail(f"{seeds[0] if seeds else '?'}: label index is implausibly small "
                f"({len(labels)} labels, {ntab} tab:) -- the \\input walk is not reaching the "
                f"paper body, so every 'unresolved' verdict below is an artefact")
    # .aux freshness: a stale .aux would silently redefine R4's authority.
    if pdfs is None:
        pdfs = {d: pdf_text(d) for d in PDF if PDF[d].is_file()}
    for lab, (doc, num) in sorted(rend.items()):
        if not lab.startswith("tab:") or doc not in pdfs:
            continue
        if not re.search(r"TABLE\s+" + re.escape(num) + r"\b", pdfs[doc]):
            r1.fail(f"{AUX[doc]}: assigns {lab} -> {doc} Tab. {num}, but '{PDF[doc].name}' "
                    f"contains no 'TABLE {num}'; the .aux is stale against the shipped PDF")

    refs = harvest(paths)
    r5.note(f"{len(refs)} label reference(s) in {len({r.path for r in refs})} file(s) of "
            f"{len(paths)} scanned")
    if len(refs) < 40 or len({r.path for r in refs}) < 5:
        r5.fail(f"{REPO}: only {len(refs)} reference(s) in "
                f"{len({r.path for r in refs})} file(s) -- the source list has stopped matching "
                f"the release layout and this gate is judging almost nothing")

    unresolved: Dict[str, List[Ref]] = {}
    for r in refs:
        if r.label not in labels:
            unresolved.setdefault(r.label, []).append(r)
    for lab, rs in sorted(unresolved.items(), key=lambda kv: -len(kv[1])):
        sites = ", ".join(x.site for x in rs[:6]) + (" ..." if len(rs) > 6 else "")
        r2.fail(f"{rs[0].site}: '{lab}' is referenced {len(rs)}x but no \\label{{{lab}}} exists "
                f"in the paper | sites: {sites}")
    r2.note(f"{len(refs) - sum(len(v) for v in unresolved.values())} of {len(refs)} reference(s) "
            f"resolve")

    # R3 -- row indices, measured against the tabular.
    judged = 0
    for r in refs:
        for m in ROW_REF.finditer(r.text):
            # "tab:main_results rows 90-91, supp tab:supp_b6_val" carries two labels and one row
            # range. Attaching the range to BOTH reported a defect against the wrong table; the
            # row index belongs to the nearest label on its LEFT.
            owners = [(mm.end(), mm.group(1)) for mm in LABEL_REF.finditer(r.text)
                      if mm.end() <= m.start()]
            if not owners or owners[-1][1] != r.label:
                continue
            if r.label not in labels:
                continue                      # R2 owns a label that does not exist at all
            n = table_rows(r.label, tex)
            if n < 0:
                continue
            judged += 1
            for grp in (m.group(1), m.group(2)):
                if grp and int(grp) > n:
                    r3.fail(f"{r.site}: cites '{r.label} row {grp}', but that table's tabular "
                            f"carries only {n} row break(s) | {r.text[:110]}")
    r3.note(f"{judged} row reference(s) judged against a measured row count")

    # R4 -- rendered numbers, only where a label pins the meaning.
    abstain = foreign = 0
    lines = ref_lines(paths)
    by_file: Dict[Path, List[Tuple[int, str]]] = {}
    for path, i, text in lines:
        by_file.setdefault(path, []).append((i, text))
    for path, rows in by_file.items():
        for idx, (i, text) in enumerate(rows):
            for m in RENDERED.finditer(text):
                kind = "tab:" if m.group(1).lower().startswith("tab") else "fig:"
                # "TALoS Tab.4 protocol" sits on the same line as our own tab:portable_s2d2
                # label and is TALoS's table 4, not ours. A capitalised proper noun immediately
                # in front of the reference is an attribution to another work; judging it
                # invented a defect on configs/eval/semanticposs_seq02.yaml:19 on this gate's
                # first run.
                lead = re.search(r"([A-Z][A-Za-z0-9\-]{2,})\s*$", text[:m.start()])
                if lead and lead.group(1) not in ("Paper", "Supp", "Supplementary", "Main",
                                                  "See", "The", "In", "Our", "This"):
                    foreign += 1
                    continue
                own = [l for l in LABEL_REF.findall(text) if l.startswith(kind)]
                if len(own) == 1:
                    label, how = own[0], "same line"
                else:
                    # Look back LOOKBACK lines. configs put the label on the `_paper_table:`
                    # line and the rendered number in the `_description:` line beneath it
                    # (configs/eval/bev_secondary.yaml), so a same-line-only rule sees none of
                    # the drift these fields were written to prevent. Only an UNAMBIGUOUS
                    # window counts: exactly one label of the matching kind.
                    # ...and ONLY in a YAML config. A markdown table gives every row its own
                    # "| Supp. Tab. VI (step reduction) |" cell, and looking back across rows attributed
                    # one row's rendered number to another row's label -- two invented defects on
                    # docs/REPRODUCIBILITY.md:243-244. In YAML the two lines are one record.
                    if path.suffix != ".yaml":
                        abstain += 1
                        continue
                    back: List[str] = []
                    for j in range(max(0, idx - LOOKBACK), idx):
                        back += [l for l in LABEL_REF.findall(rows[j][1]) if l.startswith(kind)]
                    if len(set(back)) != 1 or own:
                        abstain += 1
                        continue
                    label, how = back[0], f"label {idx - max(0, idx - LOOKBACK)} line(s) above"
                if label not in rend:
                    abstain += 1
                    continue
                doc, num = rend[label]
                if m.group(2) != num:
                    r4.fail(f"{path}:{i}: writes '{m.group(0)}' for '{label}' ({how}), which "
                            f"LaTeX renders as {doc} "
                            f"{'Tab.' if kind == 'tab:' else 'Fig.'} {num} | {text.strip()[:110]}")
    nolabel = sum(1 for _, _, t in lines if RENDERED.search(t) and not LABEL_REF.search(t))
    r4.note(f"{abstain} pairing(s) ABSTAINED (no unambiguous label, or the label carries no .aux "
            f"entry); {foreign} attributed to another work by the name in front of them; "
            f"{nolabel} line(s) give a rendered number with NO label anywhere near and are NOT "
            f"judged -- most are third-party ('TALoS Tab. 4'), and guessing which document they "
            f"mean is how a gate invents a defect")
    return [r1, r2, r3, r4, r5]


def ref_lines(paths: Sequence[Path]) -> List[Tuple[Path, int, str]]:
    out = []
    for p in paths:
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            out.append((p, i, line))
    return out


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
    tmp = Path(tempfile.mkdtemp(prefix="paper_labels_", dir=base))
    missed: List[str] = []
    pdfs = {d: pdf_text(d) for d in PDF if PDF[d].is_file()}
    rend = rendered_index()

    def write(rel: str, text: str) -> Path:
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def trip(name: str, label: str | None = None, **kw) -> None:
        res = {r.name: r for r in run(**kw)}
        shown = label or name
        if name in res and not res[name].ok:
            print(f"  TRIPPED  {shown}")
        else:
            print(f"  MISSED   {shown}")
            missed.append(shown)

    # R1 -- point the index at a seed that defines almost nothing. The fault is asserted to have
    # changed the index, so it cannot pass vacuously.
    stub = write("stub.tex", "\\label{tab:only_one}\n")
    assert len(paper_labels([stub])) < 100
    trip("paper-index-live", paths=[], seeds=[stub], rendered={}, pdfs=pdfs)

    # R1b -- a stale .aux: claim a table number the PDF does not carry.
    bogus = dict(rend)
    bogus["tab:invented"] = ("main", "XCIX")
    assert bogus != rend, "R1b fault did not perturb the rendered index"
    trip("paper-index-live", paths=[], seeds=PAPER_SEEDS, rendered=bogus, pdfs=pdfs,
         label="paper-index-live (stale .aux arm)")

    # R2 -- an unresolvable label.
    real = paper_labels(tex_sources())
    assert "tab:does_not_exist" not in real, "R2 fixture label exists; pick another"
    f2 = write("r2.md", "See paper tab:does_not_exist for the headline row.\n")
    trip("labels-resolve", paths=[f2], rendered=rend, pdfs=pdfs)

    # ... and a label that DOES exist must not trip it.
    assert "tab:main_results" in real
    f2ok = write("r2ok.md", "See paper tab:main_results for the headline row.\n")
    res = {r.name: r for r in run([f2ok], rendered=rend, pdfs=pdfs)}
    if res["labels-resolve"].ok:
        print("  TRIPPED  labels-resolve/negative-control")
    else:
        print("  MISSED   labels-resolve/negative-control")
        missed.append("labels-resolve/negative-control")

    # R3 -- an impossible row index against a REAL table.
    n = table_rows("tab:main_results", tex_sources())
    assert n > 0, "R3 fixture: could not measure a row count"
    f3 = write("r3.yaml", f"_paper_table: tab:main_results row {n + 500}\n")
    trip("table-rows-exist", paths=[f3], rendered=rend, pdfs=pdfs)

    # R4 -- a rendered number that contradicts the .aux.
    doc, num = rend["tab:main_results"]
    wrong = "XCIX" if ROMAN.match(num) else "99"
    assert wrong != num
    f4 = write("r4.md", f"The headline is paper Tab. {wrong} (tab:main_results).\n")
    trip("rendered-number-agrees", paths=[f4], rendered=rend, pdfs=pdfs)

    # R5 -- an empty harvest.
    trip("reference-census", paths=[write("r5.md", "no pointers here\n")], rendered=rend, pdfs=pdfs)

    shutil.rmtree(tmp, ignore_errors=True)
    total = 7
    print(f"SELFTEST OK: {total - len(missed)}/{total} checks provably fail when broken"
          if not missed else
          f"SELFTEST FAILED: {len(missed)} check(s) did not fail when broken: {missed}")
    return 1 if missed else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    raise SystemExit(report(run(ref_sources())))

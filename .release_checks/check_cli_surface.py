#!/usr/bin/env python3
"""GATE: an argparse `choices` value the dispatch has no branch for.

DEFECT THIS EXISTS FOR (measured, v2.3.8 / HEAD 07725af):
  LINE NUMBERS IN THIS BLOCK ARE A DATED SNAPSHOT of the checkout named above, not
  navigation. Several have already moved: follow the SYMBOL, the heading or the quoted
  text, and re-derive the location with `grep -n`. Every check below RE-MEASURES the
  live artefacts, so nothing here is load-bearing for a verdict.
  `--tta flip_y` is a first-class part of the released CLI surface --
    scripts/eval.py:48        parser.add_argument("--tta", choices=["none","flip_y","d4"])
    scripts/infer.py:103      p.add_argument("--tta", choices=["none","flip_y","d4"])
    src/gssc/inference/evaluate.py:279   documents it in the docstring
    tests/test_config_loader.py:52       asserts configs may declare it
  but every consumer is TWO-way:
    src/gssc/inference/evaluate.py:375   if tta_mode == "d4": ... else: <plain generator>
    scripts/infer.py:172                 if tta == "d4":     ... else: <plain generator>
  and the banner at evaluate.py:333 prints `tta=%s` with the requested mode. So
  `--tta flip_y` runs NO test-time augmentation, prints "tta=flip_y", and returns a
  number that reads as a flip-TTA measurement. `grep -rn flip_y` over the repo confirms
  the string is compared NOWHERE outside augmentation code: there is no handler at all.

WHY A GATE AND NOT A ONE-LINE FIX: the shape is generic. A k-way `choices` list served
by a 2-way `if/else` is invisible to argparse (the value validates), invisible to the
tests (they only assert membership) and invisible to a green run (it produces a number).
So this gate does not look for flip_y. It enumerates EVERY statically-known `choices`
list under scripts/, follows the option value through assignment aliases and one hop of
argument passing, collects every literal the code actually dispatches on, and reports
any option whose dispatch cannot distinguish two or more of its own advertised values.

THE RULE, AND WHY IT IS SHAPED THIS WAY
  A trailing `else` is a legitimate handler for exactly ONE residual choice (that is how
  `--tta none` is served). It cannot serve two: whichever of the two the user asks for,
  the same code runs, so at least one of them is silently wrong. Hence:
      residual = choices - {literals the code compares against}
      residual <= 1                     -> OK  (the else is that one value's branch)
      dispatch falls through to a raise  -> OK  (the mode is refused, loudly)
      otherwise                         -> FAIL, naming the unhandled values
  Options nothing dispatches on at all (e.g. --log-level, fed to getattr(logging, ...))
  are consumed as VALUES, not dispatched, so branch coverage is not the right question
  for them; they are reported as such and not failed. That distinction is measured (zero
  dispatch sites), not assumed per-option -- which is why it tracks the code by itself:
  --synthetic-pool was listed here as a second value-only example until 2026-08-23, when
  the DataPort DOI landed and its branch grew a real per-variant comparison; the gate
  moved it to "2/2 choices branch explicitly" with no edit to the rule.

BLIND SPOT, DELIBERATELY MADE VISIBLE: a `choices=` expression this file cannot resolve
statically means the gate cannot see that option at all -- so `choices_all_enumerable`
FAILS rather than passing quietly. (reproduce_table.py:190 uses
`choices=selectable_names()`; the resolver below evaluates that form -- a return of a
set-union of module-level collections -- so it is enumerable today.)

ROOTS, AND WHAT IS NOT PART OF THE PUBLIC RELEASE
-------------------------------------------------
Every root below is an environment variable with a repo-relative default, so this gate
measures the checkout it ships in rather than one particular machine.  Absolute paths
were hardcoded here once; a relocated clone then audited a tree it was not running in,
and the paths themselves disclosed the maintainer's local layout to every visitor.

    GSSC_REPO        the release checkout under test        default: this file's repository
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("GSSC_REPO") or Path(__file__).resolve().parents[1])
SCRIPTS = REPO / "scripts"
# The dispatch for a script's option routinely lives in the library, not the script:
# eval.py's --tta is consumed 300 lines away in src/gssc/inference/evaluate.py. Any gate
# that only reads the script it found the flag in would have declared flip_y "handled".
LIB = REPO / "src" / "gssc"

MAX_DETAIL = 3


# ---------------------------------------------------------------- corpus


def _corpus() -> list[tuple[str, str]]:
    """(relpath, source) for every .py the dispatch could live in."""
    out = []
    for base in (SCRIPTS, LIB):
        for p in sorted(base.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            out.append((str(p.relative_to(REPO)), p.read_text(encoding="utf-8", errors="replace")))
    return out


def _parse(corpus: list[tuple[str, str]]) -> list[tuple[str, ast.Module]]:
    trees = []
    for rel, src in corpus:
        try:
            trees.append((rel, ast.parse(src)))
        except SyntaxError as e:  # a file that cannot parse is a hole in the gate
            raise SystemExit(f"check_cli_surface: cannot parse {rel}:{e.lineno}: {e.msg}")
    return trees


# ---------------------------------------------------------------- static tables


def _str_seq(node: ast.AST) -> set[str] | None:
    """{'a','b'} for a literal list/tuple/set of strings, else None."""
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return None
    vals = set()
    for e in node.elts:
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            vals.add(e.value)
        else:
            return None
    return vals


def _collect_tables(trees) -> dict[str, dict[str, set[str]]]:
    """Module-level `NAME = {...}` / `NAME = [...]` string-key collections, per file.

    Needed because dict dispatch (`_SPLIT_TO_GEN[split]`, `if name in TABLE_MAP`) is a
    branch per key even though there is no `if` in sight.
    """
    tables: dict[str, dict[str, set[str]]] = {}
    for rel, tree in trees:
        per: dict[str, set[str]] = {}
        for node in tree.body:
            targets = []
            if isinstance(node, ast.Assign):
                targets, val = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, val = [node.target], node.value
            else:
                continue
            for t in targets:
                if not isinstance(t, ast.Name):
                    continue
                if isinstance(val, ast.Dict):
                    keys = {k.value for k in val.keys
                            if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                    if keys and len(keys) == len(val.keys):
                        per[t.id] = keys
                else:
                    seq = _str_seq(val)
                    if seq:
                        per[t.id] = seq
        tables[rel] = per
    return tables


def _funcs(trees) -> dict[str, list[tuple[str, ast.FunctionDef]]]:
    """name -> [(relpath, def)] for every function/method in the corpus (one-hop targets)."""
    out: dict[str, list[tuple[str, ast.FunctionDef]]] = {}
    for rel, tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.setdefault(node.name, []).append((rel, node))
    return out


# ---------------------------------------------------------------- choices discovery


class Option:
    def __init__(self, rel, line, flag, dest, choices, dynamic_src=None):
        self.rel, self.line, self.flag, self.dest = rel, line, flag, dest
        self.choices = choices          # set[str] | None when unresolved
        self.dynamic_src = dynamic_src  # text of the unresolved expression


def _dest_of(call: ast.Call) -> tuple[str, str] | None:
    for kw in call.keywords:
        if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value), str(kw.value.value)
    for a in call.args:
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            flag = a.value
            return flag, flag.lstrip("-").replace("-", "_")
    return None


def _resolve_dynamic_choices(expr: ast.AST, rel: str, funcs, tables) -> set[str] | None:
    """Resolve `choices=selectable_names()`-style expressions.

    Only the shape that exists: a zero-arg local function whose body is a single
    `return sorted({*A, *B, ...})` over module-level string collections. Anything else
    returns None and is reported as un-enumerable rather than assumed fine.
    """
    if not isinstance(expr, ast.Call):
        return None
    fname = expr.func.id if isinstance(expr.func, ast.Name) else None
    if fname is None or fname not in funcs:
        return None
    for frel, fdef in funcs[fname]:
        body = [s for s in fdef.body if not isinstance(s, ast.Expr)]
        if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
            continue
        val = body[0].value
        if isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id in ("sorted", "list", "set", "tuple"):
            val = val.args[0] if val.args else None
        acc: set[str] = set()
        for e in getattr(val, "elts", []):
            inner = e.value if isinstance(e, ast.Starred) else e
            if isinstance(inner, ast.Name) and inner.id in tables.get(frel, {}):
                acc |= tables[frel][inner.id]
            elif isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                acc.add(inner.value)
            else:
                return None
        return acc or None
    return None


def _find_options(trees, funcs, tables) -> list[Option]:
    opts = []
    for rel, tree in trees:
        if not rel.startswith("scripts/"):
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                continue
            ch = next((kw.value for kw in node.keywords if kw.arg == "choices"), None)
            if ch is None:
                continue
            named = _dest_of(node)
            if named is None:
                continue
            flag, dest = named
            lits = _str_seq(ch)
            if lits is None:
                lits = _resolve_dynamic_choices(ch, rel, funcs, tables)
                src = ast.unparse(ch)
                opts.append(Option(rel, node.lineno, flag, dest, lits, None if lits else src))
            else:
                opts.append(Option(rel, node.lineno, flag, dest, lits))
    return opts


# ---------------------------------------------------------------- alias closure


def _refs(node: ast.AST, subjects: set[str]) -> bool:
    """Does this expression read one of the subject names (bare, or as `args.<subj>`)?"""
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in subjects:
            return True
        if isinstance(n, ast.Attribute) and n.attr in subjects:
            return True
    return False


def _alias_closure(dest: str, trees, funcs) -> set[str]:
    """Names the option value can be flowing under, by assignment or one call hop.

    Deliberately shallow (3 rounds, direct references only). A wider closure would
    absorb unrelated variables and their literals, and every absorbed literal makes
    the gate BLINDER, not noisier: coverage can only grow. That asymmetry is why this
    is kept tight instead of "clever".
    """
    subjects = {dest}
    for _ in range(3):
        grew = False
        for rel, tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and _refs(node.value, subjects):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id not in subjects:
                            subjects.add(t.id)
                            grew = True
                if isinstance(node, ast.Call):
                    fname = (node.func.id if isinstance(node.func, ast.Name)
                             else node.func.attr if isinstance(node.func, ast.Attribute) else None)
                    if fname not in funcs:
                        continue
                    for _frel, fdef in funcs[fname]:
                        params = [a.arg for a in fdef.args.args] + [a.arg for a in fdef.args.kwonlyargs]
                        for i, a in enumerate(node.args):
                            if _refs(a, subjects) and i < len(params) and params[i] not in subjects:
                                subjects.add(params[i])
                                grew = True
                        for kw in node.keywords:
                            if kw.arg and _refs(kw.value, subjects) and kw.arg not in subjects:
                                subjects.add(kw.arg)
                                grew = True
        if not grew:
            break
    return subjects


def _subject_name(node: ast.AST, subjects: set[str]) -> str | None:
    if isinstance(node, ast.Name) and node.id in subjects:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in subjects:
        return node.attr
    return None


# ---------------------------------------------------------------- dispatch sites


class Site:
    def __init__(self, rel, line, kind, lits):
        self.rel, self.line, self.kind, self.lits = rel, line, kind, lits

    def where(self):
        return f"{self.rel}:{self.line}"


def _dispatch_sites(subjects: set[str], trees, tables) -> tuple[list[Site], list[str]]:
    """Every literal the code actually branches on, plus the sites whose fall-through raises."""
    sites: list[Site] = []
    raising: list[str] = []
    for rel, tree in trees:
        per_tables = tables.get(rel, {})
        for node in ast.walk(tree):
            # x == "d4" / x != "d4" / x in ("a","b") / x in TABLE
            if isinstance(node, ast.Compare) and _subject_name(node.left, subjects):
                for op, cmp in zip(node.ops, node.comparators):
                    if not isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)):
                        continue
                    if isinstance(cmp, ast.Constant) and isinstance(cmp.value, str):
                        sites.append(Site(rel, node.lineno, "compare", {cmp.value}))
                    elif (seq := _str_seq(cmp)) is not None:
                        sites.append(Site(rel, node.lineno, "compare", seq))
                    elif isinstance(cmp, ast.Name) and cmp.id in per_tables:
                        sites.append(Site(rel, node.lineno, f"in {cmp.id}", set(per_tables[cmp.id])))
            # TABLE[x] / TABLE.get(x)
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                    and node.value.id in per_tables and _subject_name(node.slice, subjects):
                sites.append(Site(rel, node.lineno, f"{node.value.id}[]", set(per_tables[node.value.id])))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "get" and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id in per_tables and node.args \
                    and _subject_name(node.args[0], subjects):
                sites.append(Site(rel, node.lineno, f"{node.func.value.id}.get",
                                  set(per_tables[node.func.value.id])))
            # match x: case "d4": ...
            if isinstance(node, ast.Match) and _subject_name(node.subject, subjects):
                for case in node.cases:
                    pat = case.pattern
                    if isinstance(pat, ast.MatchValue) and isinstance(pat.value, ast.Constant) \
                            and isinstance(pat.value.value, str):
                        sites.append(Site(rel, node.lineno, "match", {pat.value.value}))
                    if isinstance(pat, ast.MatchAs) and pat.pattern is None \
                            and _body_aborts(case.body):
                        raising.append(f"{rel}:{node.lineno}")
            # if/elif chain terminating in a raising else
            if isinstance(node, ast.If) and _pure_dispatch(node.test, subjects):
                term = node
                while len(term.orelse) == 1 and isinstance(term.orelse[0], ast.If):
                    term = term.orelse[0]
                if term.orelse and _body_aborts(term.orelse):
                    raising.append(f"{rel}:{node.lineno}")
        # a dispatching function whose LAST statement is a bare raise refuses anything
        # it did not match (reproduce_table.resolve_table:181 is exactly this shape).
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body \
                    and isinstance(node.body[-1], ast.Raise) \
                    and any(_subject_name(n.left, subjects)
                            for n in ast.walk(node) if isinstance(n, ast.Compare)):
                raising.append(f"{rel}:{node.body[-1].lineno}")
    return sites, raising


def _pure_dispatch(test: ast.AST, subjects: set[str]) -> bool:
    """`tta == "d4"` yes; `tta != "d4" and gen_split == "train"` NO.

    infer.py:166 is that compound form, and its body raises. Counting it as a raising
    fall-through would have declared --tta fully handled -- the exact way this gate
    could have been fooled into green. A guard with extra conjuncts is not the
    dispatch's else arm.
    """
    return isinstance(test, ast.Compare) and _subject_name(test.left, subjects) is not None \
        and all(isinstance(o, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for o in test.ops)


def _body_aborts(body: list[ast.stmt]) -> bool:
    for st in body:
        if isinstance(st, (ast.Raise, ast.Assert)):
            return True
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
            f = st.value.func
            if isinstance(f, ast.Attribute) and f.attr in ("exit", "abort"):
                return True
            if isinstance(f, ast.Name) and f.id in ("exit", "quit"):
                return True
    return False


# ---------------------------------------------------------------- the checks


def analyse(corpus: list[tuple[str, str]]) -> list[tuple[str, bool, str]]:
    """-> [(check name, ok, detail)]. Pure function of the corpus, so --selftest can
    feed it a synthetic one."""
    trees = _parse(corpus)
    tables = _collect_tables(trees)
    funcs = _funcs(trees)
    opts = _find_options(trees, funcs, tables)
    results: list[tuple[str, bool, str]] = []

    if not opts:
        results.append(("choices_found", False,
                        "no argparse choices= found under scripts/ -- the gate is looking "
                        "at the wrong tree or the CLI moved"))
        return results
    results.append(("choices_found", True, f"{len(opts)} option(s) with choices"))

    unresolved = [o for o in opts if o.choices is None]
    results.append((
        "choices_all_enumerable",
        not unresolved,
        "; ".join(f"{o.rel}:{o.line} {o.flag}: choices={o.dynamic_src} not statically "
                  f"resolvable, so its branch coverage is UNMEASURED" for o in unresolved),
    ))

    for o in opts:
        if o.choices is None:
            continue
        name = f"dispatch_covers:{o.rel}:{o.line}:{o.flag}"
        subjects = _alias_closure(o.dest, trees, funcs)
        sites, raising = _dispatch_sites(subjects, trees, tables)
        sites = [s for s in sites if s.lits & o.choices]
        covered: set[str] = set()
        for s in sites:
            covered |= s.lits & o.choices
        residual = sorted(o.choices - covered)
        if not sites:
            results.append((name, True,
                            f"consumed as a value ({len(o.choices)} choices, no dispatch "
                            f"found on {sorted(subjects)[:4]}...) -- branch coverage N/A"))
        elif len(residual) <= 1:
            results.append((name, True,
                            f"{len(covered)}/{len(o.choices)} choices branch explicitly"
                            + (f"; '{residual[0]}' left to the else arm" if residual else "")))
        elif raising:
            results.append((name, True,
                            f"{len(residual)} choices have no branch {residual[:MAX_DETAIL]} "
                            f"but the fall-through raises at {raising[0]}"))
        else:
            where = ", ".join(sorted({s.where() for s in sites}))[:160]
            results.append((name, False,
                            f"{o.rel}:{o.line} advertises {sorted(o.choices)}; dispatch at "
                            f"{where} branches only on {sorted(covered)}; "
                            f"{residual} run the SAME fall-through path and cannot be "
                            f"distinguished -- at least one silently does the wrong thing"))
    return results


# ---------------------------------------------------------------- selftest


_CLEAN = '''
import argparse
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tta", choices=["none", "flip_y", "d4"])
    a = p.parse_args()
    mode = a.tta
    if mode == "d4":
        run_d4()
    elif mode == "flip_y":
        run_flip()
    else:
        run_plain()
'''

_DYN = '''
import argparse
def pick():
    return sorted({*OTHER})
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--x", choices=pick())
'''


def _one(corpus, prefix):
    return [(n, ok, d) for n, ok, d in analyse(corpus) if n.startswith(prefix)]


def selftest() -> int:
    missed = []

    # 1. dispatch_covers -- drop the middle arm, exactly the live flip_y shape.
    broken = _CLEAN.replace('    elif mode == "flip_y":\n        run_flip()\n', "")
    assert broken != _CLEAN, "selftest fault did not change the source (pattern drifted)"
    ok_clean = all(ok for _, ok, _ in _one([("scripts/s.py", _CLEAN)], "dispatch_covers"))
    got = _one([("scripts/s.py", broken)], "dispatch_covers")
    hit = bool(got) and not all(ok for _, ok, _ in got)
    print(f"  {'TRIPPED' if (hit and ok_clean) else 'MISSED  '} dispatch_covers"
          f"{'' if ok_clean else '   (clean corpus also failed -- instrument is not specific)'}")
    missed += [] if (hit and ok_clean) else ["dispatch_covers"]

    # 2. choices_all_enumerable -- an unresolvable choices= expression must be reported,
    #    not silently skipped. Clean control: the same file with a literal list.
    lit = _DYN.replace("choices=pick()", 'choices=["a", "b"]')
    assert lit != _DYN
    clean_ok = all(ok for _, ok, _ in _one([("scripts/s.py", lit)], "choices_all_enumerable"))
    dyn_ok = all(ok for _, ok, _ in _one([("scripts/s.py", _DYN)], "choices_all_enumerable"))
    hit = clean_ok and not dyn_ok
    print(f"  {'TRIPPED' if hit else 'MISSED  '} choices_all_enumerable")
    missed += [] if hit else ["choices_all_enumerable"]

    # 3. choices_found -- an empty corpus must not read as success.
    hit = not all(ok for _, ok, _ in _one([("scripts/s.py", "x = 1\n")], "choices_found"))
    print(f"  {'TRIPPED' if hit else 'MISSED  '} choices_found")
    missed += [] if hit else ["choices_found"]

    n = 3
    print(f"SELFTEST OK: {n - len(missed)}/{n} checks provably fail when broken")
    return 1 if missed else 0


def main() -> int:
    results = analyse(_corpus())
    bad = 0
    for name, ok, detail in results:
        if ok:
            print(f"  PASS  {name}" + (f"   ({detail})" if detail else ""))
        else:
            bad += 1
            print(f"  FAIL  {name}   ({detail})")
    print(f"OK: 0 failing check(s)" if not bad else f"FAILED: {bad} failing check(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(selftest() if "--selftest" in sys.argv else main())

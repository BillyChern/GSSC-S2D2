#!/usr/bin/env python3
"""GATE: `load_state_dict(..., strict=False)` whose result nobody looks at.

DEFECT THIS EXISTS FOR (measured, v2.3.8 / HEAD 07725af):
  src/gssc/inference/d4_tta.py:111,114 and
  src/gssc/inference/generate_predictions.py:186,189 load the deployment weights with
  `strict=False` and DISCARD the returned `_IncompatibleKeys`, while the architecture is
  hardcoded three lines above (d4_tta.py:100-105, `base_channels=32, time_emb_dim=128,
  ...`) instead of read from the checkpoint's own config.json. Those two facts compose
  into the worst failure mode available to an eval script: an architecture that does not
  match the checkpoint loads ANYWAY, the unmatched tensors stay at random init, and the
  run prints a plausible mIoU. Nothing raises; nothing is even logged.

  THE EVIDENCE THAT THIS IS NOT HYPOTHETICAL IS IN THIS REPO. It already happened once,
  on the BEV path, and the fix is still sitting there: src/gssc/inference/evaluate_bev.py
  defines `_assert_bound` (line 79) whose docstring records the incident verbatim -- the
  denoiser was rebuilt at input_resolution=64 / cond_channels=128 against a 256/64
  checkpoint, "48 attention tensors stayed at initialisation", and the path "still
  returned a number". The shipped checkpoint config confirms it:
  checkpoints/bev/bev_s2d2_scpnet/config.json carries a `note_release_default_wrong`
  field saying exactly that, plus the `reconstruction` block that now feeds the loader.
  So the correct guard EXISTS, is BATTLE-TESTED, and was never applied to the two
  inference paths that produce the headline numbers.

WHAT THIS GATE ASSERTS
  strict_load_bound:<file>:<line>    every strict=False call binds its result
  strict_load_guarded:<file>:<line>  every bound result reaches an ABORTING guard
                                     (raise / assert / a corpus function that raises).
                                     Logging the key counts is NOT a guard: a warning
                                     does not stop the number from being printed, and
                                     four sites in this repo do exactly that today.
  exemplar_guard_present             `_assert_bound`-shaped guard still exists somewhere
                                     in src/gssc/ -- so "fixing" the gate by deleting the
                                     exemplar is caught.
  arch_from_checkpoint:<file>:<line> a release path that speaks our checkpoint layout
                                     must not hardcode kwargs the shipped config.json
                                     files declare. Key set and values are READ from
                                     GSSC-S2D2-assets/checkpoints/**/config.json, never
                                     hardcoded here; a literal that contradicts a shipped
                                     value is called out separately in the detail.

SEVERITY POLICY (measured, not per-file taste)
  BLOCKING on the release surface: scripts/ and src/gssc/inference/.
  ADVISORY (printed as NOTE, does not fail) for:
    - training code (src/gssc/training/, src/gssc/models/, src/gssc/data/), where
      partial loads are deliberate -- e.g. train_scene_completion.py:923 loads a teacher
      that legitimately lacks student-only modules -- and blocking them would force a
      wrong fix during a release push;
    - vendored upstream code, detected by a NOTICE file next to it
      (src/gssc/_improved_diffusion/NOTICE), not by a hardcoded path. Delete the NOTICE
      and those sites become blocking, which is the correct behaviour: un-vendored code
      is ours.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "gssc"
SCRIPTS = REPO / "scripts"
# data/checkpoints is a symlink into GSSC-S2D2-assets; resolve() follows it.
CKPT_ROOT = (REPO / "data" / "checkpoints").resolve()

RELEASE_PREFIXES = ("scripts/", "src/gssc/inference/")
# A file that reads these keys is loading OUR checkpoint layout, so OUR config.json sits
# beside the weights it just loaded and there is no excuse for a hardcoded architecture.
# Third-party baseline dumpers (JS3C's `model*.pth`, LMSCNet's ckpt["model"]) do not
# match, and are exempted from the architecture check only -- never from bound/guarded.
GSSC_CKPT_MARKERS = ("ema_shadow", "model_state_dict")


# ---------------------------------------------------------------- corpus


def _vendored(p: Path) -> bool:
    """Upstream code, evidenced by a NOTICE beside it -- not by a path this gate knows."""
    for anc in [p.parent, *p.parents]:
        if anc == REPO.parent:
            break
        if (anc / "NOTICE").exists():
            return True
        if anc == REPO:
            break
    return False


def corpus() -> list[tuple[str, str, bool]]:
    """(relpath, source, advisory)."""
    out = []
    for base in (SRC, SCRIPTS):
        for p in sorted(base.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            rel = str(p.relative_to(REPO))
            advisory = _vendored(p) or not rel.startswith(RELEASE_PREFIXES)
            out.append((rel, p.read_text(encoding="utf-8", errors="replace"), advisory))
    return out


def _link(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]


def _enclosing_func(node: ast.AST):
    cur = getattr(node, "parent", None)
    while cur is not None and not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
        cur = getattr(cur, "parent", None)
    return cur


def _stmt_of(node: ast.AST):
    cur = node
    while cur is not None and not isinstance(cur, ast.stmt):
        cur = getattr(cur, "parent", None)
    return cur


# ---------------------------------------------------------------- guards


def _raising_funcs(trees) -> set[str]:
    """Functions that can abort. `_assert_bound` is one; `logger.info` wrappers are not."""
    names = set()
    for _rel, tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(isinstance(n, (ast.Raise, ast.Assert)) for n in ast.walk(node)):
                    names.add(node.name)
    return names


def _names_of(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [e.id for e in target.elts if isinstance(e, ast.Name)]
    return []


def _mentions(node: ast.AST, names: set[str]) -> bool:
    return any(isinstance(n, ast.Name) and n.id in names for n in ast.walk(node))


def _guarded(bound: set[str], func: ast.AST, raising: set[str]) -> str | None:
    """Return the guard's description, or None. An ABORT is required, not a log line."""
    for node in ast.walk(func):
        if isinstance(node, ast.Assert) and _mentions(node, bound):
            return f"assert at line {node.lineno}"
        if isinstance(node, ast.If) and _mentions(node.test, bound):
            if any(isinstance(s, (ast.Raise, ast.Assert)) for s in ast.walk(node)):
                # only count a raise that lives in the arm this test controls
                for s in node.body + node.orelse:
                    if isinstance(s, (ast.Raise, ast.Assert)):
                        return f"if/raise at line {node.lineno}"
        if isinstance(node, ast.Call):
            fname = (node.func.id if isinstance(node.func, ast.Name)
                     else node.func.attr if isinstance(node.func, ast.Attribute) else None)
            if fname in raising and any(_mentions(a, bound) for a in node.args) \
                    and not (isinstance(node.func, ast.Attribute)
                             and isinstance(node.func.value, ast.Name)
                             and node.func.value.id in ("logger", "log", "logging")):
                return f"{fname}() at line {node.lineno}"
    return None


# ---------------------------------------------------------------- shipped configs


def shipped_config_keys(root: Path = CKPT_ROOT) -> dict[str, set]:
    """key -> set of declared scalar values, over every shipped checkpoint config.json.

    Read, never assumed: if the author adds a checkpoint whose config declares a new
    architecture key, the architecture check widens by itself.
    """
    keys: dict[str, set] = {}
    if not root.exists():
        return keys
    for cfg in sorted(root.rglob("config.json")):
        try:
            d = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        scopes = [d] + [d[k] for k in ("train_config", "reconstruction") if isinstance(d.get(k), dict)]
        for scope in scopes:
            for k, v in scope.items():
                if isinstance(v, (int, float, bool, str)):
                    keys.setdefault(k, set()).add(v)
    return keys


# ---------------------------------------------------------------- analysis


def analyse(files: list[tuple[str, str, bool]], cfg_keys: dict[str, set],
            require_assets: bool = True) -> list[tuple[str, bool, str, bool]]:
    """-> [(name, ok, detail, advisory)]. Pure function of its inputs so --selftest can
    hand it a synthetic corpus."""
    trees = []
    for rel, src, adv in files:
        try:
            t = ast.parse(src)
        except SyntaxError as e:
            raise SystemExit(f"check_strict_load: cannot parse {rel}:{e.lineno}: {e.msg}")
        _link(t)
        trees.append((rel, t))
    adv_of = {rel: adv for rel, _s, adv in files}
    raising = _raising_funcs(trees)
    results: list[tuple[str, bool, str, bool]] = []

    # The exemplar guard must survive. If it is deleted or reduced to logging, every
    # "apply the existing guard" instruction in the release plan has lost its referent.
    exemplar = []
    for rel, tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and any(isinstance(n, ast.Raise) for n in ast.walk(node)) \
                    and "missing_keys" in ast.dump(node) and "unexpected_keys" in ast.dump(node):
                exemplar.append(f"{rel}:{node.lineno} {node.name}")
    results.append(("exemplar_guard_present", bool(exemplar),
                    exemplar[0] if exemplar else
                    "no function in the corpus raises on missing/unexpected keys -- the "
                    "_assert_bound exemplar (was src/gssc/inference/evaluate_bev.py:79) is gone",
                    False))

    n_sites = 0
    for rel, tree in trees:
        advisory = adv_of[rel]
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load_state_dict"):
                continue
            strict = next((kw.value for kw in node.keywords if kw.arg == "strict"), None)
            if not (isinstance(strict, ast.Constant) and strict.value is False):
                continue  # strict=True already raises on mismatch: nothing to guard
            n_sites += 1
            stmt = _stmt_of(node)
            where = f"{rel}:{node.lineno}"
            bound = _names_of(stmt.targets[0]) if isinstance(stmt, ast.Assign) and stmt.targets else []
            results.append((
                f"strict_load_bound:{where}", bool(bound),
                (f"binds {bound}" if bound else
                 f"{where}: load_state_dict(strict=False) result DISCARDED -- a checkpoint "
                 f"that does not match this architecture loads silently and the run still "
                 f"prints a score. Bind the result and guard it like "
                 f"src/gssc/inference/evaluate_bev.py:247 (_assert_bound)"),
                advisory))
            if not bound:
                continue
            func = _enclosing_func(node)
            g = _guarded(set(bound), func, raising) if func is not None else None
            results.append((
                f"strict_load_guarded:{where}", g is not None,
                (f"guarded by {g}" if g else
                 f"{where}: {bound} is bound but never aborts on a mismatch (logging the "
                 f"key counts still returns a number). Pass it to the existing guard "
                 f"_assert_bound (src/gssc/inference/evaluate_bev.py:79)"),
                advisory))

    if not n_sites:
        results.append(("strict_false_sites_found", False,
                        "no load_state_dict(strict=False) found at all -- the corpus is "
                        "wrong or the call moved behind a wrapper this gate cannot see",
                        False))
    else:
        results.append(("strict_false_sites_found", True, f"{n_sites} site(s) inspected", False))

    # ---- architecture must come from the checkpoint, not from a literal
    if require_assets and not cfg_keys:
        results.append(("shipped_configs_readable", False,
                        f"no config.json under {CKPT_ROOT} -- the architecture check "
                        f"cannot run, so nothing here is verified", False))
        return results
    if require_assets:
        results.append(("shipped_configs_readable", True,
                        f"{len(cfg_keys)} distinct keys across shipped config.json", False))

    for rel, tree in trees:
        if adv_of[rel] or not rel.startswith(RELEASE_PREFIXES):
            continue
        src_text = ast.dump(tree)
        if not any(m in src_text for m in GSSC_CKPT_MARKERS):
            continue  # not our checkpoint layout -> our config.json is not beside it
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id[:1].isupper()):
                continue
            func = _enclosing_func(node)
            if func is None or not any(
                    isinstance(n, ast.Call) and (
                        (isinstance(n.func, ast.Attribute) and n.func.attr in ("load_state_dict", "load"))
                        or (isinstance(n.func, ast.Name) and n.func.id == "load_file"))
                    for n in ast.walk(func)):
                continue  # constructed somewhere that never touches a checkpoint
            if not any(kw.arg for kw in node.keywords):
                continue  # Path(x), FileNotFoundError(msg): not a parameterised build
            hard = []
            for kw in node.keywords:
                if kw.arg is None or not isinstance(kw.value, ast.Constant):
                    continue
                if kw.arg not in cfg_keys:
                    continue
                declared = sorted(cfg_keys[kw.arg], key=str)
                # NOT "this contradicts the checkpoint": the same key name can belong to
                # two different models (base_channels is 32 for the 3D denoiser and 128
                # in the BEV reconstruction block), so a mismatch here is NOT evidence of
                # a wrong value. What IS evidence: a key whose declared value VARIES
                # across shipped checkpoints cannot be correct as one literal.
                note = (f" [varies across shipped checkpoints: {declared[:4]}]"
                        if len(declared) > 1 else f" [shipped: {declared[0]!r}]")
                hard.append(f"{kw.arg}{note}")
            if hard:
                results.append((
                    f"arch_from_checkpoint:{rel}:{node.lineno}", False,
                    f"{rel}:{node.lineno} {node.func.id}(...) hardcodes {hard} -- these keys "
                    f"are declared by the shipped checkpoint config.json, so read them from "
                    f"the checkpoint (see evaluate_bev.py, which reads its 'reconstruction' "
                    f"block) instead of pinning them in the release path",
                    False))
            else:
                results.append((f"arch_from_checkpoint:{rel}:{node.lineno}", True,
                                f"{node.func.id}(...) pins no config-declared key", False))
    return results


# ---------------------------------------------------------------- selftest

_CLEAN = '''
def _assert_bound(name, result, path):
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(name)

def load_model(p, device):
    ckpt = torch.load(p)
    state_dict = ckpt["model_state_dict"]
    model = Net()
    res = model.load_state_dict(state_dict, strict=False)
    _assert_bound("model", res, p)
    return model
'''


def _sub(results, prefix):
    return [(n, ok, d) for n, ok, d, _a in results if n.startswith(prefix)]


def selftest() -> int:
    missed = []
    keys = {"num_classes": {20}}

    def run(src, cfg=None, req=False):
        return analyse([("src/gssc/inference/x.py", src, False)],
                       keys if cfg is None else cfg, require_assets=req)

    clean = run(_CLEAN)
    if not all(ok for _n, ok, _d, _a in clean):
        print("  MISSED   (baseline corpus is not clean: "
              + "; ".join(n for n, ok, _d, _a in clean if not ok) + ")")
        return 1

    # 1. bound -- discard the result.
    broken = _CLEAN.replace("    res = model.load_state_dict(state_dict, strict=False)\n"
                            '    _assert_bound("model", res, p)\n',
                            "    model.load_state_dict(state_dict, strict=False)\n")
    assert broken != _CLEAN, "fault did not change the source (pattern drifted)"
    hit = not all(ok for _n, ok, _d in _sub(run(broken), "strict_load_bound"))
    print(f"  {'TRIPPED' if hit else 'MISSED  '} strict_load_bound")
    missed += [] if hit else ["strict_load_bound"]

    # 2. guarded -- keep the binding, downgrade the guard to a log line (the live shape
    #    at scripts/eval_semanticposs.py:152 and three others).
    broken = _CLEAN.replace('    _assert_bound("model", res, p)\n',
                            "    logger.info('%d missing', len(res.missing_keys))\n")
    assert broken != _CLEAN, "fault did not change the source (pattern drifted)"
    sub = _sub(run(broken), "strict_load_guarded")
    hit = bool(sub) and not all(ok for _n, ok, _d in sub)
    print(f"  {'TRIPPED' if hit else 'MISSED  '} strict_load_guarded")
    missed += [] if hit else ["strict_load_guarded"]

    # 3. exemplar -- delete the guard function itself.
    broken = _CLEAN[_CLEAN.index("def load_model"):]
    assert "raise" not in broken
    hit = not all(ok for _n, ok, _d in _sub(run(broken), "exemplar_guard_present"))
    print(f"  {'TRIPPED' if hit else 'MISSED  '} exemplar_guard_present")
    missed += [] if hit else ["exemplar_guard_present"]

    # 4. arch_from_checkpoint -- pin a key the (synthetic) shipped config declares.
    broken = _CLEAN.replace("model = Net()", "model = Net(num_classes=20)")
    assert broken != _CLEAN, "fault did not change the source (pattern drifted)"
    hit = not all(ok for _n, ok, _d in _sub(run(broken), "arch_from_checkpoint"))
    print(f"  {'TRIPPED' if hit else 'MISSED  '} arch_from_checkpoint")
    missed += [] if hit else ["arch_from_checkpoint"]

    # 5. assets missing must FAIL, not skip quietly.
    hit = not all(ok for _n, ok, _d in _sub(run(_CLEAN, cfg={}, req=True), "shipped_configs_readable"))
    print(f"  {'TRIPPED' if hit else 'MISSED  '} shipped_configs_readable")
    missed += [] if hit else ["shipped_configs_readable"]

    # 6. sites_found -- a corpus with no strict=False must not read as success.
    hit = not all(ok for _n, ok, _d in _sub(run("x = 1\n"), "strict_false_sites_found"))
    print(f"  {'TRIPPED' if hit else 'MISSED  '} strict_false_sites_found")
    missed += [] if hit else ["strict_false_sites_found"]

    n = 6
    print(f"SELFTEST OK: {n - len(missed)}/{n} checks provably fail when broken")
    return 1 if missed else 0


def main() -> int:
    results = analyse(corpus(), shipped_config_keys())
    bad = 0
    for name, ok, detail, advisory in results:
        if ok:
            print(f"  PASS  {name}" + (f"   ({detail})" if detail else ""))
        elif advisory:
            print(f"  NOTE  {name}   ({detail})  [advisory: training/vendored, see docstring]")
        else:
            bad += 1
            print(f"  FAIL  {name}   ({detail})")
    print("OK: 0 failing check(s)" if not bad else f"FAILED: {bad} failing check(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(selftest() if "--selftest" in sys.argv else main())

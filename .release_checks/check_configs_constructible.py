#!/usr/bin/env python3
"""GATE: a shipped config that cannot even build its diffusion object.

DEFECT THIS EXISTS FOR (measured, v2.3.8 / HEAD 07725af). EVERY `file:line` in this block is
a SNAPSHOT of that checkout and several have already rotted -- the defect is FIXED in the
current tree, so follow the SYMBOL names, never the line numbers:
  MultinomialDiffusion3DV2.__init__ logged the noise schedule with

      logger.debug(..., alphas_cumprod[0].item(), alphas_cumprod[50].item(),
                        alphas_cumprod[99].item(), ...)

  Python evaluates those arguments BEFORE logging decides whether DEBUG is enabled, so
  the literal indices 50 and 99 are read unconditionally. The schedule has exactly
  `num_timesteps` entries. Measured on this checkout (torch, in-process):

      T=100 -> ctor OK      T=50 -> IndexError: index 50 is out of bounds for dim 0 with size 50
                            T=10 -> IndexError: index 50 is out of bounds for dim 0 with size 10

  configs/train/T10.yaml declares `num_timesteps: 10` and configs/train/T50.yaml
  declares 50. Both are SHIPPED configs with paper provenance notes in their headers, and
  NEITHER CAN START. A user following the repo gets a traceback out of a logging call.

  The same hardcoding was in the sampler: `MultinomialDiffusion3DV2.sample_algo2` built the
  timestep schedule as `range(99, -1, -1)` / `np.linspace(99, 0, n_steps)`, so even with the
  logger fixed it would index alphas[99] for a T=10 model. `configs/eval/
  timestep_ablation.yaml` already documented this ("does NOT read num_timesteps from
  the checkpoint") -- the defect was known and written down, just not fixed or gated.

  BOTH ARE FIXED TODAY and the gate is what keeps them fixed: `multinomial.py` now derives
  `t_last = num_timesteps - 1` in both places. This gate CONSTRUCTS every shipped config
  and RUNS the sampler, so it re-measures rather than replaying this transcript.

WHAT THIS GATE DOES
  It does not lint. For EVERY shipped config under configs/{train,eval,infer}/ it reads
  the schedule values the config itself declares and CONSTRUCTS the diffusion object at
  those values in this process, then runs the release sampler (`sample_algo2`) on a 4x4x2
  toy grid with a stub denoiser, for each step count that config declares. A config that
  cannot be instantiated is a config that cannot be run.

  constructible:<config>              ctor at the config's own num_timesteps
  sampler_runs:<config>               sample_algo2 at each step count the config declares
  schedule_indices_in_range:<config>  every declared timestep INDEX (train_timesteps_list,
                                      *_skew_idx, *_start_step, *_t_start) is < num_timesteps
  no_hardcoded_schedule_literals      static: no literal equal to the ctor's DEFAULT
                                      num_timesteps (or T-1, or T//2) is used as an index /
                                      range bound inside the diffusion module. This is the
                                      one static check here, and it is load-bearing: fixing
                                      only the logger.debug turns the dynamic checks green
                                      while sample_algo2:1167 still hardcodes 99. The
                                      forbidden values are DERIVED from the class signature,
                                      not written down here.

WHICH CLASS: the release inference paths -- `load_model` in generate_predictions.py and in
d4_tta.py -- construct MultinomialDiffusion3DV2 unconditionally, and
train_scene_completion.py picks V2 whenever `diffusion_version: v2`, which every shipped
train config declares. (Symbols, not line numbers: all three of the pointers that stood here
had rotted.)
So V2 is the default here and V1 is used only if a config explicitly says v1.

EXPECTED TODAY: FAIL on configs/train/T10.yaml and configs/train/T50.yaml, and on
no_hardcoded_schedule_literals. If those go green without multinomial.py changing,
this gate has been broken, not the defect fixed.

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
import inspect
import logging
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("GSSC_REPO") or Path(__file__).resolve().parents[1])
CONFIGS = REPO / "configs"
DIFF_SRC = REPO / "src" / "gssc" / "diffusion" / "multinomial.py"
CONFIG_DIRS = ("train", "eval", "infer")

# Scratch stays out of /tmp: a full /tmp has deadlocked this box repeatedly.
os.environ.setdefault("TMPDIR", str(Path.home() / ".cache"))
sys.path.insert(0, str(REPO / "src"))

# Toy grid: the point is that the schedule arithmetic runs, not that the output is good.
B, H, W, D = 1, 4, 4, 2
# Keys a config uses to declare "how many correction steps".
STEP_KEYS = ("correction_steps", "algo2_eval_steps", "steps")
# Keys that name a timestep INDEX rather than a count.
INDEX_KEY_SUFFIXES = ("_list", "_idx", "_start_step", "_t_start")


def _lazy_imports():
    import torch
    import torch.nn as nn
    import yaml
    from gssc.diffusion import multinomial as M
    return torch, nn, yaml, M


# ---------------------------------------------------------------- config discovery


def shipped_configs(root: Path = CONFIGS):
    """[(relpath, dict)] for every shipped config. Read, never enumerated by hand."""
    _t, _n, yaml, _M = _lazy_imports()
    out = []
    for sub in CONFIG_DIRS:
        for p in sorted((root / sub).glob("*.yaml")):
            try:
                d = yaml.safe_load(p.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001 - a config that will not parse is a finding
                out.append((str(p.relative_to(REPO)), {"__parse_error__": str(e)}))
                continue
            out.append((str(p.relative_to(REPO)), d if isinstance(d, dict) else {}))
    return out


def _int_list(val) -> list[int]:
    """Timestep indices out of an int, a list, or the "0,11,22" string form used by
    train_timesteps_list."""
    if isinstance(val, bool):
        return []
    if isinstance(val, int):
        return [val]
    if isinstance(val, str):
        out = []
        for tok in val.split(","):
            tok = tok.strip()
            if tok.lstrip("-").isdigit():
                out.append(int(tok))
        return out
    if isinstance(val, (list, tuple)):
        return [v for v in val if isinstance(v, int) and not isinstance(v, bool)]
    return []


# ---------------------------------------------------------------- static detector


def hardcoded_schedule_literals(src: str) -> list[str]:
    """Literals equal to the DEFAULT num_timesteps (or T-1 / T//2) used as an index.

    The forbidden set is read off each class's own `num_timesteps` default, so it tracks
    the code instead of pinning 99 in this gate. Only index-shaped positions count
    (subscript slices, range()/linspace() bounds): `if n_steps >= 100` is a count
    comparison, not a schedule index, and flagging it would train the reader to ignore
    this check.
    """
    tree = ast.parse(src)
    findings = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        init = next((n for n in cls.body
                     if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        default_T = None
        if init is not None:
            args = init.args.args[1:] + init.args.kwonlyargs
            defaults = list(init.args.defaults)
            pad = len(args) - len(defaults)
            for i, a in enumerate(args):
                if a.arg == "num_timesteps" and i >= pad:
                    dv = defaults[i - pad]
                    if isinstance(dv, ast.Constant) and isinstance(dv.value, int):
                        default_T = dv.value
        if default_T is None:
            continue
        forbidden = {default_T, default_T - 1, default_T // 2}
        for node in ast.walk(cls):
            spots = []
            if isinstance(node, ast.Subscript):
                spots = [(node.slice, "index")]
            elif isinstance(node, ast.Call):
                fname = (node.func.id if isinstance(node.func, ast.Name)
                         else node.func.attr if isinstance(node.func, ast.Attribute) else "")
                if fname in ("range", "linspace", "arange"):
                    spots = [(a, f"{fname}() bound") for a in node.args]
            for expr, kind in spots:
                if isinstance(expr, ast.Constant) and expr.value in forbidden \
                        and isinstance(expr.value, int) and not isinstance(expr.value, bool):
                    findings.append(
                        f"{cls.name}: line {expr.lineno}: literal {expr.value} used as "
                        f"{kind}; = default num_timesteps({default_T}) family, so any "
                        f"config with a smaller T indexes out of range")
    return findings


# ---------------------------------------------------------------- dynamic checks


def _build(M, torch, cfg: dict):
    """Construct the diffusion object the way the release code does, with THIS config's
    values. kwargs are the intersection of the ctor signature and the config keys, so a
    new schedule knob is picked up without editing this gate."""
    version = str(cfg.get("diffusion_version", "v2"))
    cls = M.MultinomialDiffusion3DV2 if version == "v2" else M.MultinomialDiffusion3D
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for name in sig.parameters:
        if name in ("self", "class_weights"):
            continue
        if name in cfg and isinstance(cfg[name], (int, float, str, bool)):
            kwargs[name] = cfg[name]
    return cls(**kwargs), kwargs


def _stub(nn, torch, K: int):
    class Stub(nn.Module):
        def forward(self, x_t, t, bev, lidar, **kw):
            return torch.zeros(x_t.shape[0], K, *x_t.shape[2:])
    return Stub()


def analyse(configs, diff_src: str) -> list[tuple[str, bool, str]]:
    torch, nn, _yaml, M = _lazy_imports()
    # The defect hides inside logger.debug's ARGUMENTS, which Python evaluates whatever
    # the log level is. Silencing the logger must therefore NOT make this gate green --
    # that is why the level is raised here: if a future "fix" only guards the log call,
    # the ctor still has to survive.
    logging.getLogger("gssc.diffusion.multinomial").setLevel(logging.CRITICAL)

    results = []
    lits = hardcoded_schedule_literals(diff_src)
    results.append(("no_hardcoded_schedule_literals", not lits,
                    " | ".join(f"{DIFF_SRC.name}: {f}" for f in lits[:4])
                    or "no schedule index is pinned to the default T"))

    for rel, cfg in configs:
        if "__parse_error__" in cfg:
            results.append((f"constructible:{rel}", False, f"{rel}: {cfg['__parse_error__']}"))
            continue
        T_declared = cfg.get("num_timesteps")
        obj = None
        try:
            obj, kwargs = _build(M, torch, cfg)
            results.append((f"constructible:{rel}", True,
                            f"T={kwargs.get('num_timesteps', 'default')} "
                            f"{type(obj).__name__} built"))
        except Exception as e:  # noqa: BLE001 - any exception is the finding
            results.append((
                f"constructible:{rel}", False,
                f"{rel} declares num_timesteps={T_declared!r} and "
                f"{type(e).__name__}: {e} -- this shipped config cannot start. The defect "
                f"class to look for: a LITERAL timestep index (50, 99) in "
                f"MultinomialDiffusion3DV2.__init__'s schedule-logging block, inside an "
                f"eagerly-evaluated logger.debug -- grep multinomial.py for "
                f"`alphas_cumprod[`, not a line number"))

        # Every step count the config declares, exercised for real. The check LINE is
        # emitted whatever happens: a check that silently disappears when its subject
        # breaks reads as "nothing wrong here" on the next run.
        steps = sorted({s for k in STEP_KEYS for s in _int_list(cfg.get(k)) if s})
        if not steps:
            results.append((f"sampler_runs:{rel}", True,
                            f"config declares no step count ({'/'.join(STEP_KEYS)})"))
        elif obj is None:
            results.append((f"sampler_runs:{rel}", False,
                            f"{rel}: NOT EXERCISED -- the ctor failed above, so the "
                            f"{steps} declared step count(s) were never reached"))
        else:
            if not hasattr(obj, "sample_algo2"):
                results.append((f"sampler_runs:{rel}", True,
                                f"{type(obj).__name__} has no sample_algo2 (v1 path)"))
            else:
                bad = []
                for n in steps:
                    try:
                        obj.sample_algo2(
                            _stub(nn, torch, obj.num_classes),
                            torch.zeros(B, H, W, dtype=torch.long),
                            torch.zeros(B, 1, H, W, D),
                            torch.zeros(B, H, W, D, dtype=torch.long),
                            (B, H, W, D), torch.device("cpu"), n_steps=n)
                    except Exception as e:  # noqa: BLE001
                        bad.append(f"n_steps={n}: {type(e).__name__}: {e}")
                results.append((
                    f"sampler_runs:{rel}", not bad,
                    f"{sorted(steps)} step count(s) ran"
                    if not bad else
                    f"{rel}: sample_algo2 fails at {bad[:2]} -- the defect class to look "
                    f"for: MultinomialDiffusion3DV2.sample_algo2 building its timestep list "
                    f"as range(99,-1,-1)/linspace(99,0,n) from a LITERAL rather than from "
                    f"self.num_timesteps; grep multinomial.py for `linspace(` inside that "
                    f"method"))

        # declared timestep INDICES must exist in this config's schedule
        T = T_declared if isinstance(T_declared, int) else _default_T(M, cfg)
        oob = []
        for k, v in cfg.items():
            if not isinstance(k, str) or not k.endswith(INDEX_KEY_SUFFIXES):
                continue
            for idx in _int_list(v):
                if T is not None and idx >= T:
                    oob.append(f"{k}={idx} >= num_timesteps={T}")
        results.append((f"schedule_indices_in_range:{rel}", not oob,
                        "; ".join(f"{rel}: {o}" for o in oob[:3])
                        or f"all declared indices < T={T}"))
    return results


def _default_T(M, cfg) -> int | None:
    cls = M.MultinomialDiffusion3DV2 if str(cfg.get("diffusion_version", "v2")) == "v2" \
        else M.MultinomialDiffusion3D
    p = inspect.signature(cls.__init__).parameters.get("num_timesteps")
    return p.default if p and isinstance(p.default, int) else None


# ---------------------------------------------------------------- selftest


def _sub(res, prefix):
    return [(n, ok, d) for n, ok, d in res if n.startswith(prefix)]


def selftest() -> int:
    src = DIFF_SRC.read_text(encoding="utf-8")
    missed = []
    good = [("synthetic/ok.yaml", {"diffusion_version": "v2", "num_timesteps": 100,
                                   "correction_steps": 2})]
    base = analyse(good, src)
    for name in ("constructible", "sampler_runs", "schedule_indices_in_range"):
        if not all(ok for _n, ok, _d in _sub(base, name)):
            print(f"  MISSED   {name}   (control config already fails -- not specific)")
            missed.append(name)

    # 1. constructible: a T the schedule arithmetic can never accept. A *string* T stays
    #    a permanent fault; injecting T=10 would go vacuous the day the defect is fixed.
    faulted = [("synthetic/bad_T.yaml", {"diffusion_version": "v2",
                                         "num_timesteps": "banana"})]
    r = _sub(analyse(faulted, src), "constructible")
    hit = bool(r) and not all(ok for _n, ok, _d in r)
    print(f"  {'TRIPPED' if hit else 'MISSED  '} constructible")
    missed += [] if hit else ["constructible"]

    # 2. sampler_runs: a step count the sampler cannot honour, permanently.
    faulted = [("synthetic/bad_steps.yaml", {"diffusion_version": "v2", "num_timesteps": 100,
                                             "correction_steps": [2, -7]})]
    r = _sub(analyse(faulted, src), "sampler_runs")
    hit = bool(r) and not all(ok for _n, ok, _d in r)
    print(f"  {'TRIPPED' if hit else 'MISSED  '} sampler_runs")
    missed += [] if hit else ["sampler_runs"]

    # 3. schedule_indices_in_range: an index past the end of the declared schedule.
    faulted = [("synthetic/bad_idx.yaml", {"diffusion_version": "v2", "num_timesteps": 100,
                                           "train_timesteps_skew_idx": 400})]
    r = _sub(analyse(faulted, src), "schedule_indices_in_range")
    hit = bool(r) and not all(ok for _n, ok, _d in r)
    print(f"  {'TRIPPED' if hit else 'MISSED  '} schedule_indices_in_range")
    missed += [] if hit else ["schedule_indices_in_range"]

    # 4. no_hardcoded_schedule_literals: prove the detector reads the CODE, both ways.
    clean = (
        "class D:\n"
        "    def __init__(self, num_timesteps: int = 100):\n"
        "        a = f(num_timesteps)\n"
        "        x = a[num_timesteps - 1]\n"
        "        ts = list(range(num_timesteps - 1, -1, -1))\n")
    dirty = clean.replace("a[num_timesteps - 1]", "a[99]")
    assert dirty != clean, "fault did not change the source (pattern drifted)"
    ok_clean = not hardcoded_schedule_literals(clean)
    ok_dirty = bool(hardcoded_schedule_literals(dirty))
    hit = ok_clean and ok_dirty
    print(f"  {'TRIPPED' if hit else 'MISSED  '} no_hardcoded_schedule_literals"
          f"{'' if ok_clean else '   (fires on the FIXED shape too -- not specific)'}")
    missed += [] if hit else ["no_hardcoded_schedule_literals"]

    n = 4
    print(f"SELFTEST OK: {n - len(missed)}/{n} checks provably fail when broken")
    return 1 if missed else 0


def main() -> int:
    cfgs = shipped_configs()
    if not cfgs:
        print(f"  FAIL  configs_found   (no yaml under {CONFIGS}/{{{','.join(CONFIG_DIRS)}}})")
        print("FAILED: 1 failing check(s)")
        return 1
    print(f"  PASS  configs_found   ({len(cfgs)} shipped config(s))")
    bad = 0
    for name, ok, detail in analyse(cfgs, DIFF_SRC.read_text(encoding="utf-8")):
        if ok:
            print(f"  PASS  {name}" + (f"   ({detail})" if detail else ""))
        else:
            bad += 1
            print(f"  FAIL  {name}   ({detail})")
    print("OK: 0 failing check(s)" if not bad else f"FAILED: {bad} failing check(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(selftest() if "--selftest" in sys.argv else main())

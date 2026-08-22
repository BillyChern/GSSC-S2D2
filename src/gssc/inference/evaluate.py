"""GSSC-S2D2 evaluation pipeline.

Loads a checkpoint, runs S2D2 correction sampling on SemanticKITTI val, and
returns per-class IoU + mIoU + completion IoU.

The evaluation is a two-stage pipeline:

1. ``gssc.inference.generate_predictions`` (S2D2 correction sampler) writes ``.label``
   files for each frame in the requested split.
2. ``external/semantic_kitti_api/evaluate_completion.py`` reads those
   ``.label`` files and produces the official scoring numbers.

This module wires the two together into a single :func:`run_evaluation`
call, matching the eval/ Hydra configs in the repo.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_eval_completion_output(text: str) -> dict[str, float]:
    """Extract mIoU + IoU_cmpl + per-class IoU from evaluate_completion.py output.

    The official semantic_kitti_api script prints lines like::

        IoU class 1 [car] = 0.514
        IoU class 8 [motorcyclist] = 0.124
        Precision =\t86.04
        Recall   =\t57.59
        IoU Cmpltn =\t52.66
        mIoU SSC =\t38.54

    Per-class IoU is printed as a fraction in [0, 1]; aggregate metrics
    are already in percent. We normalise everything to percent.

    Args:
        text: Captured stdout/stderr of the scoring process.

    Returns:
        Mapping from metric name to percentage value (e.g. ``{"mIoU": 38.54,
        "IoU_cmpl": 52.66, "IoU_car": 51.4, "IoU_motorcyclist": 12.4, ...}``).
    """
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        # Aggregate metrics: tab-separated, already in percent.
        m = re.match(r"\s*mIoU\s+SSC\s*=\s*([\d.]+)", line)
        if m:
            metrics["mIoU"] = float(m.group(1))
            continue
        m = re.match(r"\s*IoU\s+Cmpltn\s*=\s*([\d.]+)", line)
        if m:
            metrics["IoU_cmpl"] = float(m.group(1))
            continue
        m = re.match(r"\s*Precision\s*=\s*([\d.]+)", line)
        if m:
            metrics["Precision"] = float(m.group(1))
            continue
        m = re.match(r"\s*Recall\s*=\s*([\d.]+)", line)
        if m:
            metrics["Recall"] = float(m.group(1))
            continue
        # Per-class IoU: "IoU class 1 [car] = 0.514" — fraction → percent.
        m = re.match(r"\s*IoU class \d+\s*\[([^\]]+)\]\s*=\s*([\d.]+)", line)
        if m:
            cls_name = m.group(1).strip().replace("-", "_").replace(" ", "_")
            metrics[f"IoU_{cls_name}"] = float(m.group(2)) * 100.0
    return metrics


def _resolve_config(config: str) -> dict:
    """Load a Hydra-style YAML config from the configs/ tree."""
    cfg_path = REPO_ROOT / "configs" / f"{config}.yaml"
    if not cfg_path.exists():
        candidates = sorted(p.relative_to(REPO_ROOT / "configs")
                            for p in (REPO_ROOT / "configs").rglob("*.yaml"))
        raise FileNotFoundError(
            f"Config not found: {cfg_path}\n"
            f"Available: {candidates}"
        )
    return yaml.safe_load(cfg_path.read_text())


def _run_step_sweep(
    checkpoint: str,
    config: str,
    data_root: str,
    step_values: list[int],
    output: str | None = None,
    gpu: str = "0",
    tta: str | None = None,
    metrics: list[str] | None = None,
    keep_predictions: bool = False,
) -> dict:
    """Run :func:`run_evaluation` once per N and aggregate per-N mIoU.

    Drives the list-valued ``correction_steps`` step sweep (e.g.
    ``eval/step_sweep`` -> tab:step_reduction). Each N reuses the full
    single-N pipeline via a recursive :func:`run_evaluation` call with an
    explicit ``steps=N`` override, so the generate + scoring logic is not
    duplicated.

    Args:
        step_values: Correction-step counts to sweep, in run order.
        output: If set, the aggregated sweep JSON is written here; per-N JSON
            dumps are skipped (only the summary is persisted).
        (other args mirror :func:`run_evaluation`.)

    Returns:
        Mapping with a ``"sweep"`` key (``{N: per-N metric dict}``), a
        ``"mIoU_by_steps"`` convenience map (``{N: mIoU}``), plus the metrics
        of the best-mIoU N hoisted to the top level for downstream consumers.
    """
    logger.info("Step sweep over correction_steps=%s", step_values)
    sweep: dict[int, dict] = {}
    miou_by_steps: dict[int, float] = {}
    for n in step_values:
        logger.info("=== Step sweep: N=%d ===", n)
        per_n = run_evaluation(
            checkpoint=checkpoint,
            config=config,
            data_root=data_root,
            output=None,  # avoid clobbering the aggregate JSON per-N
            gpu=gpu,
            steps=n,
            tta=tta,
            metrics=metrics,
            keep_predictions=keep_predictions,
        )
        sweep[n] = per_n
        if "mIoU" in per_n:
            miou_by_steps[n] = per_n["mIoU"]

    logger.info("Step sweep summary (correction_steps -> mIoU):")
    for n in step_values:
        if n in miou_by_steps:
            logger.info("  N=%-4d %6.2f %%", n, miou_by_steps[n])

    # Hoist the best-mIoU run's metrics to the top level so callers that read
    # ``metrics["mIoU"]`` (e.g. scripts/eval.py) still get a sensible scalar.
    result: dict = {}
    if miou_by_steps:
        best_n = max(miou_by_steps, key=lambda k: miou_by_steps[k])
        result.update(sweep[best_n])
        result["best_steps"] = best_n
    result["sweep"] = sweep
    result["mIoU_by_steps"] = miou_by_steps

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        logger.info("Wrote step-sweep metrics to %s", out_path)

    return result


#: Measured bytes per written ``.label`` frame at 256x256x32 (857 frames = 3.4 GB).
_BYTES_PER_FRAME = 4_060_000

#: Headroom kept free beyond the estimate: the larger of this and _RESERVE_FRACTION of the
#: filesystem. Both exist because "it fits" and "it is safe to run" are different questions.
_MIN_RESERVE = 8 * 2**30
_RESERVE_FRACTION = 0.10


def _count_frames(data_root: Path, sequences: str) -> int:
    """Number of frames Stage 1 will write across `sequences`.

    Counts ``velodyne/*.bin`` because that is the input Stage 1 iterates. A sequence whose
    directory is missing contributes 0 rather than raising: this feeds a disk-space
    estimate, and a wrong estimate must never be the thing that stops an evaluation.

    Args:
        data_root: Repository data root containing ``SemanticKITTI/``.
        sequences: Comma-separated sequence ids, e.g. ``"08"``.

    Returns:
        Total frame count; 0 if nothing could be counted.
    """
    total = 0
    for seq in (s.strip() for s in sequences.split(",") if s.strip()):
        # Both layouts are in the wild: the raw SemanticKITTI download nests everything under
        # dataset/, while a repo-local checkout usually drops that level. Try both rather than
        # assuming, because guessing wrong yields 0 frames -- and 0 frames makes the caller's
        # space check demand 0 bytes, i.e. a guard that silently always passes.
        for base in (data_root / "SemanticKITTI" / "sequences",
                     data_root / "SemanticKITTI" / "dataset" / "sequences"):
            seq_dir = base / seq / "velodyne"
            if seq_dir.is_dir():
                total += sum(1 for _ in seq_dir.glob("*.bin"))
                break
    return total


def _assert_scratch_space(target: Path, n_frames: int) -> None:
    """Fail before writing if `target`'s filesystem cannot hold the predictions.

    Stage 1 writes one ``.label`` per frame, so a full SemanticKITTI val sequence needs
    roughly 16 GB. Without this check that lands in the system temp directory, which on a
    container is usually the small overlay backing ``/`` -- the run fills the root
    filesystem and takes the machine down with it rather than failing. Raising here turns
    a wedged host into a message naming the remedy.

    Args:
        target: Directory the predictions will be written under.
        n_frames: Number of frames Stage 1 will write.

    Raises:
        RuntimeError: If free space is below the estimated requirement.
    """
    if n_frames <= 0:
        logger.warning(
            "could not count frames under %s, so the disk-space check is inactive for this "
            "run; if the temp filesystem is small, redirect TMPDIR before starting", target)
        return

    need = n_frames * _BYTES_PER_FRAME
    usage = shutil.disk_usage(target)
    # Reserve headroom rather than merely fitting. A run that fits with 5 GiB to spare still
    # leaves the filesystem near-full for everything else on the host, and if that filesystem
    # is the root overlay the consequence is a wedged machine, not a failed command. Sizing
    # this from measurement: the 4071-frame val run needs 15.4 GiB and /tmp offered 21.4 GiB,
    # so a bare fits/does-not-fit test passes the exact case that motivated this check.
    reserve = max(_MIN_RESERVE, int(usage.total * _RESERVE_FRACTION))
    on_root = os.stat(target).st_dev == os.stat("/").st_dev
    if usage.free >= need + reserve:
        if on_root:
            logger.warning(
                "scratch resolves onto the ROOT filesystem (%s); writing %.1f GiB there will "
                "leave %.1f GiB. Prefer TMPDIR on a data volume", target, need / 2**30,
                (usage.free - need) / 2**30)
        return

    where = " -- and it is the ROOT filesystem, so filling it wedges the host" if on_root else ""
    raise RuntimeError(
        f"{target} has {usage.free / 2**30:.1f} GiB free but this evaluation writes about "
        f"{need / 2**30:.1f} GiB ({n_frames} frames) and reserves {reserve / 2**30:.1f} GiB "
        f"of headroom{where}. Point TMPDIR at a larger volume "
        f"(TMPDIR=/path/with/room python scripts/eval.py ...), or pass --keep-predictions to "
        f"write under the data root instead."
    )


def run_evaluation(
    checkpoint: str,
    config: str,
    data_root: str,
    output: str | None = None,
    gpu: str = "0",
    steps: int | None = None,
    tta: str | None = None,
    metrics: list[str] | None = None,
    keep_predictions: bool = False,
) -> dict:
    """Run a SemanticKITTI evaluation end-to-end.

    Args:
        checkpoint: Path to the trained checkpoint (``.pt`` or ``.safetensors``).
        config: Hydra-style config name, e.g. ``"eval/val_1step"``.
        data_root: Repository data root (must contain ``SemanticKITTI/`` and
            ``scpnet_predictions/``).
        output: Optional path to dump per-class metrics as JSON.
        gpu: CUDA device id (default ``"0"``).
        steps: Correction-step count override; ``None`` uses the config default.
        tta: TTA mode override; one of ``{"none", "d4"}``.
        metrics: Advisory only. The official semantic-kitti-api scorer emits
            mIoU, completion IoU and per-class IoU in a single pass, so this
            list neither subsets the computation nor the returned dict; it is
            kept for forward compatibility with a future subset selector.
        keep_predictions: If True, keep the intermediate ``.label`` files
            in ``<data_root>/predictions/<config_name>/``. If False
            (default), they go to a temporary directory.

    Returns:
        Mapping from metric name (``"mIoU"``, ``"IoU_cmpl"``, per-class
        ``"IoU_<class>"``) to percentage value. When the config sets
        ``correction_steps`` to a *list* (and no ``--steps`` override is
        given), the pipeline runs once per N and returns the step-sweep
        mapping from :func:`_run_step_sweep` instead: a ``"sweep"`` key
        (``{N: per-N metrics}``), a ``"mIoU_by_steps"`` map, and the
        best-mIoU run's metrics hoisted to the top level.

    Raises:
        FileNotFoundError: If the checkpoint or data root is missing.
        RuntimeError: If generate_predictions or evaluate_completion fails.
    """
    metrics = metrics or ["miou", "completion_iou", "per_class"]
    ckpt_path = Path(checkpoint).resolve()
    data_root_path = Path(data_root).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not data_root_path.exists():
        raise FileNotFoundError(f"Data root not found: {data_root_path}")

    cfg = _resolve_config(config)
    sequences = str(cfg.get("sequences", "08"))

    # Step-sweep support: a config may set ``correction_steps`` to a *list* of
    # N values (e.g. eval/step_sweep.yaml reproduces tab:step_reduction). When
    # no explicit ``--steps`` override is given, run the full pipeline once per
    # N, aggregate the per-N mIoU, and return a sweep mapping. A scalar config
    # (or any ``--steps`` override) keeps the single-N path below.
    cfg_steps = cfg.get("correction_steps", cfg.get("algo2_steps", 1))
    if steps is None and isinstance(cfg_steps, (list, tuple)):
        return _run_step_sweep(
            checkpoint=checkpoint,
            config=config,
            data_root=data_root,
            step_values=[int(n) for n in cfg_steps],
            output=output,
            gpu=gpu,
            tta=tta,
            metrics=metrics,
            keep_predictions=keep_predictions,
        )

    n_steps = steps if steps is not None else int(cfg_steps)
    tta_mode = tta if tta is not None else str(cfg.get("tta", "none"))
    if tta_mode not in ("none", "d4"):
        # Fail before the banner: it prints `tta=<mode>`, and the dispatch below used to
        # fall through to the plain no-TTA generator for anything that was not "d4"
        # (`flip_y` was advertised in the CLI choices and documented in this docstring,
        # but nothing anywhere dispatched on it). A mode with no implementation must
        # never be reported as if it ran.
        raise ValueError(
            f"Unsupported TTA mode {tta_mode!r}. Implemented modes: 'none' (plain "
            f"single-pass generation) and 'd4' (8-element dihedral group, "
            f"gssc.inference.d4_tta). Refusing to run a different pipeline under this "
            f"mode's name."
        )

    logger.info("Eval config: %s", config)
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Sequences=%s steps=%d tta=%s", sequences, n_steps, tta_mode)

    pred_dir_ctx: tempfile.TemporaryDirectory[str] | None = None
    if keep_predictions:
        pred_dir = data_root_path / "predictions" / config.replace("/", "_")
        pred_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Predictions kept at: %s", pred_dir)
    else:
        scratch_root = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
        _assert_scratch_space(scratch_root, _count_frames(data_root_path, sequences))
        pred_dir_ctx = tempfile.TemporaryDirectory(prefix="gssc_eval_preds_")
        pred_dir = Path(pred_dir_ctx.name)
        logger.info("Predictions in temp dir: %s (auto-cleaned on success)", pred_dir)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

    # Resolve the base-prediction directory. Modern configs (since v1.1.0)
    # specify ``base_pred_dir`` to support cross-base reproduction (JS3C-Net,
    # PaSCo, ...). Legacy configs fall back to ``scpnet_pred_dir``; both feed
    # the same downstream subprocess.
    base_pred_dir_cfg = cfg.get("base_pred_dir") or cfg.get("scpnet_pred_dir")
    if base_pred_dir_cfg is None:
        scpnet_dir = data_root_path / "scpnet_predictions"
    else:
        base_pred_dir_path = Path(base_pred_dir_cfg)
        if base_pred_dir_path.is_absolute():
            scpnet_dir = base_pred_dir_path
        else:
            scpnet_dir = data_root_path.parent / base_pred_dir_path
            if not scpnet_dir.exists():
                scpnet_dir = data_root_path / base_pred_dir_path.name
    semantic_kitti_root = data_root_path / "SemanticKITTI"
    if not semantic_kitti_root.exists():
        semantic_kitti_root = data_root_path

    cfg_split = str(cfg.get("split", "valid"))
    gen_split = "valid" if cfg_split == "val" else cfg_split
    bev_source = str(cfg.get("bev_source", "derived"))
    bev_root_cfg = cfg.get("bev_root")

    if tta_mode == "d4":
        cmd = [
            sys.executable, "-m", "gssc.inference.d4_tta",
            "--checkpoint", str(ckpt_path),
            "--cold_steps", str(n_steps),
            "--output_dir", str(pred_dir),
            "--data_root", str(semantic_kitti_root),
            "--scpnet_dir", str(scpnet_dir),
            "--split", gen_split,
            "--bev_source", bev_source,
        ]
    else:
        cmd = [
            sys.executable, "-m", "gssc.inference.generate_predictions",
            "--checkpoint", str(ckpt_path),
            "--cold_steps", str(n_steps),
            "--output_dir", str(pred_dir),
            "--data_root", str(semantic_kitti_root),
            "--scpnet_dir", str(scpnet_dir),
            "--split", gen_split,
            "--algo2",
            "--bev_from_base",
            "--bev_source", bev_source,
        ]
    if bev_root_cfg:
        bev_root_path = Path(str(bev_root_cfg))
        if not bev_root_path.is_absolute():
            bev_root_path = data_root_path / bev_root_path
        cmd.extend(["--bev_root", str(bev_root_path)])
    # When predictions are persisted, an interrupted/repeat run can resume by
    # skipping frames that already have a .label written.
    if keep_predictions:
        cmd.append("--skip_existing")

    logger.info("Stage 1 (generate): %s", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        if pred_dir_ctx is not None:
            pred_dir_ctx.cleanup()
        raise RuntimeError(
            f"generate_predictions failed (exit {proc.returncode})\n"
            f"STDOUT: {proc.stdout[-2000:]}\n"
            f"STDERR: {proc.stderr[-2000:]}"
        )
    logger.info("Stage 1 done. Predictions written under %s", pred_dir)

    eval_script = REPO_ROOT / "external" / "semantic_kitti_api" / "evaluate_completion.py"
    eval_dir = eval_script.parent
    eval_datacfg = eval_dir / "config" / "semantic-kitti.yaml"

    score_cmd = [
        sys.executable, str(eval_script),
        "--dataset", str(semantic_kitti_root),
        "--predictions", str(pred_dir),
        "--split", gen_split,
        "--datacfg", str(eval_datacfg),
    ]
    logger.info("Stage 2 (score): %s", " ".join(score_cmd))
    score_proc = subprocess.run(score_cmd, env=env, capture_output=True, text=True, cwd=str(eval_dir))
    combined = score_proc.stdout + "\n" + score_proc.stderr
    if score_proc.returncode != 0:
        # Keep predictions on scoring failure so the user can rerun stage 2
        # without spending another ~10 min on stage 1.
        if pred_dir_ctx is not None:
            persist = data_root_path / "predictions" / f"{config.replace('/', '_')}_failed"
            persist.parent.mkdir(parents=True, exist_ok=True)
            try:
                Path(pred_dir).rename(persist)
                logger.error("Stage 2 failed; predictions retained at: %s", persist)
            except OSError:
                logger.error("Stage 2 failed; predictions still under: %s", pred_dir)
        raise RuntimeError(
            f"evaluate_completion failed (exit {score_proc.returncode})\n"
            f"STDOUT: {score_proc.stdout[-2000:]}\n"
            f"STDERR: {score_proc.stderr[-2000:]}"
        )

    parsed = _parse_eval_completion_output(combined)
    if not parsed:
        if pred_dir_ctx is not None:
            pred_dir_ctx.cleanup()
        raise RuntimeError(
            "Could not parse mIoU from evaluate_completion output. "
            f"Last 2000 chars:\n{combined[-2000:]}"
        )

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(parsed, indent=2))
        logger.info("Wrote per-class metrics to %s", out_path)

    for k in ("mIoU", "IoU_cmpl"):
        if k in parsed:
            logger.info("%-12s %.2f %%", k, parsed[k])

    if pred_dir_ctx is not None:
        pred_dir_ctx.cleanup()

    return parsed

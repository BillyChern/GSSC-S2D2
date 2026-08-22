#!/usr/bin/env python3
"""Zero-shot SSCBench-KITTI360 val eval driver.

Single-command orchestrator for the cross-dataset experiment:

    SemanticKITTI-trained SCPNet  --zero-shot-->  S²D² (gssc_31k_mf_step40000)  -->  KITTI-360 GT

Stages
------
1. **SCPNet zero-shot** on KITTI-360 raw LiDAR (HDL-64E, identical 256x256x32
   grid). Reuses the SCPNet runner's ``prepare_input`` voxelizer verbatim
   (grid + region already match — see
   ``gssc.data.kitti360``). Emits ``<seq>/<frame>_pred.npy`` in SemanticKITTI
   train space (0-19). Skipped per-frame if already present.
2. **S²D² correction** per frame: ``MultinomialDiffusion3DV2.sample_algo2`` with
   the released ``gssc_31k_mf_step40000`` EMA checkpoint, cold_diffusion + bev_from_base,
   N=1. Identical call to ``gssc.inference.generate_predictions``.
3. **Class projection + writeback**: remap SemanticKITTI-train pred -> KITTI-360
   train space (``gssc.data.kitti360_class_map.remap_scpnet_to_kitti360``), then
   to KITTI-360 original 0-255 labels (``KITTI360_LEARNING_MAP_INV``); write
   ``.label`` + a bit-PACKED ``.invalid`` copy the official scorer can read.
4. **Score** with ``external/semantic_kitti_api/evaluate_completion.py`` against
   a KITTI-360 datacfg (``--datacfg``). That scorer's own headline
   (``mIoU SSC``) is the mean over **all 18** KITTI-360 classes, which includes
   other-structure and other-object -- two classes a SemanticKITTI-trained model
   can never emit, so they are structurally 0 and dilute the mean. Stage 5 below
   therefore re-averages the scorer's per-class IoUs over the **16 shared**
   SemanticKITTI-cap-KITTI-360 classes and reports THAT as the headline, which is
   the quantity the paper's cross-dataset sentence quotes. Both means are printed,
   each labelled with its class set; neither is derived from the other by
   assumption.
5. **Report** the 16-shared-class mIoU, read back from the ``scores.txt`` the
   scorer writes (per-class IoUs), never recomputed from the predictions.

This driver is intentionally separate from ``scripts/eval.py`` so the committed
SemanticKITTI path (seq 08, SemanticKITTI learning_map) is untouched.

Requires the SSCBench-KITTI360 release on disk (see
``configs/eval/kitti360_zeroshot_1step.yaml`` paths) and the KITTI-360
``datacfg`` (the SSCBench ``dataset/configs/kitti360.yaml``) shipped at
``external/semantic_kitti_api/config/kitti360.yaml``.

The SCPNet zero-shot stage shells out to the bundled SCPNet runner,
``src/gssc/inference/run_scpnet.py`` (kept a separate process because it installs
spconv-v1->v2 monkey-patches at import time). Point ``$SCPNET_RUNNER`` at a different
runner to override it. The runner needs an upstream SCPNet checkout; see its module
docstring for ``--scpnet-repo`` / ``$SCPNET_REPO``.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gssc.data.kitti360 import (
    GRID_SIZE,
    Kitti360EvalConfig,
    Kitti360SSCDataset,
    read_kitti360_invalid,
    repack_invalid_for_semantic_kitti_api,
)
from gssc.data.kitti360_class_map import (
    KITTI360_LEARNING_MAP_INV,
    SCORABLE_K360_CLASSES,
    STRUCTURAL_ZERO_K360_CLASSES,
    remap_scpnet_to_kitti360,
)

logger = logging.getLogger("eval_kitti360")


def seq_to_int_dir(drive_name: str) -> str:
    """KITTI-360 drive name -> integer seq dir the scorer expects (e.g. ..0006.. -> '06').

    ``evaluate_completion.py`` formats ``DATA["split"][split]`` (integers, e.g.
    ``[6]``) as ``{0:02d}`` and reads ``sequences/06/voxels``. So GT + predictions
    must be staged under the 2-digit integer dir, not the drive name.
    """
    import re
    m = re.search(r"drive_(\d+)_sync", drive_name)
    if not m:
        raise ValueError(f"Cannot parse KITTI-360 drive index from {drive_name!r}")
    # ..._drive_0006_sync -> 6 -> '06'.
    return f"{int(m.group(1)):02d}"

# SCPNet inference requires spconv-v1->v2 monkey-patches, so we shell out to a
# thin SCPNet runner instead of importing it (keeping those patches isolated).
# The runner ships with this repo; $SCPNET_RUNNER overrides it.
SCPNET_RUNNER = Path(os.environ.get(
    "SCPNET_RUNNER", str(REPO_ROOT / "src" / "gssc" / "inference" / "run_scpnet.py")))


def unpack_bits(compressed: np.ndarray) -> np.ndarray:
    """Bit-unpack a SemanticKITTI-format occupancy grid -> flat 0/1 uint8."""
    out = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    for i in range(8):
        out[i::8] = (compressed >> (7 - i)) & 1
    return out


def load_occupancy_bin(bin_path: str) -> torch.Tensor:
    """Load the SSCBench-KITTI360 occupancy ``voxels/.bin`` -> [1,1,256,256,32].

    SSCBench ``voxels/.bin`` follows the SemanticKITTI bit-packed convention
    (same as the SemanticKITTI occupancy input the S²D² LiDAR encoder expects).
    """
    compressed = np.fromfile(bin_path, dtype=np.uint8)
    binary = unpack_bits(compressed).reshape(GRID_SIZE).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)


def run_scpnet_zeroshot(dataset: Kitti360SSCDataset, out_dir: Path, gpu: int) -> None:
    """Stage 1: SCPNet zero-shot on KITTI-360 raw LiDAR -> <seq>/<frame>_pred.npy.

    Shells out to the SemanticKITTI SCPNet runner once per frame batch. The
    runner's ``prepare_input`` voxelizer is grid/region-identical to KITTI-360,
    so no SCPNet code change is needed; only the velodyne path differs.

    The SCPNet runner enumerates a SemanticKITTI ``sequences/<seq>/velodyne``
    layout and writes ``<out_dir>/<2-digit-seq>/<frame>_pred.npy``. For KITTI-360
    we stage a symlinked ``sequences/06/{velodyne,voxels}`` tree (velodyne scans
    here are renamed to 6-digit ids that align 1:1 with the voxel frames, so the
    runner's per-frame ``velodyne/<frame_id>.bin`` lookup is exact). Predictions
    are keyed by the 2-digit integer seq dir (``seq_to_int_dir``), matching the
    runner output. This function asserts the predictions exist and raises an
    actionable error otherwise (it does NOT silently fabricate).
    """
    missing = [
        f for f in dataset.frames
        if not (out_dir / seq_to_int_dir(f["sequence"]) / f"{f['frame_id']}_pred.npy").exists()
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(dataset)} SCPNet KITTI-360 predictions missing under "
            f"{out_dir}. Run the SCPNet zero-shot dump first. Stage a SemanticKITTI-"
            f"style symlink tree, then call the runner, e.g.:\n"
            f"  STAGE=<repo>/data/kitti360_scpnet_staging\n"
            f"  mkdir -p $STAGE/sequences/06\n"
            f"  ln -sfn <K360>/data_3d_raw/<drive>/velodyne_points/data $STAGE/sequences/06/velodyne\n"
            f"  ln -sfn <K360>/data_2d_raw/<drive>/voxels               $STAGE/sequences/06/voxels\n"
            f"  git clone --depth 1 https://github.com/SCPNet/Codes-for-SCPNet "
            f"external/Codes-for-SCPNet\n"
            f"  python {SCPNET_RUNNER} \\\n"
            f"    --scpnet-repo external/Codes-for-SCPNet \\\n"
            f"    --checkpoint external/Codes-for-SCPNet/model_load_dir/pretrained.pth \\\n"
            f"    --config external/Codes-for-SCPNet/config/semantickitti-multiscan.yaml \\\n"
            f"    --data_root $STAGE --sequences 06 \\\n"
            f"    --output_dir {out_dir} --gpu {gpu}\n"
            "(SCPNet's prepare_input voxelizer is grid-identical to KITTI-360; "
            "only the velodyne enumeration differs.)"
        )
    logger.info("Stage 1 OK: %d SCPNet KITTI-360 predictions present.", len(dataset))


def run_s2d2_and_writeback(
    dataset: Kitti360SSCDataset,
    scpnet_dir: Path,
    checkpoint: str,
    pred_out_dir: Path,
    n_steps: int,
    gpu: int,
) -> None:
    """Stages 2-3: S²D² correction + class projection + .label/.invalid writeback."""
    from safetensors.torch import load_file

    from gssc.diffusion.multinomial import MultinomialDiffusion3DV2
    from gssc.models.s2d2_unet import SceneCompletionUNetSparse
    from gssc.utils.checkpoint import assert_bound, config_value, load_checkpoint_config

    device = torch.device(f"cuda:{gpu}")

    # EMA deployment weights, mirroring gssc.inference.d4_tta.load_model:
    #   * safetensors (released v1.1.0 model_ema.safetensors) is already a full
    #     278-key state (EMA params + BN buffers) — load directly.
    #   * .pt (v1.0.0 training ckpt, e.g. the headline gssc_31k_mf_step40000.pt)
    #     holds model_state_dict (278: weights + BN buffers) AND ema_shadow (233:
    #     learnable params only). Load model_state_dict for the BN buffers, then
    #     overwrite every learnable param present in ema_shadow with its EMA value.
    #     This is the canonical EMA-inference state; using model_state_dict alone
    #     would silently deploy the NON-EMA raw weights.
    ckpt = None
    if checkpoint.endswith(".safetensors"):
        state_dict = load_file(checkpoint)
    else:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]

    # Architecture from whatever the checkpoint declares about itself (its sibling
    # config.json, or ckpt["config"]); the literals below are only the fallback for a
    # checkpoint that declares nothing. Same source as gssc.inference.d4_tta.load_model,
    # so this zero-shot driver cannot drift from the SemanticKITTI path whose number the
    # cross-dataset claim is compared against.
    cfg = load_checkpoint_config(checkpoint, ckpt)
    model = SceneCompletionUNetSparse(
        num_classes=config_value(cfg, "num_classes", 20),
        base_channels=config_value(cfg, "base_channels", 32),
        time_emb_dim=128,
        lidar_base_channels=16, lidar_out_channels=32, lidar_in_channels=1,
        no_bev=config_value(cfg, "no_bev", False),
        ssc_cond_channels=20,
        ssc_multiscale=config_value(cfg, "ssc_multiscale", False),
    ).to(device)

    # strict=False is kept (EMA shadows may omit non-float buffers) but the result is
    # CHECKED, not logged: an architecture that does not match the checkpoint used to
    # load silently, leave the unmatched tensors at random initialisation, and still
    # print a plausible zero-shot mIoU. A warning does not stop that number.
    load_res = model.load_state_dict(state_dict, strict=False)
    assert_bound("model", load_res, checkpoint)
    if ckpt is not None:
        if "ema_shadow" in ckpt and ckpt["ema_shadow"] is not None:
            swapped = 0
            for name, p in model.named_parameters():
                if name in ckpt["ema_shadow"]:
                    p.data.copy_(ckpt["ema_shadow"][name])
                    swapped += 1
            logger.info(
                "Swapped in EMA (ema_shadow) weights for %d/%d learnable params.",
                swapped, sum(1 for _ in model.named_parameters()),
            )
        else:
            logger.warning("No ema_shadow in checkpoint; deploying RAW model_state_dict weights.")
    model.train(False)

    diffusion = MultinomialDiffusion3DV2(
        num_classes=config_value(cfg, "num_classes", 20),
        num_timesteps=config_value(cfg, "num_timesteps", 100),
        beta_max=config_value(cfg, "beta_max", 0.1),
    ).to(device)

    try:
        from tqdm import tqdm
        frame_iter = tqdm(dataset.frames, desc=f"S2D2 N={n_steps}")
    except ImportError:
        frame_iter = dataset.frames

    for f in frame_iter:
        seq, frame_id = f["sequence"], f["frame_id"]
        scpnet_path = scpnet_dir / seq_to_int_dir(seq) / f"{frame_id}_pred.npy"
        scpnet_pred = np.load(scpnet_path)  # [256,256,32] uint8, SemanticKITTI train 0-19

        lidar = load_occupancy_bin(f["occupancy_bin_path"]).to(device)
        scp_tensor = torch.from_numpy(scpnet_pred.astype(np.int64)).unsqueeze(0).to(device)
        scp_oh = F.one_hot(scp_tensor, 20).float().permute(0, 4, 1, 2, 3)

        # Derived BEV: topmost-non-empty class from the SemanticKITTI-space base 3D pred.
        bev = torch.zeros(1, 256, 256, dtype=torch.long, device=device)
        for z in range(scp_tensor.shape[3] - 1, -1, -1):
            layer = scp_tensor[0, :, :, z]
            mask = (layer > 0) & (bev[0] == 0)
            bev[0][mask] = layer[mask]

        with torch.no_grad():
            pred = diffusion.sample_algo2(
                model, bev, lidar, scpnet_pred=scp_tensor,
                shape=(1, 256, 256, 32), device=device,
                n_steps=n_steps, show_progress=False, ssc_pred=scp_oh,
            )
        pred_sk = pred.cpu().numpy()[0]  # [256,256,32] SemanticKITTI train space 0-19

        # Project SemanticKITTI -> KITTI-360 train space, then to original 0-255.
        pred_k360_train = remap_scpnet_to_kitti360(pred_sk)
        pred_original = KITTI360_LEARNING_MAP_INV[pred_k360_train.astype(np.int64)]

        out_seq_dir = pred_out_dir / "sequences" / seq_to_int_dir(seq) / "predictions"
        out_seq_dir.mkdir(parents=True, exist_ok=True)
        pred_original.reshape(-1).astype(np.uint16).tofile(out_seq_dir / f"{frame_id}.label")

    logger.info("Stages 2-3 OK: wrote %d KITTI-360 .label predictions.", len(dataset))


def prepare_gt_for_scorer(dataset: Kitti360SSCDataset, gt_root: Path) -> None:
    """Stage 4a: stage KITTI-360 GT into the scorer's expected layout.

    The GSSC scorer reads ``<gt_root>/sequences/<seq>/voxels/<frame>.label`` +
    ``.invalid`` and BIT-UNPACKS ``.invalid``. KITTI-360 ships ``.invalid`` RAW,
    so we repack it; ``.label`` is copied as-is (already flat uint16). The GT
    .label is in KITTI-360 ORIGINAL labels (0-255), matching our prediction
    writeback and the KITTI-360 datacfg learning_map.
    """
    for f in dataset.frames:
        seq, frame_id = f["sequence"], f["frame_id"]
        dst = gt_root / "sequences" / seq_to_int_dir(seq) / "voxels"
        dst.mkdir(parents=True, exist_ok=True)
        # .label: copy raw bytes (flat uint16, KITTI-360 original labels).
        (dst / f"{frame_id}.label").write_bytes(Path(f["label_path"]).read_bytes())
        # .invalid: raw uint8 grid -> bit-packed for the scorer.
        invalid_grid = read_kitti360_invalid(f["invalid_path"])
        repack_invalid_for_semantic_kitti_api(invalid_grid, dst / f"{frame_id}.invalid")
    logger.info("Stage 4a OK: staged %d GT frames (.invalid repacked).", len(dataset))


def score(gt_root: Path, pred_root: Path, datacfg: Path, gpu: int,
          scores_dir: Path) -> int:
    """Stage 4b: official scorer with the KITTI-360 datacfg.

    The scorer's printed ``mIoU SSC`` is ``class_jaccard[1:].mean()`` -- an 18-class
    mean under this datacfg. ``--output`` is passed explicitly so its ``scores.txt``
    (per-class IoUs) lands somewhere this driver owns; :func:`report_shared_class_miou`
    reads it back. Without ``--output`` the scorer defaults to ``.`` and would write
    into ``external/semantic_kitti_api/``.
    """
    eval_script = REPO_ROOT / "external" / "semantic_kitti_api" / "evaluate_completion.py"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    scores_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(eval_script),
        "--dataset", str(gt_root),
        "--predictions", str(pred_root),
        "--split", "valid",     # NB: requires the KITTI-360 datacfg to list seq 06 under 'valid'
        "--datacfg", str(datacfg),
        "--output", str(scores_dir),
    ]
    logger.info("Stage 4b (score): %s", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, cwd=str(eval_script.parent))
    return proc.returncode


def class_names_by_train_index(datacfg: Path) -> dict[int, str]:
    """KITTI-360 train index -> class name, exactly as the scorer keys its scores.txt.

    The scorer writes ``iou_<class_strings[class_inv_remap[i]]>``; both tables come from
    the datacfg, so reading them from the same file is the only way this mapping cannot
    drift away from the scorer's own naming.
    """
    data = yaml.safe_load(datacfg.read_text())
    labels, inv = data["labels"], data["learning_map_inv"]
    return {int(k): labels[v] for k, v in inv.items()}


def shared_class_miou(scores: dict, names: dict[int, str],
                      scorable: Sequence[int]) -> tuple[float, dict[int, float]]:
    """Mean IoU over the shared classes only, plus the per-class values it averaged.

    Raises:
        KeyError: if a scorable class has no entry in the scorer's output. Better a loud
            failure than a mean quietly taken over fewer classes than it claims.
    """
    per = {int(k): float(scores["iou_" + names[int(k)]]) for k in scorable}
    return sum(per.values()) / len(per), per


def report_shared_class_miou(scores_dir: Path, datacfg: Path) -> None:
    """Stage 5: print the 16-shared-class headline beside the scorer's 18-class mean.

    The docstring of this script used to claim a 16-shared-class mIoU while printing only
    what the scorer printed, which is the 18-class mean -- a number roughly 2/18 lower,
    because other-structure and other-object are structurally 0 for a SemanticKITTI-trained
    model. Both are shown, each named by its class set, so neither can be quoted as the
    other.
    """
    scores_path = scores_dir / "scores.txt"
    if not scores_path.is_file():
        logger.warning("Stage 5 skipped: %s was not written by the scorer; the printed "
                       "mIoU above is the %d-class mean, NOT the shared-class headline.",
                       scores_path, len(SCORABLE_K360_CLASSES) + len(STRUCTURAL_ZERO_K360_CLASSES))
        return
    scores = yaml.safe_load(scores_path.read_text())
    names = class_names_by_train_index(datacfg)
    shared, per = shared_class_miou(scores, names, SCORABLE_K360_CLASSES)
    n_all = len(SCORABLE_K360_CLASSES) + len(STRUCTURAL_ZERO_K360_CLASSES)
    logger.info("=" * 72)
    logger.info(" SSCBench-KITTI360 zero-shot")
    logger.info("   mIoU  (%2d shared classes, headline) : %.2f %%",
                len(per), 100.0 * shared)
    logger.info("   mIoU  (%2d classes, scorer default)  : %.2f %%   "
                "[includes %s, structurally 0]", n_all, 100.0 * float(scores["iou_mean"]),
                ", ".join(names[k] for k in STRUCTURAL_ZERO_K360_CLASSES))
    logger.info("   Completion IoU                      : %.2f %%",
                100.0 * float(scores["iou_completion"]))
    logger.info("   scores.txt: %s", scores_path)
    logger.info("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="eval/kitti360_zeroshot_1step")
    ap.add_argument("--checkpoint", required=True,
                    help="S²D² checkpoint, e.g. data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors")
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    ap.add_argument("--datacfg", default=None,
                    help="KITTI-360 learning_map yaml (SSCBench dataset/configs/kitti360.yaml). "
                         "Defaults to external/semantic_kitti_api/config/kitti360.yaml")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(name)s] %(message)s")

    cfg = yaml.safe_load((REPO_ROOT / "configs" / f"{args.config}.yaml").read_text())
    n_steps = args.steps if args.steps is not None else int(cfg.get("correction_steps", 1))
    data_root = Path(args.data_root)

    def _resolve(p: str) -> Path:
        """Honour absolute paths; resolve repo-relative paths against REPO_ROOT."""
        pp = Path(p)
        return pp if pp.is_absolute() else (REPO_ROOT / pp)

    ds_cfg = Kitti360EvalConfig(
        kitti360_root=str(_resolve(cfg["kitti360_root"])),
        match_table=str(_resolve(cfg["match_table"])),
        split=cfg.get("split", "val"),
    )
    dataset = Kitti360SSCDataset(ds_cfg)
    logger.info("SSCBench-KITTI360 %s: %d frames (%s)", ds_cfg.split, len(dataset), dataset.sequences)
    logger.info("Scorable shared classes: %s", SCORABLE_K360_CLASSES)
    logger.info("Structural-zero classes (model can't emit): %s", STRUCTURAL_ZERO_K360_CLASSES)

    scpnet_dir = _resolve(cfg.get("base_pred_dir", "data/scpnet_kitti360_predictions"))
    pred_root = data_root / "predictions" / "kitti360_zeroshot"
    gt_root = data_root / "kitti360_gt_staged"
    datacfg = Path(args.datacfg) if args.datacfg else \
        REPO_ROOT / "external" / "semantic_kitti_api" / "config" / "kitti360.yaml"

    scores_dir = data_root / "eval_results" / "kitti360_zeroshot"

    run_scpnet_zeroshot(dataset, scpnet_dir, args.gpu)
    run_s2d2_and_writeback(dataset, scpnet_dir, args.checkpoint, pred_root, n_steps, args.gpu)
    prepare_gt_for_scorer(dataset, gt_root)
    rc = score(gt_root, pred_root, datacfg, args.gpu, scores_dir)
    if rc == 0:
        report_shared_class_miou(scores_dir, datacfg)
    sys.exit(rc)


if __name__ == "__main__":
    main()

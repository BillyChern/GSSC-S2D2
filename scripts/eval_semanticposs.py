#!/usr/bin/env python3
"""Zero-shot SemanticKITTI->SemanticPOSS eval driver.

Single-command orchestrator replicating the TALoS (jang2024talos, NeurIPS 2024)
Table 4 cross-dataset protocol:

    SemanticKITTI-trained SCPNet  --zero-shot-->  S²D² (gssc_31k_mf_step40000)  -->  POSS seq-02 GT

Stages
------
0. **Materialise POSS SSC GT** (one-time): SemanticPOSS ships only
   ``velodyne/.bin`` + per-point ``labels/.label``. ``gssc.data.semanticposs``
   voxelises each scan into a single-scan 256x256x32 SSC GT
   (``voxels/<frame>.label`` POSS-original 0-255, ``.invalid`` bit-packed
   ray-cast mask, ``.bin`` bit-packed occupancy) — the SemanticKITTI SSC grid +
   region SCPNet/TALoS use. Idempotent (skips frames already written).
1. **SCPNet zero-shot** on POSS raw LiDAR (40-beam Pandora, same .bin format).
   Reuses the bundled ``src/gssc/inference/run_scpnet.py`` (its ``prepare_input``
   voxelizer + the SCPNet grid/region are unchanged). Emits ``<seq>/<frame>_pred.npy`` in
   SemanticKITTI train space (0-19). Skipped per-frame if already present.
2. **S²D² correction** per frame: ``MultinomialDiffusion3DV2.sample_algo2`` with
   the released ``gssc_31k_mf_step40000`` checkpoint, cold_diffusion + bev_from_base,
   N=1. Identical call to ``gssc.inference.generate_predictions``.
3. **Class projection + writeback**: SemanticKITTI-train pred -> POSS train space
   (``gssc.data.poss_class_map.remap_scpnet_to_poss``, TALoS Table A.1), then to
   POSS original 0-255 labels (``SEMANTICPOSS_LEARNING_MAP_INV``); write
   ``.label`` the official scorer reads.
4. **Score** with ``external/semantic_kitti_api/evaluate_completion.py`` against
   the POSS datacfg (``external/semantic_kitti_api/config/semantic-poss.yaml``),
   reporting cIoU (IoU Cmpltn) + mIoU over the 11 POSS classes.

Separate from ``scripts/eval.py`` so the committed SemanticKITTI path (seq 08,
SemanticKITTI learning_map) is untouched.

Requires the raw SemanticPOSS release on disk (see
``configs/eval/semanticposs_seq02.yaml`` paths). The SCPNet zero-shot stage shells
out to the bundled runner ``src/gssc/inference/run_scpnet.py`` -- a separate process
because it installs spconv-v1->v2 monkey-patches at import time. Override the runner,
its checkpoint and its config with ``$SCPNET_RUNNER`` / ``$SCPNET_CKPT`` /
``$SCPNET_CFG``. The runner also needs an upstream SCPNet checkout for the network
definition (``--scpnet-repo`` / ``$SCPNET_REPO``).
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gssc.data.poss_class_map import (
    SCORABLE_POSS_CLASSES,
    SEMANTICPOSS_LEARNING_MAP_INV,
    STRUCTURAL_ZERO_POSS_CLASSES,
    remap_scpnet_to_poss,
)
from gssc.data.semanticposs import (
    SemanticPossEvalConfig,
    SemanticPossSSCDataset,
    load_occupancy_bin,
)

logger = logging.getLogger("eval_semanticposs")

# SCPNet inference needs the spconv v1->v2 monkey-patches in its runner, so we
# shell out to keep them isolated. The runner and the base weights ship with this
# repo; the SCPNet *config* comes from the upstream checkout. Override any of the
# three via the environment.
SCPNET_RUNNER = Path(os.environ.get(
    "SCPNET_RUNNER", str(REPO_ROOT / "src" / "gssc" / "inference" / "run_scpnet.py")))
SCPNET_CKPT = os.environ.get("SCPNET_CKPT", str(REPO_ROOT / "data" / "checkpoints" / "scpnet_v2_port.pth"))
SCPNET_CFG = os.environ.get(
    "SCPNET_CFG", str(REPO_ROOT / "external" / "Codes-for-SCPNet" / "config" / "semantickitti-multiscan.yaml"))


def materialise_gt(dataset: SemanticPossSSCDataset) -> None:
    """Stage 0: voxelise the POSS GT into voxel_gt_root (single-scan 256x256x32)."""
    n = dataset.materialise_gt(skip_existing=True)
    logger.info("Stage 0 OK: materialised %d new POSS voxel-GT frames (%d total).",
                n, len(dataset))


def run_scpnet_zeroshot(dataset: SemanticPossSSCDataset, scpnet_dir: Path,
                        voxel_gt_root: Path, gpu: int) -> None:
    """Stage 1: SCPNet zero-shot on POSS raw LiDAR -> <seq>/<frame>_pred.npy.

    The SCPNet runner enumerates frames from ``<data_root>/sequences/<seq>/voxels/
    *.bin`` and reads points from ``.../velodyne/<frame>.bin``. We point it at a
    root whose ``sequences/<seq>/velodyne`` is the raw POSS velodyne and whose
    ``sequences/<seq>/voxels`` is the materialised POSS voxel GT (Stage 0), so no
    SCPNet code change is needed. The runner must therefore see both dirs under
    one root — provision a symlinked view or pass a root that already nests them.

    This function asserts the predictions exist and raises an actionable error
    otherwise (it does NOT silently fabricate).
    """
    missing = [
        f for f in dataset.frames
        if not (scpnet_dir / f["sequence"] / f"{f['frame_id']}_pred.npy").exists()
    ]
    if missing:
        seqs = sorted({f["sequence"] for f in dataset.frames})
        raise RuntimeError(
            f"{len(missing)}/{len(dataset)} SCPNet POSS predictions missing under "
            f"{scpnet_dir}. Run the SCPNet zero-shot dump first, e.g.:\n"
            f"  git clone --depth 1 https://github.com/SCPNet/Codes-for-SCPNet "
            f"external/Codes-for-SCPNet\n"
            f"  python {SCPNET_RUNNER} \\\n"
            f"    --scpnet-repo external/Codes-for-SCPNet \\\n"
            f"    --checkpoint {SCPNET_CKPT} \\\n"
            f"    --config {SCPNET_CFG} \\\n"
            f"    --data_root <poss-root-with-velodyne-and-materialised-voxels> \\\n"
            f"    --output_dir {scpnet_dir} --sequences {','.join(seqs)} --gpu {gpu}\n"
            "(SCPNet prepare_input voxelizer is grid-identical; it enumerates "
            "voxels/*.bin and reads velodyne/<frame>.bin — both must live under "
            "one sequences/<seq> root.)"
        )
    logger.info("Stage 1 OK: %d SCPNet POSS predictions present.", len(dataset))


def run_s2d2_and_writeback(dataset: SemanticPossSSCDataset, scpnet_dir: Path,
                           checkpoint: str, pred_out_dir: Path, n_steps: int,
                           gpu: int) -> None:
    """Stages 2-3: S²D² correction + SemanticKITTI->POSS projection + .label writeback."""
    from gssc.diffusion.multinomial import MultinomialDiffusion3DV2
    from gssc.models.s2d2_unet import SceneCompletionUNetSparse
    from gssc.utils.checkpoint import assert_bound, config_value, load_checkpoint_config

    device = torch.device(f"cuda:{gpu}")

    # EMA deployment weights, mirroring scripts/eval_kitti360.py run_s2d2_and_writeback:
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
        from safetensors.torch import load_file
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
            for name, param in model.named_parameters():
                if name in ckpt["ema_shadow"]:
                    param.data.copy_(ckpt["ema_shadow"][name])
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

    for f in dataset.frames:
        seq, frame_id = f["sequence"], f["frame_id"]
        scpnet_pred = np.load(scpnet_dir / seq / f"{frame_id}_pred.npy")  # [256,256,32] 0-19

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
        pred_sk = pred.cpu().numpy()[0]  # [256,256,32] SemanticKITTI train 0-19

        # Project SemanticKITTI -> POSS train space, then to POSS original 0-255.
        pred_poss_train = remap_scpnet_to_poss(pred_sk)
        pred_original = SEMANTICPOSS_LEARNING_MAP_INV[pred_poss_train.astype(np.int64)]

        out_seq_dir = pred_out_dir / "sequences" / seq / "predictions"
        out_seq_dir.mkdir(parents=True, exist_ok=True)
        pred_original.reshape(-1).astype(np.uint16).tofile(out_seq_dir / f"{frame_id}.label")

    logger.info("Stages 2-3 OK: wrote %d POSS .label predictions.", len(dataset))


def score(gt_root: Path, pred_root: Path, datacfg: Path, gpu: int) -> int:
    """Stage 4: official scorer with the POSS datacfg (cIoU + 11-class mIoU)."""
    eval_script = REPO_ROOT / "external" / "semantic_kitti_api" / "evaluate_completion.py"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable, str(eval_script),
        "--dataset", str(gt_root),
        "--predictions", str(pred_root),
        "--split", "valid",      # POSS datacfg lists seq 02 under 'valid'
        "--datacfg", str(datacfg),
    ]
    logger.info("Stage 4 (score): %s", " ".join(cmd))
    return subprocess.run(cmd, env=env, cwd=str(eval_script.parent)).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="eval/semanticposs_seq02")
    ap.add_argument("--checkpoint", required=True,
                    help="S²D² checkpoint, e.g. data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors")
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    ap.add_argument("--datacfg", default=None,
                    help="POSS learning_map yaml. Defaults to the config's `datacfg`.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--gt-only", action="store_true",
                    help="Run Stage 0 (materialise POSS GT) only, then exit.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(name)s] %(message)s")

    cfg = yaml.safe_load((REPO_ROOT / "configs" / f"{args.config}.yaml").read_text())
    n_steps = args.steps if args.steps is not None else int(cfg.get("correction_steps", 1))
    data_root = Path(args.data_root)

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (data_root.parent / pp if (data_root.parent / pp).exists() else data_root / pp.name)

    poss_root = _resolve(cfg["poss_root"])
    voxel_gt_root = _resolve(cfg["voxel_gt_root"])
    scpnet_dir = _resolve(cfg.get("base_pred_dir", "data/scpnet_poss_predictions"))
    pred_root = data_root / "predictions" / "semanticposs_zeroshot"
    datacfg = Path(args.datacfg) if args.datacfg else \
        (REPO_ROOT / cfg.get("datacfg", "external/semantic_kitti_api/config/semantic-poss.yaml"))

    ds_cfg = SemanticPossEvalConfig(
        poss_root=str(poss_root),
        voxel_gt_root=str(voxel_gt_root),
        split=cfg.get("split", "val"),
        sequences=tuple(str(cfg["sequences"]).split(",")) if cfg.get("sequences") else (),
    )
    dataset = SemanticPossSSCDataset(ds_cfg)
    logger.info("SemanticPOSS %s: %d frames (%s)", ds_cfg.split, len(dataset), dataset.sequences)
    logger.info("Scorable POSS classes: %s", SCORABLE_POSS_CLASSES)
    logger.info("Structural-zero POSS classes (model can't emit): %s",
                STRUCTURAL_ZERO_POSS_CLASSES)

    materialise_gt(dataset)
    if args.gt_only:
        logger.info("--gt-only: stopping after Stage 0.")
        return

    run_scpnet_zeroshot(dataset, scpnet_dir, voxel_gt_root, args.gpu)
    run_s2d2_and_writeback(dataset, scpnet_dir, args.checkpoint, pred_root, n_steps, args.gpu)
    rc = score(voxel_gt_root, pred_root, datacfg, args.gpu)
    sys.exit(rc)


if __name__ == "__main__":
    main()

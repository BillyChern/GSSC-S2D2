"""BEV second-task evaluator (S2D2 applied to LiDAR-only BEV refinement).

Runs the S2D2 correction sampler (a specialisation of Cold Diffusion's non-noise correction-sampling procedure to our linear simplex interpolant) on a 2D BEV diffusion checkpoint and
reports BEV mIoU on SemanticKITTI val seq 08. This is the LiDAR-only BEV
refinement pipeline that produces the 36.09 % number in paper Sec. 4
"Secondary application: BEV semantic segmentation".

The pipeline is:

  base BEV (from SCPNet 3D pred, top-most non-empty class projection)
        |
        | x_src
        v
  S2D2 (2D denoising UNet + sparse 3D LiDAR encoder, MultinomialDiffusion2D)
        |
        | x_hat_0
        v
  refined BEV --> argmax --> BEV mIoU on seq 08

Compared to the 3D headline path:
  - 2D denoising UNet (gssc.models.bev_unet_v2.create_modular_bev_unet)
  - 2D forward / posterior (gssc.models.bev_multinomial_diffusion_2d)
  - Same SCPNet base, same Sparse3DEncoder backbone
  - Same correction-sampling step (specialised from Cold Diffusion to our linear interpolant)
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Training-space class names (matches semantic_kitti_api learning_map_inv).
CLASS_NAMES_19 = [
    "car", "bicycle", "motorcycle", "truck", "other_vehicle",
    "person", "bicyclist", "motorcyclist", "road", "parking",
    "sidewalk", "other_ground", "building", "fence", "vegetation",
    "trunk", "terrain", "pole", "traffic_sign",
]


def _voxel_to_bev_topmost(voxel: np.ndarray) -> np.ndarray:
    """Project a [H, W, D] training-space voxel grid to a [H, W] BEV map.

    For each (x, y) column, take the highest-z non-empty class. This matches
    the headline base-derived BEV used at training time.
    """
    H, W, D = voxel.shape
    bev = np.zeros((H, W), dtype=np.int64)
    for z in range(D - 1, -1, -1):
        layer = voxel[..., z]
        mask = (layer > 0) & (bev == 0)
        bev[mask] = layer[mask]
    return bev


def _build_label_to_train_lut() -> np.ndarray:
    """Build a 256-entry LUT mapping raw SemanticKITTI labels to 0-19 training space.

    Uses ``LEARNING_MAP_INV`` (training-space -> raw label) from
    ``generate_predictions`` to derive the inverse without a Python dict
    or :func:`numpy.vectorize` (avoids per-pixel Python overhead and lets
    the remap stay vectorised).
    """
    from gssc.inference.generate_predictions import LEARNING_MAP_INV
    lut = np.zeros(256, dtype=np.int64)
    for train_idx, raw_label in enumerate(LEARNING_MAP_INV):
        lut[int(raw_label)] = train_idx
    return lut


def _assert_bound(name: str, result: object, ckpt_path: object) -> None:
    """Refuse to score a module whose weights did not fully bind.

    ``load_state_dict(strict=False)`` is required here (EMA shadows may omit
    non-float buffers) but it also silently tolerates an architecture that does not
    match the checkpoint. That is how this path came to reconstruct the denoiser at
    ``input_resolution=64`` / ``cond_channels=128`` against a 256/64 checkpoint and
    still return a number: 48 attention tensors stayed at initialisation.

    Args:
        name: Module label used in the error message.
        result: The ``_IncompatibleKeys`` returned by ``load_state_dict``.
        ckpt_path: Checkpoint being loaded, echoed so the message is actionable.

    Raises:
        RuntimeError: If any key is missing or unexpected.
    """
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if not missing and not unexpected:
        logger.info("%s: all weights bound", name)
        return
    raise RuntimeError(
        f"{name}: {len(missing)} missing and {len(unexpected)} unexpected key(s) when "
        f"loading {ckpt_path}. The architecture does not match the checkpoint, so any "
        f"score from it would be meaningless. Check the reconstruction keys in the "
        f"checkpoint's 'config' (input_resolution, model_size, conditioning_type, "
        f"use_self_conditioning, lidar_channels). "
        f"First missing: {missing[:3]}; first unexpected: {unexpected[:3]}"
    )


def evaluate_bev(
    checkpoint: str,
    data_root: str,
    n_steps: int = 1,
    sequence: str = "08",
    output: str | None = None,
    gpu: str = "0",
    max_frames: int | None = None,
) -> dict[str, float]:
    """Run BEV S2D2 evaluation end-to-end.

    Args:
        checkpoint: Path to a BEV diffusion checkpoint (``.pt`` from
            ``train_bev_secondary.py``).
        data_root: Visitor data root containing ``SemanticKITTI/`` and
            ``scpnet_predictions/``.
        n_steps: S2D2 correction-sampling steps (1 = single forward pass).
        sequence: SemanticKITTI sequence id (default ``"08"`` = val).
        output: Optional path to dump per-class IoUs as JSON.
        gpu: CUDA device id.
        max_frames: Optional cap for quick smoke runs.

    Returns:
        Dict with ``"mIoU"`` (mean over 19 valid classes) and
        ``"IoU_<class>"`` for each class.

    Raises:
        FileNotFoundError: missing checkpoint or data root.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    import torch

    from gssc.models.bev_lidar_encoder import SparseLiDAREncoder
    from gssc.models.bev_multinomial_diffusion_2d import MultinomialDiffusion2D
    from gssc.models.bev_unet_v2 import create_modular_bev_unet

    ckpt_path = Path(checkpoint).resolve()
    data_root_path = Path(data_root).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"BEV checkpoint not found: {ckpt_path}")
    if not data_root_path.exists():
        raise FileNotFoundError(f"Data root not found: {data_root_path}")

    voxels_dir = data_root_path / "SemanticKITTI" / "sequences" / sequence / "voxels"
    scpnet_dir = data_root_path / "scpnet_predictions" / sequence
    if not voxels_dir.exists():
        raise FileNotFoundError(f"SemanticKITTI voxels dir missing: {voxels_dir}")
    if not scpnet_dir.exists():
        raise FileNotFoundError(f"SCPNet predictions dir missing: {scpnet_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("BEV eval: checkpoint=%s sequence=%s n_steps=%d device=%s",
                ckpt_path, sequence, n_steps, device.type)

    # Load checkpoint and reconstruct the BEV S2D2 model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    num_classes = int(cfg.get("num_classes", 20))
    num_timesteps = int(cfg.get("num_timesteps", 100))
    base_channels = int(cfg.get("base_channels", 128))
    # The shipped BEV run uses cond_channels=64 and input_resolution=256. Both were
    # previously left at their defaults here (128 from this function, 64 from the UNet
    # factory), and load_state_dict(strict=False) accepted the result SILENTLY: 12
    # cond_proj tensors shape-mismatched and 48 attention tensors stayed at init, so the
    # eval scored a half-initialised model. Read the reconstruction spec from the
    # checkpoint, and keep the old values only as fallbacks for older files.
    lidar_channels = int(cfg.get("lidar_channels", 128))
    input_resolution = int(cfg.get("input_resolution", 64))
    model_size = str(cfg.get("model_size", "base"))
    conditioning_type = str(cfg.get("conditioning_type", "sum"))
    use_self_conditioning = bool(cfg.get("use_self_conditioning", True))

    # Build the 2D diffusion process (used for the schedule constants the
    # checkpoint was trained against — even though the headline 1-step
    # path consumes only the denoiser argmax, future N>1 sampling needs it).
    _ = MultinomialDiffusion2D(
        num_classes=num_classes, num_timesteps=num_timesteps,
    ).to(device)
    # Note: create_modular_bev_unet's body channel count is fixed by
    # ``model_size``; ``base_channels`` from the checkpoint config is
    # informational and only checked for consistency.
    del base_channels  # silence "unused"; model_size carries the body width
    denoiser = create_modular_bev_unet(
        num_classes=num_classes,
        input_resolution=input_resolution,
        conditioning_type=conditioning_type,
        use_self_conditioning=use_self_conditioning,
        model_size=model_size,
        cond_channels=lidar_channels,
    ).to(device)
    lidar_encoder = SparseLiDAREncoder(
        in_channels=1, base_channels=32, out_channels=lidar_channels,
    ).to(device)

    # Load weights (EMA preferred when present).
    #
    # strict=False is kept because some checkpoints carry EMA shadows that omit
    # non-float buffers, but a SILENT partial load is exactly how this path shipped a
    # half-initialised denoiser. Report what did not bind, and refuse to score on it:
    # a wrong number that looks right is worse than a crash.
    if "denoiser_ema" in ckpt:
        den_res = denoiser.load_state_dict(ckpt["denoiser_ema"], strict=False)
        enc_res = lidar_encoder.load_state_dict(ckpt["lidar_encoder_ema"], strict=False)
        logger.info("Loaded EMA weights")
    else:
        den_res = denoiser.load_state_dict(ckpt["denoiser_state_dict"], strict=False)
        enc_res = lidar_encoder.load_state_dict(ckpt["lidar_encoder_state_dict"], strict=False)
        logger.info("Loaded training weights")
    _assert_bound("denoiser", den_res, ckpt_path)
    _assert_bound("lidar_encoder", enc_res, ckpt_path)
    denoiser.train(False)
    lidar_encoder.train(False)

    # Frame iteration
    frame_files = sorted(voxels_dir.glob("*.bin"))
    if max_frames is not None:
        frame_files = frame_files[:max_frames]
    logger.info("Scoring %d frames from sequence %s", len(frame_files), sequence)

    label_to_train = _build_label_to_train_lut()
    iou_accum: defaultdict[int, dict[str, int]] = defaultdict(
        lambda: {"intersection": 0, "union": 0}
    )

    for fpath in frame_files:
        frame_id = fpath.stem
        # Load LiDAR voxels (binary occupancy from .bin)
        compressed = np.fromfile(fpath, dtype=np.uint8)
        binary = np.unpackbits(compressed).reshape(256, 256, 32).astype(np.float32)
        lidar = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W,D]

        # Load SCPNet 3D prediction -> derive base BEV (top-most non-empty)
        scp = np.load(scpnet_dir / f"{frame_id}_pred.npy")
        bev_base = _voxel_to_bev_topmost(scp)
        x_src = torch.from_numpy(bev_base).long().unsqueeze(0).to(device)  # [1,H,W]

        # Load GT BEV by projecting GT 3D voxels (training space)
        gt_label_path = (data_root_path / "SemanticKITTI" / "sequences" / sequence
                         / "voxels" / f"{frame_id}.label")
        if not gt_label_path.exists():
            continue
        gt_voxel_raw = np.fromfile(gt_label_path, dtype=np.uint16).reshape(256, 256, 32)
        # Vectorised remap raw 0-255 -> training 0-19 via LUT
        gt_voxel = label_to_train[gt_voxel_raw.clip(0, 255).astype(np.int64)]
        gt_bev = _voxel_to_bev_topmost(gt_voxel)

        # S2D2 correction sampling on BEV
        with torch.no_grad():
            cond = lidar_encoder(lidar)  # [1, C, H, W]
            x_t = x_src.clone()
            timesteps = torch.linspace(
                num_timesteps - 1, 0, n_steps + 1, dtype=torch.long, device=device
            )[:-1]
            for t in timesteps:
                t_b = t.unsqueeze(0)
                logits = denoiser(x_t, t_b, cond)
                if isinstance(logits, tuple):
                    logits = logits[0]
                x_hat = logits.argmax(dim=1)
                x_t = x_hat
            pred_bev = x_t.squeeze(0).cpu().numpy().astype(np.int64)

        # Accumulate per-class IoU
        for c in range(20):
            valid_gt = (gt_bev == c)
            tp = int(((pred_bev == c) & valid_gt).sum())
            fp = int(((pred_bev == c) & (gt_bev != c) & (gt_bev != 0)).sum())
            fn = int(((pred_bev != c) & valid_gt).sum())
            iou_accum[c]["intersection"] += tp
            iou_accum[c]["union"] += tp + fp + fn

    # Aggregate
    metrics: dict[str, float] = {}
    iou_values = []
    for c in range(1, 20):
        denom = iou_accum[c]["union"]
        iou = (iou_accum[c]["intersection"] / denom * 100.0) if denom > 0 else 0.0
        metrics[f"IoU_{CLASS_NAMES_19[c - 1]}"] = round(iou, 2)
        iou_values.append(iou)
    metrics["mIoU"] = round(float(np.mean(iou_values)), 2)
    metrics["num_frames"] = float(len(frame_files))

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        logger.info("Wrote BEV per-class metrics to %s", out_path)

    logger.info("BEV mIoU: %.2f %% over 19 valid classes (seq %s, %d frames)",
                metrics["mIoU"], sequence, int(metrics["num_frames"]))
    return metrics

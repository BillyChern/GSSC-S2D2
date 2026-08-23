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
from pathlib import Path
from typing import Any

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
        checkpoint: Path to a BEV diffusion checkpoint -- either the trainer's
            ``.pt`` (``train_bev_secondary.py``) or the released
            ``.safetensors`` beside its ``config.json``.
        data_root: Visitor data root containing ``SemanticKITTI/`` and
            ``scpnet_predictions/``.
        n_steps: S2D2 correction-sampling steps (1 = single forward pass).
        sequence: SemanticKITTI sequence id (default ``"08"`` = val).
        output: Optional path to dump per-class IoUs as JSON.
        gpu: CUDA device id.
        max_frames: Optional cap for quick smoke runs. Takes the FIRST n frames in sorted
            order. This is NOT the protocol the published BEV numbers were measured under --
            see the note below -- so a ``max_frames=100`` result does not reproduce them.

    Returns:
        Dict with ``"mIoU"`` (mean over 19 valid classes) and
        ``"IoU_<class>"`` for each class.

    Raises:
        FileNotFoundError: missing checkpoint or data root.

    Note:
        THE PUBLISHED BEV NUMBERS COME FROM A DIFFERENT FRAME SET THAN THIS FUNCTION SCORES.
        ``train_bev_secondary.run_algo2_on_samples`` evaluates
        ``RandomState(42).choice(len(val_dataset), 100, replace=False)`` -- a seeded random
        sample of 100 frames, not the first 100.

        Adding a ``--sample-seed`` flag here would NOT be enough to reproduce them, because
        the two index spaces differ in three ways and the seed indexes a list, not a set of
        frame ids:

        1. root   -- the dataset reads ``SemanticKITTI_3D/256/<seq>/``, this reads the voxel
                     directory passed in;
        2. glob   -- the dataset enumerates ``*_bev.npy``, this enumerates ``*.bin``;
        3. filter -- the dataset DROPS any frame whose ``_voxels.npy`` or ``_bev_top.npy`` is
                     missing, so its indices are a filtered subsequence.

        Seeding this list would therefore select a different 100 frames and return a
        plausible number that reproduces nothing. Reproducing the published figure requires
        rebuilding the dataset's sample list exactly, and a previous reimplementation of this
        protocol scored 33.61% where the training log recorded 34.75% -- a 1.1 pp gap that is
        itself evidence of how easily this diverges. Until that is done and checked against
        the log, the honest statement is the one in the supplement: those figures are scored
        on 100 seeded val frames by the training-time evaluator.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    import torch

    from gssc.models.bev_lidar_encoder import SparseLiDAREncoder
    from gssc.models.bev_multinomial_diffusion_2d import MultinomialDiffusion2D
    from gssc.models.bev_unet_v2 import create_modular_bev_unet

    # The published 36.09 % was produced by this metric object, so the evaluator uses
    # the same one rather than a second IoU implementation. The previous local version
    # scored `fp = (pred==c) & (gt!=c) & (gt!=0)`, which discards every false positive
    # landing on a GT-empty cell and is therefore strictly more lenient.
    from gssc.training.train_bev_secondary import BEVMetrics
    from gssc.utils.checkpoint import assert_bound, config_value, load_checkpoint_config

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

    # Load checkpoint and reconstruct the BEV S2D2 model. Two shipped layouts:
    #   * the trainer's ``.pt``  -- EMA sub-dicts plus an in-file ``config``;
    #   * the asset bundle's ``.safetensors`` -- keys flattened to
    #     ``"<state_dict_key>.<param>"``, architecture in the sibling ``config.json``.
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        ckpt: dict[str, Any] = {}
        for flat_key, tensor in load_file(str(ckpt_path)).items():
            block, _, param = flat_key.partition(".")
            ckpt.setdefault(block, {})[param] = tensor
        logger.info("safetensors layout: %d block(s) %s", len(ckpt), sorted(ckpt))
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = load_checkpoint_config(ckpt_path, ckpt if isinstance(ckpt, dict) else None)
    num_classes = config_value(cfg, "num_classes", 20)
    num_timesteps = config_value(cfg, "num_timesteps", 100)
    base_channels = config_value(cfg, "base_channels", 128)
    # The shipped BEV run uses cond_channels=64 and input_resolution=256. Both were
    # previously left at their defaults here (128 from this function, 64 from the UNet
    # factory), and load_state_dict(strict=False) accepted the result SILENTLY: 12
    # cond_proj tensors shape-mismatched and 48 attention tensors stayed at init, so the
    # eval scored a half-initialised model. Read the reconstruction spec from the
    # checkpoint, and keep the old values only as fallbacks for older files.
    lidar_channels = config_value(cfg, "lidar_channels", 128)
    input_resolution = config_value(cfg, "input_resolution", 64)
    model_size = config_value(cfg, "model_size", "base")
    conditioning_type = config_value(cfg, "conditioning_type", "sum")
    use_self_conditioning = config_value(cfg, "use_self_conditioning", True)

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
    assert_bound("denoiser", den_res, ckpt_path)
    assert_bound("lidar_encoder", enc_res, ckpt_path)
    denoiser.train(False)
    lidar_encoder.train(False)

    # Frame iteration
    frame_files = sorted(voxels_dir.glob("*.bin"))
    if max_frames is not None:
        frame_files = frame_files[:max_frames]
    logger.info("Scoring %d frames from sequence %s", len(frame_files), sequence)

    label_to_train = _build_label_to_train_lut()
    bev_metrics = BEVMetrics(num_classes=num_classes)

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

        # Accumulate into the training-time confusion matrix (no invalid mask, exactly
        # as run_algo2_on_samples does), so this evaluator and the published number
        # share one IoU definition.
        bev_metrics.update(pred_bev, gt_bev)

    # Aggregate
    iou = bev_metrics.get_iou()
    metrics: dict[str, float] = {}
    for c, value in sorted(iou["class_iou"].items()):
        name = CLASS_NAMES_19[c - 1] if c - 1 < len(CLASS_NAMES_19) else f"class_{c}"
        metrics[f"IoU_{name}"] = round(float(value) * 100.0, 2)
    metrics["mIoU"] = round(float(iou["mIoU"]) * 100.0, 2)
    metrics["num_frames"] = float(len(frame_files))

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2))
        logger.info("Wrote BEV per-class metrics to %s", out_path)

    logger.info("BEV mIoU: %.2f %% over %d valid classes (seq %s, %d frames)",
                metrics["mIoU"], num_classes - 1, sequence, int(metrics["num_frames"]))
    logger.info(
        "Protocol: every frame of sequence %s, IoU by gssc.training.train_bev_secondary."
        "BEVMetrics (no invalid mask). The published BEV figures (34.75 base -> 36.09 "
        "refined) were scored by the SAME metric object but on 100 seeded val samples "
        "(RandomState(42)) inside the trainer, so this frame set is not that one -- see "
        "this function's docstring.", sequence)
    return metrics

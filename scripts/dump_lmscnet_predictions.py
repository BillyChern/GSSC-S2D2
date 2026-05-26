"""Dump LMSCNet SSC predictions as per-frame x_src .npy files.

Reproduces a third base for the paper Tab. III cross-base demonstration
(LMSCNet+S2D2 alongside SCPNet+S2D2 and JS3C-Net+S2D2). Uses the publicly
released multiscale LMSCNet checkpoint (Roldao et al., 2020) downloaded
from the Google Drive folder linked in the official LMSCNet README.

GT data invariant
-----------------
This script only reads SemanticKITTI .bin voxel-occupancy files (sparse
LiDAR input). It does NOT read .label files. Predictions are pure
forward-pass outputs of the pretrained LMSCNet model -- no ground-truth
leakage into x_src.

Output format
-------------
data/lmscnet_predictions/{seq}/{frame_id}_pred.npy containing a
(256, 256, 32) int64 array of per-voxel class indices in [0, 19].
Voxel-grid axis ordering matches SemanticKITTI (X, Y, Z) = (256, 256, 32).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

LMSCNET_REPO = Path("/workspace/reference/LMSCNet").resolve()
sys.path.insert(0, str(LMSCNET_REPO))

import LMSCNet.data.io_data as SemKittiIO
from LMSCNet.models.LMSCNet import LMSCNet

logger = logging.getLogger("dump_lmscnet")

EXPECTED_SHAPE = (256, 256, 32)
NUM_CLASSES = 20
GRID_DIMS_WHD = (256, 32, 256)

LMSCNET_CLASS_FREQUENCIES = np.array([
    5.41773033e+09, 1.57835390e+07, 1.25136000e+05, 1.18809000e+05,
    6.46799000e+05, 8.21951000e+05, 2.62978000e+05, 2.83696000e+05,
    2.04750000e+05, 6.16887030e+07, 4.50296100e+06, 4.48836500e+07,
    2.26992300e+06, 5.68402180e+07, 1.57196520e+07, 1.58442623e+08,
    2.06162300e+06, 3.69705220e+07, 1.15198800e+06, 3.34146000e+05,
])


def load_lmscnet_model(weights_path, device):
    model = LMSCNet(
        class_num=NUM_CLASSES,
        input_dimensions=list(GRID_DIMS_WHD),
        class_frequencies=LMSCNET_CLASS_FREQUENCIES,
    )
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cleaned = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning("Missing keys: %s", missing[:5])
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected[:5])
    model.to(device)
    model.train(False)  # inference mode (set training=False)
    return model


def read_occupancy(bin_path):
    occ = SemKittiIO._read_occupancy_SemKITTI(str(bin_path))
    W, H, D = GRID_DIMS_WHD
    occ = occ.reshape([W, D, H])
    occ = np.moveaxis(occ, [0, 1, 2], [0, 2, 1])
    return occ


@torch.no_grad()
def infer_one(model, bin_path, device):
    occ = read_occupancy(bin_path)
    x = torch.from_numpy(occ).float()[None, None].to(device)
    scores = model({"3D_OCCUPANCY": x})
    logits = scores["pred_semantic_1_1"]
    pred_whd = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    pred_xyz = np.transpose(pred_whd, (0, 2, 1))
    assert pred_xyz.shape == EXPECTED_SHAPE, f"Bad shape {pred_xyz.shape}"
    assert 0 <= pred_xyz.min() and pred_xyz.max() < NUM_CLASSES, (
        f"Bad label range [{pred_xyz.min()}, {pred_xyz.max()}]"
    )
    return pred_xyz


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="/workspace/reference/LMSCNet/pretrained_models/LMSCNet.pth")
    p.add_argument("--semantickitti_root", default="/workspace/GSSC-S2D2/data/SemanticKITTI")
    p.add_argument("--output_dir", default="/workspace/GSSC-S2D2/data/lmscnet_predictions")
    p.add_argument("--sequences", nargs="+", default=["08"])
    p.add_argument("--gpu", default="0")
    p.add_argument("--smoke_test", action="store_true")
    return p.parse_args()


def expand_sequences(seq_args):
    out = []
    for s in seq_args:
        if s == "val":
            out.extend(["08"])
        elif s == "train":
            out.extend(["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"])
        elif s == "trainval":
            out.extend(["00", "01", "02", "03", "04", "05", "06", "07", "09", "10", "08"])
        else:
            out.append(s.zfill(2))
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = load_lmscnet_model(args.weights, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Loaded LMSCNet from %s (%.3f M params)", args.weights, n_params)

    seqs = expand_sequences(args.sequences)
    logger.info("Sequences: %s", seqs)

    sk_root = Path(args.semantickitti_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    total, skipped, written = 0, 0, 0
    t0 = time.time()
    for seq in seqs:
        seq_in = sk_root / "sequences" / seq / "voxels"
        if not seq_in.exists():
            logger.error("Missing input dir: %s", seq_in)
            continue
        bins = sorted(seq_in.glob("*.bin"))
        seq_out = out_root / seq
        seq_out.mkdir(parents=True, exist_ok=True)
        logger.info("[%s] %d frames", seq, len(bins))
        for bin_path in bins:
            total += 1
            frame_id = bin_path.stem
            out_path = seq_out / f"{frame_id}_pred.npy"
            if out_path.exists():
                skipped += 1
                continue
            pred = infer_one(model, bin_path, device)
            np.save(out_path, pred)
            written += 1
            if args.smoke_test:
                unique, counts = np.unique(pred, return_counts=True)
                logger.info("Smoke test on %s/%s:", seq, frame_id)
                logger.info("  output shape: %s", pred.shape)
                logger.info("  dtype: %s", pred.dtype)
                logger.info("  unique classes: %s", dict(zip(unique.tolist(), counts.tolist())))
                logger.info("  total non-empty: %d (%.2f%%)", int((pred != 0).sum()), 100 * (pred != 0).mean())
                return
            if written % 200 == 0:
                elapsed = time.time() - t0
                rate = written / max(elapsed, 1e-6)
                logger.info("  ... %d frames written (%.2f fps)", written, rate)
    elapsed = time.time() - t0
    logger.info("Done. total=%d, written=%d, skipped=%d, elapsed=%.1f s", total, written, skipped, elapsed)


if __name__ == "__main__":
    main()

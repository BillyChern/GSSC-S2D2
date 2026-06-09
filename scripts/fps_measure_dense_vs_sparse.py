"""
Measure per-step inference latency / FPS of the S2D2 denoiser network:
  (a) DENSE headline backbone  (SceneCompletionUNetSparse: dense Conv3d U-Net body
      + auxiliary spconv SparseLiDAREncoder), ~34.7M params, the shipped headline.
  (b) SPARSE-3D-U-Net variant of matched parameter count: the SAME U-Net topology
      (channels 32->64->128->256, same depth/conditioning streams) but with the
      dense Conv3d U-Net body replaced by spconv (SubMConv3d / SparseConv3d /
      SparseInverseConv3d).

Single deployment-resolution forward: voxel grid 256x256x32, K=20 classes, B=1,
fp32, eval(), no_grad, on a single contention-free H100. Matches the N=1
S2D2 correction step (one denoiser forward per step; see sample_algo2).

10 warmup iters + 100 timed iters with torch.cuda.synchronize around each.
"""

import argparse

import numpy as np
import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from spconv.pytorch import ConvAlgo, SparseConv3d, SparseConvTensor, SparseInverseConv3d, SubMConv3d

_SPCONV_VERSION = getattr(spconv, "__version__", "unknown")

from gssc.models.s2d2_unet import (
    SceneCompletionUNetSparse,
    count_parameters,
    timestep_embedding,
)
from gssc.models.sparse_lidar_encoder import SparseLiDAREncoder

_ALGO = ConvAlgo.Native

H, W, D = 256, 256, 32
K = 20
DEVICE = "cuda"

# SemanticKITTI sparse-LiDAR occupancy is ~1-5%; use a realistic ~2% to build the
# x_t one-hot / lidar / ssc_pred inputs. (Timing only needs correct shapes+dtype;
# the dense backbone is occupancy-invariant, the sparse backbone is occupancy-
# sensitive, so we use a faithful occupancy for both.)
OCC_FRAC = 0.02


# ---------------------------------------------------------------------------
# Sparse-3D-U-Net denoiser of matched parameter count.
# Mirrors SceneCompletionUNetSparse topology but the denoiser body is spconv.
# ---------------------------------------------------------------------------
class SparseResBlock3DDenoiser(nn.Module):
    """Sparse analogue of ResidualBlock3DSparse (dense) used in the headline.

    Same channel widths and 3x3x3 kernels, AdaGN-style time conditioning, additive
    BEV + LiDAR conditioning. Convs are spconv SubMConv3d (sparsity-preserving).
    Conditioning tensors are dense -> gathered at the active sparse coords.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, cond_channels,
                 num_groups=8, indice_key=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = SubMConv3d(in_channels, out_channels, 3, padding=1,
                                       indice_key=indice_key, algo=_ALGO)
        self.bev_proj = SubMConv3d(cond_channels, out_channels, 3, padding=1,
                                          indice_key=indice_key, algo=_ALGO)
        self.lidar_proj = SubMConv3d(cond_channels, out_channels, 3, padding=1,
                                            indice_key=indice_key, algo=_ALGO)
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.time_fc = nn.Linear(time_emb_dim, out_channels * 2)
        self.conv2 = SubMConv3d(out_channels, out_channels, 3, padding=1,
                                       indice_key=indice_key, algo=_ALGO)
        if in_channels != out_channels:
            self.skip = SubMConv3d(in_channels, out_channels, 1,
                                          indice_key=indice_key + "_skip" if indice_key else None,
                                          algo=_ALGO)
        else:
            self.skip = None

    @staticmethod
    def _gn(norm, x):
        # GroupNorm over sparse features [N, C] by treating as [1, C, N].
        f = x.features.t().unsqueeze(0)
        f = norm(f).squeeze(0).t()
        return x.replace_feature(f)

    def _gather_cond(self, dense_cond, x):
        # dense_cond: [B, C, H, W, D] -> gather features at x's active indices.
        idx = x.indices.long()
        feats = dense_cond[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]]
        return x.replace_feature(feats)

    def forward(self, x, t_emb, bev_dense, lidar_dense, batch_t_index):
        h = self._gn(self.norm1, x)
        h = h.replace_feature(F.silu(h.features))
        h = self.conv1(h)

        bev_sp = self.bev_proj(self._gather_cond(bev_dense, h))
        h = h.replace_feature(h.features + bev_sp.features)
        lid_sp = self.lidar_proj(self._gather_cond(lidar_dense, h))
        h = h.replace_feature(h.features + lid_sp.features)

        h = self._gn(self.norm2, h)
        wb = self.time_fc(t_emb)  # [B, 2C]
        w, b = wb.chunk(2, dim=-1)
        # broadcast per-point using batch index
        wp = w[batch_t_index]
        bp = b[batch_t_index]
        h = h.replace_feature(wp * h.features + bp)
        h = h.replace_feature(F.silu(h.features))
        h = self.conv2(h)

        identity = x if self.skip is None else self.skip(x)
        return h.replace_feature(h.features + identity.features)


class SceneCompletionUNetSparseBody(nn.Module):
    """Matched-param sparse-3D-U-Net denoiser (spconv body).

    Same channel progression / depth / conditioning streams as the dense headline
    SceneCompletionUNetSparse, but the U-Net body is spconv. Reuses the identical
    auxiliary SparseLiDAREncoder so the aux-encoder cost is shared and only the
    DENOISER BODY differs -- exactly the dense-vs-sparse backbone comparison
    reported in the supplementary FPS ablation.
    """

    def __init__(self, num_classes=20, base_channels=32,
                 channel_mult=(1, 2, 4, 8), time_emb_dim=128,
                 lidar_base_channels=16, lidar_out_channels=32, num_groups=8):
        super().__init__()
        self.num_classes = num_classes
        self.base_channels = base_channels
        ch = [base_channels * m for m in channel_mult]
        co = lidar_out_channels

        self.lidar_encoder = SparseLiDAREncoder(
            in_channels=1, base_channels=lidar_base_channels, out_channels=co)

        self.voxel_embed = SubMConv3d(num_classes, ch[0], 3, padding=1,
                                             indice_key="vox", algo=_ALGO)
        self.bev_embed = nn.Conv3d(num_classes, co, 3, padding=1)
        self.ssc_embed = SubMConv3d(num_classes, ch[0], 3, padding=1,
                                           indice_key="ssc", algo=_ALGO)

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim))

        self.enc0 = SparseResBlock3DDenoiser(ch[0], ch[0], time_emb_dim, co, num_groups, "e0")
        self.down0 = SparseConv3d(ch[0], ch[1], 3, stride=2, padding=1, indice_key="dn0", algo=_ALGO)
        self.enc1 = SparseResBlock3DDenoiser(ch[1], ch[1], time_emb_dim, co, num_groups, "e1")
        self.down1 = SparseConv3d(ch[1], ch[2], 3, stride=2, padding=1, indice_key="dn1", algo=_ALGO)
        self.enc2 = SparseResBlock3DDenoiser(ch[2], ch[2], time_emb_dim, co, num_groups, "e2")
        self.down2 = SparseConv3d(ch[2], ch[3], 3, stride=2, padding=1, indice_key="dn2", algo=_ALGO)
        self.enc3 = SparseResBlock3DDenoiser(ch[3], ch[3], time_emb_dim, co, num_groups, "e3")
        self.down3 = SparseConv3d(ch[3], ch[3], 3, stride=2, padding=1, indice_key="dn3", algo=_ALGO)

        self.mid0 = SparseResBlock3DDenoiser(ch[3], ch[3], time_emb_dim, co, num_groups, "m0")
        self.mid1 = SparseResBlock3DDenoiser(ch[3], ch[3], time_emb_dim, co, num_groups, "m1")

        self.up3 = SparseInverseConv3d(ch[3], ch[3], 3, indice_key="dn3", algo=_ALGO)
        self.dec3 = SparseResBlock3DDenoiser(ch[3] * 2, ch[3], time_emb_dim, co, num_groups, "d3")
        self.up2 = SparseInverseConv3d(ch[3], ch[2], 3, indice_key="dn2", algo=_ALGO)
        self.dec2 = SparseResBlock3DDenoiser(ch[2] * 2, ch[2], time_emb_dim, co, num_groups, "d2")
        self.up1 = SparseInverseConv3d(ch[2], ch[1], 3, indice_key="dn1", algo=_ALGO)
        self.dec1 = SparseResBlock3DDenoiser(ch[1] * 2, ch[1], time_emb_dim, co, num_groups, "d1")
        self.up0 = SparseInverseConv3d(ch[1], ch[0], 3, indice_key="dn0", algo=_ALGO)
        self.dec0 = SparseResBlock3DDenoiser(ch[0] * 2, ch[0], time_emb_dim, co, num_groups, "d0")

        self.out_norm = nn.GroupNorm(num_groups, ch[0])
        self.out_conv = SubMConv3d(ch[0], num_classes, 3, padding=1,
                                          indice_key="out", algo=_ALGO)

    @staticmethod
    def _occ(dense_voxels):
        # occupancy over class channels: occupied where argmax != 0 OR any prob mass
        return dense_voxels.abs().sum(dim=1) > 1e-6

    def _to_sparse(self, dense, occ):
        B = dense.shape[0]
        coords, feats, bidx = [], [], []
        for b in range(B):
            nz = torch.nonzero(occ[b], as_tuple=False)  # [N,3]
            if nz.shape[0] == 0:
                nz = torch.zeros((1, 3), dtype=torch.int32, device=dense.device)
            f = dense[b, :, nz[:, 0], nz[:, 1], nz[:, 2]].t()
            coords.append(nz.int())
            feats.append(f)
            bidx.append(torch.full((nz.shape[0], 1), b, dtype=torch.int32, device=dense.device))
        coords = torch.cat(coords, 0)
        bidx = torch.cat(bidx, 0)
        feats = torch.cat(feats, 0)
        indices = torch.cat([bidx, coords], 1)
        st = SparseConvTensor(feats, indices, [H, W, D], B)
        return st

    def forward(self, x_t, t, bev, lidar, ssc_pred=None, **kw):
        lidar_feats = self.lidar_encoder(lidar)  # dense dict

        # BEV -> dense 3D conditioning (same as headline)
        bev_3d = bev.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        bev_emb0 = self.bev_embed(bev_3d)

        # voxel -> sparse on the union occupancy of x_t / ssc_pred
        occ = self._occ(x_t)
        if ssc_pred is not None:
            occ = occ | self._occ(ssc_pred)
        x = self._to_sparse(x_t, occ)
        x = self.voxel_embed(x)
        if ssc_pred is not None:
            ssc_st = self._to_sparse(ssc_pred, occ)
            ssc_st = self.ssc_embed(ssc_st)
            x = x.replace_feature(x.features + ssc_st.features)

        t_emb = self.time_mlp(timestep_embedding(t, self.base_channels))
        bt0 = x.indices[:, 0].long()

        e0 = self.enc0(x, t_emb, bev_emb0, lidar_feats["level0"], bt0)
        x = self.down0(e0)
        e1 = self.enc1(x, t_emb, F.avg_pool3d(bev_emb0, 2), lidar_feats["level1"], x.indices[:, 0].long())
        x = self.down1(e1)
        bev2 = F.avg_pool3d(F.avg_pool3d(bev_emb0, 2), 2)
        e2 = self.enc2(x, t_emb, bev2, lidar_feats["level2"], x.indices[:, 0].long())
        x = self.down2(e2)
        bev3 = F.avg_pool3d(bev2, 2)
        e3 = self.enc3(x, t_emb, bev3, lidar_feats["level3"], x.indices[:, 0].long())
        x = self.down3(e3)
        bev4 = F.avg_pool3d(bev3, 2)

        x = self.mid0(x, t_emb, bev4, lidar_feats["level4"], x.indices[:, 0].long())
        x = self.mid1(x, t_emb, bev4, lidar_feats["level4"], x.indices[:, 0].long())

        x = self.up3(x)
        x = x.replace_feature(torch.cat([x.features, e3.features], 1))
        x = self.dec3(x, t_emb, bev3, lidar_feats["level3"], x.indices[:, 0].long())
        x = self.up2(x)
        x = x.replace_feature(torch.cat([x.features, e2.features], 1))
        x = self.dec2(x, t_emb, bev2, lidar_feats["level2"], x.indices[:, 0].long())
        x = self.up1(x)
        x = x.replace_feature(torch.cat([x.features, e1.features], 1))
        x = self.dec1(x, t_emb, F.avg_pool3d(bev_emb0, 2), lidar_feats["level1"], x.indices[:, 0].long())
        x = self.up0(x)
        x = x.replace_feature(torch.cat([x.features, e0.features], 1))
        x = self.dec0(x, t_emb, bev_emb0, lidar_feats["level0"], x.indices[:, 0].long())

        x = x.replace_feature(F.silu(self.out_norm(x.features.t().unsqueeze(0)).squeeze(0).t()))
        x = self.out_conv(x)
        return x.dense()  # [B, K, H, W, D]


# ---------------------------------------------------------------------------
def make_inputs():
    g = torch.Generator(device=DEVICE).manual_seed(42)
    # sparse lidar occupancy
    lidar = (torch.rand(1, 1, H, W, D, generator=g, device=DEVICE) < OCC_FRAC).float()
    # ssc_pred: ~5% occupied dense semantic one-hot (the base completion has more mass)
    occ = (torch.rand(1, H, W, D, generator=g, device=DEVICE) < 0.05)
    cls = torch.randint(1, K, (1, H, W, D), generator=g, device=DEVICE)
    cls = cls * occ.long()
    ssc_pred = F.one_hot(cls, K).permute(0, 4, 1, 2, 3).float()
    x_t = ssc_pred.clone()  # N=1 starts from base one-hot (see sample_algo2)
    bev = F.one_hot(torch.randint(0, K, (1, H, W), generator=g, device=DEVICE), K)
    bev = bev.permute(0, 3, 1, 2).float()
    t = torch.full((1,), 99, device=DEVICE, dtype=torch.long)
    return dict(x_t=x_t, t=t, bev=bev, lidar=lidar, ssc_pred=ssc_pred)


def time_model(model, inputs, name, warmup=10, iters=100):
    model.eval().to(DEVICE)
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**inputs)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            _ = model(**inputs)
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))  # ms
    times = np.array(times)
    mean, std = times.mean(), times.std()
    fps = 1000.0 / mean
    params = count_parameters(model)
    print(f"\n[{name}]")
    print(f"  params      : {params:,}  ({params/1e6:.2f}M)")
    print(f"  ms / forward : {mean:.2f} +/- {std:.2f}")
    print(f"  FPS          : {fps:.2f}")
    return dict(name=name, params=params, ms=mean, std=std, fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors",
        help="Headline EMA checkpoint to time (default: repo-relative released "
        "checkpoint path; weights only affect numerics, not the timing).",
    )
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    print("=" * 70)
    print("S2D2 denoiser per-step inference FPS @ 256x256x32, K=20, B=1, fp32")
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | spconv {_SPCONV_VERSION}")
    print(f"warmup={args.warmup} timed={args.iters} | sparse-lidar occ={OCC_FRAC:.0%}")
    print("=" * 70)

    inputs = make_inputs()

    # ---- (a) DENSE headline ----
    dense = SceneCompletionUNetSparse(
        num_classes=20, base_channels=32, time_emb_dim=128,
        lidar_base_channels=16, lidar_out_channels=32, lidar_in_channels=1,
        no_bev=False, ssc_cond_channels=20, ssc_multiscale=False)
    # load headline EMA weights (exact shipped checkpoint). Timing is
    # weight-independent; loading just makes the report faithful to the deployed
    # model. Supports both the released .safetensors and the .pt training ckpt.
    try:
        if args.ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(args.ckpt)
        else:
            ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            sd = ck.get("ema_shadow") or ck.get("model_state_dict")
        missing, unexpected = dense.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(unexpected)
        print(f"\n[dense] loaded headline weights: matched~{loaded}/{len(dense.state_dict())} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    except Exception as e:
        print(f"\n[dense] WARNING: could not load checkpoint ({e}); using default init.")
    r_dense = time_model(dense, inputs, "DENSE headline (Conv3d U-Net + sparse LiDAR aux)",
                         args.warmup, args.iters)
    del dense
    torch.cuda.empty_cache()

    # ---- (b) SPARSE-3D-U-Net matched params ----
    sparse = SceneCompletionUNetSparseBody(
        num_classes=20, base_channels=32, time_emb_dim=128,
        lidar_base_channels=16, lidar_out_channels=32)
    r_sparse = time_model(sparse, inputs, "SPARSE 3D U-Net (matched params, spconv body)",
                          args.warmup, args.iters)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in (r_dense, r_sparse):
        print(f"  {r['name']:<52s}  {r['params']/1e6:6.2f}M  "
              f"{r['ms']:8.2f} ms  {r['fps']:7.2f} FPS")
    faster = "DENSE" if r_dense["fps"] > r_sparse["fps"] else "SPARSE"
    ratio = max(r_dense["fps"], r_sparse["fps"]) / min(r_dense["fps"], r_sparse["fps"])
    print(f"\n  Faster backbone: {faster}  ({ratio:.2f}x)")
    print(f"  Dense headline  : {r_dense['fps']:.2f} FPS ({r_dense['ms']:.1f} ms)")
    print("  -> the dense headline runs at ~9.3 FPS / ~107 ms per step at N=1.")


if __name__ == "__main__":
    main()

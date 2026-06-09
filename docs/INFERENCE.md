# Inference recipes

## Real-time deployment (single S²D² correction step, N=1 no TTA, 9.33 FPS marginal on H100)

```bash
python scripts/eval.py eval/val_1step --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors
```

Returns 38.54% val mIoU on SemanticKITTI seq 08. The same N=1 no-TTA setting
scores **38.8 % on the hidden test** — to our knowledge the best single-frame
single-sample result on the leaderboard to date (+0.9 over TALoS 37.9).

## Multi-step S²D² correction sampling (peak quality)

```bash
python scripts/eval.py eval/step_sweep --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors
```

Returns the full sweep: 38.54 (N=1), 38.59 (N=2), 38.65 (N=4 peak), 38.16 (N=100).

## D4 TTA (test-server best, 39.2% test; N=1 no-TTA already reaches 38.8)

```bash
# Generate .label files
python scripts/infer.py infer/test_d4tta \
    --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors \
    --output predictions/test_d4tta/

# Bundle for submission
cd predictions/test_d4tta && zip -r submission.zip sequences/
```

Submit `submission.zip` to the [SemanticKITTI Codabench leaderboard](https://www.codabench.org/competitions/13814/#/results-tab).

## Custom data

If you have your own SCPNet predictions and SemanticKITTI-format frames,
point the data root at them:

```bash
python scripts/eval.py eval/val_1step \
    --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors \
    --data-root /path/to/your_data
```

## JS3C-Net cross-base (paper tab:portable_s2d2, cross-base rows)

```bash
# Verify the JS3C predictions are dumped (see docs/REPRODUCIBILITY.md)
ls data/js3cnet_predictions/08 | head

# N=1, no TTA — expect the paper headline 22.7 -> 26.1 % val mIoU (+3.3 pp)
# under the official semantic-kitti-api with GT BEV (precise eval output 26.05 %,
# the number the paper rounds to 26.1; 26.72 % under the paper's internal
# SSCMetrics footnote)
python scripts/eval.py eval/js3c_val_1step \
    --checkpoint data/checkpoints/gssc_js3c/gssc_js3c_s2d2_real/model_ema.safetensors

# N=4 + D4 TTA — expect ≥ 24.32 % (TTA monotone gain on the at-deploy derived-BEV path)
python scripts/eval.py eval/js3c_val_d4tta \
    --checkpoint data/checkpoints/gssc_js3c/gssc_js3c_s2d2_real/model_ema.safetensors
```

> **GT BEV vs. derived BEV.** `eval/js3c_val_1step` uses `bev_source: gt`
> (the paper tab:portable_s2d2 protocol), which conditions on a ground-truth BEV
> oracle: its headline value is the paper's **22.7 → 26.1 % (+3.3 pp)** under the
> official `semantic-kitti-api` (precise eval output **26.05 %**, the number the
> paper rounds to 26.1; **26.72 %** under the paper's internal SSCMetrics, a
> footnote ship-both number) — a paper-protocol number, not an at-deploy result.
> For the honest deployment number use `eval/js3c_val_realistic`
> (`bev_source: derived`), which lands at **24.32 %** under the official
> `semantic-kitti-api` evaluator. See `docs/REPRODUCIBILITY.md` and
> `docs/MODEL_ZOO.md` for the full GT-vs-derived breakdown.

The all-in-one driver runs the same eval with a pre-flight check on the
JS3C predictions (`tab:cross_base_js3c` is the `reproduce_table.py` command
key for the JS3C-Net row of the paper's cross-base table `tab:portable_s2d2`):

```bash
python scripts/reproduce_table.py tab:cross_base_js3c
```

## LMSCNet cross-base (paper tab:portable_s2d2, third base)

```bash
# Verify the LMSCNet predictions are dumped (see docs/REPRODUCIBILITY.md)
ls data/lmscnet_predictions/08 | head

# N=1, no TTA — expect 16.59 % val mIoU (paper rounds to 16.6; +1.8 pp over the
# LMSCNet base 14.76 %, re-scored from on-disk predictions through the official
# semantic-kitti-api — this supersedes the earlier 12.10 base/+4.49 summary)
# NOTE: the released LMSCNet model_ema.safetensors is missing its BN buffers
# (scores 11.04 under strict=False); load the full-state checkpoint to reach
# 16.59 — see the LMSCNet known-issue in docs/MODEL_ZOO.md.
python scripts/eval.py eval/lmscnet_val_1step \
    --checkpoint data/checkpoints/gssc_lmsc/gssc_lmsc_s2d2_real/model_ema.safetensors
```

LMSCNet conditions on a derived BEV (`bev_from_base: true`, height-pooled
from LMSCNet's own 3D prediction — never GT BEV), so 16.59 % is already an
at-deploy number with no GT-BEV oracle caveat. The all-in-one driver runs
the same eval with a pre-flight check on the LMSCNet predictions
(`tab:cross_base_lmsc` is the `reproduce_table.py` command key for the
LMSCNet row of the paper's cross-base table `tab:portable_s2d2`):

```bash
python scripts/reproduce_table.py tab:cross_base_lmsc
```

## Single-frame demo

See `examples/quickstart.ipynb` for an end-to-end demo on one frame.

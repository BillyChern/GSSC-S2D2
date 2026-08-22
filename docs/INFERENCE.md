# Inference recipes

## Single-step deployment (one S²D² correction step, N=1, no TTA)

```bash
python scripts/eval.py eval/val_1step --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors
```

Returns 38.54% val mIoU on SemanticKITTI seq 08. The same N=1 no-TTA setting
scores **38.8 % on the hidden test** — to our knowledge the best *causal,
single-sweep, single-sample* result on the leaderboard to date, +2.1 pp over the
previous best published score under that restriction (SCPNet, 36.7).

**This is not a real-time configuration, and the paper does not claim it is.**
The correction pass costs 107 ms, which drops the deployed base+refiner pipeline
to **3.23 FPS end-to-end** on an idle H100, against the frozen base's 4.95 FPS.
The refiner's marginal **9.33 FPS** is the throughput of the incremental UNet
pass alone, excluding the shared frozen-base forward; the paper explicitly
declines to credit the pipeline with it, calling it "an incremental pass, not a
deployable rate". Neither rate matches the sensor's 10 Hz sweep cadence.

## Multi-step S²D² correction sampling (peak quality)

```bash
python scripts/eval.py eval/step_sweep --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors
```

Returns the full sweep: 38.54 (N=1), 38.59 (N=2), 38.65 (N=4 peak), 38.16 (N=100).

## Hidden-test submission, headline configuration (N=1, no TTA — 38.8% test)

This is the configuration behind the paper's headline test number and the
superlative quoted above, and it is the row the paper's deployment predicate
admits. The paper's reproduction matrix names this exact config.

```bash
# Generate .label files
python scripts/infer.py infer/test_1step \
    --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors \
    --output predictions/test_n1/

# Bundle for submission
cd predictions/test_n1 && zip -r submission.zip sequences/
```

Submit `submission.zip` to the [SemanticKITTI Codabench leaderboard](https://www.codabench.org/competitions/13814/#/results-tab).
Expected: **38.8 % mIoU, 58.9 % completion IoU**.

## Hidden-test submission, D4 TTA (test-server best, 39.2%)

Four correction steps plus an eight-view D4 ensemble. It is a real result we
report, but the paper **excludes it from bolding** and from the deployment
predicate that indexes the single-sweep superlative, so it is the extra rather
than the headline. Use the `infer/test_1step` recipe above if you are checking
the paper's main claim.

```bash
# Generate .label files
python scripts/infer.py infer/test_d4tta \
    --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors \
    --output predictions/test_d4tta/

# Bundle for submission
cd predictions/test_d4tta && zip -r submission.zip sequences/
```

Submit `submission.zip` the same way. Expected: **39.2 % mIoU, 59.0 %
completion IoU**.

> **`scripts/reproduce_table.py tab:main_results` runs the D4 TTA flow, not
> this one.** The driver maps the paper's main-results label to
> `infer/test_d4tta`, so it regenerates the 39.2 row. To regenerate the 38.8
> headline row, run the `infer/test_1step` command above by hand.

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

# N=1, no TTA, GT BEV. eval/js3c_val_1step is an alias of eval/js3c_val_paper and
# sets `bev_source: gt`, so what it prints is a GT-BEV DIAGNOSTIC of ours: expect
# 26.05 % as a diagnostic under the official semantic-kitti-api. The paper does
# not print it. It is NOT the paper's headline for this base, and it is NOT the
# protocol behind the 26.72 % continuity figure (that one is derived BEV -- see
# the note below). For the headline, run the derived-BEV command below.
python scripts/eval.py eval/js3c_val_1step \
    --checkpoint data/checkpoints/gssc_js3c/gssc_js3c_s2d2_real/model_ema.safetensors

# N=1, no TTA, derived BEV — the paper headline for this base: 22.7 -> 24.3 % val
# mIoU (+1.6 pp), precise eval output 24.32 %.
python scripts/eval.py eval/js3c_val_realistic \
    --checkpoint data/checkpoints/gssc_js3c/gssc_js3c_s2d2_real/model_ema.safetensors

# N=4 + D4 TTA — expect ≥ 24.32 % (TTA monotone gain on the at-deploy derived-BEV
# path; the D4 path forces derived BEV, GT BEV being incompatible with it)
python scripts/eval.py eval/js3c_val_d4tta \
    --checkpoint data/checkpoints/gssc_js3c/gssc_js3c_s2d2_real/model_ema.safetensors
```

> **Three numbers, and what actually separates them.**
> The paper's headline for this base is **22.7 → 24.3 % (+1.6 pp)**: **derived
> BEV**, scored by the **official `semantic-kitti-api`**, reproduced by
> `eval/js3c_val_realistic` (`bev_source: derived`, precise output **24.32 %**)
> -- which is also the protocol the released checkpoint was trained under
> (`configs/train/js3c_real.yaml` sets `bev_from_base: true`).
> The **26.7 %** figure the supplement carries above it is the **same
> derived-BEV setting** read by the paper's internal training-time evaluator
> (`SSCMetrics`, precise **26.72 %**), retained there for continuity only.
> **The evaluator is what separates 26.7 from 24.3, not the BEV source.**
> `eval/js3c_val_1step` (= `eval/js3c_val_paper`) is a third thing: it sets
> `bev_source: gt`, conditions on a ground-truth BEV oracle, and lands at a
> **26.05 %** diagnostic of ours under the official `semantic-kitti-api`. The
> paper prints neither "26.05" nor "26.1" anywhere, and this GT-BEV protocol is
> not what produced 26.72. (The paper's own GT-BEV row, 61.8 %, is a
> *separately trained* variant it labels an upper bound, not this checkpoint.)
> See `docs/REPRODUCIBILITY.md` and `docs/MODEL_ZOO.md` for the full breakdown.

The all-in-one driver takes the paper's cross-base table label directly and
runs the same eval with a pre-flight check on each base's predictions. The
label covers more than this section: it reproduces the LMSCNet row and then the
JS3C-Net row, so it needs both prediction dumps on disk.

```bash
python scripts/reproduce_table.py tab:portable_s2d2
```

## LMSCNet cross-base (paper tab:portable_s2d2, third base)

```bash
# Verify the LMSCNet predictions are dumped (see docs/REPRODUCIBILITY.md)
ls data/lmscnet_predictions/08 | head

# N=1, no TTA — expect 16.59 % val mIoU (paper rounds to 16.6; +1.8 pp over the
# LMSCNet base 14.76 %, re-scored from on-disk predictions through the official
# semantic-kitti-api — this supersedes the earlier 12.10 base/+4.49 summary)
# NOTE: the released LMSCNet model_ema.safetensors ships complete (278 tensors,
# 45 BN buffers) and reproduces 16.59 directly; no full-state-checkpoint
# workaround is needed.
python scripts/eval.py eval/lmscnet_val_1step \
    --checkpoint data/checkpoints/gssc_lmsc/gssc_lmsc_s2d2_real/model_ema.safetensors
```

LMSCNet conditions on a derived BEV (`bev_from_base: true`, height-pooled
from LMSCNet's own 3D prediction — never GT BEV), so 16.59 % is already an
at-deploy number with no GT-BEV oracle caveat. The all-in-one driver reaches
this row through the same paper label as the JS3C-Net section above
(`tab:portable_s2d2`), which reproduces the LMSCNet row and the JS3C-Net row in
one go, each behind a pre-flight check on that base's predictions:

```bash
python scripts/reproduce_table.py tab:portable_s2d2
```

## Single-frame demo

See `examples/quickstart.ipynb` for an end-to-end demo on one frame.

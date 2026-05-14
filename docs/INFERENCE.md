# Inference recipes

## Real-time deployment (single S²D² correction step, 9.33 FPS marginal on H100)

```bash
python scripts/eval.py eval/val_1step --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors
```

Returns 38.54% val mIoU on SemanticKITTI seq 08.

## Multi-step S²D² correction sampling (peak quality)

```bash
python scripts/eval.py eval/step_sweep --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors
```

Returns the full sweep: 38.54 (N=1), 38.59 (N=2), 38.65 (N=4 peak), 38.16 (N=100).

## D4 TTA (test-server SOTA, 39.2% test)

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

## JS3C-Net cross-base (paper Tab. III rows 90-91)

```bash
# Verify the JS3C predictions are dumped (see docs/REPRODUCIBILITY.md)
ls data/js3cnet_predictions/08 | head

# N=1, no TTA — expect 26.72 % val mIoU (+3.99 pp over JS3C-Net base 22.73 %)
python scripts/eval.py eval/js3c_val_1step \
    --checkpoint data/checkpoints/gssc_js3c/gssc_js3c_s2d2_real/model_ema.safetensors

# N=4 + D4 TTA — expect ≥ 26.72 % (TTA monotone gain)
python scripts/eval.py eval/js3c_val_d4tta \
    --checkpoint data/checkpoints/gssc_js3c/gssc_js3c_s2d2_real/model_ema.safetensors
```

The all-in-one driver runs the same eval with a pre-flight check on the
JS3C predictions:

```bash
python scripts/reproduce_table.py tab:cross_base_js3c
```

## Single-frame demo

See `examples/00_quickstart.ipynb` for an end-to-end demo on one frame.

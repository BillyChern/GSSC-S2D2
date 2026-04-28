# Inference recipes

## Real-time deployment (single S²D² correction step, 9.33 FPS marginal on H100)

```bash
python scripts/eval.py eval/val_1step --checkpoint data/checkpoints/gssc_31k_mf_step40000.safetensors
```

Returns 38.54% val mIoU on SemanticKITTI seq 08.

## Multi-step S²D² correction sampling (peak quality)

```bash
python scripts/eval.py eval/step_sweep --checkpoint data/checkpoints/gssc_31k_mf_step40000.safetensors
```

Returns the full sweep: 38.54 (N=1), 38.59 (N=2), 38.65 (N=4 peak), 38.16 (N=100).

## D4 TTA (test-server SOTA, 39.2% test)

```bash
# Generate .label files
python scripts/infer.py infer/test_d4tta \
    --checkpoint data/checkpoints/gssc_31k_mf_step40000.safetensors \
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
    --checkpoint data/checkpoints/gssc_31k_mf_step40000.safetensors \
    --data-root /path/to/your_data
```

## Single-frame demo

See `examples/00_quickstart.ipynb` for an end-to-end demo on one frame.

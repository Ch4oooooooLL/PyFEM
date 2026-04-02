# Research Protocol Upgrade

This project now defaults to a stricter experimental protocol for damage-identification studies.

## What Changed

- Normalization statistics are fit on the training subset only.
- Generated datasets can include healthy and near-healthy samples.
- Dataset generation stores sample-level metadata for downstream OOD splits.
- Training supports `damage_holdout` splits so severe-damage samples can be reserved for OOD testing.
- Training supports `online_prefix` mode for causal prefix observation tasks.
- Validation can select a decision threshold automatically instead of relying on a fixed `0.95`.
- Checkpoints persist preprocessing statistics, task mode, and model arguments for inference.
- Condition prediction now distinguishes true-target metrics from FEM-proxy comparison metrics.
- `pinn_v2` is the recommended physics-regularized baseline. The old `pinn` path is retained as `legacy_pinn`.

## Recommended Commands

```bash
run.bat dataset -j 4
run.bat train --model gt --epochs 100
run.bat train --model pinn_v2 --epochs 100
run.bat predict
```

## Notes

- Old checkpoints without preprocessing metadata still work if `dataset_npz` is available.
- Non-causal checkpoints are no longer treated as true online detectors during condition prediction. Their history output is a repeated final-state proxy.

# Environment Notes

This repo currently uses two different conda environments.

## 1. Training and image generation: `eupmu-h200`

Use `eupmu-h200` for:

- `train-scripts/random_label_eu.py`
- `eval-scripts/generate-images.py`
- `eval-scripts/imageclassify.py`

Runtime-verified on this machine:

- Python `3.10.20`
- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `diffusers==0.14.0`
- `transformers==4.26.1`
- `torchmetrics==0.6.0`
- `datasets==2.14.7`
- `pytorch_lightning==1.4.2`
- `numpy==1.24.4`
- `pandas==2.0.3`
- `opencv-python-headless` imports as `cv2==4.7.0`
- `wandb` currently imports as `0.25.1`

Runtime checks that passed:

- `python eval-scripts/imageclassify.py --help`
- `python train-scripts/random_label_eu.py --help` is expected to use this env

Setup reference:

- [scripts/setup_eupmu_h200_env.sh](/home/tianbai2/EUPMU_reproduce_fix/SD_Class-wise_unlearn_NSFW_unlearn/EUPMU/scripts/setup_eupmu_h200_env.sh)

## 2. FID only: `fid-eval`

Use `fid-eval` only for:

- `eval-scripts/compute-fid.py`
- `eval-scripts/compute-fid-per-class.py`

Reason:

- `eupmu-h200` has `torchmetrics==0.6.0`
- `compute-fid.py` fails there with:
  `ImportError: cannot import name 'FrechetInceptionDistance' from 'torchmetrics.image.fid'`
- `fid-eval` has a newer `torchmetrics`, which fixes the broken FID path

Runtime-verified on this machine:

- Python `3.12.13`
- `torch==2.11.0+cu130`
- `torchvision==0.26.0+cu130`
- `torchmetrics==1.9.0`
- `torch_fidelity==0.4.0`
- `numpy==2.4.4`
- `pandas==3.0.2`
- `datasets==4.8.4`

Notably, this env is leaner and does not currently include training-only packages like `diffusers`, `cv2`, or `pytorch_lightning`.

Runtime checks that passed:

- `python eval-scripts/compute-fid.py --help`
- `python eval-scripts/compute-fid-per-class.py --help`

## Current run convention

The current validated tench workflow is driven by:

- [validate_tench_sweep.sh](/home/tianbai2/EUPMU_reproduce_fix/SD_Class-wise_unlearn_NSFW_unlearn/EUPMU/validate_tench_sweep.sh)

The new all-class runner is:

- [unlearning_all_classes_eu.sh](/home/tianbai2/EUPMU_reproduce_fix/SD_Class-wise_unlearn_NSFW_unlearn/EUPMU/unlearning_all_classes_eu.sh)

It is aligned to the best class-0 run:

- `epoch=5`
- `lr=1e-05`
- `w_lr=10.0`
- `error=0.5`
- `alpha=0.01`

and skips class `0` because that run already exists.

## Command split

Use this split consistently:

```bash
conda run -n eupmu-h200 python train-scripts/random_label_eu.py ...
conda run -n eupmu-h200 python eval-scripts/generate-images.py ... --num_samples 10
conda run -n eupmu-h200 python eval-scripts/imageclassify.py ...
conda run -n fid-eval python eval-scripts/compute-fid.py ...
```

`generate-images.py` currently loops ten times internally, so `--num_samples 10` produces 100 generated images per prompt.

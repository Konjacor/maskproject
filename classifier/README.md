# FaceMask Classifier Workspace

This workspace contains capture, dataset preparation, training, export, and live inference tools for the upper-face expression classifier.

## Layout

- `config/` - classifier configs
- `data/raw/` - captured labeled ROI images
- `data/processed/train/` - train split
- `data/processed/val/` - validation split
- `models/checkpoints/` - training checkpoints
- `models/export/` - exported SavedModel and TFLite files
- `runs/` - debug snapshots and runtime outputs
- `scripts/` - runnable tools
- `src/classifier/` - reusable classifier modules

## Install

Use a dedicated environment for classifier work and install:

- `pip install -r classifier/requirements.txt`

## Capture dataset

Run:

- `python3 classifier/scripts/capture_dataset.py --config classifier/config/default.json`

Controls:

- `1` -> label `neutral`
- `2` -> label `sleepy`
- `3` -> label `surprised`
- `4` -> label `cheerful`
- `s` -> save one sample
- `a` -> save a short burst
- `q` or `Esc` -> quit

The script uses the detector backend to crop the upper-face ROI and saves grayscale inputs into `classifier/data/raw/<label>/`.

## Prepare train/val split

Run:

- `python3 classifier/scripts/prepare_dataset.py --config classifier/config/default.json`

## Train classifier

Run:

- `python3 classifier/scripts/train_classifier.py --config classifier/config/default.json`

This trains a `MobileNetV3 Small` classifier and writes checkpoints to `classifier/models/checkpoints/`.

## Export TFLite

Run:

- `python3 classifier/scripts/export_tflite.py --config classifier/config/default.json`

Exports:

- SavedModel
- float TFLite
- INT8 TFLite
- `labels.json`

## Live inference

Run:

- `python3 classifier/scripts/live_inference.py --config classifier/config/default.json`

This reuses the detector backend, crops upper-face ROI, runs the TFLite classifier, prints class probabilities, and saves debug snapshots.

## Recommended workflow

1. Capture raw samples for each label in the real camera environment.
2. Prepare train/val split.
3. Train the baseline classifier.
4. Export TFLite models.
5. Run live inference on Raspberry Pi.
6. Compare classifier output against your existing state-machine behavior.

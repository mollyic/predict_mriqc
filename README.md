# MRIQC predict

This script loads **best saved models** (one per modality) and applies them to a **MRIQC output file** to produce prediction files (per-modality and a combined file).

## Requirements

* Python 3.9+
* `pandas`, `joblib`
* Your **input CSV** must contain a `modality` column (used to match the correct model per modality).

## Inputs

* `--input_csv`
  Path to the new data you want to score.

* `--dv`
  Dependent variable tag used when selecting models (`motion` or `quality`).

* `--classifier`
  Model family (`binary` or `multiclass`).

* `--out_dir` *(optional)*
  Folder to write predictions. Defaults to `predictions/`.

## Usage

```bash
python predict.py \
  --input_csv /path/to/new_data.csv \
  --dv motion \
  --classifier multiclass \
  --out_dir output/predictions_new
```

## What it does

1. Gets best predictive model from `input/best_models.csv`, filters to rows matching `--dv` and `--classifier`.
2. For each `modality` present in your input CSV:
   * Loads the corresponding model .
   * Writes a per-modality CSV with predictions and probabilities.

## Outputs

* Per-modality files:
  `preds_<MODALITY>_dv-<dv>_cls-<classifier>.csv`
* Combined file:
  `preds_ALL_dv-<dv>_cls-<classifier>.csv`

Each file includes:

* `modality`, `classifier`, `dv`
* `y_pred` (predicted class/label)
* Probability columns (e.g., `proba_class_0`, `proba_class_1`) when supported
* `model_type`, `model_file`

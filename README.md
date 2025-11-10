# MRIQC predict

This script predicts scan quality and motion for T1w, T2w and FLAIR scans. Models were trained on **MRIQC data** and scans were **rated by trained neuroradiologists**. It requires **MRIQC output files** to produce prediction files (per-modality and a combined file). For each `modality` present in your input MRIQC file the script loads the corresponding model and writes an output CSV with predictions and probabilities


## Requirements

* Python 3.9+
* `pandas`, `joblib`

## Inputs

* `--input_csv / -i`
  Path to the new data you want to score.

* `-dv`
  Dependent variable tag used when selecting models (`motion` or `quality`).

* `--classifier / -c`
  Model family (`binary` or `multiclass`).

* `--out_dir / -o` *(optional)*
  Folder to write predictions. Defaults to `predictions/`.

## Usage

```bash
python predict.py \
  -i /path/to/new_data.csv \
  -dv motion \
  -c multiclass 
```

## Outputs

* Per-modality files:
  `predictions_<MODALITY>_dv-<dv>_cls-<classifier>_infile<you_input_csv_name>.csv`
* Combined file:
  `predictions_dv-<dv>_cls-<classifier>_infile<you_input_csv_name>.csv`

## Results 
Output csv contain numerical and qualitative rating columns: 
* `rating`: predicted class/label (1 = positive class, 0 = negative class)
* `rating_label`: a qualitative rating label derived from `rating` consistent with the labels applied by neuroradiologists during trainig 
    * *Motion:*  Severe, Moderate, Mild, None
    * *Quality: * Very poor, Suboptimal, Acceptable, Above average, Excellent
* Probability columns for each possible class (e.g., `proba_class_0`, `proba_class_1`) when supported

"""
predict.py

Usage:
  python predict_from_file.py \
      --input_csv /path/to/new_data.csv \
      --best_models_csv best_models.csv \
      --dv motion \
      --classifier multiclass \
      --out_dir output/predictions_new

Notes:
- Expects your best_models.csv to have columns: modality, classifier, dv, model_file
- Uses project settings for drop_cols.
- Does NOT compute ROC AUC (no ground truth in the new input).
"""

import argparse
import pathlib
import pandas as pd
import joblib


TEST_FILE = "/Users/mollyireland/Documents/work/rqa_classifier-main/input/radiolqa_classifier_desc-allhubs_20250729.csv"
drop_cols = [
    'modality','bids_name', 'avg_quality', 'factor_quality', 'avg_susceptibility', 
    'factor_susceptibility', 'avg_motion', 'factor_motion',
    'avg_flow_ghosting', 'factor_flow_ghosting', 'aep_id', 
    'spacing_x', 'spacing_y', 'spacing_z',
    'size_x', 'size_y', 'size_z',]
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_csv", 
        #default=TEST_FILE, 
        required=True, 
        help="Path to the input CSV to score."
        )
    p.add_argument(
        "--best_models_csv", 
        default="input/best_models.csv",
        #required=True, 
        help="CSV with winning models (e.g., best_models.csv)."
        )
    p.add_argument(
        "--dv", 
        required=True, 
        choices=["motion", "quality"], 
        help="Dependent variable tag used in best_models.csv (e.g., 'motion' or 'quality')."
        )
    p.add_argument(
        "--classifier", 
        required=True, 
        choices=["binary", "multiclass"], 
        help="Classifier family."
        )

    p.add_argument(
        "--out_dir", 
        default='predictions/', 
        help="Directory to write predictions.")
    return p.parse_args()

def main():
    args = parse_args()

    input_csv = pathlib.Path(args.input_csv)
    best_models_csv = pathlib.Path(args.best_models_csv)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(input_csv)
    if "modality" not in df_all.columns:
        raise ValueError("Input CSV must contain a 'modality' column to match models by modality.")

    winners = pd.read_csv(best_models_csv)

    winners = winners[(winners["dv"] == args.dv) & (winners["classifier"] == args.classifier)]
    if winners.empty:
        raise ValueError(f"No rows in {best_models_csv} for dv='{args.dv}' and classifier='{args.classifier}'.")

    preds_all = []

    for modality, df_mod in df_all.groupby("modality"):
       
        match = winners[winners["modality"] == modality]
        if match.empty:
            print(f"[WARN] No model for modality='{modality}' (dv={args.dv}, classifier={args.classifier}). Skipping.")
            continue
        if len(match) > 1:
            print(f"[WARN] Multiple models found for modality='{modality}' (dv={args.dv}, classifier={args.classifier}). Using the first one.")
            quit()

        row = match.iloc[0]
        model_path = f"input/models/{row["model_file"].split('/')[-1]}"
        model_type = row["model"]
        print(f"\n== Modality: {modality}")
        print(f"   Using model: {model_path}")

        model = joblib.load(model_path)

        # drop the project configured columns
        X = df_mod.drop(columns=drop_cols, errors="ignore")

        # Predict
        y_pred = model.predict(X)
        y_proba = None
        proba_cols = {}
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
            # Use the classes_ from the fitted estimator to name probability columns
            model_classes = getattr(model, "classes_", None)
            if model_classes is None:
                # Try to access through final step if this is a Pipeline
                try:
                    model_classes = model.named_steps["clf"].classes_
                except Exception:
                    model_classes = None

            if model_classes is not None:
                for i, cls in enumerate(model_classes):
                    proba_cols[f"proba_class_{cls}"] = y_proba[:, i]
            else:
                # Fallback: generic proba columns
                for i in range(y_proba.shape[1]):
                    proba_cols[f"proba_col_{i}"] = y_proba[:, i]

        # predictions df
        preds_df = pd.DataFrame({
            "modality": modality,
            "classifier": args.classifier,
            "dv": args.dv,
            "y_pred": y_pred if model_type != 'CatBoostOrdinal' else y_pred.ravel(), 
            "model_type": model_type,
            "model_file": model_path,
        }, index=X.index).assign(**proba_cols)


        if f'avg_{args.dv}' in df_mod.columns:
            preds_df = pd.concat(
                [df_mod[[f'avg_{args.dv}']].reset_index(drop=True),
                preds_df.reset_index(drop=True)],
                axis=1
            )

        if 'bids_name' in df_mod.columns:
            preds_df = pd.concat(
                [df_mod[['bids_name']].reset_index(drop=True),
                preds_df.reset_index(drop=True)],
                axis=1
            )
        else:
            print(f"[WARN] 'bids_name' not found for modality={modality}; proceeding without it.")

        per_mod_out = out_dir / f"preds_{modality}_dv-{args.dv}_cls-{args.classifier}.csv"
        preds_df.to_csv(per_mod_out, index=False)
        print(f"   Wrote: {per_mod_out}")

        preds_all.append(preds_df)

    if preds_all:
        combined = pd.concat(preds_all, axis=0, ignore_index=True)
        combined_out = out_dir / f"preds_ALL_dv-{args.dv}_cls-{args.classifier}.csv"
        combined.to_csv(combined_out, index=False)
        print(f"\nCombined predictions written to: {combined_out}")
    else:
        print("\nNo predictions generated (no matching models or modalities).")

if __name__ == "__main__":
    main()

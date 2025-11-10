"""
predict.py

Usage:
  python predict_from_file.py \
      --input_csv /path/to/new_data.csv \
      --models_csv best_models.csv \
      --dv motion \
      --classifier multiclass \
      --out_dir output/predictions_new

"""

import argparse
import pandas as pd
import joblib
from pathlib import Path

TEST_FILE="test_results_desc-simple.csv"
drop_cols = [
    'modality','bids_name', 'avg_quality', 'factor_quality', 'avg_susceptibility', 
    'factor_susceptibility', 'avg_motion', 'factor_motion',
    'avg_flow_ghosting', 'factor_flow_ghosting', 'aep_id', 
    'spacing_x', 'spacing_y', 'spacing_z',
    'size_x', 'size_y', 'size_z']

ratings_dict = {
    "quality": {
        0: "Very poor",
        1: "Suboptimal",
        2: "Acceptable",
        3: "Above average",
        4: "Excellent"
    },
    "motion": {
        0: "Severe",
        1: "Moderate",
        2: "Mild",
        3: "None"
        }
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_csv", "-i",
        #default=TEST_FILE, 
        required=True, 
        type=Path,
        help="Path to the input CSV to score."
        )
    p.add_argument(
        "-dv", 
        required=True, 
        choices=["motion", "quality"], 
        help="Dependent variable to test: `motion` or `quality`"
        )
    p.add_argument(
        "--classifier", "-c",
        required=True, 
        choices=["binary", "multiclass"], 
        help="Classifier choice: `binary` or `multiclass`."
        )
    p.add_argument(
        "--out_dir", "-o",
        default='predictions/', 
        help="Directory to write predictions."
        )
    
    p.add_argument(
        "--models_csv", 
        default="src/results_models-overallbest_r-2_add-f1filter_desc-predict.csv",
        help="DEBUG: CSV with winning models, do not change unless updating models to run predictions."
        )
    return p.parse_args()

def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    models_csv = Path(args.models_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    ext = Path(input_csv).suffix.lower()
    sep = "\t" if ext in (".tsv", ".tab") else ","
    df_all = pd.read_csv(input_csv, sep=sep)

    if "modality" not in df_all.columns:
        df_all["modality"] = df_all["bids_name"].str.extract(r"_([A-Za-z0-9]+)$", expand=False)
    
    counts = df_all["modality"].value_counts(dropna=False)
    total  = counts.sum()
    print("\n=====================================================\n Dataset description:")
    for mod, n in counts.items():
        pct = 100 * n / total
        print(f"{str(mod):>10} | {n:>4} ({pct:5.1f}%) ")
    print("\n[NOTE] Modality is derived from bids_name column suffix (e.g. '_T1w'). If your filenames do not follow this convention, please manually add a `modality` column.")

    winners = pd.read_csv(models_csv)
    winners = winners[(winners["dv"] == args.dv) & (winners["classifier"] == args.classifier)]
    if winners.empty:
        raise ValueError(f"No rows in {models_csv} for dv='{args.dv}' and classifier='{args.classifier}'.")


    preds_all = []

    if args.classifier == 'multiclass':
        ratings_labels = ratings_dict[args.dv]
    else: 
        ratings_labels = {0: "Unacceptable", 1: "Acceptable"},
    print("\n=====================================================\n Generating predictions...")
    for modality, df_mod in df_all.groupby("modality"):
       
        match = winners[winners["modality"] == modality]
        print(f"\n + Modality: {modality}")

        if match.empty:
            print(f"   [WARN] No model for modality='{modality}' (dv={args.dv}, classifier={args.classifier}). Skipping.")
            continue
        if len(match) > 1:
            print(f"   [WARN] Multiple models found for modality='{modality}' (dv={args.dv}, classifier={args.classifier}). Using the first one.")
            quit()

        row = match.iloc[0]
        model_path = f"src/models/{row["model_name"]}"
        model_type = row["model"]
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
            model_classes = getattr(model, "classes_", None)
            if model_classes is None:
                try:
                    model_classes = model.named_steps["clf"].classes_
                except Exception:
                    model_classes = None

            if model_classes is not None:
                for i, cls in enumerate(model_classes):
                    proba_cols[f"proba_class_{cls}"] = y_proba[:, i]
            else:
                for i in range(y_proba.shape[1]):
                    proba_cols[f"proba_col_{i}"] = y_proba[:, i]
        # predictions df
        preds_df = pd.DataFrame({
            "modality": modality,
            "classifier": args.classifier,
            "dv": args.dv,
            "rating": y_pred if model_type != 'CatBoostOrdinal' else y_pred.ravel(), 
        }, index=X.index).assign(**proba_cols)

        preds_df["rating_label"] = preds_df["rating"].map(ratings_labels)
        preds_df = pd.concat(
            [df_mod[['bids_name']].reset_index(drop=True),
            preds_df.reset_index(drop=True)],
            axis=1
        )

        order = ['bids_name', 'modality', 'classifier', 'dv', 'rating', 'rating_label']
        rest = [c for c in preds_df.columns if c not in order]
        preds_df = preds_df[order + rest]

        if f'avg_{args.dv}' in df_mod.columns:
            preds_df = pd.concat(
                [df_mod[[f'avg_{args.dv}']].reset_index(drop=True),
                preds_df.reset_index(drop=True)],
                axis=1
            )

        per_mod_out = out_dir / f"predictions_{modality}_dv-{args.dv}_cls-{args.classifier}_infile-{input_csv.stem}.csv"
        preds_df.to_csv(per_mod_out, index=False)
        print(f"   Wrote: {per_mod_out}")

        preds_all.append(preds_df)

    if preds_all:
        combined = pd.concat(preds_all, axis=0, ignore_index=True)
        combined_out = out_dir / f"predictions_dv-{args.dv}_cls-{args.classifier}_infile-{input_csv.stem}.csv"
        combined.to_csv(combined_out, index=False)
        print(f"\n=====================================================\n\nDone! Combined predictions written to: {combined_out}")
    else:
        print("\n=====================================================\nNo predictions generated (no matching models or modalities).")

if __name__ == "__main__":
    main()

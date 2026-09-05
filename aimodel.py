"""
aimodel.py

Self-contained inference wrapper around the trained CPRI screening pipeline.

This does NOT retrain anything. It assumes the two final models were already
trained (see Data_cleaning.ipynb + main.ipynb) and saved with joblib to:
    Model/final_classifier_extratrees.pkl
    Model/final_regressor_randomforest.pkl

The preprocessing constants below (medians, threshold) were fitted ONCE on
the original 1000-row training set and are frozen here deliberately — do not
recompute them from a new/hidden file. Re-fitting medians on a new,
un-labelled file would silently change what "Invalid" means from run to run,
and the threshold was tuned specifically against the original training
label distribution.

Usage (from your teammate's UI or anywhere else):

    from aimodel import Model

    model = Model()
    output_path = model("C:/path/to/some_new_hidden_data.xlsx")
    # or equivalently:
    output_path = model.predict_file("some_new_hidden_data.csv")

`output_path` is the path to the saved predictions CSV
(Test_ID, Predicted_Reference_Parameter, Validity_Label).
"""

import os
import pandas as pd
import numpy as np
import joblib


class Model:
    # ---- Frozen preprocessing constants, fitted on the original training set ----
    # (Data_cleaning.ipynb, medians computed from Training_Data only)
    S1_MEDIAN = 13.627749999999999
    S2_MEDIAN = 13.57855
    S3_MEDIAN = 16.9269

    # Decision threshold: median of tuned per-fold thresholds across
    # 5 seeds x 5 folds for the ExtraTreesClassifier (main.ipynb, Step 11a)
    CLASSIFIER_THRESHOLD = 0.1967

    SENSOR_COLS = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]

    INPUT_COLS = [
        "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C",
        "Test_Duration_min", "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"
    ]

    FEATURE_COLS = [
        "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C", "Test_Duration_min",
        "Sensor_S1", "Sensor_S2", "Sensor_S3",
        "S1_missing", "S2_missing", "S3_missing", "S4_missing",
        "is_duplicate_input"
    ]

    def __init__(self, model_dir="Model"):
        """
        Loads the two pretrained models from `model_dir`. Raises a clear
        error immediately if they're missing, rather than failing later
        mid-prediction.
        """
        clf_path = os.path.join(model_dir, "final_classifier_extratrees.pkl")
        reg_path = os.path.join(model_dir, "final_regressor_randomforest.pkl")

        if not os.path.exists(clf_path):
            raise FileNotFoundError(
                f"Classifier model not found at '{clf_path}'. "
                f"Train and save it first (see main.ipynb, Step 11b)."
            )
        if not os.path.exists(reg_path):
            raise FileNotFoundError(
                f"Regressor model not found at '{reg_path}'. "
                f"Train and save it first (see main.ipynb)."
            )

        self.classifier = joblib.load(clf_path)
        self.regressor = joblib.load(reg_path)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    def load_data(self, filename):
        """
        Loads a new/hidden dataset from either .csv or .xlsx/.xls.

        For Excel files, tries a sheet literally named 'Test_Data' first
        (matching the original dataset's convention); if that sheet doesn't
        exist, falls back to the first sheet in the workbook. This avoids
        assuming the hidden file necessarily mirrors the original workbook's
        sheet names.
        """
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(filename)

        elif ext in (".xlsx", ".xls"):
            try:
                df = pd.read_excel(filename, sheet_name="Test_Data")
            except ValueError:
                df = pd.read_excel(filename, sheet_name=0)

        else:
            raise ValueError(
                f"Unsupported file type '{ext}'. Expected .csv, .xlsx, or .xls."
            )

        if "Test_ID" not in df.columns:
            # Hidden data should have Test_ID, but don't hard-fail if it's
            # missing — generate a stable placeholder so the pipeline can
            # still run end-to-end rather than crashing on a UI upload.
            df = df.reset_index(drop=True)
            df.insert(0, "Test_ID", [f"ROW-{i:05d}" for i in range(len(df))])

        missing_cols = [c for c in self.INPUT_COLS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input file is missing required columns: {missing_cols}. "
                f"Expected columns: {self.INPUT_COLS}"
            )

        return df

    # ------------------------------------------------------------------
    # 2. Clean — same rules as Data_cleaning.ipynb, applied generically
    #    (no row/Test_ID-specific logic anywhere in here)
    # ------------------------------------------------------------------
    def clean(self, df):
        df = df.copy()

        # Fix physically impossible values: any negative sensor reading is
        # clipped to 0. Generic rule, not tied to any specific Test_ID.
        for c in self.SENSOR_COLS:
            df.loc[df[c] < 0, c] = 0

        # Missingness flags — captured BEFORE imputation, since missingness
        # itself is the strongest fault signal found during EDA.
        for c in self.SENSOR_COLS:
            flag_name = c.replace("Sensor_", "") + "_missing"
            df[flag_name] = df[c].isna().astype(int)

        # Duplicate-input flag — checked WITHIN this file only, since a
        # hidden dataset has no relationship to the original training rows.
        df["is_duplicate_input"] = df.duplicated(subset=self.INPUT_COLS, keep=False).astype(int)

        # Impute S1/S2/S3 using the FROZEN training medians (never recomputed
        # from the new file — see module docstring for why).
        df["Sensor_S1"] = df["Sensor_S1"].fillna(self.S1_MEDIAN)
        df["Sensor_S2"] = df["Sensor_S2"].fillna(self.S2_MEDIAN)
        df["Sensor_S3"] = df["Sensor_S3"].fillna(self.S3_MEDIAN)

        # Drop Sensor_S4's raw value (near-zero correlation with the target);
        # its missingness flag (S4_missing) was already captured above.
        df = df.drop(columns=["Sensor_S4"])

        # Final safety check before handing off to the models
        missing_features = [c for c in self.FEATURE_COLS if c not in df.columns]
        if missing_features:
            raise RuntimeError(
                f"Cleaning did not produce expected feature columns: {missing_features}"
            )
        if df[self.FEATURE_COLS].isna().sum().sum() > 0:
            raise RuntimeError("Unexpected NaNs remain in feature columns after cleaning.")

        return df

    # ------------------------------------------------------------------
    # 3. Predict
    # ------------------------------------------------------------------
    def predict(self, df_clean):
        X = df_clean[self.FEATURE_COLS]

        reg_preds = self.regressor.predict(X)

        probs = self.classifier.predict_proba(X)[:, 1]
        class_preds = (probs >= self.CLASSIFIER_THRESHOLD).astype(int)
        labels = pd.Series(class_preds).map({0: "Valid", 1: "Invalid"}).values

        result = pd.DataFrame({
            "Test_ID": df_clean["Test_ID"].values,
            "Predicted_Reference_Parameter": reg_preds,
            "Validity_Label": labels
        })

        return result

    # ------------------------------------------------------------------
    # 4. End-to-end entry point
    # ------------------------------------------------------------------
    def predict_file(self, input_path, output_path=None):
        """
        Full pipeline: load -> clean -> predict -> save -> return saved path.

        If output_path is not given, saves next to the input file as
        '<input_filename>_predictions.csv'.
        """
        raw = self.load_data(input_path)
        cleaned = self.clean(raw)
        result = self.predict(cleaned)

        if output_path is None:
            base = os.path.splitext(os.path.basename(input_path))[0]
            output_path = f"{base}_predictions.csv"

        result.to_csv(output_path, index=False)
        return output_path

    def __call__(self, input_path, output_path=None):
        """Allows: Model()(path_to_file) -> saved_output_path"""
        return self.predict_file(input_path, output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python aimodel.py <path_to_input_file.csv_or_.xlsx> [output_path.csv]")
        sys.exit(1)

    input_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else None

    model = Model()
    saved_path = model.predict_file(input_file, out_file)
    print(f"Predictions saved to: {saved_path}")

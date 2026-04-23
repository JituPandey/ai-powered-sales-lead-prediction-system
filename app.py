import os
import pickle
import warnings

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── globals ──────────────────────────────────────────────────────────────────
_model        = None
_preprocessor = None
_feature_info = None
_dropdown     = {}

PATHS = {
    "model"      : "model.pkl",
    "preprocessor": "preprocessor.pkl",
    "feature_info": "feature_info.pkl",
    "dropdown"   : "dropdown_options.pkl",
    "data"       : "sales_leads.csv",
}

# ── training ──────────────────────────────────────────────────────────────────
def _train():
    global _model, _preprocessor, _feature_info, _dropdown

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBClassifier

    print("[*] Loading dataset …")
    df = pd.read_csv(PATHS["data"])

    # ── collect dropdown options (before any drops) ──────────────────────────
    ui_cats = [
        "Lead Origin", "Lead Source", "Last Activity", "Last Notable Activity",
        "Country", "Specialization", "How did you hear about X Education",
        "What is your current occupation",
        "What matters most to you in choosing a course",
        "Tags", "Lead Profile", "City", "Do Not Email", "Do Not Call",
    ]
    for col in ui_cats:
        if col in df.columns:
            vals = sorted(df[col].dropna().unique().tolist())
            _dropdown[col] = vals

    # ── preprocessing ────────────────────────────────────────────────────────
    df = df.drop(columns=["Prospect ID", "Lead Number"])
    X  = df.drop(columns=["Converted"])
    y  = df["Converted"]

    missing_pct  = X.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 40].index.tolist()
    X = X.drop(columns=high_missing)

    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    _preprocessor = ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ])

    print("[*] Preprocessing …")
    X_proc = _preprocessor.fit_transform(X)

    print("[*] Training XGBoost …")
    _model = XGBClassifier(
        n_estimators=100, eval_metric="logloss", random_state=42
    )
    _model.fit(X_proc, y)

    _feature_info = {
        "num_features": num_features,
        "cat_features": cat_features,
        "dropped"     : high_missing,
        "all_columns" : X.columns.tolist(),
    }

    # ── persist ──────────────────────────────────────────────────────────────
    for key, obj in [
        ("model", _model), ("preprocessor", _preprocessor),
        ("feature_info", _feature_info), ("dropdown", _dropdown),
    ]:
        with open(PATHS[key], "wb") as f:
            pickle.dump(obj, f)

    print("[OK] Model trained and saved.")


def _load():
    global _model, _preprocessor, _feature_info, _dropdown
    if all(os.path.exists(PATHS[k]) for k in ["model", "preprocessor", "feature_info", "dropdown"]):
        print("[*] Loading saved model …")
        with open(PATHS["model"],       "rb") as f: _model        = pickle.load(f)
        with open(PATHS["preprocessor"],"rb") as f: _preprocessor = pickle.load(f)
        with open(PATHS["feature_info"],"rb") as f: _feature_info = pickle.load(f)
        with open(PATHS["dropdown"],    "rb") as f: _dropdown     = pickle.load(f)
        print("[OK] Model loaded.")
    else:
        _train()


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", dropdown=_dropdown)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # build a row with NaN for everything not supplied
    row = {col: [np.nan] for col in _feature_info["all_columns"]}
    for key, val in data.items():
        if key in row and val not in ("", None):
            row[key] = [val]

    df_input = pd.DataFrame(row)

    # coerce numeric columns
    for col in _feature_info["num_features"]:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

    X_proc = _preprocessor.transform(df_input)
    prob   = float(_model.predict_proba(X_proc)[0][1])
    pred   = int(prob >= 0.5)

    # risk tier
    if prob >= 0.75:
        tier = "Hot Lead"
    elif prob >= 0.50:
        tier = "Warm Lead"
    elif prob >= 0.30:
        tier = "Cool Lead"
    else:
        tier = "Cold Lead"

    return jsonify({
        "prediction" : pred,
        "probability": round(prob, 4),
        "confidence" : f"{prob * 100:.1f}%",
        "tier"       : tier,
        "label"      : "High Conversion Likely" if pred == 1 else "Low Conversion Likelihood",
    })


# ── entry-point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _load()
    print("\n[OK] Server ready --> http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)

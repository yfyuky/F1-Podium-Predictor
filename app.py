from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -------------------------
# Load data / model on startup
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DF_PATH = DATA_DIR / "feature_engineered_dataset.csv"
MODEL_PATH = MODELS_DIR / "final_gb_model.joblib"
THRESHOLD = 0.30

df_feat = pd.read_csv(DF_PATH)
model = joblib.load(MODEL_PATH)

print("Loaded dataset from:", DF_PATH)
print("Loaded model from:", MODEL_PATH)

app = Flask(__name__)
CORS(app)

# -------------------------
# Signal functions (same as notebook)
# -------------------------
def driver_form_signal(row):
    if row["round"] <= 3:
        return "UNKNOWN"
    if row["driver_points_last3"] >= 18:
        return "STRONG"
    elif row["driver_points_last3"] >= 8:
        return "MODERATE"
    else:
        return "WEAK"

def constructor_momentum_signal(row):
    if row["round"] <= 3:
        return "UNKNOWN"
    if row["constructor_podiums_last3"] >= 0.5:
        return "HIGH"
    elif row["constructor_podiums_last3"] >= 0.2:
        return "MEDIUM"
    else:
        return "LOW"

def grid_advantage_signal(row):
    if row["grid"] <= 3:
        return "FRONT"
    elif row["grid"] <= 10:
        return "MIDFIELD"
    else:
        return "BACK"

def consistency_signal(row):
    if row["round"] <= 3:
        return "UNKNOWN"
    if row["driver_finishpos_last3"] <= 4:
        return "HIGH"
    elif row["driver_finishpos_last3"] <= 9:
        return "MEDIUM"
    else:
        return "LOW"

def generate_prediction_output(row_dict, podium_proba, threshold=THRESHOLD):
    # row_dict is a plain dict (JSON-friendly)
    signals = {
        "driver_form": driver_form_signal(row_dict),
        "constructor_momentum": constructor_momentum_signal(row_dict),
        "grid_positioning": grid_advantage_signal(row_dict),
        "consistency": consistency_signal(row_dict),
    }

    decision = "PODIUM_LIKELY" if podium_proba >= threshold else "PODIUM_UNLIKELY"

    if podium_proba >= 0.75 or podium_proba <= 0.15:
        confidence = "HIGH"
    elif podium_proba >= 0.55 or podium_proba <= 0.30:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    facts = {
        "grid": int(row_dict["grid"]),
        "qual_position": int(float(row_dict["qual_position"])),
        "driver_points_last3": float(row_dict["driver_points_last3"]),
        "constructor_points_last3": float(row_dict["constructor_points_last3"]),
        "driver_podiums_last3": float(row_dict["driver_podiums_last3"]),
        "driver_finishpos_last3": float(row_dict["driver_finishpos_last3"]),
        "constructor_podiums_last3": float(row_dict["constructor_podiums_last3"]),
        "grid_inverse": float(row_dict["grid_inverse"]),
    }

    reasons = []
    if signals["grid_positioning"] == "FRONT":
        reasons.append("strong starting position advantage")
    if signals["driver_form"] == "STRONG":
        reasons.append("strong recent driver performance")
    if signals["constructor_momentum"] == "HIGH":
        reasons.append("strong recent constructor momentum")
    if signals["consistency"] == "HIGH":
        reasons.append("high recent finishing consistency")
    if not reasons:
        reasons.append("no strong performance signals detected")

    summary = (
        f"Podium probability {podium_proba:.2f}. "
        f"Driver form {signals['driver_form']}, "
        f"constructor momentum {signals['constructor_momentum']}, "
        f"grid positioning {signals['grid_positioning']}, "
        f"consistency {signals['consistency']}."
    )

    return {
        "probability": float(podium_proba),
        "decision_threshold": float(threshold),
        "decision": decision,
        "confidence_level": confidence,
        "signals": signals,
        "facts": facts,
        "reasons": reasons,
        "summary": summary,
    }

# -------------------------
# Helper: find a row
# -------------------------
def find_row(season, rnd, driver_id):
    subset = df_feat[
        (df_feat["season"] == season) &
        (df_feat["round"] == rnd) &
        (df_feat["driverId"] == driver_id)
    ]
    if subset.empty:
        return None
    return subset.iloc[0]

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/predict")
def predict():
    try:
        season = int(request.args.get("season"))
        rnd = int(request.args.get("round"))
        driver_id = request.args.get("driverId")
        if not driver_id:
            return jsonify({"error": "Missing driverId"}), 400
    except Exception:
        return jsonify({"error": "Invalid or missing season/round/driverId"}), 400

    row = find_row(season, rnd, driver_id)
    if row is None:
        return jsonify({"error": "No matching row found"}), 404

    # Build model input row as a 1-row DataFrame
    X = pd.DataFrame([row.drop("podium").to_dict()])

    proba = float(model.predict_proba(X)[:, 1][0])

    row_dict = row.to_dict()
    out = generate_prediction_output(row_dict, proba, threshold=THRESHOLD)
    out["request"] = {"season": season, "round": rnd, "driverId": driver_id}

    return jsonify(out)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
from flask import Flask, request, jsonify, render_template_string
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
# Simple frontend
# -------------------------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>F1 Podium Predictor</title>
    <style>
        :root {
            color-scheme: light dark;
        }
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 32px 16px;
            background: #f5f7fb;
            color: #1a1a1a;
        }
        .container {
            max-width: 860px;
            margin: 0 auto;
        }
        .card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
        h1, h2, h3 {
            margin-top: 0;
        }
        .subtitle {
            color: #555;
            margin-bottom: 24px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
        }
        label {
            font-weight: 600;
            display: block;
            margin-bottom: 8px;
        }
        input, select, button {
            width: 100%;
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #d0d7de;
            font-size: 15px;
            box-sizing: border-box;
        }
        button {
            cursor: pointer;
            font-weight: 700;
            border: none;
            background: #111827;
            color: white;
            margin-top: 8px;
        }
        button:hover {
            opacity: 0.92;
        }
        .muted {
            color: #666;
            font-size: 14px;
        }
        .result-box {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 18px;
            background: #fafafa;
            margin-top: 16px;
        }
        .pill {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
            margin-right: 8px;
            margin-bottom: 8px;
            background: #e5e7eb;
        }
        .error {
            color: #b00020;
            font-weight: 600;
        }
        ul {
            padding-left: 20px;
        }
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            background: #0f172a;
            color: #f8fafc;
            padding: 14px;
            border-radius: 10px;
            overflow-x: auto;
        }
        @media (max-width: 600px) {
            body {
                padding: 16px 12px;
            }
            .card {
                padding: 18px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>F1 Podium Predictor</h1>
            <p class="subtitle">
                Prototype interface for your Formula 1 podium prediction system.
                Enter a season, round, and driver ID to request a prediction from the Flask backend.
            </p>

            <form id="predict-form">
                <div class="grid">
                    <div>
                        <label for="season">Season</label>
                        <select id="season" name="season" required></select>
                    </div>

                    <div>
                        <label for="round">Round</label>
                        <select id="round" name="round" required></select>
                    </div>

                    <div>
                        <label for="driverId">Driver ID</label>
                        <select id="driverId" name="driverId" required></select>
                    </div>
                </div>

                <button type="submit">Predict Podium Probability</button>
            </form>

            <p class="muted">
                This prototype reads available options directly from the feature-engineered dataset.
            </p>
        </div>

        <div class="card">
            <h2>Prediction Result</h2>
            <div id="status" class="muted">Waiting for input.</div>
            <div id="result"></div>
        </div>
    </div>

    <script>
        const state = {
            rows: [],
        };

        const seasonSelect = document.getElementById("season");
        const roundSelect = document.getElementById("round");
        const driverSelect = document.getElementById("driverId");
        const form = document.getElementById("predict-form");
        const statusBox = document.getElementById("status");
        const resultBox = document.getElementById("result");

        function uniqueSorted(values, numeric = false) {
            const arr = [...new Set(values)];
            if (numeric) {
                return arr.sort((a, b) => Number(a) - Number(b));
            }
            return arr.sort();
        }

        function setOptions(selectEl, values) {
            selectEl.innerHTML = "";
            values.forEach(value => {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = value;
                selectEl.appendChild(option);
            });
        }

        function filterRows() {
            const season = seasonSelect.value;
            const round = roundSelect.value;
            return state.rows.filter(row =>
                String(row.season) === String(season) &&
                String(row.round) === String(round)
            );
        }

        function updateRounds() {
            const season = seasonSelect.value;
            const rounds = uniqueSorted(
                state.rows
                    .filter(row => String(row.season) === String(season))
                    .map(row => row.round),
                true
            );
            setOptions(roundSelect, rounds);
            updateDrivers();
        }

        function updateDrivers() {
            const rows = filterRows();
            const drivers = uniqueSorted(rows.map(row => row.driverId));
            setOptions(driverSelect, drivers);
        }

        function renderPrediction(data) {
            const probabilityPct = (data.probability * 100).toFixed(1);

            const reasonsHtml = (data.reasons || [])
                .map(reason => `<li>${reason}</li>`)
                .join("");

            const signals = data.signals || {};
            const facts = data.facts || {};

            resultBox.innerHTML = `
                <div class="result-box">
                    <h3>${data.decision}</h3>
                    <p><strong>Probability:</strong> ${probabilityPct}%</p>
                    <p><strong>Confidence:</strong> ${data.confidence_level}</p>
                    <p><strong>Threshold:</strong> ${data.decision_threshold}</p>
                    <p><strong>Summary:</strong> ${data.summary}</p>

                    <div>
                        <span class="pill">Driver Form: ${signals.driver_form ?? "N/A"}</span>
                        <span class="pill">Constructor Momentum: ${signals.constructor_momentum ?? "N/A"}</span>
                        <span class="pill">Grid Positioning: ${signals.grid_positioning ?? "N/A"}</span>
                        <span class="pill">Consistency: ${signals.consistency ?? "N/A"}</span>
                    </div>

                    <h3>Reasons</h3>
                    <ul>${reasonsHtml}</ul>

                    <h3>Key Facts</h3>
                    <pre>${JSON.stringify(facts, null, 2)}</pre>
                </div>
            `;
        }

        async function loadMetadata() {
            statusBox.textContent = "Loading available seasons, rounds, and drivers...";
            try {
                const response = await fetch("/metadata");
                const data = await response.json();

                state.rows = data.rows || [];

                if (!state.rows.length) {
                    statusBox.innerHTML = '<span class="error">No dataset rows available.</span>';
                    return;
                }

                const seasons = uniqueSorted(state.rows.map(row => row.season), true);
                setOptions(seasonSelect, seasons);
                updateRounds();

                seasonSelect.addEventListener("change", updateRounds);
                roundSelect.addEventListener("change", updateDrivers);

                statusBox.textContent = "Ready. Select race inputs and run a prediction.";
            } catch (error) {
                statusBox.innerHTML = `<span class="error">Failed to load metadata: ${error.message}</span>`;
            }
        }

        form.addEventListener("submit", async (event) => {
            event.preventDefault();

            const season = seasonSelect.value;
            const round = roundSelect.value;
            const driverId = driverSelect.value;

            statusBox.textContent = "Generating prediction...";
            resultBox.innerHTML = "";

            const params = new URLSearchParams({
                season,
                round,
                driverId
            });

            try {
                const response = await fetch(`/predict?${params.toString()}`);
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || "Prediction request failed.");
                }

                statusBox.textContent = "Prediction completed successfully.";
                renderPrediction(data);
            } catch (error) {
                statusBox.innerHTML = `<span class="error">${error.message}</span>`;
            }
        });

        loadMetadata();
    </script>
</body>
</html>
"""

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

def get_metadata_rows():
    meta = (
        df_feat[["season", "round", "driverId"]]
        .drop_duplicates()
        .sort_values(["season", "round", "driverId"])
    )
    return meta.to_dict(orient="records")

# -------------------------
# Routes
# -------------------------
@app.get("/")
def index():
    return render_template_string(INDEX_HTML)

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/metadata")
def metadata():
    return jsonify({"rows": get_metadata_rows()})

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
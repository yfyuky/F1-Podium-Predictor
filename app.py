from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import joblib
from pathlib import Path
from feature_builder import build_2025_lineup_map, build_feature_row

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
lineup_map = build_2025_lineup_map(df_feat)
model = joblib.load(MODEL_PATH)

print("Loaded dataset from:", DF_PATH)
print("Loaded model from:", MODEL_PATH)
print("Running app file:", __file__)

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
                Prototype interface for the Formula 1 podium prediction system.
                Enter a circuit, driver, qualifying position, and grid position to request a prediction from the Flask backend.
            </p>

            <form id="predict-form">
                <div class="grid">
                    <div>
                        <label for="circuitId">Circuit</label>
                        <select id="circuitId" name="circuitId" required></select>
                    </div>
                    <div>
                        <label for="driverId">Driver</label>
                        <select id="driverId" name="driverId" required></select>
                    </div>
                    <div>
                        <label for="qual_position">Qualifying Position</label>
                        <input type="number" id="qual_position" name="qual_position" min="1" max="20" required />
                    </div>
                    <div>
                        <label for="grid">Grid Position</label>
                        <input type="number" id="grid" name="grid" min="1" max="20" required />
                    </div>
                </div>
                <button type="submit">Predict Podium Probability</button>
            </form>
            <p class="muted">
                Enter a circuit, driver, qualifying position, and grid position to simulate a podium prediction using the 2025 lineup.
            </p>
        </div>

        <div class="card">
            <h2>Prediction Result</h2>
            <div id="status" class="muted">Waiting for input.</div>
            <div id="result"></div>
        </div>
    </div>

    <script>
        const form = document.getElementById("predict-form");
        const statusBox = document.getElementById("status");
        const resultBox = document.getElementById("result");

        function setOptions(selectEl, values) {
            selectEl.innerHTML = "";
            values.forEach(value => {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = value;
                selectEl.appendChild(option);
            });
        }

        function renderPrediction(data) {
            const probabilityPct = (data.probability * 100).toFixed(1);
            const rawProbabilityPct = typeof data.raw_probability === "number"
                ? (data.raw_probability * 100).toFixed(1)
                : null;
            const adjustedProbabilityPct = typeof data.adjusted_probability === "number"
                ? (data.adjusted_probability * 100).toFixed(1)
                : null;
            const adjustmentPts = typeof data.context_adjustment === "number"
                ? (data.context_adjustment * 100).toFixed(1)
                : null;

            const signals = data.signals || {};
            const facts = data.facts || {};
            const fmt = (value) => (typeof value === "number" ? value.toFixed(2) : "N/A");

            resultBox.innerHTML = `
                <div class="result-box">
                    <h3>${data.decision} (${probabilityPct}%)</h3>
                    ${rawProbabilityPct !== null ? `<p><strong>Raw Model Probability:</strong> ${rawProbabilityPct}%</p>` : ""}
                    ${adjustedProbabilityPct !== null ? `<p><strong>Context-Adjusted Probability:</strong> ${adjustedProbabilityPct}%</p>` : ""}
                    ${adjustmentPts !== null ? `<p><strong>Context Adjustment:</strong> ${adjustmentPts}% points</p>` : ""}
                    <p><strong>Confidence:</strong> ${data.confidence_level}</p>
                    <p><strong>Threshold:</strong> ${data.decision_threshold}</p>
                    <p><strong>Summary:</strong> ${data.summary}</p>
                    <p><strong>Model Insight:</strong> ${data.explanation_short}</p>

                    <div>
                        <span class="pill">Driver Form: ${signals.driver_form ?? "N/A"}</span>
                        <span class="pill">Constructor Momentum: ${signals.constructor_momentum ?? "N/A"}</span>
                        <span class="pill">Grid Positioning: ${signals.grid_positioning ?? "N/A"}</span>
                        <span class="pill">Consistency: ${signals.consistency ?? "N/A"}</span>
                    </div>

                    <h3>Explanation</h3>
                    <p>${data.explanation || "No explanation available."}</p>

                    ${data.contradiction_note ? `<p><strong>Model Note:</strong> ${data.contradiction_note}</p>` : ""}

                    <h3>Key Supporting Factors</h3>
                    <ul>${(data.main_support || []).map(item => `<li>${item}</li>`).join("")}</ul>

                    <h3>Primary Risk Factors</h3>
                    <ul>${(data.main_risks || []).map(item => `<li>${item}</li>`).join("")}</ul>
                    
                    <h3>Positive Factors</h3>
                    <ul>${(data.positive_factors || []).map(item => `<li>${item}</li>`).join("")}</ul>
                    
                    <h3>Risk Factors</h3>
                    <ul>${(data.risk_factors || []).map(item => `<li>${item}</li>`).join("")}</ul>

                    <h3>Model Evidence</h3>
                    <ul>
                        <li>Grid Position: ${facts.grid ?? "N/A"}</li>
                        <li>Qualifying Position: ${facts.qual_position ?? "N/A"}</li>
                        <li>Driver Points (Last 3): ${fmt(facts.driver_points_last3)}</li>
                        <li>Constructor Points (Last 3): ${fmt(facts.constructor_points_last3)}</li>
                        <li>Driver Podiums (Last 3): ${fmt(facts.driver_podiums_last3)}</li>
                        <li>Driver Avg Finish (Last 3): ${fmt(facts.driver_finishpos_last3)}</li>
                    </ul>

                    <h3>Debug Data</h3>
                    <pre>${JSON.stringify(facts, null, 2)}</pre>
                </div>
            `;
        }

        async function loadMetadata() {
            statusBox.textContent = "Loading available circuits and drivers...";
            try {
                const response = await fetch("/metadata");
                const data = await response.json();

                if (!data.drivers || !data.circuits) {
                    statusBox.innerHTML = '<span class="error">No metadata available.</span>';
                    return;
                }

                setOptions(document.getElementById("circuitId"), data.circuits);
                setOptions(document.getElementById("driverId"), data.drivers);

                statusBox.textContent = "Ready. Select race inputs and run a prediction.";
            } catch (error) {
                statusBox.innerHTML = `<span class="error">Failed to load metadata: ${error.message}</span>`;
            }
        }

        form.addEventListener("submit", async (event) => {
            event.preventDefault();

            const circuitId = document.getElementById("circuitId").value;
            const driverId = document.getElementById("driverId").value;
            const qual_position = document.getElementById("qual_position").value;
            const grid = document.getElementById("grid").value;

            statusBox.textContent = "Generating prediction...";
            resultBox.innerHTML = "";

            const params = new URLSearchParams({
                circuitId,
                driverId,
                qual_position,
                grid
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
    if row["driver_points_last3"] >= 12:
        return "STRONG"
    elif row["driver_points_last3"] >= 5:
        return "MODERATE"
    else:
        return "WEAK"

def constructor_momentum_signal(row):
    if row["round"] <= 3:
        return "UNKNOWN"
    if row["constructor_points_last3"] >= 10:
        return "HIGH"
    elif row["constructor_points_last3"] >= 3:
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
    if row["driver_finishpos_last3"] <= 6:
        return "HIGH"
    elif row["driver_finishpos_last3"] <= 10:
        return "MEDIUM"
    else:
        return "LOW"

def apply_context_adjustment(proba, grid, qual_position):
    grid_penalty = max(0, grid - 3) * 0.012
    qual_penalty = max(0, qual_position - 3) * 0.010

    front_bonus = 0.0
    if grid <= 3:
        front_bonus += (4 - grid) * 0.02
    if qual_position <= 3:
        front_bonus += (4 - qual_position) * 0.015

    adjusted = proba - grid_penalty - qual_penalty + front_bonus
    adjusted = max(0.0, min(1.0, adjusted))
    adjustment = adjusted - proba
    return float(adjusted), float(adjustment)

def generate_prediction_output(row_dict, podium_proba, threshold=THRESHOLD, raw_proba=None, context_adjustment=None):
    signals = {
        "driver_form": driver_form_signal(row_dict),
        "constructor_momentum": constructor_momentum_signal(row_dict),
        "grid_positioning": grid_advantage_signal(row_dict),
        "consistency": consistency_signal(row_dict),
    }

    if podium_proba >= 0.70:
        decision = "High Podium Potential"
    elif podium_proba >= 0.50:
        decision = "Moderate Podium Potential"
    elif podium_proba >= 0.30:
        decision = "Low Podium Potential"
    else:
        decision = "Unlikely Podium Outcome"

    # base confidence from probability
    if podium_proba >= 0.75 or podium_proba <= 0.15:
        confidence = "High"
    elif podium_proba >= 0.55 or podium_proba <= 0.30:
        confidence = "Moderate"
    else:
        confidence = "Low"

    # adjust confidence based on disagreement
    if raw_proba is not None and context_adjustment is not None:
        if abs(context_adjustment) > 0.25:
            confidence = "Low (Model and contextual factors are in conflict)"
        elif abs(context_adjustment) > 0.15:
            confidence = "Moderate (Some conflict between model and context)"

    facts = {
        "grid": int(row_dict["grid"]),
        "qual_position": int(float(row_dict["qual_position"])),
        "driver_points_last3": float(row_dict["driver_points_last3"]),
        "constructor_points_last3": float(row_dict["constructor_points_last3"]),
        "driver_podiums_last3": float(row_dict["driver_podiums_last3"]),
        "driver_finishpos_last3": float(row_dict["driver_finishpos_last3"]),
        "constructor_podiums_last3": float(row_dict["constructor_podiums_last3"]),
        "driver_track_avg_finish": float(row_dict["driver_track_avg_finish"]),
        "driver_track_podium_rate": float(row_dict["driver_track_podium_rate"]),
        "constructor_track_avg_finish": float(row_dict["constructor_track_avg_finish"]),
        "constructor_track_podium_rate": float(row_dict["constructor_track_podium_rate"]),
        "grid_inverse": float(row_dict["grid_inverse"]),
    }

    positive_factors = []
    risk_factors = []

    if row_dict["grid"] <= 3:
        positive_factors.append("front-row or near-front grid position")
    elif row_dict["grid"] >= 11:
        risk_factors.append("starting from the lower half of the grid")

    if row_dict["qual_position"] <= 3:
        positive_factors.append("strong qualifying performance")
    elif row_dict["qual_position"] >= 11:
        risk_factors.append("weaker qualifying position")

    if row_dict["driver_points_last3"] >= 12:
        positive_factors.append("strong recent driver points form")
    elif row_dict["driver_points_last3"] < 5:
        risk_factors.append("limited recent driver scoring form")

    if row_dict["driver_podiums_last3"] >= 1:
        positive_factors.append("recent podium finishes in previous races")

    if row_dict["constructor_points_last3"] >= 10:
        positive_factors.append("strong recent constructor momentum")
    elif row_dict["constructor_points_last3"] < 3:
        risk_factors.append("limited recent constructor momentum")

    if row_dict["driver_finishpos_last3"] <= 6:
        positive_factors.append("good recent finishing consistency")
    elif row_dict["driver_finishpos_last3"] > 10:
        risk_factors.append("inconsistent recent finishing performance")

    if row_dict["driver_track_podium_rate"] >= 0.30:
        positive_factors.append("encouraging historical podium rate at this circuit")
    elif row_dict["driver_track_avg_finish"] > 10:
        risk_factors.append("limited historical success at this circuit")

    if row_dict["constructor_track_podium_rate"] >= 0.30:
        positive_factors.append("constructor has competitive circuit history")

    if not positive_factors:
        positive_factors.append("no major positive indicators were detected")

    if not risk_factors:
        risk_factors.append("no major risk indicators were detected")

    main_support = []
    main_risks = []

    if row_dict["driver_points_last3"] >= 12:
        main_support.append("strong recent driver points form")
    if row_dict["constructor_points_last3"] >= 10:
        main_support.append("strong recent constructor momentum")
    if row_dict["driver_finishpos_last3"] <= 6:
        main_support.append("good recent finishing consistency")

    if row_dict["grid"] >= 11:
        main_risks.append("poor starting grid position")
    if row_dict["qual_position"] >= 11:
        main_risks.append("weak qualifying result")
    if row_dict["driver_track_avg_finish"] > 10:
        main_risks.append("limited historical circuit performance")

    if not main_support:
        main_support.append("no dominant supporting signal identified")
    if not main_risks:
        main_risks.append("no dominant risk signal identified")

    contradiction_note = None
    if podium_proba >= threshold and row_dict["grid"] >= 11:
        contradiction_note = (
            "There is a clear contradiction between strong historical performance signals "
            "and an unfavourable starting position. This suggests that past performance is "
            "offsetting a significant race-day disadvantage."
        )
    elif podium_proba < threshold and row_dict["grid"] <= 3:
        contradiction_note = (
            "Despite a strong starting position, the model remains cautious due to weaker "
            "underlying performance indicators. This highlights the importance of consistency "
            "and historical performance over isolated race conditions."
        )

    if raw_proba is not None and context_adjustment is not None:
        explanation_short = (
            f"The predicted podium probability is {podium_proba:.2f}, adjusted from an initial "
            f"model estimate of {raw_proba:.2f}. Contextual factors such as grid and qualifying "
            f"position contributed a {context_adjustment:+.2f} adjustment."
        )
    else:
        explanation_short = (
            f"The model estimates a podium probability of {podium_proba:.2f} "
            f"based on grid position, qualifying result, recent form, and historical performance signals."
        )

    if podium_proba >= threshold:
        explanation = (
            f"The model predicts a competitive likelihood of achieving a podium finish. "
            f"Key supporting factors include {', '.join(positive_factors[:3])}. "
            f"However, some limiting factors such as {', '.join(risk_factors[:2])} remain present. "
            f"Overall, the balance of performance indicators supports a positive outcome."
        )
    else:
        explanation = (
            f"The model predicts a reduced likelihood of a podium finish under the current race conditions. "
            f"The primary limiting factors include {', '.join(risk_factors[:3])}. "
            f"Although some positive indicators exist, such as {', '.join(positive_factors[:2])}, "
            f"they are insufficient to outweigh the identified risks."
        )

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
        "main_support": main_support,
        "main_risks": main_risks,
        "positive_factors": positive_factors,
        "risk_factors": risk_factors,
        "contradiction_note": contradiction_note,
        "summary": summary,
        "explanation_short": explanation_short,
        "explanation": explanation,
    }
    
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
    drivers = sorted(df_feat[df_feat["season"] == 2025]["driverId"].dropna().unique().tolist())
    circuits = sorted(df_feat["circuitId"].dropna().unique().tolist())
    return jsonify({
        "drivers": drivers,
        "circuits": circuits
    })

@app.get("/predict")
def predict():
    try:
        circuit_id = request.args.get("circuitId")
        driver_id = request.args.get("driverId")
        qual_position = float(request.args.get("qual_position"))
        grid = int(request.args.get("grid"))

        if not circuit_id or not driver_id:
            return jsonify({"error": "Missing circuitId or driverId"}), 400
        if qual_position <= 0 or qual_position > 20:
            return jsonify({"error": "Qualifying position must be between 1 and 20"}), 400
        if grid <= 0 or grid > 20:
            return jsonify({"error": "Grid must be between 1 and 20"}), 400
    except Exception:
        return jsonify({"error": "Invalid inputs"}), 400

    try:
        row_dict = build_feature_row(
            df_feat=df_feat,
            lineup_map=lineup_map,
            circuit_id=circuit_id,
            driver_id=driver_id,
            qual_position=qual_position,
            grid=grid
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    X = pd.DataFrame([row_dict])

    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_)

    raw_proba = float(model.predict_proba(X)[:, 1][0])
    proba, context_adjustment = apply_context_adjustment(raw_proba, grid, qual_position)

    out = generate_prediction_output(
        row_dict,
        proba,
        threshold=THRESHOLD,
        raw_proba=raw_proba,
        context_adjustment=context_adjustment,
    )
    out["request"] = {
        "circuitId": circuit_id,
        "driverId": driver_id,
        "qual_position": qual_position,
        "grid": grid
    }
    out["raw_probability"] = float(raw_proba)
    out["adjusted_probability"] = float(proba)
    out["context_adjustment"] = float(context_adjustment)
    out["resolved_constructor"] = row_dict["constructorId"]

    return jsonify(out)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
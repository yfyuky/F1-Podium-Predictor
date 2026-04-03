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
            --bg: #0a0a0f;
            --bg-soft: #11131a;
            --card: rgba(255, 255, 255, 0.06);
            --card-strong: rgba(255, 255, 255, 0.09);
            --border: rgba(255, 255, 255, 0.10);
            --text: #f5f7fb;
            --muted: #b7bdc9;
            --accent: #e10600;
            --accent-2: #ff5a5f;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
            --radius-xl: 24px;
            --radius-lg: 18px;
            --radius-md: 14px;
            --maxw: 1240px;
        }

        * {
            box-sizing: border-box;
        }

        html {
            scroll-behavior: smooth;
        }

        body {
            margin: 0;
            font-family: Inter, Arial, sans-serif;
            background:
                radial-gradient(circle at top left, rgba(225, 6, 0, 0.22), transparent 28%),
                radial-gradient(circle at top right, rgba(255, 255, 255, 0.05), transparent 18%),
                linear-gradient(180deg, #090a0f 0%, #0d1017 55%, #090a0f 100%);
            color: var(--text);
        }

        body::before {
            content: "";
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 40px 40px;
            pointer-events: none;
        }

        a {
            color: inherit;
            text-decoration: none;
        }

        .page-shell {
            max-width: var(--maxw);
            margin: 0 auto;
            padding: 24px 20px 56px;
        }

        .topbar {
            position: relative;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            padding: 14px 18px;
            margin-bottom: 24px;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.04);
            backdrop-filter: blur(10px);
            border-radius: 18px;
            box-shadow: var(--shadow);
        }

        .topbar::after {
            content: "";
            position: absolute;
            left: 0;
            bottom: -1px;
            height: 3px;
            width: 120px;
            background: linear-gradient(90deg, #e10600, transparent);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 800;
            letter-spacing: 0.4px;
        }

        .brand-mark {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            display: grid;
            place-items: center;
            font-size: 18px;
            font-weight: 900;
            box-shadow: 0 10px 24px rgba(225, 6, 0, 0.35);
        }

        .brand-sub {
            font-size: 12px;
            color: var(--muted);
            font-weight: 500;
            letter-spacing: 0.2px;
        }

        .nav {
            display: flex;
            gap: 18px;
            color: var(--muted);
            font-size: 14px;
        }

        .nav a:hover {
            color: var(--text);
        }

        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 30px;
            padding: 50px 36px;
            margin-bottom: 26px;

            background:
                linear-gradient(120deg, rgba(225, 6, 0, 0.35), rgba(10,10,15,0.9)),
                #0a0a0f;

            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 30px 70px rgba(0,0,0,0.6);
        }

        .hero-bg-text {
            position: absolute;
            right: -40px;
            top: 10px;
            font-size: 180px;
            font-weight: 900;
            letter-spacing: -6px;
            color: rgba(255,255,255,0.04);
            pointer-events: none;
        }

        .hero::before {
            content: "";
            position: absolute;
            right: -90px;
            top: -90px;
            width: 320px;
            height: 320px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.12), transparent 65%);
            pointer-events: none;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 28px;
            align-items: center;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            color: #fff;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.4px;
            margin-bottom: 16px;
        }

        .eyebrow-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #fff;
        }

        .hero h1 {
            font-size: clamp(30px, 4.2vw, 54px);
            line-height: 1.04;
            margin: 0 0 14px;
            letter-spacing: -1.2px;
        }

        .hero p {
            margin: 0;
            max-width: 700px;
            color: var(--muted);
            font-size: 16px;
            line-height: 1.65;
        }

        .hero-stats {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }

        .hero-stat {
            padding: 18px;
            border-radius: 20px;
            background: rgba(255,255,255,0.06);
            border: 1px solid var(--border);
        }

        .hero-stat-label {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 8px;
        }

        .hero-stat-value {
            font-size: 24px;
            font-weight: 800;
            letter-spacing: -0.8px;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 0.92fr 1.08fr;
            gap: 22px;
            margin-bottom: 22px;
        }

        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
        }

        .card-inner {
            padding: 24px;
        }

        .card-inner::before {
            content: "";
            display: block;
            width: 40px;
            height: 3px;
            background: #e10600;
            margin-bottom: 16px;
        }

        .section-title {
            margin: 0 0 6px;
            font-size: 24px;
            font-weight: 900;
            letter-spacing: -0.5px;
        }

        .section-subtitle {
            margin: 0 0 22px;
            color: var(--muted);
            font-size: 14px;
            line-height: 1.6;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
        }

        .field {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .field.full {
            grid-column: span 2;
        }

        label {
            font-size: 13px;
            color: #e8ebf2;
            font-weight: 700;
            letter-spacing: 0.2px;
        }

        input, select {
            width: 100%;
            padding: 15px 15px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.05);
            font-weight: 600;
            color: var(--text);
            font-size: 15px;
            outline: none;
        }

        input:focus, select:focus {
            border-color: rgba(225, 6, 0, 0.7);
            box-shadow: 0 0 0 4px rgba(225, 6, 0, 0.15);
        }

        select option {
            color: #111;
        }

        .button-row {
            display: flex;
            gap: 12px;
            margin-top: 22px;
        }

        button {
            appearance: none;
            border: none;
            border-radius: 14px;
            padding: 15px 18px;
            font-size: 15px;
            font-weight: 800;
            cursor: pointer;
            transition: transform 0.15s ease, opacity 0.15s ease, box-shadow 0.15s ease;
        }

        button:hover {
            transform: translateY(-1px);
            opacity: 0.96;
        }

        .btn-primary {
            background: linear-gradient(135deg, #e10600, #ff3b3b);
            color: white;
            font-weight: 900;
            letter-spacing: 0.5px;
            box-shadow: 0 12px 30px rgba(225,6,0,0.4);
            flex: 1;
        }

        .btn-secondary {
            background: rgba(255,255,255,0.08);
            color: var(--text);
            border: 1px solid var(--border);
        }

        .helper-text {
            margin-top: 14px;
            color: var(--muted);
            font-size: 13px;
            line-height: 1.6;
        }

        .status {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            border-radius: 999px;
            background: rgba(255,255,255,0.07);
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 18px;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--warning);
            box-shadow: 0 0 18px rgba(245, 158, 11, 0.65);
        }

        .status.ready .status-dot {
            background: var(--success);
            box-shadow: 0 0 18px rgba(34, 197, 94, 0.65);
        }

        .status.error .status-dot {
            background: var(--danger);
            box-shadow: 0 0 18px rgba(239, 68, 68, 0.65);
        }

        .empty-state {
            min-height: 360px;
            display: grid;
            place-items: center;
            text-align: center;
            border: 1px dashed rgba(255,255,255,0.12);
            border-radius: 20px;
            background: rgba(255,255,255,0.03);
            color: var(--muted);
            padding: 20px;
        }

        .empty-state h3 {
            margin: 0 0 10px;
            color: var(--text);
            font-size: 22px;
        }

        .prediction-panel {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .decision-card {
            position: relative;
            border-radius: 22px;
            padding: 26px;
            background:
                linear-gradient(135deg, rgba(225,6,0,0.35), rgba(0,0,0,0.6)),
                #111;
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow:
                0 25px 60px rgba(225,6,0,0.25),
                inset 0 1px 0 rgba(255,255,255,0.06);
        }

        .decision-card::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            width: 6px;
            height: 100%;
            background: #e10600;
        }

        .decision-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.1px;
            color: #ffd9d9;
            margin-bottom: 12px;
            font-weight: 800;
        }

        .decision-value {
            font-size: clamp(30px, 3.1vw, 44px);
            line-height: 1.06;
            margin: 0 0 14px;
            letter-spacing: -1.1px;
            font-weight: 900;
        }

        .decision-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .meta-chip {
            display: inline-flex;
            align-items: center;
            padding: 10px 13px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.10);
            font-size: 13px;
            color: #f8fafc;
            font-weight: 700;
        }

        .meta-chip.conf-high {
            background: rgba(34, 197, 94, 0.16);
            border-color: rgba(34, 197, 94, 0.30);
        }

        .meta-chip.conf-moderate {
            background: rgba(59, 130, 246, 0.16);
            border-color: rgba(59, 130, 246, 0.30);
        }

        .meta-chip.conf-low {
            background: rgba(245, 158, 11, 0.18);
            border-color: rgba(245, 158, 11, 0.30);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
        }

        .metric-card {
            padding: 20px;
            border-radius: 18px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.10);
            min-height: 108px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .metric-label {
            color: var(--muted);
            font-size: 11px;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.9px;
            line-height: 1.4;
        }

        .metric-value {
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.7px;
            line-height: 1.1;
        }

        .split-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }

        .subcard {
            padding: 20px;
            border-radius: 18px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }

        .subcard h3 {
            margin: 0 0 14px;
            font-size: 17px;
            letter-spacing: -0.3px;
        }

        .subcard p {
            margin: 0;
            color: var(--muted);
            line-height: 1.65;
            font-size: 14px;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            padding: 10px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.10);
            color: #f3f6fb;
        }

        ul.clean-list {
            list-style: none;
            margin: 0;
            padding: 0;
            display: grid;
            gap: 10px;
        }

        ul.clean-list li {
            padding: 12px 14px;
            border-radius: 14px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            color: #eef2f8;
            line-height: 1.5;
            font-size: 14px;
        }

        details {
            border-radius: 18px;
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--border);
            overflow: hidden;
        }

        summary {
            cursor: pointer;
            list-style: none;
            padding: 18px 20px;
            font-weight: 800;
            font-size: 15px;
        }

        summary::-webkit-details-marker {
            display: none;
        }

        .details-content {
            padding: 0 20px 20px;
        }

        .tech-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 14px;
        }

        .tech-item {
            padding: 14px;
            border-radius: 14px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .tech-item strong {
            display: block;
            font-size: 12px;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.7px;
            margin-bottom: 8px;
        }

        .tech-item span {
            font-weight: 800;
            color: var(--text);
            font-size: 16px;
        }

        pre {
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
            background: #090b11;
            color: #f8fafc;
            padding: 16px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.08);
            overflow-x: auto;
            font-size: 12px;
            line-height: 1.55;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 18px;
        }

        .info-card {
            padding: 20px;
            border-radius: 20px;
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
        }

        .info-card h3 {
            margin: 0 0 10px;
            font-size: 18px;
            letter-spacing: -0.4px;
        }

        .info-card p {
            margin: 0;
            color: var(--muted);
            font-size: 14px;
            line-height: 1.7;
        }

        .footer-note {
            margin-top: 22px;
            color: var(--muted);
            text-align: center;
            font-size: 13px;
            line-height: 1.6;
        }

        .error-text {
            color: #ffd6d6;
        }

        @media (max-width: 1100px) {
            .hero-grid,
            .main-grid,
            .info-grid {
                grid-template-columns: 1fr;
            }

            .metrics-grid,
            .split-grid,
            .tech-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 760px) {
            .page-shell {
                padding: 16px 14px 40px;
            }

            .topbar {
                flex-direction: column;
                align-items: flex-start;
            }

            .nav {
                flex-wrap: wrap;
            }

            .hero {
                padding: 28px 20px;
            }

            .card-inner {
                padding: 18px;
            }

            .form-grid {
                grid-template-columns: 1fr;
            }

            .field.full {
                grid-column: span 1;
            }

            .hero-stats {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="page-shell">
        <header class="topbar">
            <div class="brand">
                <div class="brand-mark">F1</div>
                <div>
                    <div>F1 Podium Predictor</div>
                    <div class="brand-sub">Machine learning prototype for podium probability prediction</div>
                </div>
            </div>
            <nav class="nav">
                <a href="#predictor">Predictor</a>
                <a href="#insights">Insights</a>
                <a href="#technical">Model Breakdown</a>
            </nav>
        </header>

        <section class="hero">
            <div class="hero-bg-text">PODIUM</div>
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">
                        <span class="eyebrow-dot"></span>
                        Formula 1 Race Outcome Analytics
                    </div>
                    <h1>Race to the Podium</h1>
                    <p>
                        Predict Formula 1 podium potential using historical race performance,
                        constructor momentum, track-specific indicators, and race-context inputs
                        such as qualifying and grid position.
                    </p>
                </div>
                <div class="hero-stats">
                    <div class="hero-stat">
                        <div class="hero-stat-label">Model Type</div>
                        <div class="hero-stat-value">Gradient Boosting</div>
                    </div>
                    <div class="hero-stat">
                        <div class="hero-stat-label">Prediction Focus</div>
                        <div class="hero-stat-value">Podium Probability</div>
                    </div>
                    <div class="hero-stat">
                        <div class="hero-stat-label">Feature Basis</div>
                        <div class="hero-stat-value">Track + Form + Grid</div>
                    </div>
                    <div class="hero-stat">
                        <div class="hero-stat-label">System Output</div>
                        <div class="hero-stat-value">Explainable Result</div>
                    </div>
                </div>
            </div>
        </section>

        <section id="predictor" class="main-grid">
            <div class="card">
                <div class="card-inner">
                    <h2 class="section-title">Race Setup</h2>
                    <p class="section-subtitle">
                        Select a circuit and driver, then enter qualifying and starting grid positions
                        to generate a podium probability estimate for the race scenario.
                    </p>

                    <form id="predict-form">
                        <div class="form-grid">
                            <div class="field full">
                                <label for="circuitId">Circuit</label>
                                <select id="circuitId" name="circuitId" required></select>
                            </div>

                            <div class="field full">
                                <label for="driverId">Driver</label>
                                <select id="driverId" name="driverId" required></select>
                            </div>

                            <div class="field">
                                <label for="qual_position">Qualifying Position</label>
                                <input type="number" id="qual_position" name="qual_position" min="1" max="20" required />
                            </div>

                            <div class="field">
                                <label for="grid">Grid Position</label>
                                <input type="number" id="grid" name="grid" min="1" max="20" required />
                            </div>
                        </div>

                        <div class="button-row">
                            <button type="submit" class="btn-primary">Run Podium Analysis</button>
                            <button type="button" class="btn-secondary" id="reset-btn">Reset</button>
                        </div>
                    </form>

                    <p class="helper-text">
                        This interface is designed for presentation and demonstration purposes.
                        The output combines the model estimate with race-context adjustments and
                        explainability signals.
                    </p>
                </div>
            </div>

            <div class="card">
                <div class="card-inner">
                    <div id="statusWrap" class="status">
                        <span class="status-dot"></span>
                        <span id="statusText">Loading available circuits and drivers...</span>
                    </div>

                    <div id="result" class="empty-state">
                        <div>
                            <h3>Prediction Ready</h3>
                            <p>
                                Once you submit a race scenario, the system will display the decision,
                                probability breakdown, supporting signals, risk signals, and technical evidence.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <section id="insights" class="info-grid">
            <div class="info-card">
                <h3>Model Overview</h3>
                <p>
                    The system uses a Gradient Boosting classifier trained on historical Formula 1 data
                    from the 2022–2025 regulation period to estimate podium likelihood.
                </p>
            </div>

            <div class="info-card">
                <h3>Key Signals</h3>
                <p>
                    Predictions are influenced by recent driver form, constructor momentum, track history,
                    qualifying position, starting grid position, and race consistency indicators.
                </p>
            </div>

            <div class="info-card">
                <h3>Explainability Layer</h3>
                <p>
                    The output includes supporting factors, risk factors, contradiction notes,
                    and a natural language explanation to improve interpretability during evaluation.
                </p>
            </div>
        </section>

        <div class="footer-note">
            F1 Podium Predictor — presentation-ready prototype for podium prediction and explainable race analytics.
        </div>
    </div>

    <script>
        const form = document.getElementById("predict-form");
        const resultBox = document.getElementById("result");
        const resetBtn = document.getElementById("reset-btn");
        const statusWrap = document.getElementById("statusWrap");
        const statusText = document.getElementById("statusText");

        function setStatus(message, type = "default") {
            statusWrap.className = "status";
            if (type === "ready") statusWrap.classList.add("ready");
            if (type === "error") statusWrap.classList.add("error");
            statusText.textContent = message;
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

        function fmtNum(value, digits = 2) {
            return typeof value === "number" ? value.toFixed(digits) : "N/A";
        }

        function titleCase(value) {
            if (!value) return "N/A";
            return value.charAt(0).toUpperCase() + value.slice(1);
        }

        function getConfidenceClass(confidence) {
            if (!confidence) return "";
            const c = confidence.toLowerCase();
            if (c.includes("high")) return "conf-high";
            if (c.includes("moderate")) return "conf-moderate";
            if (c.includes("low")) return "conf-low";
            return "";
        }

        function renderList(items) {
            if (!items || !items.length) {
                return "<ul class='clean-list'><li>No items available.</li></ul>";
            }
            return `<ul class="clean-list">${items.map(item => `<li>${item}</li>`).join("")}</ul>`;
        }

        function renderPrediction(data) {
            const probabilityPct = (data.probability * 100).toFixed(1);
            const rawProbabilityPct = typeof data.raw_probability === "number"
                ? (data.raw_probability * 100).toFixed(1)
                : "N/A";
            const adjustedProbabilityPct = typeof data.adjusted_probability === "number"
                ? (data.adjusted_probability * 100).toFixed(1)
                : "N/A";
            const adjustmentPts = typeof data.context_adjustment === "number"
                ? (data.context_adjustment * 100).toFixed(1)
                : "N/A";

            const signals = data.signals || {};
            const facts = data.facts || {};

            resultBox.className = "prediction-panel";
            resultBox.innerHTML = `
                <div class="decision-card">
                    <div class="decision-label">Podium Projection</div>
                    <h2 class="decision-value">${data.decision} (${probabilityPct}%)</h2>
                    <div class="decision-meta">
                        <span class="meta-chip ${getConfidenceClass(data.confidence_level)}">Confidence: ${data.confidence_level || "N/A"}</span>
                        <span class="meta-chip">Constructor: ${titleCase(data.resolved_constructor)}</span>
                    </div>
                </div>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Raw Model Probability</div>
                        <div class="metric-value">${rawProbabilityPct}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Context-Adjusted Probability</div>
                        <div class="metric-value">${adjustedProbabilityPct}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Adjustment Impact</div>
                        <div class="metric-value">${adjustmentPts}%</div>
                    </div>
                </div>

                <div class="subcard">
                    <h3>Model Insight</h3>
                    <p>${data.explanation_short || "No short explanation available."}</p>
                </div>

                <div class="split-grid">
                    <div class="subcard">
                        <h3>Explanation</h3>
                        <p>${data.explanation || "No explanation available."}</p>
                    </div>

                    <div class="subcard">
                        <h3>Performance Signals</h3>
                        <div class="pill-row">
                            <span class="pill">Driver Form: ${signals.driver_form ?? "N/A"}</span>
                            <span class="pill">Constructor Momentum: ${signals.constructor_momentum ?? "N/A"}</span>
                            <span class="pill">Grid Positioning: ${signals.grid_positioning ?? "N/A"}</span>
                            <span class="pill">Consistency: ${signals.consistency ?? "N/A"}</span>
                        </div>
                    </div>
                </div>

                <div class="split-grid">
                    <div class="subcard">
                        <h3>Key Supporting Factors</h3>
                        ${renderList(data.main_support)}
                    </div>

                    <div class="subcard">
                        <h3>Primary Risk Factors</h3>
                        ${renderList(data.main_risks)}
                    </div>
                </div>

                ${data.contradiction_note ? `
                    <div class="subcard">
                        <h3>Contradiction Insight</h3>
                        <p>${data.contradiction_note}</p>
                    </div>
                ` : ""}

                <details id="technical">
                    <summary>Model Breakdown</summary>
                    <div class="details-content">
                        <div class="tech-grid">
                            <div class="tech-item">
                                <strong>Decision Threshold</strong>
                                <span>${data.decision_threshold ?? "N/A"}</span>
                            </div>
                            <div class="tech-item">
                                <strong>Grid Position</strong>
                                <span>${facts.grid ?? "N/A"}</span>
                            </div>
                            <div class="tech-item">
                                <strong>Qualifying Position</strong>
                                <span>${facts.qual_position ?? "N/A"}</span>
                            </div>
                            <div class="tech-item">
                                <strong>Driver Points (Last 3)</strong>
                                <span>${fmtNum(facts.driver_points_last3)}</span>
                            </div>
                            <div class="tech-item">
                                <strong>Constructor Points (Last 3)</strong>
                                <span>${fmtNum(facts.constructor_points_last3)}</span>
                            </div>
                            <div class="tech-item">
                                <strong>Driver Podiums (Last 3)</strong>
                                <span>${fmtNum(facts.driver_podiums_last3)}</span>
                            </div>
                            <div class="tech-item">
                                <strong>Driver Avg Finish (Last 3)</strong>
                                <span>${fmtNum(facts.driver_finishpos_last3)}</span>
                            </div>
                            <div class="tech-item">
                                <strong>Driver Track Avg Finish</strong>
                                <span>${fmtNum(facts.driver_track_avg_finish)}</span>
                            </div>
                            <div class="tech-item">
                                <strong>Driver Track Podium Rate</strong>
                                <span>${fmtNum(facts.driver_track_podium_rate)}</span>
                            </div>
                            <div class="tech-item">
                                <strong>Constructor Track Avg Finish</strong>
                                <span>${fmtNum(facts.constructor_track_avg_finish)}</span>
                            </div>
                            <div class="tech-item">
                                <strong>Constructor Track Podium Rate</strong>
                                <span>${fmtNum(facts.constructor_track_podium_rate)}</span>
                            </div>
                        </div>

                        <pre>${JSON.stringify(facts, null, 2)}</pre>
                    </div>
                </details>
            `;
        }

        function resetUI() {
            form.reset();
            resultBox.className = "empty-state";
            resultBox.innerHTML = `
                <div>
                    <h3>Prediction Ready</h3>
                    <p>
                        Once you submit a race scenario, the system will display the decision,
                        probability breakdown, supporting signals, risk signals, and technical evidence.
                    </p>
                </div>
            `;
            setStatus("Ready. Select race inputs and generate a prediction.", "ready");
        }

        async function loadMetadata() {
            setStatus("Loading available circuits and drivers...");
            try {
                const response = await fetch("/metadata");
                const data = await response.json();

                if (!data.drivers || !data.circuits) {
                    setStatus("No metadata available.", "error");
                    return;
                }

                setOptions(document.getElementById("circuitId"), data.circuits);
                setOptions(document.getElementById("driverId"), data.drivers);

                setStatus("Ready. Select race inputs and generate a prediction.", "ready");
            } catch (error) {
                setStatus(`Failed to load metadata: ${error.message}`, "error");
            }
        }

        form.addEventListener("submit", async (event) => {
            event.preventDefault();

            const circuitId = document.getElementById("circuitId").value;
            const driverId = document.getElementById("driverId").value;
            const qual_position = document.getElementById("qual_position").value;
            const grid = document.getElementById("grid").value;

            setStatus("Generating prediction...");
            resultBox.className = "empty-state";
            resultBox.innerHTML = `
                <div>
                    <h3>Running Prediction</h3>
                    <p>The system is processing the selected race scenario.</p>
                </div>
            `;

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

                renderPrediction(data);
                setStatus("Prediction completed successfully.", "ready");
            } catch (error) {
                resultBox.className = "empty-state";
                resultBox.innerHTML = `
                    <div>
                        <h3 class="error-text">Prediction Failed</h3>
                        <p class="error-text">${error.message}</p>
                    </div>
                `;
                setStatus("Prediction failed. Please review the input values.", "error");
            }
        });

        resetBtn.addEventListener("click", resetUI);

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
        confidence = "High Confidence"
    elif podium_proba >= 0.55 or podium_proba <= 0.30:
        confidence = "Moderate Confidence"
    else:
        confidence = "Low Confidence"

    # adjust confidence based on disagreement
    if raw_proba is not None and context_adjustment is not None:
        if abs(context_adjustment) > 0.25:
            confidence = "Low Confidence (Model and contextual factors are in conflict)"
        elif abs(context_adjustment) > 0.15:
            confidence = "Moderate Confidence (Some conflict between model and context)"

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
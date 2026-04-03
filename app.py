from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import joblib
from pathlib import Path
from feature_builder import build_2025_lineup_map, build_feature_row, AVG_FINISH_FALLBACK

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

RAW_PATH = DATA_DIR / "jolpica_podium_dataset_2022_2025.csv"
df_raw = pd.read_csv(RAW_PATH)

print("Loaded dataset from:", DF_PATH)
print("Loaded model from:", MODEL_PATH)
print("Running app file:", __file__)

app = Flask(__name__)
CORS(app)

# -------------------------
# Simple frontend
# -------------------------
INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>F1 Podium Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&display=swap" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg:        #15151E;
      --bg-raised: #1E1E2E;
      --bg-card:   #22222F;
      --bg-input:  #2A2A3A;
      --border:    #2E2E3E;
      --border-hi: #3E3E55;
      --red:       #E8002D;
      --red-dim:   rgba(232,0,45,0.12);
      --red-dark:  #B0001F;
      --g1:        #CCCCCC;
      --white:     #FFFFFF;
      --g2:        #888899;
      --g3:        #44445A;
      --ghost:     rgba(255,255,255,0.035);
      --font:      'Titillium Web', sans-serif;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--white); font-family: var(--font); min-height: 100vh; overflow-x: hidden; }

    /* NAV */
    nav {
      position: sticky; top: 0; z-index: 200;
      background: var(--bg); border-bottom: 1px solid var(--border);
      height: 56px; display: flex; align-items: center; padding: 0 32px; gap: 32px;
    }
    .f1-badge {
      background: var(--red); color: white;
      padding: 5px 11px 5px 9px;
      clip-path: polygon(6px 0%,100% 0%,calc(100% - 6px) 100%,0% 100%);
      font-size: 14px; font-weight: 900; letter-spacing: 0.05em;
    }
    .nav-title {
      font-size: 11px; font-weight: 700;
      letter-spacing: 0.25em; text-transform: uppercase;
      color: var(--g2); margin-left: 10px;
    }
    .nav-links { margin-left: auto; display: flex; }
    .nav-links a {
      font-size: 11px; font-weight: 700;
      letter-spacing: 0.22em; text-transform: uppercase;
      color: var(--g2); text-decoration: none;
      padding: 0 16px; height: 56px;
      display: flex; align-items: center;
      border-bottom: 3px solid transparent;
      transition: color .15s, border-color .15s;
    }
    .nav-links a:hover { color: var(--white); }
    .nav-links a.active { color: var(--white); border-bottom-color: var(--red); }

    /* HERO — matches image3 "RACE WEEKEND" dark card */
    .hero {
      position: relative;
      background: var(--bg);
      padding: 52px 40px 44px;
      border-bottom: 1px solid var(--border);
      overflow: hidden;
    }
    .hero::before {
      content: '';
      position: absolute; right: -80px; bottom: -120px;
      width: 560px; height: 560px; border-radius: 50%;
      background: radial-gradient(circle, rgba(176,0,31,0.35) 0%, transparent 70%);
      pointer-events: none;
    }
    .hero::after {
      content: 'PREDICTOR';
      position: absolute; right: -20px; bottom: -28px;
      font-size: 130px; font-weight: 900;
      color: var(--ghost); letter-spacing: -0.03em;
      line-height: 1; pointer-events: none; user-select: none; white-space: nowrap;
    }
    .hero-eyebrow {
      font-size: 10px; font-weight: 700;
      letter-spacing: 0.3em; text-transform: uppercase;
      color: var(--red); margin-bottom: 14px;
    }
    .hero-title {
      font-size: clamp(36px,6vw,72px); font-weight: 900;
      letter-spacing: -0.02em; line-height: 0.95;
      text-transform: uppercase; margin-bottom: 14px;
    }
    .hero-subtitle {
      font-size: 11px; font-weight: 700;
      letter-spacing: 0.2em; text-transform: uppercase;
      color: var(--g2); margin-bottom: 36px;
    }
    .hero-stats { display: flex; gap: 0; }
    .hero-stat {
      padding-right: 28px; margin-right: 28px;
      border-right: 1px solid var(--border-hi);
    }
    .hero-stat:last-child { border-right: none; }
    .hero-stat-val { font-size: 26px; font-weight: 900; line-height: 1; }
    .hero-stat-val em { color: var(--red); font-style: normal; }
    .hero-stat-label {
      font-size: 9px; font-weight: 700;
      letter-spacing: 0.2em; text-transform: uppercase;
      color: var(--g2); margin-top: 5px;
    }

    /* LAYOUT */
    .layout { display: grid; grid-template-columns: 360px 1fr; min-height: calc(100vh - 56px - 160px); }

    /* SIDEBAR */
    .sidebar {
      background: var(--bg-raised);
      border-right: 1px solid var(--border);
      padding: 32px 28px;
    }
    .sec-tag {
      font-size: 9px; font-weight: 700;
      letter-spacing: 0.35em; text-transform: uppercase;
      color: var(--g3); margin-bottom: 22px;
      display: flex; align-items: center; gap: 10px;
    }
    .sec-tag::after { content: ''; flex: 1; height: 1px; background: var(--border); }

    .field { margin-bottom: 18px; }
    .field label {
      display: block; font-size: 9px; font-weight: 700;
      letter-spacing: 0.25em; text-transform: uppercase;
      color: var(--g2); margin-bottom: 7px;
    }
    .field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }

    .f1-input {
      width: 100%; background: var(--bg-input);
      border: 1px solid var(--border-hi);
      color: var(--white); font-family: var(--font);
      font-size: 13px; font-weight: 600;
      padding: 11px 14px; outline: none;
      transition: border-color .15s, background .15s;
      appearance: none; -webkit-appearance: none; border-radius: 0;
    }
    .f1-input:focus { border-color: var(--red); background: #1f1424; }
    select.f1-input {
      cursor: pointer;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23888899'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 12px center;
      padding-right: 32px;
    }
    select.f1-input option { background: #22222F; }

    /* CTA — angled F1 button from image1 "MERCHANDISE" */
    .cta-btn {
      width: 100%; background: var(--red); color: white;
      border: none; font-family: var(--font);
      font-size: 11px; font-weight: 700;
      letter-spacing: 0.3em; text-transform: uppercase;
      padding: 15px 24px; cursor: pointer; margin-top: 8px;
      clip-path: polygon(10px 0%,100% 0%,calc(100% - 10px) 100%,0% 100%);
      position: relative; overflow: hidden;
      transition: background .15s;
    }
    .cta-btn::after {
      content: ''; position: absolute; inset: 0;
      background: linear-gradient(90deg,transparent,rgba(255,255,255,0.12),transparent);
      transform: translateX(-100%); transition: transform .5s;
    }
    .cta-btn:hover { background: #ff0033; }
    .cta-btn:hover::after { transform: translateX(100%); }
    .cta-btn:active { transform: scale(0.98); }
    .cta-btn:disabled { opacity: .35; cursor: not-allowed; }

    /* Mini stat cards — image1 bottom-left info card */
    .mini-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--border); margin-top: 28px; }
    .mini-card { background: var(--bg-card); padding: 14px 16px; }
    .mini-card-val { font-size: 20px; font-weight: 900; color: var(--red); line-height: 1; }
    .mini-card-label { font-size: 9px; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; color: var(--g2); margin-top: 5px; }

    /* MAIN */
    .main { background: var(--bg); padding: 32px; }

    /* Idle */
    .idle-state {
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      min-height: 420px; color: var(--g3); text-align: center; gap: 16px;
    }
    .idle-icon {
      width: 64px; height: 64px; border: 2px solid var(--border-hi);
      border-radius: 50%; display: flex; align-items: center; justify-content: center;
      font-size: 24px; color: var(--red);
    }
    .idle-state p { font-size: 11px; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; line-height: 1.8; }

    /* Loading */
    .loading-state { display: none; flex-direction: column; align-items: center; justify-content: center; min-height: 420px; gap: 18px; }
    .loading-state.show { display: flex; }
    .race-loader { width: 180px; height: 3px; background: var(--border); position: relative; overflow: hidden; }
    .race-loader-bar {
      position: absolute; height: 100%; width: 70px;
      background: linear-gradient(90deg,transparent,var(--red),var(--red),transparent);
      animation: carRace .9s linear infinite;
    }
    @keyframes carRace { 0%{left:-70px} 100%{left:180px} }
    .loading-label { font-size: 10px; font-weight: 700; letter-spacing: 0.3em; text-transform: uppercase; color: var(--g2); }

    /* Error */
    .error-state { display: none; background: var(--red-dim); border-left: 3px solid var(--red); padding: 18px 20px; margin-bottom: 24px; gap: 12px; align-items: flex-start; }
    .error-state.show { display: flex; }
    .error-icon { color: var(--red); font-size: 18px; flex-shrink: 0; }
    .error-body { font-size: 13px; color: #ff6b6b; line-height: 1.5; }
    .error-body strong { display: block; font-size: 10px; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 4px; }

    /* Results */
    .results { display: none; }
    .results.show { display: block; }

    /* Decision hero — image3 top "RACE WEEKEND" large heading card */
    .decision-hero {
      position: relative; background: var(--bg-raised);
      border: 1px solid var(--border);
      padding: 32px 32px 24px; margin-bottom: 1px; overflow: hidden;
    }
    .decision-hero::before {
      content: ''; position: absolute;
      left: 0; top: 0; bottom: 0; width: 4px; background: var(--red);
    }
    /* Ghost numeral like image2 large rank numbers */
    .decision-hero::after {
      content: attr(data-ghost);
      position: absolute; right: 20px; bottom: -20px;
      font-size: 160px; font-weight: 900;
      color: var(--ghost); letter-spacing: -0.04em;
      line-height: 1; pointer-events: none; user-select: none;
    }
    .dh-eye { font-size: 9px; font-weight: 700; letter-spacing: 0.3em; text-transform: uppercase; color: var(--red); margin-bottom: 10px; }
    .dh-dec {
      font-size: clamp(24px,4vw,40px); font-weight: 900;
      text-transform: uppercase; letter-spacing: -0.01em;
      line-height: 1; margin-bottom: 14px;
    }
    .driver-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }
    .driver-pill {
      display: flex; align-items: center; gap: 8px;
      background: var(--bg-input); border: 1px solid var(--border-hi);
      padding: 6px 12px; font-size: 12px; font-weight: 700; letter-spacing: 0.08em;
    }
    .dp-code { font-size: 13px; font-weight: 900; letter-spacing: 0.12em; }
    .dp-sep { color: var(--g3); }
    .dp-team { color: var(--g2); font-weight: 600; }

    .dh-badges { display: flex; gap: 8px; flex-wrap: wrap; }
    .badge { font-size: 9px; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; padding: 4px 10px; border: 1px solid; }
    .b-red   { color: var(--red);   border-color: var(--red);   background: var(--red-dim); }
    .b-green { color: #4ade80; border-color: #4ade80; background: rgba(74,222,128,0.08); }
    .b-amber { color: #fbbf24; border-color: #fbbf24; background: rgba(251,191,36,0.08); }
    .b-grey  { color: var(--g2);  border-color: var(--border-hi); }

    /* Probability bar row */
    .prob-row {
      background: var(--bg-raised); border: 1px solid var(--border); border-top: none;
      padding: 20px 32px; display: flex; align-items: center; gap: 28px; margin-bottom: 1px;
    }
    .prob-num { font-size: 48px; font-weight: 900; line-height: 1; width: 100px; flex-shrink: 0; }
    .prob-num small { font-size: 22px; font-weight: 700; color: var(--g2); }
    .prob-track-wrap { flex: 1; }
    .prob-track-label { display: flex; justify-content: space-between; font-size: 9px; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; color: var(--g2); margin-bottom: 8px; }
    .prob-track { height: 4px; background: var(--border-hi); position: relative; }
    .prob-fill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--red); transition: width 1.2s cubic-bezier(0.4,0,0.2,1); width: 0; }
    .prob-thresh { position: absolute; left: 30%; top: -5px; bottom: -5px; width: 2px; background: rgba(251,191,36,0.7); }
    .prob-thresh::before { content: '30%'; position: absolute; top: -18px; left: 50%; transform: translateX(-50%); font-size: 8px; font-weight: 700; color: #fbbf24; white-space: nowrap; letter-spacing: 0.1em; }

    /* Adj row — image3 session list columns */
    .adj-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 1px; background: var(--border); margin-bottom: 1px; }
    .adj-cell { background: var(--bg-raised); padding: 16px 24px; }
    .adj-label { font-size: 9px; font-weight: 700; letter-spacing: 0.25em; text-transform: uppercase; color: var(--g2); margin-bottom: 6px; }
    .adj-val { font-size: 22px; font-weight: 900; }
    .adj-pos { color: #4ade80; }
    .adj-neg { color: var(--red); }

    /* Signals — image2 team row layout */
    .signals-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1px; background: var(--border); margin-bottom: 20px; }
    .signal-cell {
      background: var(--bg-raised); padding: 18px 16px; text-align: center;
      position: relative; overflow: hidden;
    }
    /* ghost rank numeral inside signal — image2 large position numbers */
    .signal-cell::after {
      content: attr(data-rank);
      position: absolute; right: 6px; bottom: -8px;
      font-size: 56px; font-weight: 900; color: var(--ghost);
      line-height: 1; pointer-events: none;
    }
    .sig-label { font-size: 9px; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; color: var(--g2); margin-bottom: 10px; }
    .sig-icon  { font-size: 22px; margin-bottom: 6px; }
    .sig-val   { font-size: 12px; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; }
    .c-green { color: #4ade80; } .c-amber { color: #fbbf24; } .c-red { color: var(--red); } .c-blue { color: #60a5fa; } .c-grey { color: var(--g2); }

    /* Contradiction strip */
    .contraband { display: none; background: var(--red-dim); border-left: 3px solid var(--red); padding: 12px 18px; margin-bottom: 20px; gap: 10px; align-items: flex-start; }
    .contraband.show { display: flex; }
    .contra-icon { color: var(--red); font-size: 14px; flex-shrink: 0; margin-top: 2px; }
    .contra-text { font-size: 13px; color: #ff8888; line-height: 1.5; }

    /* Two-col factors */
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
    .panel { background: var(--bg-raised); border: 1px solid var(--border); }
    .panel-head { padding: 12px 18px; border-bottom: 1px solid var(--border); font-size: 9px; font-weight: 700; letter-spacing: 0.28em; text-transform: uppercase; color: var(--g2); display: flex; align-items: center; gap: 8px; }
    .dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
    .dot-g { background: #4ade80; } .dot-r { background: var(--red); } .dot-b { background: #60a5fa; }
    .panel-body { padding: 14px 18px; }

    /* Factor list — image3 schedule row style */
    .factor-list { list-style: none; }
    .factor-list li { display: grid; grid-template-columns: 18px 1fr; gap: 10px; padding: 9px 0; font-size: 13px; color: var(--g1); border-bottom: 1px solid var(--border); align-items: baseline; line-height: 1.4; }
    .factor-list li:last-child { border-bottom: none; }
    .fi { font-size: 11px; font-weight: 700; }
    .fi-g { color: #4ade80; } .fi-r { color: var(--red); }

    /* Evidence table */
    .ev-panel { background: var(--bg-raised); border: 1px solid var(--border); margin-bottom: 12px; }
    .ev-table { width: 100%; border-collapse: collapse; }
    .ev-table tr { border-bottom: 1px solid var(--border); }
    .ev-table tr:last-child { border-bottom: none; }
    .ev-table td { padding: 11px 18px; font-size: 13px; }
    .ev-table td:first-child { font-size: 10px; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; color: var(--g2); width: 55%; }
    .ev-table td:last-child { font-weight: 700; text-align: right; font-size: 14px; }
    .ev-g { color: #4ade80; } .ev-w { color: #fbbf24; } .ev-r { color: var(--red); }

    /* Explanation */
    .xp-panel { background: var(--bg-raised); border: 1px solid var(--border); margin-bottom: 24px; }
    .xp-body { padding: 18px; font-size: 14px; line-height: 1.75; color: var(--g1); font-weight: 300; }
    .xp-body strong { color: var(--white); font-weight: 700; }

    /* Responsive */
    @media (max-width: 860px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { border-right: none; border-bottom: 1px solid var(--border); }
      .signals-grid { grid-template-columns: 1fr 1fr; }
      .two-col { grid-template-columns: 1fr; }
      nav { padding: 0 16px; }
      .nav-links a { padding: 0 10px; }
      .hero { padding: 36px 20px; }
      .hero::after { display: none; }
      .main, .sidebar { padding: 20px; }
      .prob-row { flex-direction: column; gap: 14px; }
      .prob-num { width: auto; font-size: 40px; }
    }
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 4px; }

    /* Decision Thresholds Table */
    .threshold-section {
      margin-top: 30px;
      padding: 32px 28px;
      background: var(--bg-raised);
      border-top: 1px solid var(--border);
    }
    .threshold-section .section-title {
      font-size: clamp(24px, 3vw, 32px);
      font-weight: 900;
      letter-spacing: -0.02em;
      margin-bottom: 8px;
      text-transform: uppercase;
    }
    .threshold-section .section-subtitle {
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      color: var(--g2);
      margin-bottom: 22px;
    }
    .threshold-table {
      margin-top: 16px;
      border-radius: 0;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.04);
    }
    .threshold-row {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr 2fr;
      padding: 16px 18px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      font-size: 14px;
      color: var(--white);
    }
    .threshold-row:last-child {
      border-bottom: none;
    }
    .threshold-row.header {
      background: rgba(255,255,255,0.08);
      font-weight: 800;
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.6px;
      color: #e10600;
      border-left: 4px solid #e10600;
    }
    .threshold-row div {
      display: flex;
      align-items: center;
    }
    
    /* Mini Bar Charts */
    .form-charts {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-top: 20px;
      margin-bottom: 20px;
    }
    .chart-box {
      padding: 16px;
      background: rgba(255,255,255,0.05);
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.1);
    }
    .chart-box h3 {
      margin: 0 0 14px;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.6px;
      text-transform: uppercase;
      color: #e10600;
    }
    .chart-box canvas {
      width: 100% !important;
      height: 120px !important;
    }
  </style>
</head>
<body>

<nav>
  <span class="f1-badge">F1</span>
  <span class="nav-title">Podium Predictor</span>
  <div class="nav-links">
    <a href="#" class="active">Predict</a>
    <a href="#">2025 Season</a>
    <a href="#">Model</a>
  </div>
</nav>

<div class="hero">
  <div class="hero-eyebrow">Machine Learning · 2025 Season</div>
  <h1 class="hero-title">F1 Podium<br>Predictor</h1>
  <p class="hero-subtitle">Formula 1 Grand Prix · Race Outcome Prediction</p>
  <div class="hero-stats">
    <div class="hero-stat">
      <div class="hero-stat-val"><em>0.947</em></div>
      <div class="hero-stat-label">ROC AUC Score</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val"><em>72.5</em>%</div>
      <div class="hero-stat-label">F1 Score</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val">1,838</div>
      <div class="hero-stat-label">Training Rows</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val">19</div>
      <div class="hero-stat-label">Model Features</div>
    </div>
  </div>
</div>

<div class="layout">
  <aside class="sidebar">
    <div class="sec-tag">Race Inputs</div>
    <form id="pform">
      <div class="field">
        <label>Circuit</label>
        <select class="f1-input" id="circuitId" required>
          <option value="" disabled selected>Loading circuits…</option>
        </select>
      </div>
      <div class="field">
        <label>Driver</label>
        <select class="f1-input" id="driverId" required>
          <option value="" disabled selected>Loading drivers…</option>
        </select>
      </div>
      <div class="field">
        <div class="field-row">
          <div>
            <label>Qualifying P</label>
            <input class="f1-input" type="number" id="qual_position" min="1" max="20" placeholder="1–20" required />
          </div>
          <div>
            <label>Grid P</label>
            <input class="f1-input" type="number" id="grid" min="1" max="20" placeholder="1–20" required />
          </div>
        </div>
      </div>
      <button type="submit" class="cta-btn" id="submit-btn">Run Prediction →</button>
    </form>
    <div class="mini-cards">
      <div class="mini-card"><div class="mini-card-val" id="stat-drivers">—</div><div class="mini-card-label">2025 Drivers</div></div>
      <div class="mini-card"><div class="mini-card-val" id="stat-circuits">—</div><div class="mini-card-label">Circuits</div></div>
      <div class="mini-card"><div class="mini-card-val">30%</div><div class="mini-card-label">Threshold</div></div>
      <div class="mini-card"><div class="mini-card-val">GB</div><div class="mini-card-label">Model</div></div>
    </div>
  </aside>

  <main class="main">
    <div class="idle-state" id="s-idle">
      <div class="idle-icon">◈</div>
      <p>Configure race inputs<br>and run a prediction</p>
    </div>

    <div class="loading-state" id="s-loading">
      <div class="race-loader"><div class="race-loader-bar"></div></div>
      <div class="loading-label">Analysing telemetry…</div>
    </div>

    <div class="error-state" id="s-error">
      <div class="error-icon">⚠</div>
      <div class="error-body"><strong>Prediction Failed</strong><span id="err-msg"></span></div>
    </div>

    <div class="results" id="s-results">

      <div class="decision-hero" id="dh" data-ghost="">
        <div class="dh-eye">Prediction Result</div>
        <div class="dh-dec" id="r-decision">—</div>
        <div class="driver-row">
          <div class="driver-pill"><span class="dp-code" id="r-code">—</span><span class="dp-sep">·</span><span class="dp-team" id="r-team">—</span></div>
          <div class="driver-pill"><span class="dp-team" id="r-circuit">—</span></div>
        </div>
        <div class="dh-badges">
          <span class="badge" id="r-conf-badge">—</span>
          <span class="badge b-grey">Threshold 30%</span>
        </div>
      </div>

      <div class="prob-row">
        <div class="prob-num" id="r-pct">—<small>%</small></div>
        <div class="prob-track-wrap">
          <div class="prob-track-label"><span>Podium Probability</span><span id="r-pct2">0%</span></div>
          <div class="prob-track">
            <div class="prob-fill" id="r-bar"></div>
            <div class="prob-thresh"></div>
          </div>
        </div>
      </div>

      <div class="adj-grid">
        <div class="adj-cell"><div class="adj-label">Model Raw</div><div class="adj-val" id="r-raw">—</div></div>
        <div class="adj-cell"><div class="adj-label">Context Δ</div><div class="adj-val" id="r-delta">—</div></div>
        <div class="adj-cell"><div class="adj-label">Adjusted</div><div class="adj-val" id="r-adj">—</div></div>
      </div>

      <div class="signals-grid">
        <div class="signal-cell" data-rank="1"><div class="sig-label">Driver Form</div><div class="sig-icon" id="r-si-form">—</div><div class="sig-val" id="r-sv-form">—</div></div>
        <div class="signal-cell" data-rank="2"><div class="sig-label">Constructor</div><div class="sig-icon" id="r-si-cons">—</div><div class="sig-val" id="r-sv-cons">—</div></div>
        <div class="signal-cell" data-rank="3"><div class="sig-label">Grid Position</div><div class="sig-icon" id="r-si-grid">—</div><div class="sig-val" id="r-sv-grid">—</div></div>
        <div class="signal-cell" data-rank="4"><div class="sig-label">Consistency</div><div class="sig-icon" id="r-si-cns2">—</div><div class="sig-val" id="r-sv-cns2">—</div></div>
      </div>

      <div class="two-col">
        <div class="panel">
          <div class="panel-head"><div class="dot dot-b"></div>Driver Form Trend</div>
          <div class="panel-body" style="height:200px;">
            <canvas id="driverChart"></canvas>
          </div>
        </div>
        <div class="panel">
          <div class="panel-head"><div class="dot dot-b"></div>Constructor Form Trend</div>
          <div class="panel-body" style="height:200px;">
            <canvas id="constructorChart"></canvas>
          </div>
        </div>
      </div>

      <div class="contraband" id="r-contra">
        <div class="contra-icon">!</div>
        <div class="contra-text" id="r-contra-text"></div>
      </div>

      <div class="two-col">
        <div class="panel">
          <div class="panel-head"><div class="dot dot-g"></div>Positive Factors</div>
          <div class="panel-body"><ul class="factor-list" id="r-pos"></ul></div>
        </div>
        <div class="panel">
          <div class="panel-head"><div class="dot dot-r"></div>Risk Factors</div>
          <div class="panel-body"><ul class="factor-list" id="r-risk"></ul></div>
        </div>
      </div>

      <div class="ev-panel">
        <div class="panel-head"><div class="dot dot-b"></div>Model Evidence</div>
        <table class="ev-table"><tbody id="r-ev"></tbody></table>
      </div>

      <div class="xp-panel">
        <div class="panel-head"><div class="dot dot-b"></div>Model Explanation</div>
        <div class="xp-body" id="r-explain">—</div>
      </div>

    </div>
  </main>
</div>

<div class="threshold-section">
  <h2 class="section-title">Decision Thresholds</h2>
  <p class="section-subtitle">
    The model categorizes podium potential based on probability ranges.
  </p>

  <div class="threshold-table">
    <div class="threshold-row header">
      <div>Label</div>
      <div>Probability Range</div>
      <div>Interpretation</div>
    </div>

    <div class="threshold-row">
      <div>Very Likely Chance of a Podium Finish</div>
      <div>≥ 70%</div>
      <div>Strong likelihood based on performance and context</div>
    </div>

    <div class="threshold-row">
      <div>Likely Chance of a Podium Finish</div>
      <div>50% – 69.9%</div>
      <div>Balanced scenario — outcome could go either way</div>
    </div>

    <div class="threshold-row">
      <div>Moderate Chance of a Podium Finish</div>
      <div>30% – 49.9%</div>
      <div>Possible but requires favourable conditions</div>
    </div>

    <div class="threshold-row">
      <div>Unlikely Chance of a Podium Finish</div>
      <div>&lt; 30%</div>
      <div>Weak indicators for podium finish</div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let driverChartInstance = null;
let constructorChartInstance = null;

const ICON = {
  STRONG:'🔥', HIGH:'▲', FRONT:'🏁',
  MODERATE:'⚡', MEDIUM:'◆', MIDFIELD:'→',
  WEAK:'↓', LOW:'▽', BACK:'⬇',
  UNKNOWN:'?', LIMITED:'↓'
};

const COL = {
  STRONG:'c-green', HIGH:'c-green', FRONT:'c-green',
  MODERATE:'c-amber', MEDIUM:'c-amber', MIDFIELD:'c-blue',
  WEAK:'c-red', LOW:'c-red', BACK:'c-red',
  UNKNOWN:'c-grey', LIMITED:'c-red'
};

const DRIVER_NAME_MAP = {
  "max_verstappen": "Max Verstappen",
  "max_ve": "Max Verstappen",
  "lewis_hamilton": "Lewis Hamilton",
  "hamilton": "Lewis Hamilton",
  "charles_leclerc": "Charles Leclerc",
  "leclerc": "Charles Leclerc",
  "carlos_sainz": "Carlos Sainz",
  "sainz": "Carlos Sainz",
  "lando_norris": "Lando Norris",
  "norris": "Lando Norris",
  "george_russell": "George Russell",
  "russell": "George Russell"
};

function ui(s) {
  if (s === 'idle') destroyCharts();
  $('s-idle').style.display = s==='idle' ? 'flex' : 'none';
  $('s-loading').classList.toggle('show', s==='loading');
  $('s-error').classList.toggle('show', s==='error');
  $('s-results').classList.toggle('show', s==='results');
}

function formatName(value) {
  if (!value) return "N/A";
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, c => c.toUpperCase());
}

function formatDriver(value) {
  if (!value) return "N/A";
  return DRIVER_NAME_MAP[value] || formatName(value);
}

function destroyCharts() {
  if (driverChartInstance) {
    driverChartInstance.destroy();
    driverChartInstance = null;
  }
  if (constructorChartInstance) {
    constructorChartInstance.destroy();
    constructorChartInstance = null;
  }
}

function renderFormCharts(d) {
  destroyCharts();

  const driverCanvas = document.getElementById('driverChart');
  const constructorCanvas = document.getElementById('constructorChart');

  if (!driverCanvas || !constructorCanvas) return;

  const driverTrend = d.driver_points_trend || [];
  const constructorTrend = d.constructor_points_trend || [];
  const labels = d.trend_labels || ['Race -3', 'Race -2', 'Race -1'];

  driverChartInstance = new Chart(driverCanvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Driver Points',
        data: driverTrend,
        tension: 0.35,
        borderColor: '#E8002D',
        backgroundColor: 'rgba(232,0,45,0.15)',
        fill: true,
        pointRadius: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#888899' }, grid: { color: '#2E2E3E' } },
        y: { ticks: { color: '#888899' }, grid: { color: '#2E2E3E' }, beginAtZero: true }
      }
    }
  });

  constructorChartInstance = new Chart(constructorCanvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Constructor Points',
        data: constructorTrend,
        tension: 0.35,
        borderColor: '#FFFFFF',
        backgroundColor: 'rgba(255,255,255,0.10)',
        fill: true,
        pointRadius: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#888899' }, grid: { color: '#2E2E3E' } },
        y: { ticks: { color: '#888899' }, grid: { color: '#2E2E3E' }, beginAtZero: true }
      }
    }
  });
}

function setSig(iconId, valId, raw) {
  const v = (raw||'UNKNOWN').toUpperCase();
  $(iconId).textContent = ICON[v]||'?';
  $(valId).textContent  = v;
  $(valId).className    = 'sig-val '+(COL[v]||'c-grey');
}

function render(d) {
  const pct = (d.probability * 100).toFixed(1);
  const pctNum = parseFloat(pct);
  $('dh').dataset.ghost = pct+'%';
  $('r-decision').textContent = d.decision||'—';
  $('r-code').textContent     = formatDriver(d.request?.driverId);
  $('r-team').textContent     = formatName(d.resolved_constructor);
  $('r-circuit').textContent  = formatName(d.request?.circuitId);

  const cb = $('r-conf-badge');
  cb.textContent = 'Confidence: '+(d.confidence_level||'—');
  const cl = (d.confidence_level||'').toLowerCase();
  cb.className = 'badge '+(cl.startsWith('high')?'b-green':cl.startsWith('moderate')?'b-amber':'b-red');

  $('r-pct').innerHTML = pct+'<small>%</small>';
  $('r-pct2').textContent = pct+'%';
  setTimeout(()=>{ $('r-bar').style.width = pct+'%'; }, 80);

  const rp = ((d.raw_probability ?? d.probability) * 100).toFixed(1);
  const ap = ((d.adjusted_probability ?? d.probability) * 100).toFixed(1);
  const dl = ((d.context_adjustment ?? 0) * 100).toFixed(1);
  $('r-raw').textContent = rp+'%';
  $('r-adj').textContent = ap+'%';
  const de = $('r-delta');
  de.textContent = ((parseFloat(dl) >= 0 ? '+' : '') + dl + '%');
  de.className = 'adj-val ' + (parseFloat(dl) >= 0 ? 'adj-pos' : 'adj-neg');

  const sg = d.signals||{};
  setSig('r-si-form','r-sv-form',sg.driver_form);
  setSig('r-si-cons','r-sv-cons',sg.constructor_momentum);
  setSig('r-si-grid','r-sv-grid',sg.grid_positioning);
  setSig('r-si-cns2','r-sv-cns2',sg.consistency);
  
  renderFormCharts(d);

  if (d.contradiction_note) { $('r-contra').classList.add('show'); $('r-contra-text').textContent=d.contradiction_note; }
  else { $('r-contra').classList.remove('show'); }

  const pf=$('r-pos'); pf.innerHTML='';
  (d.positive_factors||[]).forEach(f=>{ const li=document.createElement('li'); li.innerHTML=`<span class="fi fi-g">✓</span><span>${f[0].toUpperCase()+f.slice(1)}</span>`; pf.appendChild(li); });

  const rf=$('r-risk'); rf.innerHTML='';
  (d.risk_factors||[]).forEach(f=>{ const li=document.createElement('li'); li.innerHTML=`<span class="fi fi-r">✕</span><span>${f[0].toUpperCase()+f.slice(1)}</span>`; rf.appendChild(li); });

  const fts=d.facts||{};
  const rows=[
    ['Grid Position',fts.grid,v=>v<=3?'ev-g':v>=11?'ev-r':''],
    ['Qualifying Position',fts.qual_position,v=>v<=3?'ev-g':v>=11?'ev-r':''],
    ['Driver Points (L3)',(fts.driver_points_last3||0).toFixed(1),v=>parseFloat(v)>=12?'ev-g':parseFloat(v)<5?'ev-r':'ev-w'],
    ['Constructor Points (L3)',(fts.constructor_points_last3||0).toFixed(1),v=>parseFloat(v)>=10?'ev-g':parseFloat(v)<3?'ev-r':'ev-w'],
    ['Driver Podiums (L3)',(fts.driver_podiums_last3||0).toFixed(2),v=>parseFloat(v)>=0.33?'ev-g':''],
    ['Avg Finish Pos (L3)',(fts.driver_finishpos_last3||0).toFixed(1),v=>parseFloat(v)<=6?'ev-g':parseFloat(v)>10?'ev-r':'ev-w'],
    ['Track Podium Rate',((fts.driver_track_podium_rate||0)*100).toFixed(0)+'%',v=>parseFloat(v)>=30?'ev-g':''],
    ['Constr. Track Podium',((fts.constructor_track_podium_rate||0)*100).toFixed(0)+'%',v=>parseFloat(v)>=30?'ev-g':''],
  ];
  const tb=$('r-ev'); tb.innerHTML='';
  rows.forEach(([label,val,cls])=>{
    const tr=document.createElement('tr');
    const c=cls?cls(val):'';
    tr.innerHTML=`<td>${label}</td><td class="${c}">${val??'—'}</td>`;
    tb.appendChild(tr);
  });

  $('r-explain').innerHTML = d.explanation||d.summary||'—';
}

$('pform').addEventListener('submit', async e=>{
  e.preventDefault();
  const btn=$('submit-btn'); btn.disabled=true;
  ui('loading');
  const p=new URLSearchParams({circuitId:$('circuitId').value,driverId:$('driverId').value,qual_position:$('qual_position').value,grid:$('grid').value});
  try {
    const res=await fetch('/predict?'+p);
    const data=await res.json();
    if(!res.ok) throw new Error(data.error||'Prediction failed');
    render(data); ui('results');
  } catch(err) { $('err-msg').textContent=err.message; ui('error'); }
  finally { btn.disabled=false; }
});

async function loadMetadata() {
    statusBox.textContent = "Loading available circuits and drivers...";
    try {
        const response = await fetch("/metadata");
        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        const data = await response.json();

        if (!data.drivers?.length || !data.circuits?.length) {
            statusBox.innerHTML = '<span class="error">Metadata loaded but lists are empty. Check server logs.</span>';
            return;
        }

        setOptions(document.getElementById("circuitId"), data.circuits);
        setOptions(document.getElementById("driverId"), data.drivers);
        statusBox.textContent = "Ready. Select race inputs and run a prediction.";
    } catch (error) {
        statusBox.innerHTML = `<span class="error">Failed to load metadata: ${error.message}</span>`;
    }
}
      $('stat-drivers').textContent=data.drivers.length;

ui('idle');
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
        decision = "Very Likely Chance of a Podium Finish"
    elif podium_proba >= 0.50:
        decision = "Likely Chance of a Podium Finish"
    elif podium_proba >= 0.30:
        decision = "Moderate Chance of a Podium Finish"
    else:
        decision = "Unlikely Chance of a Podium Finish"

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
    # Fall back to all seasons if 2025 has no data yet
    drivers_2025 = df_feat[df_feat["season"] == 2025]["driverId"].dropna().unique().tolist()
    drivers = sorted(drivers_2025 if drivers_2025 else df_feat["driverId"].dropna().unique().tolist())
    circuits = sorted(df_feat["circuitId"].dropna().unique().tolist())
    return jsonify({"drivers": drivers, "circuits": circuits})

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
        grid=grid,
        df_raw=df_raw
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

    out["driver_points_trend"] = row_dict.get("driver_points_trend", [0.0, 0.0, 0.0])
    out["constructor_points_trend"] = row_dict.get("constructor_points_trend", [0.0, 0.0, 0.0])
    out["driver_finish_trend"] = row_dict.get("driver_finish_trend", [AVG_FINISH_FALLBACK] * 3)
    out["trend_labels"] = ["Race -3", "Race -2", "Race -1"]

    print("DEBUG: out keys before jsonify:", list(out.keys())[-5:])
    print("DEBUG: trend_labels value:", out.get("trend_labels"))
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
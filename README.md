# F1 Podium Predictor

A machine learning web application that predicts the probability of a Formula 1 driver achieving a podium finish, built with Flask and a Gradient Boosting classifier trained on data from the 2022–2025 regulation era.

---

## Project Overview

This project explores race outcome prediction in Formula 1 using supervised classification models. The objective is to estimate whether a driver will finish in the top three positions (podium) for a given race, presented through an interactive dark-themed web interface.

The workflow follows a complete data science pipeline:

1. Data source identification
2. Data collection via API
3. Data cleaning and preprocessing
4. Feature engineering
5. Model development and comparison
6. Model evaluation and selection
7. Flask web application deployment

---

## Data Source

Race data is retrieved from the **Jolpica Ergast-compatible API**, which provides structured historical Formula 1 data including:

- Race results
- Drivers and constructors
- Grid positions
- Qualifying positions
- Points and finishing positions

Data is restricted to the **2022–2025 regulation era** to reduce variability from major technical rule changes. No web scraping was required.

---

## Prediction Task

Binary classification problem:

- `1` → Driver finished on the podium (P1–P3)
- `0` → Driver did not finish on the podium

---

## Features Used

The model uses 19 engineered features across four categories:

**Driver Performance**
- Rolling points scored over last 3 races
- Recent podium frequency (last 3 races)
- Average finishing position (last 3 races)

**Constructor Performance**
- Constructor rolling points (last 3 races)
- Constructor podium frequency (last 3 races)

**Track-Specific**
- Driver historical average finish at circuit
- Driver podium rate at circuit
- Constructor average finish at circuit
- Constructor podium rate at circuit
- Driver–constructor synergy metrics

**Race Inputs**
- Grid position and its inverse (`grid_inverse`)
- Qualifying position
- Season and round

---

## Model Performance

Three models were trained and compared on a temporal split (train: 2022–2024, test: 2025):

| Model               | ROC AUC | Precision | Recall | F1 Score |
|---------------------|---------|-----------|--------|----------|
| Gradient Boosting   | 0.947   | 0.626     | 0.861  | 0.725    |
| Random Forest       | 0.946   | 0.685     | 0.847  | 0.758    |
| Logistic Regression | 0.937   | 0.746     | 0.736  | 0.741    |

**Final model: Gradient Boosting** — selected for its superior ROC AUC and recall, ensuring fewer missed podium predictions. Classification threshold set at **0.30**.

**Top predictive features by importance:**
1. Grid inverse (35.1%)
2. Qualifying position (18.5%)
3. Driver points last 3 races (7.8%)
4. Driver–constructor avg finish (7.5%)
5. Constructor points last 3 races (4.6%)

---

## Tech Stack

**Backend**
- Python, Flask (`render_template_string` single-file architecture)
- scikit-learn (Gradient Boosting classifier)
- pandas, numpy, joblib

**Frontend**
- HTML/CSS/JavaScript (embedded in Flask)
- Chart.js (form trend charts)
- Custom dark F1-themed UI with SVG circuit visualisations

**Data**
- Jolpica Ergast F1 API
- 1,838 driver–race observations (2022–2025)

---

## Project Structure

```
f1-podium-predictor/
├── app.py                              # Flask app (backend + frontend)
├── feature_builder.py                  # Feature engineering for predictions
├── podium_predictor.ipynb              # Full ML pipeline notebook
├── requirements.txt
├── data/
│   ├── jolpica_podium_dataset_2022_2025.csv   # Raw collected data
│   ├── clean_podium_dataset.csv               # Cleaned data
│   └── feature_engineered_dataset.csv         # Final modelling dataset
├── models/
│   └── final_gb_model.joblib           # Trained Gradient Boosting model
└── images/
    └── images.png                      # F1 logo asset
```

---

## Setup and Installation

### Prerequisites

- Python 3.9+

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the data pipeline (optional — pre-built datasets included)

```bash
jupyter notebook podium_predictor.ipynb
```

This notebook handles data collection, cleaning, feature engineering, model training, and saves the model to `models/final_gb_model.joblib`.

### Run the web application

```bash
python app.py
```

Then open your browser at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Using the Web App

1. Select a **circuit** from the dropdown
2. Select a **driver** from the 2025 lineup
3. Enter the **qualifying position** (1–20)
4. Enter the **grid position** (1–20)
5. Click **Run Prediction →**

The app returns:
- Podium probability score with animated progress bar
- Raw vs. context-adjusted probability
- Driver form, constructor momentum, grid position, and consistency signals
- Form trend charts (last 3 races)
- Positive factors and risk factors
- Model evidence table
- Plain-language prediction explanation

---

## Web App Sections

Beyond the prediction form, the app includes:

- **Grid Positions & Points Breakdown** — interactive starting grid visualiser with F1 points system
- **Understanding Formula 1** — race weekend, qualifying, tyres, DRS, flags, and championship explainers
- **Factors That Can Change Everything** — weather, safety car, reliability, pit stops, and other race variables the model cannot capture
- **Decision Thresholds** — guide to interpreting the probability output categories

---

## Prediction Categories

| Label                        | Probability | Interpretation                                                  |
|------------------------------|-------------|------------------------------------------------------------------|
| Very Likely Podium Chance    | ≥ 70%       | Strong indicators across form, grid, and circuit history        |
| Likely Podium Chance         | 50–69.9%    | Competitive scenario, outcome genuinely open                    |
| Moderate Podium Chance       | 30–49.9%    | Possible with favourable conditions or race disruption          |
| Unlikely Podium Chance       | < 30%       | Weak indicators, requires major race events to materialise      |

---

## Limitations

- Predictions are based on **pre-race historical data only**
- Real-time factors (weather, safety cars, mechanical failures, pit strategy, first-corner incidents) are **not modelled**
- New drivers or newly introduced circuits have limited historical data and may use fallback values
- Model is scoped to the **2022–2025 regulation era** — performance may degrade for future seasons with major regulation changes

---

## Notes

Datasets are generated programmatically via API calls. The Jolpica API enforces rate limits; the notebook handles retries automatically.

The trained model file (`models/final_gb_model.joblib`) must exist before running `app.py`. Run the notebook first if it is missing.
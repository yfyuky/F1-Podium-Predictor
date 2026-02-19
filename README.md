# F1 Podium Predictor

Machine learning project that predicts the probability of a Formula 1 driver achieving a podium finish based on historical race data.

## Project Overview

This project explores race outcome prediction in Formula 1 using supervised classification models.  
The objective is to estimate whether a driver will finish in the top three positions (podium) for a given race.

The workflow follows a typical data science pipeline:

1. Data source identification  
2. Data collection via API  
3. Data cleaning and preprocessing  
4. Feature engineering  
5. Baseline model development  
6. Model evaluation

## Data Source

Race data is retrieved from the **Jolpica Ergast-compatible API**, which provides structured historical Formula 1 data including:

- Race results  
- Drivers and constructors  
- Grid positions  
- Qualifying positions  
- Points and finishing positions  

No web scraping was required.

## Prediction Task

Binary classification problem:

- `1` → Driver finished on the podium (P1–P3)  
- `0` → Driver did not finish on the podium  

## Features Used

Examples of predictive variables:

- Season and round  
- Circuit  
- Driver and constructor identifiers  
- Grid position  
- Qualifying position  
- Rolling performance metrics (recent form)

## Technologies & Libraries

- Python  
- pandas  
- numpy  
- scikit-learn  
- requests  

## Reproducing the Project

Install dependencies:

pip install -r requirements.txt


Run the data pipeline (if scripts are provided):

python data_collection.py
python feature_engineering.py

Or open the notebook:

podium_predictor.ipynb


## Notes

Datasets are generated programmatically via API calls and are not stored directly in the repository.

## Example Prediction Output

Example model evaluation metrics from baseline classifier:

- ROC AUC: 0.93
- High recall for podium class
- Moderate precision due to class imbalance

Confusion matrix and classification reports are generated within the notebook.

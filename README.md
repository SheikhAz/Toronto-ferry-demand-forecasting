# Toronto Ferry Demand Forecasting

**Short-term ticket demand forecasting and predictive decision support for the Toronto Island Park ferry service.**

Prepared for Toronto Parks, Forestry & Recreation. The project moves ferry operations from *reactive* queue management to *predictive* operational intelligence.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red)

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Key Findings from EDA](#key-findings-from-eda)
- [Project Pipeline](#project-pipeline)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Model Details](#model-details)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Recommendations](#recommendations)
- [Limitations and Future Work](#limitations-and-future-work)
- [License](#license)

---

## Overview

The ferry service holds roughly ten years of ticket sales and redemption records, but congestion at the terminal has been handled by reacting to the visible length of the queue. This project uses that history to forecast demand **15 minutes to 2 hours ahead**, so that staff and vessels can be deployed before a surge arrives.

The work covers:

1. Data auditing and cleaning
2. Resampling to a continuous 15-minute time series
3. Exploratory data analysis (seasonality, hourly, weekday, peak and momentum effects)
4. Feature engineering (time encodings, lags, rolling statistics)
5. Benchmarking six forecasting models and tuning the best one
6. Deployment of the final model in a Streamlit decision-support dashboard

---

## Key Results

All models were evaluated on a held-out test set using a **chronological** train-test split (lower is better).

| Model                         | MAE       | RMSE      |
|-------------------------------|-----------|-----------|
| SARIMA                        | 50.16     | 112.54    |
| Facebook Prophet              | 40.45     | 81.32     |
| Naive baseline (t-1)          | 21.42     | 86.27     |
| Moving average baseline (1H)  | 19.49     | 69.75     |
| Random Forest                 | 17.39     | 63.25     |
| XGBoost (baseline)            | 16.90     | 63.02     |
| **XGBoost (optimised)**       | **16.90** | **62.49** |

MAE and RMSE are in tickets per 15-minute interval.

**Takeaways**

- Tuned XGBoost gave the best result. SARIMA and Prophet scored worse than the naive baseline, because 15-minute ferry data is long stretches of near-zero activity broken by sudden surges, not smooth oscillations.
- Tuning improved MAE by only 0.01, so the default model was already close to the best possible with the current features. Gains are more likely to come from new features (weather, holidays) than from more tuning.
- RMSE is almost four times MAE: most errors are small, but a minority, concentrated in rare surges, are very large.

---

## Key Findings from EDA

| Finding | Evidence |
|---------|----------|
| **Strong summer seasonality** | Daily volumes reach about 15,000-25,000 tickets in summer and fall to near zero in winter. 2020 is suppressed by the pandemic. |
| **Midday peak** | Average volume peaks at about 11:00-13:00 (roughly 90-97 tickets per 15 min), ramping up from about 09:00. |
| **Peak events cluster at midday** | Intervals above the 95th percentile occur mostly at 11:00-12:00. |
| **Weekend skew** | Saturday and Sunday average about 48-52 tickets per interval, roughly 1.6-2x a typical weekday. |
| **Walk-up behaviour** | Sales and redemptions correlate at **+0.92**, so buyers go almost straight to the gate. |
| **Momentum** | Spikes are usually part of a sustained 1-2 hour wave, not isolated events. |

### Figures

> Copy the charts into `reports/figures/` so the links below render on GitHub.

| | |
|---|---|
| ![Daily demand](reports/figures/daily_demand.png) | ![Weekday demand](reports/figures/weekday_demand.png) |
| ![Hourly sales](reports/figures/hourly_sales.png) | ![Hourly redemptions](reports/figures/hourly_redemptions.png) |
| ![Peak demand](reports/figures/peak_demand.png) | ![Sales vs redemption](reports/figures/sales_vs_redemption.png) |
| ![Rolling demand](reports/figures/rolling_demand.png) | ![Forecast vs actual](reports/figures/model_forecast_comparison.png) |

![Model comparison](reports/figures/multi_model_comparison.png)

---

## Project Pipeline

### 1. Data cleaning

Before any analysis the raw dataset was checked with:

| Check | Method | Result |
|-------|--------|--------|
| Missing values | `df.isnull().sum()` | 0 |
| Duplicates | `df.duplicated().sum()` | 0 |
| Summary statistics | `df.describe()` | Reviewed |
| Types and size | `df.info()` | Confirmed |

### 2. Resampling

Event-level records were aggregated into **15-minute intervals** of Sales Count and Redemption Count to create a continuous time series. Daily and hourly summaries were also produced for EDA.

### 3. Feature engineering

| Family | Features |
|--------|----------|
| Temporal encodings | Hour of day, day of week, month, weekend flag |
| Lag features | t-1, t-2, t-4, t-8 (15 min, 30 min, 1 h, 2 h earlier) |
| Rolling statistics | 1-hour and 2-hour rolling mean and standard deviation |

Rolling statistics are computed on **shifted** data so the model never sees the value it is trying to predict (no target leakage). The output is saved as `data/processed/engineered_ferry.csv`.

### 4. Modelling

Six models were compared: Naive, Moving Average, SARIMA, Facebook Prophet, Random Forest and XGBoost. The split is strictly chronological (train on the past, test on the future).

### 5. Hyperparameter tuning

`RandomizedSearchCV` with `TimeSeriesSplit` ran 20 candidate combinations across 3 folds (60 fits).

---

## Repository Structure

> Adjust to match your actual layout and file names.

```
toronto-ferry-demand-forecasting/
├── data/
│   ├── raw/                  # Source data (see "Data" below)
│   └── processed/
│       └── engineered_ferry.csv
├── notebooks/                # EDA, feature engineering, modelling
├── models/                   # Serialised model and feature pipeline (joblib)
├── app/
│   └── app.py                # Streamlit dashboard
├── reports/
│   ├── figures/              # Charts used in this README and the paper
│   ├── Ferry_Demand_Forecasting_Research_Paper.docx
│   └── Executive_Summary_Ferry_Forecasting.docx
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9 or later
- `pip` and (recommended) a virtual environment

### Installation

```bash
git clone https://github.com/<your-username>/toronto-ferry-demand-forecasting.git
cd toronto-ferry-demand-forecasting

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Main libraries used: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`, `prophet`, `statsmodels`, `joblib`, `streamlit`.

### Data

Place the raw ticket data in `data/raw/`. The dataset is not included in this repository if it is restricted for public release. See the data-source note from the data owner for access.

### Run the pipeline

Run the notebooks in this order:

1. EDA and resampling
2. Feature engineering (creates `data/processed/engineered_ferry.csv`)
3. Model building and tuning (saves the model to `models/`)

### Launch the dashboard

```bash
streamlit run app/app.py
```

---

## Model Details

Final tuned **XGBoost** parameters:

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 500 |
| `max_depth` | 5 |
| `learning_rate` | 0.01 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.7 |
| `min_child_weight` | 3 |

Metrics: **MAE 16.90, RMSE 62.49** per 15-minute interval.

---

## Streamlit Dashboard

The optimised model and its feature pipeline are serialised with `joblib` and served in a Streamlit app that lets operators:

- **Select a forecast horizon** from the next 15 minutes up to 2 hours.
- **View a 95% confidence band**, calculated from the model RMSE, to estimate the worst-case peak risk.
- **See automated readiness alerts**: 🟢 NORMAL, 🟡 ELEVATED or 🔴 CRITICAL (deploy the secondary ferry).

---

## Recommendations

1. **Protect the midday window.** Avoid shift changes and lunch breaks between 10:30 and 13:00, especially on weekends.
2. **Deploy the dashboard to gate managers**, and act on the 1-hour forecast rather than the visible queue.
3. **Act on the upper confidence bound.** When the 95% upper bound exceeds capacity, put the secondary ferry crew on standby.
4. **Pilot off-peak incentives.** Test pricing or promotions for weekend trips before 10:00 or after 15:00.
5. **Review data anomalies.** A few sales-vs-redemption points and one extreme spike in July 2023 look like system artefacts.

---

## Limitations and Future Work

- **Under-prediction of extreme surges.** The model smooths isolated spikes. Quantile regression or peak-focused models could help.
- **No external drivers.** Weather, public holidays, events and ferry schedule or capacity are not yet inputs.
- **Wide confidence band.** A single global RMSE gives a band of roughly ±120 tickets. Quantile or conformal prediction intervals that vary with demand level would be tighter in quiet periods.
- **Pandemic-era data.** The 2020-2021 seasons are structurally different; test with and without them.
- **Evaluation window.** The forecast plots use a winter slice. Report summer-peak performance separately.
- **Further models.** Try LightGBM, LSTM or temporal-fusion networks against the same split.

---

## License

Add a license here (for example, MIT) if the code is to be shared. Check data-sharing permissions separately for the dataset.

## Acknowledgements

Prepared for Toronto Parks, Forestry & Recreation.

<h1 align="center">Renewable Energy Prediction Using Deep Learning & Generative AI</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Flask-2.3+-black?style=flat-square&logo=flask" />
  <img src="https://img.shields.io/badge/TensorFlow-2.13+-orange?style=flat-square&logo=tensorflow" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikitlearn" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

<p align="center">
  A full-stack web application that predicts <strong>solar and wind energy output</strong> across German states using classical ML, deep learning (LSTM), and real historical grid data — with a generative profile engine for future-year forecasting.
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=BQPUwUDnKRE" target="_blank">▶ Watch Video Demo</a>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Models & Results](#models--results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Screenshots](#screenshots)
- [Team](#team)

---

## Overview

Solar and wind energy are central to the global shift toward clean power — but their intermittent nature makes planning difficult. This project builds a prediction system that:

- Forecasts **monthly solar and wind energy generation (MWh)** for any German state and any year (2015–2030+)
- Uses **real historical data** from the Open Power System Data platform to derive seasonal profiles
- Applies **location-specific capacity factors** for each of 12 German states
- Combines **ML/DL models** with a **generative capacity-growth engine** to simulate future years beyond the training data

Germany was chosen as the focus because it leads Europe in renewable adoption (~46% of energy from renewables) and serves as a strong indicator of global trends.

---

## Key Features

- **Location-aware predictions** — 12 German states each with distinct solar irradiance and wind corridor profiles
- **Year-range forecasting** — supports past years (historical) and future years (generative extrapolation up to 2030+)
- **Multiple ML/DL models** — Ridge/Lasso Regression, Decision Tree, SVM, Random Forest, LSTM
- **Automated model selection** — models benchmarked on RMSE, MAE, and accuracy; best model retained per energy type
- **Interactive web UI** — Flask backend + HTML/CSS/JS frontend with real-time chart generation
- **Keras Tuner hyperparameter optimization** — automated tuning for the LSTM model

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask, Flask-CORS |
| Frontend | HTML5, CSS3, Bootstrap, JavaScript |
| ML Models | scikit-learn (Ridge, Lasso, SVM, Random Forest, Decision Tree) |
| Deep Learning | TensorFlow 2.13+, LSTM, Keras Tuner |
| Data & Viz | pandas, NumPy, Matplotlib, Seaborn, statsmodels |
| Model Persistence | joblib |
| Dataset | Open Power System Data (Germany, hourly, 2015–2024) |

---

## Dataset

Data sourced from [Open Power System Data](https://open-power-system-data.org/) — an open platform covering power systems across 37 European countries. The Germany dataset was selected for its completeness and renewable energy diversity.

**Key columns used:**
- `utc_timestamp` — hourly time index
- `solar_capacity`, `wind_capacity` — installed capacity in MW
- `solar_profile`, `wind_profile` — normalized generation profiles
- `wind_onshore_profile`, `wind_offshore_profile` — onshore/offshore wind breakdown
- `solar_generation_actual`, `wind_generation_actual` — target variables

Installed capacity by year (Germany national, MW):

| Year | Solar | Wind |
|------|-------|------|
| 2015 | 37,400 | 44,900 |
| 2020 | 53,700 | 62,200 |
| 2024 | 81,900 | 71,900 |

---

## Data Preprocessing

### Handling Null Values
- **Solar generation**: nulls filled using the prior-day value; first-of-month nulls defaulted to 0 (pre-dawn assumption); remaining NaNs replaced with column mean.
- **Wind generation**: nulls replaced with column mean throughout.

### Correlation Analysis
Feature correlation to target variables was computed to identify the most predictive inputs:
- Solar: `solar_profile` showed highest correlation with solar output
- Wind: `wind_profile`, `wind_onshore_profile`, `wind_onshore_generation` were top predictors

![Correlation Heatmap](https://user-images.githubusercontent.com/87893594/211166795-6f554565-dee5-4d3d-8998-0795c909fd10.png)

### Feature Importance
Random Forest feature importance scores were used alongside correlation analysis to filter out low-signal variables, reducing model bias and improving generalization.

---

## Models & Results

Six models were trained and evaluated on standard regression metrics:

| Model | Energy Type | Best Metric |
|---|---|---|
| Ridge Regression | Solar & Wind | Baseline |
| Lasso Regression | Solar & Wind | Baseline |
| Decision Tree | Solar & Wind | — |
| SVM | **Wind** | Best classical ML for wind |
| Random Forest | **Solar** | Best classical ML for solar |
| **LSTM** | **Solar** | ~98.2% accuracy |

**Key finding:** SVM was best for wind energy prediction; Random Forest for solar. The LSTM deep learning model achieved ~98.2% accuracy on solar generation.

![Model Results 1](https://user-images.githubusercontent.com/74909133/211185941-6d9292c6-acee-476a-9dd6-467f804c80b5.jpeg)
![Model Results 2](https://user-images.githubusercontent.com/74909133/211185945-1c4de132-e170-4307-b1b0-94b7b6e4fef5.jpeg)
![LSTM True vs Predicted](https://user-images.githubusercontent.com/74909133/211186083-d5179172-9220-4d6b-b62a-08490c9e6b7e.jpeg)
<p align="center"><em>LSTM — True values vs Predicted values</em></p>

---

## Project Structure

```
├── app.py                          # Flask backend — API routes & prediction logic
├── requirements.txt                # Python dependencies
├── time_series_60min_singleindex_filtered.csv  # Germany hourly energy dataset
├── models/
│   ├── solar_rf.pkl                # Trained Random Forest (solar)
│   ├── solar_scaler.pkl            # Feature scaler (solar)
│   ├── wind_rf.pkl                 # Trained Random Forest (wind)
│   └── wind_scaler.pkl             # Feature scaler (wind)
├── website/
│   ├── index.html                  # Main landing page
│   ├── output.html                 # Solar prediction results page
│   ├── output_wind.html            # Wind prediction results page
│   ├── css/                        # Stylesheets
│   ├── js/                         # Frontend scripts
│   └── img/                        # Static assets
├── MAIN .ipynb                     # Full model training & evaluation notebook
├── LSTM Code .ipynb                # LSTM deep learning notebook
├── ANOTHER.ipynb                   # Supplementary analysis notebook
└── Collab Notebook-Arunkumar.ipynb # Collaborative exploration notebook
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/arunkumaraluru/Renewable-Energy-prediction-using-Generative-AI.git
cd Renewable-Energy-prediction-using-Generative-AI

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Open your browser at `http://localhost:5000`.

---

## How It Works

1. **User selects** an energy type (solar/wind), a German state, and a year range via the web UI.
2. **The backend** loads the pre-trained Random Forest model and the real dataset's monthly seasonal profiles.
3. **Capacity for future years** is extrapolated linearly from known national installed-capacity data.
4. **State-specific capacity factors** scale the national baseline to regional conditions (e.g. Bavaria has the highest solar CF; Schleswig-Holstein has the highest wind CF).
5. **Monthly predictions** (MWh) are generated and returned as JSON, then rendered as charts on the results page.

---

## Screenshots

<img width="1439" alt="Cover Page" src="https://user-images.githubusercontent.com/74909133/211185833-7900ed1d-5cef-4d75-b128-0e1953ec526d.png">
<img width="1440" alt="About Section" src="https://user-images.githubusercontent.com/74909133/211185843-3c5464cd-8dbc-4c21-bac7-02035c74433d.png">
<img width="1440" alt="Solar Predictor" src="https://user-images.githubusercontent.com/74909133/211185851-e1f6a5cb-2258-4bef-ae44-5aa2d266cc92.png">
<img width="1439" alt="Advantages" src="https://user-images.githubusercontent.com/74909133/211185852-9a6cc865-e25b-4e64-b2fa-603197d3117d.png">
<img width="1440" alt="Wind Predictor" src="https://user-images.githubusercontent.com/74909133/211185854-f65a083c-1ef2-4a82-b974-cde3e9249800.png">

---

## Team

- Aluru Arunkumar
- Mehak Aggarwal
- Sonanshi Goel
- Shambhavi Rai
- Princy Singhal

---

<p align="center">Made with dedication for a sustainable energy future 🌱</p>

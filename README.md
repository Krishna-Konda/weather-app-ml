# WeatherML – Machine Learning Weather Prediction System

A complete ML weather prediction system built with **Flask**, **scikit-learn**, and **Pandas**.
Predicts **temperature** (regression) and **rainfall** (classification) from historical weather data.

---

## Project Structure

```
weather_prediction/
├── app.py                  # Flask web application
├── train_models.py         # Model training script
├── requirements.txt
├── data/
│   └── historical_weather.csv   # 10 years of synthetic weather data
├── models/
│   ├── temp_model.pkl      # Random Forest – temperature
│   ├── temp_scaler.pkl
│   ├── temp_features.pkl
│   ├── rain_model.pkl      # Gradient Boosting – rainfall
│   ├── rain_scaler.pkl
│   ├── rain_features.pkl
│   └── summary.json        # Model accuracy metrics
└── templates/
    └── index.html          # Flask HTML template
```

---

## ML Models

| Task              | Algorithm           | Metric                      |
|-------------------|---------------------|-----------------------------|
| Temperature       | Random Forest       | MAE ≈ 1.85°C, R² ≈ 0.93    |
| Rainfall (binary) | Gradient Boosting   | Accuracy ≈ 65%              |

### Features Used

**Temperature prediction:**  
month, day_of_year, humidity, pressure, wind_speed, cloud_cover,  
temp_lag1, temp_lag7, pressure_diff, temp_7d_avg, sin_doy, cos_doy

**Rainfall prediction:**  
month, day_of_year, temperature, humidity, pressure, wind_speed, cloud_cover,  
rain_lag1, humidity_lag1, pressure_diff, rain_7d_sum, temp_7d_avg, sin_doy, cos_doy

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (generates data + trains both models)
python train_models.py

# 3. Start the Flask server
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Features

- **Single Prediction** – Predict temperature or rainfall probability given weather conditions
- **7-Day Forecast** – Full week forecast with temperature and rain probability cards
- **Historical Charts** – Last 30 days of temperature and rainfall trends
- **Model Metrics Dashboard** – Live R², MAE, and accuracy badges in the header
- **Interactive Sliders** – Adjust all parameters in real-time
- **Responsive Design** – Works on desktop and mobile

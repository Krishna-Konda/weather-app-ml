"""
Train temperature and rainfall prediction models using synthetic historical weather data.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import os

np.random.seed(42)

# ─── Generate Historical Weather Data ─────────────────────────────────────────
def generate_weather_data(n_days=3650):  # 10 years of data
    dates = pd.date_range(start='2015-01-01', periods=n_days, freq='D')
    day_of_year = dates.dayofyear.values
    year        = dates.year.values

    # Seasonal temperature pattern (sine wave) + trend + noise
    seasonal_temp = 22 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    yearly_trend  = 0.03 * (year - 2015)
    noise_temp    = np.random.normal(0, 2.5, n_days)
    temperature   = seasonal_temp + yearly_trend + noise_temp

    # Humidity (inversely correlated with temp, seasonal)
    humidity = 70 - 0.4 * (temperature - 22) + np.random.normal(0, 8, n_days)
    humidity = np.clip(humidity, 20, 100)

    # Pressure (hPa) – random walk + seasonal
    pressure = 1013 + 8 * np.sin(2 * np.pi * day_of_year / 365) + \
               np.cumsum(np.random.normal(0, 0.3, n_days))
    pressure = 1013 + (pressure - pressure.mean()) / pressure.std() * 5

    # Wind speed (km/h)
    wind_speed = np.abs(np.random.normal(15, 7, n_days))

    # Cloud cover (%)
    cloud_cover = np.clip(humidity * 0.7 + np.random.normal(0, 15, n_days), 0, 100)

    # Rainfall: probabilistic based on humidity, cloud cover, pressure drop
    rain_prob = (humidity / 100) * 0.4 + (cloud_cover / 100) * 0.4 + \
                (1 - (pressure - 1005) / 20) * 0.2
    rain_prob = np.clip(rain_prob, 0, 1)
    rain_mm   = np.where(
        np.random.random(n_days) < rain_prob,
        np.random.exponential(8, n_days),
        0
    )
    rain_mm   = np.round(rain_mm, 1)

    df = pd.DataFrame({
        'date':        dates,
        'day_of_year': day_of_year,
        'month':       dates.month,
        'season':      pd.cut(dates.month, bins=[0,3,6,9,12],
                              labels=['Winter','Spring','Summer','Autumn']).astype(str),
        'temperature': np.round(temperature, 1),
        'humidity':    np.round(humidity, 1),
        'pressure':    np.round(pressure, 1),
        'wind_speed':  np.round(wind_speed, 1),
        'cloud_cover': np.round(cloud_cover, 1),
        'rainfall_mm': rain_mm,
        'will_rain':   (rain_mm > 0).astype(int)
    })
    return df

# ─── Feature Engineering ──────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df['temp_lag1']     = df['temperature'].shift(1)
    df['temp_lag7']     = df['temperature'].shift(7)
    df['rain_lag1']     = df['rainfall_mm'].shift(1)
    df['humidity_lag1'] = df['humidity'].shift(1)
    df['pressure_diff'] = df['pressure'].diff()
    df['temp_7d_avg']   = df['temperature'].rolling(7).mean()
    df['rain_7d_sum']   = df['rainfall_mm'].rolling(7).sum()
    df['sin_doy']       = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_doy']       = np.cos(2 * np.pi * df['day_of_year'] / 365)
    return df.dropna()

# ─── Train & Save Models ──────────────────────────────────────────────────────
def train_and_save():
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    print("Generating historical weather data...")
    df = generate_weather_data(3650)
    df.to_csv('data/historical_weather.csv', index=False)
    print(f"  → {len(df)} days saved to data/historical_weather.csv")

    df_feat = engineer_features(df)

    TEMP_FEATURES = [
        'month','day_of_year','humidity','pressure','wind_speed','cloud_cover',
        'temp_lag1','temp_lag7','pressure_diff','temp_7d_avg','sin_doy','cos_doy'
    ]
    RAIN_FEATURES = [
        'month','day_of_year','temperature','humidity','pressure','wind_speed',
        'cloud_cover','rain_lag1','humidity_lag1','pressure_diff','rain_7d_sum',
        'temp_7d_avg','sin_doy','cos_doy'
    ]

    # ── Temperature Regression ────────────────────────────────────────────────
    print("\nTraining Temperature Prediction Model (Random Forest)...")
    X_temp = df_feat[TEMP_FEATURES]
    y_temp = df_feat['temperature']

    scaler_temp = StandardScaler()
    X_temp_sc   = scaler_temp.fit_transform(X_temp)

    X_tr, X_te, y_tr, y_te = train_test_split(X_temp_sc, y_temp, test_size=0.15, random_state=42)
    rf_temp = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3,
                                     n_jobs=-1, random_state=42)
    rf_temp.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, rf_temp.predict(X_te))
    print(f"  → Temperature MAE: {mae:.2f}°C  |  R²: {rf_temp.score(X_te, y_te):.4f}")

    joblib.dump(rf_temp,      'models/temp_model.pkl')
    joblib.dump(scaler_temp,  'models/temp_scaler.pkl')
    joblib.dump(TEMP_FEATURES,'models/temp_features.pkl')

    # ── Rainfall Classification ───────────────────────────────────────────────
    print("\nTraining Rainfall Prediction Model (Gradient Boosting)...")
    X_rain = df_feat[RAIN_FEATURES]
    y_rain = df_feat['will_rain']

    scaler_rain = StandardScaler()
    X_rain_sc   = scaler_rain.fit_transform(X_rain)

    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_rain_sc, y_rain, test_size=0.15, random_state=42)
    gb_rain = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.08,
                                          subsample=0.8, random_state=42)
    gb_rain.fit(X_tr2, y_tr2)
    acc = accuracy_score(y_te2, gb_rain.predict(X_te2))
    print(f"  → Rainfall Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    joblib.dump(gb_rain,      'models/rain_model.pkl')
    joblib.dump(scaler_rain,  'models/rain_scaler.pkl')
    joblib.dump(RAIN_FEATURES,'models/rain_features.pkl')

    # ── Save Summary ──────────────────────────────────────────────────────────
    summary = {
        'temp_mae': round(mae, 3),
        'temp_r2':  round(rf_temp.score(X_te, y_te), 4),
        'rain_acc': round(acc, 4),
        'train_samples': len(df_feat),
        'data_years': '2015–2024'
    }
    import json
    with open('models/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n✅ All models trained and saved to models/")
    return summary

if __name__ == '__main__':
    train_and_save()

"""
Train per-city weather prediction models using city-specific climate profiles.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib, json, os
from city_config import CITIES

np.random.seed(42)

TEMP_FEATURES = [
    'month','day_of_year','humidity','pressure','wind_speed','cloud_cover',
    'temp_lag1','temp_lag7','pressure_diff','temp_7d_avg','sin_doy','cos_doy'
]
RAIN_FEATURES = [
    'month','day_of_year','temperature','humidity','pressure','wind_speed',
    'cloud_cover','rain_lag1','humidity_lag1','pressure_diff','rain_7d_sum',
    'temp_7d_avg','sin_doy','cos_doy'
]

def generate_city_data(city_name, cfg, n_days=3650):
    dates      = pd.date_range('2015-01-01', periods=n_days, freq='D')
    doy        = dates.dayofyear.values
    month      = dates.month.values
    year       = dates.year.values

    # Temperature: city baseline + amplitude + trend + noise
    # Southern hemisphere: flip phase
    phase = 80 if cfg['lat'] >= 0 else 260
    seasonal_temp = cfg['avg_temp'] + cfg['temp_amplitude'] * \
                    np.sin(2 * np.pi * (doy - phase) / 365)
    trend         = 0.03 * (year - 2015)
    temperature   = seasonal_temp + trend + np.random.normal(0, 2, n_days)

    humidity = cfg['avg_humidity'] - 0.3*(temperature - cfg['avg_temp']) + \
               np.random.normal(0, 7, n_days)
    humidity = np.clip(humidity, 15, 100)

    pressure = cfg['avg_pressure'] + 5*np.sin(2*np.pi*doy/365) + \
               np.cumsum(np.random.normal(0, 0.25, n_days))
    pressure = cfg['avg_pressure'] + (pressure - pressure.mean()) / pressure.std() * 4

    wind_speed  = np.abs(np.random.normal(cfg['avg_wind'], 6, n_days))
    cloud_cover = np.clip(humidity*0.65 + np.random.normal(0,14,n_days), 0, 100)

    # Rain: weighted toward city's rain season peak month
    peak   = cfg['rain_season_peak']
    rain_seasonal = 0.5 + 0.5*np.cos(2*np.pi*(month - peak)/12)
    rain_prob = np.clip(
        (humidity/100)*0.35 + (cloud_cover/100)*0.35 + rain_seasonal*0.3, 0, 1
    ) * cfg['rain_multiplier']
    rain_prob = np.clip(rain_prob, 0, 0.95)
    rain_mm   = np.where(np.random.random(n_days) < rain_prob,
                         np.random.exponential(7, n_days), 0)

    df = pd.DataFrame({
        'date':        dates,
        'city':        city_name,
        'day_of_year': doy,
        'month':       month,
        'temperature': np.round(temperature, 1),
        'humidity':    np.round(humidity, 1),
        'pressure':    np.round(pressure, 1),
        'wind_speed':  np.round(wind_speed, 1),
        'cloud_cover': np.round(cloud_cover, 1),
        'rainfall_mm': np.round(rain_mm, 1),
        'will_rain':   (rain_mm > 0).astype(int),
    })
    return df

def engineer_features(df):
    df = df.copy().sort_values('date').reset_index(drop=True)
    df['temp_lag1']      = df['temperature'].shift(1)
    df['temp_lag7']      = df['temperature'].shift(7)
    df['rain_lag1']      = df['rainfall_mm'].shift(1)
    df['humidity_lag1']  = df['humidity'].shift(1)
    df['pressure_diff']  = df['pressure'].diff()
    df['temp_7d_avg']    = df['temperature'].rolling(7).mean()
    df['rain_7d_sum']    = df['rainfall_mm'].rolling(7).sum()
    df['sin_doy']        = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_doy']        = np.cos(2 * np.pi * df['day_of_year'] / 365)
    return df.dropna()

def train_city(city_name, cfg):
    df      = generate_city_data(city_name, cfg)
    df_feat = engineer_features(df)

    results = {}

    # ── Temperature ────────────────────────────────────────────
    X = df_feat[TEMP_FEATURES]
    y = df_feat['temperature']
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.15, random_state=42)
    m = RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=-1, random_state=42)
    m.fit(Xtr, ytr)
    mae = mean_absolute_error(yte, m.predict(Xte))
    r2  = m.score(Xte, yte)
    results['temp_mae'] = round(mae, 3)
    results['temp_r2']  = round(r2, 4)
    joblib.dump(m,  f'models/{city_name}_temp_model.pkl')
    joblib.dump(sc, f'models/{city_name}_temp_scaler.pkl')

    # ── Rainfall ───────────────────────────────────────────────
    X2  = df_feat[RAIN_FEATURES]
    y2  = df_feat['will_rain']
    sc2 = StandardScaler()
    X2s = sc2.fit_transform(X2)
    Xtr2, Xte2, ytr2, yte2 = train_test_split(X2s, y2, test_size=0.15, random_state=42)
    m2 = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                                     subsample=0.8, random_state=42)
    m2.fit(Xtr2, ytr2)
    acc = accuracy_score(yte2, m2.predict(Xte2))
    results['rain_acc']      = round(acc, 4)
    results['train_samples'] = len(df_feat)
    joblib.dump(m2,  f'models/{city_name}_rain_model.pkl')
    joblib.dump(sc2, f'models/{city_name}_rain_scaler.pkl')

    # Save last 60 days for history chart
    tail = df.tail(60).copy()
    tail['date'] = tail['date'].dt.strftime('%Y-%m-%d')
    results['history'] = tail[['date','temperature','rainfall_mm','humidity']].to_dict('records')

    return results

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('data',   exist_ok=True)

    # Save feature lists once
    joblib.dump(TEMP_FEATURES, 'models/temp_features.pkl')
    joblib.dump(RAIN_FEATURES, 'models/rain_features.pkl')

    all_summary = {}
    for city, cfg in CITIES.items():
        print(f"  Training {city} ({cfg['climate']})...", end=' ', flush=True)
        res = train_city(city, cfg)
        all_summary[city] = {**cfg, **res}
        print(f"T-MAE={res['temp_mae']}°C  Rain={res['rain_acc']*100:.1f}%")

    with open('models/city_summary.json', 'w') as f:
        json.dump(all_summary, f, indent=2, default=str)

    print(f"\n✅ Trained {len(CITIES)} city models")

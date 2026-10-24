"""
WeatherML City-Wise Prediction System – app.py
"""
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib, json, os, io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from city_config import CITIES

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load shared feature lists ──────────────────────────────────────────────────
TEMP_FEATURES = joblib.load(f'{BASE}/models/temp_features.pkl')
RAIN_FEATURES = joblib.load(f'{BASE}/models/rain_features.pkl')

with open(f'{BASE}/models/city_summary.json') as f:
    CITY_SUMMARY = json.load(f)

# ── Lazy-load per-city models ──────────────────────────────────────────────────
_model_cache = {}
def get_models(city):
    if city not in _model_cache:
        _model_cache[city] = {
            'temp_model':  joblib.load(f'{BASE}/models/{city}_temp_model.pkl'),
            'temp_scaler': joblib.load(f'{BASE}/models/{city}_temp_scaler.pkl'),
            'rain_model':  joblib.load(f'{BASE}/models/{city}_rain_model.pkl'),
            'rain_scaler': joblib.load(f'{BASE}/models/{city}_rain_scaler.pkl'),
        }
    return _model_cache[city]

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_season(month, lat):
    # Flip seasons for southern hemisphere
    if lat < 0:
        month = ((month + 5) % 12) + 1
    return {12:'Winter',1:'Winter',2:'Winter',
            3:'Spring',4:'Spring',5:'Spring',
            6:'Summer',7:'Summer',8:'Summer',
            9:'Autumn',10:'Autumn',11:'Autumn'}[month]

def city_month_stats(city, month):
    hist = CITY_SUMMARY[city].get('history', [])
    if not hist:
        cfg = CITIES[city]
        return {
            'avg_temp': cfg['avg_temp'], 'avg_humidity': cfg['avg_humidity'],
            'avg_pressure': cfg['avg_pressure'], 'avg_wind': cfg['avg_wind'],
            'avg_cloud': cfg['avg_cloud'], 'avg_rain': 3.0,
        }
    df = pd.DataFrame(hist)
    df['month'] = pd.to_datetime(df['date']).dt.month
    sub = df[df['month'] == month]
    if sub.empty:
        sub = df
    return {
        'avg_temp':     round(sub['temperature'].mean(), 1),
        'avg_humidity': round(sub['humidity'].mean(), 1),
        'avg_pressure': CITIES[city]['avg_pressure'],
        'avg_wind':     CITIES[city]['avg_wind'],
        'avg_cloud':    CITIES[city]['avg_cloud'],
        'avg_rain':     round(sub['rainfall_mm'].mean(), 1),
    }

def build_temp_vec(city, month, doy, humidity, pressure, wind, cloud):
    s = city_month_stats(city, month)
    v = {'month': month, 'day_of_year': doy, 'humidity': humidity,
         'pressure': pressure, 'wind_speed': wind, 'cloud_cover': cloud,
         'temp_lag1': s['avg_temp'], 'temp_lag7': s['avg_temp'],
         'pressure_diff': 0.0, 'temp_7d_avg': s['avg_temp'],
         'sin_doy': np.sin(2*np.pi*doy/365), 'cos_doy': np.cos(2*np.pi*doy/365)}
    return pd.DataFrame([v])[TEMP_FEATURES]

def build_rain_vec(city, month, doy, temperature, humidity, pressure, wind, cloud):
    s = city_month_stats(city, month)
    v = {'month': month, 'day_of_year': doy, 'temperature': temperature,
         'humidity': humidity, 'pressure': pressure, 'wind_speed': wind,
         'cloud_cover': cloud, 'rain_lag1': s['avg_rain'],
         'humidity_lag1': s['avg_humidity'], 'pressure_diff': 0.0,
         'rain_7d_sum': s['avg_rain']*7, 'temp_7d_avg': s['avg_temp'],
         'sin_doy': np.sin(2*np.pi*doy/365), 'cos_doy': np.cos(2*np.pi*doy/365)}
    return pd.DataFrame([v])[RAIN_FEATURES]

def make_chart(title, x_vals, y_vals, ylabel, color, fill=False):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    if fill:
        ax.fill_between(range(len(y_vals)), y_vals, alpha=0.2, color=color)
    ax.plot(range(len(y_vals)), y_vals, color=color, linewidth=2,
            marker='o', markersize=4)
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals, rotation=30, ha='right', fontsize=7.5)
    ax.set_title(title, color='#e6edf3', fontsize=11, pad=8, fontweight='bold')
    ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.grid(axis='y', color='#21262d', linewidth=0.7)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    city_list = sorted(CITIES.keys())
    city_data = {c: {'country': CITIES[c]['country'],
                     'climate': CITIES[c]['climate'],
                     'lat': CITIES[c]['lat'],
                     'lon': CITIES[c]['lon']} for c in CITIES}
    return render_template('index.html',
                           city_list=city_list,
                           city_data=json.dumps(city_data),
                           city_summary=json.dumps(CITY_SUMMARY))

@app.route('/city_info')
def city_info():
    city = request.args.get('city', 'Mumbai')
    cfg  = CITIES.get(city, CITIES['Mumbai'])
    summ = CITY_SUMMARY.get(city, {})
    hist = summ.get('history', [])

    temp_chart, rain_chart = None, None
    if hist:
        df = pd.DataFrame(hist)
        df = df.tail(30)
        labels = pd.to_datetime(df['date']).dt.strftime('%b %d').tolist()
        temp_chart = make_chart(f'{city} – Temperature (Last 30 Days)',
                                labels, df['temperature'].tolist(),
                                'Temp (°C)', '#58a6ff', fill=True)
        rain_chart = make_chart(f'{city} – Rainfall (Last 30 Days)',
                                labels, df['rainfall_mm'].tolist(),
                                'Rainfall (mm)', '#3fb950', fill=True)

    return jsonify({
        'city': city,
        'country': cfg['country'],
        'climate': cfg['climate'],
        'lat': cfg['lat'], 'lon': cfg['lon'],
        'avg_temp': cfg['avg_temp'],
        'avg_humidity': cfg['avg_humidity'],
        'avg_pressure': cfg['avg_pressure'],
        'avg_wind': cfg['avg_wind'],
        'avg_cloud': cfg['avg_cloud'],
        'temp_mae': summ.get('temp_mae'),
        'temp_r2':  summ.get('temp_r2'),
        'rain_acc': summ.get('rain_acc'),
        'temp_chart': temp_chart,
        'rain_chart': rain_chart,
    })

@app.route('/predict', methods=['POST'])
def predict():
    d    = request.get_json()
    city = d.get('city', 'Mumbai')
    m    = get_models(city)
    cfg  = CITIES[city]

    month   = int(d['month'])
    doy     = int(d['day_of_year'])
    hum     = float(d['humidity'])
    pres    = float(d['pressure'])
    wind    = float(d['wind_speed'])
    cloud   = float(d['cloud_cover'])
    ptype   = d.get('type', 'temperature')

    summ    = CITY_SUMMARY.get(city, {})

    if ptype == 'temperature':
        X   = build_temp_vec(city, month, doy, hum, pres, wind, cloud)
        Xs  = m['temp_scaler'].transform(X)
        pred = round(float(m['temp_model'].predict(Xs)[0]), 1)
        conf = round(summ.get('temp_r2', 0.9) * 100, 1)
        return jsonify({'type': 'temperature', 'prediction': pred,
                        'label': f'{pred} °C', 'unit': '°C',
                        'confidence': conf, 'season': get_season(month, cfg['lat']),
                        'city': city})

    elif ptype == 'rainfall':
        Xt  = build_temp_vec(city, month, doy, hum, pres, wind, cloud)
        Xts = m['temp_scaler'].transform(Xt)
        pred_temp = float(m['temp_model'].predict(Xts)[0])

        X   = build_rain_vec(city, month, doy, pred_temp, hum, pres, wind, cloud)
        Xs  = m['rain_scaler'].transform(X)
        prob      = round(float(m['rain_model'].predict_proba(Xs)[0][1]) * 100, 1)
        will_rain = bool(m['rain_model'].predict(Xs)[0])
        conf      = round(summ.get('rain_acc', 0.75) * 100, 1)
        return jsonify({'type': 'rainfall', 'prediction': prob,
                        'label': 'Rain Expected' if will_rain else 'No Rain',
                        'unit': '%', 'confidence': conf,
                        'will_rain': will_rain, 'probability': prob,
                        'season': get_season(month, cfg['lat']), 'city': city})

    return jsonify({'error': 'Invalid type'}), 400

@app.route('/forecast', methods=['POST'])
def forecast():
    d    = request.get_json()
    city = d.get('city', 'Mumbai')
    m    = get_models(city)
    cfg  = CITIES[city]
    base = datetime(int(d['year']), int(d['month']), int(d['day']))
    hum  = float(d['humidity'])
    pres = float(d['pressure'])
    wind = float(d['wind_speed'])
    cloud= float(d['cloud_cover'])

    temps, rains, dates = [], [], []
    for i in range(7):
        dd  = base + timedelta(days=i)
        doy = dd.timetuple().tm_yday
        mo  = dd.month

        Xt  = build_temp_vec(city, mo, doy, hum, pres, wind, cloud)
        Xts = m['temp_scaler'].transform(Xt)
        t   = round(float(m['temp_model'].predict(Xts)[0]), 1)

        Xr  = build_rain_vec(city, mo, doy, t, hum, pres, wind, cloud)
        Xrs = m['rain_scaler'].transform(Xr)
        rp  = round(float(m['rain_model'].predict_proba(Xrs)[0][1]) * 100, 1)

        temps.append(t)
        rains.append(rp)
        dates.append(dd.strftime('%a %d %b'))

    return jsonify({
        'dates': dates, 'temperatures': temps, 'rain_probs': rains,
        'temp_chart': make_chart(f'{city} – 7-Day Temperature', dates, temps, 'Temp (°C)', '#58a6ff', True),
        'rain_chart': make_chart(f'{city} – 7-Day Rain Probability', dates, rains, 'Prob (%)', '#3fb950', True),
    })

@app.route('/compare', methods=['POST'])
def compare():
    """Compare prediction across multiple cities for same conditions."""
    d     = request.get_json()
    cities = d.get('cities', list(CITIES.keys())[:6])
    month = int(d['month']); doy = int(d['day_of_year'])
    hum   = float(d['humidity']); pres = float(d['pressure'])
    wind  = float(d['wind_speed']); cloud = float(d['cloud_cover'])

    results = []
    for city in cities:
        try:
            m   = get_models(city)
            cfg = CITIES[city]
            summ= CITY_SUMMARY.get(city, {})

            Xt  = build_temp_vec(city, month, doy, hum, pres, wind, cloud)
            Xts = m['temp_scaler'].transform(Xt)
            t   = round(float(m['temp_model'].predict(Xts)[0]), 1)

            Xr  = build_rain_vec(city, month, doy, t, hum, pres, wind, cloud)
            Xrs = m['rain_scaler'].transform(Xr)
            rp  = round(float(m['rain_model'].predict_proba(Xrs)[0][1]) * 100, 1)

            results.append({'city': city, 'country': cfg['country'],
                             'temperature': t, 'rain_prob': rp,
                             'climate': cfg['climate'],
                             'will_rain': rp > 50})
        except Exception as e:
            results.append({'city': city, 'error': str(e)})

    results.sort(key=lambda x: x.get('temperature', 0), reverse=True)

    # Bar chart – temperatures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2))
    fig.patch.set_facecolor('#0d1117')
    names = [r['city'] for r in results if 'temperature' in r]
    temps = [r['temperature'] for r in results if 'temperature' in r]
    rains = [r['rain_prob'] for r in results if 'rain_prob' in r]

    colors_t = ['#ff7b72' if t > 30 else '#58a6ff' if t < 15 else '#d29922' for t in temps]
    colors_r = ['#3fb950' if r > 60 else '#8b949e' for r in rains]

    ax1.barh(names, temps, color=colors_t, edgecolor='#30363d')
    ax1.set_facecolor('#0d1117'); ax1.set_title('Temperature (°C)', color='#e6edf3', fontweight='bold')
    ax1.tick_params(colors='#8b949e'); [sp.set_edgecolor('#30363d') for sp in ax1.spines.values()]
    for i, v in enumerate(temps):
        ax1.text(v + 0.3, i, f'{v}°', va='center', color='#e6edf3', fontsize=8)

    ax2.barh(names, rains, color=colors_r, edgecolor='#30363d')
    ax2.set_facecolor('#0d1117'); ax2.set_title('Rain Probability (%)', color='#e6edf3', fontweight='bold')
    ax2.tick_params(colors='#8b949e'); [sp.set_edgecolor('#30363d') for sp in ax2.spines.values()]
    for i, v in enumerate(rains):
        ax2.text(v + 0.5, i, f'{v}%', va='center', color='#e6edf3', fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0d1117')
    plt.close(); buf.seek(0)
    chart = base64.b64encode(buf.read()).decode()

    return jsonify({'results': results, 'chart': chart})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

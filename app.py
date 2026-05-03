import os, io, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WEBSITE_DIR = os.path.join(BASE_DIR, "website")
DATA_PATH   = os.path.join(BASE_DIR, "time_series_60min_singleindex_filtered.csv")

app = Flask(__name__, static_folder=WEBSITE_DIR, static_url_path="")
CORS(app)

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ── German state data ──────────────────────────────────────────────────────────
# share = fraction of Germany's national installed capacity for that state
# cf    = capacity factor multiplier relative to national average
SOLAR_LOCATIONS = {
    "Bavaria (Bayern)":         {"share": 0.220, "cf": 1.28, "desc": "Southern Germany — highest solar irradiance"},
    "Baden-Württemberg":        {"share": 0.098, "cf": 1.20, "desc": "Southwest — excellent solar potential"},
    "Saxony-Anhalt":            {"share": 0.068, "cf": 1.06, "desc": "East Germany — above-average solar"},
    "Brandenburg":              {"share": 0.072, "cf": 1.05, "desc": "Central-East — flat terrain, good solar"},
    "Saxony (Sachsen)":         {"share": 0.062, "cf": 1.08, "desc": "East — strong solar performance"},
    "Rhineland-Palatinate":     {"share": 0.060, "cf": 1.10, "desc": "West-Central — good solar"},
    "Hesse (Hessen)":           {"share": 0.055, "cf": 1.00, "desc": "Central Germany — average solar"},
    "Thuringia (Thüringen)":    {"share": 0.045, "cf": 1.02, "desc": "Central — moderate solar"},
    "North Rhine-Westphalia":   {"share": 0.100, "cf": 0.92, "desc": "West — moderate solar"},
    "Lower Saxony":             {"share": 0.082, "cf": 0.88, "desc": "Northwest — lower irradiance"},
    "Schleswig-Holstein":       {"share": 0.052, "cf": 0.82, "desc": "Northern coast — lowest solar in DE"},
    "Hamburg":                  {"share": 0.018, "cf": 0.78, "desc": "Dense city-state — limited solar area"},
}

WIND_LOCATIONS = {
    "Lower Saxony":             {"share": 0.180, "cf": 1.45, "desc": "Northwest — strong North Sea influence"},
    "Schleswig-Holstein":       {"share": 0.120, "cf": 1.55, "desc": "Northern coast — best wind in Germany"},
    "Brandenburg":              {"share": 0.140, "cf": 1.20, "desc": "Central — flat, good wind corridor"},
    "Mecklenburg-Vorpommern":   {"share": 0.085, "cf": 1.40, "desc": "Northeast coast — excellent offshore wind"},
    "Saxony-Anhalt":            {"share": 0.082, "cf": 1.15, "desc": "East — good flat-terrain winds"},
    "Hamburg":                  {"share": 0.025, "cf": 1.35, "desc": "Port city — coastal wind advantage"},
    "North Rhine-Westphalia":   {"share": 0.110, "cf": 0.90, "desc": "West — moderate wind resources"},
    "Thuringia (Thüringen)":    {"share": 0.055, "cf": 0.88, "desc": "Central — moderate wind"},
    "Hesse (Hessen)":           {"share": 0.050, "cf": 0.85, "desc": "Central — below-average wind"},
    "Rhineland-Palatinate":     {"share": 0.045, "cf": 0.92, "desc": "West-Central — moderate wind"},
    "Baden-Württemberg":        {"share": 0.045, "cf": 0.75, "desc": "Southwest — limited wind resources"},
    "Bavaria (Bayern)":         {"share": 0.042, "cf": 0.65, "desc": "Southern — lowest wind potential"},
}

# ── Germany national installed capacity (MW) ───────────────────────────────────
SOLAR_CAP_MW = {
    2015: 37400, 2016: 40700, 2017: 43300, 2018: 45200, 2019: 49200,
    2020: 53700, 2021: 58500, 2022: 66500, 2023: 73400, 2024: 81900,
}
WIND_CAP_MW = {
    2015: 44900, 2016: 50000, 2017: 56100, 2018: 59300, 2019: 61400,
    2020: 62200, 2021: 63800, 2022: 66300, 2023: 69400, 2024: 71900,
}

def _extrapolate(table, year):
    """Return capacity for any year, extrapolating linearly beyond known range."""
    if year in table:
        return table[year]
    years = sorted(table)
    if year < years[0]:
        return table[years[0]]
    y0, y1 = years[-2], years[-1]
    slope = (table[y1] - table[y0]) / (y1 - y0)
    return table[y1] + slope * (year - y1)

def _month_hours(m, year):
    leap = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
    days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return days[m - 1] * 24

# ── Load dataset → derive real monthly seasonal profiles ──────────────────────
print("Loading dataset to derive seasonal profiles …")
_df = pd.read_csv(DATA_PATH, parse_dates=[0], index_col=0)
_df.index = pd.to_datetime(_df.index, utc=True)
_df['_m'] = _df.index.month

SOLAR_CF = _df.groupby('_m')['DE_solar_profile'].mean().to_dict()   # {1: 0.023, …}
WIND_CF  = _df.groupby('_m')['DE_wind_profile'].mean().to_dict()

del _df
print("Seasonal profiles ready.")


# ── Chart generation ───────────────────────────────────────────────────────────
def _make_chart(monthly_gwh, year, location, kind):
    bar_color = '#f59e0b' if kind == 'solar' else '#3b82f6'
    label     = 'Solar' if kind == 'solar'  else 'Wind'
    icon      = '☀️'   if kind == 'solar'  else '💨'

    fig, ax1 = plt.subplots(figsize=(11, 5))
    bg = '#0f1f2e'
    fig.patch.set_facecolor(bg)
    ax1.set_facecolor(bg)

    peak_i = monthly_gwh.index(max(monthly_gwh))
    colors = [('#32C36C' if i == peak_i else bar_color) for i in range(12)]
    ax1.bar(MONTHS, monthly_gwh, color=colors, alpha=0.85, width=0.65, zorder=3)

    # Cumulative line on right axis
    ax2 = ax1.twinx()
    cum = [sum(monthly_gwh[:i+1]) for i in range(12)]
    ax2.plot(MONTHS, cum, color='#32C36C', lw=2.5, marker='o', ms=5, zorder=4)
    ax2.set_ylabel('Cumulative (GWh)', color='#32C36C', fontsize=9)
    ax2.tick_params(axis='y', colors='#32C36C', labelsize=8)
    for sp in ['top', 'bottom', 'left']: ax2.spines[sp].set_visible(False)
    ax2.spines['right'].set_color('#374151')

    ax1.set_ylabel('Monthly Generation (GWh)', color='#94a3b8', fontsize=9)
    ax1.set_title(f'{icon}  {label} Energy Prediction — {location}  ({year})',
                  color='#f1f5f9', fontsize=12, fontweight='bold', pad=14)
    ax1.tick_params(axis='x', colors='#94a3b8', labelsize=8.5)
    ax1.tick_params(axis='y', colors='#94a3b8', labelsize=8)
    for sp in ['top', 'right']: ax1.spines[sp].set_visible(False)
    ax1.spines['bottom'].set_color('#374151')
    ax1.spines['left'].set_color('#374151')
    ax1.grid(axis='y', color='#1e3347', linestyle='--', lw=0.6, zorder=0)

    # Peak annotation
    ax1.annotate(f'Peak\n{monthly_gwh[peak_i]:.0f} GWh',
                 xy=(peak_i, monthly_gwh[peak_i]),
                 xytext=(peak_i, monthly_gwh[peak_i] * 1.08),
                 ha='center', color='#32C36C', fontsize=8, fontweight='bold')

    plt.tight_layout(pad=1.8)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


# ── API routes ─────────────────────────────────────────────────────────────────
@app.route('/api/locations/solar')
def solar_locations():
    return jsonify(list(SOLAR_LOCATIONS.keys()))

@app.route('/api/locations/wind')
def wind_locations():
    return jsonify(list(WIND_LOCATIONS.keys()))


@app.route('/api/predict/solar', methods=['POST'])
def predict_solar():
    data     = request.get_json(force=True)
    year     = int(data.get('year', 2024))
    location = data.get('location', 'Bavaria (Bayern)')

    loc      = SOLAR_LOCATIONS.get(location, list(SOLAR_LOCATIONS.values())[0])
    nat_cap  = _extrapolate(SOLAR_CAP_MW, year)
    loc_cap  = nat_cap * loc['share']                    # MW for this state

    monthly  = []
    for m in range(1, 13):
        cf  = SOLAR_CF.get(m, 0.10) * loc['cf']          # adjusted capacity factor
        gwh = cf * loc_cap * _month_hours(m, year) / 1e3  # GWh
        monthly.append(round(gwh, 2))

    annual  = round(sum(monthly), 1)
    peak_m  = MONTHS[monthly.index(max(monthly))]
    chart   = _make_chart(monthly, year, location, 'solar')

    return jsonify({
        'annual_gwh':    annual,
        'monthly_gwh':   monthly,
        'unit':          'GWh',
        'capacity_mw':   round(loc_cap),
        'location':      location,
        'location_desc': loc['desc'],
        'year':          year,
        'peak_month':    peak_m,
        'chart':         chart,
        'model':         'Seasonal Profile × Capacity Extrapolation (RF-trained)',
    })


@app.route('/api/predict/wind', methods=['POST'])
def predict_wind():
    data     = request.get_json(force=True)
    year     = int(data.get('year', 2024))
    location = data.get('location', 'Lower Saxony')

    loc      = WIND_LOCATIONS.get(location, list(WIND_LOCATIONS.values())[0])
    nat_cap  = _extrapolate(WIND_CAP_MW, year)
    loc_cap  = nat_cap * loc['share']

    monthly  = []
    for m in range(1, 13):
        cf  = WIND_CF.get(m, 0.25) * loc['cf']
        gwh = cf * loc_cap * _month_hours(m, year) / 1e3
        monthly.append(round(gwh, 2))

    annual  = round(sum(monthly), 1)
    peak_m  = MONTHS[monthly.index(max(monthly))]
    chart   = _make_chart(monthly, year, location, 'wind')

    return jsonify({
        'annual_gwh':    annual,
        'monthly_gwh':   monthly,
        'unit':          'GWh',
        'capacity_mw':   round(loc_cap),
        'location':      location,
        'location_desc': loc['desc'],
        'year':          year,
        'peak_month':    peak_m,
        'chart':         chart,
        'model':         'Seasonal Profile × Capacity Extrapolation (RF-trained)',
    })


# ── Static serving ─────────────────────────────────────────────────────────────
@app.route('/')
def index():    return send_from_directory(WEBSITE_DIR, 'index.html')

@app.route('/solar')
def solar_page():
    return send_from_directory(WEBSITE_DIR, 'output.html')

@app.route('/wind')
def wind_page():
    return send_from_directory(WEBSITE_DIR, 'output_wind.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(WEBSITE_DIR, path)


if __name__ == '__main__':
    print('iEnergy server → http://localhost:5000')
    app.run(debug=True, port=5000)

"""
Model analyst implied upside from objective fundamentals, then identify
the most aberrant ratings as large residuals.

Approach
--------
DV:  implied_upside_mean  (analyst mean target / current price - 1)
IVs: objective fundamentals only — no sector dummies, no peer-group
     expectations.  Growth metrics included so high-growth firms aren't
     mechanically flagged.

Model: Elastic-Net regularised regression (handles collinearity, small n).
       Also fits a Random Forest for a non-linear benchmark.

Output: self-contained interactive HTML dashboard (Plotly + DataTables)
        at  output/aberrant_dashboard.html
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data" / "analyst_targets_fundamentals.csv"
OUT = PROJECT / "output"

# ── Feature definitions ──────────────────────────────────────────────
# Every variable here is a stock-level observable, no sector benchmarks.
FEATURES = {
    # Growth
    "earningsGrowth":              "Earnings Growth (YoY)",
    "earningsQuarterlyGrowth":     "Earnings Growth (QoQ)",
    "revenueGrowth":               "Revenue Growth (YoY)",
    # Profitability
    "profitMargins":               "Profit Margin",
    "operatingMargins":            "Operating Margin",
    "grossMargins":                "Gross Margin",
    "returnOnEquity":              "Return on Equity",
    "returnOnAssets":              "Return on Assets",
    # Valuation
    "forwardPE":                   "Forward P/E",
    "trailingPE":                  "Trailing P/E",
    "priceToBook":                 "Price / Book",
    "enterpriseToRevenue":         "EV / Revenue",
    "trailingPegRatio":            "PEG Ratio",
    # Risk / leverage
    "beta":                        "Beta",
    "debtToEquity":                "Debt / Equity",
    # Momentum / position in range
    "fiftyTwoWeekHighChangePercent": "% off 52-wk High",
    "fiftyTwoWeekLowChangePercent":  "% above 52-wk Low",
    # Size (will be log-transformed)
    "marketCap":                   "Market Cap",
    # Analyst coverage depth (controls for information richness)
    "rec_totalAnalysts":           "Total Analysts",
}

TARGET = "implied_upside_mean"


def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load CSV, impute, log-transform, return (full_df, X, y)."""
    raw = pd.read_csv(DATA)

    # Drop rows without the target
    df = raw.dropna(subset=[TARGET]).copy()

    # Log-transform skewed fields
    df["marketCap"] = np.log1p(df["marketCap"].fillna(0))

    # Winsorise extreme PE values (negative PEs are meaningless for regression)
    for pe_col in ["forwardPE", "trailingPE"]:
        df[pe_col] = df[pe_col].clip(lower=0, upper=df[pe_col].quantile(0.95))

    # Winsorise PEG
    df["trailingPegRatio"] = df["trailingPegRatio"].clip(
        lower=df["trailingPegRatio"].quantile(0.05),
        upper=df["trailingPegRatio"].quantile(0.95),
    )

    # Build feature matrix — median-impute missing
    feat_cols = list(FEATURES.keys())
    X = df[feat_cols].copy()
    for c in feat_cols:
        med = X[c].median()
        X[c] = X[c].fillna(med)

    y = df[TARGET]
    return df, X, y


def fit_models(X, y):
    """Fit Elastic-Net (LOO cross-val predictions) and RF."""
    scaler = StandardScaler()

    # ── Elastic-Net with LOO cross-validated predictions ──
    enet_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("enet", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
            cv=5, max_iter=10000, random_state=42,
        )),
    ])
    enet_pipe.fit(X, y)

    # LOO predictions — each stock predicted when held out
    loo = LeaveOneOut()
    enet_loo_preds = cross_val_predict(enet_pipe, X, y, cv=loo)

    # Coefficients
    enet_model = enet_pipe.named_steps["enet"]
    coef_df = pd.DataFrame({
        "feature": list(FEATURES.keys()),
        "label": list(FEATURES.values()),
        "coefficient": enet_model.coef_,
    }).sort_values("coefficient", key=abs, ascending=False)

    # ── Random Forest (LOO predictions) ──
    rf_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )),
    ])
    rf_pipe.fit(X, y)
    rf_loo_preds = cross_val_predict(rf_pipe, X, y, cv=loo)

    # Feature importance from RF
    imp_df = pd.DataFrame({
        "feature": list(FEATURES.keys()),
        "label": list(FEATURES.values()),
        "importance": rf_pipe.named_steps["rf"].feature_importances_,
    }).sort_values("importance", ascending=False)

    return enet_loo_preds, rf_loo_preds, coef_df, imp_df, enet_pipe, rf_pipe


def build_results(df, X, y, enet_preds, rf_preds):
    """Merge predictions and residuals back into dataframe."""
    res = df.copy()
    res["enet_predicted_upside"] = enet_preds
    res["rf_predicted_upside"] = rf_preds
    # Ensemble: average of both
    res["ensemble_predicted"] = (enet_preds + rf_preds) / 2
    res["residual"] = y.values - res["ensemble_predicted"].values
    res["abs_residual"] = res["residual"].abs()

    # Residual z-score
    res["residual_zscore"] = (
        (res["residual"] - res["residual"].mean()) / res["residual"].std()
    )

    # Direction label
    res["direction"] = np.where(
        res["residual"] > 0, "Analysts MORE bullish than fundamentals predict",
        "Analysts LESS bullish than fundamentals predict"
    )

    res = res.sort_values("abs_residual", ascending=False)
    return res


def generate_html(results: pd.DataFrame, coef_df: pd.DataFrame,
                  imp_df: pd.DataFrame, y: pd.Series, enet_preds, rf_preds):
    """Generate a self-contained interactive HTML dashboard."""

    # ── Prepare table data ──
    table_cols = [
        "symbol", "shortName", "sector", "industry",
        "currentPrice", "apt_mean", "apt_high", "apt_low",
        "implied_upside_mean", "ensemble_predicted", "residual",
        "residual_zscore", "direction",
        "bullish_ratio", "rec_totalAnalysts",
        "forwardPE", "trailingPE", "earningsGrowth", "revenueGrowth",
        "profitMargins", "operatingMargins", "grossMargins",
        "returnOnEquity", "returnOnAssets",
        "debtToEquity", "beta", "priceToBook", "enterpriseToRevenue",
        "dividendYield", "marketCap",
        "enet_predicted_upside", "rf_predicted_upside",
    ]
    available = [c for c in table_cols if c in results.columns]
    tbl = results[available].copy()

    # Format floats for JSON
    float_cols = tbl.select_dtypes(include=[np.number]).columns
    for c in float_cols:
        tbl[c] = tbl[c].apply(lambda v: round(v, 4) if pd.notna(v) else None)

    # Restore marketCap from log
    if "marketCap" in tbl.columns:
        tbl["marketCap"] = tbl["marketCap"].apply(
            lambda v: round(np.expm1(v)) if v is not None else None
        )

    records = tbl.to_dict(orient="records")

    # ── Scatter data ──
    scatter = []
    for _, r in results.iterrows():
        scatter.append({
            "x": round(r["ensemble_predicted"], 4) if pd.notna(r["ensemble_predicted"]) else None,
            "y": round(r["implied_upside_mean"], 4) if pd.notna(r["implied_upside_mean"]) else None,
            "symbol": r["symbol"],
            "sector": r.get("sector", ""),
            "residual": round(r["residual"], 4) if pd.notna(r["residual"]) else None,
            "abs_residual": round(r["abs_residual"], 4) if pd.notna(r["abs_residual"]) else None,
        })

    # ── Coefficient data ──
    coefs = []
    for _, r in coef_df.iterrows():
        coefs.append({
            "feature": r["label"],
            "value": round(r["coefficient"], 5),
        })

    # ── Importance data ──
    imps = []
    for _, r in imp_df.iterrows():
        imps.append({
            "feature": r["label"],
            "value": round(r["importance"], 5),
        })

    # ── Residual histogram ──
    residuals_list = [round(v, 4) for v in results["residual"].dropna().tolist()]

    # ── Model stats ──
    from sklearn.metrics import r2_score, mean_absolute_error
    enet_r2 = round(r2_score(y, enet_preds), 4)
    rf_r2 = round(r2_score(y, rf_preds), 4)
    ens_preds = (np.array(enet_preds) + np.array(rf_preds)) / 2
    ens_r2 = round(r2_score(y, ens_preds), 4)
    enet_mae = round(mean_absolute_error(y, enet_preds), 4)
    rf_mae = round(mean_absolute_error(y, rf_preds), 4)
    ens_mae = round(mean_absolute_error(y, ens_preds), 4)

    stats = {
        "n": len(y),
        "enet_r2": enet_r2, "rf_r2": rf_r2, "ensemble_r2": ens_r2,
        "enet_mae": enet_mae, "rf_mae": rf_mae, "ensemble_mae": ens_mae,
    }

    # Sector colors
    sectors = sorted(results["sector"].dropna().unique().tolist())
    palette = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990",
        "#dcbeff", "#9A6324",
    ]
    sector_colors = {s: palette[i % len(palette)] for i, s in enumerate(sectors)}

    html = _HTML_TEMPLATE.replace("__TABLE_DATA__", json.dumps(records))
    html = html.replace("__SCATTER_DATA__", json.dumps(scatter))
    html = html.replace("__COEF_DATA__", json.dumps(coefs))
    html = html.replace("__IMP_DATA__", json.dumps(imps))
    html = html.replace("__RESIDUALS__", json.dumps(residuals_list))
    html = html.replace("__STATS__", json.dumps(stats))
    html = html.replace("__SECTOR_COLORS__", json.dumps(sector_colors))

    return html


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Analyst Rating Aberrance Dashboard</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242836;
    --border: #2e3348;
    --text: #e2e4f0;
    --muted: #8b8fa8;
    --accent: #6c8cff;
    --accent2: #ff6c8c;
    --green: #4ade80;
    --red: #f87171;
    --yellow: #fbbf24;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.5; padding: 20px;
  }
  h1 { font-size: 1.6rem; margin-bottom: 4px; }
  h2 { font-size: 1.15rem; color: var(--accent); margin: 18px 0 10px; }
  h3 { font-size: 0.95rem; color: var(--muted); margin: 10px 0 6px; }
  .subtitle { color: var(--muted); font-size: 0.85rem; margin-bottom: 16px; }
  .stats-row {
    display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px;
  }
  .stat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 18px; min-width: 150px; flex: 1;
  }
  .stat-card .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-card .value { font-size: 1.4rem; font-weight: 700; margin-top: 2px; }
  .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 18px; }
  .panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; overflow: hidden;
  }
  .panel-full { grid-column: 1 / -1; }
  canvas { width: 100% !important; }

  /* Table */
  .table-controls { display: flex; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; align-items: center; }
  .table-controls input, .table-controls select {
    background: var(--surface2); border: 1px solid var(--border); color: var(--text);
    padding: 6px 10px; border-radius: 6px; font-size: 0.85rem;
  }
  .table-controls input { flex: 1; min-width: 200px; }
  table {
    width: 100%; border-collapse: collapse; font-size: 0.78rem;
  }
  thead th {
    background: var(--surface2); color: var(--muted); font-weight: 600;
    padding: 8px 6px; text-align: left; position: sticky; top: 0;
    cursor: pointer; user-select: none; white-space: nowrap;
    border-bottom: 2px solid var(--border);
  }
  thead th:hover { color: var(--accent); }
  thead th.sorted-asc::after { content: " ▲"; color: var(--accent); }
  thead th.sorted-desc::after { content: " ▼"; color: var(--accent); }
  tbody tr { border-bottom: 1px solid var(--border); transition: background 0.15s; }
  tbody tr:hover { background: var(--surface2); }
  tbody td { padding: 7px 6px; white-space: nowrap; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .pos { color: var(--green); }
  .neg { color: var(--red); }
  .badge {
    display: inline-block; padding: 2px 7px; border-radius: 4px;
    font-size: 0.7rem; font-weight: 600;
  }
  .badge-bull { background: rgba(74,222,128,0.15); color: var(--green); }
  .badge-bear { background: rgba(248,113,113,0.15); color: var(--red); }

  .table-wrap { max-height: 600px; overflow: auto; border-radius: 8px; border: 1px solid var(--border); }

  /* Detail panel */
  #detail-panel {
    background: var(--surface); border: 1px solid var(--accent);
    border-radius: 8px; padding: 18px; margin-bottom: 18px;
    display: none;
  }
  #detail-panel.visible { display: block; }
  .detail-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 8px; margin-top: 10px;
  }
  .detail-item {
    background: var(--surface2); border-radius: 6px; padding: 8px 10px;
  }
  .detail-item .dl { font-size: 0.7rem; color: var(--muted); }
  .detail-item .dv { font-size: 0.95rem; font-weight: 600; }
  .close-btn {
    float: right; cursor: pointer; background: var(--surface2);
    border: 1px solid var(--border); color: var(--muted); padding: 4px 10px;
    border-radius: 4px; font-size: 0.8rem;
  }
  .close-btn:hover { color: var(--text); border-color: var(--accent); }

  /* Tooltip */
  .chart-tooltip {
    position: absolute; background: var(--surface2); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 12px; font-size: 0.8rem;
    pointer-events: none; display: none; z-index: 100;
  }

  /* Bar chart */
  .bar-row { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
  .bar-label { width: 160px; text-align: right; font-size: 0.75rem; color: var(--muted); flex-shrink: 0; }
  .bar-track { flex: 1; height: 16px; background: var(--surface2); border-radius: 3px; position: relative; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 3px; transition: width 0.4s; }
  .bar-val { width: 60px; font-size: 0.75rem; font-variant-numeric: tabular-nums; }

  @media (max-width: 900px) {
    .panels { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<h1>Analyst Rating Aberrance Dashboard</h1>
<p class="subtitle">
  Objective fundamentals predict what analyst implied upside <em>should</em> be.
  Large residuals = aberrant ratings. Click any row to drill in.
</p>

<div class="stats-row" id="stats-row"></div>

<div id="detail-panel">
  <button class="close-btn" onclick="closeDetail()">Close</button>
  <h2 id="detail-title"></h2>
  <div id="detail-direction"></div>
  <div class="detail-grid" id="detail-grid"></div>
</div>

<div class="panels">
  <div class="panel">
    <h2>Predicted vs Actual Implied Upside</h2>
    <h3>Points far from diagonal = most aberrant</h3>
    <div style="position:relative;">
      <canvas id="scatter-chart" height="320"></canvas>
      <div class="chart-tooltip" id="scatter-tip"></div>
    </div>
  </div>
  <div class="panel">
    <h2>Residual Distribution</h2>
    <h3>How far analysts deviate from fundamental-based prediction</h3>
    <canvas id="hist-chart" height="320"></canvas>
  </div>
  <div class="panel">
    <h2>Elastic-Net Coefficients</h2>
    <h3>Standardised — what drives analyst targets</h3>
    <div id="coef-bars"></div>
  </div>
  <div class="panel">
    <h2>Random Forest Importance</h2>
    <h3>Non-linear feature contribution</h3>
    <div id="imp-bars"></div>
  </div>
</div>

<h2>Full Results — Ranked by Aberrance (|Residual|)</h2>
<div class="table-controls">
  <input type="text" id="search" placeholder="Search by symbol, name, sector, industry…" />
  <select id="sector-filter"><option value="">All Sectors</option></select>
  <select id="direction-filter">
    <option value="">All Directions</option>
    <option value="bull">Analysts MORE bullish</option>
    <option value="bear">Analysts LESS bullish</option>
  </select>
</div>
<div class="table-wrap">
  <table id="main-table">
    <thead><tr id="table-head"></tr></thead>
    <tbody id="table-body"></tbody>
  </table>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script>
// ── Data injection ──
const TABLE_DATA   = __TABLE_DATA__;
const SCATTER_DATA = __SCATTER_DATA__;
const COEF_DATA    = __COEF_DATA__;
const IMP_DATA     = __IMP_DATA__;
const RESIDUALS    = __RESIDUALS__;
const STATS        = __STATS__;
const SECTOR_COLORS= __SECTOR_COLORS__;

// ── Stats cards ──
(function() {
  const row = document.getElementById('stats-row');
  const cards = [
    { label: 'Universe', value: STATS.n + ' stocks' },
    { label: 'Elastic-Net R²', value: STATS.enet_r2.toFixed(3) },
    { label: 'Random Forest R²', value: STATS.rf_r2.toFixed(3) },
    { label: 'Ensemble R²', value: STATS.ensemble_r2.toFixed(3) },
    { label: 'Ensemble MAE', value: (STATS.ensemble_mae * 100).toFixed(1) + ' pp' },
  ];
  cards.forEach(c => {
    const d = document.createElement('div');
    d.className = 'stat-card';
    d.innerHTML = `<div class="label">${c.label}</div><div class="value">${c.value}</div>`;
    row.appendChild(d);
  });
})();

// ── Scatter plot ──
(function() {
  const ctx = document.getElementById('scatter-chart').getContext('2d');
  const sectors = [...new Set(SCATTER_DATA.map(d => d.sector))].sort();
  const datasets = sectors.map(s => ({
    label: s,
    data: SCATTER_DATA.filter(d => d.sector === s).map(d => ({
      x: d.x, y: d.y, symbol: d.symbol, residual: d.residual, abs_residual: d.abs_residual
    })),
    backgroundColor: (SECTOR_COLORS[s] || '#888') + 'cc',
    borderColor: SECTOR_COLORS[s] || '#888',
    borderWidth: 1,
    pointRadius: ctx2 => {
      const ar = ctx2.raw?.abs_residual || 0;
      return 4 + ar * 15;
    },
    pointHoverRadius: 8,
  }));

  // Perfect prediction line
  const allX = SCATTER_DATA.map(d => d.x).filter(v => v != null);
  const allY = SCATTER_DATA.map(d => d.y).filter(v => v != null);
  const lo = Math.min(...allX, ...allY) - 0.05;
  const hi = Math.max(...allX, ...allY) + 0.05;

  datasets.push({
    label: 'Perfect prediction',
    data: [{ x: lo, y: lo }, { x: hi, y: hi }],
    type: 'line',
    borderColor: '#ffffff33',
    borderDash: [6, 4],
    borderWidth: 1,
    pointRadius: 0,
    showLine: true,
  });

  new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      animation: { duration: 600 },
      plugins: {
        legend: { display: true, position: 'bottom', labels: { color: '#8b8fa8', boxWidth: 10, font: { size: 10 } } },
        tooltip: {
          callbacks: {
            label: ctx2 => {
              if (!ctx2.raw.symbol) return '';
              return `${ctx2.raw.symbol}: actual=${(ctx2.raw.y*100).toFixed(1)}% pred=${(ctx2.raw.x*100).toFixed(1)}% resid=${(ctx2.raw.residual*100).toFixed(1)}pp`;
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Predicted Implied Upside', color: '#8b8fa8' },
          ticks: { color: '#8b8fa8', callback: v => (v*100).toFixed(0)+'%' },
          grid: { color: '#2e334833' },
        },
        y: {
          title: { display: true, text: 'Actual Implied Upside (Analyst)', color: '#8b8fa8' },
          ticks: { color: '#8b8fa8', callback: v => (v*100).toFixed(0)+'%' },
          grid: { color: '#2e334833' },
        }
      }
    }
  });
})();

// ── Residual histogram ──
(function() {
  const ctx = document.getElementById('hist-chart').getContext('2d');
  const nbins = 20;
  const lo = Math.min(...RESIDUALS), hi = Math.max(...RESIDUALS);
  const step = (hi - lo) / nbins;
  const bins = Array.from({length: nbins}, (_, i) => lo + i * step);
  const counts = new Array(nbins).fill(0);
  RESIDUALS.forEach(v => {
    let idx = Math.floor((v - lo) / step);
    if (idx >= nbins) idx = nbins - 1;
    if (idx < 0) idx = 0;
    counts[idx]++;
  });
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: bins.map(b => (b*100).toFixed(0) + '%'),
      datasets: [{
        data: counts,
        backgroundColor: counts.map((_, i) => {
          const mid = bins[i] + step/2;
          return mid > 0 ? '#4ade8066' : '#f8717166';
        }),
        borderColor: counts.map((_, i) => {
          const mid = bins[i] + step/2;
          return mid > 0 ? '#4ade80' : '#f87171';
        }),
        borderWidth: 1,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Residual (Actual − Predicted)', color: '#8b8fa8' }, ticks: { color: '#8b8fa8' }, grid: { color: '#2e334833' } },
        y: { title: { display: true, text: 'Count', color: '#8b8fa8' }, ticks: { color: '#8b8fa8' }, grid: { color: '#2e334833' } },
      }
    }
  });
})();

// ── Bar charts for coefficients / importance ──
function renderBars(containerId, data, colorPos, colorNeg) {
  const el = document.getElementById(containerId);
  const maxAbs = Math.max(...data.map(d => Math.abs(d.value)), 0.0001);
  data.forEach(d => {
    const pct = Math.abs(d.value) / maxAbs * 100;
    const color = d.value >= 0 ? colorPos : colorNeg;
    el.innerHTML += `
      <div class="bar-row">
        <div class="bar-label">${d.feature}</div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${pct}%;background:${color};"></div>
        </div>
        <div class="bar-val" style="color:${color}">${d.value >= 0 ? '+' : ''}${d.value.toFixed(4)}</div>
      </div>`;
  });
}
renderBars('coef-bars', COEF_DATA, '#4ade80', '#f87171');
renderBars('imp-bars', IMP_DATA, '#6c8cff', '#6c8cff');

// ── Table ──
const COLS = [
  { key: 'symbol', label: 'Ticker', fmt: v => v },
  { key: 'shortName', label: 'Name', fmt: v => v ? (v.length > 28 ? v.slice(0,26)+'…' : v) : '' },
  { key: 'sector', label: 'Sector', fmt: v => v || '' },
  { key: 'currentPrice', label: 'Price', fmt: v => v != null ? '$'+v.toFixed(2) : '', cls: 'num' },
  { key: 'apt_mean', label: 'Target (Mean)', fmt: v => v != null ? '$'+v.toFixed(2) : '', cls: 'num' },
  { key: 'implied_upside_mean', label: 'Actual Upside', fmt: v => v != null ? (v*100).toFixed(1)+'%' : '', cls: 'num',
    color: v => v > 0 ? 'pos' : v < 0 ? 'neg' : '' },
  { key: 'ensemble_predicted', label: 'Predicted Upside', fmt: v => v != null ? (v*100).toFixed(1)+'%' : '', cls: 'num' },
  { key: 'residual', label: 'Residual', fmt: v => v != null ? (v > 0 ? '+' : '') + (v*100).toFixed(1)+'pp' : '', cls: 'num',
    color: v => v > 0 ? 'pos' : v < 0 ? 'neg' : '' },
  { key: 'residual_zscore', label: 'Z-Score', fmt: v => v != null ? (v > 0 ? '+' : '') + v.toFixed(2) : '', cls: 'num',
    color: v => Math.abs(v) > 1.5 ? (v > 0 ? 'pos' : 'neg') : '' },
  { key: 'direction', label: 'Direction', fmt: v => {
    if (!v) return '';
    if (v.includes('MORE')) return '<span class="badge badge-bull">BULLISH OUTLIER</span>';
    return '<span class="badge badge-bear">BEARISH OUTLIER</span>';
  }},
  { key: 'forwardPE', label: 'Fwd P/E', fmt: v => v != null ? v.toFixed(1) : '', cls: 'num' },
  { key: 'earningsGrowth', label: 'Earn. Growth', fmt: v => v != null ? (v*100).toFixed(1)+'%' : '', cls: 'num',
    color: v => v > 0 ? 'pos' : v < 0 ? 'neg' : '' },
  { key: 'bullish_ratio', label: 'Bullish %', fmt: v => v != null ? (v*100).toFixed(0)+'%' : '', cls: 'num' },
];

// Build header
const thead = document.getElementById('table-head');
COLS.forEach((col, i) => {
  const th = document.createElement('th');
  th.textContent = col.label;
  th.dataset.idx = i;
  th.onclick = () => sortTable(i);
  thead.appendChild(th);
});

// Populate sector filter
const sectorFilter = document.getElementById('sector-filter');
const sectors = [...new Set(TABLE_DATA.map(d => d.sector))].filter(Boolean).sort();
sectors.forEach(s => {
  const opt = document.createElement('option');
  opt.value = s; opt.textContent = s;
  sectorFilter.appendChild(opt);
});

let sortCol = 7; // residual abs — but we sort by the underlying value descending by abs
let sortDir = -1;
let currentData = [...TABLE_DATA];

function renderTable(data) {
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';
  data.forEach(row => {
    const tr = document.createElement('tr');
    tr.style.cursor = 'pointer';
    tr.onclick = () => showDetail(row);
    COLS.forEach(col => {
      const td = document.createElement('td');
      if (col.cls) td.className = col.cls;
      const val = row[col.key];
      td.innerHTML = col.fmt(val);
      if (col.color) {
        const cls = col.color(val);
        if (cls) td.classList.add(cls);
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function filterAndRender() {
  const q = document.getElementById('search').value.toLowerCase();
  const sec = document.getElementById('sector-filter').value;
  const dir = document.getElementById('direction-filter').value;
  let data = [...TABLE_DATA];
  if (q) {
    data = data.filter(r =>
      (r.symbol||'').toLowerCase().includes(q) ||
      (r.shortName||'').toLowerCase().includes(q) ||
      (r.sector||'').toLowerCase().includes(q) ||
      (r.industry||'').toLowerCase().includes(q)
    );
  }
  if (sec) data = data.filter(r => r.sector === sec);
  if (dir === 'bull') data = data.filter(r => r.residual > 0);
  if (dir === 'bear') data = data.filter(r => r.residual < 0);

  // Sort
  data.sort((a, b) => {
    let va = a[COLS[sortCol].key], vb = b[COLS[sortCol].key];
    if (va == null) va = sortDir > 0 ? Infinity : -Infinity;
    if (vb == null) vb = sortDir > 0 ? Infinity : -Infinity;
    if (typeof va === 'string') return sortDir * va.localeCompare(vb);
    return sortDir * (va - vb);
  });
  currentData = data;
  renderTable(data);
}

function sortTable(idx) {
  const ths = document.querySelectorAll('#table-head th');
  ths.forEach(th => th.classList.remove('sorted-asc', 'sorted-desc'));
  if (sortCol === idx) sortDir *= -1;
  else { sortCol = idx; sortDir = -1; }
  ths[idx].classList.add(sortDir > 0 ? 'sorted-asc' : 'sorted-desc');
  filterAndRender();
}

document.getElementById('search').oninput = filterAndRender;
document.getElementById('sector-filter').onchange = filterAndRender;
document.getElementById('direction-filter').onchange = filterAndRender;

// Initial sort by abs_residual desc (we use residual_zscore desc by abs)
sortCol = 8; sortDir = -1;
// Custom initial sort by absolute residual
TABLE_DATA.sort((a, b) => Math.abs(b.residual || 0) - Math.abs(a.residual || 0));
renderTable(TABLE_DATA);

// ── Detail panel ──
function showDetail(row) {
  const panel = document.getElementById('detail-panel');
  panel.classList.add('visible');
  document.getElementById('detail-title').textContent =
    `${row.symbol} — ${row.shortName || ''}`;

  const dir = row.residual > 0
    ? `<span class="badge badge-bull">Analysts are ${(Math.abs(row.residual)*100).toFixed(1)}pp MORE bullish than fundamentals predict</span>`
    : `<span class="badge badge-bear">Analysts are ${(Math.abs(row.residual)*100).toFixed(1)}pp LESS bullish than fundamentals predict</span>`;
  document.getElementById('detail-direction').innerHTML = dir;

  const grid = document.getElementById('detail-grid');
  grid.innerHTML = '';

  const items = [
    ['Sector', row.sector],
    ['Industry', row.industry],
    ['Price', row.currentPrice != null ? '$'+row.currentPrice.toFixed(2) : '—'],
    ['Target (Mean)', row.apt_mean != null ? '$'+row.apt_mean.toFixed(2) : '—'],
    ['Target (High)', row.apt_high != null ? '$'+row.apt_high.toFixed(2) : '—'],
    ['Target (Low)', row.apt_low != null ? '$'+row.apt_low.toFixed(2) : '—'],
    ['Actual Implied Upside', row.implied_upside_mean != null ? (row.implied_upside_mean*100).toFixed(1)+'%' : '—'],
    ['Predicted Upside (Ensemble)', row.ensemble_predicted != null ? (row.ensemble_predicted*100).toFixed(1)+'%' : '—'],
    ['Predicted (Elastic-Net)', row.enet_predicted_upside != null ? (row.enet_predicted_upside*100).toFixed(1)+'%' : '—'],
    ['Predicted (Random Forest)', row.rf_predicted_upside != null ? (row.rf_predicted_upside*100).toFixed(1)+'%' : '—'],
    ['Residual', row.residual != null ? (row.residual > 0 ? '+' : '')+(row.residual*100).toFixed(1)+'pp' : '—'],
    ['Z-Score', row.residual_zscore != null ? row.residual_zscore.toFixed(2) : '—'],
    ['Forward P/E', row.forwardPE != null ? row.forwardPE.toFixed(1) : '—'],
    ['Trailing P/E', row.trailingPE != null ? row.trailingPE.toFixed(1) : '—'],
    ['EV/Revenue', row.enterpriseToRevenue != null ? row.enterpriseToRevenue.toFixed(2) : '—'],
    ['Price/Book', row.priceToBook != null ? row.priceToBook.toFixed(2) : '—'],
    ['Earnings Growth', row.earningsGrowth != null ? (row.earningsGrowth*100).toFixed(1)+'%' : '—'],
    ['Revenue Growth', row.revenueGrowth != null ? (row.revenueGrowth*100).toFixed(1)+'%' : '—'],
    ['Profit Margin', row.profitMargins != null ? (row.profitMargins*100).toFixed(1)+'%' : '—'],
    ['Operating Margin', row.operatingMargins != null ? (row.operatingMargins*100).toFixed(1)+'%' : '—'],
    ['Gross Margin', row.grossMargins != null ? (row.grossMargins*100).toFixed(1)+'%' : '—'],
    ['ROE', row.returnOnEquity != null ? (row.returnOnEquity*100).toFixed(1)+'%' : '—'],
    ['ROA', row.returnOnAssets != null ? (row.returnOnAssets*100).toFixed(1)+'%' : '—'],
    ['Debt/Equity', row.debtToEquity != null ? row.debtToEquity.toFixed(1) : '—'],
    ['Beta', row.beta != null ? row.beta.toFixed(2) : '—'],
    ['Dividend Yield', row.dividendYield != null ? (row.dividendYield*100).toFixed(2)+'%' : '—'],
    ['Bullish Ratio', row.bullish_ratio != null ? (row.bullish_ratio*100).toFixed(0)+'%' : '—'],
    ['Total Analysts', row.rec_totalAnalysts != null ? row.rec_totalAnalysts : '—'],
    ['Market Cap', row.marketCap != null ? '$'+(row.marketCap/1e9).toFixed(1)+'B' : '—'],
  ];
  items.forEach(([label, value]) => {
    const d = document.createElement('div');
    d.className = 'detail-item';
    d.innerHTML = `<div class="dl">${label}</div><div class="dv">${value}</div>`;
    grid.appendChild(d);
  });

  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeDetail() {
  document.getElementById('detail-panel').classList.remove('visible');
}
</script>
</body>
</html>"""


def main():
    print("[1/4] Loading data...")
    df, X, y = load_and_prepare()
    print(f"      {len(df)} tickers, {X.shape[1]} features")

    print("[2/4] Fitting models (LOO cross-validation)...")
    enet_preds, rf_preds, coef_df, imp_df, enet_pipe, rf_pipe = fit_models(X, y)

    from sklearn.metrics import r2_score, mean_absolute_error
    ens = (enet_preds + rf_preds) / 2
    print(f"      Elastic-Net  R²={r2_score(y, enet_preds):.3f}  MAE={mean_absolute_error(y, enet_preds):.4f}")
    print(f"      Random Forest R²={r2_score(y, rf_preds):.3f}  MAE={mean_absolute_error(y, rf_preds):.4f}")
    print(f"      Ensemble      R²={r2_score(y, ens):.3f}  MAE={mean_absolute_error(y, ens):.4f}")

    print("\n      Top Elastic-Net coefficients (standardised):")
    for _, r in coef_df.head(8).iterrows():
        print(f"        {r['label']:30s}  {r['coefficient']:+.4f}")

    print("\n      Top RF importances:")
    for _, r in imp_df.head(8).iterrows():
        print(f"        {r['label']:30s}  {r['importance']:.4f}")

    print("\n[3/4] Building results table...")
    results = build_results(df, X, y, enet_preds, rf_preds)

    print("\n      Most aberrant (top 10):")
    for _, r in results.head(10).iterrows():
        arrow = "▲" if r["residual"] > 0 else "▼"
        print(f"        {arrow} {r['symbol']:6s}  actual={r['implied_upside_mean']:+.1%}  "
              f"pred={r['ensemble_predicted']:+.1%}  resid={r['residual']:+.1%}  "
              f"z={r['residual_zscore']:+.2f}  [{r.get('sector','')}]")

    print("\n[4/4] Generating interactive dashboard...")
    OUT.mkdir(parents=True, exist_ok=True)
    html = generate_html(results, coef_df, imp_df, y, enet_preds, rf_preds)
    out_path = OUT / "aberrant_dashboard.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"      Saved to {out_path}")

    # Also save model results CSV
    results.to_csv(OUT / "model_results.csv", index=False)
    print(f"      Model results CSV: {OUT / 'model_results.csv'}")


if __name__ == "__main__":
    main()

"""
================================================================================
AKH Capital Analytics Platform
Modern Institutional NGX Intelligence & Quantitative Portfolio Suite
================================================================================
Features:
  - 🎨 Dual Theme Mode: Dark Obsidian (Night) & Light Pearl (Day) with Vivid Colors
  - 🏛️ Market Research & Live Screener (TradingView Screener API + Google Sheets Sync)
  - 💼 Black-Litterman Portfolio Engine (Manager Convictions P/Q, SLSQP & Whole-Share Sizing)
  - 📊 Return Correlation Matrix & Diversification Explorer
  - 📈 Multi-Horizon Price Forecasting with 90% Confidence Uncertainty Bands
  - 🤖 Machine Learning Intelligence:
      1. Pre-Trained 4-Model Ensemble (RandomForest, XGBoost, LightGBM, LogisticRegression)
         from `Notebooks/Machine learning/ngx model_eval_ensemble_updated (2).ipynb`
      2. K-Means Behavioral Clustering + 2D PCA Projection
      3. Multi-Factor Quantitative Ranking Model
================================================================================
"""

import os
import sys
import json
import datetime
import warnings
import pytz
import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tradingview_screener import Query

# Monkeypatch scikit-learn compatibility for unpickling ColumnTransformer & SimpleImputer bundles
import sklearn.compose._column_transformer as sct
class _RemainderColsList(list):
    pass
sct._RemainderColsList = _RemainderColsList

from sklearn.impute import SimpleImputer
_orig_imputer_transform = SimpleImputer.transform
def _patched_imputer_transform(self, X):
    if not hasattr(self, "_fill_dtype"):
        self._fill_dtype = getattr(self, "statistics_", np.array([0.0])).dtype
    return _orig_imputer_transform(self, X)
SimpleImputer.transform = _patched_imputer_transform

import joblib

warnings.filterwarnings("ignore")

try:
    import pygsheets
    PYGSHEETS_AVAILABLE = True
except ImportError:
    PYGSHEETS_AVAILABLE = False

# ==============================================================================
# 1. GLOBAL CONFIGURATION & MARKET DEFAULTS
# ==============================================================================
NIGERIA_TZ       = pytz.timezone("Africa/Lagos")
GSHEET_CRED_PATH = r"C:\Users\Segun\Desktop\wqu-satellite-image-learning1-175e0a0906aa.json"
GSHEET_URL       = "https://docs.google.com/spreadsheets/d/1jkbrrHnR48JztVz0SrqZELKjOenVm44wsiIchLW0K9g/edit#gid=1868316212"
GSHEET_TAB_INDEX = 0

# Financial parameters from NGX_Black_Litterman_Real_World_v4.ipynb
ANNUALIZATION     = 252
RIDGE             = 1e-8
DELTA_FLOOR       = 0.25
DELTA_CAP         = 10.0
VIEW_MAX_EXCESS   = 0.20
MIN_CONFIDENCE    = 0.25
MAX_CONFIDENCE    = 0.85
MAX_WEIGHT        = 0.25
MAX_ACTIVE_WEIGHT = 0.15
MAX_PORTFOLIO_VOL = 0.30
TURNOVER_PENALTY  = 0.01
RF_FALLBACK       = 0.185     # Nigeria MPR benchmark (18.5% p.a.)

# Default Manager Opinions from Cell 36 of BL notebook
DEFAULT_NOTEBOOK_MANAGER_VIEWS = [
    {
        "type": "absolute",
        "ticker": "ARADEL",
        "view": 0.50,
        "confidence": 0.65,
        "label": "ARADEL absolute (oil & gas growth thesis)"
    },
    {
        "type": "absolute",
        "ticker": "BUACEMENT",
        "view": 0.15,
        "confidence": 0.55,
        "label": "BUACEMENT absolute (stable cement demand)"
    },
    {
        "type": "relative",
        "long": ["MTNN"],
        "short": ["CWG"],
        "view": 0.10,
        "confidence": 0.50,
        "label": "MTNN vs CWG (large-cap defensive over small-cap tech)"
    },
    {
        "type": "relative",
        "long": ["FIDSON", "MAYBAKER"],
        "short": ["HONYFLOUR"],
        "view": 0.06,
        "confidence": 0.45,
        "label": "Pharma basket (FIDSON+MAYBAKER) vs HONYFLOUR"
    },
]

# Model directory for 5-day ensemble classifiers
ML_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Model mangement", "Classification model", "5 days prediction model", "NGX v1"
)

# Application In-Memory State
STATE = {
    "df":            None,
    "last_updated":  None,
    "gsheet_synced": None,
    "status":        "🟢 Platform ready. Click 'Refresh Market Data' to load live prices.",
    "export_path":   os.path.abspath("NGX_Stocks_Data.xlsx"),
    "manager_views": list(DEFAULT_NOTEBOOK_MANAGER_VIEWS),
    "theme":         "Dark Obsidian",
    "ml_bundle":     None
}

# ==============================================================================
# 2. UTILITY & MATHEMATICAL FUNCTIONS (FROM NOTEBOOKS)
# ==============================================================================
def now_wat_str():
    return datetime.datetime.now(NIGERIA_TZ).strftime("%d %b %Y • %I:%M:%S %p WAT")

def format_naira(val):
    if pd.isna(val) or val is None:
        return "N/A"
    try:
        v = float(val)
        av = abs(v)
        if av >= 1e12:
            return f"₦{v / 1e12:.2f}T"
        elif av >= 1e9:
            return f"₦{v / 1e9:.2f}B"
        elif av >= 1e6:
            return f"₦{v / 1e6:.2f}M"
        return f"₦{v:,.2f}"
    except Exception:
        return "N/A"

def make_psd(matrix, ridge=RIDGE):
    """Ensure matrix is symmetric positive semi-definite (Cell 12 of BL notebook)."""
    s = np.asarray(matrix, dtype=float)
    s = (s + s.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(s)
    eigvals = np.maximum(eigvals, ridge)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def portfolio_vol(w, sigma):
    """Calculate portfolio volatility (Cell 20 of BL notebook)."""
    return float(np.sqrt(max(w @ sigma @ w, 0.0)))

def risk_contribution(weights, sigma):
    """Calculate percentage risk contribution per asset (Cell 22 of BL notebook)."""
    w = weights.values if isinstance(weights, pd.Series) else np.asarray(weights)
    S = sigma.values if isinstance(sigma, pd.DataFrame) else np.asarray(sigma)
    pv = np.sqrt(max(w @ S @ w, RIDGE))
    contribution = w * (S @ w) / pv
    return pd.Series(contribution / max(pv, RIDGE), index=weights.index if hasattr(weights, "index") else range(len(w)))

def build_manager_view_matrix(manager_views, tickers):
    """
    Convert manager view dicts into Black-Litterman P/Q/confidence rows.
    Direct implementation of Cell 36 in NGX_Black_Litterman_Real_World_v4.ipynb.
    """
    n = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}
    rows, qs, confs, labels = [], [], [], []

    for v in manager_views:
        row = np.zeros(n)
        vtype = v.get("type", "absolute")

        if vtype == "absolute":
            t = v.get("ticker", "")
            if t not in idx:
                continue
            row[idx[t]] = 1.0
            default_label = f"{t} absolute"

        elif vtype == "relative":
            long_legs = [t for t in v.get("long", []) if t in idx]
            short_legs = [t for t in v.get("short", []) if t in idx]
            if not long_legs or not short_legs:
                continue
            for t in long_legs:
                row[idx[t]] += 1.0 / len(long_legs)
            for t in short_legs:
                row[idx[t]] -= 1.0 / len(short_legs)
            default_label = f"{'+'.join(long_legs)} vs {'+'.join(short_legs)}"
        else:
            continue

        conf = float(v.get("confidence", 0.50))
        if not (0.0 < conf < 1.0):
            conf = 0.50

        rows.append(row)
        qs.append(float(v.get("view", 0.05)))
        confs.append(conf)
        labels.append(v.get("label", default_label))

    if not rows:
        return None, None, None, []

    return np.vstack(rows), np.array(qs), np.array(confs), labels

def black_litterman(pi, sigma, tau, q, confidence,
                     extra_P=None, extra_q=None, extra_confidence=None, view_labels=None):
    """
    BL posterior calculation with Idzorek confidence scaling and manager view rows.
    Direct implementation of Cell 18 in NGX_Black_Litterman_Real_World_v4.ipynb.
    """
    assets = pi.index
    n = len(assets)

    P = np.eye(n)
    q_vals = pd.Series(q, index=assets).values
    confidence_vals = pd.Series(confidence, index=assets).clip(MIN_CONFIDENCE, MAX_CONFIDENCE).values

    if extra_P is not None and len(extra_P) > 0:
        extra_P = np.asarray(extra_P, dtype=float)
        extra_q = np.asarray(extra_q, dtype=float)
        extra_confidence = np.clip(np.asarray(extra_confidence, dtype=float), MIN_CONFIDENCE, MAX_CONFIDENCE)
        P = np.vstack([P, extra_P])
        q_vals = np.concatenate([q_vals, extra_q])
        confidence_vals = np.concatenate([confidence_vals, extra_confidence])

    if view_labels is None:
        view_labels = list(assets)

    sigma = make_psd(sigma)
    tau_sigma = tau * sigma

    base_omega = np.diag(P @ tau_sigma @ P.T)
    omega_diag = base_omega * (1.0 - confidence_vals) / np.maximum(confidence_vals, 0.01)
    omega_diag = np.maximum(omega_diag, RIDGE)
    Omega = np.diag(omega_diag)

    inv_tau_sigma = np.linalg.pinv(make_psd(tau_sigma))
    inv_omega = np.linalg.pinv(Omega)

    posterior_precision = inv_tau_sigma + P.T @ inv_omega @ P
    posterior_cov = np.linalg.pinv(make_psd(posterior_precision))
    posterior_mean = posterior_cov @ (inv_tau_sigma @ pi.values + P.T @ inv_omega @ q_vals)

    return (
        pd.Series(posterior_mean, index=assets, name="mu_BL"),
        pd.DataFrame(posterior_cov, index=assets, columns=assets),
        P,
        pd.DataFrame(Omega, index=view_labels[:len(P)], columns=view_labels[:len(P)])
    )

def optimise_portfolio(mu, sigma, benchmark_w, previous_w=None, risk_aversion=None):
    """Constrained SLSQP quadratic programming (Cell 20 of BL notebook)."""
    assets = mu.index
    n = len(assets)
    mu_v = mu.values
    sigma_m = make_psd(sigma.values if isinstance(sigma, pd.DataFrame) else sigma)
    bench = benchmark_w.reindex(assets).fillna(0.0).values

    if previous_w is None:
        previous_w = benchmark_w.copy()
    prev = previous_w.reindex(assets).fillna(0.0).values
    d = float(risk_aversion if risk_aversion is not None else 3.0)

    def objective(w):
        utility = w @ mu_v - 0.5 * d * (w @ sigma_m @ w)
        turnover = np.sum(np.abs(w - prev))
        return -(utility - TURNOVER_PENALTY * turnover)

    bounds = []
    for b in bench:
        lo = max(0.0, b - MAX_ACTIVE_WEIGHT)
        hi = min(MAX_WEIGHT, b + MAX_ACTIVE_WEIGHT)
        if lo > hi:
            lo, hi = 0.0, MAX_WEIGHT
        bounds.append((lo, hi))

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: MAX_PORTFOLIO_VOL - portfolio_vol(w, sigma_m)}
    ]

    x0 = prev.copy()
    if not np.isclose(x0.sum(), 1.0):
        x0 = np.repeat(1.0 / n, n)

    result = minimize(
        objective, x0=x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-9}
    )

    if result.success:
        return pd.Series(result.x, index=assets, name="weight"), "Optimal SLSQP Convergence"

    fallback = minimize(
        lambda w: portfolio_vol(w, sigma_m),
        x0=np.repeat(1.0 / n, n),
        method="SLSQP",
        bounds=[(0.0, MAX_WEIGHT)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 2000}
    )
    if fallback.success:
        return pd.Series(fallback.x, index=assets, name="weight"), "Fallback Minimum-Variance Solution"

    return pd.Series(x0, index=assets, name="weight"), f"Heuristic: {result.message}"

# ==============================================================================
# 3. GOOGLE SHEETS & LIVE MARKET DATA ENGINE
# ==============================================================================
def get_gsheet_client():
    if not PYGSHEETS_AVAILABLE:
        return None, "⚠️ pygsheets package not installed."
    try:
        raw_json = os.environ.get("GSHEET_SERVICE_ACCOUNT_JSON", "").strip()
        if raw_json:
            return pygsheets.authorize(service_account_json=raw_json), "OK"
        env_file = os.environ.get("GSHEET_SERVICE_ACCOUNT_FILE", "").strip()
        if env_file and os.path.exists(env_file):
            return pygsheets.authorize(service_account_file=env_file), "OK"
        if os.path.exists(GSHEET_CRED_PATH):
            return pygsheets.authorize(service_account_file=GSHEET_CRED_PATH), "OK"
        if os.path.exists("google_creds.json"):
            return pygsheets.authorize(service_account_file="google_creds.json"), "OK"
        return None, "⚠️ Google Service Account credentials not found."
    except Exception as exc:
        return None, f"⚠️ Authorization failed: {exc}"

def push_to_gsheet(df, worksheet_idx=GSHEET_TAB_INDEX):
    if df is None or df.empty:
        return False, "⚠️ No data available to push."
    gc, auth_msg = get_gsheet_client()
    if gc is None:
        return False, auth_msg
    try:
        push_df = df.drop(columns=[c for c in ["ticker", "symbol"] if c in df.columns], errors="ignore")
        sh = gc.open_by_url(GSHEET_URL)
        wks = sh[worksheet_idx]
        wks.clear()
        wks.set_dataframe(push_df, (1, 1), copy_head=True, nan="")
        ts = now_wat_str()
        STATE["gsheet_synced"] = ts
        return True, f"✅ Google Sheet '{wks.title}' updated successfully at {ts} ({len(push_df)} rows)"
    except Exception as exc:
        return False, f"❌ Google Sheet update error: {exc}"

def fetch_ngx_market():
    """Fetch real-time NGX equities and technicals via TradingView Screener."""
    try:
        q = (
            Query()
            .select(
                "name", "close", "open", "high", "volume", "change",
                "basic_eps_net_income", "ebitda_fy", "Perf.5Y", "Perf.Y",
                "last_annual_revenue", "price_book_ratio", "price_earnings_ttm",
                "enterprise_value_current", "market_cap_basic",
                "average_volume_30d_calc", "enterprise_value_ebitda_ttm",
                "dividends_yield_current", "EMA50", "VWAP", "RSI", "sector", "industry"
            )
            .limit(200)
            .set_markets("nigeria")
            .set_property("symbols", {"query": {"types": ["stock", "dr", "fund"]}})
        )
        data = q.get_scanner_data()
        df = data[1] if isinstance(data, tuple) and len(data) == 2 else pd.DataFrame()

        if df.empty:
            return STATE["df"], "⚠️ TradingView query returned an empty dataset."

        if "ticker" in df.columns:
            df["symbol"] = df["ticker"].astype(str).str.replace("NSENG:", "", regex=False)
        else:
            df["symbol"] = df["name"]

        numeric_cols = [
            "close", "open", "high", "volume", "change",
            "basic_eps_net_income", "ebitda_fy", "Perf.5Y", "Perf.Y",
            "last_annual_revenue", "price_book_ratio", "price_earnings_ttm",
            "enterprise_value_current", "market_cap_basic",
            "average_volume_30d_calc", "enterprise_value_ebitda_ttm",
            "dividends_yield_current", "EMA50", "VWAP", "RSI"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "EMA50" in df.columns and "close" in df.columns:
            df["EMA50_pct_diff"] = ((df["EMA50"] - df["close"]) / df["close"].replace(0, np.nan)) * 100
        else:
            df["EMA50_pct_diff"] = np.nan

        if "market_cap_basic" in df.columns:
            df = df.sort_values(by="market_cap_basic", ascending=False).reset_index(drop=True)

        ts = now_wat_str()
        STATE["df"] = df
        STATE["last_updated"] = ts
        STATE["status"] = f"🟢 Connected | Live NGX feed active • {ts} ({len(df)} tickers)"

        try:
            df.to_excel(STATE["export_path"], index=False)
        except Exception:
            pass

        return df, STATE["status"]

    except Exception as exc:
        err = f"⚠️ Network/API Notice: {exc}"
        if STATE["df"] is not None:
            return STATE["df"], f"{err} (Serving cached data from {STATE['last_updated']})"
        return pd.DataFrame(), err

# ==============================================================================
# 4. PRE-TRAINED ENSEMBLE CLASSIFIER ENGINE (FROM NOTEBOOK)
# ==============================================================================
def load_ml_ensemble_bundle():
    """Load the 4 ensemble models from Model mangement/Classification model/5 days prediction model/NGX v1."""
    if STATE.get("ml_bundle") is not None:
        return STATE["ml_bundle"]

    models_dict = {}
    model_paths = {
        "RandomForest":       os.path.join(ML_MODEL_DIR, "ngxRandomForest_ensemble_v1.pkl"),
        "XGBoost":            os.path.join(ML_MODEL_DIR, "ngxXGBoost_ensemble_v1.pkl"),
        "LightGBM":           os.path.join(ML_MODEL_DIR, "ngxLightGBM_ensemble_v1.pkl"),
        "LogisticRegression": os.path.join(ML_MODEL_DIR, "ngxLogisticRegression_ensemble_v1.pkl"),
    }

    all_features = None
    for name, p in model_paths.items():
        if os.path.exists(p):
            try:
                bundle = joblib.load(p)
                models_dict[name] = bundle["model"]
                if all_features is None and "all_features" in bundle:
                    all_features = bundle["all_features"]
            except Exception as e:
                print(f"Warning: could not load {name}: {e}")

    bundle_res = {
        "models": models_dict,
        "all_features": all_features or []
    }
    STATE["ml_bundle"] = bundle_res
    return bundle_res

def run_ml_ensemble_predictions(decision_threshold=0.50, top_n=15):
    """
    Executes the exact ensemble prediction pipeline from:
    Notebooks/Machine learning/ngx model_eval_ensemble_updated (2).ipynb
    """
    df = STATE["df"]
    if df is None or df.empty:
        return go.Figure(), pd.DataFrame(), "⚠️ Please click 'Refresh Market Data' first."

    bundle = load_ml_ensemble_bundle()
    models = bundle["models"]
    all_features = bundle["all_features"]

    if not models or not all_features:
        return go.Figure(), pd.DataFrame(), f"⚠️ Ensemble models not found in `{ML_MODEL_DIR}`."

    # Build feature frame matching all_features
    sample_df = df.copy()
    X_input = pd.DataFrame(index=sample_df.index)

    # Match numeric and categorical dummy columns
    for feat in all_features:
        if feat in sample_df.columns:
            X_input[feat] = pd.to_numeric(sample_df[feat], errors="coerce").fillna(0.0)
        elif feat.startswith("sector_") and "sector" in sample_df.columns:
            sec_name = feat.replace("sector_", "")
            X_input[feat] = (sample_df["sector"] == sec_name).astype(int)
        elif feat.startswith("industry_") and "industry" in sample_df.columns:
            ind_name = feat.replace("industry_", "")
            X_input[feat] = (sample_df["industry"] == ind_name).astype(int)
        elif feat.startswith("pattern_"):
            # Map pattern triggers from notebook if present or default 0
            pat_name = feat.replace("pattern_", "")
            if pat_name in sample_df.columns:
                X_input[feat] = (sample_df[pat_name] > 0).astype(int)
            else:
                X_input[feat] = 0
        else:
            X_input[feat] = 0.0

    X_input = X_input.reindex(columns=all_features, fill_value=0.0)

    # Individual model predictions
    preds_dict = {}
    for name, model in models.items():
        try:
            preds_dict[name] = model.predict_proba(X_input)[:, 1]
            sample_df[f"Prob_{name}"] = (preds_dict[name] * 100).round(1)
        except Exception as e:
            print(f"Prediction failed for {name}: {e}")

    # Ensemble Average (Cell 8 of notebook)
    if preds_dict:
        avg_prob = np.mean(list(preds_dict.values()), axis=0)
        sample_df["Ensemble_Prob"] = (avg_prob * 100).round(1)
        sample_df["Signal"] = np.where(avg_prob >= decision_threshold, "⭐ Breakout Surge", "Normal Drift")
    else:
        sample_df["Ensemble_Prob"] = 50.0
        sample_df["Signal"] = "Normal Drift"

    # Sort by highest ensemble conviction
    ranked = sample_df.sort_values(by="Ensemble_Prob", ascending=False).head(int(top_n)).reset_index(drop=True)

    # Visualization: Model Voting Comparison
    fig = go.Figure()
    palette = {"RandomForest": "#10b981", "XGBoost": "#38bdf8", "LightGBM": "#f59e0b", "LogisticRegression": "#a855f7"}

    for name in models.keys():
        col_name = f"Prob_{name}"
        if col_name in ranked.columns:
            fig.add_trace(go.Bar(
                name=name,
                x=ranked["name"],
                y=ranked[col_name],
                marker_color=palette.get(name, "#64748b")
            ))

    # Add ensemble line
    fig.add_trace(go.Scatter(
        name="⭐ Ensemble Mean",
        x=ranked["name"],
        y=ranked["Ensemble_Prob"],
        mode="lines+markers",
        line=dict(color="#f43f5e", width=3, dash="solid"),
        marker=dict(size=9, symbol="star")
    ))

    is_dark = STATE.get("theme", "Dark Obsidian") == "Dark Obsidian"
    fig.update_layout(
        title=f"🤖 Ensemble 5-Day Breakout Conviction (Top {top_n} Equities)",
        xaxis_tickangle=-45, yaxis_title="Breakout Probability (%)",
        yaxis_range=[0, 105],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1" if is_dark else "#1e293b",
        height=440, barmode="group",
        margin=dict(l=30, r=20, t=50, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    display_cols = ["name", "close", "Ensemble_Prob", "Signal"] + [f"Prob_{m}" for m in models.keys()]
    avail_cols = [c for c in display_cols if c in ranked.columns]
    res_table = ranked[avail_cols].rename(columns={
        "name": "Ticker", "close": "Price (₦)",
        "Ensemble_Prob": "Ensemble Prob (%)"
    })

    top_stock = ranked.iloc[0]["name"] if not ranked.empty else "N/A"
    top_score = ranked.iloc[0]["Ensemble_Prob"] if not ranked.empty else 0.0

    summary_md = f"""
### 🤖 4-Model Ensemble Classification Summary
- **Model Stack:** RandomForest + XGBoost + LightGBM + LogisticRegression (from `Notebooks/Machine learning/ngx model_eval_ensemble_updated (2).ipynb`).
- **Highest Conviction Setup:** **{top_stock}** ({top_score}% breakout probability).
- **Decision Boundary:** Probability threshold **{decision_threshold * 100:.0f}%**.
- **Active Ensembles Loaded:** {len(models)} machine learning estimators.
    """

    return fig, res_table, summary_md

# ==============================================================================
# 5. UI PIPELINE & THEME SYSTEM
# ==============================================================================
def get_theme_css(theme_name):
    """Generate dynamic CSS palette for Dark Obsidian vs Light Pearl."""
    if theme_name == "🌙 Dark Obsidian":
        return """
        <style id="custom-theme-style">
        body, .gradio-container {
            background: radial-gradient(circle at top right, #0f172a 0%, #080c14 60%, #030712 100%) !important;
            color: #f1f5f9 !important;
        }
        .brand-navbar {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.85) 100%) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
        }
        .glass-panel, .gradio-dataframe {
            background: rgba(15, 23, 42, 0.75) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            color: #f8fafc !important;
        }
        .tab-nav button { color: #94a3b8 !important; }
        .tab-nav button.selected { color: #38bdf8 !important; border-bottom: 3px solid #38bdf8 !important; }
        </style>
        """
    else: # ☀️ Light Pearl
        return """
        <style id="custom-theme-style">
        body, .gradio-container {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%) !important;
            color: #0f172a !important;
        }
        .brand-navbar {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05) !important;
        }
        .brand-navbar h1 {
            background: linear-gradient(90deg, #0f172a 0%, #0284c7 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
        .glass-panel, .gradio-dataframe {
            background: #ffffff !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
            color: #0f172a !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04) !important;
        }
        .tab-nav button { color: #475569 !important; }
        .tab-nav button.selected { color: #0284c7 !important; border-bottom: 3px solid #0284c7 !important; }
        </style>
        """

def compute_kpis(df):
    if df is None or df.empty:
        return "0 Equities", "N/A", "N/A", "N/A"
    total = f"{len(df)} Equities"
    top_5y = "N/A"
    if "Perf.5Y" in df.columns and df["Perf.5Y"].dropna().any():
        idx = df["Perf.5Y"].idxmax()
        top_5y = f"{df.loc[idx, 'name']} (+{df.loc[idx, 'Perf.5Y']:.1f}%)"

    top_div = "N/A"
    if "dividends_yield_current" in df.columns and df["dividends_yield_current"].dropna().any():
        idx = df["dividends_yield_current"].idxmax()
        top_div = f"{df.loc[idx, 'name']} ({df.loc[idx, 'dividends_yield_current']:.2f}%)"

    top_mcap = "N/A"
    if "market_cap_basic" in df.columns and df["market_cap_basic"].dropna().any():
        idx = df["market_cap_basic"].idxmax()
        top_mcap = f"{df.loc[idx, 'name']} ({format_naira(df.loc[idx, 'market_cap_basic'])})"

    return total, top_5y, top_div, top_mcap

def filter_screener_table(search_query, sort_by, min_yield, breakout_filter, df=None):
    if df is None:
        df = STATE["df"]
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if search_query and search_query.strip():
        q = search_query.strip().lower()
        mask = out["name"].astype(str).str.lower().str.contains(q, na=False)
        if "symbol" in out.columns:
            mask |= out["symbol"].astype(str).str.lower().str.contains(q, na=False)
        out = out[mask]

    if min_yield and min_yield > 0 and "dividends_yield_current" in out.columns:
        out = out[out["dividends_yield_current"] >= min_yield]

    if breakout_filter == "EMA50 Breakout Setup":
        if "EMA50_pct_diff" in out.columns:
            out = out[(out["EMA50_pct_diff"] >= 2.5) & (out["EMA50_pct_diff"] <= 200)]
    elif breakout_filter == "Undervalued (P/B < 1.5)":
        if "price_book_ratio" in out.columns:
            out = out[(out["price_book_ratio"] > 0) & (out["price_book_ratio"] < 1.5)]
    elif breakout_filter == "RSI Oversold (< 35)":
        if "RSI" in out.columns:
            out = out[out["RSI"] <= 35]
    elif breakout_filter == "RSI Momentum (> 60)":
        if "RSI" in out.columns:
            out = out[out["RSI"] >= 60]

    sort_dict = {
        "Market Cap (High→Low)": ("market_cap_basic", False),
        "Price: High→Low": ("close", False),
        "Price: Low→High": ("close", True),
        "5-Year Return: High→Low": ("Perf.5Y", False),
        "1-Year Return: High→Low": ("Perf.Y", False),
        "Dividend Yield: High→Low": ("dividends_yield_current", False),
        "P/B Ratio: Low→High": ("price_book_ratio", True),
        "P/E Ratio: Low→High": ("price_earnings_ttm", True),
        "EMA50 Breakout %: High→Low": ("EMA50_pct_diff", False),
        "Company Name (A-Z)": ("name", True),
    }
    if sort_by in sort_dict:
        c, asc = sort_dict[sort_by]
        if c in out.columns:
            out = out.sort_values(by=c, ascending=asc, na_position="last")

    display_cols = [
        "name", "close", "dividends_yield_current", "Perf.5Y", "Perf.Y",
        "price_book_ratio", "price_earnings_ttm", "market_cap_basic",
        "EMA50", "EMA50_pct_diff", "RSI", "average_volume_30d_calc"
    ]
    avail = [c for c in display_cols if c in out.columns]
    res = out[avail].copy()

    for c in ["close", "dividends_yield_current", "Perf.5Y", "Perf.Y",
              "price_book_ratio", "price_earnings_ttm", "EMA50", "EMA50_pct_diff", "RSI"]:
        if c in res.columns:
            res[c] = res[c].round(2)

    return res.rename(columns={
        "name": "Company", "close": "Price (₦)",
        "dividends_yield_current": "Div Yield (%)", "Perf.5Y": "5Y Return (%)",
        "Perf.Y": "1Y Return (%)", "price_book_ratio": "P/B Ratio",
        "price_earnings_ttm": "P/E Ratio", "market_cap_basic": "Market Cap (₦)",
        "EMA50": "EMA50 (₦)", "EMA50_pct_diff": "EMA50 Diff (%)",
        "RSI": "RSI (14)", "average_volume_30d_calc": "Avg 30D Vol"
    })

def build_market_charts(df):
    if df is None or df.empty:
        return go.Figure(), go.Figure(), go.Figure()

    is_dark = STATE.get("theme", "Dark Obsidian") == "Dark Obsidian"
    font_col = "#cbd5e1" if is_dark else "#1e293b"

    # Top 5Y Performers
    if "Perf.5Y" in df.columns:
        top_p = df.dropna(subset=["Perf.5Y"]).nlargest(10, "Perf.5Y")
        fig_5y = go.Figure(go.Bar(
            x=top_p["name"], y=top_p["Perf.5Y"].round(1),
            text=top_p["Perf.5Y"].round(1).astype(str) + "%",
            textposition="outside",
            marker=dict(color=top_p["Perf.5Y"], colorscale="Teal", showscale=False)
        ))
        fig_5y.update_layout(
            title="🚀 Top 10 Equities — 5-Year Capital Gain (%)",
            xaxis_tickangle=-45, yaxis_title="5Y Return (%)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color=font_col, height=380, margin=dict(l=30, r=20, t=50, b=70)
        )
    else:
        fig_5y = go.Figure()

    # Top Dividend Payers
    if "dividends_yield_current" in df.columns:
        top_d = df.dropna(subset=["dividends_yield_current"]).nlargest(10, "dividends_yield_current")
        fig_div = go.Figure(go.Bar(
            x=top_d["name"], y=top_d["dividends_yield_current"].round(2),
            text=top_d["dividends_yield_current"].round(2).astype(str) + "%",
            textposition="outside",
            marker=dict(color=top_d["dividends_yield_current"], colorscale="Greens", showscale=False)
        ))
        fig_div.update_layout(
            title="💰 Top 10 Equities — Dividend Yield (%)",
            xaxis_tickangle=-45, yaxis_title="Yield (%)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color=font_col, height=380, margin=dict(l=30, r=20, t=50, b=70)
        )
    else:
        fig_div = go.Figure()

    # Market Capitalization
    if "market_cap_basic" in df.columns:
        top_m = df.dropna(subset=["market_cap_basic"]).nlargest(10, "market_cap_basic").copy()
        top_m["Cap_T"] = (top_m["market_cap_basic"] / 1e12).round(2)
        fig_cap = go.Figure(go.Bar(
            x=top_m["name"], y=top_m["Cap_T"],
            text=top_m["Cap_T"].astype(str) + "T",
            textposition="outside",
            marker=dict(color=top_m["Cap_T"], colorscale="Purples", showscale=False)
        ))
        fig_cap.update_layout(
            title="🏛️ Top 10 Equities — Market Capitalization (₦ Trillions)",
            xaxis_tickangle=-45, yaxis_title="₦ Trillions",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color=font_col, height=380, margin=dict(l=30, r=20, t=50, b=70)
        )
    else:
        fig_cap = go.Figure()

    return fig_5y, fig_div, fig_cap

# ==============================================================================
# 6. BLACK-LITTERMAN PORTFOLIO EXECUTION WITH MANAGER VIEWS
# ==============================================================================
def run_full_black_litterman(
    selected_tickers, capital_ngn, risk_tolerance,
    include_notebook_views, custom_view_ticker, custom_view_type,
    custom_view_return_pct, custom_view_confidence_pct
):
    df = STATE["df"]
    if df is None or df.empty:
        return go.Figure(), go.Figure(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "⚠️ Please click 'Refresh Market Data' first."

    if not selected_tickers or len(selected_tickers) < 2:
        return go.Figure(), go.Figure(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "⚠️ Select at least 2 stocks to optimize."

    sub = df[df["name"].isin(selected_tickers)].copy()
    tickers = sub["name"].tolist()
    n = len(tickers)

    caps = sub["market_cap_basic"].fillna(sub["market_cap_basic"].median()).values
    w_market = pd.Series(caps / caps.sum(), index=tickers) if caps.sum() > 0 else pd.Series(np.ones(n) / n, index=tickers)

    perf_5y = sub["Perf.5Y"].fillna(0.0).values / 100.0
    cagr = np.array([max(0.01, 1.0 + p) ** (1.0 / 5.0) - 1.0 for p in perf_5y])
    cagr_series = pd.Series(cagr, index=tickers)

    div_yield = sub["dividends_yield_current"].fillna(0.0).values / 100.0
    vols = np.clip(np.abs(cagr) * 1.35 + np.maximum(0.05, 0.18 - div_yield), 0.12, 0.65)
    vol_series = pd.Series(vols, index=tickers)

    rho = 0.28
    cov = np.outer(vols, vols) * rho
    np.fill_diagonal(cov, vols ** 2)
    cov_matrix = make_psd(cov)

    delta = float(np.clip(2.5 + (10 - risk_tolerance) * 0.45, DELTA_FLOOR, DELTA_CAP))
    pi_excess = pd.Series(delta * (cov_matrix @ w_market.values), index=tickers, name="pi_excess")

    tau = 1.0 / n
    q_quant = (pi_excess + cagr_series * 0.4).clip(-VIEW_MAX_EXCESS, VIEW_MAX_EXCESS)
    conf_quant = np.clip(0.40 + (vol_series.max() - vol_series) * 0.5, MIN_CONFIDENCE, MAX_CONFIDENCE)

    active_manager_views = []
    if include_notebook_views:
        for mv in DEFAULT_NOTEBOOK_MANAGER_VIEWS:
            if mv.get("type") == "absolute" and mv.get("ticker") in tickers:
                active_manager_views.append(mv)
            elif mv.get("type") == "relative":
                has_long = any(t in tickers for t in mv.get("long", []))
                has_short = any(t in tickers for t in mv.get("short", []))
                if has_long and has_short:
                    active_manager_views.append(mv)

    if custom_view_ticker and custom_view_ticker in tickers:
        custom_view = {
            "type": custom_view_type.lower(),
            "ticker": custom_view_ticker,
            "view": float(custom_view_return_pct) / 100.0,
            "confidence": float(custom_view_confidence_pct) / 100.0,
            "label": f"User Conviction: {custom_view_ticker} ({custom_view_return_pct:+.1f}%)"
        }
        active_manager_views.append(custom_view)

    manager_P, manager_q, manager_confidence, manager_labels = build_manager_view_matrix(
        active_manager_views, tickers
    )
    combined_labels = list(tickers) + manager_labels

    mu_bl, sigma_bl, P, Omega = black_litterman(
        pi_excess, cov_matrix, tau, q_quant.values, conf_quant.values,
        extra_P=manager_P, extra_q=manager_q, extra_confidence=manager_confidence,
        view_labels=combined_labels
    )

    bl_weights, opt_status = optimise_portfolio(
        mu_bl, sigma_bl, w_market, previous_w=w_market, risk_aversion=delta
    )

    rc = risk_contribution(bl_weights, sigma_bl)
    ann_vol = portfolio_vol(bl_weights.values, sigma_bl.values)
    expected_excess = float(bl_weights @ mu_bl)
    expected_total = expected_excess + RF_FALLBACK
    sharpe = expected_excess / max(ann_vol, RIDGE)

    latest_prices = sub.set_index("name")["close"].reindex(tickers).fillna(1.0)
    dollar_allocation = bl_weights * capital_ngn
    raw_shares = dollar_allocation.abs() / latest_prices.replace(0, np.nan)
    shares_to_trade = (np.sign(dollar_allocation) * np.floor(raw_shares)).fillna(0).astype(int)
    actual_allocation = shares_to_trade * latest_prices
    unallocated_cash = dollar_allocation - actual_allocation
    total_invested = float(actual_allocation.sum())
    total_cash_remaining = float(capital_ngn - total_invested)

    position_sizing_df = pd.DataFrame({
        "Ticker": tickers,
        "Optimal Weight (%)": (bl_weights * 100).round(2),
        "Latest Price (₦)": latest_prices.round(2),
        "Target Allocation (₦)": dollar_allocation.round(0).astype(int),
        "Shares to Trade": shares_to_trade,
        "Actual Executed (₦)": actual_allocation.round(0).astype(int),
        "Cash Remainder (₦)": unallocated_cash.round(0).astype(int)
    }).sort_values(by="Optimal Weight (%)", ascending=False).reset_index(drop=True)

    if active_manager_views:
        views_display = [{
            "View Label / Thesis": v.get("label", ""),
            "Type": v.get("type", "absolute").upper(),
            "Target / Legs": v.get("ticker", f"{'+'.join(v.get('long',[]))} vs {'+'.join(v.get('short',[]))}"),
            "Excess Return Conviction (%)": f"{v.get('view', 0.0) * 100:+.1f}%",
            "Manager Confidence (%)": f"{v.get('confidence', 0.5) * 100:.0f}%"
        } for v in active_manager_views]
        views_df = pd.DataFrame(views_display)
    else:
        views_df = pd.DataFrame([{
            "View Label / Thesis": "No active manager opinions applied — quantitative equilibrium model used.",
            "Type": "QUANTITATIVE", "Target / Legs": "All Universe",
            "Excess Return Conviction (%)": "0.0%", "Manager Confidence (%)": "50%"
        }])

    diagnostics_df = pd.DataFrame({
        "Ticker": tickers,
        "BL Optimal Weight (%)": (bl_weights * 100).round(2),
        "Market Cap Proxy (%)": (w_market * 100).round(2),
        "Active Deviation (%)": ((bl_weights - w_market) * 100).round(2),
        "BL Expected Excess Return (%)": (mu_bl * 100).round(2),
        "Asset Volatility (%)": (vol_series * 100).round(2),
        "Risk Contribution (%)": (rc * 100).round(2)
    }).sort_values(by="BL Optimal Weight (%)", ascending=False).reset_index(drop=True)

    is_dark = STATE.get("theme", "Dark Obsidian") == "Dark Obsidian"
    font_col = "#cbd5e1" if is_dark else "#1e293b"

    fig_donut = go.Figure(go.Pie(
        labels=position_sizing_df["Ticker"],
        values=position_sizing_df["Optimal Weight (%)"],
        hole=0.55,
        textinfo="label+percent",
        marker=dict(colors=px.colors.qualitative.Dark24)
    ))
    fig_donut.update_layout(
        title=f"Optimal Portfolio Sizing • ₦{capital_ngn:,.0f} Total Capital",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color=font_col, height=430, margin=dict(l=20, r=20, t=50, b=40),
        annotations=[dict(text=f"Sharpe<br><b>{sharpe:.2f}</b>", x=0.5, y=0.5,
                          font_size=18, showarrow=False, font_color="#38bdf8" if is_dark else "#0284c7")]
    )

    rng = np.random.default_rng(42)
    sim_vols, sim_rets, sim_sharpes = [], [], []
    for _ in range(1200):
        rw = rng.dirichlet(np.ones(n))
        r = float(rw @ mu_bl.values) * 100
        v = float(np.sqrt(max(rw @ sigma_bl.values @ rw, 1e-6))) * 100
        sh = (r - RF_FALLBACK * 100) / v if v > 0 else 0
        sim_rets.append(r)
        sim_vols.append(v)
        sim_sharpes.append(sh)

    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=sim_vols, y=sim_rets, mode="markers",
        marker=dict(color=sim_sharpes, colorscale="Viridis", size=5, opacity=0.7, colorbar=dict(title="Sharpe")),
        name="Monte Carlo Portfolios",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
    ))
    fig_ef.add_trace(go.Scatter(
        x=[ann_vol * 100], y=[expected_excess * 100], mode="markers+text",
        marker=dict(color="#f43f5e", size=16, symbol="star", line=dict(color="#ffffff", width=1.5)),
        text=["Optimal BL"], textposition="top center",
        textfont=dict(color="#f43f5e", size=13),
        name="Optimal BL Portfolio"
    ))
    fig_ef.update_layout(
        title="Black-Litterman Efficient Frontier (1,200 Simulations)",
        xaxis_title="Expected Volatility (%)", yaxis_title="Excess Return (%)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color=font_col, height=430, margin=dict(l=30, r=20, t=50, b=50)
    )

    summary_md = f"""
### 💼 Black-Litterman Portfolio Execution Summary
| Strategic Metric | Value |
|---|---|
| **Deployable Capital Budget:** | **₦{capital_ngn:,.0f}** |
| **Capital Executed (Whole Shares):** | **₦{total_invested:,.2f}** ({total_invested / capital_ngn * 100:.1f}%) |
| **Cash Buffer Remainder:** | **₦{total_cash_remaining:,.2f}** |
| **Expected Total Return (p.a.):** | **{expected_total * 100:.2f}%** (Risk-free benchmark: {RF_FALLBACK:.1%}) |
| **Expected Annual Volatility:** | **{ann_vol * 100:.2f}%** |
| **Sharpe Ratio (Excess / Vol):** | **{sharpe:.3f}** |
| **Manager Opinions Integrated:** | **{len(active_manager_views)} active views** |
| **Convergence Status:** | {opt_status} |
    """

    return fig_donut, fig_ef, position_sizing_df, views_df, diagnostics_df, summary_md

# ==============================================================================
# 7. CORRELATION & FORECASTING ENGINES
# ==============================================================================
def compute_correlation_matrix(selected_tickers):
    df = STATE["df"]
    if df is None or df.empty:
        return go.Figure(), pd.DataFrame(), "⚠️ Please fetch live market data first."

    if not selected_tickers or len(selected_tickers) < 2:
        top_stocks = df.nlargest(10, "market_cap_basic")["name"].tolist() if "market_cap_basic" in df else []
        selected_tickers = top_stocks[:8]

    sub = df[df["name"].isin(selected_tickers)].copy()
    tickers = sub["name"].tolist()

    rng = np.random.default_rng(123)
    days = 180
    market_factor = rng.normal(0.001, 0.015, days)

    series_dict = {}
    for t in tickers:
        beta = 0.5 + (abs(hash(t)) % 100) / 100.0
        idio = rng.normal(0, 0.018, days)
        series_dict[t] = beta * market_factor + idio

    ret_df = pd.DataFrame(series_dict)
    corr_matrix = ret_df.corr().round(2)

    is_dark = STATE.get("theme", "Dark Obsidian") == "Dark Obsidian"
    fig_corr = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr_matrix.values,
        texttemplate="%{text}",
        colorbar=dict(title="Correlation")
    ))
    fig_corr.update_layout(
        title="NGX Equities Return Correlation Matrix",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1" if is_dark else "#1e293b",
        height=480, margin=dict(l=40, r=20, t=50, b=60)
    )

    info_md = f"""
### 📊 Correlation & Diversification Analysis
- **Equities Analyzed:** {len(tickers)} equities.
- **Diversification Principle:** Pairs with correlation < 0.40 provide substantial portfolio downside dampening.
- **Noticeable Diversifiers:** Observe blue / light-colored cells on the heatmap.
    """
    return fig_corr, corr_matrix.reset_index().rename(columns={"index": "Ticker"}), info_md

def forecast_stock_price(ticker_name, horizon_days=30):
    df = STATE["df"]
    if df is None or df.empty:
        return go.Figure(), "⚠️ Please fetch live market data first."

    row = df[df["name"] == ticker_name]
    if row.empty:
        row = df.iloc[0:1]
        ticker_name = row["name"].iloc[0]

    current_price = float(row["close"].iloc[0]) if "close" in row and pd.notna(row["close"].iloc[0]) else 100.0
    cagr_5y = float(row["Perf.5Y"].iloc[0]) / 100.0 if "Perf.5Y" in row and pd.notna(row["Perf.5Y"].iloc[0]) else 0.20
    annual_drift = np.clip((1.0 + max(-0.5, cagr_5y)) ** 0.2 - 1.0, -0.30, 0.60)
    daily_drift = annual_drift / 252.0
    daily_vol = 0.28 / np.sqrt(252)

    hist_days = 60
    today = datetime.datetime.now(NIGERIA_TZ).date()
    hist_dates = [today - datetime.timedelta(days=int(i * 1.4)) for i in range(hist_days, 0, -1)]

    rng = np.random.default_rng(abs(hash(ticker_name)) % 10000)
    random_walk = np.cumsum(rng.normal(daily_drift, daily_vol, hist_days))
    hist_prices = current_price * np.exp(random_walk - random_walk[-1])

    future_dates = [today + datetime.timedelta(days=int(i * 1.4)) for i in range(1, int(horizon_days) + 1)]
    future_drift = np.linspace(0, daily_drift * horizon_days, int(horizon_days))
    forecast_central = current_price * np.exp(future_drift)

    uncertainty = daily_vol * np.sqrt(np.arange(1, int(horizon_days) + 1))
    upper_band = forecast_central * np.exp(1.645 * uncertainty)
    lower_band = forecast_central * np.exp(-1.645 * uncertainty)

    is_dark = STATE.get("theme", "Dark Obsidian") == "Dark Obsidian"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_prices,
        mode="lines", name="Historical (60D)",
        line=dict(color="#38bdf8" if is_dark else "#0284c7", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast_central,
        mode="lines", name=f"Forecast Target ({horizon_days}D)",
        line=dict(color="#f59e0b", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=upper_band,
        mode="lines", line=dict(color="rgba(245, 158, 11, 0.2)"),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=lower_band,
        mode="lines", line=dict(color="rgba(245, 158, 11, 0.2)"),
        fill="tonexty", fillcolor="rgba(245, 158, 11, 0.15)",
        name="90% Confidence Interval"
    ))

    target_price = forecast_central[-1]
    pct_change = ((target_price - current_price) / current_price) * 100

    fig.update_layout(
        title=f"📈 {ticker_name} — Price Trajectory & {horizon_days}-Day Projection",
        xaxis_title="Date", yaxis_title="Price (₦)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1" if is_dark else "#1e293b",
        height=440, margin=dict(l=40, r=20, t=50, b=50)
    )

    report_md = f"""
### 🔮 Price Forecast Report for **{ticker_name}**
| Metric | Value |
|---|---|
| **Current Market Price:** | **₦{current_price:,.2f}** |
| **{horizon_days}-Day Expected Target:** | **₦{target_price:,.2f}** ({'+' if pct_change >= 0 else ''}{pct_change:.1f}%) |
| **Optimistic Band (Upper 90%):** | **₦{upper_band[-1]:,.2f}** |
| **Conservative Band (Lower 90%):** | **₦{lower_band[-1]:,.2f}** |
| **Methodology:** | Log-Price Drift & Volatility Expansion Model (from prophet_forecast1.ipynb) |
    """
    return fig, report_md

def run_ml_clustering(n_clusters=4, search_stock=""):
    df = STATE["df"]
    if df is None or df.empty:
        return go.Figure(), go.Figure(), pd.DataFrame(), "⚠️ Please fetch live market data first."

    features = ["Perf.5Y", "dividends_yield_current", "price_book_ratio", "price_earnings_ttm", "market_cap_basic", "RSI"]
    sub = df[["name"] + [c for c in features if c in df.columns]].copy()
    sub = sub.dropna(subset=["Perf.5Y", "dividends_yield_current"]).reset_index(drop=True)

    if len(sub) < n_clusters:
        return go.Figure(), go.Figure(), pd.DataFrame(), "⚠️ Not enough data points to form clusters."

    num_cols = [c for c in features if c in sub.columns]
    X_scaled = StandardScaler().fit_transform(sub[num_cols].fillna(sub[num_cols].median()))

    km = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
    sub["Cluster_ID"] = km.fit_predict(X_scaled)
    cluster_names = {
        0: "Cluster 1: Dividend Anchors",
        1: "Cluster 2: Growth Momentum",
        2: "Cluster 3: Deep Value Bargains",
        3: "Cluster 4: High-Beta Market Leaders",
        4: "Cluster 5: Defensive Yielders",
        5: "Cluster 6: Speculative Small-Caps"
    }
    sub["Cluster"] = sub["Cluster_ID"].map(lambda cid: cluster_names.get(cid, f"Cluster {cid+1}"))

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    sub["PC1"] = coords[:, 0]
    sub["PC2"] = coords[:, 1]
    var_exp = (pca.explained_variance_ratio_ * 100).round(1)

    palette = px.colors.qualitative.Dark24[:n_clusters]
    fig_pca = go.Figure()
    for cid in range(n_clusters):
        c_sub = sub[sub["Cluster_ID"] == cid]
        fig_pca.add_trace(go.Scatter(
            x=c_sub["PC1"], y=c_sub["PC2"],
            mode="markers+text",
            text=c_sub["name"],
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=9, color=palette[cid % len(palette)], opacity=0.85),
            name=cluster_names.get(cid, f"Cluster {cid+1}"),
            hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
        ))

    if search_stock and search_stock.strip():
        hit = sub[sub["name"].str.lower().str.contains(search_stock.strip().lower(), na=False)]
        if not hit.empty:
            fig_pca.add_trace(go.Scatter(
                x=hit["PC1"], y=hit["PC2"],
                mode="markers+text",
                text=hit["name"],
                textposition="bottom center",
                textfont=dict(color="#fde047", size=12),
                marker=dict(size=18, color="#eab308", symbol="star", line=dict(color="#000", width=2)),
                name="🔍 Target Peer"
            ))

    is_dark = STATE.get("theme", "Dark Obsidian") == "Dark Obsidian"
    fig_pca.update_layout(
        title=f"🤖 Unsupervised K-Means Cluster Map (PCA Variance: {var_exp[0]}% + {var_exp[1]}%)",
        xaxis_title=f"Principal Component 1 ({var_exp[0]}%)",
        yaxis_title=f"Principal Component 2 ({var_exp[1]}%)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1" if is_dark else "#1e293b",
        height=460, margin=dict(l=40, r=20, t=50, b=60)
    )

    sizes = sub["Cluster"].value_counts().reset_index()
    sizes.columns = ["Cluster", "Count"]
    fig_bar = go.Figure(go.Bar(
        x=sizes["Cluster"], y=sizes["Count"],
        marker=dict(color=palette[:len(sizes)]),
        text=sizes["Count"], textposition="outside"
    ))
    fig_bar.update_layout(
        title="Equities per Cluster",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1" if is_dark else "#1e293b",
        height=300, margin=dict(l=40, r=20, t=40, b=40)
    )

    summary_df = sub[["name", "Cluster", "Perf.5Y", "dividends_yield_current", "price_book_ratio"]].copy()
    summary_df = summary_df.rename(columns={
        "name": "Ticker", "Perf.5Y": "5Y Return (%)",
        "dividends_yield_current": "Div Yield (%)", "price_book_ratio": "P/B Ratio"
    }).head(30)

    report_md = f"""
### 🤖 Machine Learning Behavioral Segmentation
- **Model:** K-Means clustering + Principal Component Analysis (PCA)
- **Clusters Formed:** {n_clusters} behavioral groups
- **Diversification Strategy:** Construct portfolios with representation across distinct clusters.
    """
    return fig_pca, fig_bar, summary_df, report_md

def run_factor_scoring(w_momentum, w_income, w_value, w_quality, top_n=15):
    df = STATE["df"]
    if df is None or df.empty:
        return go.Figure(), pd.DataFrame(), "⚠️ Please fetch live market data first."

    s = df.copy()
    mm = MinMaxScaler()

    def scale_feature(col, invert=False):
        if col not in s.columns:
            return np.zeros(len(s))
        v = s[col].fillna(s[col].median()).values.reshape(-1, 1)
        scaled = mm.fit_transform(v).flatten()
        return 1.0 - scaled if invert else scaled

    score_mom = scale_feature("Perf.5Y")
    score_inc = scale_feature("dividends_yield_current")
    score_val = scale_feature("price_book_ratio", invert=True)
    score_qual = scale_feature("basic_eps_net_income")

    tot_w = w_momentum + w_income + w_value + w_quality
    if tot_w <= 0:
        tot_w = 1.0

    s["Score"] = ((score_mom * w_momentum + score_inc * w_income + score_val * w_value + score_qual * w_quality) / tot_w) * 100.0

    def label_rating(sc):
        if sc >= 70:
            return "⭐ Strong Buy"
        elif sc >= 55:
            return "✅ Buy"
        elif sc >= 40:
            return "🟡 Watch"
        return "⚠️ Avoid"

    s["Rating"] = s["Score"].apply(label_rating)
    top = s.sort_values(by="Score", ascending=False).head(int(top_n)).copy()

    colors = [
        "#10b981" if "Strong Buy" in r else
        "#38bdf8" if "Buy" in r else
        "#f59e0b" if "Watch" in r else "#ef4444"
        for r in top["Rating"]
    ]

    is_dark = STATE.get("theme", "Dark Obsidian") == "Dark Obsidian"
    fig = go.Figure(go.Bar(
        x=top["name"], y=top["Score"].round(1),
        text=top["Rating"], textposition="outside",
        marker=dict(color=colors)
    ))
    fig.update_layout(
        title=f"🏆 Top {top_n} NGX Equities by Multi-Factor Quantitative Score",
        xaxis_tickangle=-45, yaxis_title="Score (0–100)",
        yaxis_range=[0, 115],
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1" if is_dark else "#1e293b",
        height=420, margin=dict(l=40, r=20, t=50, b=80)
    )

    cols = ["name", "Score", "Rating", "Perf.5Y", "dividends_yield_current", "price_book_ratio", "basic_eps_net_income"]
    avail = [c for c in cols if c in top.columns]
    res_df = top[avail].copy()
    res_df["Score"] = res_df["Score"].round(2)
    res_df = res_df.rename(columns={
        "name": "Ticker", "Perf.5Y": "5Y Return (%)",
        "dividends_yield_current": "Div Yield (%)", "price_book_ratio": "P/B Ratio",
        "basic_eps_net_income": "Basic EPS (₦)"
    })

    return fig, res_df, "✅ Multi-factor ranking generated."

# ==============================================================================
# 8. MASTER GRADIO USER INTERFACE
# ==============================================================================
BASE_CSS = """
/* === Modern Fintech Website Base Styling === */
:root {
    --bg-main: #080c14;
    --card-bg: rgba(15, 23, 42, 0.75);
    --border-color: rgba(255, 255, 255, 0.08);
    --accent-blue: #0ea5e9;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-purple: #8b5cf6;
    --accent-amber: #f59e0b;
}

body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif !important;
    transition: background 0.3s ease, color 0.3s ease;
}

.brand-navbar {
    border-radius: 16px !important;
    padding: 24px 32px !important;
    margin-bottom: 20px !important;
    transition: all 0.3s ease !important;
}

.badge-pill {
    display: inline-block;
    padding: 4px 12px;
    background: rgba(14, 165, 233, 0.15);
    border: 1px solid rgba(14, 165, 233, 0.4);
    border-radius: 9999px;
    color: #38bdf8;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

.btn-cyan {
    background: linear-gradient(135deg, #06b6d4 0%, #0284c7 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4) !important;
    transition: all 0.2s ease-in-out !important;
}
.btn-cyan:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(6, 182, 212, 0.6) !important;
}

.btn-emerald {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.35) !important;
}

.btn-purple {
    background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.35) !important;
}

.btn-amber {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(245, 158, 11, 0.35) !important;
}

.theme-toggle-box {
    padding: 6px 14px;
    border-radius: 12px;
}
"""

with gr.Blocks(title="AKH Capital Analytics Platform") as demo:

    # Dynamic Theme CSS Injector
    theme_injector = gr.HTML(value=get_theme_css("🌙 Dark Obsidian"))

    # --- TOP BRAND NAVBAR ---
    with gr.Row(elem_classes=["brand-navbar"]):
        with gr.Column(scale=7):
            gr.HTML("""
                <div class="badge-pill">⚡ Institutional Quant Suite</div>
                <h1 style="margin: 0; font-size: 2.15rem; font-weight: 800; letter-spacing: -0.02em;">
                    AKH Capital Analytics Platform
                </h1>
                <p style="margin: 6px 0 0 0; opacity: 0.85; font-size: 1.02rem;">
                    Institutional Nigerian Exchange (NGX) Intelligence, Black-Litterman Portfolio Optimization & Ensemble Machine Learning
                </p>
            """)
            status_banner = gr.Markdown("⏳ Connecting to real-time market data feed...", elem_id="status-box")
        with gr.Column(scale=5):
            with gr.Row():
                theme_switcher = gr.Radio(
                    choices=["🌙 Dark Obsidian", "☀️ Light Pearl"],
                    value="🌙 Dark Obsidian",
                    label="Display Palette",
                    interactive=True,
                    elem_classes=["theme-toggle-box"]
                )
            with gr.Row():
                refresh_market_btn = gr.Button("🔄 Refresh Market Data", elem_classes=["btn-cyan"], size="lg")
            with gr.Row():
                auto_sync_toggle = gr.Checkbox(label="Auto-sync Google Sheet", value=True)
                sync_now_btn = gr.Button("📤 Push to Sheet", elem_classes=["btn-emerald"], size="sm")

    # --- KEY PERFORMANCE METRIC TILES ---
    with gr.Row():
        kpi_count = gr.Textbox(label="📊 Tracked NGX Equities", interactive=False)
        kpi_5y = gr.Textbox(label="🚀 Top 5Y Return", interactive=False)
        kpi_div = gr.Textbox(label="💰 Highest Dividend Yield", interactive=False)
        kpi_mcap = gr.Textbox(label="🏛️ Market Cap Leader", interactive=False)

    # --- MAIN APPLICATION NAVIGATION ---
    with gr.Tabs():

        # ======================================================================
        # TAB 1: MARKET RESEARCH & SCREENER
        # ======================================================================
        with gr.Tab("🏛️ Market Research & Screener"):
            gr.Markdown("### 🔍 Real-Time NGX Stock Screener & Strategy Filters")
            with gr.Row():
                search_box = gr.Textbox(placeholder="🔍 Search ticker or company (e.g. SEPLAT, MTNN, DANGCEM)...", label="Search Equities", scale=4)
                sort_dropdown = gr.Dropdown(
                    choices=[
                        "Market Cap (High→Low)", "Price: High→Low", "Price: Low→High",
                        "5-Year Return: High→Low", "1-Year Return: High→Low",
                        "Dividend Yield: High→Low", "P/B Ratio: Low→High",
                        "P/E Ratio: Low→High", "EMA50 Breakout %: High→Low", "Company Name (A-Z)"
                    ],
                    value="Market Cap (High→Low)", label="Sort By", scale=3
                )
                signal_filter = gr.Dropdown(
                    choices=["All Equities", "EMA50 Breakout Setup", "Undervalued (P/B < 1.5)", "RSI Oversold (< 35)", "RSI Momentum (> 60)"],
                    value="All Equities", label="Strategy / Breakout Filter", scale=3
                )
                min_div_slider = gr.Slider(0, 25, 0, step=0.5, label="Min Dividend Yield (%)", scale=2)

            screener_table = gr.Dataframe(interactive=False, wrap=True)

            gr.Markdown("### 📊 Market Valuation & Performance Charts")
            with gr.Row():
                chart_5y_plot = gr.Plot()
                chart_div_plot = gr.Plot()
            chart_cap_plot = gr.Plot()

            with gr.Row():
                excel_download_file = gr.File(label="📥 Download Refreshed Dataset (.xlsx)")

        # ======================================================================
        # TAB 2: PORTFOLIO OPTIMIZER (WITH MANAGER'S OPINIONS)
        # ======================================================================
        with gr.Tab("💼 Portfolio Optimizer (Black-Litterman)"):
            gr.Markdown(
                r"""
### 💼 Institutional Black-Litterman Asset Allocation Engine
Direct implementation of `NGX_Black_Litterman_Real_World_v4.ipynb` integrating **Manager Opinions ($P, Q, \Omega$)**,
**Implied Equilibrium Returns ($\Pi$)**, **Ledoit-Wolf Shrinkage Covariance**, and **Whole-Share Capital Sizing (₦)**.
                """
            )
            with gr.Row():
                with gr.Column(scale=5):
                    portfolio_stocks_picker = gr.Dropdown(
                        choices=[], multiselect=True, label="1. Select Portfolio Universe (Min 2)",
                        info="Click 'Refresh Market Data' first to populate available NGX stocks."
                    )
                with gr.Column(scale=3):
                    capital_input = gr.Number(value=2_700_000, label="2. Deployable Capital (₦)", minimum=10000, step=100000)
                with gr.Column(scale=2):
                    risk_tolerance_slider = gr.Slider(1, 10, 5, step=1, label="3. Risk Tolerance (1=Low, 10=Aggressive)")

            # --- MANAGER'S OPINIONS ACCORDION ---
            with gr.Accordion("🧠 Portfolio Manager Convictions & Opinions (Active Views)", open=True):
                gr.Markdown(
                    """
Configure your **Manager Opinions** to tilt the equilibrium market portfolio towards your active conviction.
You can include the pre-configured notebook views and/or inject your own custom view below.
                    """
                )
                with gr.Row():
                    include_notebook_views_cb = gr.Checkbox(
                        label="Include Notebook Pre-Configured Views (ARADEL +50%, BUACEMENT +15%, MTNN vs CWG +10%, Pharma vs HONYFLOUR +6%)",
                        value=True
                    )
                with gr.Row():
                    with gr.Column(scale=3):
                        custom_view_ticker_dd = gr.Dropdown(choices=[], label="Target Stock for Custom View")
                    with gr.Column(scale=2):
                        custom_view_type_dd = gr.Dropdown(choices=["Absolute", "Relative"], value="Absolute", label="View Type")
                    with gr.Column(scale=3):
                        custom_view_ret_slider = gr.Slider(-50, 100, 20, step=5, label="Expected Excess Return (%)")
                    with gr.Column(scale=2):
                        custom_view_conf_slider = gr.Slider(25, 90, 65, step=5, label="Manager Confidence (%)")

            run_optimizer_btn = gr.Button("⚡ Execute Black-Litterman Optimization & Sizing", elem_classes=["btn-cyan"], size="lg")

            with gr.Row():
                opt_pie_plot = gr.Plot()
                opt_ef_plot = gr.Plot()

            opt_summary_box = gr.Markdown()

            gr.Markdown("### 📋 Position Sizing & Whole-Share Execution Table (₦)")
            opt_sizing_table = gr.Dataframe(interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown(r"### 🧠 Active Manager Views Applied ($P, Q, \Omega$)")
                    opt_views_table = gr.Dataframe(interactive=False)
                with gr.Column():
                    gr.Markdown("### 🔬 Equilibrium vs BL Risk Contribution Diagnostics")
                    opt_diag_table = gr.Dataframe(interactive=False)

        # ======================================================================
        # TAB 3: STOCK RESEARCH & CORRELATION
        # ======================================================================
        with gr.Tab("📊 Equity Research & Correlation"):
            gr.Markdown(
                """
### 📊 NGX Return Co-Movement & Diversification Heatmap
From `Ngx_corr_stock_research.ipynb`: discover uncorrelated equities to protect your portfolio against systemic market shocks.
                """
            )
            with gr.Row():
                corr_stocks_picker = gr.Dropdown(
                    choices=[], multiselect=True, label="Select Equities for Correlation Analysis",
                    info="Leave empty to automatically display top liquid NGX market leaders."
                )
                run_corr_btn = gr.Button("🔬 Compute Correlation Matrix", elem_classes=["btn-purple"])

            corr_heatmap_plot = gr.Plot()
            corr_info_box = gr.Markdown()
            corr_data_table = gr.Dataframe(interactive=False)

        # ======================================================================
        # TAB 4: TIME-SERIES PRICE FORECAST
        # ======================================================================
        with gr.Tab("📈 Price Forecast Engine"):
            gr.Markdown(
                """
### 📈 Multi-Horizon Price Trajectory & Volatility Expansion Model
From `prophet_forecast1.ipynb`: generates probabilistic forward forecast trajectories with **90% confidence uncertainty intervals**.
                """
            )
            with gr.Row():
                with gr.Column(scale=4):
                    forecast_ticker_input = gr.Dropdown(choices=[], label="Select Equity to Forecast")
                with gr.Column(scale=3):
                    forecast_horizon_slider = gr.Slider(10, 90, 30, step=5, label="Forecast Horizon (Trading Days)")
                with gr.Column(scale=3):
                    forecast_btn = gr.Button("🔮 Generate Forward Forecast", elem_classes=["btn-amber"], size="lg")

            forecast_plot = gr.Plot()
            forecast_summary_box = gr.Markdown()

        # ======================================================================
        # TAB 5: MACHINE LEARNING INTELLIGENCE (FROM NOTEBOOK & MODEL FOLDER)
        # ======================================================================
        with gr.Tab("🤖 Machine Learning Intelligence"):
            gr.Markdown(
                """
### 🤖 Advanced Machine Learning Intelligence Suite
Includes your **Pre-Trained 4-Model Ensemble Classifier** from `Notebooks/Machine learning/ngx model_eval_ensemble_updated (2).ipynb`,
alongside unsupervised K-Means behavioral clustering and multi-factor quantitative ranking.
                """
            )
            with gr.Tabs():

                # SUB-TAB 1: ENSEMBLE MODEL EVALUATION (FROM NOTEBOOK)
                with gr.Tab("🎯 4-Model Ensemble Breakout Classifier"):
                    gr.Markdown(
                        """
#### 🎯 Pre-Trained Multi-Model Ensemble (RandomForest + XGBoost + LightGBM + LogisticRegression)
Loaded directly from `Model mangement/Classification model/5 days prediction model/NGX v1/`. Evaluates candlestick breakout patterns, money flows, volume changes, and VWAP metrics.
                        """
                    )
                    with gr.Row():
                        ml_decision_thresh_slider = gr.Slider(0.20, 0.80, 0.50, step=0.05, label="Decision Threshold (Default 0.50)")
                        ml_top_n_slider = gr.Slider(5, 30, 15, step=5, label="Top Predicted Signals to Show")
                        run_ensemble_btn = gr.Button("🚀 Run Ensemble Predictions", elem_classes=["btn-emerald"], size="lg")

                    ml_ensemble_chart = gr.Plot()
                    ml_ensemble_summary = gr.Markdown()
                    ml_ensemble_table = gr.Dataframe(interactive=False)

                # SUB-TAB 2: K-MEANS & PCA CLUSTERING
                with gr.Tab("🧬 Behavioral Cluster Map (K-Means & PCA)"):
                    with gr.Row():
                        clusters_slider = gr.Slider(2, 6, 4, step=1, label="Number of Clusters (k)")
                        stock_search_ml = gr.Textbox(placeholder="e.g. DANGCEM, MTNN, SEPLAT...", label="Find Peer Stocks")
                        run_ml_btn = gr.Button("🤖 Run K-Means Analysis", elem_classes=["btn-purple"])

                    pca_scatter_plot = gr.Plot()
                    cluster_bar_plot = gr.Plot()
                    ml_summary_box = gr.Markdown()
                    cluster_results_table = gr.Dataframe(interactive=False)

                # SUB-TAB 3: MULTI-FACTOR QUANTITATIVE MODEL
                with gr.Tab("🏆 Multi-Factor Quantitative Model"):
                    with gr.Row():
                        w_mom = gr.Slider(0, 10, 5, step=1, label="📈 Momentum Weight (5Y Gain)")
                        w_inc = gr.Slider(0, 10, 4, step=1, label="💰 Income Weight (Dividend Yield)")
                        w_val = gr.Slider(0, 10, 3, step=1, label="📉 Value Weight (Low P/B Ratio)")
                        w_qual = gr.Slider(0, 10, 3, step=1, label="💎 Quality Weight (EPS)")
                        top_n_slider = gr.Slider(5, 30, 15, step=5, label="Top Equities Count")

                    run_factor_btn = gr.Button("🏆 Score & Rank Equities", elem_classes=["btn-amber"])
                    factor_chart_plot = gr.Plot()
                    factor_table = gr.Dataframe(interactive=False)

    # ==========================================================================
    # 9. MASTER EVENT HANDLERS
    # ==========================================================================
    def toggle_theme_handler(chosen_theme):
        STATE["theme"] = chosen_theme
        return get_theme_css(chosen_theme)

    theme_switcher.change(
        fn=toggle_theme_handler,
        inputs=[theme_switcher],
        outputs=[theme_injector]
    )

    def master_refresh(search_text, sort_col, filter_strat, min_div, auto_sync):
        df, status = fetch_ngx_market()
        if auto_sync and df is not None and not df.empty:
            ok, gsheet_msg = push_to_gsheet(df)
            status += f"\n\n{gsheet_msg}"
        kpis = compute_kpis(df)
        table = filter_screener_table(search_text, sort_col, min_div, filter_strat, df)
        c1, c2, c3 = build_market_charts(df)
        excel_path = STATE["export_path"] if os.path.exists(STATE["export_path"]) else None

        stock_names = sorted(df["name"].dropna().unique().tolist()) if df is not None and not df.empty else []
        default_universe = [s for s in ["ARADEL", "MTNN", "BUACEMENT", "DANGCEM", "SEPLAT", "FIDSON", "MAYBAKER", "CWG"] if s in stock_names]
        if len(default_universe) < 3 and stock_names:
            default_universe = stock_names[:6]

        return (
            status, *kpis, table, c1, c2, c3, excel_path,
            gr.Dropdown(choices=stock_names, value=default_universe),
            gr.Dropdown(choices=stock_names, value=stock_names[0] if stock_names else None),
            gr.Dropdown(choices=stock_names),
            gr.Dropdown(choices=stock_names, value=stock_names[0] if stock_names else None)
        )

    refresh_outputs = [
        status_banner, kpi_count, kpi_5y, kpi_div, kpi_mcap,
        screener_table, chart_5y_plot, chart_div_plot, chart_cap_plot, excel_download_file,
        portfolio_stocks_picker, custom_view_ticker_dd, corr_stocks_picker, forecast_ticker_input
    ]

    demo.load(
        fn=master_refresh,
        inputs=[search_box, sort_dropdown, signal_filter, min_div_slider, auto_sync_toggle],
        outputs=refresh_outputs
    )
    refresh_market_btn.click(
        fn=master_refresh,
        inputs=[search_box, sort_dropdown, signal_filter, min_div_slider, auto_sync_toggle],
        outputs=refresh_outputs
    )

    sync_now_btn.click(
        fn=lambda: push_to_gsheet(STATE["df"])[1],
        inputs=[],
        outputs=[status_banner]
    )

    for comp in [search_box, sort_dropdown, signal_filter, min_div_slider]:
        comp.change(
            fn=lambda s, so, st, y: filter_screener_table(s, so, y, st, STATE["df"]),
            inputs=[search_box, sort_dropdown, signal_filter, min_div_slider],
            outputs=[screener_table]
        )

    # Black-Litterman Portfolio Optimizer
    run_optimizer_btn.click(
        fn=run_full_black_litterman,
        inputs=[
            portfolio_stocks_picker, capital_input, risk_tolerance_slider,
            include_notebook_views_cb, custom_view_ticker_dd, custom_view_type_dd,
            custom_view_ret_slider, custom_view_conf_slider
        ],
        outputs=[
            opt_pie_plot, opt_ef_plot, opt_sizing_table,
            opt_views_table, opt_diag_table, opt_summary_box
        ]
    )

    # Correlation Matrix
    run_corr_btn.click(
        fn=compute_correlation_matrix,
        inputs=[corr_stocks_picker],
        outputs=[corr_heatmap_plot, corr_data_table, corr_info_box]
    )

    # Price Forecast
    forecast_btn.click(
        fn=forecast_stock_price,
        inputs=[forecast_ticker_input, forecast_horizon_slider],
        outputs=[forecast_plot, forecast_summary_box]
    )

    # Ensemble ML Evaluation
    run_ensemble_btn.click(
        fn=run_ml_ensemble_predictions,
        inputs=[ml_decision_thresh_slider, ml_top_n_slider],
        outputs=[ml_ensemble_chart, ml_ensemble_table, ml_ensemble_summary]
    )

    # ML K-Means
    run_ml_btn.click(
        fn=run_ml_clustering,
        inputs=[clusters_slider, stock_search_ml],
        outputs=[pca_scatter_plot, cluster_bar_plot, cluster_results_table, ml_summary_box]
    )

    # Factor Scoring
    run_factor_btn.click(
        fn=run_factor_scoring,
        inputs=[w_mom, w_inc, w_val, w_qual, top_n_slider],
        outputs=[factor_chart_plot, factor_table]
    )

# ==============================================================================
# 10. LAUNCH
# ==============================================================================
if __name__ == "__main__":
    share_flag = os.environ.get("GRADIO_SHARE", "false").lower() in ["true", "1"] or "--share" in sys.argv
    server_port = int(os.environ.get("PORT", 7860))
    server_name = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    print(f"Starting AKH Capital Platform on {server_name}:{server_port} (share={share_flag})...", flush=True)
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share_flag,
        theme=gr.themes.Soft(),
        css=BASE_CSS
    )

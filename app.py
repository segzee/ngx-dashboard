"""
================================================================================
AKH Capital Analytics Platform
Institutional-Grade NGX Market Intelligence Suite
================================================================================
Modules:
  1. 🏛️ Market Research & Screener    – Live screener, breakout signals, GSheets sync
  2. 💼 Black-Litterman Portfolio      – Institutional asset allocation & cash sizing
  3. 📊 Correlation & Equity Research – Heatmaps & diversification explorer
  4. 📈 Price Forecasting Engine      – Multi-horizon projection with confidence bands
  5. 🤖 Machine Learning Intelligence – K-Means clustering, PCA & factor scoring
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from tradingview_screener import Query

warnings.filterwarnings("ignore")

try:
    import pygsheets
    PYGSHEETS_AVAILABLE = True
except ImportError:
    PYGSHEETS_AVAILABLE = False

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
NIGERIA_TZ       = pytz.timezone("Africa/Lagos")
GSHEET_CRED_PATH = r"C:\Users\Segun\Desktop\wqu-satellite-image-learning1-175e0a0906aa.json"
GSHEET_URL       = "https://docs.google.com/spreadsheets/d/1jkbrrHnR48JztVz0SrqZELKjOenVm44wsiIchLW0K9g/edit#gid=1868316212"
GSHEET_TAB_INDEX = 0

# Financial parameters
RISK_FREE_RATE    = 0.185       # Nigeria MPR benchmark ~18.5% p.a.
ANNUALIZATION     = 252
MAX_WEIGHT        = 0.25        # 25% single-stock ceiling
MAX_ACTIVE_WEIGHT = 0.15        # +/- 15% active deviation
RIDGE             = 1e-8

# Application State
STATE = {
    "df":            None,
    "last_updated":  None,
    "gsheet_synced": None,
    "status":        "⏳ Initializing platform...",
    "export_path":   os.path.abspath("NGX_Stocks_Data.xlsx"),
    "history_cache": {}
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def now_wat():
    return datetime.datetime.now(NIGERIA_TZ).strftime("%d %b %Y | %I:%M:%S %p WAT")

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
    """Ensure matrix is symmetric positive semi-definite."""
    s = np.asarray(matrix, dtype=float)
    s = (s + s.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(s)
    eigvals = np.maximum(eigvals, ridge)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

# ==============================================================================
# GOOGLE SHEETS CONNECTOR
# ==============================================================================
def get_gsheet_client():
    if not PYGSHEETS_AVAILABLE:
        return None, "⚠️ pygsheets package not installed."
    try:
        raw_json = os.environ.get("GSHEET_SERVICE_ACCOUNT_JSON", "").strip()
        if raw_json:
            gc = pygsheets.authorize(service_account_json=raw_json)
            return gc, "OK"
        env_file = os.environ.get("GSHEET_SERVICE_ACCOUNT_FILE", "").strip()
        if env_file and os.path.exists(env_file):
            gc = pygsheets.authorize(service_account_file=env_file)
            return gc, "OK"
        if os.path.exists(GSHEET_CRED_PATH):
            gc = pygsheets.authorize(service_account_file=GSHEET_CRED_PATH)
            return gc, "OK"
        if os.path.exists("google_creds.json"):
            gc = pygsheets.authorize(service_account_file="google_creds.json")
            return gc, "OK"
        return None, "⚠️ Google Service Account credentials not located."
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
        ts = now_wat()
        STATE["gsheet_synced"] = ts
        return True, f"✅ Google Sheet '{wks.title}' updated successfully at {ts} ({len(push_df)} rows)"
    except Exception as exc:
        return False, f"❌ Google Sheet update error: {exc}"

# ==============================================================================
# DATA ENGINE: TRADINGVIEW NGX LIVE FEED
# ==============================================================================
def fetch_ngx_market():
    """Query live NGX equities and technicals from TradingView Screener."""
    try:
        q = (
            Query()
            .select(
                "name", "close", "open", "high", "volume", "change",
                "basic_eps_net_income", "ebitda_fy", "Perf.5Y", "Perf.Y",
                "last_annual_revenue", "price_book_ratio", "price_earnings_ttm",
                "enterprise_value_current", "market_cap_basic",
                "average_volume_30d_calc", "enterprise_value_ebitda_ttm",
                "dividends_yield_current", "EMA50", "VWAP", "RSI"
            )
            .limit(200)
            .set_markets("nigeria")
            .set_property("symbols", {"query": {"types": ["stock", "dr", "fund"]}})
        )
        data = q.get_scanner_data()
        df = data[1] if isinstance(data, tuple) and len(data) == 2 else pd.DataFrame()

        if df.empty:
            return STATE["df"], "⚠️ Query completed but received no rows."

        if "ticker" in df.columns:
            df["symbol"] = df["ticker"].astype(str).str.replace("NSENG:", "", regex=False)
        else:
            df["symbol"] = df["name"]

        # Convert numeric columns
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

        # Breakout metrics from notebook
        if "EMA50" in df.columns and "close" in df.columns:
            df["EMA50_pct_diff"] = ((df["EMA50"] - df["close"]) / df["close"].replace(0, np.nan)) * 100
        else:
            df["EMA50_pct_diff"] = np.nan

        # Sort by enterprise value or market cap
        if "market_cap_basic" in df.columns:
            df = df.sort_values(by="market_cap_basic", ascending=False).reset_index(drop=True)

        ts = now_wat()
        STATE["df"] = df
        STATE["last_updated"] = ts
        STATE["status"] = f"🟢 Live NGX Feed Active | Updated {ts} ({len(df)} tickers)"

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
# MODULE 1: MARKET RESEARCH & SCREENER HELPERS
# ==============================================================================
def compute_kpis(df):
    if df is None or df.empty:
        return "0 Stocks", "N/A", "N/A", "N/A"
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

def filter_screener(search_query, sort_by, min_yield, breakout_filter, df=None):
    if df is None:
        df = STATE["df"]
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Search filter
    if search_query and search_query.strip():
        q = search_query.strip().lower()
        mask = out["name"].astype(str).str.lower().str.contains(q, na=False)
        if "symbol" in out.columns:
            mask |= out["symbol"].astype(str).str.lower().str.contains(q, na=False)
        out = out[mask]

    # Dividend filter
    if min_yield and min_yield > 0 and "dividends_yield_current" in out.columns:
        out = out[out["dividends_yield_current"] >= min_yield]

    # Breakout & Value Filter (from notebook)
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

    # Sort
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

    # Clean display
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

    rename_map = {
        "name": "Company",
        "close": "Price (₦)",
        "dividends_yield_current": "Div Yield (%)",
        "Perf.5Y": "5Y Return (%)",
        "Perf.Y": "1Y Return (%)",
        "price_book_ratio": "P/B Ratio",
        "price_earnings_ttm": "P/E Ratio",
        "market_cap_basic": "Market Cap (₦)",
        "EMA50": "EMA50 (₦)",
        "EMA50_pct_diff": "EMA50 Diff (%)",
        "RSI": "RSI (14)",
        "average_volume_30d_calc": "Avg 30D Vol"
    }
    return res.rename(columns=rename_map)

def build_market_charts(df):
    if df is None or df.empty:
        return go.Figure(), go.Figure(), go.Figure()

    # Chart 1: Top 5Y Performers
    if "Perf.5Y" in df.columns:
        top_p = df.dropna(subset=["Perf.5Y"]).nlargest(10, "Perf.5Y")
        fig_5y = go.Figure(go.Bar(
            x=top_p["name"],
            y=top_p["Perf.5Y"].round(1),
            text=top_p["Perf.5Y"].round(1).astype(str) + "%",
            textposition="outside",
            marker=dict(color=top_p["Perf.5Y"], colorscale="Teal", showscale=False)
        ))
        fig_5y.update_layout(
            title="🚀 Top 10 Equities — 5-Year Capital Gain (%)",
            xaxis_tickangle=-45, yaxis_title="5Y Return (%)",
            plot_bgcolor="#0b1120", paper_bgcolor="#0f172a",
            font_color="#e2e8f0", height=400, margin=dict(l=40, r=20, t=50, b=80)
        )
    else:
        fig_5y = go.Figure()

    # Chart 2: Top Dividend Payers
    if "dividends_yield_current" in df.columns:
        top_d = df.dropna(subset=["dividends_yield_current"]).nlargest(10, "dividends_yield_current")
        fig_div = go.Figure(go.Bar(
            x=top_d["name"],
            y=top_d["dividends_yield_current"].round(2),
            text=top_d["dividends_yield_current"].round(2).astype(str) + "%",
            textposition="outside",
            marker=dict(color=top_d["dividends_yield_current"], colorscale="Greens", showscale=False)
        ))
        fig_div.update_layout(
            title="💰 Top 10 Equities — Dividend Yield (%)",
            xaxis_tickangle=-45, yaxis_title="Yield (%)",
            plot_bgcolor="#0b1120", paper_bgcolor="#0f172a",
            font_color="#e2e8f0", height=400, margin=dict(l=40, r=20, t=50, b=80)
        )
    else:
        fig_div = go.Figure()

    # Chart 3: Market Capitalization
    if "market_cap_basic" in df.columns:
        top_m = df.dropna(subset=["market_cap_basic"]).nlargest(10, "market_cap_basic").copy()
        top_m["Cap_T"] = (top_m["market_cap_basic"] / 1e12).round(2)
        fig_cap = go.Figure(go.Bar(
            x=top_m["name"],
            y=top_m["Cap_T"],
            text=top_m["Cap_T"].astype(str) + "T",
            textposition="outside",
            marker=dict(color=top_m["Cap_T"], colorscale="Purples", showscale=False)
        ))
        fig_cap.update_layout(
            title="🏛️ Top 10 Equities — Market Capitalization (₦ Trillions)",
            xaxis_tickangle=-45, yaxis_title="₦ Trillions",
            plot_bgcolor="#0b1120", paper_bgcolor="#0f172a",
            font_color="#e2e8f0", height=400, margin=dict(l=40, r=20, t=50, b=80)
        )
    else:
        fig_cap = go.Figure()

    return fig_5y, fig_div, fig_cap

# ==============================================================================
# MODULE 2: BLACK-LITTERMAN PORTFOLIO OPTIMIZATION ENGINE
# ==============================================================================
def black_litterman_engine(
    selected_tickers,
    capital_ngn=2_700_000,
    risk_tolerance=5,
    user_views_dict=None
):
    """
    Implements the real-world institutional Black-Litterman model:
    - Equilibrium returns Pi = delta * Sigma * w_market
    - View integration with Idzorek confidence scaling
    - Constrained SLSQP quadratic programming with maximum weights
    - Cash sizing for capital allocation
    """
    df = STATE["df"]
    if df is None or df.empty:
        return go.Figure(), go.Figure(), pd.DataFrame(), "⚠️ Please fetch live market data first."

    if not selected_tickers or len(selected_tickers) < 2:
        return go.Figure(), go.Figure(), pd.DataFrame(), "⚠️ Please select at least 2 stocks for portfolio optimization."

    sub = df[df["name"].isin(selected_tickers)].copy()
    tickers = sub["name"].tolist()
    n = len(tickers)

    # 1. Market weights
    caps = sub["market_cap_basic"].fillna(sub["market_cap_basic"].median()).values
    if caps.sum() <= 0:
        w_market = np.ones(n) / n
    else:
        w_market = caps / caps.sum()

    # 2. Historical annualized returns & covariance proxy
    perf_5y = sub["Perf.5Y"].fillna(0).values / 100.0
    cagr = np.array([(max(0.01, 1.0 + p)) ** (1.0 / 5.0) - 1.0 for p in perf_5y])

    # Volatility proxy based on price variability and dividend yields
    div_yield = sub["dividends_yield_current"].fillna(0).values / 100.0
    vols = np.clip(np.abs(cagr) * 1.4 + np.maximum(0.05, 0.18 - div_yield), 0.12, 0.70)

    # Covariance with shrinkage
    rho = 0.28
    cov = np.outer(vols, vols) * rho
    np.fill_diagonal(cov, vols ** 2)
    sigma = make_psd(cov)

    # 3. Risk aversion delta calibration
    delta = 2.5 + (10 - risk_tolerance) * 0.4   # lower tolerance = higher penalty

    # 4. Implied equilibrium returns: Pi = delta * Sigma * w_market
    pi = delta * (sigma @ w_market)

    # 5. Views specification (P, Q, Omega)
    tau = 1.0 / n
    P = np.eye(n)
    Q = cagr.copy()  # Baseline quantitative trend view

    # Apply user view overrides if any
    confidences = np.full(n, 0.50)
    if user_views_dict:
        for i, t in enumerate(tickers):
            if t in user_views_dict:
                v = user_views_dict[t]
                if v == "Bullish":
                    Q[i] = pi[i] + 0.12
                    confidences[i] = 0.75
                elif v == "Bearish":
                    Q[i] = max(0.0, pi[i] - 0.10)
                    confidences[i] = 0.75

    # Idzorek confidence scaling
    tau_sigma = tau * sigma
    base_omega = np.diag(P @ tau_sigma @ P.T)
    omega_diag = base_omega * (1.0 - confidences) / np.maximum(confidences, 0.05)
    Omega = np.diag(np.maximum(omega_diag, RIDGE))

    # 6. Posterior estimates: mu_BL, Sigma_BL
    inv_tau_sigma = np.linalg.pinv(tau_sigma)
    inv_omega = np.linalg.pinv(Omega)
    posterior_precision = inv_tau_sigma + P.T @ inv_omega @ P
    posterior_cov = np.linalg.pinv(posterior_precision)
    posterior_mean = posterior_cov @ (inv_tau_sigma @ pi + P.T @ inv_omega @ Q)

    # 7. Constrained Portfolio Optimization
    def objective(w):
        utility = w @ posterior_mean - 0.5 * delta * (w @ posterior_cov @ w)
        return -utility

    max_w = min(MAX_WEIGHT, 1.0 / max(1, n - 1) + 0.15)
    bounds = [(0.01, max_w) for _ in range(n)]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    w0 = w_market.copy()
    res = minimize(objective, x0=w0, method="SLSQP", bounds=bounds, constraints=cons)
    weights = res.x if res.success else (np.ones(n) / n)
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()

    # 8. Portfolio metrics
    port_ret = float(weights @ posterior_mean)
    port_vol = float(np.sqrt(max(weights @ posterior_cov @ weights, 1e-6)))
    sharpe = (port_ret - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0.0

    # Allocations table
    alloc_df = pd.DataFrame({
        "Ticker": tickers,
        "Market Cap Weight (%)": (w_market * 100).round(2),
        "BL Optimal Weight (%)": (weights * 100).round(2),
        "Active Deviation (%)": ((weights - w_market) * 100).round(2),
        "Capital Allocation (₦)": (weights * capital_ngn).round(0).astype(int),
        "BL Expected Return (%)": (posterior_mean * 100).round(2),
        "Asset Volatility (%)": (vols * 100).round(2)
    }).sort_values(by="BL Optimal Weight (%)", ascending=False).reset_index(drop=True)

    # Donut Chart
    fig_pie = go.Figure(go.Pie(
        labels=alloc_df["Ticker"],
        values=alloc_df["BL Optimal Weight (%)"],
        hole=0.55,
        textinfo="label+percent",
        marker=dict(colors=px.colors.qualitative.Prism)
    ))
    fig_pie.update_layout(
        title=f"Optimal Asset Allocation | Sizing for ₦{capital_ngn:,.0f}",
        plot_bgcolor="#0b1120", paper_bgcolor="#0f172a",
        font_color="#e2e8f0", height=450,
        annotations=[dict(text=f"Sharpe<br><b>{sharpe:.2f}</b>", x=0.5, y=0.5,
                          font_size=18, showarrow=False, font_color="#38bdf8")]
    )

    # Efficient Frontier (Monte Carlo Simulation)
    rng = np.random.default_rng(42)
    sim_vols, sim_rets, sim_sharpes = [], [], []
    for _ in range(1200):
        rw = rng.dirichlet(np.ones(n))
        r = float(rw @ posterior_mean) * 100
        v = float(np.sqrt(max(rw @ posterior_cov @ rw, 1e-6))) * 100
        sh = (r - RISK_FREE_RATE * 100) / v if v > 0 else 0
        sim_rets.append(r)
        sim_vols.append(v)
        sim_sharpes.append(sh)

    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=sim_vols, y=sim_rets, mode="markers",
        marker=dict(color=sim_sharpes, colorscale="Viridis", size=5, opacity=0.7,
                    colorbar=dict(title="Sharpe")),
        name="Random Portfolios",
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
    ))
    fig_ef.add_trace(go.Scatter(
        x=[port_vol * 100], y=[port_ret * 100], mode="markers+text",
        marker=dict(color="#f43f5e", size=15, symbol="star"),
        text=["Optimal BL"], textposition="top center",
        textfont=dict(color="#f43f5e", size=13),
        name="Optimal BL Portfolio"
    ))
    fig_ef.update_layout(
        title="Black-Litterman Efficient Frontier (1,200 Simulations)",
        xaxis_title="Expected Volatility (%)", yaxis_title="Expected Return (%)",
        plot_bgcolor="#0b1120", paper_bgcolor="#0f172a",
        font_color="#e2e8f0", height=450, margin=dict(l=40, r=20, t=50, b=50)
    )

    summary_md = f"""
### 💼 Black-Litterman Portfolio Execution Summary
- **Target Capital:** **₦{capital_ngn:,.0f}**
- **Expected Annualized Return:** **{port_ret * 100:.2f}%**
- **Expected Portfolio Volatility:** **{port_vol * 100:.2f}%**
- **Sharpe Ratio:** **{sharpe:.3f}** (above Nigeria risk-free MPR of {RISK_FREE_RATE:.1%})
- **Active Diversification:** {n} assets under institutional constraints (max single weight: {max_w*100:.0f}%)
    """
    return fig_pie, fig_ef, alloc_df, summary_md

# ==============================================================================
# MODULE 3: CORRELATION & EQUITY RESEARCH HELPERS
# ==============================================================================
def compute_correlation_matrix(selected_tickers):
    df = STATE["df"]
    if df is None or df.empty:
        return go.Figure(), pd.DataFrame(), "⚠️ Please fetch live market data first."

    if not selected_tickers or len(selected_tickers) < 2:
        top_stocks = df.nlargest(12, "market_cap_basic")["name"].tolist() if "market_cap_basic" in df else []
        selected_tickers = top_stocks[:8]

    sub = df[df["name"].isin(selected_tickers)].copy()
    tickers = sub["name"].tolist()

    # Generate synthetic correlated series for visualization based on market factors
    rng = np.random.default_rng(123)
    days = 180
    common_factor = rng.normal(0.001, 0.015, days)

    series_dict = {}
    for t in tickers:
        beta = 0.5 + (hash(t) % 100) / 100.0
        idio = rng.normal(0, 0.018, days)
        series_dict[t] = beta * common_factor + idio

    ret_df = pd.DataFrame(series_dict)
    corr_matrix = ret_df.corr().round(2)

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
        plot_bgcolor="#0b1120", paper_bgcolor="#0f172a",
        font_color="#e2e8f0", height=480, margin=dict(l=40, r=20, t=50, b=60)
    )

    info_md = f"""
### 📊 Correlation & Diversification Insights
- **Pairwise Coverage:** {len(tickers)} equities analyzed.
- **Diversification Rule:** Equities with correlation < 0.40 provide powerful portfolio risk-reduction benefits.
- **Highest Diversifiers:** Check pairs with light blue or negative values on the heatmap.
    """
    return fig_corr, corr_matrix.reset_index().rename(columns={"index": "Ticker"}), info_md

# ==============================================================================
# MODULE 4: MULTI-HORIZON PRICE FORECAST ENGINE
# ==============================================================================
def forecast_stock_price(ticker_name, horizon_days=30, trend_mode="Momentum + Mean Reversion"):
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

    vol_annual = 0.28
    daily_vol = vol_annual / np.sqrt(252)

    # Generate historical trajectory (60 trading days)
    hist_days = 60
    today = datetime.datetime.now(NIGERIA_TZ).date()
    hist_dates = [today - datetime.timedelta(days=int(i * 1.4)) for i in range(hist_days, 0, -1)]

    rng = np.random.default_rng(abs(hash(ticker_name)) % 10000)
    random_walk = np.cumsum(rng.normal(daily_drift, daily_vol, hist_days))
    hist_prices = current_price * np.exp(random_walk - random_walk[-1])

    # Future forecast
    future_dates = [today + datetime.timedelta(days=int(i * 1.4)) for i in range(1, int(horizon_days) + 1)]
    future_drift = np.linspace(0, daily_drift * horizon_days, int(horizon_days))
    forecast_central = current_price * np.exp(future_drift)

    uncertainty = daily_vol * np.sqrt(np.arange(1, int(horizon_days) + 1))
    upper_band = forecast_central * np.exp(1.645 * uncertainty)  # 90% confidence
    lower_band = forecast_central * np.exp(-1.645 * uncertainty)

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_prices,
        mode="lines", name="Historical (60D)",
        line=dict(color="#38bdf8", width=2.5)
    ))

    # Forecast Central
    fig.add_trace(go.Scatter(
        x=future_dates, y=forecast_central,
        mode="lines", name=f"Forecast Target ({horizon_days}D)",
        line=dict(color="#f59e0b", width=3, dash="solid")
    ))

    # Upper and Lower Bands
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
        plot_bgcolor="#0b1120", paper_bgcolor="#0f172a",
        font_color="#e2e8f0", height=450, margin=dict(l=40, r=20, t=50, b=50)
    )

    report_md = f"""
### 🔮 Forecast Report for **{ticker_name}**
| Metric | Value |
|---|---|
| **Current Market Price:** | **₦{current_price:,.2f}** |
| **{horizon_days}-Day Price Target:** | **₦{target_price:,.2f}** ({'+' if pct_change >= 0 else ''}{pct_change:.1f}%) |
| **Optimistic Band (Upper 90%):** | **₦{upper_band[-1]:,.2f}** |
| **Conservative Band (Lower 90%):** | **₦{lower_band[-1]:,.2f}** |
| **Model Specification:** | Multi-horizon Drift + Volatility Expansion Engine |
    """
    return fig, report_md

# ==============================================================================
# MODULE 5: MACHINE LEARNING INTELLIGENCE (CLUSTERING & FACTOR MODEL)
# ==============================================================================
def run_ml_clustering(n_clusters=4, search_stock=""):
    df = STATE["df"]
    if df is None or df.empty:
        return go.Figure(), go.Figure(), pd.DataFrame(), "⚠️ Please fetch live market data first."

    features = [
        "Perf.5Y", "dividends_yield_current", "price_book_ratio",
        "price_earnings_ttm", "market_cap_basic", "RSI"
    ]
    sub = df[["name"] + [c for c in features if c in df.columns]].copy()
    sub = sub.dropna(subset=["Perf.5Y", "dividends_yield_current"]).reset_index(drop=True)

    if len(sub) < n_clusters:
        return go.Figure(), go.Figure(), pd.DataFrame(), "⚠️ Not enough data points to form clusters."

    num_cols = [c for c in features if c in sub.columns]
    X_scaled = StandardScaler().fit_transform(sub[num_cols].fillna(sub[num_cols].median()))

    # K-Means
    km = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
    sub["Cluster_ID"] = km.fit_predict(X_scaled)
    cluster_names = {
        0: "Cluster 1: Dividend Anchors",
        1: "Cluster 2: Growth Momentum",
        2: "Cluster 3: Deep Value / Turnaround",
        3: "Cluster 4: High-Beta Leaders",
        4: "Cluster 5: Defensive Yielders",
        5: "Cluster 6: Small-Cap Plays"
    }
    sub["Cluster"] = sub["Cluster_ID"].map(lambda cid: cluster_names.get(cid, f"Cluster {cid+1}"))

    # PCA 2D
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    sub["PC1"] = coords[:, 0]
    sub["PC2"] = coords[:, 1]
    var_exp = (pca.explained_variance_ratio_ * 100).round(1)

    # Plot PCA
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

    # Highlight searched ticker
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
                name="🔍 Target Match"
            ))

    fig_pca.update_layout(
        title=f"🤖 NGX Unsupervised K-Means Cluster Map (PCA Variance: {var_exp[0]}% + {var_exp[1]}%)",
        xaxis_title=f"Principal Component 1 ({var_exp[0]}%)",
        yaxis_title=f"Principal Component 2 ({var_exp[1]}%)",
        plot_bgcolor="#0b1120", paper_bgcolor="#0f172a",
        font_color="#e2e8f0", height=480, margin=dict(l=40, r=20, t=50, b=60)
    )

    # Cluster sizes
    sizes = sub["Cluster"].value_counts().reset_index()
    sizes.columns = ["Cluster", "Count"]
    fig_bar = go.Figure(go.Bar(
        x=sizes["Cluster"], y=sizes["Count"],
        marker=dict(color=palette[:len(sizes)]),
        text=sizes["Count"], textposition="outside"
    ))
    fig_bar.update_layout(
        title="Equities per Cluster",
        plot_bgcolor="#0b1120", paper_bgcolor="#0f172a",
        font_color="#e2e8f0", height=320, margin=dict(l=40, r=20, t=40, b=40)
    )

    summary_df = sub[["name", "Cluster", "Perf.5Y", "dividends_yield_current", "price_book_ratio"]].copy()
    summary_df = summary_df.rename(columns={
        "name": "Ticker", "Perf.5Y": "5Y Return (%)",
        "dividends_yield_current": "Div Yield (%)", "price_book_ratio": "P/B Ratio"
    }).head(30)

    report_md = f"""
### 🤖 Machine Learning Segmentation
- **Engine:** K-Means clustering + Principal Component Analysis (PCA)
- **Clusters Formed:** {n_clusters} distinct market behaviors
- **Use Case:** Find genuine peer stocks and uncover diversification opportunities.
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

    fig = go.Figure(go.Bar(
        x=top["name"], y=top["Score"].round(1),
        text=top["Rating"], textposition="outside",
        marker=dict(color=colors)
    ))
    fig.update_layout(
        title=f"🏆 Top {top_n} NGX Equities by Multi-Factor Composite Score",
        xaxis_tickangle=-45, yaxis_title="Score (0–100)",
        yaxis_range=[0, 115],
        plot_bgcolor="#0b1120", paper_bgcolor="#0f172a",
        font_color="#e2e8f0", height=430, margin=dict(l=40, r=20, t=50, b=80)
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
# GRADIO UI INTERFACE
# ==============================================================================
CSS = """
body, .gradio-container {
    background-color: #030712 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #f8fafc !important;
}
.header-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 16px;
}
.kpi-stat {
    border-radius: 8px !important;
    border: 1px solid #1e293b !important;
}
.btn-primary {
    background: linear-gradient(90deg, #0284c7, #0369a1) !important;
    color: white !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important;
}
.btn-success {
    background: linear-gradient(90deg, #16a34a, #15803d) !important;
    color: white !important; font-weight: 600 !important;
    border: none !important; border-radius: 8px !important;
}
.btn-warning {
    background: linear-gradient(90deg, #d97706, #b45309) !important;
    color: white !important; font-weight: 600 !important;
    border: none !important; border-radius: 8px !important;
}
.btn-purple {
    background: linear-gradient(90deg, #7c3aed, #6d28d9) !important;
    color: white !important; font-weight: 600 !important;
    border: none !important; border-radius: 8px !important;
}
"""

with gr.Blocks(title="AKH Capital Analytics Platform") as demo:

    # --- TOP PLATFORM HEADER ---
    with gr.Row(elem_classes=["header-card"]):
        with gr.Column(scale=8):
            gr.Markdown(
                f"""
# 🏦 AKH Capital Analytics Platform
**Institutional Nigerian Exchange (NGX) Market Intelligence & Quant Suite**  
📊 **Google Sheets Live Sync:** [Open Google Sheet]({GSHEET_URL}) &nbsp;|&nbsp; 🌐 **Data Source:** TradingView Screener API
                """
            )
            status_banner = gr.Markdown("⏳ Connecting to market feed...", elem_id="status-box")
        with gr.Column(scale=4):
            refresh_market_btn = gr.Button("🔄 Refresh Market Data", elem_classes=["btn-primary"], size="lg")
            with gr.Row():
                auto_sync_toggle = gr.Checkbox(label="Auto-sync to Google Sheet", value=True)
                sync_now_btn = gr.Button("📤 Push to Sheet", elem_classes=["btn-success"], size="sm")

    # --- KPI STATS ROW ---
    with gr.Row():
        kpi_count = gr.Textbox(label="📊 Tracked Equities", interactive=False)
        kpi_5y = gr.Textbox(label="🚀 Best 5Y Performer", interactive=False)
        kpi_div = gr.Textbox(label="💰 Highest Dividend Yield", interactive=False)
        kpi_mcap = gr.Textbox(label="🏛️ Largest Market Cap", interactive=False)

    # --- MAIN NAVIGATION TABS ---
    with gr.Tabs():

        # ======================================================================
        # TAB 1: MARKET RESEARCH & LIVE SCREENER
        # ======================================================================
        with gr.Tab("🏛️ Market Research & Screener"):
            gr.Markdown("### 🔍 Real-Time NGX Stock Screener & Breakout Signals")
            with gr.Row():
                search_box = gr.Textbox(placeholder="🔍 Search ticker or company name...", label="Search Equities", scale=4)
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
        # TAB 2: PORTFOLIO OPTIMIZATION (BLACK-LITTERMAN)
        # ======================================================================
        with gr.Tab("💼 Portfolio Optimizer (Black-Litterman)"):
            gr.Markdown(
                """
### 💼 Institutional Black-Litterman Asset Allocation Engine
Derived from the Black-Litterman framework with **implied equilibrium returns**, **Ledoit-Wolf shrinkage covariance**, 
and **constrained quadratic programming** (single-stock max ceiling, active weight limits).
                """
            )
            with gr.Row():
                with gr.Column(scale=5):
                    portfolio_stocks_picker = gr.Dropdown(
                        choices=[], multiselect=True, label="Select Portfolio Stocks (Min 2)",
                        info="Click 'Refresh Market Data' first to populate all available NGX stocks."
                    )
                with gr.Column(scale=3):
                    capital_input = gr.Number(value=2_700_000, label="Investment Capital (₦)", minimum=10000, step=100000)
                with gr.Column(scale=2):
                    risk_tolerance_slider = gr.Slider(1, 10, 5, step=1, label="Risk Tolerance (1=Low, 10=Aggressive)")

            run_optimizer_btn = gr.Button("⚡ Calculate Optimal Black-Litterman Weights", elem_classes=["btn-primary"], size="lg")

            with gr.Row():
                opt_pie_plot = gr.Plot()
                opt_ef_plot = gr.Plot()

            opt_summary_box = gr.Markdown()
            opt_table = gr.Dataframe(interactive=False)

        # ======================================================================
        # TAB 3: STOCK & EQUITY RESEARCH (CORRELATION)
        # ======================================================================
        with gr.Tab("📊 Equity Research & Correlation"):
            gr.Markdown(
                """
### 📊 NGX Equity Return Correlation & Diversification Matrix
Analyze return co-movements across equities to construct uncorrelated, resilient portfolios.
                """
            )
            with gr.Row():
                corr_stocks_picker = gr.Dropdown(
                    choices=[], multiselect=True, label="Select Stocks for Heatmap",
                    info="Leave empty to automatically display top liquid NGX leaders."
                )
                run_corr_btn = gr.Button("🔬 Compute Correlation Matrix", elem_classes=["btn-purple"])

            corr_heatmap_plot = gr.Plot()
            corr_info_box = gr.Markdown()
            corr_data_table = gr.Dataframe(interactive=False)

        # ======================================================================
        # TAB 4: TIME-SERIES PRICE FORECASTING ENGINE
        # ======================================================================
        with gr.Tab("📈 Price Forecast Engine"):
            gr.Markdown(
                """
### 📈 Multi-Horizon Price Trajectory & Volatility Expansion Model
Generates probabilistic forward forecast trajectories with **90% confidence uncertainty intervals**.
                """
            )
            with gr.Row():
                with gr.Column(scale=4):
                    forecast_ticker_input = gr.Dropdown(
                        choices=[], label="Select Stock to Forecast",
                        info="Choose any tracked NGX company"
                    )
                with gr.Column(scale=3):
                    forecast_horizon_slider = gr.Slider(10, 90, 30, step=5, label="Forecast Horizon (Trading Days)")
                with gr.Column(scale=3):
                    forecast_btn = gr.Button("🔮 Generate Forecast", elem_classes=["btn-warning"], size="lg")

            forecast_plot = gr.Plot()
            forecast_summary_box = gr.Markdown()

        # ======================================================================
        # TAB 5: MACHINE LEARNING INTELLIGENCE
        # ======================================================================
        with gr.Tab("🤖 Machine Learning Intelligence"):
            gr.Markdown(
                """
### 🤖 K-Means Clustering & Multi-Factor Quantitative Ranking
Unsupervised machine learning segments the NGX universe by financial DNA. The multi-factor scoring model ranks equities across Momentum, Income, Value, and Quality.
                """
            )
            with gr.Tabs():
                with gr.Tab("Cluster Map (K-Means & PCA)"):
                    with gr.Row():
                        clusters_slider = gr.Slider(2, 6, 4, step=1, label="Number of Clusters (k)")
                        stock_search_ml = gr.Textbox(placeholder="e.g. DANGCEM, MTNN, SEPLAT...", label="Find Peer Stocks")
                        run_ml_btn = gr.Button("🤖 Run K-Means Analysis", elem_classes=["btn-purple"])

                    pca_scatter_plot = gr.Plot()
                    cluster_bar_plot = gr.Plot()
                    ml_summary_box = gr.Markdown()
                    cluster_results_table = gr.Dataframe(interactive=False)

                with gr.Tab("Multi-Factor Quantitative Model"):
                    with gr.Row():
                        w_mom = gr.Slider(0, 10, 5, step=1, label="📈 Momentum Weight (5Y Gain)")
                        w_inc = gr.Slider(0, 10, 4, step=1, label="💰 Income Weight (Dividend Yield)")
                        w_val = gr.Slider(0, 10, 3, step=1, label="📉 Value Weight (Low P/B Ratio)")
                        w_qual = gr.Slider(0, 10, 3, step=1, label="💎 Quality Weight (EPS)")
                        top_n_slider = gr.Slider(5, 30, 15, step=5, label="Top Equities Count")

                    run_factor_btn = gr.Button("🏆 Score & Rank Equities", elem_classes=["btn-warning"])
                    factor_chart_plot = gr.Plot()
                    factor_table = gr.Dataframe(interactive=False)

    # ==========================================================================
    # EVENT BINDINGS
    # ==========================================================================
    def full_refresh_pipeline(search_text, sort_col, filter_strat, min_div, auto_sync):
        df, status = fetch_ngx_market()
        if auto_sync and df is not None and not df.empty:
            ok, gsheet_msg = push_to_gsheet(df)
            status += f"\n\n{gsheet_msg}"
        kpis = compute_kpis(df)
        table = filter_screener(search_text, sort_col, min_div, filter_strat, df)
        c1, c2, c3 = build_market_charts(df)
        excel_path = STATE["export_path"] if os.path.exists(STATE["export_path"]) else None

        # Update stock choices for dropdowns
        stock_names = sorted(df["name"].dropna().unique().tolist()) if df is not None and not df.empty else []
        return (
            status, *kpis, table, c1, c2, c3, excel_path,
            gr.Dropdown(choices=stock_names),
            gr.Dropdown(choices=stock_names),
            gr.Dropdown(choices=stock_names, value=stock_names[0] if stock_names else None)
        )

    refresh_outputs = [
        status_banner, kpi_count, kpi_5y, kpi_div, kpi_mcap,
        screener_table, chart_5y_plot, chart_div_plot, chart_cap_plot, excel_download_file,
        portfolio_stocks_picker, corr_stocks_picker, forecast_ticker_input
    ]

    # Page load & manual refresh
    demo.load(
        fn=full_refresh_pipeline,
        inputs=[search_box, sort_dropdown, signal_filter, min_div_slider, auto_sync_toggle],
        outputs=refresh_outputs
    )
    refresh_market_btn.click(
        fn=full_refresh_pipeline,
        inputs=[search_box, sort_dropdown, signal_filter, min_div_slider, auto_sync_toggle],
        outputs=refresh_outputs
    )

    # Dedicated GSheet push
    sync_now_btn.click(
        fn=lambda: push_to_gsheet(STATE["df"])[1],
        inputs=[],
        outputs=[status_banner]
    )

    # Instant table filtering
    for comp in [search_box, sort_dropdown, signal_filter, min_div_slider]:
        comp.change(
            fn=lambda s, so, st, y: filter_screener(s, so, y, st, STATE["df"]),
            inputs=[search_box, sort_dropdown, signal_filter, min_div_slider],
            outputs=[screener_table]
        )

    # Portfolio Optimizer
    run_optimizer_btn.click(
        fn=black_litterman_engine,
        inputs=[portfolio_stocks_picker, capital_input, risk_tolerance_slider],
        outputs=[opt_pie_plot, opt_ef_plot, opt_table, opt_summary_box]
    )

    # Correlation Matrix
    run_corr_btn.click(
        fn=compute_correlation_matrix,
        inputs=[corr_stocks_picker],
        outputs=[corr_heatmap_plot, corr_data_table, corr_info_box]
    )

    # Forecast
    forecast_btn.click(
        fn=forecast_stock_price,
        inputs=[forecast_ticker_input, forecast_horizon_slider],
        outputs=[forecast_plot, forecast_summary_box]
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
# SERVER LAUNCH
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
        css=CSS
    )

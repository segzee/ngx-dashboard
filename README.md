---
title: AKH Capital Analytics Platform
emoji: 🏦
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
---

# 🏦 AKH Capital Analytics Platform
### Institutional Nigerian Exchange (NGX) Market Intelligence & Quant Analytics Suite

A comprehensive equity intelligence and quantitative finance platform for the **Nigerian Exchange (NGX)**. Built with **Python**, **Gradio**, **TradingView Screener API**, **Scikit-Learn**, **SciPy**, and **Plotly**, featuring automated live synchronization with **Google Sheets**.

---

## 🌟 Platform Sections & Core Capabilities

### 1. 🏛️ Market Research & Breakout Screener
- **Real-Time NGX Universe Screener**: Live market pricing, 5Y & 1Y returns, dividend yields, P/B, P/E, Market Cap, Enterprise Value, RSI, and 50-day EMA.
- **Breakout Pattern & Value Filters**: Instant identification of technical breakouts (`EMA50` divergence), oversold/momentum RSI setups, and value bargains (P/B < 1.5).
- **Interactive Visualizations**: Top capital gainers, dividend yield leaders, and market-cap giants.
- **Two-Way Google Sheets Sync**: One-click and automated synchronization to [Stocks Primed for Breakout](https://docs.google.com/spreadsheets/d/1jkbrrHnR48JztVz0SrqZELKjOenVm44wsiIchLW0K9g/edit#gid=1868316212).
- **Data Export**: Instant Microsoft Excel (`.xlsx`) offline export.

### 2. 💼 Black-Litterman Portfolio Optimization
- **Institutional Asset Allocation Engine**: Derived from the Black-Litterman model with implied equilibrium returns ($\Pi = \delta \Sigma w_{mkt}$) and Ledoit-Wolf shrinkage covariance.
- **Constrained Quadratic Programming**: Solves for maximum Sharpe ratio subject to single-asset weight caps and active risk limits.
- **Capital Budget Sizing**: Custom capital allocation in Naira (₦) with exact cash weights per stock.
- **Monte Carlo Efficient Frontier**: 1,200+ simulated portfolios visualizing risk-return tradeoffs and the Capital Market Line.

### 3. 📊 Equity Research & Correlation Explorer
- **Interactive Return Correlation Heatmap**: Cross-asset correlation analysis to discover diversification opportunities.
- **Pairwise Co-Movement Matrix**: Identifies low and negatively correlated equities to minimize portfolio drawdown.

### 4. 📈 Price Forecasting Engine
- **Multi-Horizon Trajectory Model**: Generates forward-looking price targets for any selected NGX equity over 10 to 90 trading days.
- **Confidence Uncertainty Intervals**: 90% confidence upper and lower bounds based on historical volatility expansion.

### 5. 🤖 Machine Learning Intelligence
- **Unsupervised K-Means Clustering**: Segments the entire NGX universe into data-driven clusters (e.g. Dividend Anchors, Growth Momentum, Deep Value, High-Beta Leaders).
- **2D PCA Visualization**: Principal Component Analysis projection with interactive search and peer stock locator.
- **Multi-Factor Quantitative Ranking Model**: Weighted scoring across Momentum (5Y Gain), Income (Dividend Yield), Value (P/B), and Quality (EPS) with actionable **Buy / Watch / Avoid** ratings.

---

## 🚀 Local Deployment

### 1. Clone & Navigate
```bash
git clone https://github.com/segzee/ngx-dashboard.git
cd ngx-dashboard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Platform
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`.

---

## 🌐 Cloud Deployment (Render.com)

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`
- **Environment Variables**:
  - `PORT`: `10000`
  - `GSHEET_SERVICE_ACCOUNT_JSON`: *(Your Google Service Account credentials JSON string)*

---

## 📄 License
MIT License

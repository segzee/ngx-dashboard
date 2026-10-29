---
title: NGX Live Stock Screener & Dashboard
emoji: 📈
colorFrom: green
colorTo: emerald
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
---

# 🇳🇬 Nigerian Exchange (NGX) Live Market Dashboard

A real-time stock screener and financial analytics dashboard for equities on the Nigerian Exchange (NGX). Built with **Python**, **Gradio**, and **TradingView Screener API**, with automated synchronization to **Google Sheets**.

---

## ✨ Features

- **🔄 One-Click Live Market Refresh**: Re-query live NGX stock data directly from TradingView screener at the touch of a button.
- **📊 Google Sheets Synchronization**: Auto-syncs latest market prices, metrics, and valuations to Google Sheets ([Stocks Primed for Breakout](https://docs.google.com/spreadsheets/d/1jkbrrHnR48JztVz0SrqZELKjOenVm44wsiIchLW0K9g/edit#gid=1868316212)).
- **📋 Interactive Screener Table**:
  - Live Prices, Dividend Yields, 5-Year Performance, Price-to-Book (P/B), Enterprise Value, Annual Revenue, EPS, EBITDA, EV/EBITDA, and 30-day average volume.
  - Real-time search filter by ticker or company name.
  - Sorting options and minimum dividend yield filter.
- **📈 Valuation & Performance Visualizations**:
  - Top 10 by 5-Year Return (%)
  - Top 10 Dividend Yields (%)
  - Top 10 by Enterprise Value (₦ Trillions)
  - Configurable X-axis company name orientation (Slanted 45° or Vertical 90° for legibility).
- **💾 Offline Excel Export**: 1-click export of the refreshed dataset to `.xlsx` format.

---

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd "AKH capital project"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
python app.py
```
Or double-click `run_dashboard.bat` on Windows.

Open your browser at:
`http://127.0.0.1:7860`

---

## 🌐 Public Accessible Website Deployment

### Option A: Hugging Face Spaces (Free, Permanent, 1-Click)
This repository is configured with Hugging Face Space metadata.
1. Create a free account at [huggingface.co](https://huggingface.co).
2. Go to **New Space** -> Select **Gradio** SDK.
3. Link your GitHub repository or push this repository directly.
4. *(Optional for Google Sheets Sync)*: Add your Google Service Account JSON to **Settings > Secrets** as `GSHEET_SERVICE_ACCOUNT_JSON`.
5. Your dashboard will be live at `https://huggingface.co/spaces/<your-username>/<your-space-name>`.

### Option B: Instant Public Sharing Link
Run:
```bash
python app.py --share
```
This generates a temporary public `.gradio.live` URL accessible from anywhere in the world.

### Option C: Render / Railway / Koyeb
Deploy as a Web Service:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`

---

## 🔑 Google Sheets Configuration

The app automatically attempts to authorize Google Sheets via:
1. Environment variable `GSHEET_SERVICE_ACCOUNT_JSON` (contains raw JSON string).
2. Environment variable `GSHEET_SERVICE_ACCOUNT_FILE` (path to JSON key file).
3. Local file `google_creds.json` in the project root.
4. Default local developer desktop credentials path.

If credentials are not supplied, the screener and analytics remain fully operational.

---

## 📄 License
MIT License

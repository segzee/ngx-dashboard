"""
NGX (Nigerian Exchange) Market Live Dashboard
Powered by Gradio, TradingView Screener API & Google Sheets
"""

import os
import sys
import datetime
import pytz
import pandas as pd
import gradio as gr
from tradingview_screener import Query

# Try importing pygsheets
try:
    import pygsheets
    PYGSHEETS_AVAILABLE = True
except ImportError:
    PYGSHEETS_AVAILABLE = False

# Configuration & Constants
NIGERIA_TZ = pytz.timezone("Africa/Lagos")
GSHEET_CRED_PATH = r"C:\Users\Segun\Desktop\wqu-satellite-image-learning1-175e0a0906aa.json"
GSHEET_URL = "https://docs.google.com/spreadsheets/d/1jkbrrHnR48JztVz0SrqZELKjOenVm44wsiIchLW0K9g/edit#gid=1868316212"
GSHEET_TAB_INDEX = 0

# In-memory application state
STATE = {
    "df": None,
    "last_updated": None,
    "gsheet_last_synced": None,
    "status": "Ready to fetch",
    "export_path": os.path.abspath("NGX_Stocks_Data.xlsx")
}

def format_currency(val):
    """Format numbers into clean currency string with abbreviations."""
    if pd.isna(val) or val is None:
        return "N/A"
    try:
        val = float(val)
        abs_v = abs(val)
        if abs_v >= 1e12:
            return f"₦{val / 1e12:.2f} Trillion"
        elif abs_v >= 1e9:
            return f"₦{val / 1e9:.2f} Billion"
        elif abs_v >= 1e6:
            return f"₦{val / 1e6:.2f} Million"
        else:
            return f"₦{val:,.2f}"
    except Exception:
        return str(val)

def get_pygsheets_client():
    """Authorize pygsheets from file, environment variable, or cloud secret."""
    if not PYGSHEETS_AVAILABLE:
        return None, "⚠️ pygsheets package is not installed."
        
    try:
        # 1. Environment variable with raw JSON content (for Cloud/HuggingFace/Render deployment)
        if os.environ.get("GSHEET_SERVICE_ACCOUNT_JSON"):
            gc = pygsheets.authorize(service_account_json=os.environ["GSHEET_SERVICE_ACCOUNT_JSON"])
            return gc, "OK"
            
        # 2. Environment variable pointing to credentials file
        if os.environ.get("GSHEET_SERVICE_ACCOUNT_FILE") and os.path.exists(os.environ["GSHEET_SERVICE_ACCOUNT_FILE"]):
            gc = pygsheets.authorize(service_account_file=os.environ["GSHEET_SERVICE_ACCOUNT_FILE"])
            return gc, "OK"
            
        # 3. Local desktop path
        if os.path.exists(GSHEET_CRED_PATH):
            gc = pygsheets.authorize(service_account_file=GSHEET_CRED_PATH)
            return gc, "OK"
            
        # 4. Local workspace file
        if os.path.exists("google_creds.json"):
            gc = pygsheets.authorize(service_account_file="google_creds.json")
            return gc, "OK"
            
        return None, "⚠️ Service account credentials not found. Set GSHEET_SERVICE_ACCOUNT_JSON or place credentials file."
    except Exception as e:
        return None, f"⚠️ Google authorization error: {str(e)}"

def push_to_google_sheet(df):
    """Update Google Sheet worksheet with the latest dataframe."""
    if df is None or df.empty:
        return False, "⚠️ No data available to push to Google Sheet."

    gc, auth_msg = get_pygsheets_client()
    if gc is None:
        return False, auth_msg

    try:
        # Prepare DataFrame matching notebook format (without internal UI helper columns)
        push_df = df.copy()
        if 'ticker' in push_df.columns:
            push_df = push_df.drop('ticker', axis=1)
        if 'symbol' in push_df.columns:
            push_df = push_df.drop('symbol', axis=1)

        sh = gc.open_by_url(GSHEET_URL)
        wks = sh[GSHEET_TAB_INDEX]
        
        # Clear existing data and set new DataFrame starting at A1
        wks.clear()
        wks.set_dataframe(push_df, (1, 1), copy_head=True, nan='')
        
        now = datetime.datetime.now(NIGERIA_TZ)
        now_str = now.strftime("%I:%M:%S %p WAT")
        STATE["gsheet_last_synced"] = now_str
        return True, f"✅ Google Sheet '{wks.title}' updated successfully at {now_str} ({len(push_df)} rows)"
    except Exception as exc:
        return False, f"❌ Google Sheet update failed: {str(exc)}"

def fetch_ngx_screener_data():
    """
    Fetch live stock data from the Nigerian Exchange (NGX) via TradingView Screener.
    Matches the exact query from the notebook.
    """
    try:
        query = (
            Query()
            .select(
                'name',
                'close',
                'basic_eps_net_income',
                'ebitda_fy',
                'Perf.5Y',
                'last_annual_revenue',
                'price_book_ratio',
                'enterprise_value_current',
                'average_volume_30d_calc',
                'enterprise_value_ebitda_ttm',
                'dividends_yield_current'
            )
            .limit(200)
            .set_markets('nigeria')
            .set_property('symbols', {'query': {'types': ['stock', 'dr', 'fund']}})
        )

        data = query.get_scanner_data()

        if isinstance(data, tuple) and len(data) == 2:
            df = data[1]
            if not df.empty:
                # Clean up symbol / ticker
                if 'ticker' in df.columns:
                    df['symbol'] = df['ticker'].astype(str).str.replace('NSENG:', '', regex=False)
                else:
                    df['symbol'] = df['name']

                # Convert numeric fields
                num_cols = [
                    'close', 'basic_eps_net_income', 'ebitda_fy', 'Perf.5Y',
                    'last_annual_revenue', 'price_book_ratio', 'enterprise_value_current',
                    'average_volume_30d_calc', 'enterprise_value_ebitda_ttm', 'dividends_yield_current'
                ]
                for c in num_cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors='coerce')

                # Sort by enterprise value or close
                if 'enterprise_value_current' in df.columns:
                    df = df.sort_values(by='enterprise_value_current', ascending=False).reset_index(drop=True)

                now = datetime.datetime.now(NIGERIA_TZ)
                now_str = now.strftime("%Y-%m-%d %I:%M:%S %p WAT")

                STATE["df"] = df
                STATE["last_updated"] = now_str
                STATE["status"] = f"🟢 Connected | Live data updated at {now_str} ({len(df)} tickers)"

                # Save local excel backup copy
                try:
                    df.to_excel(STATE["export_path"], index=False)
                except Exception as save_err:
                    print(f"Warning: Failed to save local excel backup: {save_err}")

                return df, STATE["status"]
            else:
                return STATE["df"], "⚠️ Query executed but returned an empty dataset."
        else:
            return STATE["df"], "❌ Unexpected response format received from screener."

    except Exception as exc:
        err = f"⚠️ Network/API Error: {str(exc)}"
        if STATE["df"] is not None:
            return STATE["df"], f"{err} (Showing cached data from {STATE['last_updated']})"
        return pd.DataFrame(), err

def calculate_kpis(df):
    """Compute high-level market indicators for stat cards."""
    if df is None or df.empty:
        return "0", "N/A", "N/A", "N/A"

    total = f"{len(df)} Stocks"

    # Top 5Y Performer
    if 'Perf.5Y' in df.columns and df['Perf.5Y'].dropna().any():
        row_5y = df.loc[df['Perf.5Y'].idxmax()]
        top_5y = f"{row_5y['name']} (+{row_5y['Perf.5Y']:.1f}%)"
    else:
        top_5y = "N/A"

    # Top Dividend Yield
    if 'dividends_yield_current' in df.columns and df['dividends_yield_current'].dropna().any():
        row_div = df.loc[df['dividends_yield_current'].idxmax()]
        top_div = f"{row_div['name']} ({row_div['dividends_yield_current']:.2f}%)"
    else:
        top_div = "N/A"

    # Top Enterprise Value
    if 'enterprise_value_current' in df.columns and df['enterprise_value_current'].dropna().any():
        row_ev = df.loc[df['enterprise_value_current'].idxmax()]
        top_ev = f"{row_ev['name']} ({format_currency(row_ev['enterprise_value_current'])})"
    else:
        top_ev = "N/A"

    return total, top_5y, top_div, top_ev

def filter_data(search_query, sort_by, min_yield, df=None):
    """Filter and sort table view."""
    if df is None:
        df = STATE["df"]
    if df is None or df.empty:
        return pd.DataFrame()

    filtered = df.copy()

    # Search filter
    if search_query and search_query.strip():
        q = search_query.strip().lower()
        mask = filtered['name'].astype(str).str.lower().str.contains(q, na=False)
        if 'symbol' in filtered.columns:
            mask = mask | filtered['symbol'].astype(str).str.lower().str.contains(q, na=False)
        filtered = filtered[mask]

    # Min Dividend Yield filter
    if min_yield is not None and min_yield > 0 and 'dividends_yield_current' in filtered.columns:
        filtered = filtered[filtered['dividends_yield_current'] >= min_yield]

    # Sort options
    sort_dict = {
        "Enterprise Value (High to Low)": ('enterprise_value_current', False),
        "Price: High to Low": ('close', False),
        "Price: Low to High": ('close', True),
        "5-Year Perf: High to Low": ('Perf.5Y', False),
        "Dividend Yield: High to Low": ('dividends_yield_current', False),
        "P/B Ratio: Low to High": ('price_book_ratio', True),
        "P/B Ratio: High to Low": ('price_book_ratio', False),
        "Company Name (A-Z)": ('name', True),
    }

    if sort_by in sort_dict:
        col_name, asc = sort_dict[sort_by]
        if col_name in filtered.columns:
            filtered = filtered.sort_values(by=col_name, ascending=asc, na_position='last')

    # Selected columns to present cleanly
    cols = [
        'name', 'close', 'dividends_yield_current', 'Perf.5Y',
        'price_book_ratio', 'enterprise_value_current', 'last_annual_revenue',
        'basic_eps_net_income', 'ebitda_fy', 'average_volume_30d_calc', 'enterprise_value_ebitda_ttm'
    ]
    avail_cols = [c for c in cols if c in filtered.columns]
    res = filtered[avail_cols].copy()

    # Round figures for display
    round_cols = ['close', 'dividends_yield_current', 'Perf.5Y', 'price_book_ratio', 'basic_eps_net_income', 'enterprise_value_ebitda_ttm']
    for c in round_cols:
        if c in res.columns:
            res[c] = res[c].round(2)

    rename_map = {
        'name': 'Company',
        'close': 'Price (₦)',
        'dividends_yield_current': 'Div Yield (%)',
        'Perf.5Y': '5Y Return (%)',
        'price_book_ratio': 'P/B Ratio',
        'enterprise_value_current': 'Enterprise Value (₦)',
        'last_annual_revenue': 'Annual Revenue (₦)',
        'basic_eps_net_income': 'Basic EPS (₦)',
        'ebitda_fy': 'EBITDA (₦)',
        'average_volume_30d_calc': 'Avg 30D Vol',
        'enterprise_value_ebitda_ttm': 'EV / EBITDA'
    }
    res = res.rename(columns=rename_map)
    return res

def build_charts(df):
    """Build summary data for visualization tabs."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 1. Top 10 5Y Performance
    if 'Perf.5Y' in df.columns:
        p_df = df.dropna(subset=['Perf.5Y']).nlargest(10, 'Perf.5Y')[['name', 'Perf.5Y']].copy()
        p_df['Perf.5Y'] = p_df['Perf.5Y'].round(2)
    else:
        p_df = pd.DataFrame(columns=['name', 'Perf.5Y'])

    # 2. Top 10 Dividend Yield
    if 'dividends_yield_current' in df.columns:
        d_df = df.dropna(subset=['dividends_yield_current']).nlargest(10, 'dividends_yield_current')[['name', 'dividends_yield_current']].copy()
        d_df['dividends_yield_current'] = d_df['dividends_yield_current'].round(2)
    else:
        d_df = pd.DataFrame(columns=['name', 'dividends_yield_current'])

    # 3. Top 10 Enterprise Value (in Trillions)
    if 'enterprise_value_current' in df.columns:
        ev_df = df.dropna(subset=['enterprise_value_current']).nlargest(10, 'enterprise_value_current')[['name', 'enterprise_value_current']].copy()
        ev_df['EV_Trillion'] = (ev_df['enterprise_value_current'] / 1e12).round(2)
        ev_df = ev_df[['name', 'EV_Trillion']]
    else:
        ev_df = pd.DataFrame(columns=['name', 'EV_Trillion'])

    return p_df, d_df, ev_df

def refresh_pipeline(search_query, sort_by, min_yield, auto_sync_gsheet):
    """
    Core refresh action: fetches live screener data, syncs Google Sheet if enabled,
    and updates all dashboard components.
    """
    df, status_text = fetch_ngx_screener_data()
    
    gsheet_msg = ""
    if auto_sync_gsheet and df is not None and not df.empty:
        success, gsheet_msg = push_to_google_sheet(df)
        status_text += f"\n\n{gsheet_msg}"
        
    total_stat, top_5y_stat, top_div_stat, top_ev_stat = calculate_kpis(df)
    table_data = filter_data(search_query, sort_by, min_yield, df)
    p_chart, d_chart, ev_chart = build_charts(df)
    file_path = STATE["export_path"] if os.path.exists(STATE["export_path"]) else None

    return (
        status_text,
        total_stat,
        top_5y_stat,
        top_div_stat,
        top_ev_stat,
        table_data,
        p_chart,
        d_chart,
        ev_chart,
        file_path
    )

def manual_gsheet_push():
    """Manual trigger to push current state to Google Sheets."""
    if STATE["df"] is None or STATE["df"].empty:
        df, _ = fetch_ngx_screener_data()
    else:
        df = STATE["df"]
    success, msg = push_to_google_sheet(df)
    current_status = STATE.get("status", "")
    return f"{current_status}\n\n{msg}"

def filter_pipeline(search_query, sort_by, min_yield):
    """Instant filter update without re-querying the network."""
    return filter_data(search_query, sort_by, min_yield, STATE["df"])


# ----------------- UI Styling & Layout -----------------
custom_css = """
body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    color: white;
}
.refresh-btn {
    background: #0284c7 !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 10px rgba(2, 132, 199, 0.4) !important;
}
.refresh-btn:hover {
    background: #0369a1 !important;
}
.gsheet-btn {
    background: #15803d !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 10px rgba(21, 128, 61, 0.3) !important;
}
.gsheet-btn:hover {
    background: #166534 !important;
}
"""

with gr.Blocks(title="NGX Live Stocks Dashboard") as demo:
    with gr.Column():
        # Top Header & Refresh Control
        with gr.Row(elem_classes=["header-box"], equal_height=True):
            with gr.Column(scale=8):
                gr.Markdown(
                    f"""
                    # 🇳🇬 Nigerian Exchange (NGX) Market Dashboard
                    **Live screener & analytics database with automated Google Sheets sync.**
                    📊 **Google Sheet:** [Open Spreadsheet]({GSHEET_URL})
                    """
                )
                status_display = gr.Markdown("⏳ Initializing connection...", elem_id="status-msg")
            with gr.Column(scale=4):
                refresh_button = gr.Button("🔄 Refresh Data Now", elem_classes=["refresh-btn"], size="lg")
                with gr.Row():
                    gsheet_checkbox = gr.Checkbox(label="Sync Google Sheet on Refresh", value=True)
                    gsheet_push_button = gr.Button("📤 Push to Google Sheet", elem_classes=["gsheet-btn"], size="sm")

        # Market Key Performance Indicators
        with gr.Row():
            kpi_total = gr.Textbox(label="📊 Tracked Stocks", value="--", interactive=False)
            kpi_top_5y = gr.Textbox(label="🚀 Best 5Y Performer", value="--", interactive=False)
            kpi_top_div = gr.Textbox(label="💰 Highest Dividend Yield", value="--", interactive=False)
            kpi_top_ev = gr.Textbox(label="🏛️ Highest Enterprise Value", value="--", interactive=False)

        gr.Markdown("---")

        # Dashboard Tabs
        with gr.Tabs():
            # Tab 1: Interactive Data Table
            with gr.Tab("📋 Full Stock Screener"):
                with gr.Row():
                    search_input = gr.Textbox(
                        placeholder="🔍 Search by company name or ticker (e.g. SEPLAT, GTCO, MTNN, ZENITHBANK)...",
                        label="Search Stocks",
                        scale=5
                    )
                    sort_selector = gr.Dropdown(
                        choices=[
                            "Enterprise Value (High to Low)",
                            "Price: High to Low",
                            "Price: Low to High",
                            "5-Year Perf: High to Low",
                            "Dividend Yield: High to Low",
                            "P/B Ratio: Low to High",
                            "P/B Ratio: High to Low",
                            "Company Name (A-Z)"
                        ],
                        value="Enterprise Value (High to Low)",
                        label="Sort Table By",
                        scale=4
                    )
                    min_yield_filter = gr.Slider(
                        minimum=0,
                        maximum=25,
                        value=0,
                        step=0.5,
                        label="Min Dividend Yield (%)",
                        scale=3
                    )

                data_table = gr.Dataframe(
                    headers=[
                        "Company", "Price (₦)", "Div Yield (%)", "5Y Return (%)",
                        "P/B Ratio", "Enterprise Value (₦)", "Annual Revenue (₦)",
                        "Basic EPS (₦)", "EBITDA (₦)", "Avg 30D Vol", "EV / EBITDA"
                    ],
                    datatype=["str"] + ["number"] * 10,
                    interactive=False,
                    wrap=True
                )

            # Tab 2: Visualizations
            with gr.Tab("📊 Performance Charts"):
                with gr.Row():
                    angle_choice = gr.Radio(
                        choices=["Slanted (45°)", "Vertical (90°)"],
                        value="Slanted (45°)",
                        label="📐 Company Name Label Orientation",
                        interactive=True
                    )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🚀 Top 10 Stocks by 5-Year Return (%)")
                        chart_5y = gr.BarPlot(
                            x="name",
                            y="Perf.5Y",
                            title="5-Year Performance (%)",
                            tooltip=["name", "Perf.5Y"],
                            x_title="Company",
                            y_title="Return (%)",
                            x_label_angle=45,
                            height=420
                        )
                    with gr.Column():
                        gr.Markdown("### 💵 Top 10 Dividend Yields (%)")
                        chart_div = gr.BarPlot(
                            x="name",
                            y="dividends_yield_current",
                            title="Dividend Yield (%)",
                            tooltip=["name", "dividends_yield_current"],
                            x_title="Company",
                            y_title="Dividend Yield (%)",
                            x_label_angle=45,
                            height=420
                        )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🏛️ Top 10 by Enterprise Value (₦ Trillions)")
                        chart_ev = gr.BarPlot(
                            x="name",
                            y="EV_Trillion",
                            title="Enterprise Value (₦ Trillions)",
                            tooltip=["name", "EV_Trillion"],
                            x_title="Company",
                            y_title="₦ Trillions",
                            x_label_angle=45,
                            height=420
                        )

            # Tab 3: Data Export & Google Sheets info
            with gr.Tab("💾 Export & Google Sheets"):
                gr.Markdown("### 📊 Google Sheets Live Synchronization")
                gr.Markdown(
                    f"""
                    - **Spreadsheet Title:** `Stocks Primed for Breakout`
                    - **Active Worksheet:** `NGX Database` (Tab 1)
                    - **Direct Link:** [Open in Google Sheets]({GSHEET_URL})
                    - Whenever you click **Refresh Data Now** (with 'Sync Google Sheet on Refresh' checked), the spreadsheet updates automatically.
                    """
                )
                gr.Markdown("---")
                gr.Markdown("### 📥 Download Offline Excel File")
                gr.Markdown(
                    "You can also download the latest refreshed dataset as a Microsoft Excel workbook (.xlsx):"
                )
                excel_download = gr.File(label="Download Excel Spreadsheet (.xlsx)")

    # ----------------- Wiring Event Triggers -----------------
    outputs_list = [
        status_display,
        kpi_total,
        kpi_top_5y,
        kpi_top_div,
        kpi_top_ev,
        data_table,
        chart_5y,
        chart_div,
        chart_ev,
        excel_download
    ]

    # 1. Manual Refresh Button Click
    refresh_button.click(
        fn=refresh_pipeline,
        inputs=[search_input, sort_selector, min_yield_filter, gsheet_checkbox],
        outputs=outputs_list
    )

    # 2. Page Load trigger (Runs refresh automatically when user visits or browser-refreshes the page)
    demo.load(
        fn=refresh_pipeline,
        inputs=[search_input, sort_selector, min_yield_filter, gsheet_checkbox],
        outputs=outputs_list
    )

    # 3. Dedicated Push to Google Sheet Button
    gsheet_push_button.click(
        fn=manual_gsheet_push,
        inputs=[],
        outputs=[status_display]
    )

    # 4. Instant Filter / Search / Sort changes
    search_input.change(
        fn=filter_pipeline,
        inputs=[search_input, sort_selector, min_yield_filter],
        outputs=[data_table]
    )
    sort_selector.change(
        fn=filter_pipeline,
        inputs=[search_input, sort_selector, min_yield_filter],
        outputs=[data_table]
    )
    min_yield_filter.change(
        fn=filter_pipeline,
        inputs=[search_input, sort_selector, min_yield_filter],
        outputs=[data_table]
    )

    # 5. Label Angle Switcher (45° Slanted vs 90° Vertical)
    def on_angle_change(angle_str):
        deg = 90 if "90" in str(angle_str) else 45
        return (
            gr.BarPlot(x_label_angle=deg),
            gr.BarPlot(x_label_angle=deg),
            gr.BarPlot(x_label_angle=deg)
        )

    angle_choice.change(
        fn=on_angle_change,
        inputs=[angle_choice],
        outputs=[chart_5y, chart_div, chart_ev]
    )


if __name__ == "__main__":
    share_flag = os.environ.get("GRADIO_SHARE", "false").lower() in ["true", "1"] or "--share" in sys.argv
    server_port = int(os.environ.get("PORT", 7860))
    server_name = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    
    print(f"Starting server on {server_name}:{server_port} (share={share_flag})...", flush=True)
    launch_res = demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share_flag,
        theme=gr.themes.Soft(),
        css=custom_css
    )
    if isinstance(launch_res, tuple) and len(launch_res) >= 3:
        _, local_url, share_url = launch_res[:3]
        print(f"Local URL: {local_url}", flush=True)
        if share_url:
            print(f"PUBLIC_URL: {share_url}", flush=True)

import gradio as gr
from tradingview_screener import Query, col
import pandas as pd
import pygsheets
import os

INITIAL_GSHEET_COLUMNS = [
    'name', 'close', 'basic_eps_net_income', 'ebitda_fy', 'Perf.5Y',
    'last_annual_revenue', 'price_book_ratio', 'enterprise_value_current',
    'average_volume_30d_calc', 'enterprise_value_ebitda_ttm', 'dividends_yield_current'
]
MODEL_PREDICTION_COLUMNS = {'Ensemble_Prob', 'Signal'}

# Define the function to fetch and process data
def get_ngx_data():
    # Build the query
    query = (Query()
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

    # Fetch stock data from NGX
    data = query.get_scanner_data()

    # Extract the actual DataFrame
    if isinstance(data, tuple) and len(data) == 2:
        stock_df2 = data[1]  # Extract DataFrame
        if not stock_df2.empty:
            stock_df2 = stock_df2.drop(columns=[col for col in ['ticker'] if col in stock_df2.columns], errors='ignore')
            stock_df2 = stock_df2.drop(columns=[col for col in stock_df2.columns if col in MODEL_PREDICTION_COLUMNS or col.startswith('Prob_')], errors='ignore')
            stock_df2 = stock_df2.reindex(columns=[col for col in INITIAL_GSHEET_COLUMNS if col in stock_df2.columns])

            # Authorize with service account and push to Google Sheets
            service_account_file_path = "wqu-satellite-image-learning1-175e0a0906aa.json"

            if os.path.exists(service_account_file_path):
                gc = pygsheets.authorize(service_account_file=service_account_file_path)
                sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1jkbrrHnR48JztVz0SrqZELKjOenVm44wsiIchLW0K9g/edit#gid=1868316212")
                wks = sh[0]
                wks.set_dataframe(stock_df2, (1, 1))
                return stock_df2, "✅ NGX Stock Data Retrieved and Updated Successfully!"
            else:
                return pd.DataFrame(), f"❌ Service account file not found at: {service_account_file_path}"
        else:
            return pd.DataFrame(), "❌ No data found for NGX stocks."
    else:
        return pd.DataFrame(), "❌ Unexpected data format received from TradingView."

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# NGX Stock Data Dashboard")
    gr.Markdown("Click the 'Refresh Data' button to fetch the latest stock data from TradingView and update the Google Sheet.")

    status_output = gr.Textbox(label="Status", interactive=False)
    output_dataframe = gr.DataFrame(headers="first", interactive=False)
    refresh_button = gr.Button("Refresh Data")

    refresh_button.click(
        fn=get_ngx_data,
        inputs=[],
        outputs=[output_dataframe, status_output]
    )

    # Initial data load
    demo.load(
        fn=get_ngx_data,
        inputs=[],
        outputs=[output_dataframe, status_output]
    )

if __name__ == "__main__":
    demo.launch()

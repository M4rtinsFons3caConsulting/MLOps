"""
This is a pipeline 'data_ingestion'
generated using Kedro 0.19.14
"""

import yfinance as yf

def collect_data(
    symbols: list
    ,start_date: str
):
    data = {}
    
    for symbol in symbols:
        symbol_df = yf.download(symbol, start=start_date)

        # Standardize column names
        if len(symbol_df.columns) == 6:
            symbol_df.columns = ['close', 'high', 'low', 'open', 'adj_close', 'volume']
        else:
            symbol_df.columns = ['close', 'high', 'low', 'open', 'volume']

        symbol_df.index.name = 'date'

        data[f"{symbol}.csv"] = symbol_df

    return data
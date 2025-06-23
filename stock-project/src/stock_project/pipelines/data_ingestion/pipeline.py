"""
Pipeline '01_data_ingestion' generated using Kedro 0.19.14.

This pipeline performs the following:
- Downloads historical OHLCV data from yfinance for a given list of symbols.
- Computes the effective start date based on user input and the last ingestion checkpoint.
- Ingests data from the effective start to the latest available date.
- Returns a long-format DataFrame with raw market data.
- Updates the last_ingestion_date for future incremental ingestion.

Inputs:
- params:symbols: List of ticker symbols to download.
- params:user_start_date: User-defined earliest ingestion date (YYYY-MM-DD).
- last_ingestion_date: Date of the last successful ingestion, tracked persistently.

Outputs:
- raw_data: Ingested OHLCV data in long format with columns [date, ticker, open, high, low, close, volume].
- last_ingestion_date: Latest available date from the new data, to be reused as ingestion checkpoint.
"""

from kedro.pipeline import node, Pipeline, pipeline
from .nodes import collect_yf_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=collect_yf_data
            ,inputs=dict(
                symbols="params:symbols"
                , user_start_date="params:user_start_date"
                , is_to_feature_store="params:is_to_feature_store"
            )
            ,outputs=[
                "raw_data"
                ,"last_ingestion_date"
                ,"ingestion_interation_count"
            ]
            ,name="collect_data_node"
        )
    ])

"""
Pipeline 'data_ingestion' for data ingestion and wrangling.

This pipeline performs the following steps:
- Downloads historical OHLCV data from yfinance for specified ticker symbols.
- Computes the effective start date based on user input and the last ingestion checkpoint.
- Ingests data incrementally from the effective start date to the latest available date.
- Returns a long-format DataFrame with raw market data.
- Updates the last_ingestion_date for future incremental ingestion.
- Optionally stores data to a feature store and tracks feature store versions.

Inputs:
- params:symbols (List[str]): List of ticker symbols to download.
- params:user_start_date (str): User-defined earliest ingestion date (YYYY-MM-DD).
- params:is_to_feature_store (bool): Flag to enable feature store upload.
- last_ingestion_date (str, optional): Date of last successful ingestion (tracked persistently).

Outputs:
- raw_data (pd.DataFrame): Ingested OHLCV data in long format with columns [date, symbol, open, high, low, close, volume].
- last_ingestion_date (str): Latest available date from the new data, for incremental ingestion checkpoint.
- feature_store_versions (dict): Current versions of feature groups in the feature store.
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
                ,"feature_store_versions"
            ]
            ,name="collect_data_node"
        )
    ])

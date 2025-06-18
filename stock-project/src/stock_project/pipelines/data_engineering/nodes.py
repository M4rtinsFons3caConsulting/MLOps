"""
This is a pipeline 'data_preprocessing'
generated using Kedro 0.19.14
"""

import time
import pandas as pd

import mlflow

from .indicators import apply_indicators_to_group


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    with mlflow.start_run(run_name="add_technical_indicators", nested=True):
        start_time = time.time()
        
        unique_symbols = data["symbol"].nunique()
        mlflow.log_param("unique_symbols", unique_symbols)
        mlflow.log_param("input_shape", data.shape)

        result = data.groupby("symbol", group_keys=False).apply(apply_indicators_to_group)

        mlflow.log_param("output_shape", result.shape)
        mlflow.log_metric("processing_time_seconds", time.time() - start_time)

    return result

"""
This is a pipeline 'data_wrangling'
generated using Kedro 0.19.14
"""

import logging
from pathlib import Path
import pandas as pd
import mlflow

logger = logging.getLogger(__name__)

def combine_files(
    raw_data_dir: str
) -> pd.DataFrame:
    raw_path = Path(raw_data_dir)
    csv_files = list(raw_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

    dataframes = []
    filenames = []

    for file in csv_files:
        df = pd.read_csv(file)
        df["Symbol"] = file.stem
        dataframes.append(df)
        filenames.append(file.name)

    merged_df = pd.concat(dataframes, ignore_index=True)

    # Log to kedro-mlflow
    mlflow.log_param("num_input_files", len(filenames))
    mlflow.log_param("input_file_names", filenames)
    mlflow.log_metric("combined_rows", merged_df.shape[0])
    mlflow.log_metric("combined_cols", merged_df.shape[1])

    # Save the data
    preview_path = "data/02_intermediate/stock.csv"
    merged_df.to_csv(preview_path, index=False)
    mlflow.log_artifact(preview_path, artifact_path="previews")

    logger.info(f"Merged {len(filenames)} files from '{raw_data_dir}' into shape {merged_df.shape}")

    return merged_df


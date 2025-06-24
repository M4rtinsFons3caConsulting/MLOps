"""
This is a pipeline 'data_preprocessing'
generated using Kedro 0.19.14
"""

# STL
import time
from pathlib import Path
import logging
import os

# Preprocessing
import pandas as pd
import numpy as np
np.NaN = np.nan
import pandas_ta as ta

# Pipeline
from kedro.framework.session import KedroSession
from kedro.io.core import DatasetError
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

# Logging
import mlflow

# Feature Store + Expectations
from stock_project.pipelines.data_ingestion.nodes import to_feature_store, build_expectation_suite
from great_expectations.data_context import DataContext

# Configs
conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]

# Logger
logger = logging.getLogger(__name__)


def apply_indicators_to_group(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply a comprehensive set of technical analysis indicators to the input OHLCV DataFrame.
    
    Each indicator is calculated with multiple parameter scalings (p) to capture short, medium,
    and long-term trends. The indicators cover trend, momentum, volatility, volume, and cycle analysis.
    
    Indicators included:
    - EMA (Exponential Moving Average): Weighted moving average emphasizing recent prices.
    - HMA (Hull Moving Average): Smoothed moving average reducing lag.
    - MACD (Moving Average Convergence Divergence): Momentum trend indicator showing relationship between two EMAs.
    - RSI (Relative Strength Index): Momentum oscillator identifying overbought/oversold conditions.
    - Stoch (Stochastic Oscillator): Momentum indicator comparing closing price to price range.
    - Supertrend: Trend following indicator based on ATR volatility.
    - MOM (Momentum): Measures rate of change in price.
    - ROC (Rate of Change): Percentage change over a period.
    - WILLIAMS %R: Momentum indicator showing overbought/oversold levels.
    - AO (Awesome Oscillator): Momentum indicator comparing 5 and 34 period moving averages.
    - KAMA (Kaufman Adaptive Moving Average): Moving average adapting to volatility.
    - BBANDS (Bollinger Bands): Volatility bands around a moving average.
    - ATR (Average True Range): Measures volatility.
    - Donchian Channels: Highest high and lowest low over a period, used for breakout signals.
    - CMF (Chaikin Money Flow): Volume-weighted average of accumulation/distribution over a period.
    - MFI (Money Flow Index): Volume-based RSI measuring buying and selling pressure.
    - Z-Score: Standard score showing how far data is from mean.
    - ADOSC (Chaikin A/D Oscillator): Momentum indicator using accumulation/distribution line.
    - UO (Ultimate Oscillator): Combines multiple time frame momentum oscillators.
    - OBV (On Balance Volume): Measures buying and selling pressure as cumulative volume.
    - Squeeze: Identifies periods of low volatility likely to precede breakouts.
    - VWAP (Volume Weighted Average Price): Average price weighted by volume, important for intraday analysis.
    
    Args:
        data (pd.DataFrame): DataFrame containing OHLCV data with columns ['open', 'high', 'low', 'close', 'volume'] and a 'date' column.
        
    Returns:
        pd.DataFrame: Original DataFrame with all technical indicators appended as new columns. Index set to 'date'.
    """
    logger.info(f"Creating new features for ticker {data['symbol'].iloc[0]}...")    

    data = data.sort_values("date").copy()
    data.set_index('date', inplace=True)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Missing required OHLCV columns in: {data.columns}")
    
    for p in [1, 3, 6]:
        data.ta.ema(length=10*p, append=True)        # EMA - faster than SMA, better for recent weekly shifts
        data.ta.hma(length=10*p, append=True)        # HMA - smooths without much lag
        data.ta.macd(fast=6*p, slow=13*p, signal=4*p, append=True)  # MACD - momentum and trend crossover
        data.ta.rsi(length=10*p, append=True)        # RSI - tuned for weekly sentiment
        data.ta.stoch(k=10*p, d=3*p, append=True)    # Stoch - short-term momentum
        data.ta.supertrend(length=3*p, multiplier=1.5*p, append=True)  # Supertrend - tuned for shorter trend cycles
        data.ta.mom(length=10*p, append=True)        # Momentum
        data.ta.roc(length=10*p, append=True)        # Rate of change - volatility/strength
        data.ta.willr(length=10*p, append=True)      # Williams %R - overbought/oversold weekly
        data.ta.ao(fast=3*p, slow=10*p, append=True) # AO - short vs long trend shift
        data.ta.kama(length=10*p, append=True)       # Adaptive to market noise
        data.ta.bbands(length=10*p, std=2, append=True)  # Bollinger Bands
        data.ta.atr(length=10*p, append=True)        # ATR - weekly volatility
        data.ta.donchian(lower_length=10*p, upper_length=10*p, append=True)  # Breakout strategy
        data.ta.cmf(length=10*p, append=True)        # Volume flow
        data.ta.mfi(length=10*p, append=True)        # Money Flow Index
        data.ta.adosc(fast=2*p, slow=4*p, append=True)  # A/D Oscillator
        data.ta.zscore(length=10*p, append=True)     # Detect outlier behavior weekly

    for p1, p2 in zip([3, 5, 10], [9, 17, 34]):
        data.ta.adosc(fast=p1, slow=p2, append=True)  # Chaikin Accumulation/Distribution Oscillator

    data.ta.uo(append=True)      # Ultimate Oscillator
    data.ta.obv(append=True)     # On Balance Volume
    data.ta.squeeze(append=True) # Squeeze Momentum Indicator

    data.ta.vwap(append=True)    # Volume Weighted Average Price

    logger.info(f"Finished creating new features for ticker {data['symbol'].iloc[0]}.")
    
    return data

# TODO: Review repeated 'adosc' with different arguments (l. 105, 109)

def perform_feature_engineering(
    data: pd.DataFrame
    ,is_to_feature_store: bool = False
) -> pd.DataFrame:
    data = data.copy()
    project_path = Path(__file__).resolve().parents[4]

    # Inject previous context
    with KedroSession.create(project_path=project_path) as session:
        catalog = session.load_context().catalog
                
        # Try to load the feature store versioning
        try:
            versions = catalog.load("feature_store_versions")
        except DatasetError:
            versions = {}

    with mlflow.start_run(run_name="add_technical_indicators", nested=True):
        start_time = time.time()
        
        unique_symbols = data["symbol"].nunique()
        mlflow.log_param("unique_symbols", unique_symbols)
        mlflow.log_param("input_shape", data.shape)

        # Perform feature engineering
        result = data.groupby("symbol", group_keys=False).apply(apply_indicators_to_group)

        mlflow.log_param("output_shape", result.shape)
        mlflow.log_metric("processing_time_seconds", time.time() - start_time)

        if is_to_feature_store:
            logger.info("Retrieving feature store versions...")

            # Initialize versions if missing
            if "numerical_features" not in versions:
                versions["numerical_features"] = 1
                
            logger.info("Feature store versions retrieved.")

            # Setting the feature store
            categorical_features = result.select_dtypes(include=['object','string','category']).columns.tolist()
            data_numeric = result.drop(
                columns=categorical_features
            )
            data_numeric.reset_index(drop=False, inplace=True)

            # Replace invalid hopsworks characters
            data_numeric.columns = [col.replace('.', '_') for col in data_numeric.columns]

            # Initialize GE context (adjust path as needed)
            context_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../gx"))
            context = DataContext(context_root_dir=context_root_dir)

            # Load the expectation suite
            validation_expectation_suite_numerical = context.get_expectation_suite(
                expectation_suite_name="numerical_expectations"
            )

            mlflow.log_artifacts(context_root_dir, artifact_path="great_expectations")

            numerical_feature_descriptions = []
            
            logger.info("Uploading numerical features to feature store...")
            # Update feature store
            object_fs_numerical_features = to_feature_store(
                data_numeric
                ,"numerical_features"
                ,versions["numerical_features"]
                ,"Numerical Features"
                ,numerical_feature_descriptions
                ,validation_expectation_suite_numerical
                ,credentials["feature_store"]
            )
            # Update feature store version
            versions["numerical_features"] += 1

            logger.info("Numerical features upload complete.")

    return result, versions


def create_target(
    data: pd.DataFrame,
    prediction_horizon: int = 5,
    threshold: float = 0.01,
    is_to_feature_store: bool = False
) -> pd.DataFrame:
    """
    Generate binary momentum labels for QQQ based on future return.

    Args:
        data: DataFrame with columns ['date', 'symbol', 'close', ...]
        prediction_horizon: Number of days ahead to calculate the return.
        threshold: Return threshold to classify as label 1 (positive momentum).

    Returns:
        DataFrame with ['date', 'label'] columns for QQQ only.
    """
    project_path = Path(__file__).resolve().parents[4]

    # Inject previous context
    with KedroSession.create(project_path=project_path) as session:
        catalog = session.load_context().catalog

        # Try to load the feature store versioning
        try:
            versions = catalog.load("feature_store_versions")
        except DatasetError:
            versions = {}

    with mlflow.start_run(run_name="create_target", nested=True):
        mlflow.log_param("prediction_horizon", prediction_horizon)
        mlflow.log_param("threshold", threshold)

        logger.info(f"Creating target with prediction_horizon={prediction_horizon}, threshold={threshold}")

        # Filter only QQQ data
        qqq = data[data['symbol'] == 'QQQ'].sort_values(by='date').copy()

        # Calculate future return
        qqq['future_return'] = qqq['close'].shift(-prediction_horizon) / qqq['close'] - 1

        # Generate binary label
        qqq['label'] = (qqq['future_return'] > threshold).astype(int)

        # Shift label back to align with current date features
        qqq['label'] = qqq['label'].shift(prediction_horizon)

        # Drop rows with missing label (last <prediction_horizon> rows)
        qqq = qqq.dropna(subset=['label'])


        # Save the index and labels
        data_label = qqq[['date', 'label']]

        logger.info(f"Created target labels DataFrame with {len(data_label)} rows")

        # Log metrics for insight
        label_distribution = data_label['label'].value_counts(normalize=True).to_dict()
        mlflow.log_metric("positive_label_ratio", label_distribution.get(1, 0))
        mlflow.log_metric("negative_label_ratio", label_distribution.get(0, 0))

        # Create expectation suite
        validation_expectation_suite_label = build_expectation_suite(
            df=data_label
            ,suite_name="label_expectations"
            ,datasource_name="target_feature"
            ,data_asset_name="target_feature_asset"
        )

        context_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../gx"))
        mlflow.log_artifacts(context_root_dir, artifact_path="great_expectations")

        if is_to_feature_store:
            logger.info("Retrieving feature store versions...")

            # Initialize versions if missing
            if "target_feature" not in versions:
                versions["target_feature"] = 1
                
            logger.info("Feature store versions retrieved.")

            target_feature = ['label']
            target_feature_description = []
            logger.info("Uploading target feature to feature store...")

            object_fs_target_feature = to_feature_store(
                data_label
                ,"target_feature"
                ,versions["target_feature"]
                ,"Target Feature"
                ,target_feature_description
                ,validation_expectation_suite_label
                ,credentials["feature_store"]
            )
            # Update feature store version
            versions["target_feature"] += 1

            logger.info("Target feature upload complete.")

    return data_label, versions


def widden_df(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert a long-format DataFrame of time series data into a wide format.

    The input DataFrame is expected to have columns including 'date' and 'symbol',
    along with other feature columns. This function pivots the DataFrame so that each
    row corresponds to a unique date, and each column represents a feature for a specific symbol,
    named in the format "{ticker}_{feature}".

    Args:
        data (pd.DataFrame): Long-format DataFrame containing columns 'date', 'symbol',
                             and one or more feature columns.

    Returns:
        pd.DataFrame: Wide-format DataFrame indexed by 'date' with columns named
                      as "{symbol}_{feature}".
    """
    logger.info(f"Starting to pivot data with shape: {data.shape}")

    with mlflow.start_run(run_name="widen_dataframe", nested=True):
        # Pivot to wide format: each row is a date, each column is feature_ticker
        data_wide = data.reset_index().pivot(index='date', columns='symbol')

        logger.info(f"Pivoted data shape (before flattening columns): {data_wide.shape}")
        mlflow.log_metric("pivoted_rows", data_wide.shape[0])
        mlflow.log_metric("pivoted_columns", data_wide.shape[1])

        # Flatten MultiIndex columns
        data_wide.columns = [f"{ticker}_{feature}" for feature, ticker in data_wide.columns]

        # Reset index to make 'date' a column again
        data_wide.reset_index(inplace=True)

        logger.info(f"Finished widening data, resulting shape: {data_wide.shape}")
        mlflow.log_metric("final_rows", data_wide.shape[0])
        mlflow.log_metric("final_columns", data_wide.shape[1])

    return data_wide


def handle_missing_values(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Cleans the input DataFrame by handling missing values and removing unwanted columns.

    Steps:
    1. Drops all columns containing 'SUPERT' in their name, except those containing 'SUPERTd'.
    2. Fills missing values in columns containing 'SUPERTd' with 0.
    3. Drops all remaining rows with missing values.

    Args:
        data (pd.DataFrame): The input DataFrame containing technical indicators.

    Returns:
        pd.DataFrame: Cleaned DataFrame with reduced missing values and removed columns.
    """
    # Drop all columns starting with 'SUPERT' but keep those starting with 'SUPERTd'
    cols_to_drop = [col for col in data.columns if 'SUPERT' in col and 'SUPERTd' not in col]
    data.drop(columns=cols_to_drop, inplace=True)

    # Fill NaNs in SUPERTd columns with 0
    supertd_cols = [col for col in data.columns if 'SUPERTd' in col]
    data[supertd_cols] = data[supertd_cols].fillna(0)

    # All remaining missing values are at the beginning of the time series, so those are dropped
    data.dropna(inplace=True)

    return data


def prepare_model_input(
    data: pd.DataFrame
    ,is_to_feature_store: bool = False
) -> pd.DataFrame:
    """
    Join engineered features with QQQ binary labels on date.

    Args:
        data: DataFrame with date and multi-ticker features.
        labels: DataFrame with ['date', 'label'] for QQQ.

    Returns:
        DataFrame.
    """
    raw_data = data.copy()

    engineered_data, versions_engineering = perform_feature_engineering(
        data=raw_data
        ,is_to_feature_store=is_to_feature_store
    )
    data_labels, versions_target = create_target(
        data=raw_data
        ,is_to_feature_store=is_to_feature_store
    )
    data_wide = widden_df(engineered_data)
    data_final = handle_missing_values(data_wide)

    # Get final feature stores versions
    versions = {
        key: max(versions_engineering.get(key, 0), versions_target.get(key, 0))
        for key in set(versions_engineering) | set(versions_target)
    }

    logger.info("Starting to prepare model input by merging features and labels.")

    with mlflow.start_run(run_name="prepare_model_input", nested=True):
        # Merge features and label on date
        merged = pd.merge(data_final, data_labels, on='date', how='inner')
        logger.info(f"Merged data shape: {merged.shape}")

        # Drop rows with missing label
        model_data = merged.dropna(subset=['label']).reset_index(drop=True)
        logger.info(f"Data shape after dropping missing labels: {model_data.shape}")

    logger.info("Finished preparing model input.")
    
    return model_data, versions

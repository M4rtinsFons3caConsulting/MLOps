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
    - SMA (Simple Moving Average): Average price over specified periods to identify trend direction.
    - EMA (Exponential Moving Average): Weighted moving average emphasizing recent prices.
    - WMA (Weighted Moving Average): Moving average with linearly decreasing weights.
    - HMA (Hull Moving Average): Smoothed moving average reducing lag.
    - VWMA (Volume Weighted Moving Average): Moving average weighted by volume.
    - MACD (Moving Average Convergence Divergence): Momentum trend indicator showing relationship between two EMAs.
    - ADX (Average Directional Index): Measures trend strength regardless of direction.
    - CCI (Commodity Channel Index): Identifies cyclical trends and overbought/oversold levels.
    - RSI (Relative Strength Index): Momentum oscillator identifying overbought/oversold conditions.
    - Stoch (Stochastic Oscillator): Momentum indicator comparing closing price to price range.
    - Ichimoku: Comprehensive indicator showing support, resistance, and trend direction.
    - Supertrend: Trend following indicator based on ATR volatility.
    - PSAR (Parabolic SAR): Indicates trend direction and potential reversal points.
    - MOM (Momentum): Measures rate of change in price.
    - ROC (Rate of Change): Percentage change over a period.
    - WILLIAMS %R: Momentum indicator showing overbought/oversold levels.
    - AO (Awesome Oscillator): Momentum indicator comparing 5 and 34 period moving averages.
    - KAMA (Kaufman Adaptive Moving Average): Moving average adapting to volatility.
    - CG (Center of Gravity): Indicator showing cyclical price behavior.
    - BBANDS (Bollinger Bands): Volatility bands around a moving average.
    - ATR (Average True Range): Measures volatility.
    - KC (Keltner Channels): Volatility based envelopes around an EMA.
    - Donchian Channels: Highest high and lowest low over a period, used for breakout signals.
    - RVI (Relative Vigor Index): Measures strength of a trend by comparing closing and opening prices.
    - CMF (Chaikin Money Flow): Volume-weighted average of accumulation/distribution over a period.
    - MFI (Money Flow Index): Volume-based RSI measuring buying and selling pressure.
    - EOM (Ease of Movement): Combines volume and price change to gauge ease of price movement.
    - NVI (Negative Volume Index): Emphasizes price changes on days with lower volume.
    - Fisher Transform: Converts price into a Gaussian normal distribution for turning points.
    - Decay: Applies decay functions (linear and exponential) to smooth the data.
    - Vortex: Identifies start of a new trend or continuation.
    - Z-Score: Standard score showing how far data is from mean.
    - Entropy: Measures disorder or randomness in the data.
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
    data.index = range(len(data))
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Missing required OHLCV columns in: {data.columns}")
    
    for p in [1, 2, 4]:
        data.ta.sma(length=50*p, append=True)      # Simple Moving Average
        data.ta.ema(length=25*p, append=True)      # Exponential Moving Average
        data.ta.wma(length=25*p, append=True)      # Weighted Moving Average
        data.ta.hma(length=25*p, append=True)      # Hull Moving Average
        data.ta.vwma(length=25*p, append=True)     # Volume Weighted Moving Average
        data.ta.macd(fast=12*p, slow=26*p, signal=9*p, append=True)  # MACD
        data.ta.adx(length=14*p, append=True)      # Average Directional Index
        data.ta.cci(length=10*p, append=True)      # Commodity Channel Index
        data.ta.rsi(length=14*p, append=True)      # Relative Strength Index
        data.ta.stoch(k=14*p, d=3*p, append=True)  # Stochastic Oscillator
        data.ta.ichimoku(tenkan=9*p, kijun=26*p, senkou=52*p, append=True)  # Ichimoku Cloud
        data.ta.supertrend(length=7 * p, multiplier=2 * p, append=True)    # Supertrend
        data.ta.psar(step=0.02*p, max_step=0.2*p, append=True)             # Parabolic SAR
        data.ta.mom(length=25*p, append=True)      # Momentum
        data.ta.roc(length=15*p, append=True)      # Rate of Change
        data.ta.willr(length=7*p, append=True)     # Williams %R
        data.ta.ao(fast=3*p, slow=17*p, append=True)  # Awesome Oscillator
        data.ta.kama(length=5*p, append=True)      # Kaufman Adaptive Moving Average
        data.ta.cg(length=5*p, append=True)        # Center of Gravity
        data.ta.bbands(length=20*p, std=1*p, append=True)  # Bollinger Bands
        data.ta.atr(length=7*p, append=True)       # Average True Range
        data.ta.kc(length=20*p, scalar=1.5*p, append=True)  # Keltner Channels
        data.ta.donchian(lower_length=20*p, upper_length=20*p, append=True)  # Donchian Channels
        data.ta.rvi(length=7*p, append=True)       # Relative Vigor Index
        data.ta.cmf(length=15*p, append=True)      # Chaikin Money Flow
        data.ta.mfi(length=14*p, append=True)      # Money Flow Index
        data.ta.eom(length=14*p, append=True)      # Ease of Movement
        data.ta.nvi(length=128*p, append=True)     # Negative Volume Index
        data.ta.fisher(length=9*p, append=True)    # Fisher Transform
        data.ta.decay(length=5*p, mode="linear", append=True)      # Decay (Linear)
        data.ta.decay(length=5*p, mode="exponential", append=True) # Decay (Exponential)
        data.ta.vortex(length=14*p, append=True)   # Vortex Indicator
        data.ta.zscore(length=20*p, append=True)   # Z-Score
        data.ta.entropy(length=10*p, append=True)  # Entropy

    for p1, p2 in zip([3, 5, 10], [9, 17, 34]):
        data.ta.adosc(fast=p1, slow=p2, append=True)  # Chaikin Accumulation/Distribution Oscillator

    data.ta.uo(append=True)      # Ultimate Oscillator
    data.ta.obv(append=True)     # On Balance Volume
    data.ta.squeeze(append=True) # Squeeze Momentum Indicator

    data.set_index('date', inplace=True)
    data.ta.vwap(append=True)    # Volume Weighted Average Price

    logger.info(f"Finished creating new features for ticker {data['symbol'].iloc[0]}.")
    
    return data


def perform_feature_engineering(
    data: pd.DataFrame
    ,is_to_feature_store: bool = False
) -> pd.DataFrame:
    data = data.copy()
    project_path = Path(__file__).resolve().parents[4]

    # Inject previous context
    with KedroSession.create(project_path=project_path) as session:
        catalog = session.load_context().catalog
                
        # Try to load the iteration counter
        try:
            preprocessing_interation_count = catalog.load("preprocessing_interation_count") + 1
        except DatasetError:
            preprocessing_interation_count = 1

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
            # Setting the feature store
            categorical_features = result.select_dtypes(include=['object','string','category']).columns.tolist()
            data_numeric = result.drop(
                columns=categorical_features
            )

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
                ,preprocessing_interation_count
                ,"Numerical Features"
                ,numerical_feature_descriptions
                ,validation_expectation_suite_numerical
                ,credentials["feature_store"]
            )

            object_fs_numerical_features.insert(
                features=data_numeric,
                write_options={"schema_evolution": True},
                mode="overwrite"
            )

            logger.info("Numerical features upload complete.")

    return result, preprocessing_interation_count


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
            target_feature = ['label']
            target_feature_description = []
            logger.info("Uploading target feature to feature store...")

            object_fs_target_feature = to_feature_store(
                data_label
                ,"target_feature"
                ,1
                ,"Target Feature"
                ,target_feature_description
                ,validation_expectation_suite_label
                ,credentials["feature_store"]
            )

            logger.info("Target feature upload complete.")

    return data_label


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


def prepare_model_input(
    data: pd.DataFrame
    ,labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Join engineered features with QQQ binary labels on date.

    Args:
        data: DataFrame with date and multi-ticker features.
        labels: DataFrame with ['date', 'label'] for QQQ.

    Returns:
        DataFrame.
    """
    logger.info("Starting to prepare model input by merging features and labels.")

    with mlflow.start_run(run_name="prepare_model_input", nested=True):
        # Merge features and label on date
        merged = pd.merge(data, labels, on='date', how='inner')
        logger.info(f"Merged data shape: {merged.shape}")

        # Drop rows with missing label
        model_data = merged.dropna(subset=['label']).reset_index(drop=True)
        logger.info(f"Data shape after dropping missing labels: {model_data.shape}")

    logger.info("Finished preparing model input.")
    return model_data

"""
This is a pipeline 'data_preprocessing'
generated using Kedro 0.19.14
"""

import time

import pandas as pd
import numpy as np
np.NaN = np.nan

import mlflow

import pandas_ta as ta


def apply_indicators_to_group(
    data: pd.DataFrame
) -> pd.DataFrame:
    data = data.sort_values("date").copy()
    data.index = range(len(data))
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Missing required OHLCV columns in: {data.columns}")
    
    for p in [1, 2, 4]:
        data.ta.sma(length=50*p, append=True)
        data.ta.ema(length=25*p, append=True)
        data.ta.wma(length=25*p, append=True)
        data.ta.hma(length=25*p, append=True)
        data.ta.vwma(length=25*p, append=True)
        data.ta.macd(fast=12*p, slow=26*p, signal=9*p, append=True)
        data.ta.adx(length=14*p, append=True)
        data.ta.cci(length=10*p, append=True)
        data.ta.rsi(length=14*p, append=True)
        data.ta.stoch(k=14*p, d=3*p, append=True)
        data.ta.ichimoku(tenkan=9*p, kijun=26*p, senkou=52*p, append=True)
        data.ta.supertrend(length=7 * p, multiplier=2 * p, append=True)
        data.ta.psar(step=0.02*p, max_step=0.2*p, append=True)
        data.ta.mom(length=25*p, append=True)
        data.ta.roc(length=15*p, append=True)
        data.ta.willr(length=7*p, append=True)
        data.ta.ao(fast=3*p, slow=17*p, append=True)
        data.ta.kama(length=5*p, append=True)
        data.ta.cg(length=5*p, append=True)
        data.ta.bbands(length=20*p, std=1*p, append=True)
        data.ta.atr(length=7*p, append=True)
        data.ta.kc(length=20*p, scalar=1.5*p, append=True)
        data.ta.donchian(lower_length=20*p, upper_length=20*p, append=True)
        data.ta.rvi(length=7*p, append=True)
        data.ta.cmf(length=15*p, append=True)
        data.ta.mfi(length=14*p, append=True)
        data.ta.eom(length=14*p, append=True)
        data.ta.nvi(length=128*p, append=True)
        data.ta.fisher(length=9*p, append=True)
        data.ta.decay(length=5*p, mode="linear", append=True)
        data.ta.decay(length=5*p, mode="exponential", append=True)
        data.ta.vortex(length=14*p, append=True)
        data.ta.zscore(length=20*p, append=True)
        data.ta.entropy(length=10*p, append=True)

    for p1, p2 in zip([3, 5, 10], [9, 17, 34]):
        data.ta.adosc(fast=p1, slow=p2, append=True)

    data.ta.uo(append=True)
    data.ta.obv(append=True)
    data.ta.squeeze(append=True)

    data.set_index('date', inplace=True)
    data.ta.vwap(append=True)
    
    return data


def perform_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
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


def create_target(
    data: pd.DataFrame
    ,prediction_horizon: int = 5
    ,threshold: float = 0.01
) -> pd.DataFrame:
    """
    Generate binary momentum labels for QQQ based on future return.

    Args:
        market_data: DataFrame with columns ['date', 'ticker', 'close', ...]
        prediction_horizon: Number of days ahead to calculate the return.
        threshold: Return threshold to classify as label 1 (positive momentum).

    Returns:
        DataFrame with ['date', 'label'] columns for QQQ only.
    """
    # Filter only QQQ data
    qqq = data[data['symbol'] == 'QQQ'].copy()

    # Sort
    qqq = qqq.sort_values(by='date')

    # Calculate future return
    qqq['future_return'] = qqq['close'].shift(-prediction_horizon) / qqq['close'] - 1

    # Generate binary label
    qqq['label'] = (qqq['future_return'] > threshold).astype(int)

    # Shift label back to align with current date features
    qqq['label'] = qqq['label'].shift(prediction_horizon)

    # Drop rows with missing label (last <prediction_horizon> rows)
    qqq = qqq.dropna(subset=['label'])

    # Save the index and labels
    label_df = qqq[['date', 'label']].reset_index(drop=True)

    return label_df


def widden_df(
    data: pd.DataFrame
) -> pd.DataFrame:
    # Pivot to wide format: each row is a date, each column is feature_ticker
    data_wide = data.pivot(index='date', columns='symbol')

    # Flatten MultiIndex columns
    data_wide.columns = [f"{ticker}_{feature}" for feature, ticker in data_wide.columns]

    # Reset index to make 'date' a column again
    data_wide.reset_index(inplace=True)

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
    # Ensure date columns are datetime
    data['date'] = pd.to_datetime(data['date'])
    labels['date'] = pd.to_datetime(labels['date'])

    # Merge features and label on date
    merged = pd.merge(data, labels, on='date', how='inner')

    # Drop rows with missing label
    model_data = merged.dropna(subset=['label']).reset_index(drop=True)

    return model_data

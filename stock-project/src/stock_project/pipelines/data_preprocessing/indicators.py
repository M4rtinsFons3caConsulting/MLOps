import pandas as pd
import pandas_ta as ta

def apply_indicators_to_group(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values("date").copy()
    data.columns = [c.lower() for c in data.columns]  # normalize case
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Missing required OHLCV columns in: {data.columns}")

    # Trend indicators
    for length in [50, 100, 200]:
        data.ta.sma(length=length, append=True)

    for length in [25, 50, 100]:
        data.ta.ema(length=length, append=True)
        data.ta.wma(length=length, append=True)
        data.ta.hma(length=length, append=True)
        data.ta.vwma(length=length, append=True)

    # MACD
    for f, s1, s2 in zip([])
    data.ta.macd(fast=24, slow=52, signal=18, append=True)
    data.ta.macd(fast=12, slow=26, signal=9, append=True)
    data.ta.macd(fast=48, slow=104, signal=36, append=True)

    # ADX, RSI
    for length in [14, 28, 56]:
        data.ta.adx(length=length, append=True)
        data.ta.rsi(length=length, append=True)

    # CCI
    for length in [10, 20, 40]:
        data.ta.cci(length=length, append=True)

    # Stochastic
    for k, d in zip([14, 28, 56], [3, 6, 12]):
        data.ta.stoch(k=k, d=int(k/4), append=True)

    data.ta.ichimoku(tenkan=18, kijun=52, senkou=104, append=True)
    data.ta.ichimoku(tenkan=9, kijun=26, senkou=52, append=True)
    data.ta.ichimoku(tenkan=36, kijun=104, senkou=208, append=True)



    # Momentum indicators
    for length in [25, 50, 100]:
        data.ta.mom(length=length, append=True)
        data.ta.kama(length=length, append=True)
        data.ta.cg(length=length, append=True)

    # Volatility indicators
    for length in [20, 40, 80]:
        data.ta.bbands(length=length, append=True)
        data.ta.kc(length=length, append=True)
        data.ta.atr(length=length, append=True)
        data.ta.zscore(length=length, append=True)

    # Volume indicators
    for length in [14, 28, 56]:
        data.ta.mfi(length=length, append=True)
        data.ta.cmf(length=length * 2, append=True) 

    data.ta.obv(append=True)
    data.ta.vwap(append=True)

    # Other indicators
    data.ta.vortex(length=14, append=True)
    data.ta.squeeze(append=True)

    return data

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.date
    return data.groupby("symbol", group_keys=False).apply(apply_indicators_to_group)

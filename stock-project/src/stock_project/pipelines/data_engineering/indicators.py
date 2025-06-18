import pandas as pd
import numpy as np
np.NaN = np.nan

import pandas_ta as ta



def apply_indicators_to_group(data: pd.DataFrame) -> pd.DataFrame:
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
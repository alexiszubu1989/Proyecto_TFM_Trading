import pandas as pd
from mvpfx.data import simulate_ohlcv
from mvpfx.indicators import compute_all_indicators
from mvpfx.config import get_cfg

def test_indicators_shapes():
    cfg = get_cfg()
    df = simulate_ohlcv(500, cfg["timeframe"], cfg["data"]["seed"])
    out = compute_all_indicators(df, cfg)
    for col in ["ema_fast","ema_slow","rsi","macd","macd_signal","macd_hist","atr","bb_mid","bb_upper","bb_lower"]:
        assert col in out.columns
    assert out["rsi"].notna().sum() > 0

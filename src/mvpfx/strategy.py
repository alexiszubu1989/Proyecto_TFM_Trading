from __future__ import annotations

# --- Bootstrap ---
import os, sys
if __package__ is None or __package__ == "":
    _CUR = os.path.dirname(os.path.abspath(__file__))
    _SRC = os.path.dirname(_CUR)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
# ---------------

import pandas as pd
import numpy as np
from mvpfx.config import get_cfg
from mvpfx.data import load_data
from mvpfx.indicators import compute_all_indicators

def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))

def cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))

def regime_trending(df: pd.DataFrame, threshold: float) -> pd.Series:
    return (df["ema_fast"] - df["ema_slow"]).abs() / df["close"].abs() >= threshold

def generate_signals(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = get_cfg()
    st, rk = cfg["strategy"], cfg["risk"]
    atr_pct = df["atr"] / df["close"].abs()
    filt_vol = atr_pct >= st["min_atr_pct"]
    reg = regime_trending(df, st["regime_threshold"])
    long_cross = cross_up(df["ema_fast"], df["ema_slow"])
    short_cross = cross_down(df["ema_fast"], df["ema_slow"])
    macd_ok_long = df["macd"] >= df["macd_signal"] if st["macd_confirm"] else pd.Series(True, index=df.index)
    macd_ok_short = df["macd"] <= df["macd_signal"] if st["macd_confirm"] else pd.Series(True, index=df.index)
    rsi_ok_long = df["rsi"] >= st["rsi_long_min"]
    rsi_ok_short = df["rsi"] <= st["rsi_short_max"]
    cond_long = long_cross & rsi_ok_long & macd_ok_long & filt_vol & reg
    cond_short = short_cross & rsi_ok_short & macd_ok_short & filt_vol & reg

    signal = pd.Series(0, index=df.index, dtype=int).mask(cond_long, 1).mask(cond_short, -1)
    long_score = (long_cross.astype(int)+rsi_ok_long.astype(int)+macd_ok_long.astype(int)+filt_vol.astype(int)+reg.astype(int))/5.0
    short_score = (short_cross.astype(int)+rsi_ok_short.astype(int)+macd_ok_short.astype(int)+filt_vol.astype(int)+reg.astype(int))/5.0
    score = pd.Series(0.0, index=df.index).mask(signal==1, long_score).mask(signal==-1, short_score)

    sl_long = df["close"] - cfg["risk"]["atr_sl_mult"]*df["atr"]
    tp_long = df["close"] + cfg["risk"]["atr_tp_mult"]*df["atr"]
    sl_short = df["close"] + cfg["risk"]["atr_sl_mult"]*df["atr"]
    tp_short = df["close"] - cfg["risk"]["atr_tp_mult"]*df["atr"]
    sl = pd.Series(np.nan, index=df.index).mask(signal==1, sl_long).mask(signal==-1, sl_short)
    tp = pd.Series(np.nan, index=df.index).mask(signal==1, tp_long).mask(signal==-1, tp_short)

    out = df.copy()
    out["signal"] = signal
    out["score"] = score.clip(0,1).fillna(0.0)
    out["sl"], out["tp"] = sl, tp
    return out

if __name__ == "__main__":
    from mvpfx.indicators import compute_all_indicators
    cfg = get_cfg()
    base = load_data()
    feats = compute_all_indicators(base, cfg)
    sigs = generate_signals(feats, cfg)
    print(sigs[["close","signal","score","sl","tp"]].tail(10))
    print("Conteo señales:", sigs["signal"].value_counts(dropna=False).to_dict())


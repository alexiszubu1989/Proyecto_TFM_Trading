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

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    return (100 - 100/(1+rs)).fillna(50.0)

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ef, es = ema(close, fast), ema(close, slow)
    line = ef - es
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    return true_range(h,l,c).ewm(span=period, adjust=False).mean()

def bollinger(c: pd.Series, period: int = 20, k: float = 2.0):
    mid = c.rolling(window=period, min_periods=period).mean()
    std = c.rolling(window=period, min_periods=period).std(ddof=0)
    return mid, mid+k*std, mid-k*std

def tick_volume(v: pd.Series | None) -> pd.Series:
    return (v.astype(float) if v is not None else pd.Series(1.0, index=None))

def compute_all_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = df["close"]
    ef = ema(c, cfg["indicators"]["ema_fast"])
    es = ema(c, cfg["indicators"]["ema_slow"])
    r = rsi(c, cfg["indicators"]["rsi_period"])
    m, ms, mh = macd(c, cfg["indicators"]["ema_fast"], cfg["indicators"]["ema_slow"], cfg["indicators"]["macd_signal"])
    a = atr(df["high"], df["low"], c, cfg["indicators"]["atr_period"])
    bbm, bbu, bbl = bollinger(c, cfg["indicators"]["bb_period"], cfg["indicators"]["bb_k"])
    vol = tick_volume(df.get("volume"))
    out = df.copy()
    out["ema_fast"], out["ema_slow"], out["rsi"] = ef, es, r
    out["macd"], out["macd_signal"], out["macd_hist"] = m, ms, mh
    out["atr"], out["bb_mid"], out["bb_upper"], out["bb_lower"] = a, bbm, bbu, bbl
    out["tick_volume"] = vol
    return out

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Calcular indicadores y exportar")
    p.add_argument("--out", type=str, help="Ruta CSV salida con indicadores")
    args = p.parse_args()
    cfg = get_cfg()
    df = load_data()
    feats = compute_all_indicators(df, cfg)
    print(feats[["close","ema_fast","ema_slow","rsi","macd","macd_signal","atr"]].tail(5))
    if args.out:
        feats.to_csv(args.out, index_label="timestamp")
        print(f"Guardado en {args.out}")


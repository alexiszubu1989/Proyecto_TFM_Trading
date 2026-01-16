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

# ==================== NUEVOS INDICADORES ====================

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
               k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator (%K y %D).
    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA de %K
    """
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    stoch_k = ((close - lowest_low) / (highest_high - lowest_low + 1e-10)) * 100
    stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()
    return stoch_k.fillna(50.0), stoch_d.fillna(50.0)

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, 
               period: int = 14) -> pd.Series:
    """
    Williams %R - Oscilador de momentum.
    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    Rango: -100 a 0 (sobrevendido < -80, sobrecomprado > -20)
    """
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    wr = ((highest_high - close) / (highest_high - lowest_low + 1e-10)) * -100
    return wr.fillna(-50.0)

def adx(high: pd.Series, low: pd.Series, close: pd.Series, 
        period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index (ADX) con +DI y -DI.
    ADX mide la fuerza de la tendencia (no la dirección).
    ADX > 25 = tendencia fuerte, ADX < 20 = mercado lateral
    """
    tr = true_range(high, low, close)
    
    # Movimiento direccional
    up_move = high.diff()
    down_move = -low.diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    # Suavizado
    atr_val = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / (atr_val + 1e-10))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / (atr_val + 1e-10))
    
    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx_val = dx.ewm(span=period, adjust=False).mean()
    
    return adx_val.fillna(25.0), plus_di.fillna(25.0), minus_di.fillna(25.0)

def cci(high: pd.Series, low: pd.Series, close: pd.Series, 
        period: int = 20) -> pd.Series:
    """
    Commodity Channel Index (CCI).
    CCI > 100 = sobrecomprado, CCI < -100 = sobrevendido
    """
    tp = (high + low + close) / 3  # Typical Price
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    cci_val = (tp - sma_tp) / (0.015 * mad + 1e-10)
    return cci_val.fillna(0.0)

def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Momentum simple: diferencia entre precio actual y precio de N períodos atrás.
    """
    return close - close.shift(period)

def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Rate of Change (ROC) - Tasa de cambio porcentual.
    """
    return ((close - close.shift(period)) / (close.shift(period) + 1e-10)) * 100

# ==================== FIN NUEVOS INDICADORES ====================

def compute_all_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]
    ind_cfg = cfg["indicators"]
    
    # --- Indicadores Originales ---
    ef = ema(c, ind_cfg["ema_fast"])
    es = ema(c, ind_cfg["ema_slow"])
    r = rsi(c, ind_cfg["rsi_period"])
    m, ms, mh = macd(c, ind_cfg["ema_fast"], ind_cfg["ema_slow"], ind_cfg["macd_signal"])
    a = atr(h, l, c, ind_cfg["atr_period"])
    bbm, bbu, bbl = bollinger(c, ind_cfg["bb_period"], ind_cfg["bb_k"])
    vol = tick_volume(df.get("volume"))
    
    # --- Nuevos Indicadores ---
    stoch_k, stoch_d = stochastic(h, l, c, ind_cfg.get("stoch_k", 14), ind_cfg.get("stoch_d", 3))
    wr = williams_r(h, l, c, ind_cfg.get("williams_period", 14))
    adx_val, plus_di, minus_di = adx(h, l, c, ind_cfg.get("adx_period", 14))
    cci_val = cci(h, l, c, ind_cfg.get("cci_period", 20))
    mom = momentum(c, ind_cfg.get("momentum_period", 10))
    roc_val = roc(c, ind_cfg.get("roc_period", 10))
    
    # --- Construir DataFrame de salida ---
    out = df.copy()
    
    # Indicadores originales
    out["ema_fast"], out["ema_slow"], out["rsi"] = ef, es, r
    out["macd"], out["macd_signal"], out["macd_hist"] = m, ms, mh
    out["atr"], out["bb_mid"], out["bb_upper"], out["bb_lower"] = a, bbm, bbu, bbl
    out["tick_volume"] = vol
    
    # Nuevos indicadores
    out["stoch_k"], out["stoch_d"] = stoch_k, stoch_d
    out["williams_r"] = wr
    out["adx"], out["plus_di"], out["minus_di"] = adx_val, plus_di, minus_di
    out["cci"] = cci_val
    out["momentum"] = mom
    out["roc"] = roc_val
    
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


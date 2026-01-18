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
from typing import Literal
from mvpfx.config import get_cfg

TF = Literal["M1", "M5", "M15", "H1"]

def timeframe_to_minutes(tf: str) -> int:
    return {"M1":1, "M5":5, "M15":15, "H1":60}[tf.upper()]

def fetch_yfinance(symbol: str, timeframe: TF, bars: int = 3000) -> pd.DataFrame:
    """
    Obtiene datos OHLCV usando yfinance (Yahoo Finance).
    
    Args:
        symbol: Símbolo del instrumento (ej: "EURUSD=X" para Forex, "AAPL" para acciones)
        timeframe: M1, M5, M15, H1
        bars: Número de barras a descargar
    
    Returns:
        DataFrame con índice timestamp y columnas [open, high, low, close, volume]
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance no está instalado. Ejecuta: pip install yfinance"
        )
    
    # Mapeo de timeframes a intervalos de yfinance
    interval_map = {
        "M1": "1m",
        "M5": "5m",
        "M15": "15m",
        "H1": "1h"
    }
    
    interval = interval_map.get(timeframe.upper())
    if not interval:
        raise ValueError(f"Timeframe no soportado: {timeframe}")
    
    # Calcular período requerido
    minutes = timeframe_to_minutes(timeframe)
    total_minutes = bars * minutes
    
    # yfinance tiene límites: máx 7 días para 1m, 60 días para intervalos < 1h
    if interval == "1m":
        period = min(total_minutes // (24 * 60), 7)
        period_str = f"{period}d" if period > 0 else "1d"
    elif interval in ["5m", "15m"]:
        period = min(total_minutes // (24 * 60), 60)
        period_str = f"{period}d" if period > 0 else "5d"
    else:  # 1h o mayor
        period = min(total_minutes // (24 * 60), 730)
        period_str = f"{period}d" if period > 0 else "30d"
    
    # Normalizar símbolo para Forex
    if symbol.upper().replace(".", "") in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]:
        yf_symbol = symbol.replace(".", "").upper() + "=X"
    else:
        yf_symbol = symbol
    
    # Descargar datos
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=period_str, interval=interval)
    
    if df.empty:
        raise ValueError(f"No se obtuvieron datos para {yf_symbol} con intervalo {interval}")
    
    # Normalizar columnas (yfinance usa mayúsculas)
    df.columns = df.columns.str.lower()
    df = df[["open", "high", "low", "close", "volume"]]
    
    # Normalizar índice a UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    
    # Limitar a número de barras solicitadas
    if len(df) > bars:
        df = df.tail(bars)
    
    return df

def load_data() -> pd.DataFrame:
    cfg = get_cfg()
    src = cfg["data"]["source"]
    tf = cfg["timeframe"]
    if src == "yfinance":
        symbol = cfg["symbol"]
        bars = cfg["data"].get("bars", 3000)
        df = fetch_yfinance(symbol, tf, bars)
    elif src == "ib":
        # Import lazy para evitar problemas de event loop en FastAPI
        import asyncio
        import sys
        if sys.version_info >= (3, 10):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
        from mvpfx.broker_ib import get_historical_bars
        df = get_historical_bars(symbol=cfg["symbol"], timeframe=tf)
    else:
        raise ValueError(f"data.source desconocido: {src}. Usa 'yfinance' o 'ib'")
    df = df[["open","high","low","close"]].join(df.get("volume"))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Preview/Export OHLCV")
    p.add_argument("--source", choices=["ib","yfinance"], help="Override data.source")
    p.add_argument("--bars", type=int, help="Override bars for yfinance")
    p.add_argument("--out", type=str, help="Ruta CSV de salida")
    args = p.parse_args()
    cfg = get_cfg()
    if args.source: cfg["data"]["source"] = args.source
    if args.bars: cfg["data"]["bars"] = args.bars
    df = load_data()
    print(df.head())
    print(df.tail(3))
    if args.out:
        df_out = df.copy()
        df_out.to_csv(args.out, index_label="timestamp")
        print(f"Guardado en {args.out}")


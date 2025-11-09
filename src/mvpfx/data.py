from __future__ import annotations

# --- Bootstrap ---
import os, sys
if __package__ is None or __package__ == "":
    _CUR = os.path.dirname(os.path.abspath(__file__))
    _SRC = os.path.dirname(_CUR)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
# ---------------

import numpy as np
import pandas as pd
from typing import Literal
from mvpfx.config import get_cfg

TF = Literal["M1", "M5", "M15", "H1"]

def timeframe_to_minutes(tf: str) -> int:
    return {"M1":1, "M5":5, "M15":15, "H1":60}[tf.upper()]

def simulate_ohlcv(bars: int, timeframe: TF, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    minutes = timeframe_to_minutes(timeframe)
    idx = pd.date_range("2024-01-01", periods=bars, freq=f"{minutes}min", tz="UTC")
    vol = rng.lognormal(mean=-5.0, sigma=0.25, size=bars)
    price = 1.08 + np.cumsum(rng.normal(0, vol))
    price = np.clip(price, 1.01, 1.20)
    close = price
    open_ = np.r_[price[0], price[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, vol/2))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, vol/2))
    vol_ticks = rng.integers(50, 500, size=bars)
    df = pd.DataFrame({"open":open_, "high":high, "low":low, "close":close, "volume":vol_ticks}, index=idx)
    return df

def load_data() -> pd.DataFrame:
    cfg = get_cfg()
    src = cfg["data"]["source"]
    tf = cfg["timeframe"]
    if src == "simulated":
        df = simulate_ohlcv(cfg["data"]["bars"], tf, cfg["data"]["seed"])
    elif src == "csv":
        path = cfg["data"]["csv_path"]
        df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
        df = df.tz_localize("UTC") if df.index.tz is None else df.tz_convert("UTC")
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
        raise ValueError(f"data.source desconocido: {src}")
    df = df[["open","high","low","close"]].join(df.get("volume"))
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Preview/Export OHLCV")
    p.add_argument("--source", choices=["simulated","csv","ib"], help="Override data.source")
    p.add_argument("--bars", type=int, help="Override bars for simulated")
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


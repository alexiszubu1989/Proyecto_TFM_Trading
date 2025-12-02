from __future__ import annotations

# --- Bootstrap ---
import os, sys
if __package__ is None or __package__ == "":
    _CUR = os.path.dirname(os.path.abspath(__file__))
    _SRC = os.path.dirname(_CUR)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
# ---------------

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from mvpfx.config import get_cfg
from mvpfx.data import load_data
from mvpfx.indicators import compute_all_indicators
from mvpfx.strategy import generate_signals
from mvpfx.risk import position_size, enforce_daily_limits

@dataclass
class BTResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict

def compute_metrics(equity: pd.Series) -> dict:
    if len(equity) == 0:
        return {"CAGR": 0.0, "Sharpe": 0.0, "Sortino": 0.0, "MaxDrawdown": 0.0, "Bars": 0}
    ret = equity.pct_change().fillna(0.0)
    ann = 252
    cagr = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    sharpe = (ret.mean() / (ret.std(ddof=0)+1e-9)) * np.sqrt(ann)
    downside = ret[ret < 0].std(ddof=0)
    sortino = (ret.mean() / (downside+1e-9)) * np.sqrt(ann)
    dd = (equity / equity.cummax() - 1.0).min()
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "Sortino": float(sortino), "MaxDrawdown": float(dd), "Bars": int(len(ret))}

def run_backtest() -> BTResult:
    cfg = get_cfg()
    df = load_data()
    df = compute_all_indicators(df, cfg)
    df = generate_signals(df, cfg).iloc[cfg["warmup_bars"]:]
    ex, rk = cfg["execution"], cfg["risk"]
    equity = rk["capital"]
    position, units, entry = 0, 0, np.nan
    records, equity_curve = [], []
    trade_log = pd.DataFrame(columns=["side","entry","exit","pnl"])

    for ts, row in df.iterrows():
        price = row["close"]
        ask = price + ex["simulate_spread"]/2 + ex["simulate_slippage"]
        bid = price - ex["simulate_spread"]/2 - ex["simulate_slippage"]

        if position != 0:
            if position == 1 and (bid <= row["sl"] or bid >= row["tp"]):
                exit_price = bid; pnl = (exit_price - entry) * units
                equity += pnl; records.append({"time":ts,"type":"exit","price":float(exit_price),"pnl":float(pnl)})
                trade_log.loc[ts] = {"side":"long","entry":entry,"exit":exit_price,"pnl":pnl}
                position, units = 0, 0
            elif position == -1 and (ask >= row["sl"] or ask <= row["tp"]):
                exit_price = ask; pnl = (entry - exit_price) * units
                equity += pnl; records.append({"time":ts,"type":"exit","price":float(exit_price),"pnl":float(pnl)})
                trade_log.loc[ts] = {"side":"short","entry":entry,"exit":exit_price,"pnl":pnl}
                position, units = 0, 0

        if position == 0 and not enforce_daily_limits(trade_log, rk["capital"]):
            sig = int(row["signal"])
            if sig == 1:
                units = position_size(equity, ask, row["atr"], cfg); entry = ask; position = 1
                records.append({"time":ts,"type":"entry_long","price":float(entry),"units":units})
            elif sig == -1:
                units = position_size(equity, bid, row["atr"], cfg); entry = bid; position = -1
                records.append({"time":ts,"type":"entry_short","price":float(entry),"units":units})

        equity_curve.append((ts, equity))

    eq = pd.Series({t:v for t,v in equity_curve})
    trades = pd.DataFrame(records)
    metrics = compute_metrics(eq)
    last_eq = float(eq.iloc[-1]) if len(eq) > 0 else rk["capital"]
    with open("backtest_report.json","w",encoding="utf-8") as f:
        json.dump({"metrics":metrics,"last_equity":last_eq}, f, indent=2)
    return BTResult(equity_curve=eq, trades=trades, metrics=metrics)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run backtest")
    p.add_argument("--print", action="store_true", help="Imprime métricas")
    args = p.parse_args()
    res = run_backtest()
    if args.print:
        print(res.metrics)
    print("OK: backtest_report.json generado.")

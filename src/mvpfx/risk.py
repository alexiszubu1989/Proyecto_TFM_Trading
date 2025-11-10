from __future__ import annotations

# --- Bootstrap ---
import os, sys
if __package__ is None or __package__ == "":
    _CUR = os.path.dirname(os.path.abspath(__file__))
    _SRC = os.path.dirname(_CUR)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
# ---------------

import math
from mvpfx.config import get_cfg

def position_size(equity: float, price: float, atr: float, cfg: dict | None = None) -> int:
    if cfg is None:
        cfg = get_cfg()
    rk = cfg["risk"]
    risk_amount = equity * rk["risk_per_trade"]
    stop_distance = max(1e-6, rk["atr_sl_mult"] * atr)
    units = math.floor(risk_amount / stop_distance)
    units = max(rk["min_position_units"], min(units, rk["max_position_units"]))
    return int(units)

def enforce_daily_limits(trade_log_df, equity0: float, cfg: dict | None = None) -> bool:
    if cfg is None:
        cfg = get_cfg()
    rk = cfg["risk"]
    if trade_log_df is None or len(trade_log_df) == 0:
        return False
    today = trade_log_df.index[-1].date()
    day = trade_log_df[trade_log_df.index.date == today]
    if len(day) >= rk["max_trades_per_day"]:
        return True
    pnl_day = (day["pnl"].sum()) / equity0
    return pnl_day <= -rk["daily_loss_limit"]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--equity", type=float, default=10000)
    p.add_argument("--price", type=float, default=1.08)
    p.add_argument("--atr", type=float, default=0.001)
    args = p.parse_args()
    print("Units:", position_size(args.equity, args.price, args.atr))


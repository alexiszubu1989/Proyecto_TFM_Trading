from __future__ import annotations

# --- Bootstrap ---
import os, sys
if __package__ is None or __package__ == "":
    _CUR = os.path.dirname(os.path.abspath(__file__))
    _SRC = os.path.dirname(_CUR)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
# ---------------

import os
import yaml
from typing import Any

_CFG: dict[str, Any] | None = None

def _project_root() -> str:
    # .../mvp-eurusd/src/mvpfx -> root = ../..
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_cfg() -> dict:
    global _CFG
    if _CFG is not None:
        return _CFG
    root = _project_root()
    cfg_path = os.path.join(root, "config.yml")
    if not os.path.exists(cfg_path):
        # Defaults mínimos si falta config.yml
        _CFG = {
            "symbol": "EURUSD",
            "timezone": "UTC",
            "timeframe": "M5",
            "warmup_bars": 200,
            "indicators": {"ema_fast": 12, "ema_slow": 26, "rsi_period": 14, "macd_signal": 9, "atr_period": 14, "bb_period": 20, "bb_k": 2.0},
            "strategy": {"rsi_long_min": 55, "rsi_short_max": 45, "macd_confirm": True, "min_atr_pct": 0.0003, "regime_threshold": 0.0001},
            "risk": {"capital": 10000.0, "risk_per_trade": 0.0075, "atr_sl_mult": 1.5, "atr_tp_mult": 2.0, "trailing_mult": 0.0,
                     "daily_loss_limit": 0.03, "max_trades_per_day": 6, "max_position_units": 100000, "min_position_units": 1000},
            "execution": {"simulate_spread": 0.00005, "simulate_slippage": 0.00002},
            "data": {"source": "simulated", "csv_path": "./data/eurusd.csv", "bars": 3000, "seed": 42},
            "api": {"host": "127.0.0.1", "port": 8000, "cors_origins": ["*"]},
            "flags": {"enable_live": False, "paper_only": True}
        }
        return _CFG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Overrides críticos por ENV
    cfg["symbol"] = os.getenv("SYMBOL", cfg.get("symbol", "EURUSD"))
    paper_env = os.getenv("PAPER", "true").lower() == "true"
    cfg.setdefault("flags", {})
    cfg["flags"]["paper_only"] = True if paper_env else cfg["flags"].get("paper_only", True)
    _CFG = cfg
    return _CFG

if __name__ == "__main__":
    import json
    cfg = get_cfg()
    print(json.dumps(cfg, indent=2))


from __future__ import annotations

import asyncio
import sys
if sys.version_info >= (3, 10):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

# --- Bootstrap ---
import os, sys
if __package__ is None or __package__ == "":
    _CUR = os.path.dirname(os.path.abspath(__file__))
    _SRC = os.path.dirname(_CUR)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
# ---------------

import os
from typing import Optional
import pandas as pd
from ib_insync import IB, Forex, MarketOrder, LimitOrder, StopOrder, util
from mvpfx.config import get_cfg

def connect_ib() -> IB:
    cfg = get_cfg()
    host = os.getenv("IB_HOST","127.0.0.1")
    port = int(os.getenv("IB_PORT","7497"))
    client_id = int(os.getenv("IB_CLIENT_ID","1001"))
    ib = IB()
    ib.connect(host, port, clientId=client_id, readonly=True, timeout=20)
    return ib

def get_symbol_contract(symbol: str):
    # IB Forex format: "EUR" base currency, "USD" quote currency
    # symbol puede venir como "EURUSD" o "EUR.USD"
    symbol = symbol.replace(".", "")  # Normalizar
    if len(symbol) == 6:  # EURUSD
        base = symbol[:3]
        quote = symbol[3:]
    else:
        raise ValueError(f"Formato de símbolo inválido: {symbol}")
    return Forex(base + quote)

def get_historical_bars(symbol: str, timeframe: str, duration: str = "2 D") -> pd.DataFrame:
    ib = connect_ib()
    c = get_symbol_contract(symbol)
    ib.qualifyContracts(c)
    barSize = {"M1":"1 min","M5":"5 mins","M15":"15 mins","H1":"1 hour"}[timeframe.upper()]
    bars = ib.reqHistoricalData(c, endDateTime="", durationStr=duration, barSizeSetting=barSize,
                                whatToShow="MIDPOINT", useRTH=False, formatDate=1)
    df = util.df(bars)
    df = df.rename(columns={"date":"timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")[["open","high","low","close","volume"]]
    ib.disconnect()
    return df

def place_order(symbol: str, side: str, qty: int, order_type: str = "MKT",
                limit_price: Optional[float]=None, stop_price: Optional[float]=None):
    cfg = get_cfg()
    if cfg["flags"]["paper_only"] or os.getenv("PAPER","true").lower()=="true":
        pass
    else:
        raise RuntimeError("Modo LIVE deshabilitado en el MVP.")
    ib = connect_ib()
    c = get_symbol_contract(symbol); ib.qualifyContracts(c)
    if order_type.upper()=="MKT":
        order = MarketOrder("BUY" if side=="long" else "SELL", qty)
    elif order_type.upper()=="LMT":
        if limit_price is None: raise ValueError("limit_price requerido")
        order = LimitOrder("BUY" if side=="long" else "SELL", qty, limit_price)
    elif order_type.upper()=="STP":
        if stop_price is None: raise ValueError("stop_price requerido")
        order = StopOrder("BUY" if side=="long" else "SELL", qty, stop_price)
    else:
        raise ValueError(f"Tipo no soportado: {order_type}")
    trade = ib.placeOrder(c, order); ib.sleep(1.0)
    status, order_id = trade.orderStatus.status, trade.order.orderId
    ib.disconnect()
    return {"orderId": order_id, "status": status}

def cancel_order(order_id: int):
    ib = connect_ib()
    trades = [t for t in ib.trades() if t.order.orderId == order_id]
    if trades: ib.cancelOrder(trades[0].order)
    ib.disconnect(); return {"orderId": order_id, "status": "Cancelled"}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="IBKR paper smoke")
    p.add_argument("--op", choices=["ping","bars","buy","sell"], default="ping")
    p.add_argument("--tf", default="M5")
    p.add_argument("--qty", type=int, default=10000)
    args = p.parse_args()
    if args.op == "ping":
        ib = connect_ib(); print("Conectado a IBKR (paper)"); ib.disconnect()
    elif args.op == "bars":
        df = get_historical_bars(get_cfg()["symbol"], args.tf, "1 D"); print(df.tail())
    elif args.op in ("buy","sell"):
        side = "long" if args.op=="buy" else "short"
        resp = place_order(get_cfg()["symbol"], side=side, qty=args.qty, order_type="MKT")
        print(resp)


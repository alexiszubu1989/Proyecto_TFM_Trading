from __future__ import annotations

# --- Bootstrap ---
import os, sys
if __package__ is None or __package__ == "":
    _CUR = os.path.dirname(os.path.abspath(__file__))
    _SRC = os.path.dirname(_CUR)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
# ---------------

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from mvpfx.config import get_cfg
from mvpfx.data import load_data
from mvpfx.indicators import compute_all_indicators
from mvpfx.strategy import generate_signals
from mvpfx.llm_stub import explain_trade

app = FastAPI(title="EURUSD MVP API", version="0.1.1")
cfg = get_cfg()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg["api"]["cors_origins"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class Signal(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    price: float
    signal: int
    score: float
    sl: float | None
    tp: float | None

class OrderRequest(BaseModel):
    side: str; qty: int; order_type: str = "MKT"; limit_price: float | None = None; stop_price: float | None = None

class OrderResponse(BaseModel):
    orderId: int | None = None; status: str

class Explanation(BaseModel):
    json: dict; text: str

@app.get("/signals", response_model=list[Signal])
def get_signals():
    df = load_data()
    df = compute_all_indicators(df, cfg)
    df = generate_signals(df, cfg).iloc[cfg["warmup_bars"]:]
    out = []
    for ts, row in df.tail(200).iterrows():
        out.append(Signal(
            timestamp=ts.isoformat(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            price=float(row["close"]),
            signal=int(row["signal"]),
            score=float(row["score"]),
            sl=float(row["sl"]) if row["signal"]!=0 else None,
            tp=float(row["tp"]) if row["signal"]!=0 else None
        ))
    return out

@app.post("/orders", response_model=OrderResponse)
def post_order(req: OrderRequest):
    return OrderResponse(orderId=None, status="SimulatedAccepted")

@app.get("/explanations", response_model=Explanation)
def get_explanations():
    data = explain_trade(
        strategy="EMA Cross + RSI + MACD", signal="long",
        indicators={"ema_fast":12,"ema_slow":26,"rsi":62,"macd":0.0005},
        risk={"risk_pct":0.0075,"sl_atr_mult":1.5,"tp_atr_mult":2.0},
        confidence=0.82
    )
    return Explanation(json=data["json"], text=data["text"])

if __name__ == "__main__":
    import uvicorn
    cfg = get_cfg()
    uvicorn.run("mvpfx.api:app", host=cfg["api"]["host"], port=cfg["api"]["port"], reload=True)


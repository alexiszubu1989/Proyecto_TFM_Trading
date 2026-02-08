from __future__ import annotations

# --- Bootstrap ---
import os, sys
if __package__ is None or __package__ == "":
    _CUR = os.path.dirname(os.path.abspath(__file__))
    _SRC = os.path.dirname(_CUR)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
# ---------------

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd
from mvpfx.config import get_cfg
from mvpfx.data import load_data
from mvpfx.indicators import compute_all_indicators
from mvpfx.strategy import generate_signals, STRATEGIES
from mvpfx.llm_stub import explain_trade, analyze_signals, clear_analysis_cache
from mvpfx.backtest import run_backtest_for_api

app = FastAPI(title="MVP Trading API", version="4.2.0")
cfg = get_cfg()

# Obtener ruta del directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DASHBOARD_DIR = BASE_DIR / "dashboard"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Servir archivos estáticos del dashboard
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

class StrategyVote(BaseModel):
    name: str
    signal: int  # 1=LONG, -1=SHORT, 0=NEUTRAL
    score: float
    vote: str  # "LONG", "SHORT", "NEUTRAL"

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
    strategy_votes: list[StrategyVote] | None = None
    long_votes: int | None = None
    short_votes: int | None = None
    neutral_votes: int | None = None

class OrderRequest(BaseModel):
    side: str; qty: int; order_type: str = "MKT"; limit_price: float | None = None; stop_price: float | None = None

class OrderResponse(BaseModel):
    orderId: int | None = None; status: str

class Explanation(BaseModel):
    json: dict; text: str

class AnalysisRequest(BaseModel):
    """Request para análisis de señales con LLM"""
    ticker: str
    signals: list[dict]
    use_cache: bool = True

class AnalysisResponse(BaseModel):
    """Response del análisis LLM"""
    analysis: str
    cached: bool
    error: str | None = None
    signal_count: int

# Lista de tickers disponibles para el dashboard
AVAILABLE_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ",
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
    "BTC-USD", "ETH-USD", "SPY", "QQQ", "DIA", "IWM"
]

# Mapeo de intervalos del dashboard a yfinance
INTERVAL_MAP = {
    "1m": {"yf_interval": "1m", "max_days": 7, "tf_code": "M1"},
    "5m": {"yf_interval": "5m", "max_days": 60, "tf_code": "M5"},
    "15m": {"yf_interval": "15m", "max_days": 60, "tf_code": "M15"},
    "1h": {"yf_interval": "1h", "max_days": 730, "tf_code": "H1"},
    "1d": {"yf_interval": "1d", "max_days": 3650, "tf_code": "D1"},
    "1mo": {"yf_interval": "1mo", "max_days": 7300, "tf_code": "MO"}
}

@app.get("/tickers")
def get_tickers():
    """Obtener lista de tickers disponibles"""
    return {"tickers": AVAILABLE_TICKERS}

@app.get("/intervals")
def get_intervals():
    """Obtener intervalos de tiempo disponibles"""
    return {
        "intervals": [
            {"value": "1m", "label": "1 Minuto"},
            {"value": "5m", "label": "5 Minutos"},
            {"value": "15m", "label": "15 Minutos"},
            {"value": "1h", "label": "1 Hora"},
            {"value": "1d", "label": "1 Día"},
            {"value": "1mo", "label": "1 Mes"}
        ]
    }

@app.get("/signals", response_model=list[Signal])
def get_signals(
    ticker: Optional[str] = Query(None, description="Símbolo del ticker (ej: AAPL, MSFT)"),
    years: Optional[float] = Query(None, description="Años de datos históricos (0.5, 1, 2, 5, etc.)"),
    interval: Optional[str] = Query(None, description="Intervalo de velas (1m, 5m, 15m, 1h, 1d, 1mo)")
):
    """Obtener señales de trading con datos frescos de yfinance"""
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Usar parámetros o valores por defecto de config
    symbol = ticker if ticker else cfg["symbol"]
    yf_interval = interval if interval else "5m"  # Default 5 minutos
    period_years = years if years else 0.1  # Default ~1 mes
    
    # Obtener configuración del intervalo
    interval_cfg = INTERVAL_MAP.get(yf_interval, INTERVAL_MAP["5m"])
    
    # Calcular fechas
    end_date = datetime.now()
    days_back = min(int(period_years * 365), interval_cfg["max_days"])
    start_date = end_date - timedelta(days=days_back)
    
    # Normalizar símbolo para Forex si es necesario
    if symbol.upper().replace(".", "") in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]:
        if not symbol.endswith("=X"):
            symbol = symbol.replace(".", "").upper() + "=X"
    
    # Descargar datos con yfinance
    try:
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval_cfg["yf_interval"]
        )
        
        if df.empty:
            # Fallback: intentar con period en lugar de start/end
            period_str = f"{days_back}d" if days_back <= 730 else f"{period_years}y"
            df = ticker_obj.history(period=period_str, interval=interval_cfg["yf_interval"])
    except Exception as e:
        print(f"Error descargando datos: {e}")
        df = None
    
    if df is None or df.empty:
        return []
    
    # Normalizar columnas
    df.columns = df.columns.str.lower()
    df = df[["open", "high", "low", "close", "volume"]]
    
    # Asegurar timezone UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    
    # Crear config temporal con timeframe correcto
    temp_cfg = cfg.copy()
    temp_cfg["timeframe"] = interval_cfg["tf_code"]
    
    # Calcular indicadores
    df = compute_all_indicators(df, temp_cfg)
    
    # Aplicar warmup ANTES de generar señales
    warmup = temp_cfg["warmup_bars"]
    if len(df) > warmup:
        df = df.iloc[warmup:].copy()
    
    # Calcular indicadores de cada estrategia individualmente para el detalle de votación
    strategy_signals = {}
    strategy_scores = {}
    for strat_name, strat_func in STRATEGIES.items():
        try:
            sig, scr, _ = strat_func(df, temp_cfg)
            strategy_signals[strat_name] = sig
            strategy_scores[strat_name] = scr
        except Exception as e:
            print(f"Error en estrategia {strat_name}: {e}")
            strategy_signals[strat_name] = pd.Series(0, index=df.index)
            strategy_scores[strat_name] = pd.Series(0.0, index=df.index)
    
    # Generar señales combinadas (votación)
    df = generate_signals(df, temp_cfg)
    
    # Retornar TODAS las barras con detalle de votación
    out = []
    for ts, row in df.iterrows():
        # Construir detalle de votación para esta barra
        votes = []
        long_count = 0
        short_count = 0
        neutral_count = 0
        
        for strat_name in STRATEGIES.keys():
            sig_val = int(strategy_signals[strat_name].loc[ts]) if ts in strategy_signals[strat_name].index else 0
            scr_val = float(strategy_scores[strat_name].loc[ts]) if ts in strategy_scores[strat_name].index else 0.0
            
            if sig_val == 1:
                vote_str = "LONG"
                long_count += 1
            elif sig_val == -1:
                vote_str = "SHORT"
                short_count += 1
            else:
                vote_str = "NEUTRAL"
                neutral_count += 1
            
            votes.append(StrategyVote(
                name=strat_name,
                signal=sig_val,
                score=round(scr_val, 4),
                vote=vote_str
            ))
        
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
            tp=float(row["tp"]) if row["signal"]!=0 else None,
            strategy_votes=votes if row["signal"] != 0 else None,
            long_votes=long_count if row["signal"] != 0 else None,
            short_votes=short_count if row["signal"] != 0 else None,
            neutral_votes=neutral_count if row["signal"] != 0 else None
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


@app.post("/analysis", response_model=AnalysisResponse)
def post_analysis(req: AnalysisRequest):
    """
    Genera un análisis detallado de las señales de trading usando LLM.
    
    El análisis incluye:
    - Explicación de cada señal
    - Estrategias evaluadas y sus votos
    - Justificación de la decisión (LONG/SHORT)
    - Gestión de riesgo (SL/TP)
    """
    # Filtrar solo señales con posición (LONG o SHORT)
    active_signals = [s for s in req.signals if s.get("signal") != 0]
    
    if not active_signals:
        return AnalysisResponse(
            analysis="No hay señales activas para analizar. El sistema no ha generado posiciones LONG ni SHORT en el período seleccionado.",
            cached=False,
            error=None,
            signal_count=0
        )
    
    # Llamar a la función de análisis
    result = analyze_signals(
        asset_name=req.ticker,
        signal_history=active_signals,
        use_cache=req.use_cache
    )
    
    return AnalysisResponse(
        analysis=result.get("analysis", ""),
        cached=result.get("cached", False),
        error=result.get("error"),
        signal_count=len(active_signals)
    )


@app.delete("/analysis/cache")
def delete_analysis_cache():
    """Limpia el caché de análisis LLM"""
    result = clear_analysis_cache()
    return result


@app.get("/")
def serve_dashboard():
    """Servir el dashboard principal"""
    index_path = DASHBOARD_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Dashboard no encontrado. Coloca index.html en /dashboard/"}

@app.get("/backtest_report.json")
def serve_backtest_report():
    """Servir el reporte de backtest"""
    report_path = BASE_DIR / "backtest_report.json"
    if report_path.exists():
        return FileResponse(str(report_path))
    return {"metrics": {}, "signals": []}


# ========== NUEVO ENDPOINT DE BACKTEST ==========

class BacktestRequest(BaseModel):
    """Parámetros opcionales para el backtest"""
    capital: Optional[float] = None
    risk_pct: Optional[float] = None
    sl_atr_mult: Optional[float] = None
    tp_atr_mult: Optional[float] = None


@app.get("/backtest/{ticker}")
def get_backtest(
    ticker: str,
    years: Optional[float] = Query(0.25, description="Años de datos históricos"),
    interval: Optional[str] = Query("1h", description="Intervalo de velas (1h, 1d recomendados)"),
    capital: Optional[float] = Query(None, description="Capital inicial (default: 10000)"),
    risk_pct: Optional[float] = Query(None, description="% de riesgo por trade (default: 0.75)"),
):
    """
    Ejecuta un backtest para el ticker especificado y retorna resultados detallados.
    
    Incluye:
    - Resumen con métricas de rendimiento (Win Rate, PnL, Sharpe, etc.)
    - Lista de todos los trades con entrada, salida, PnL y duración
    - Curva de equity simplificada para gráficos
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Obtener configuración del intervalo
    interval_cfg = INTERVAL_MAP.get(interval, INTERVAL_MAP["1h"])
    
    # Calcular fechas
    end_date = datetime.now()
    days_back = min(int(years * 365), interval_cfg["max_days"])
    start_date = end_date - timedelta(days=days_back)
    
    # Normalizar símbolo para Forex
    symbol = ticker
    if ticker.upper().replace(".", "") in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]:
        if not ticker.endswith("=X"):
            symbol = ticker.replace(".", "").upper() + "=X"
    
    # Descargar datos
    try:
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval_cfg["yf_interval"]
        )
        
        if df.empty:
            period_str = f"{days_back}d" if days_back <= 730 else f"{years}y"
            df = ticker_obj.history(period=period_str, interval=interval_cfg["yf_interval"])
    except Exception as e:
        return {"error": f"Error descargando datos: {str(e)}", "ticker": ticker}
    
    if df is None or df.empty:
        return {"error": "No hay datos disponibles para este ticker/período", "ticker": ticker}
    
    # Normalizar columnas
    df.columns = df.columns.str.lower()
    df = df[["open", "high", "low", "close", "volume"]]
    
    # Asegurar timezone UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    
    # Crear config con parámetros custom
    temp_cfg = cfg.copy()
    temp_cfg["timeframe"] = interval_cfg["tf_code"]
    
    # Aplicar parámetros personalizados si se proporcionaron
    if capital is not None:
        temp_cfg["risk"]["capital"] = capital
    if risk_pct is not None:
        temp_cfg["risk"]["risk_pct"] = risk_pct / 100  # Convertir de % a decimal
    
    # Calcular indicadores
    df = compute_all_indicators(df, temp_cfg)
    
    # Aplicar warmup
    warmup = temp_cfg["warmup_bars"]
    if len(df) > warmup:
        df = df.iloc[warmup:].copy()
    
    # Generar señales
    df = generate_signals(df, temp_cfg)
    
    # Ejecutar backtest
    try:
        result = run_backtest_for_api(df=df, cfg=temp_cfg, ticker=ticker)
        return result
    except Exception as e:
        return {"error": f"Error en backtest: {str(e)}", "ticker": ticker}


@app.post("/backtest/{ticker}")
def post_backtest(
    ticker: str,
    request: BacktestRequest,
    years: Optional[float] = Query(0.25, description="Años de datos históricos"),
    interval: Optional[str] = Query("1h", description="Intervalo de velas"),
):
    """
    Versión POST del backtest que permite configuración avanzada en el body.
    """
    return get_backtest(
        ticker=ticker,
        years=years,
        interval=interval,
        capital=request.capital,
        risk_pct=request.risk_pct
    )



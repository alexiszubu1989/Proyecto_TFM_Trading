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
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from mvpfx.config import get_cfg
from mvpfx.data import load_data
from mvpfx.indicators import compute_all_indicators
from mvpfx.strategy import generate_signals
from mvpfx.risk import position_size, enforce_daily_limits


@dataclass
class TradeRecord:
    """Registro detallado de un trade individual"""
    trade_id: int
    side: str  # "LONG" or "SHORT"
    entry_time: str
    entry_price: float
    exit_time: Optional[str]
    exit_price: Optional[float]
    units: float
    sl: float
    tp: float
    pnl: float
    pnl_pct: float
    exit_reason: Optional[str]  # "SL", "TP", "OPEN"
    duration_minutes: Optional[float]
    status: str  # "OPEN", "CLOSED"


@dataclass
class BacktestStats:
    """Estadísticas agregadas del backtest"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    cagr: float
    avg_duration_minutes: float
    initial_capital: float
    final_equity: float


@dataclass
class BTResult:
    """Resultado completo del backtest con tracking detallado"""
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict
    # Nuevos campos para tracking detallado
    trade_history: List[TradeRecord] = field(default_factory=list)
    stats: Optional[BacktestStats] = None
    equity_history: List[Dict[str, Any]] = field(default_factory=list)
    ticker: str = "UNKNOWN"


def compute_metrics(equity: pd.Series) -> dict:
    """Calcula métricas básicas de la curva de equity"""
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


def compute_detailed_stats(trade_history: List[TradeRecord], equity_curve: pd.Series, initial_capital: float) -> BacktestStats:
    """Calcula estadísticas detalladas del backtest"""
    closed_trades = [t for t in trade_history if t.status == "CLOSED"]
    
    if not closed_trades:
        return BacktestStats(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
            total_pnl=0.0, total_pnl_pct=0.0, avg_pnl=0.0, avg_win=0.0, avg_loss=0.0,
            best_trade=0.0, worst_trade=0.0, profit_factor=0.0,
            max_drawdown=0.0, max_drawdown_pct=0.0, sharpe_ratio=0.0, sortino_ratio=0.0, cagr=0.0,
            avg_duration_minutes=0.0, initial_capital=initial_capital,
            final_equity=initial_capital
        )
    
    pnls = [t.pnl for t in closed_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    durations = [t.duration_minutes for t in closed_trades if t.duration_minutes is not None]
    
    total_pnl = sum(pnls)
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    
    # Calcular métricas de equity
    metrics = compute_metrics(equity_curve)
    final_equity = float(equity_curve.iloc[-1]) if len(equity_curve) > 0 else initial_capital
    
    # Max Drawdown absoluto
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_dd = float(drawdown.min())
    max_dd_pct = float((drawdown / running_max).min()) if running_max.max() > 0 else 0.0
    
    return BacktestStats(
        total_trades=len(closed_trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=len(wins) / len(closed_trades) * 100 if closed_trades else 0.0,
        total_pnl=total_pnl,
        total_pnl_pct=(total_pnl / initial_capital) * 100,
        avg_pnl=np.mean(pnls) if pnls else 0.0,
        avg_win=np.mean(wins) if wins else 0.0,
        avg_loss=np.mean(losses) if losses else 0.0,
        best_trade=max(pnls) if pnls else 0.0,
        worst_trade=min(pnls) if pnls else 0.0,
        profit_factor=gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct * 100,
        sharpe_ratio=metrics["Sharpe"],
        sortino_ratio=metrics["Sortino"],
        cagr=metrics["CAGR"] * 100,
        avg_duration_minutes=np.mean(durations) if durations else 0.0,
        initial_capital=initial_capital,
        final_equity=final_equity
    )


def run_backtest(
    df: Optional[pd.DataFrame] = None,
    cfg: Optional[dict] = None,
    ticker: str = "UNKNOWN"
) -> BTResult:
    """
    Ejecuta el backtest con tracking detallado de trades.
    
    Args:
        df: DataFrame con datos OHLCV ya procesados (opcional, si None carga desde config)
        cfg: Configuración (opcional, si None usa config por defecto)
        ticker: Nombre del ticker para referencia
    
    Returns:
        BTResult con historial completo de trades, equity y estadísticas
    """
    if cfg is None:
        cfg = get_cfg()
    
    if df is None:
        df = load_data()
        df = compute_all_indicators(df, cfg)
        df = generate_signals(df, cfg).iloc[cfg["warmup_bars"]:]
    
    ex, rk = cfg["execution"], cfg["risk"]
    initial_capital = rk["capital"]
    equity = initial_capital
    position, units, entry = 0, 0, np.nan
    entry_time, current_sl, current_tp = None, np.nan, np.nan
    
    records, equity_curve = [], []
    trade_log = pd.DataFrame(columns=["side","entry","exit","pnl"])
    trade_history: List[TradeRecord] = []
    equity_history: List[Dict[str, Any]] = []
    trade_counter = 0

    for ts, row in df.iterrows():
        price = row["close"]
        ask = price + ex["simulate_spread"]/2 + ex["simulate_slippage"]
        bid = price - ex["simulate_spread"]/2 - ex["simulate_slippage"]
        ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)

        # ========== CIERRE DE POSICIÓN ==========
        if position != 0:
            exit_triggered = False
            exit_price = 0.0
            exit_reason = ""
            
            if position == 1:  # LONG
                if bid <= row["sl"]:
                    exit_price = row["sl"]  # Hit Stop Loss
                    exit_reason = "SL"
                    exit_triggered = True
                elif bid >= row["tp"]:
                    exit_price = row["tp"]  # Hit Take Profit
                    exit_reason = "TP"
                    exit_triggered = True
            
            elif position == -1:  # SHORT
                if ask >= row["sl"]:
                    exit_price = row["sl"]  # Hit Stop Loss
                    exit_reason = "SL"
                    exit_triggered = True
                elif ask <= row["tp"]:
                    exit_price = row["tp"]  # Hit Take Profit
                    exit_reason = "TP"
                    exit_triggered = True
            
            if exit_triggered:
                # Calcular PnL
                if position == 1:
                    pnl = (exit_price - entry) * units
                else:
                    pnl = (entry - exit_price) * units
                
                equity += pnl
                pnl_pct = (pnl / (entry * units)) * 100 if entry * units > 0 else 0.0
                
                # Calcular duración
                duration_mins = None
                if entry_time:
                    try:
                        entry_dt = pd.Timestamp(entry_time)
                        exit_dt = pd.Timestamp(ts)
                        duration_mins = (exit_dt - entry_dt).total_seconds() / 60
                    except:
                        duration_mins = None
                
                # Actualizar trade en historial
                for trade in trade_history:
                    if trade.trade_id == trade_counter and trade.status == "OPEN":
                        trade.exit_time = ts_str
                        trade.exit_price = float(exit_price)
                        trade.pnl = float(pnl)
                        trade.pnl_pct = float(pnl_pct)
                        trade.exit_reason = exit_reason
                        trade.duration_minutes = duration_mins
                        trade.status = "CLOSED"
                        break
                
                records.append({
                    "time": ts, "type": "exit", "price": float(exit_price),
                    "pnl": float(pnl), "reason": exit_reason
                })
                trade_log.loc[ts] = {
                    "side": "long" if position == 1 else "short",
                    "entry": entry, "exit": exit_price, "pnl": pnl
                }
                position, units = 0, 0

        # ========== APERTURA DE POSICIÓN ==========
        if position == 0 and not enforce_daily_limits(trade_log, rk["capital"]):
            sig = int(row["signal"])
            
            if sig == 1:  # LONG
                trade_counter += 1
                units = position_size(equity, ask, row["atr"], cfg)
                entry = ask
                entry_time = ts_str
                current_sl = row["sl"]
                current_tp = row["tp"]
                position = 1
                
                trade_history.append(TradeRecord(
                    trade_id=trade_counter,
                    side="LONG",
                    entry_time=ts_str,
                    entry_price=float(entry),
                    exit_time=None,
                    exit_price=None,
                    units=float(units),
                    sl=float(current_sl),
                    tp=float(current_tp),
                    pnl=0.0,
                    pnl_pct=0.0,
                    exit_reason=None,
                    duration_minutes=None,
                    status="OPEN"
                ))
                
                records.append({
                    "time": ts, "type": "entry_long",
                    "price": float(entry), "units": units,
                    "sl": float(current_sl), "tp": float(current_tp)
                })
            
            elif sig == -1:  # SHORT
                trade_counter += 1
                units = position_size(equity, bid, row["atr"], cfg)
                entry = bid
                entry_time = ts_str
                current_sl = row["sl"]
                current_tp = row["tp"]
                position = -1
                
                trade_history.append(TradeRecord(
                    trade_id=trade_counter,
                    side="SHORT",
                    entry_time=ts_str,
                    entry_price=float(entry),
                    exit_time=None,
                    exit_price=None,
                    units=float(units),
                    sl=float(current_sl),
                    tp=float(current_tp),
                    pnl=0.0,
                    pnl_pct=0.0,
                    exit_reason=None,
                    duration_minutes=None,
                    status="OPEN"
                ))
                
                records.append({
                    "time": ts, "type": "entry_short",
                    "price": float(entry), "units": units,
                    "sl": float(current_sl), "tp": float(current_tp)
                })

        # ========== REGISTRAR EQUITY ==========
        equity_curve.append((ts, equity))
        equity_history.append({
            "timestamp": ts_str,
            "equity": float(equity),
            "drawdown": 0.0  # Se calcula después
        })

    # ========== POST-PROCESO ==========
    eq = pd.Series({t: v for t, v in equity_curve})
    trades = pd.DataFrame(records)
    metrics = compute_metrics(eq)
    
    # Calcular drawdown en equity_history
    if equity_history:
        max_equity = initial_capital
        for eh in equity_history:
            if eh["equity"] > max_equity:
                max_equity = eh["equity"]
            eh["drawdown"] = eh["equity"] - max_equity
    
    # Calcular estadísticas detalladas
    stats = compute_detailed_stats(trade_history, eq, initial_capital)
    
    # Guardar reporte JSON mejorado
    last_eq = float(eq.iloc[-1]) if len(eq) > 0 else initial_capital
    report = {
        "ticker": ticker,
        "metrics": metrics,
        "last_equity": last_eq,
        "stats": {
            "total_trades": stats.total_trades,
            "win_rate": round(stats.win_rate, 2),
            "total_pnl": round(stats.total_pnl, 2),
            "profit_factor": round(stats.profit_factor, 2) if stats.profit_factor != float('inf') else "∞",
            "max_drawdown_pct": round(stats.max_drawdown_pct, 2),
            "sharpe_ratio": round(stats.sharpe_ratio, 2)
        }
    }
    with open("backtest_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    return BTResult(
        equity_curve=eq,
        trades=trades,
        metrics=metrics,
        trade_history=trade_history,
        stats=stats,
        equity_history=equity_history,
        ticker=ticker
    )


def run_backtest_for_api(
    df: pd.DataFrame,
    cfg: dict,
    ticker: str = "UNKNOWN"
) -> Dict[str, Any]:
    """
    Versión del backtest optimizada para la API.
    Retorna un diccionario listo para serializar a JSON.
    """
    result = run_backtest(df=df, cfg=cfg, ticker=ticker)
    
    # Convertir trade_history a lista de diccionarios
    trades_list = []
    for t in result.trade_history:
        trades_list.append({
            "id": t.trade_id,
            "side": t.side,
            "entry_time": t.entry_time,
            "entry_price": round(t.entry_price, 4),
            "exit_time": t.exit_time,
            "exit_price": round(t.exit_price, 4) if t.exit_price else None,
            "units": round(t.units, 4),
            "sl": round(t.sl, 4),
            "tp": round(t.tp, 4),
            "pnl": round(t.pnl, 2),
            "pnl_pct": round(t.pnl_pct, 2),
            "exit_reason": t.exit_reason,
            "duration_minutes": round(t.duration_minutes, 1) if t.duration_minutes else None,
            "status": t.status
        })
    
    # Simplificar equity_history (tomar cada N puntos para reducir tamaño)
    equity_simplified = []
    step = max(1, len(result.equity_history) // 100)  # Máximo 100 puntos
    for i in range(0, len(result.equity_history), step):
        eh = result.equity_history[i]
        equity_simplified.append({
            "t": eh["timestamp"],
            "e": round(eh["equity"], 2),
            "d": round(eh["drawdown"], 2)
        })
    
    # Construir respuesta
    stats = result.stats
    return {
        "ticker": ticker,
        "summary": {
            "total_trades": stats.total_trades,
            "winning_trades": stats.winning_trades,
            "losing_trades": stats.losing_trades,
            "win_rate": round(stats.win_rate, 1),
            "total_pnl": round(stats.total_pnl, 2),
            "total_pnl_pct": round(stats.total_pnl_pct, 2),
            "avg_pnl": round(stats.avg_pnl, 2),
            "avg_win": round(stats.avg_win, 2),
            "avg_loss": round(stats.avg_loss, 2),
            "best_trade": round(stats.best_trade, 2),
            "worst_trade": round(stats.worst_trade, 2),
            "profit_factor": round(stats.profit_factor, 2) if stats.profit_factor != float('inf') else 999.99,
            "max_drawdown": round(stats.max_drawdown, 2),
            "max_drawdown_pct": round(stats.max_drawdown_pct, 2),
            "sharpe_ratio": round(stats.sharpe_ratio, 2),
            "sortino_ratio": round(stats.sortino_ratio, 2),
            "cagr": round(stats.cagr, 2),
            "avg_duration_minutes": round(stats.avg_duration_minutes, 1),
            "initial_capital": stats.initial_capital,
            "final_equity": round(stats.final_equity, 2)
        },
        "trades": trades_list,
        "equity_curve": equity_simplified
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run backtest")
    p.add_argument("--print", action="store_true", help="Imprime métricas")
    p.add_argument("--detailed", action="store_true", help="Imprime detalle de trades")
    args = p.parse_args()
    
    res = run_backtest()
    
    if args.print:
        print("\n📊 MÉTRICAS DEL BACKTEST")
        print("=" * 40)
        print(f"  CAGR:         {res.metrics['CAGR']*100:.2f}%")
        print(f"  Sharpe:       {res.metrics['Sharpe']:.2f}")
        print(f"  Sortino:      {res.metrics['Sortino']:.2f}")
        print(f"  Max Drawdown: {res.metrics['MaxDrawdown']*100:.2f}%")
        print(f"  Barras:       {res.metrics['Bars']}")
    
    if args.detailed and res.stats:
        print("\n💰 ESTADÍSTICAS DETALLADAS")
        print("=" * 40)
        print(f"  Total Trades:    {res.stats.total_trades}")
        print(f"  Win Rate:        {res.stats.win_rate:.1f}%")
        print(f"  Total PnL:       ${res.stats.total_pnl:.2f}")
        print(f"  Profit Factor:   {res.stats.profit_factor:.2f}")
        print(f"  Best Trade:      ${res.stats.best_trade:.2f}")
        print(f"  Worst Trade:     ${res.stats.worst_trade:.2f}")
        print(f"  Avg Duration:    {res.stats.avg_duration_minutes:.1f} min")
        print(f"  Final Equity:    ${res.stats.final_equity:.2f}")
    
    print("\n✅ OK: backtest_report.json generado.")

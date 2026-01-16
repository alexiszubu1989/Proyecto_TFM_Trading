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
from mvpfx.indicators import compute_all_indicators

def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))

def cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))

def regime_trending(df: pd.DataFrame, threshold: float) -> pd.Series:
    return (df["ema_fast"] - df["ema_slow"]).abs() / df["close"].abs() >= threshold

# ==================== ESTRATEGIAS ====================

def strategy_ema_crossover(df: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series, str]:
    """
    ESTRATEGIA 1: EMA Crossover (Original)
    - LONG: EMA rápida cruza hacia arriba la EMA lenta
    - SHORT: EMA rápida cruza hacia abajo la EMA lenta
    - Filtros: RSI, MACD, volatilidad ATR, régimen de tendencia
    """
    st = cfg["strategy"]
    filt_vol = (df["atr"] / df["close"].abs()) >= st["min_atr_pct"]
    reg = regime_trending(df, st["regime_threshold"])
    
    long_cross = cross_up(df["ema_fast"], df["ema_slow"])
    short_cross = cross_down(df["ema_fast"], df["ema_slow"])
    
    macd_ok_long = df["macd"] >= df["macd_signal"] if st["macd_confirm"] else pd.Series(True, index=df.index)
    macd_ok_short = df["macd"] <= df["macd_signal"] if st["macd_confirm"] else pd.Series(True, index=df.index)
    
    rsi_ok_long = df["rsi"] >= st["rsi_long_min"]
    rsi_ok_short = df["rsi"] <= st["rsi_short_max"]
    
    cond_long = long_cross & rsi_ok_long & macd_ok_long & filt_vol & reg
    cond_short = short_cross & rsi_ok_short & macd_ok_short & filt_vol & reg
    
    # Score basado en 5 componentes
    long_score = (long_cross.astype(int) + rsi_ok_long.astype(int) + macd_ok_long.astype(int) + 
                  filt_vol.astype(int) + reg.astype(int)) / 5.0
    short_score = (short_cross.astype(int) + rsi_ok_short.astype(int) + macd_ok_short.astype(int) + 
                   filt_vol.astype(int) + reg.astype(int)) / 5.0
    
    signal = pd.Series(0, index=df.index, dtype=int).mask(cond_long, 1).mask(cond_short, -1)
    score = pd.Series(0.0, index=df.index).mask(signal==1, long_score).mask(signal==-1, short_score)
    
    return signal, score, "ema_crossover"


def strategy_rsi_reversal(df: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series, str]:
    """
    ESTRATEGIA 2: RSI Reversal (Reversión a la Media)
    - LONG: RSI sale de zona de sobreventa (< 30 → > 30) + Stochastic confirma
    - SHORT: RSI sale de zona de sobrecompra (> 70 → < 70) + Stochastic confirma
    - Usa Williams %R y Stochastic como confirmación
    """
    st_cfg = cfg["strategies"]["rsi_reversal"]
    
    # RSI sale de sobreventa (cruce hacia arriba de 30)
    rsi_oversold_exit = (df["rsi"] > st_cfg["rsi_oversold"]) & (df["rsi"].shift(1) <= st_cfg["rsi_oversold"])
    # RSI sale de sobrecompra (cruce hacia abajo de 70)
    rsi_overbought_exit = (df["rsi"] < st_cfg["rsi_overbought"]) & (df["rsi"].shift(1) >= st_cfg["rsi_overbought"])
    
    # Confirmación con Stochastic
    stoch_oversold = df["stoch_k"] < st_cfg["stoch_oversold"]
    stoch_overbought = df["stoch_k"] > st_cfg["stoch_overbought"]
    stoch_cross_up = cross_up(df["stoch_k"], df["stoch_d"])
    stoch_cross_down = cross_down(df["stoch_k"], df["stoch_d"])
    
    # Williams %R confirmación
    wr_oversold = df["williams_r"] < st_cfg["williams_oversold"]
    wr_overbought = df["williams_r"] > st_cfg["williams_overbought"]
    
    # ADX filtro - solo operar en tendencias o rangos según configuración
    adx_filter = df["adx"] < st_cfg["adx_max"]  # Reversión funciona mejor en rangos
    
    # Condiciones finales
    cond_long = (rsi_oversold_exit | (stoch_oversold & stoch_cross_up)) & wr_oversold & adx_filter
    cond_short = (rsi_overbought_exit | (stoch_overbought & stoch_cross_down)) & wr_overbought & adx_filter
    
    # Score basado en confirmaciones
    long_score = (rsi_oversold_exit.astype(int) + stoch_cross_up.astype(int) + 
                  wr_oversold.astype(int) + adx_filter.astype(int)) / 4.0
    short_score = (rsi_overbought_exit.astype(int) + stoch_cross_down.astype(int) + 
                   wr_overbought.astype(int) + adx_filter.astype(int)) / 4.0
    
    signal = pd.Series(0, index=df.index, dtype=int).mask(cond_long, 1).mask(cond_short, -1)
    score = pd.Series(0.0, index=df.index).mask(signal==1, long_score).mask(signal==-1, short_score)
    
    return signal, score, "rsi_reversal"


def strategy_macd_crossover(df: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series, str]:
    """
    ESTRATEGIA 3: MACD Crossover con Momentum
    - LONG: MACD cruza hacia arriba su línea de señal + Histograma positivo creciente
    - SHORT: MACD cruza hacia abajo su línea de señal + Histograma negativo decreciente
    - Confirmación con CCI y Momentum
    """
    st_cfg = cfg["strategies"]["macd_crossover"]
    
    # MACD cruces
    macd_cross_up = cross_up(df["macd"], df["macd_signal"])
    macd_cross_down = cross_down(df["macd"], df["macd_signal"])
    
    # Histograma creciente/decreciente
    hist_increasing = df["macd_hist"] > df["macd_hist"].shift(1)
    hist_decreasing = df["macd_hist"] < df["macd_hist"].shift(1)
    
    # CCI confirmación
    cci_bullish = df["cci"] > st_cfg["cci_long_threshold"]
    cci_bearish = df["cci"] < st_cfg["cci_short_threshold"]
    
    # Momentum positivo/negativo
    mom_positive = df["momentum"] > 0
    mom_negative = df["momentum"] < 0
    
    # ROC confirmación
    roc_positive = df["roc"] > st_cfg["roc_threshold"]
    roc_negative = df["roc"] < -st_cfg["roc_threshold"]
    
    # ADX filtro - solo en tendencias fuertes
    adx_trending = df["adx"] > st_cfg["adx_min"]
    
    # Condiciones finales
    cond_long = macd_cross_up & hist_increasing & (cci_bullish | mom_positive) & adx_trending
    cond_short = macd_cross_down & hist_decreasing & (cci_bearish | mom_negative) & adx_trending
    
    # Score
    long_score = (macd_cross_up.astype(int) + hist_increasing.astype(int) + 
                  cci_bullish.astype(int) + mom_positive.astype(int) + 
                  roc_positive.astype(int) + adx_trending.astype(int)) / 6.0
    short_score = (macd_cross_down.astype(int) + hist_decreasing.astype(int) + 
                   cci_bearish.astype(int) + mom_negative.astype(int) + 
                   roc_negative.astype(int) + adx_trending.astype(int)) / 6.0
    
    signal = pd.Series(0, index=df.index, dtype=int).mask(cond_long, 1).mask(cond_short, -1)
    score = pd.Series(0.0, index=df.index).mask(signal==1, long_score).mask(signal==-1, short_score)
    
    return signal, score, "macd_crossover"


def strategy_bollinger_breakout(df: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series, str]:
    """
    ESTRATEGIA 4: Bollinger Bands Breakout
    - LONG: Precio cierra por encima de la banda superior + ADX indica tendencia fuerte
    - SHORT: Precio cierra por debajo de la banda inferior + ADX indica tendencia fuerte
    - Confirmación con +DI/-DI y volumen
    """
    st_cfg = cfg["strategies"]["bollinger_breakout"]
    
    # Breakout de bandas
    bb_upper_break = (df["close"] > df["bb_upper"]) & (df["close"].shift(1) <= df["bb_upper"].shift(1))
    bb_lower_break = (df["close"] < df["bb_lower"]) & (df["close"].shift(1) >= df["bb_lower"].shift(1))
    
    # Precio sobre/bajo la banda (continuo)
    above_upper = df["close"] > df["bb_upper"]
    below_lower = df["close"] < df["bb_lower"]
    
    # ADX indica tendencia fuerte
    strong_trend = df["adx"] > st_cfg["adx_strong"]
    
    # Direccionalidad (+DI vs -DI)
    bullish_di = df["plus_di"] > df["minus_di"]
    bearish_di = df["minus_di"] > df["plus_di"]
    
    # Momentum confirma dirección
    mom_confirms_long = df["momentum"] > st_cfg["momentum_threshold"]
    mom_confirms_short = df["momentum"] < -st_cfg["momentum_threshold"]
    
    # Precio alejándose de la media
    distance_from_mid = (df["close"] - df["bb_mid"]).abs() / df["bb_mid"]
    significant_move = distance_from_mid > st_cfg["bb_distance_pct"]
    
    # Condiciones finales
    cond_long = (bb_upper_break | above_upper) & strong_trend & bullish_di & mom_confirms_long
    cond_short = (bb_lower_break | below_lower) & strong_trend & bearish_di & mom_confirms_short
    
    # Score
    long_score = (bb_upper_break.astype(int) + strong_trend.astype(int) + 
                  bullish_di.astype(int) + mom_confirms_long.astype(int) + 
                  significant_move.astype(int)) / 5.0
    short_score = (bb_lower_break.astype(int) + strong_trend.astype(int) + 
                   bearish_di.astype(int) + mom_confirms_short.astype(int) + 
                   significant_move.astype(int)) / 5.0
    
    signal = pd.Series(0, index=df.index, dtype=int).mask(cond_long, 1).mask(cond_short, -1)
    score = pd.Series(0.0, index=df.index).mask(signal==1, long_score).mask(signal==-1, short_score)
    
    return signal, score, "bollinger_breakout"


# ==================== SELECTOR DE ESTRATEGIAS ====================

STRATEGIES = {
    "ema_crossover": strategy_ema_crossover,
    "rsi_reversal": strategy_rsi_reversal,
    "macd_crossover": strategy_macd_crossover,
    "bollinger_breakout": strategy_bollinger_breakout
}

# Prioridad de estrategias para desempate (mayor índice = mayor prioridad)
STRATEGY_PRIORITY = {
    "ema_crossover": 4,       # Mayor prioridad - más confiable en tendencias
    "macd_crossover": 3,      # Segunda prioridad - buena confirmación
    "bollinger_breakout": 2,  # Tercera - breakouts
    "rsi_reversal": 1         # Menor - reversiones son más arriesgadas
}


def resolve_tie(df: pd.DataFrame, signals_dict: dict, scores_dict: dict, 
                tie_mask: pd.Series, tie_method: str, cfg: dict) -> tuple[pd.Series, pd.Series]:
    """
    Resuelve empates entre señales LONG y SHORT usando diferentes métodos.
    
    Métodos disponibles:
    - "score": Usa el score promedio más alto
    - "priority": Usa la estrategia con mayor prioridad
    - "adx_trend": Si ADX > 25, sigue la tendencia (+DI vs -DI); si no, no opera
    - "conservative": No opera en empate (señal = 0)
    - "momentum": Usa el indicador de momentum para decidir
    
    Args:
        df: DataFrame con indicadores
        signals_dict: Dict con señales por estrategia
        scores_dict: Dict con scores por estrategia
        tie_mask: Serie booleana indicando dónde hay empate
        tie_method: Método de desempate a usar
        cfg: Configuración
    
    Returns:
        Tuple (signal, score) resueltos
    """
    signal = pd.Series(0, index=df.index, dtype=int)
    score = pd.Series(0.0, index=df.index)
    
    if not tie_mask.any():
        return signal, score
    
    if tie_method == "score":
        # Desempate por score promedio más alto
        long_scores = []
        short_scores = []
        for strat_name, sig in signals_dict.items():
            scr = scores_dict[strat_name]
            long_scores.append(scr.where(sig == 1, 0))
            short_scores.append(scr.where(sig == -1, 0))
        
        avg_long_score = pd.concat(long_scores, axis=1).mean(axis=1)
        avg_short_score = pd.concat(short_scores, axis=1).mean(axis=1)
        
        # Donde hay empate, elegir el de mayor score
        signal = signal.mask(tie_mask & (avg_long_score > avg_short_score), 1)
        signal = signal.mask(tie_mask & (avg_short_score > avg_long_score), -1)
        score = score.mask(signal == 1, avg_long_score)
        score = score.mask(signal == -1, avg_short_score)
        
    elif tie_method == "priority":
        # Desempate por prioridad de estrategia
        for idx in df.index[tie_mask]:
            best_priority = -1
            best_signal = 0
            best_score = 0.0
            
            for strat_name, sig in signals_dict.items():
                if sig.loc[idx] != 0:
                    priority = STRATEGY_PRIORITY.get(strat_name, 0)
                    if priority > best_priority:
                        best_priority = priority
                        best_signal = sig.loc[idx]
                        best_score = scores_dict[strat_name].loc[idx]
            
            signal.loc[idx] = best_signal
            score.loc[idx] = best_score
            
    elif tie_method == "adx_trend":
        # Desempate usando ADX y direccionalidad
        adx_threshold = cfg["strategies"].get("tie_breaker", {}).get("adx_threshold", 25)
        strong_trend = df["adx"] > adx_threshold
        bullish_trend = df["plus_di"] > df["minus_di"]
        bearish_trend = df["minus_di"] > df["plus_di"]
        
        # Si hay tendencia fuerte, seguir la dirección
        signal = signal.mask(tie_mask & strong_trend & bullish_trend, 1)
        signal = signal.mask(tie_mask & strong_trend & bearish_trend, -1)
        # Si no hay tendencia fuerte, no operar (queda en 0)
        score = score.mask(signal != 0, df["adx"] / 100)  # Score basado en fuerza de tendencia
        
    elif tie_method == "momentum":
        # Desempate usando momentum
        mom_threshold = cfg["strategies"].get("tie_breaker", {}).get("momentum_threshold", 0)
        
        signal = signal.mask(tie_mask & (df["momentum"] > mom_threshold), 1)
        signal = signal.mask(tie_mask & (df["momentum"] < -mom_threshold), -1)
        score = score.mask(signal != 0, (df["momentum"].abs() / df["close"]).clip(0, 1))
        
    elif tie_method == "conservative":
        # No operar en empate - ya está en 0
        pass
    
    return signal, score


def generate_signals(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """
    Genera señales usando votación por mayoría de TODAS las estrategias.
    
    Modos de operación:
    1. combine_strategies=False: Usa solo la estrategia activa
    2. combine_strategies=True: Evalúa TODAS las estrategias y decide por mayoría
    
    Métodos de desempate disponibles:
    - "score": Mayor score promedio gana
    - "priority": Estrategia con mayor prioridad gana
    - "adx_trend": Sigue la tendencia si ADX > 25
    - "conservative": No opera en empate
    - "momentum": Sigue la dirección del momentum
    """
    if cfg is None:
        cfg = get_cfg()
    
    st = cfg["strategy"]
    rk = cfg["risk"]
    
    # Obtener configuración
    active_strategy = st.get("active_strategy", "ema_crossover")
    combine_strategies = st.get("combine_strategies", False)
    tie_break_method = st.get("tie_break_method", "score")
    
    if combine_strategies:
        # ==================== VOTACIÓN POR MAYORÍA ====================
        # Evaluar TODAS las estrategias habilitadas
        enabled_strategies = st.get("enabled_strategies", list(STRATEGIES.keys()))
        
        signals_dict = {}
        scores_dict = {}
        signals_list = []
        scores_list = []
        strategy_names = []
        
        for strat_name in enabled_strategies:
            if strat_name in STRATEGIES:
                try:
                    sig, scr, _ = STRATEGIES[strat_name](df, cfg)
                    signals_dict[strat_name] = sig
                    scores_dict[strat_name] = scr
                    signals_list.append(sig)
                    scores_list.append(scr)
                    strategy_names.append(strat_name)
                except Exception as e:
                    print(f"⚠️ Error en estrategia {strat_name}: {e}")
        
        n_strategies = len(signals_list)
        
        if n_strategies == 0:
            signal = pd.Series(0, index=df.index, dtype=int)
            score = pd.Series(0.0, index=df.index)
            voting_detail = "No strategies available"
        else:
            # Combinar señales en DataFrame
            combined_signals = pd.concat(signals_list, axis=1, keys=strategy_names)
            combined_scores = pd.concat(scores_list, axis=1, keys=strategy_names)
            
            # Contar votos
            long_votes = (combined_signals == 1).sum(axis=1)
            short_votes = (combined_signals == -1).sum(axis=1)
            neutral_votes = (combined_signals == 0).sum(axis=1)
            
            # Determinar mayoría requerida
            min_votes = st.get("min_strategy_votes", max(1, n_strategies // 2 + 1))
            
            # Inicializar señal
            signal = pd.Series(0, index=df.index, dtype=int)
            score = pd.Series(0.0, index=df.index)
            
            # Caso 1: Mayoría clara de LONG
            clear_long = (long_votes >= min_votes) & (long_votes > short_votes)
            signal = signal.mask(clear_long, 1)
            
            # Caso 2: Mayoría clara de SHORT
            clear_short = (short_votes >= min_votes) & (short_votes > long_votes)
            signal = signal.mask(clear_short, -1)
            
            # Caso 3: EMPATE (mismo número de votos LONG y SHORT, ambos >= 1)
            tie_condition = (long_votes == short_votes) & (long_votes >= 1)
            
            if tie_condition.any():
                # Aplicar método de desempate
                tie_signal, tie_score = resolve_tie(
                    df, signals_dict, scores_dict, 
                    tie_condition, tie_break_method, cfg
                )
                signal = signal.mask(tie_condition, tie_signal)
                score = score.mask(tie_condition, tie_score)
            
            # Calcular score para señales sin empate
            # Score = promedio de scores de estrategias que votaron en la misma dirección
            for idx in df.index:
                if signal.loc[idx] == 1 and not tie_condition.loc[idx]:
                    scores_agreeing = [scores_dict[s].loc[idx] for s in strategy_names 
                                       if signals_dict[s].loc[idx] == 1]
                    score.loc[idx] = np.mean(scores_agreeing) if scores_agreeing else 0.0
                elif signal.loc[idx] == -1 and not tie_condition.loc[idx]:
                    scores_agreeing = [scores_dict[s].loc[idx] for s in strategy_names 
                                       if signals_dict[s].loc[idx] == -1]
                    score.loc[idx] = np.mean(scores_agreeing) if scores_agreeing else 0.0
            
            # Guardar detalles de votación
            voting_detail = f"Strategies: {n_strategies}, MinVotes: {min_votes}, TieBreak: {tie_break_method}"
            
    else:
        # ==================== ESTRATEGIA ÚNICA ====================
        if active_strategy in STRATEGIES:
            signal, score, _ = STRATEGIES[active_strategy](df, cfg)
        else:
            signal, score, _ = strategy_ema_crossover(df, cfg)
        voting_detail = f"Single: {active_strategy}"
    
    # ==================== CALCULAR SL/TP ====================
    sl_long = df["close"] - rk["atr_sl_mult"] * df["atr"]
    tp_long = df["close"] + rk["atr_tp_mult"] * df["atr"]
    sl_short = df["close"] + rk["atr_sl_mult"] * df["atr"]
    tp_short = df["close"] - rk["atr_tp_mult"] * df["atr"]
    
    sl = pd.Series(np.nan, index=df.index).mask(signal == 1, sl_long).mask(signal == -1, sl_short)
    tp = pd.Series(np.nan, index=df.index).mask(signal == 1, tp_long).mask(signal == -1, tp_short)
    
    # ==================== CONSTRUIR OUTPUT ====================
    out = df.copy()
    out["signal"] = signal
    out["score"] = score.clip(0, 1).fillna(0.0)
    out["sl"], out["tp"] = sl, tp
    out["strategy"] = "combined" if combine_strategies else active_strategy
    
    # Añadir columnas de votación si se combinaron estrategias
    if combine_strategies and 'combined_signals' in dir():
        out["long_votes"] = long_votes
        out["short_votes"] = short_votes
        out["is_tie"] = tie_condition.astype(int)
    
    return out
    
    sl = pd.Series(np.nan, index=df.index).mask(signal == 1, sl_long).mask(signal == -1, sl_short)
    tp = pd.Series(np.nan, index=df.index).mask(signal == 1, tp_long).mask(signal == -1, tp_short)
    
    # Construir DataFrame de salida
    out = df.copy()
    out["signal"] = signal
    out["score"] = score.clip(0, 1).fillna(0.0)
    out["sl"], out["tp"] = sl, tp
    out["strategy"] = active_strategy if not combine_strategies else "combined"
    
    return out


def get_all_strategy_signals(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """
    Ejecuta TODAS las estrategias y retorna un DataFrame con las señales de cada una.
    Útil para comparar estrategias o hacer análisis.
    """
    if cfg is None:
        cfg = get_cfg()
    
    results = df.copy()
    
    for strat_name, strat_func in STRATEGIES.items():
        try:
            signal, score, _ = strat_func(df, cfg)
            results[f"signal_{strat_name}"] = signal
            results[f"score_{strat_name}"] = score
        except Exception as e:
            print(f"Error en estrategia {strat_name}: {e}")
            results[f"signal_{strat_name}"] = 0
            results[f"score_{strat_name}"] = 0.0
    
    return results

if __name__ == "__main__":
    from mvpfx.indicators import compute_all_indicators
    cfg = get_cfg()
    base = load_data()
    feats = compute_all_indicators(base, cfg)
    
    print("=" * 70)
    print("📊 ANÁLISIS DE TODAS LAS ESTRATEGIAS INDIVIDUALES")
    print("=" * 70)
    
    # Ejecutar todas las estrategias
    all_signals = get_all_strategy_signals(feats, cfg)
    
    for strat_name in STRATEGIES.keys():
        sig_col = f"signal_{strat_name}"
        if sig_col in all_signals.columns:
            counts = all_signals[sig_col].value_counts(dropna=False).to_dict()
            print(f"\n📈 {strat_name.upper()}:")
            print(f"   LONG (+1): {counts.get(1, 0):4d} | SHORT (-1): {counts.get(-1, 0):4d} | NEUTRAL (0): {counts.get(0, 0):4d}")
    
    print("\n" + "=" * 70)
    print("🗳️  VOTACIÓN POR MAYORÍA (TODAS LAS ESTRATEGIAS)")
    print("=" * 70)
    
    # Generar señales con votación
    sigs = generate_signals(feats, cfg)
    
    combine_mode = cfg["strategy"].get("combine_strategies", False)
    tie_method = cfg["strategy"].get("tie_break_method", "score")
    
    print(f"\n⚙️  Modo: {'COMBINACIÓN (Votación)' if combine_mode else 'ESTRATEGIA ÚNICA'}")
    print(f"📋 Estrategia: {sigs['strategy'].iloc[0]}")
    
    if combine_mode:
        print(f"🎯 Método de desempate: {tie_method.upper()}")
        print(f"🔢 Mínimo de votos requeridos: {cfg['strategy'].get('min_strategy_votes', 2)}")
        
        # Mostrar estadísticas de votación
        if "long_votes" in sigs.columns:
            n_ties = sigs["is_tie"].sum()
            print(f"\n📊 Estadísticas de votación:")
            print(f"   Empates detectados: {n_ties}")
            print(f"   Empates resueltos por '{tie_method}': {n_ties}")
    
    # Conteo de señales finales
    signal_counts = sigs["signal"].value_counts(dropna=False).to_dict()
    print(f"\n🎯 SEÑALES FINALES (después de votación/desempate):")
    print(f"   LONG (+1):    {signal_counts.get(1, 0):4d}")
    print(f"   SHORT (-1):   {signal_counts.get(-1, 0):4d}")
    print(f"   NEUTRAL (0):  {signal_counts.get(0, 0):4d}")
    
    # Mostrar últimas señales
    print("\n" + "-" * 70)
    print("📋 Últimas 10 barras con señales:")
    cols_to_show = ["close", "signal", "score"]
    if "long_votes" in sigs.columns:
        cols_to_show.extend(["long_votes", "short_votes", "is_tie"])
    cols_to_show.extend(["sl", "tp"])
    print(sigs[cols_to_show].tail(10))
    
    # Mostrar señales activas (no neutrales)
    active_signals = sigs[sigs["signal"] != 0]
    if len(active_signals) > 0:
        print(f"\n" + "-" * 70)
        print(f"🔔 Últimas 5 señales activas (LONG/SHORT):")
        print(active_signals[cols_to_show].tail(5))

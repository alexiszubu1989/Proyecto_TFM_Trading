"""
Módulo de Explicabilidad Avanzada para el Sistema de Trading

Este módulo implementa:
1. Explicaciones Contextuales Dinámicas de indicadores técnicos
2. Sistema de Advertencias de Riesgo con alertas automáticas
3. Explicación detallada de Votos Individuales de cada estrategia

Objetivo: Generar explicaciones transparentes y honestas que permitan al usuario
tomar decisiones informadas sobre si seguir o ignorar las recomendaciones del sistema.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np


# ============================================================================
# ENUMS Y CONSTANTES
# ============================================================================

class RiskLevel(Enum):
    """Niveles de riesgo para las alertas"""
    LOW = "bajo"
    MEDIUM = "medio"
    HIGH = "alto"
    CRITICAL = "crítico"


class SignalDirection(Enum):
    """Dirección de la señal"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


# Umbrales para clasificación de indicadores
INDICATOR_THRESHOLDS = {
    "rsi": {
        "oversold_extreme": 20,
        "oversold": 30,
        "neutral_low": 40,
        "neutral_high": 60,
        "overbought": 70,
        "overbought_extreme": 80
    },
    "adx": {
        "no_trend": 20,
        "weak_trend": 25,
        "moderate_trend": 40,
        "strong_trend": 50,
        "very_strong": 75
    },
    "stochastic": {
        "oversold_extreme": 10,
        "oversold": 20,
        "overbought": 80,
        "overbought_extreme": 90
    },
    "williams_r": {
        "oversold": -80,
        "overbought": -20
    },
    "cci": {
        "oversold_extreme": -200,
        "oversold": -100,
        "overbought": 100,
        "overbought_extreme": 200
    }
}


# ============================================================================
# DATACLASSES PARA ESTRUCTURAR LA EXPLICACIÓN
# ============================================================================

@dataclass
class RiskWarning:
    """Representa una advertencia de riesgo específica"""
    code: str                    # Código único de la alerta (ej: "RSI_DIVERGENCE")
    level: RiskLevel            # Nivel de riesgo
    title: str                  # Título corto de la alerta
    description: str            # Descripción detallada
    recommendation: str         # Acción recomendada
    indicator_values: Dict[str, Any] = field(default_factory=dict)  # Valores relevantes


@dataclass
class IndicatorExplanation:
    """Explicación contextual de un indicador"""
    name: str                   # Nombre del indicador
    value: float               # Valor actual
    interpretation: str        # Interpretación del valor
    signal_alignment: str      # "confirma", "contradice", "neutral"
    context: str               # Contexto adicional


@dataclass
class StrategyVoteExplanation:
    """Explicación detallada del voto de una estrategia"""
    strategy_name: str         # Nombre de la estrategia
    vote: str                  # "LONG", "SHORT", "NEUTRAL"
    score: float               # Score de confianza
    reasoning: str             # Razón del voto
    conditions_met: List[str]  # Condiciones que se cumplieron
    conditions_failed: List[str]  # Condiciones que NO se cumplieron
    key_indicators: Dict[str, str]  # Indicadores clave y su interpretación


@dataclass
class SignalExplanation:
    """Explicación completa de una señal de trading"""
    timestamp: str
    direction: SignalDirection
    price: float
    confidence_score: float
    
    # Componentes de la explicación
    summary: str                                    # Resumen ejecutivo
    indicator_explanations: List[IndicatorExplanation]  # Explicaciones de indicadores
    strategy_votes: List[StrategyVoteExplanation]  # Votos detallados
    risk_warnings: List[RiskWarning]               # Alertas de riesgo
    
    # Gestión de riesgo
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: Optional[float]
    
    # Conclusión final
    overall_risk_level: RiskLevel
    final_recommendation: str
    honest_disclaimer: str


# ============================================================================
# CLASE PRINCIPAL DE EXPLICABILIDAD
# ============================================================================

class TradingExplainer:
    """
    Motor de explicabilidad para el sistema de trading.
    
    Genera explicaciones transparentes, honestas y contextuales para cada señal,
    advirtiendo explícitamente sobre incertidumbres y riesgos.
    """
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.risk_config = cfg.get("risk", {})
        self.strategy_config = cfg.get("strategy", {})
    
    # ========================================================================
    # 1. EXPLICACIONES CONTEXTUALES DINÁMICAS DE INDICADORES
    # ========================================================================
    
    def explain_indicator(self, name: str, value: float, signal_direction: SignalDirection,
                         all_indicators: Dict[str, float]) -> IndicatorExplanation:
        """
        Genera una explicación contextual dinámica para un indicador específico.
        
        Args:
            name: Nombre del indicador (ej: "rsi", "macd", "adx")
            value: Valor actual del indicador
            signal_direction: Dirección de la señal generada
            all_indicators: Diccionario con todos los indicadores para contexto
        
        Returns:
            IndicatorExplanation con la interpretación completa
        """
        
        if name == "rsi":
            return self._explain_rsi(value, signal_direction, all_indicators)
        elif name == "macd":
            return self._explain_macd(value, signal_direction, all_indicators)
        elif name == "adx":
            return self._explain_adx(value, signal_direction, all_indicators)
        elif name == "ema_fast" or name == "ema_slow":
            return self._explain_ema(name, value, signal_direction, all_indicators)
        elif name == "stoch_k" or name == "stoch_d":
            return self._explain_stochastic(name, value, signal_direction, all_indicators)
        elif name == "atr":
            return self._explain_atr(value, signal_direction, all_indicators)
        elif name == "cci":
            return self._explain_cci(value, signal_direction, all_indicators)
        elif name == "momentum":
            return self._explain_momentum(value, signal_direction, all_indicators)
        elif name in ["bb_upper", "bb_lower", "bb_mid"]:
            return self._explain_bollinger(name, value, signal_direction, all_indicators)
        else:
            return IndicatorExplanation(
                name=name,
                value=value,
                interpretation=f"Valor actual: {value:.4f}",
                signal_alignment="neutral",
                context="Indicador disponible para análisis"
            )
    
    def _explain_rsi(self, value: float, signal_direction: SignalDirection,
                    indicators: Dict[str, float]) -> IndicatorExplanation:
        """Explicación contextual del RSI"""
        thresholds = INDICATOR_THRESHOLDS["rsi"]
        
        # Interpretación del valor
        if value <= thresholds["oversold_extreme"]:
            interpretation = f"RSI en {value:.2f}: SOBREVENTA EXTREMA. El activo está significativamente sobrevendido, lo que históricamente precede rebotes técnicos."
            zone = "sobreventa_extrema"
        elif value <= thresholds["oversold"]:
            interpretation = f"RSI en {value:.2f}: SOBREVENTA. Presión vendedora elevada, posible agotamiento de la tendencia bajista."
            zone = "sobreventa"
        elif value >= thresholds["overbought_extreme"]:
            interpretation = f"RSI en {value:.2f}: SOBRECOMPRA EXTREMA. El activo está significativamente sobrecomprado, lo que históricamente precede correcciones."
            zone = "sobrecompra_extrema"
        elif value >= thresholds["overbought"]:
            interpretation = f"RSI en {value:.2f}: SOBRECOMPRA. Presión compradora elevada, posible agotamiento de la tendencia alcista."
            zone = "sobrecompra"
        elif value < thresholds["neutral_low"]:
            interpretation = f"RSI en {value:.2f}: NEUTRAL-BAJO. Momentum ligeramente bajista, pero sin condiciones extremas."
            zone = "neutral_bajo"
        elif value > thresholds["neutral_high"]:
            interpretation = f"RSI en {value:.2f}: NEUTRAL-ALTO. Momentum ligeramente alcista, pero sin condiciones extremas."
            zone = "neutral_alto"
        else:
            interpretation = f"RSI en {value:.2f}: NEUTRAL. Sin señal clara de momentum, el mercado está equilibrado."
            zone = "neutral"
        
        # Alineación con la señal
        if signal_direction == SignalDirection.LONG:
            if zone in ["sobreventa", "sobreventa_extrema"]:
                alignment = "confirma"
                context = "✅ RSI en sobreventa CONFIRMA potencial de rebote alcista."
            elif zone in ["sobrecompra", "sobrecompra_extrema"]:
                alignment = "contradice"
                context = "⚠️ ADVERTENCIA: RSI en sobrecompra CONTRADICE entrada LONG. Riesgo de corrección inminente."
            else:
                alignment = "neutral"
                context = "➖ RSI neutral no aporta confirmación adicional para LONG."
        elif signal_direction == SignalDirection.SHORT:
            if zone in ["sobrecompra", "sobrecompra_extrema"]:
                alignment = "confirma"
                context = "✅ RSI en sobrecompra CONFIRMA potencial de caída."
            elif zone in ["sobreventa", "sobreventa_extrema"]:
                alignment = "contradice"
                context = "⚠️ ADVERTENCIA: RSI en sobreventa CONTRADICE entrada SHORT. Riesgo de rebote inminente."
            else:
                alignment = "neutral"
                context = "➖ RSI neutral no aporta confirmación adicional para SHORT."
        else:
            alignment = "neutral"
            context = "➖ Señal NEUTRAL - RSI no requiere alineación específica."
        
        return IndicatorExplanation(
            name="RSI (14)",
            value=value,
            interpretation=interpretation,
            signal_alignment=alignment,
            context=context
        )
    
    def _explain_macd(self, value: float, signal_direction: SignalDirection,
                     indicators: Dict[str, float]) -> IndicatorExplanation:
        """Explicación contextual del MACD"""
        macd_signal = indicators.get("macd_signal", 0)
        macd_hist = indicators.get("macd_hist", value - macd_signal)
        
        # Interpretación
        if value > macd_signal:
            if macd_hist > 0:
                interpretation = f"MACD ({value:.6f}) por ENCIMA de señal ({macd_signal:.6f}). Histograma positivo ({macd_hist:.6f}) indica momentum alcista creciente."
            else:
                interpretation = f"MACD ({value:.6f}) cruzando hacia arriba señal ({macd_signal:.6f}). Posible inicio de momentum alcista."
        elif value < macd_signal:
            if macd_hist < 0:
                interpretation = f"MACD ({value:.6f}) por DEBAJO de señal ({macd_signal:.6f}). Histograma negativo ({macd_hist:.6f}) indica momentum bajista creciente."
            else:
                interpretation = f"MACD ({value:.6f}) cruzando hacia abajo señal ({macd_signal:.6f}). Posible inicio de momentum bajista."
        else:
            interpretation = f"MACD ({value:.6f}) igual a señal ({macd_signal:.6f}). Momento de indecisión, posible cambio de dirección."
        
        # Alineación
        if signal_direction == SignalDirection.LONG:
            if value > macd_signal and macd_hist > 0:
                alignment = "confirma"
                context = "✅ MACD alcista CONFIRMA momentum favorable para LONG."
            elif value < macd_signal:
                alignment = "contradice"
                context = "⚠️ ADVERTENCIA: MACD bajista CONTRADICE entrada LONG. Momentum desfavorable."
            else:
                alignment = "neutral"
                context = "➖ MACD en transición, confirmación débil."
        elif signal_direction == SignalDirection.SHORT:
            if value < macd_signal and macd_hist < 0:
                alignment = "confirma"
                context = "✅ MACD bajista CONFIRMA momentum favorable para SHORT."
            elif value > macd_signal:
                alignment = "contradice"
                context = "⚠️ ADVERTENCIA: MACD alcista CONTRADICE entrada SHORT. Momentum desfavorable."
            else:
                alignment = "neutral"
                context = "➖ MACD en transición, confirmación débil."
        else:
            alignment = "neutral"
            context = "➖ Señal NEUTRAL - MACD no requiere alineación específica."
        
        return IndicatorExplanation(
            name="MACD",
            value=value,
            interpretation=interpretation,
            signal_alignment=alignment,
            context=context
        )
    
    def _explain_adx(self, value: float, signal_direction: SignalDirection,
                    indicators: Dict[str, float]) -> IndicatorExplanation:
        """Explicación contextual del ADX"""
        thresholds = INDICATOR_THRESHOLDS["adx"]
        plus_di = indicators.get("plus_di", 0)
        minus_di = indicators.get("minus_di", 0)
        
        # Interpretación de fuerza de tendencia
        if value < thresholds["no_trend"]:
            strength = "SIN TENDENCIA"
            interpretation = f"ADX en {value:.2f}: {strength}. Mercado en rango/consolidación. Las estrategias de tendencia tienen ALTA probabilidad de fallo."
        elif value < thresholds["weak_trend"]:
            strength = "TENDENCIA MUY DÉBIL"
            interpretation = f"ADX en {value:.2f}: {strength}. Tendencia apenas perceptible. Señales de tendencia poco confiables."
        elif value < thresholds["moderate_trend"]:
            strength = "TENDENCIA MODERADA"
            interpretation = f"ADX en {value:.2f}: {strength}. Tendencia establecida pero no dominante."
        elif value < thresholds["strong_trend"]:
            strength = "TENDENCIA FUERTE"
            interpretation = f"ADX en {value:.2f}: {strength}. Tendencia clara y dominante. Señales de tendencia más confiables."
        else:
            strength = "TENDENCIA MUY FUERTE"
            interpretation = f"ADX en {value:.2f}: {strength}. Tendencia extremadamente fuerte. PRECAUCIÓN: posible agotamiento cercano."
        
        # Direccionalidad
        if plus_di > minus_di:
            direction = f"Dirección: ALCISTA (+DI={plus_di:.2f} > -DI={minus_di:.2f})"
        else:
            direction = f"Dirección: BAJISTA (-DI={minus_di:.2f} > +DI={plus_di:.2f})"
        
        interpretation = f"{interpretation} {direction}"
        
        # Alineación con señal
        if value < thresholds["no_trend"]:
            alignment = "contradice" if signal_direction != SignalDirection.NEUTRAL else "neutral"
            context = f"⚠️ ADX bajo ({value:.2f}) indica AUSENCIA DE TENDENCIA. Estrategias de tendencia no recomendadas."
        elif signal_direction == SignalDirection.LONG:
            if plus_di > minus_di:
                alignment = "confirma"
                context = f"✅ ADX ({value:.2f}) y +DI dominante CONFIRMAN tendencia alcista."
            else:
                alignment = "contradice"
                context = f"⚠️ ADX indica tendencia pero -DI dominante CONTRADICE LONG."
        elif signal_direction == SignalDirection.SHORT:
            if minus_di > plus_di:
                alignment = "confirma"
                context = f"✅ ADX ({value:.2f}) y -DI dominante CONFIRMAN tendencia bajista."
            else:
                alignment = "contradice"
                context = f"⚠️ ADX indica tendencia pero +DI dominante CONTRADICE SHORT."
        else:
            alignment = "neutral"
            context = "➖ Señal NEUTRAL - ADX no requiere alineación específica."
        
        return IndicatorExplanation(
            name="ADX (Fuerza de Tendencia)",
            value=value,
            interpretation=interpretation,
            signal_alignment=alignment,
            context=context
        )
    
    def _explain_ema(self, name: str, value: float, signal_direction: SignalDirection,
                    indicators: Dict[str, float]) -> IndicatorExplanation:
        """Explicación contextual de las EMAs"""
        ema_fast = indicators.get("ema_fast", value if name == "ema_fast" else 0)
        ema_slow = indicators.get("ema_slow", value if name == "ema_slow" else 0)
        close = indicators.get("close", value)
        
        diff = ema_fast - ema_slow
        diff_pct = (diff / ema_slow * 100) if ema_slow != 0 else 0
        
        if ema_fast > ema_slow:
            trend = "ALCISTA"
            interpretation = f"EMA rápida ({ema_fast:.5f}) por ENCIMA de EMA lenta ({ema_slow:.5f}). Diferencia: {diff_pct:.3f}%. Tendencia de corto plazo {trend}."
        elif ema_fast < ema_slow:
            trend = "BAJISTA"
            interpretation = f"EMA rápida ({ema_fast:.5f}) por DEBAJO de EMA lenta ({ema_slow:.5f}). Diferencia: {diff_pct:.3f}%. Tendencia de corto plazo {trend}."
        else:
            trend = "NEUTRAL"
            interpretation = f"EMAs convergiendo ({ema_fast:.5f} ≈ {ema_slow:.5f}). Posible cambio de tendencia o consolidación."
        
        # Posición del precio respecto a EMAs
        if close > ema_fast and close > ema_slow:
            price_position = "Precio por encima de ambas EMAs (momentum alcista)."
        elif close < ema_fast and close < ema_slow:
            price_position = "Precio por debajo de ambas EMAs (momentum bajista)."
        else:
            price_position = "Precio entre ambas EMAs (zona de incertidumbre)."
        
        interpretation = f"{interpretation} {price_position}"
        
        # Alineación
        if signal_direction == SignalDirection.LONG:
            if ema_fast > ema_slow:
                alignment = "confirma"
                context = "✅ Cruce alcista de EMAs CONFIRMA señal LONG."
            else:
                alignment = "contradice"
                context = "⚠️ EMAs en cruce bajista CONTRADICEN señal LONG."
        elif signal_direction == SignalDirection.SHORT:
            if ema_fast < ema_slow:
                alignment = "confirma"
                context = "✅ Cruce bajista de EMAs CONFIRMA señal SHORT."
            else:
                alignment = "contradice"
                context = "⚠️ EMAs en cruce alcista CONTRADICEN señal SHORT."
        else:
            alignment = "neutral"
            context = "➖ Señal NEUTRAL - EMAs no requieren alineación específica."
        
        return IndicatorExplanation(
            name="EMA Crossover",
            value=diff,
            interpretation=interpretation,
            signal_alignment=alignment,
            context=context
        )
    
    def _explain_stochastic(self, name: str, value: float, signal_direction: SignalDirection,
                           indicators: Dict[str, float]) -> IndicatorExplanation:
        """Explicación contextual del Estocástico"""
        stoch_k = indicators.get("stoch_k", value if name == "stoch_k" else 0)
        stoch_d = indicators.get("stoch_d", value if name == "stoch_d" else 0)
        thresholds = INDICATOR_THRESHOLDS["stochastic"]
        
        # Zona actual
        if stoch_k <= thresholds["oversold_extreme"]:
            zone = "SOBREVENTA EXTREMA"
            interpretation = f"Stochastic %K={stoch_k:.2f}, %D={stoch_d:.2f}: {zone}. Condición extrema que suele preceder rebotes."
        elif stoch_k <= thresholds["oversold"]:
            zone = "SOBREVENTA"
            interpretation = f"Stochastic %K={stoch_k:.2f}, %D={stoch_d:.2f}: {zone}. Presión vendedora elevada."
        elif stoch_k >= thresholds["overbought_extreme"]:
            zone = "SOBRECOMPRA EXTREMA"
            interpretation = f"Stochastic %K={stoch_k:.2f}, %D={stoch_d:.2f}: {zone}. Condición extrema que suele preceder correcciones."
        elif stoch_k >= thresholds["overbought"]:
            zone = "SOBRECOMPRA"
            interpretation = f"Stochastic %K={stoch_k:.2f}, %D={stoch_d:.2f}: {zone}. Presión compradora elevada."
        else:
            zone = "NEUTRAL"
            interpretation = f"Stochastic %K={stoch_k:.2f}, %D={stoch_d:.2f}: {zone}. Sin condiciones extremas."
        
        # Cruce K/D
        if stoch_k > stoch_d:
            cross = "Cruce alcista (%K > %D)."
        elif stoch_k < stoch_d:
            cross = "Cruce bajista (%K < %D)."
        else:
            cross = "Sin cruce definido."
        
        interpretation = f"{interpretation} {cross}"
        
        # Alineación
        if signal_direction == SignalDirection.LONG:
            if zone in ["SOBREVENTA", "SOBREVENTA EXTREMA"] and stoch_k > stoch_d:
                alignment = "confirma"
                context = "✅ Stochastic en sobreventa con cruce alcista CONFIRMA LONG."
            elif zone in ["SOBRECOMPRA", "SOBRECOMPRA EXTREMA"]:
                alignment = "contradice"
                context = "⚠️ Stochastic en sobrecompra CONTRADICE LONG."
            else:
                alignment = "neutral"
                context = "➖ Stochastic neutral, confirmación limitada."
        elif signal_direction == SignalDirection.SHORT:
            if zone in ["SOBRECOMPRA", "SOBRECOMPRA EXTREMA"] and stoch_k < stoch_d:
                alignment = "confirma"
                context = "✅ Stochastic en sobrecompra con cruce bajista CONFIRMA SHORT."
            elif zone in ["SOBREVENTA", "SOBREVENTA EXTREMA"]:
                alignment = "contradice"
                context = "⚠️ Stochastic en sobreventa CONTRADICE SHORT."
            else:
                alignment = "neutral"
                context = "➖ Stochastic neutral, confirmación limitada."
        else:
            alignment = "neutral"
            context = "➖ Señal NEUTRAL."
        
        return IndicatorExplanation(
            name="Stochastic",
            value=stoch_k,
            interpretation=interpretation,
            signal_alignment=alignment,
            context=context
        )
    
    def _explain_atr(self, value: float, signal_direction: SignalDirection,
                    indicators: Dict[str, float]) -> IndicatorExplanation:
        """Explicación contextual del ATR"""
        close = indicators.get("close", 1)
        atr_pct = (value / close * 100) if close != 0 else 0
        
        # Clasificar volatilidad
        if atr_pct < 0.5:
            volatility = "MUY BAJA"
            interpretation = f"ATR={value:.5f} ({atr_pct:.2f}% del precio): Volatilidad {volatility}. Movimientos pequeños esperados."
        elif atr_pct < 1.0:
            volatility = "BAJA"
            interpretation = f"ATR={value:.5f} ({atr_pct:.2f}% del precio): Volatilidad {volatility}. Mercado relativamente tranquilo."
        elif atr_pct < 2.0:
            volatility = "MODERADA"
            interpretation = f"ATR={value:.5f} ({atr_pct:.2f}% del precio): Volatilidad {volatility}. Condiciones normales de trading."
        elif atr_pct < 3.0:
            volatility = "ALTA"
            interpretation = f"ATR={value:.5f} ({atr_pct:.2f}% del precio): Volatilidad {volatility}. Movimientos significativos posibles."
        else:
            volatility = "MUY ALTA"
            interpretation = f"ATR={value:.5f} ({atr_pct:.2f}% del precio): Volatilidad {volatility}. PRECAUCIÓN: movimientos extremos posibles."
        
        # El ATR no tiene alineación directa, pero afecta gestión de riesgo
        sl_distance = value * self.risk_config.get("atr_sl_mult", 1.5)
        tp_distance = value * self.risk_config.get("atr_tp_mult", 2.0)
        
        context = f"📊 SL calculado a {sl_distance:.5f} ({sl_distance/close*100:.2f}%), TP a {tp_distance:.5f} ({tp_distance/close*100:.2f}%)."
        
        return IndicatorExplanation(
            name="ATR (Volatilidad)",
            value=value,
            interpretation=interpretation,
            signal_alignment="neutral",  # ATR no tiene alineación direccional
            context=context
        )
    
    def _explain_cci(self, value: float, signal_direction: SignalDirection,
                    indicators: Dict[str, float]) -> IndicatorExplanation:
        """Explicación contextual del CCI"""
        thresholds = INDICATOR_THRESHOLDS["cci"]
        
        if value <= thresholds["oversold_extreme"]:
            zone = "SOBREVENTA EXTREMA"
            interpretation = f"CCI={value:.2f}: {zone}. Desviación extrema a la baja, posible rebote."
        elif value <= thresholds["oversold"]:
            zone = "SOBREVENTA"
            interpretation = f"CCI={value:.2f}: {zone}. Precio significativamente por debajo de su media."
        elif value >= thresholds["overbought_extreme"]:
            zone = "SOBRECOMPRA EXTREMA"
            interpretation = f"CCI={value:.2f}: {zone}. Desviación extrema al alza, posible corrección."
        elif value >= thresholds["overbought"]:
            zone = "SOBRECOMPRA"
            interpretation = f"CCI={value:.2f}: {zone}. Precio significativamente por encima de su media."
        else:
            zone = "NEUTRAL"
            interpretation = f"CCI={value:.2f}: {zone}. Precio cerca de su media típica."
        
        if signal_direction == SignalDirection.LONG:
            if zone in ["SOBREVENTA", "SOBREVENTA EXTREMA"]:
                alignment = "confirma"
                context = "✅ CCI en sobreventa CONFIRMA potencial alcista."
            elif zone in ["SOBRECOMPRA", "SOBRECOMPRA EXTREMA"]:
                alignment = "contradice"
                context = "⚠️ CCI en sobrecompra CONTRADICE LONG."
            else:
                alignment = "neutral"
                context = "➖ CCI neutral."
        elif signal_direction == SignalDirection.SHORT:
            if zone in ["SOBRECOMPRA", "SOBRECOMPRA EXTREMA"]:
                alignment = "confirma"
                context = "✅ CCI en sobrecompra CONFIRMA potencial bajista."
            elif zone in ["SOBREVENTA", "SOBREVENTA EXTREMA"]:
                alignment = "contradice"
                context = "⚠️ CCI en sobreventa CONTRADICE SHORT."
            else:
                alignment = "neutral"
                context = "➖ CCI neutral."
        else:
            alignment = "neutral"
            context = "➖ Señal NEUTRAL."
        
        return IndicatorExplanation(
            name="CCI",
            value=value,
            interpretation=interpretation,
            signal_alignment=alignment,
            context=context
        )
    
    def _explain_momentum(self, value: float, signal_direction: SignalDirection,
                         indicators: Dict[str, float]) -> IndicatorExplanation:
        """Explicación contextual del Momentum"""
        close = indicators.get("close", 1)
        mom_pct = (value / close * 100) if close != 0 else 0
        
        if value > 0:
            direction = "ALCISTA"
            interpretation = f"Momentum={value:.5f} ({mom_pct:.2f}%): {direction}. El precio actual es MAYOR que hace N períodos."
        elif value < 0:
            direction = "BAJISTA"
            interpretation = f"Momentum={value:.5f} ({mom_pct:.2f}%): {direction}. El precio actual es MENOR que hace N períodos."
        else:
            direction = "NEUTRAL"
            interpretation = f"Momentum={value:.5f}: {direction}. Sin cambio significativo."
        
        if signal_direction == SignalDirection.LONG:
            if value > 0:
                alignment = "confirma"
                context = "✅ Momentum positivo CONFIRMA LONG."
            else:
                alignment = "contradice"
                context = "⚠️ Momentum negativo CONTRADICE LONG."
        elif signal_direction == SignalDirection.SHORT:
            if value < 0:
                alignment = "confirma"
                context = "✅ Momentum negativo CONFIRMA SHORT."
            else:
                alignment = "contradice"
                context = "⚠️ Momentum positivo CONTRADICE SHORT."
        else:
            alignment = "neutral"
            context = "➖ Señal NEUTRAL."
        
        return IndicatorExplanation(
            name="Momentum",
            value=value,
            interpretation=interpretation,
            signal_alignment=alignment,
            context=context
        )
    
    def _explain_bollinger(self, name: str, value: float, signal_direction: SignalDirection,
                          indicators: Dict[str, float]) -> IndicatorExplanation:
        """Explicación contextual de Bandas de Bollinger"""
        bb_upper = indicators.get("bb_upper", 0)
        bb_lower = indicators.get("bb_lower", 0)
        bb_mid = indicators.get("bb_mid", 0)
        close = indicators.get("close", 0)
        
        band_width = bb_upper - bb_lower
        band_width_pct = (band_width / bb_mid * 100) if bb_mid != 0 else 0
        
        # Posición del precio respecto a las bandas
        if close > bb_upper:
            position = "POR ENCIMA de banda superior"
            interpretation = f"Precio ({close:.5f}) {position}. Ruptura alcista o sobrecompra potencial."
        elif close < bb_lower:
            position = "POR DEBAJO de banda inferior"
            interpretation = f"Precio ({close:.5f}) {position}. Ruptura bajista o sobreventa potencial."
        elif close > bb_mid:
            position = "entre media y banda superior"
            interpretation = f"Precio ({close:.5f}) {position}. Momentum ligeramente alcista."
        elif close < bb_mid:
            position = "entre media y banda inferior"
            interpretation = f"Precio ({close:.5f}) {position}. Momentum ligeramente bajista."
        else:
            position = "en la media"
            interpretation = f"Precio ({close:.5f}) {position}. Sin sesgo direccional."
        
        interpretation = f"{interpretation} Ancho de banda: {band_width_pct:.2f}%."
        
        if signal_direction == SignalDirection.LONG:
            if close > bb_upper:
                alignment = "confirma"
                context = "✅ Ruptura de banda superior CONFIRMA momentum alcista fuerte."
            elif close < bb_lower:
                alignment = "contradice"
                context = "⚠️ Precio bajo banda inferior CONTRADICE LONG en estrategia de breakout."
            else:
                alignment = "neutral"
                context = "➖ Precio dentro de bandas, sin ruptura."
        elif signal_direction == SignalDirection.SHORT:
            if close < bb_lower:
                alignment = "confirma"
                context = "✅ Ruptura de banda inferior CONFIRMA momentum bajista fuerte."
            elif close > bb_upper:
                alignment = "contradice"
                context = "⚠️ Precio sobre banda superior CONTRADICE SHORT en estrategia de breakout."
            else:
                alignment = "neutral"
                context = "➖ Precio dentro de bandas, sin ruptura."
        else:
            alignment = "neutral"
            context = "➖ Señal NEUTRAL."
        
        return IndicatorExplanation(
            name="Bollinger Bands",
            value=band_width,
            interpretation=interpretation,
            signal_alignment=alignment,
            context=context
        )
    
    # ========================================================================
    # 2. SISTEMA DE ADVERTENCIAS DE RIESGO
    # ========================================================================
    
    def generate_risk_warnings(self, indicators: Dict[str, float], 
                               signal_direction: SignalDirection,
                               strategy_votes: List[Dict]) -> List[RiskWarning]:
        """
        Genera advertencias de riesgo automáticas basadas en las condiciones del mercado.
        
        Args:
            indicators: Diccionario con todos los indicadores técnicos
            signal_direction: Dirección de la señal (LONG/SHORT/NEUTRAL)
            strategy_votes: Lista de votos de cada estrategia
        
        Returns:
            Lista de RiskWarning ordenada por nivel de riesgo (más crítico primero)
        """
        warnings = []
        
        # 1. ADVERTENCIA: ADX bajo (sin tendencia)
        adx = indicators.get("adx", 25)
        if adx < INDICATOR_THRESHOLDS["adx"]["no_trend"]:
            warnings.append(RiskWarning(
                code="ADX_NO_TREND",
                level=RiskLevel.HIGH,
                title="⚠️ AUSENCIA DE TENDENCIA",
                description=f"ADX en {adx:.2f} (umbral: 20). El mercado está en rango/consolidación. Las estrategias de seguimiento de tendencia tienen ALTA probabilidad de generar señales falsas en estas condiciones.",
                recommendation="Considerar NO OPERAR o reducir tamaño de posición al 50%. Alternativa: usar estrategias de reversión a la media.",
                indicator_values={"adx": adx}
            ))
        elif adx < INDICATOR_THRESHOLDS["adx"]["weak_trend"]:
            warnings.append(RiskWarning(
                code="ADX_WEAK_TREND",
                level=RiskLevel.MEDIUM,
                title="⚡ TENDENCIA DÉBIL",
                description=f"ADX en {adx:.2f} (umbral fuerte: 25). Tendencia presente pero débil.",
                recommendation="Reducir expectativas de profit. Considerar Take Profit más cercano.",
                indicator_values={"adx": adx}
            ))
        
        # 2. ADVERTENCIA: Consenso mínimo
        if strategy_votes:
            total_votes = len(strategy_votes)
            agreeing_votes = sum(1 for v in strategy_votes 
                                if v.get("vote") == signal_direction.value)
            
            if agreeing_votes <= total_votes // 2:
                warnings.append(RiskWarning(
                    code="LOW_CONSENSUS",
                    level=RiskLevel.MEDIUM,
                    title="⚡ CONSENSO MÍNIMO",
                    description=f"Solo {agreeing_votes} de {total_votes} estrategias votan {signal_direction.value}. Señal con respaldo limitado.",
                    recommendation="Considerar reducir tamaño de posición o esperar confirmación adicional.",
                    indicator_values={"agreeing_votes": agreeing_votes, "total_votes": total_votes}
                ))
        
        # 3. ADVERTENCIA: RSI contradice señal
        rsi = indicators.get("rsi", 50)
        if signal_direction == SignalDirection.LONG and rsi > 70:
            warnings.append(RiskWarning(
                code="RSI_CONTRADICTS_LONG",
                level=RiskLevel.HIGH,
                title="⚠️ RSI CONTRADICE SEÑAL",
                description=f"RSI en {rsi:.2f} (sobrecompra >70) mientras la señal es LONG. Alta probabilidad de corrección inminente.",
                recommendation="EVITAR entrada o esperar pullback a zona neutral (RSI 40-60).",
                indicator_values={"rsi": rsi}
            ))
        elif signal_direction == SignalDirection.SHORT and rsi < 30:
            warnings.append(RiskWarning(
                code="RSI_CONTRADICTS_SHORT",
                level=RiskLevel.HIGH,
                title="⚠️ RSI CONTRADICE SEÑAL",
                description=f"RSI en {rsi:.2f} (sobreventa <30) mientras la señal es SHORT. Alta probabilidad de rebote inminente.",
                recommendation="EVITAR entrada o esperar rebote a zona neutral (RSI 40-60).",
                indicator_values={"rsi": rsi}
            ))
        
        # 4. ADVERTENCIA: Divergencia entre indicadores de tendencia
        ema_fast = indicators.get("ema_fast", 0)
        ema_slow = indicators.get("ema_slow", 0)
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        
        ema_bullish = ema_fast > ema_slow
        macd_bullish = macd > macd_signal
        
        if signal_direction == SignalDirection.LONG:
            if ema_bullish != macd_bullish:
                warnings.append(RiskWarning(
                    code="INDICATOR_DIVERGENCE",
                    level=RiskLevel.MEDIUM,
                    title="⚡ DIVERGENCIA DE INDICADORES",
                    description=f"EMA indica {'alcista' if ema_bullish else 'bajista'} mientras MACD indica {'alcista' if macd_bullish else 'bajista'}. Los indicadores de tendencia no están alineados.",
                    recommendation="Esperar convergencia de indicadores antes de entrar.",
                    indicator_values={"ema_bullish": ema_bullish, "macd_bullish": macd_bullish}
                ))
        elif signal_direction == SignalDirection.SHORT:
            if ema_bullish != macd_bullish:
                warnings.append(RiskWarning(
                    code="INDICATOR_DIVERGENCE",
                    level=RiskLevel.MEDIUM,
                    title="⚡ DIVERGENCIA DE INDICADORES",
                    description=f"EMA indica {'alcista' if ema_bullish else 'bajista'} mientras MACD indica {'alcista' if macd_bullish else 'bajista'}. Los indicadores de tendencia no están alineados.",
                    recommendation="Esperar convergencia de indicadores antes de entrar.",
                    indicator_values={"ema_bullish": ema_bullish, "macd_bullish": macd_bullish}
                ))
        
        # 5. ADVERTENCIA: Volatilidad extrema
        atr = indicators.get("atr", 0)
        close = indicators.get("close", 1)
        atr_pct = (atr / close * 100) if close != 0 else 0
        
        if atr_pct > 3.0:
            warnings.append(RiskWarning(
                code="HIGH_VOLATILITY",
                level=RiskLevel.HIGH,
                title="⚠️ VOLATILIDAD EXTREMA",
                description=f"ATR representa {atr_pct:.2f}% del precio. Volatilidad muy elevada aumenta riesgo de Stop Loss prematuro.",
                recommendation="Ampliar Stop Loss a 2x ATR o reducir tamaño de posición significativamente.",
                indicator_values={"atr": atr, "atr_pct": atr_pct}
            ))
        elif atr_pct < 0.3:
            warnings.append(RiskWarning(
                code="LOW_VOLATILITY",
                level=RiskLevel.LOW,
                title="📉 VOLATILIDAD MUY BAJA",
                description=f"ATR representa solo {atr_pct:.2f}% del precio. Potencial de profit limitado.",
                recommendation="Ajustar expectativas de beneficio o buscar activos más volátiles.",
                indicator_values={"atr": atr, "atr_pct": atr_pct}
            ))
        
        # 6. ADVERTENCIA: Stochastic en zona extrema contradice señal
        stoch_k = indicators.get("stoch_k", 50)
        if signal_direction == SignalDirection.LONG and stoch_k > 80:
            warnings.append(RiskWarning(
                code="STOCH_OVERBOUGHT_LONG",
                level=RiskLevel.MEDIUM,
                title="⚡ STOCHASTIC EN SOBRECOMPRA",
                description=f"Stochastic %K en {stoch_k:.2f} (>80) sugiere agotamiento alcista. Entrada LONG arriesgada.",
                recommendation="Esperar retroceso de Stochastic a zona neutral (<70).",
                indicator_values={"stoch_k": stoch_k}
            ))
        elif signal_direction == SignalDirection.SHORT and stoch_k < 20:
            warnings.append(RiskWarning(
                code="STOCH_OVERSOLD_SHORT",
                level=RiskLevel.MEDIUM,
                title="⚡ STOCHASTIC EN SOBREVENTA",
                description=f"Stochastic %K en {stoch_k:.2f} (<20) sugiere agotamiento bajista. Entrada SHORT arriesgada.",
                recommendation="Esperar rebote de Stochastic a zona neutral (>30).",
                indicator_values={"stoch_k": stoch_k}
            ))
        
        # 7. ADVERTENCIA: Señal contra tendencia principal
        plus_di = indicators.get("plus_di", 0)
        minus_di = indicators.get("minus_di", 0)
        
        if signal_direction == SignalDirection.LONG and minus_di > plus_di and adx > 25:
            warnings.append(RiskWarning(
                code="AGAINST_TREND_LONG",
                level=RiskLevel.CRITICAL,
                title="🚨 OPERACIÓN CONTRA TENDENCIA",
                description=f"Señal LONG pero tendencia es BAJISTA (-DI={minus_di:.2f} > +DI={plus_di:.2f}, ADX={adx:.2f}). Operar contra tendencia tiene baja probabilidad de éxito.",
                recommendation="EVITAR esta operación o usarla solo como scalping con TP muy cercano.",
                indicator_values={"plus_di": plus_di, "minus_di": minus_di, "adx": adx}
            ))
        elif signal_direction == SignalDirection.SHORT and plus_di > minus_di and adx > 25:
            warnings.append(RiskWarning(
                code="AGAINST_TREND_SHORT",
                level=RiskLevel.CRITICAL,
                title="🚨 OPERACIÓN CONTRA TENDENCIA",
                description=f"Señal SHORT pero tendencia es ALCISTA (+DI={plus_di:.2f} > -DI={minus_di:.2f}, ADX={adx:.2f}). Operar contra tendencia tiene baja probabilidad de éxito.",
                recommendation="EVITAR esta operación o usarla solo como scalping con TP muy cercano.",
                indicator_values={"plus_di": plus_di, "minus_di": minus_di, "adx": adx}
            ))
        
        # Ordenar por nivel de riesgo (crítico primero)
        risk_order = {RiskLevel.CRITICAL: 0, RiskLevel.HIGH: 1, RiskLevel.MEDIUM: 2, RiskLevel.LOW: 3}
        warnings.sort(key=lambda w: risk_order.get(w.level, 4))
        
        return warnings
    
    # ========================================================================
    # 3. EXPLICACIÓN DE VOTOS INDIVIDUALES DE ESTRATEGIAS
    # ========================================================================
    
    def explain_strategy_vote(self, strategy_name: str, vote: str, score: float,
                             indicators: Dict[str, float], cfg: dict) -> StrategyVoteExplanation:
        """
        Genera una explicación detallada del voto de una estrategia específica.
        
        Args:
            strategy_name: Nombre de la estrategia
            vote: Voto emitido ("LONG", "SHORT", "NEUTRAL")
            score: Score de confianza
            indicators: Diccionario con indicadores técnicos
            cfg: Configuración del sistema
        
        Returns:
            StrategyVoteExplanation con el detalle completo
        """
        
        if strategy_name == "ema_crossover":
            return self._explain_ema_crossover_vote(vote, score, indicators, cfg)
        elif strategy_name == "rsi_reversal":
            return self._explain_rsi_reversal_vote(vote, score, indicators, cfg)
        elif strategy_name == "macd_crossover":
            return self._explain_macd_crossover_vote(vote, score, indicators, cfg)
        elif strategy_name == "bollinger_breakout":
            return self._explain_bollinger_breakout_vote(vote, score, indicators, cfg)
        else:
            return StrategyVoteExplanation(
                strategy_name=strategy_name,
                vote=vote,
                score=score,
                reasoning="Estrategia no documentada",
                conditions_met=[],
                conditions_failed=[],
                key_indicators={}
            )
    
    def _explain_ema_crossover_vote(self, vote: str, score: float, 
                                    indicators: Dict[str, float], cfg: dict) -> StrategyVoteExplanation:
        """Explicación detallada de EMA Crossover"""
        st = cfg.get("strategy", {})
        
        ema_fast = indicators.get("ema_fast", 0)
        ema_slow = indicators.get("ema_slow", 0)
        rsi = indicators.get("rsi", 50)
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        atr = indicators.get("atr", 0)
        close = indicators.get("close", 1)
        
        atr_pct = (atr / close) if close != 0 else 0
        min_atr_pct = st.get("min_atr_pct", 0.001)
        rsi_long_min = st.get("rsi_long_min", 40)
        rsi_short_max = st.get("rsi_short_max", 60)
        
        conditions_met = []
        conditions_failed = []
        
        # Evaluar condiciones
        if ema_fast > ema_slow:
            conditions_met.append(f"✅ Cruce alcista: EMA rápida ({ema_fast:.5f}) > EMA lenta ({ema_slow:.5f})")
            ema_direction = "alcista"
        elif ema_fast < ema_slow:
            conditions_met.append(f"✅ Cruce bajista: EMA rápida ({ema_fast:.5f}) < EMA lenta ({ema_slow:.5f})")
            ema_direction = "bajista"
        else:
            conditions_failed.append(f"❌ Sin cruce: EMAs convergentes ({ema_fast:.5f} ≈ {ema_slow:.5f})")
            ema_direction = "neutral"
        
        if vote == "LONG":
            if rsi >= rsi_long_min:
                conditions_met.append(f"✅ RSI favorable: {rsi:.2f} ≥ {rsi_long_min} (umbral LONG)")
            else:
                conditions_failed.append(f"❌ RSI insuficiente: {rsi:.2f} < {rsi_long_min} (umbral LONG)")
            
            if macd >= macd_signal:
                conditions_met.append(f"✅ MACD confirma: {macd:.6f} ≥ señal {macd_signal:.6f}")
            else:
                conditions_failed.append(f"❌ MACD no confirma: {macd:.6f} < señal {macd_signal:.6f}")
                
        elif vote == "SHORT":
            if rsi <= rsi_short_max:
                conditions_met.append(f"✅ RSI favorable: {rsi:.2f} ≤ {rsi_short_max} (umbral SHORT)")
            else:
                conditions_failed.append(f"❌ RSI insuficiente: {rsi:.2f} > {rsi_short_max} (umbral SHORT)")
            
            if macd <= macd_signal:
                conditions_met.append(f"✅ MACD confirma: {macd:.6f} ≤ señal {macd_signal:.6f}")
            else:
                conditions_failed.append(f"❌ MACD no confirma: {macd:.6f} > señal {macd_signal:.6f}")
        
        # Volatilidad
        if atr_pct >= min_atr_pct:
            conditions_met.append(f"✅ Volatilidad suficiente: ATR {atr_pct*100:.3f}% ≥ {min_atr_pct*100:.3f}%")
        else:
            conditions_failed.append(f"❌ Volatilidad insuficiente: ATR {atr_pct*100:.3f}% < {min_atr_pct*100:.3f}%")
        
        # Generar razonamiento
        if vote == "NEUTRAL":
            reasoning = f"La estrategia EMA Crossover votó NEUTRAL porque no se detectó un cruce claro de EMAs o las condiciones de confirmación (RSI, MACD, volatilidad) no se cumplieron simultáneamente."
        else:
            met_count = len(conditions_met)
            total = met_count + len(conditions_failed)
            reasoning = f"La estrategia EMA Crossover votó {vote} basándose en el cruce {ema_direction} de medias móviles. Se cumplieron {met_count}/{total} condiciones de validación. Score de confianza: {score:.2%}."
        
        return StrategyVoteExplanation(
            strategy_name="EMA Crossover",
            vote=vote,
            score=score,
            reasoning=reasoning,
            conditions_met=conditions_met,
            conditions_failed=conditions_failed,
            key_indicators={
                "EMA Fast": f"{ema_fast:.5f}",
                "EMA Slow": f"{ema_slow:.5f}",
                "RSI": f"{rsi:.2f}",
                "MACD": f"{macd:.6f}",
                "ATR %": f"{atr_pct*100:.3f}%"
            }
        )
    
    def _explain_rsi_reversal_vote(self, vote: str, score: float,
                                   indicators: Dict[str, float], cfg: dict) -> StrategyVoteExplanation:
        """Explicación detallada de RSI Reversal"""
        st_cfg = cfg.get("strategies", {}).get("rsi_reversal", {})
        
        rsi = indicators.get("rsi", 50)
        stoch_k = indicators.get("stoch_k", 50)
        stoch_d = indicators.get("stoch_d", 50)
        williams_r = indicators.get("williams_r", -50)
        adx = indicators.get("adx", 25)
        
        rsi_oversold = st_cfg.get("rsi_oversold", 30)
        rsi_overbought = st_cfg.get("rsi_overbought", 70)
        stoch_oversold = st_cfg.get("stoch_oversold", 20)
        stoch_overbought = st_cfg.get("stoch_overbought", 80)
        adx_max = st_cfg.get("adx_max", 40)
        
        conditions_met = []
        conditions_failed = []
        
        if vote == "LONG":
            # RSI saliendo de sobreventa
            if rsi > rsi_oversold and rsi < 50:
                conditions_met.append(f"✅ RSI saliendo de sobreventa: {rsi:.2f} (umbral: {rsi_oversold})")
            else:
                conditions_failed.append(f"❌ RSI no en zona de reversión alcista: {rsi:.2f}")
            
            # Stochastic
            if stoch_k < stoch_oversold:
                conditions_met.append(f"✅ Stochastic en sobreventa: %K={stoch_k:.2f} < {stoch_oversold}")
            else:
                conditions_failed.append(f"❌ Stochastic no confirma sobreventa: %K={stoch_k:.2f}")
            
            # Cruce Stochastic
            if stoch_k > stoch_d:
                conditions_met.append(f"✅ Cruce alcista Stochastic: %K={stoch_k:.2f} > %D={stoch_d:.2f}")
            else:
                conditions_failed.append(f"❌ Sin cruce alcista Stochastic: %K={stoch_k:.2f} ≤ %D={stoch_d:.2f}")
                
        elif vote == "SHORT":
            # RSI saliendo de sobrecompra
            if rsi < rsi_overbought and rsi > 50:
                conditions_met.append(f"✅ RSI saliendo de sobrecompra: {rsi:.2f} (umbral: {rsi_overbought})")
            else:
                conditions_failed.append(f"❌ RSI no en zona de reversión bajista: {rsi:.2f}")
            
            # Stochastic
            if stoch_k > stoch_overbought:
                conditions_met.append(f"✅ Stochastic en sobrecompra: %K={stoch_k:.2f} > {stoch_overbought}")
            else:
                conditions_failed.append(f"❌ Stochastic no confirma sobrecompra: %K={stoch_k:.2f}")
            
            # Cruce Stochastic
            if stoch_k < stoch_d:
                conditions_met.append(f"✅ Cruce bajista Stochastic: %K={stoch_k:.2f} < %D={stoch_d:.2f}")
            else:
                conditions_failed.append(f"❌ Sin cruce bajista Stochastic: %K={stoch_k:.2f} ≥ %D={stoch_d:.2f}")
        
        # ADX (reversión funciona mejor en rangos)
        if adx < adx_max:
            conditions_met.append(f"✅ ADX favorable para reversión: {adx:.2f} < {adx_max} (mercado en rango)")
        else:
            conditions_failed.append(f"❌ ADX muy alto para reversión: {adx:.2f} ≥ {adx_max} (tendencia fuerte)")
        
        if vote == "NEUTRAL":
            reasoning = f"RSI Reversal votó NEUTRAL porque no se detectaron condiciones claras de sobreventa/sobrecompra con confirmación de Stochastic, o el ADX indica tendencia fuerte donde las reversiones son menos efectivas."
        else:
            met_count = len(conditions_met)
            total = met_count + len(conditions_failed)
            reasoning = f"RSI Reversal votó {vote} detectando una posible reversión a la media. Se cumplieron {met_count}/{total} condiciones. Esta estrategia es más efectiva en mercados en rango (ADX < {adx_max})."
        
        return StrategyVoteExplanation(
            strategy_name="RSI Reversal",
            vote=vote,
            score=score,
            reasoning=reasoning,
            conditions_met=conditions_met,
            conditions_failed=conditions_failed,
            key_indicators={
                "RSI": f"{rsi:.2f}",
                "Stoch %K": f"{stoch_k:.2f}",
                "Stoch %D": f"{stoch_d:.2f}",
                "Williams %R": f"{williams_r:.2f}",
                "ADX": f"{adx:.2f}"
            }
        )
    
    def _explain_macd_crossover_vote(self, vote: str, score: float,
                                     indicators: Dict[str, float], cfg: dict) -> StrategyVoteExplanation:
        """Explicación detallada de MACD Crossover"""
        st_cfg = cfg.get("strategies", {}).get("macd_crossover", {})
        
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        macd_hist = indicators.get("macd_hist", 0)
        cci = indicators.get("cci", 0)
        momentum = indicators.get("momentum", 0)
        roc = indicators.get("roc", 0)
        adx = indicators.get("adx", 25)
        
        cci_long = st_cfg.get("cci_long_threshold", 0)
        cci_short = st_cfg.get("cci_short_threshold", 0)
        adx_min = st_cfg.get("adx_min", 20)
        roc_threshold = st_cfg.get("roc_threshold", 0)
        
        conditions_met = []
        conditions_failed = []
        
        if vote == "LONG":
            if macd > macd_signal:
                conditions_met.append(f"✅ MACD sobre señal: {macd:.6f} > {macd_signal:.6f}")
            else:
                conditions_failed.append(f"❌ MACD bajo señal: {macd:.6f} ≤ {macd_signal:.6f}")
            
            if macd_hist > 0:
                conditions_met.append(f"✅ Histograma positivo: {macd_hist:.6f}")
            else:
                conditions_failed.append(f"❌ Histograma negativo: {macd_hist:.6f}")
            
            if cci > cci_long:
                conditions_met.append(f"✅ CCI alcista: {cci:.2f} > {cci_long}")
            else:
                conditions_failed.append(f"❌ CCI no confirma: {cci:.2f} ≤ {cci_long}")
            
            if momentum > 0:
                conditions_met.append(f"✅ Momentum positivo: {momentum:.5f}")
            else:
                conditions_failed.append(f"❌ Momentum negativo: {momentum:.5f}")
                
        elif vote == "SHORT":
            if macd < macd_signal:
                conditions_met.append(f"✅ MACD bajo señal: {macd:.6f} < {macd_signal:.6f}")
            else:
                conditions_failed.append(f"❌ MACD sobre señal: {macd:.6f} ≥ {macd_signal:.6f}")
            
            if macd_hist < 0:
                conditions_met.append(f"✅ Histograma negativo: {macd_hist:.6f}")
            else:
                conditions_failed.append(f"❌ Histograma positivo: {macd_hist:.6f}")
            
            if cci < cci_short:
                conditions_met.append(f"✅ CCI bajista: {cci:.2f} < {cci_short}")
            else:
                conditions_failed.append(f"❌ CCI no confirma: {cci:.2f} ≥ {cci_short}")
            
            if momentum < 0:
                conditions_met.append(f"✅ Momentum negativo: {momentum:.5f}")
            else:
                conditions_failed.append(f"❌ Momentum positivo: {momentum:.5f}")
        
        # ADX (MACD funciona mejor en tendencias)
        if adx > adx_min:
            conditions_met.append(f"✅ ADX confirma tendencia: {adx:.2f} > {adx_min}")
        else:
            conditions_failed.append(f"❌ ADX débil para MACD: {adx:.2f} ≤ {adx_min}")
        
        if vote == "NEUTRAL":
            reasoning = f"MACD Crossover votó NEUTRAL porque no se detectó cruce claro de MACD con su línea de señal, o los indicadores de confirmación (CCI, Momentum, ADX) no validaron la dirección."
        else:
            met_count = len(conditions_met)
            total = met_count + len(conditions_failed)
            reasoning = f"MACD Crossover votó {vote} basándose en el cruce de MACD con su línea de señal, confirmado por momentum. Se cumplieron {met_count}/{total} condiciones. Esta estrategia es más efectiva en tendencias (ADX > {adx_min})."
        
        return StrategyVoteExplanation(
            strategy_name="MACD Crossover",
            vote=vote,
            score=score,
            reasoning=reasoning,
            conditions_met=conditions_met,
            conditions_failed=conditions_failed,
            key_indicators={
                "MACD": f"{macd:.6f}",
                "Señal": f"{macd_signal:.6f}",
                "Histograma": f"{macd_hist:.6f}",
                "CCI": f"{cci:.2f}",
                "Momentum": f"{momentum:.5f}",
                "ADX": f"{adx:.2f}"
            }
        )
    
    def _explain_bollinger_breakout_vote(self, vote: str, score: float,
                                         indicators: Dict[str, float], cfg: dict) -> StrategyVoteExplanation:
        """Explicación detallada de Bollinger Breakout"""
        st_cfg = cfg.get("strategies", {}).get("bollinger_breakout", {})
        
        close = indicators.get("close", 0)
        bb_upper = indicators.get("bb_upper", 0)
        bb_lower = indicators.get("bb_lower", 0)
        bb_mid = indicators.get("bb_mid", 0)
        adx = indicators.get("adx", 25)
        plus_di = indicators.get("plus_di", 0)
        minus_di = indicators.get("minus_di", 0)
        momentum = indicators.get("momentum", 0)
        
        adx_strong = st_cfg.get("adx_strong", 25)
        mom_threshold = st_cfg.get("momentum_threshold", 0)
        
        conditions_met = []
        conditions_failed = []
        
        if vote == "LONG":
            if close > bb_upper:
                conditions_met.append(f"✅ Ruptura banda superior: Precio {close:.5f} > BB Upper {bb_upper:.5f}")
            else:
                conditions_failed.append(f"❌ Sin ruptura superior: Precio {close:.5f} ≤ BB Upper {bb_upper:.5f}")
            
            if plus_di > minus_di:
                conditions_met.append(f"✅ Direccionalidad alcista: +DI={plus_di:.2f} > -DI={minus_di:.2f}")
            else:
                conditions_failed.append(f"❌ Direccionalidad no confirma: +DI={plus_di:.2f} ≤ -DI={minus_di:.2f}")
            
            if momentum > mom_threshold:
                conditions_met.append(f"✅ Momentum positivo: {momentum:.5f} > {mom_threshold}")
            else:
                conditions_failed.append(f"❌ Momentum insuficiente: {momentum:.5f} ≤ {mom_threshold}")
                
        elif vote == "SHORT":
            if close < bb_lower:
                conditions_met.append(f"✅ Ruptura banda inferior: Precio {close:.5f} < BB Lower {bb_lower:.5f}")
            else:
                conditions_failed.append(f"❌ Sin ruptura inferior: Precio {close:.5f} ≥ BB Lower {bb_lower:.5f}")
            
            if minus_di > plus_di:
                conditions_met.append(f"✅ Direccionalidad bajista: -DI={minus_di:.2f} > +DI={plus_di:.2f}")
            else:
                conditions_failed.append(f"❌ Direccionalidad no confirma: -DI={minus_di:.2f} ≤ +DI={plus_di:.2f}")
            
            if momentum < -mom_threshold:
                conditions_met.append(f"✅ Momentum negativo: {momentum:.5f} < -{mom_threshold}")
            else:
                conditions_failed.append(f"❌ Momentum insuficiente: {momentum:.5f} ≥ -{mom_threshold}")
        
        # ADX para breakouts
        if adx > adx_strong:
            conditions_met.append(f"✅ ADX confirma tendencia fuerte: {adx:.2f} > {adx_strong}")
        else:
            conditions_failed.append(f"❌ ADX débil para breakout: {adx:.2f} ≤ {adx_strong}")
        
        if vote == "NEUTRAL":
            reasoning = f"Bollinger Breakout votó NEUTRAL porque el precio está dentro de las bandas de Bollinger, no hay ruptura significativa, o el ADX/Momentum no confirman un breakout válido."
        else:
            met_count = len(conditions_met)
            total = met_count + len(conditions_failed)
            reasoning = f"Bollinger Breakout votó {vote} detectando ruptura de banda con confirmación de direccionalidad. Se cumplieron {met_count}/{total} condiciones. Los breakouts son más confiables con ADX > {adx_strong}."
        
        return StrategyVoteExplanation(
            strategy_name="Bollinger Breakout",
            vote=vote,
            score=score,
            reasoning=reasoning,
            conditions_met=conditions_met,
            conditions_failed=conditions_failed,
            key_indicators={
                "Precio": f"{close:.5f}",
                "BB Upper": f"{bb_upper:.5f}",
                "BB Lower": f"{bb_lower:.5f}",
                "BB Mid": f"{bb_mid:.5f}",
                "+DI": f"{plus_di:.2f}",
                "-DI": f"{minus_di:.2f}",
                "ADX": f"{adx:.2f}"
            }
        )
    
    # ========================================================================
    # 4. GENERADOR DE EXPLICACIÓN COMPLETA
    # ========================================================================
    
    def generate_full_explanation(self, signal_data: Dict[str, Any], 
                                  indicators: Dict[str, float],
                                  strategy_votes: List[Dict]) -> SignalExplanation:
        """
        Genera una explicación completa y honesta de una señal de trading.
        
        Args:
            signal_data: Datos de la señal (timestamp, precio, sl, tp, etc.)
            indicators: Todos los indicadores técnicos
            strategy_votes: Votos de cada estrategia
        
        Returns:
            SignalExplanation con toda la información estructurada
        """
        # Determinar dirección
        signal_value = signal_data.get("signal", 0)
        if signal_value == 1:
            direction = SignalDirection.LONG
        elif signal_value == -1:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL
        
        price = signal_data.get("price", signal_data.get("close", 0))
        sl = signal_data.get("sl")
        tp = signal_data.get("tp")
        
        # Calcular ratio riesgo/beneficio
        rr_ratio = None
        if sl and tp and price:
            sl_distance = abs(price - sl)
            tp_distance = abs(tp - price)
            rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
        
        # 1. Generar explicaciones de indicadores
        indicator_explanations = []
        key_indicators = ["rsi", "macd", "adx", "ema_fast", "stoch_k", "atr", "cci", "momentum", "bb_upper"]
        for ind_name in key_indicators:
            if ind_name in indicators:
                expl = self.explain_indicator(ind_name, indicators[ind_name], direction, indicators)
                indicator_explanations.append(expl)
        
        # 2. Generar explicaciones de votos
        vote_explanations = []
        for vote_data in strategy_votes:
            vote_expl = self.explain_strategy_vote(
                strategy_name=vote_data.get("name", "unknown"),
                vote=vote_data.get("vote", "NEUTRAL"),
                score=vote_data.get("score", 0),
                indicators=indicators,
                cfg=self.cfg
            )
            vote_explanations.append(vote_expl)
        
        # 3. Generar advertencias de riesgo
        risk_warnings = self.generate_risk_warnings(indicators, direction, strategy_votes)
        
        # 4. Determinar nivel de riesgo global
        if any(w.level == RiskLevel.CRITICAL for w in risk_warnings):
            overall_risk = RiskLevel.CRITICAL
        elif any(w.level == RiskLevel.HIGH for w in risk_warnings):
            overall_risk = RiskLevel.HIGH
        elif any(w.level == RiskLevel.MEDIUM for w in risk_warnings):
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW
        
        # 5. Generar resumen ejecutivo
        confirming = sum(1 for e in indicator_explanations if e.signal_alignment == "confirma")
        contradicting = sum(1 for e in indicator_explanations if e.signal_alignment == "contradice")
        
        summary = self._generate_summary(direction, confirming, contradicting, 
                                         len(risk_warnings), overall_risk, strategy_votes)
        
        # 6. Generar recomendación final
        final_recommendation = self._generate_recommendation(direction, overall_risk, 
                                                             risk_warnings, rr_ratio)
        
        # 7. Disclaimer honesto
        honest_disclaimer = self._generate_disclaimer(overall_risk, len(risk_warnings))
        
        return SignalExplanation(
            timestamp=signal_data.get("timestamp", "N/A"),
            direction=direction,
            price=price,
            confidence_score=signal_data.get("score", 0),
            summary=summary,
            indicator_explanations=indicator_explanations,
            strategy_votes=vote_explanations,
            risk_warnings=risk_warnings,
            stop_loss=sl,
            take_profit=tp,
            risk_reward_ratio=rr_ratio,
            overall_risk_level=overall_risk,
            final_recommendation=final_recommendation,
            honest_disclaimer=honest_disclaimer
        )
    
    def _generate_summary(self, direction: SignalDirection, confirming: int, 
                         contradicting: int, warnings_count: int,
                         risk_level: RiskLevel, votes: List[Dict]) -> str:
        """Genera resumen ejecutivo de la señal"""
        if direction == SignalDirection.NEUTRAL:
            return f"📊 **SEÑAL NEUTRAL**: El sistema recomienda NO OPERAR. Las estrategias no alcanzaron consenso suficiente. {warnings_count} advertencias de riesgo detectadas."
        
        agreeing = sum(1 for v in votes if v.get("vote") == direction.value)
        total = len(votes)
        
        return (f"📊 **SEÑAL {direction.value}** | Consenso: {agreeing}/{total} estrategias | "
                f"Indicadores: {confirming} confirman, {contradicting} contradicen | "
                f"Advertencias: {warnings_count} | Nivel de riesgo: {risk_level.value.upper()}")
    
    def _generate_recommendation(self, direction: SignalDirection, risk_level: RiskLevel,
                                warnings: List[RiskWarning], rr_ratio: Optional[float]) -> str:
        """Genera recomendación final basada en el análisis"""
        if direction == SignalDirection.NEUTRAL:
            return "🚫 **RECOMENDACIÓN: NO OPERAR**. Esperar condiciones más claras."
        
        if risk_level == RiskLevel.CRITICAL:
            return (f"🚨 **RECOMENDACIÓN: EVITAR OPERACIÓN**. Se detectaron {len(warnings)} advertencias "
                   f"incluyendo riesgo CRÍTICO. La probabilidad de éxito es significativamente reducida.")
        
        if risk_level == RiskLevel.HIGH:
            return (f"⚠️ **RECOMENDACIÓN: ALTA PRECAUCIÓN**. Considerar reducir tamaño de posición al 50% "
                   f"o esperar mejores condiciones. Ratio R:R = 1:{rr_ratio:.2f}" if rr_ratio else "")
        
        if risk_level == RiskLevel.MEDIUM:
            return (f"⚡ **RECOMENDACIÓN: PROCEDER CON CAUTELA**. Condiciones aceptables pero no óptimas. "
                   f"Respetar estrictamente Stop Loss. Ratio R:R = 1:{rr_ratio:.2f}" if rr_ratio else "")
        
        return (f"✅ **RECOMENDACIÓN: CONDICIONES FAVORABLES**. Señal válida con riesgo controlado. "
               f"Ratio R:R = 1:{rr_ratio:.2f}" if rr_ratio else "")
    
    def _generate_disclaimer(self, risk_level: RiskLevel, warnings_count: int) -> str:
        """Genera disclaimer honesto sobre las limitaciones"""
        base = ("⚖️ **DISCLAIMER DE TRANSPARENCIA**: Este análisis es generado automáticamente "
               "y tiene limitaciones inherentes. ")
        
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            return (base + f"Se detectaron {warnings_count} factores de riesgo. El sistema identificó "
                   "INCERTIDUMBRES SIGNIFICATIVAS que podrían invalidar esta señal. "
                   "El rendimiento pasado NO garantiza resultados futuros. "
                   "Considere NO operar o reducir exposición sustancialmente.")
        
        return (base + "Los indicadores técnicos pueden generar señales falsas, especialmente "
               "en condiciones de mercado atípicas o eventos noticiosos. "
               "El rendimiento pasado NO garantiza resultados futuros. "
               "Opere solo con capital que pueda permitirse perder.")


# ============================================================================
# FUNCIÓN AUXILIAR PARA FORMATEAR EXPLICACIÓN COMO TEXTO
# ============================================================================

def format_explanation_as_text(explanation: SignalExplanation) -> str:
    """
    Convierte una SignalExplanation en texto legible para el usuario.
    """
    lines = []
    
    # Encabezado
    lines.append("=" * 80)
    lines.append(f"ANÁLISIS DE SEÑAL - {explanation.timestamp}")
    lines.append("=" * 80)
    
    # Resumen
    lines.append(f"\n{explanation.summary}")
    
    # Información básica
    lines.append(f"\n📈 **Dirección**: {explanation.direction.value}")
    lines.append(f"💰 **Precio**: {explanation.price:.5f}")
    lines.append(f"📊 **Confianza**: {explanation.confidence_score:.2%}")
    
    if explanation.stop_loss and explanation.take_profit:
        lines.append(f"🛡️ **Stop Loss**: {explanation.stop_loss:.5f}")
        lines.append(f"🎯 **Take Profit**: {explanation.take_profit:.5f}")
        if explanation.risk_reward_ratio:
            lines.append(f"⚖️ **Ratio R:R**: 1:{explanation.risk_reward_ratio:.2f}")
    
    # Advertencias de riesgo
    if explanation.risk_warnings:
        lines.append(f"\n{'='*40}")
        lines.append("⚠️ ADVERTENCIAS DE RIESGO")
        lines.append("=" * 40)
        for warning in explanation.risk_warnings:
            lines.append(f"\n{warning.title}")
            lines.append(f"   Nivel: {warning.level.value.upper()}")
            lines.append(f"   {warning.description}")
            lines.append(f"   💡 {warning.recommendation}")
    
    # Explicación de indicadores
    lines.append(f"\n{'='*40}")
    lines.append("📊 ANÁLISIS DE INDICADORES")
    lines.append("=" * 40)
    for ind_expl in explanation.indicator_explanations:
        lines.append(f"\n**{ind_expl.name}**: {ind_expl.value:.5f}")
        lines.append(f"   {ind_expl.interpretation}")
        lines.append(f"   {ind_expl.context}")
    
    # Votos de estrategias
    lines.append(f"\n{'='*40}")
    lines.append("🗳️ VOTOS DE ESTRATEGIAS")
    lines.append("=" * 40)
    for vote_expl in explanation.strategy_votes:
        emoji = "🟢" if vote_expl.vote == "LONG" else "🔴" if vote_expl.vote == "SHORT" else "⚪"
        lines.append(f"\n{emoji} **{vote_expl.strategy_name}**: {vote_expl.vote} (Score: {vote_expl.score:.2%})")
        lines.append(f"   {vote_expl.reasoning}")
        
        if vote_expl.conditions_met:
            lines.append("   Condiciones cumplidas:")
            for cond in vote_expl.conditions_met:
                lines.append(f"      {cond}")
        
        if vote_expl.conditions_failed:
            lines.append("   Condiciones NO cumplidas:")
            for cond in vote_expl.conditions_failed:
                lines.append(f"      {cond}")
    
    # Recomendación final
    lines.append(f"\n{'='*40}")
    lines.append("🎯 RECOMENDACIÓN FINAL")
    lines.append("=" * 40)
    lines.append(f"\n{explanation.final_recommendation}")
    
    # Disclaimer
    lines.append(f"\n{'='*40}")
    lines.append(f"{explanation.honest_disclaimer}")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def format_explanation_as_markdown(explanation: SignalExplanation) -> str:
    """
    Convierte una SignalExplanation en formato Markdown para el dashboard.
    """
    md = []
    
    # Encabezado
    md.append(f"# Análisis de Señal - {explanation.timestamp}")
    md.append(f"\n{explanation.summary}")
    
    # Info básica
    md.append("\n## 📈 Información de la Señal")
    md.append(f"- **Dirección**: {explanation.direction.value}")
    md.append(f"- **Precio**: {explanation.price:.5f}")
    md.append(f"- **Confianza**: {explanation.confidence_score:.2%}")
    
    if explanation.stop_loss and explanation.take_profit:
        md.append(f"- **Stop Loss**: {explanation.stop_loss:.5f}")
        md.append(f"- **Take Profit**: {explanation.take_profit:.5f}")
        if explanation.risk_reward_ratio:
            md.append(f"- **Ratio R:R**: 1:{explanation.risk_reward_ratio:.2f}")
    
    # Advertencias
    if explanation.risk_warnings:
        md.append("\n## ⚠️ Advertencias de Riesgo")
        for warning in explanation.risk_warnings:
            md.append(f"\n### {warning.title}")
            md.append(f"**Nivel**: {warning.level.value.upper()}")
            md.append(f"\n{warning.description}")
            md.append(f"\n💡 **Recomendación**: {warning.recommendation}")
    
    # Indicadores
    md.append("\n## 📊 Análisis de Indicadores")
    for ind_expl in explanation.indicator_explanations:
        md.append(f"\n### {ind_expl.name}")
        md.append(f"**Valor**: {ind_expl.value:.5f}")
        md.append(f"\n{ind_expl.interpretation}")
        md.append(f"\n{ind_expl.context}")
    
    # Votos
    md.append("\n## 🗳️ Votos de Estrategias")
    for vote_expl in explanation.strategy_votes:
        emoji = "🟢" if vote_expl.vote == "LONG" else "🔴" if vote_expl.vote == "SHORT" else "⚪"
        md.append(f"\n### {emoji} {vote_expl.strategy_name}")
        md.append(f"**Voto**: {vote_expl.vote} | **Score**: {vote_expl.score:.2%}")
        md.append(f"\n{vote_expl.reasoning}")
        
        if vote_expl.conditions_met or vote_expl.conditions_failed:
            md.append("\n| Condición | Estado |")
            md.append("|-----------|--------|")
            for cond in vote_expl.conditions_met:
                md.append(f"| {cond} | ✅ |")
            for cond in vote_expl.conditions_failed:
                md.append(f"| {cond} | ❌ |")
    
    # Recomendación
    md.append("\n## 🎯 Recomendación Final")
    md.append(f"\n{explanation.final_recommendation}")
    
    # Disclaimer
    md.append("\n---")
    md.append(f"\n{explanation.honest_disclaimer}")
    
    return "\n".join(md)

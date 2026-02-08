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
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

# Importar el módulo de explicabilidad avanzada
from mvpfx.explainability import (
    TradingExplainer, 
    SignalExplanation,
    format_explanation_as_text,
    format_explanation_as_markdown,
    SignalDirection,
    RiskLevel
)

load_dotenv()

# Configurar Google AI Studio
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = None
MODEL_NAME = None

if GOOGLE_API_KEY and GOOGLE_API_KEY != "tu_api_key_aqui":
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Lista de modelos a probar en orden de preferencia (actualizados)
    MODELS_TO_TRY = [
        'gemini-2.5-flash',           # Más reciente y rápido
        'gemini-2.5-pro',             # Pro más reciente
        'gemini-2.0-flash-exp',       # Experimental 2.0
        'gemini-2.0-flash',           # Estable 2.0
        'gemini-1.5-flash-8b',        # Versión ligera
        'gemini-1.5-flash',           # Estable 1.5
        'gemini-1.5-pro',             # Pro 1.5
    ]
    
    # Primero intentar listar los modelos disponibles
    try:
        available_models = list(genai.list_models())
        print(f"📋 Modelos disponibles: {len(available_models)}")
        for m in available_models[:5]:  # Mostrar primeros 5
            if 'generateContent' in str(m.supported_generation_methods):
                print(f"   - {m.name}")
    except Exception as e:
        print(f"⚠️ No se pudo listar modelos: {e}")
    
    for model_name in MODELS_TO_TRY:
        try:
            model = genai.GenerativeModel(model_name)
            MODEL_NAME = model_name
            print(f"✅ Modelo LLM configurado: {model_name}")
            break
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                # El modelo existe pero hay límite de quota - guardarlo para usar después
                model = genai.GenerativeModel(model_name)
                MODEL_NAME = model_name
                print(f"⚠️ Modelo {model_name} configurado (quota limitada, esperando)")
                break
            else:
                print(f"⚠️ Modelo {model_name} no disponible: {error_str[:60]}")
                model = None
                continue
    
    if model is None:
        print("❌ No se pudo configurar ningún modelo de Gemini")
else:
    print("⚠️ GOOGLE_API_KEY no configurada")

def explain_trade(strategy: str, signal: str, indicators: dict, risk: dict, confidence: float):
    rationale = {
        "strategy": strategy, "signal": signal, "indicators": indicators,
        "risk": risk, "confidence": round(float(confidence), 2),
        "checklist": ["Cruce EMA", "RSI coherente", "MACD confirma", "ATR suficiente y régimen tendencial"],
        "caveats": ["Evitar noticias de alto impacto", "Spread anormal"]
    }
    
    # Si no hay API key configurada, usar texto por defecto
    if model is None:
        text = (f"Se propone {signal} con confianza {rationale['confidence']}. "
                "EMAs y MACD alineados; RSI en zona coherente. "
                "Riesgo controlado por fracción fija y SL/TP basados en ATR.")
        return {"json": rationale, "text": text}
    
    # Usar Google Gemini para generar explicación con fallback automático
    prompt = f"""
Eres un analista de trading experto. Explica esta señal de trading de forma clara y educativa:

**Estrategia**: {strategy}
**Señal**: {signal.upper()} ({"COMPRA" if signal == "long" else "VENTA"})
**Indicadores**:
{json.dumps(indicators, indent=2)}

**Gestión de Riesgo**:
{json.dumps(risk, indent=2)}

**Nivel de Confianza**: {confidence:.0%}

Proporciona:
1. Por qué los indicadores técnicos sugieren esta operación
2. Cómo la gestión de riesgo protege el capital
3. Advertencias sobre factores externos (noticias, volatilidad)

Responde en español, máximo 150 palabras, tono educativo.
"""
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
    except Exception as e:
        print(f"Error generando explicación: {e}")
        text = (f"Se propone {signal} con confianza {rationale['confidence']}. "
                f"EMAs y MACD alineados; RSI en zona coherente. "
                f"[Error al consultar IA: {str(e)[:50]}]")
    
    return {"json": rationale, "text": text}

# ==================== CACHÉ PARA ANÁLISIS ====================
_analysis_cache = {}

def _get_cache_key(asset_name: str, signals_hash: str) -> str:
    """Genera una clave única para el caché basada en el activo y las señales."""
    return f"{asset_name}_{signals_hash}"

def _compute_signals_hash(signals: list) -> str:
    """Computa un hash simple de las señales para el caché."""
    import hashlib
    signals_str = json.dumps(signals, sort_keys=True, default=str)
    return hashlib.md5(signals_str.encode()).hexdigest()[:16]


def analyze_signals(asset_name: str, signal_history: list, use_cache: bool = True) -> dict:
    """
    Analiza un historial de señales de trading usando LLM.
    
    Args:
        asset_name: Nombre del activo (ej: "AAPL", "EURUSD")
        signal_history: Lista de señales con su detalle de votación
        use_cache: Si usar caché para evitar llamadas repetidas
    
    Returns:
        dict con "analysis" (texto del análisis) y "cached" (bool)
    """
    
    # Verificar caché
    signals_hash = _compute_signals_hash(signal_history)
    cache_key = _get_cache_key(asset_name, signals_hash)
    
    if use_cache and cache_key in _analysis_cache:
        return {"analysis": _analysis_cache[cache_key], "cached": True}
    
    # Formatear el historial de señales para el prompt
    formatted_history = _format_signal_history(signal_history)
    
    # =========================================================================
    # NUEVO: Generar advertencias de riesgo estructuradas con TradingExplainer
    # =========================================================================
    from mvpfx.config import get_cfg
    cfg = get_cfg()
    explainer = TradingExplainer(cfg)
    
    risk_warnings_section = _generate_risk_warnings_for_prompt(signal_history, explainer)
    indicator_context_section = _generate_indicator_context_for_prompt(signal_history, explainer)
    
    # Construir el prompt completo ENRIQUECIDO con información del TradingExplainer
    prompt = f"""Eres un experto en análisis de comportamiento del mercado de acciones, forex y ETF. Tu misión es explicar de forma detallada, clara y sencilla las señales de trading generadas a través de backtesting.

**IMPORTANTE**: Este análisis incluye advertencias de riesgo CRÍTICAS que el sistema ha detectado automáticamente. DEBES incorporar estas advertencias en tu explicación y ser HONESTO sobre las incertidumbres.

Aquí está el historial de señales que necesitas analizar:

<signal_history>
{formatted_history}
</signal_history>

El activo que se está analizando es:
<asset_name>
{asset_name}
</asset_name>

<advertencias_riesgo_sistema>
{risk_warnings_section}
</advertencias_riesgo_sistema>

<contexto_indicadores>
{indicator_context_section}
</contexto_indicadores>

El historial de señales contiene resultados de backtest donde se calculan indicadores técnicos y se evalúan 4 estrategias diferentes para determinar si se debe tomar una posición SHORT (venta) o LONG (compra) en el comportamiento histórico del activo.

Las 4 estrategias evaluadas son:
1. **EMA Crossover**: Cruces de medias móviles exponenciales (rápida/lenta) con filtros de RSI, MACD y régimen de tendencia.
2. **RSI Reversal**: Reversión a la media usando RSI, Stochastic y Williams %R para detectar zonas de sobrecompra/sobreventa.
3. **MACD Crossover**: Cruces de MACD con su línea de señal, confirmado por CCI, Momentum y ROC.
4. **Bollinger Breakout**: Rupturas de bandas de Bollinger con confirmación de ADX y direccionalidad (+DI/-DI).

Tu tarea es explicar CADA señal individual del historial de la siguiente manera:

Para cada señal, debes:
1. **PRIMERO**: Mencionar las advertencias de riesgo detectadas para esa señal (de <advertencias_riesgo_sistema>)
2. Describir las 4 estrategias que fueron configuradas y evaluadas
3. Explicar qué indicadores técnicos utilizó cada estrategia
4. Identificar cuál fue la mejor estrategia para esa señal específica
5. Explicar claramente por qué se tomó la decisión de SHORT o LONG
6. **Ser HONESTO**: Si hay indicadores que contradicen la señal, menciónalo
7. Explicar la gestión de riesgo con el Stop Loss y Take Profit definidos
8. **FINALIZAR** con una recomendación que considere el nivel de riesgo

Antes de escribir tu explicación final, usa el espacio de <analisis> para:
- Revisar las ADVERTENCIAS DE RIESGO detectadas
- Revisar los datos de cada señal
- Identificar los valores de los indicadores técnicos
- Determinar qué estrategia tuvo mejor desempeño
- **Evaluar si la señal es de ALTO, MEDIO o BAJO riesgo**
- Organizar tu explicación de forma lógica

Tu explicación debe ser:
- En español
- Clara y sencilla (evita jerga técnica excesiva, o explícala cuando la uses)
- **HONESTA sobre las incertidumbres y riesgos**
- Contundente y directa
- Fácil de entender para alguien con conocimientos básicos de trading

Formato de respuesta:

Para cada señal en el historial, estructura tu respuesta así:

<analisis>
[Aquí analiza los datos de la señal, identifica estrategias, indicadores, ADVERTENCIAS DE RIESGO y la mejor opción]
</analisis>

<explicacion_señal>
[Aquí escribe tu explicación clara y detallada de la señal, incluyendo:
- ⚠️ ADVERTENCIAS DE RIESGO (si las hay)
- Número o identificador de la señal
- Descripción de las 4 estrategias evaluadas
- Decisión tomada (SHORT o LONG)
- Mejor estrategia y por qué
- **Indicadores que CONTRADICEN la señal** (si los hay)
- Justificación completa de la decisión
- Gestión de riesgo (SL/TP)
- 🎯 Recomendación final con nivel de riesgo]
</explicacion_señal>

Repite este formato para cada señal en el historial.

Comienza tu análisis ahora"""

    # Si no hay API key configurada, usar respuesta por defecto
    if model is None:
        default_analysis = _generate_default_analysis(asset_name, signal_history)
        if use_cache:
            _analysis_cache[cache_key] = default_analysis
        return {"analysis": default_analysis, "cached": False, "error": "LLM no configurado - usando análisis avanzado del motor de explicabilidad"}
    
    # Llamar al LLM
    print(f"🔄 Generando análisis para {asset_name} con {len(signal_history)} señales...")
    
    try:
        response = model.generate_content(prompt)
        analysis_text = response.text.strip()
        
        # Guardar en caché
        if use_cache:
            _analysis_cache[cache_key] = analysis_text
        
        return {"analysis": analysis_text, "cached": False}
    
    except Exception as e:
        print(f"Error generando análisis: {e}")
        default_analysis = _generate_default_analysis(asset_name, signal_history)
        return {"analysis": default_analysis, "cached": False, "error": f"Error LLM: {str(e)[:100]}"}


def _generate_risk_warnings_for_prompt(signals: list, explainer: TradingExplainer) -> str:
    """
    Genera sección de advertencias de riesgo estructuradas para incluir en el prompt del LLM.
    """
    warnings_parts = []
    
    for i, sig in enumerate(signals, 1):
        indicators = _extract_indicators_from_signal(sig)
        strategy_votes = sig.get("strategy_votes", [])
        
        # Determinar dirección de la señal
        if sig.get("signal") == 1:
            direction = SignalDirection.LONG
        elif sig.get("signal") == -1:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL
        
        # Generar advertencias
        warnings = explainer.generate_risk_warnings(indicators, direction, strategy_votes)
        
        if warnings:
            signal_warnings = [f"\n📍 SEÑAL #{i}:"]
            for w in warnings:
                level_emoji = {"crítico": "🚨", "alto": "⛔", "medio": "⚠️", "bajo": "ℹ️"}.get(w.level.value, "❓")
                signal_warnings.append(f"  {level_emoji} [{w.level.value.upper()}] {w.title}: {w.description}")
                signal_warnings.append(f"     💡 Recomendación: {w.recommendation}")
            warnings_parts.append("\n".join(signal_warnings))
    
    if warnings_parts:
        return "\n".join(warnings_parts)
    return "No se detectaron advertencias de riesgo significativas."


def _generate_indicator_context_for_prompt(signals: list, explainer: TradingExplainer) -> str:
    """
    Genera contexto interpretativo de indicadores para incluir en el prompt del LLM.
    """
    context_parts = []
    
    for i, sig in enumerate(signals, 1):
        indicators = _extract_indicators_from_signal(sig)
        
        if not indicators:
            continue
        
        # Determinar dirección de la señal
        if sig.get("signal") == 1:
            direction = SignalDirection.LONG
        elif sig.get("signal") == -1:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL
        
        # Generar interpretaciones para indicadores clave
        key_indicators = ["rsi", "adx", "macd", "atr"]
        interpretations = []
        
        for ind_name in key_indicators:
            if ind_name in indicators:
                try:
                    explanation = explainer.explain_indicator(
                        name=ind_name,
                        value=indicators[ind_name],
                        signal_direction=direction,
                        all_indicators=indicators
                    )
                    alignment_emoji = {"confirma": "✅", "contradice": "❌", "neutral": "➖"}.get(explanation.signal_alignment, "❓")
                    interpretations.append(f"  - {explanation.name}: {explanation.value} → {explanation.interpretation} {alignment_emoji}")
                except Exception:
                    pass
        
        if interpretations:
            context_parts.append(f"\n📍 SEÑAL #{i} - Contexto de Indicadores:")
            context_parts.extend(interpretations)
    
    if context_parts:
        return "\n".join(context_parts)
    return "Indicadores sin contexto adicional disponible."


def _format_signal_history(signals: list) -> str:
    """Formatea el historial de señales para el prompt del LLM."""
    formatted = []
    
    for i, sig in enumerate(signals, 1):
        signal_type = "LONG (Compra)" if sig.get("signal") == 1 else "SHORT (Venta)"
        
        # Formatear votos de estrategias
        votes_detail = []
        if sig.get("strategy_votes"):
            for vote in sig["strategy_votes"]:
                vote_emoji = "🟢" if vote.get("vote") == "LONG" else "🔴" if vote.get("vote") == "SHORT" else "⚪"
                votes_detail.append(f"  - {vote.get('name', 'N/A')}: {vote_emoji} {vote.get('vote', 'N/A')} (score: {vote.get('score', 0):.4f})")
        
        votes_str = "\n".join(votes_detail) if votes_detail else "  - Sin detalle de votos"
        
        # Gestión de riesgo
        sl = sig.get("sl")
        tp = sig.get("tp")
        price = sig.get("price", sig.get("close", 0))
        
        risk_info = ""
        if sl and tp and price:
            sl_pct = abs((sl - price) / price * 100) if price else 0
            tp_pct = abs((tp - price) / price * 100) if price else 0
            rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 0
            risk_info = f"""
  Gestión de Riesgo:
  - Precio entrada: {price:.5f}
  - Stop Loss: {sl:.5f} ({sl_pct:.2f}% de riesgo)
  - Take Profit: {tp:.5f} ({tp_pct:.2f}% de beneficio)
  - Ratio Riesgo/Beneficio: 1:{rr_ratio:.2f}"""
        
        entry = f"""
--- SEÑAL #{i} ---
Timestamp: {sig.get("timestamp", "N/A")}
Tipo: {signal_type}
Precio: {price:.5f}
Score de confianza: {sig.get("score", 0):.4f}

Votación de Estrategias:
  LONG: {sig.get("long_votes", 0)} votos | SHORT: {sig.get("short_votes", 0)} votos | NEUTRAL: {sig.get("neutral_votes", 0)} votos

Detalle por estrategia:
{votes_str}
{risk_info}
"""
        formatted.append(entry)
    
    return "\n".join(formatted)


def _generate_default_analysis(asset_name: str, signals: list, cfg: dict = None) -> str:
    """
    Genera un análisis detallado y explicable cuando el LLM no está disponible.
    
    Usa el motor de explicabilidad avanzada para proporcionar:
    - Explicaciones contextuales dinámicas de indicadores
    - Advertencias de riesgo automáticas
    - Explicación detallada de votos de cada estrategia
    """
    if cfg is None:
        from mvpfx.config import get_cfg
        cfg = get_cfg()
    
    explainer = TradingExplainer(cfg)
    analysis_parts = []
    
    header = f"""# 📊 Análisis Explicable de Señales para {asset_name}

**Total de señales analizadas:** {len(signals)}
**Fecha de generación:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

> 💡 **Nota de Transparencia**: Este análisis es generado por el motor de explicabilidad del sistema.
> Cada señal incluye: indicadores técnicos contextualizados, advertencias de riesgo automáticas,
> y explicación detallada del voto de cada estrategia.

---
"""
    analysis_parts.append(header)
    
    for i, sig in enumerate(signals, 1):
        # Extraer indicadores de la señal (si están disponibles)
        indicators = _extract_indicators_from_signal(sig)
        strategy_votes = sig.get("strategy_votes", [])
        
        # Generar explicación completa
        try:
            explanation = explainer.generate_full_explanation(
                signal_data=sig,
                indicators=indicators,
                strategy_votes=strategy_votes
            )
            
            # Formatear como Markdown
            signal_analysis = format_explanation_as_markdown(explanation)
            analysis_parts.append(f"\n---\n\n## Señal #{i}\n\n{signal_analysis}")
            
        except Exception as e:
            # Fallback a análisis básico si falla
            analysis_parts.append(_generate_basic_signal_analysis(i, sig, asset_name))
    
    # Resumen final
    summary = _generate_analysis_summary(signals, asset_name)
    analysis_parts.append(f"\n---\n\n{summary}")
    
    return "\n".join(analysis_parts)


def _extract_indicators_from_signal(signal: dict) -> Dict[str, float]:
    """Extrae indicadores técnicos de los datos de la señal."""
    indicators = {}
    
    # Indicadores que podrían estar en la señal
    indicator_keys = [
        "rsi", "macd", "macd_signal", "macd_hist",
        "adx", "plus_di", "minus_di",
        "ema_fast", "ema_slow",
        "stoch_k", "stoch_d", "williams_r",
        "atr", "cci", "momentum", "roc",
        "bb_upper", "bb_lower", "bb_mid",
        "open", "high", "low", "close", "price"
    ]
    
    for key in indicator_keys:
        if key in signal:
            try:
                indicators[key] = float(signal[key])
            except (ValueError, TypeError):
                pass
    
    # Si 'close' no está pero 'price' sí, usar price como close
    if "close" not in indicators and "price" in indicators:
        indicators["close"] = indicators["price"]
    
    return indicators


def _generate_basic_signal_analysis(index: int, sig: dict, asset_name: str) -> str:
    """Genera análisis básico de una señal (fallback)."""
    signal_type = "LONG" if sig.get("signal") == 1 else "SHORT" if sig.get("signal") == -1 else "NEUTRAL"
    price = sig.get("price", sig.get("close", 0))
    sl = sig.get("sl")
    tp = sig.get("tp")
    
    # Determinar estrategia ganadora
    winning_strategy = "Sistema de votación"
    if sig.get("strategy_votes"):
        for vote in sig["strategy_votes"]:
            if (signal_type == "LONG" and vote.get("vote") == "LONG") or \
               (signal_type == "SHORT" and vote.get("vote") == "SHORT"):
                if vote.get("score", 0) > 0.5:
                    winning_strategy = vote.get("name", "N/A")
                    break
    
    risk_text = ""
    if sl and tp and price:
        sl_pct = abs((sl - price) / price * 100)
        tp_pct = abs((tp - price) / price * 100)
        rr = tp_pct / sl_pct if sl_pct > 0 else 0
        risk_text = f"Stop Loss a {sl_pct:.2f}% y Take Profit a {tp_pct:.2f}%. Ratio R:R = 1:{rr:.2f}"
    
    return f"""
### Señal #{index} - {signal_type} en {asset_name}

**Fecha:** {sig.get("timestamp", "N/A")}
**Precio:** {price:.5f}
**Confianza:** {sig.get("score", 0):.2%}

**Decisión:** Se generó una señal de **{signal_type}** basada en la votación de 4 estrategias:
- 🟢 Votos LONG: {sig.get("long_votes", 0)}
- 🔴 Votos SHORT: {sig.get("short_votes", 0)}
- ⚪ Votos Neutral: {sig.get("neutral_votes", 0)}

**Estrategia dominante:** {winning_strategy}

**Gestión de Riesgo:** {risk_text if risk_text else "No definida"}

---"""


def _generate_analysis_summary(signals: list, asset_name: str) -> str:
    """Genera un resumen del análisis completo."""
    total = len(signals)
    long_count = sum(1 for s in signals if s.get("signal") == 1)
    short_count = sum(1 for s in signals if s.get("signal") == -1)
    neutral_count = total - long_count - short_count
    
    # Calcular métricas de confianza
    scores = [s.get("score", 0) for s in signals if s.get("signal") != 0]
    avg_confidence = sum(scores) / len(scores) if scores else 0
    
    return f"""## 📋 Resumen del Análisis

| Métrica | Valor |
|---------|-------|
| Total de señales | {total} |
| Señales LONG | {long_count} ({long_count/total*100:.1f}%) |
| Señales SHORT | {short_count} ({short_count/total*100:.1f}%) |
| Señales NEUTRAL | {neutral_count} ({neutral_count/total*100:.1f}%) |
| Confianza promedio | {avg_confidence:.2%} |

### ⚠️ Recordatorio de Riesgo

Este análisis se genera automáticamente basándose en indicadores técnicos históricos.
**El rendimiento pasado NO garantiza resultados futuros.**

Factores externos no considerados:
- 📰 Noticias macroeconómicas
- 🏦 Decisiones de bancos centrales
- 🌍 Eventos geopolíticos
- 📊 Spreads y liquidez del mercado

**Opere solo con capital que pueda permitirse perder.**
"""


# ============================================================================
# FUNCIONES DE EXPLICABILIDAD AVANZADA
# ============================================================================

def explain_signal_detailed(signal_data: Dict[str, Any], 
                           indicators: Dict[str, float],
                           strategy_votes: List[Dict],
                           cfg: dict = None,
                           output_format: str = "markdown") -> Dict[str, Any]:
    """
    Genera una explicación detallada y transparente de una señal de trading.
    
    Esta función implementa el sistema de explicabilidad avanzada que incluye:
    1. Explicaciones contextuales dinámicas de cada indicador
    2. Sistema de advertencias de riesgo automáticas
    3. Explicación detallada del voto de cada estrategia
    
    Args:
        signal_data: Diccionario con datos de la señal (timestamp, price, sl, tp, signal, score)
        indicators: Diccionario con todos los indicadores técnicos calculados
        strategy_votes: Lista de votos de cada estrategia
        cfg: Configuración del sistema (opcional, usa default si no se proporciona)
        output_format: "markdown", "text", o "json"
    
    Returns:
        Diccionario con:
        - 'explanation': Texto explicativo formateado
        - 'risk_level': Nivel de riesgo global
        - 'warnings': Lista de advertencias detectadas
        - 'recommendation': Recomendación final
        - 'raw': Objeto SignalExplanation completo (si format=json)
    """
    if cfg is None:
        from mvpfx.config import get_cfg
        cfg = get_cfg()
    
    explainer = TradingExplainer(cfg)
    
    # Generar explicación completa
    explanation = explainer.generate_full_explanation(
        signal_data=signal_data,
        indicators=indicators,
        strategy_votes=strategy_votes
    )
    
    # Formatear según el formato solicitado
    if output_format == "text":
        formatted = format_explanation_as_text(explanation)
    elif output_format == "markdown":
        formatted = format_explanation_as_markdown(explanation)
    else:
        formatted = None
    
    # Extraer información clave para respuesta estructurada
    result = {
        "explanation": formatted,
        "risk_level": explanation.overall_risk_level.value,
        "warnings": [
            {
                "code": w.code,
                "level": w.level.value,
                "title": w.title,
                "description": w.description,
                "recommendation": w.recommendation
            }
            for w in explanation.risk_warnings
        ],
        "recommendation": explanation.final_recommendation,
        "disclaimer": explanation.honest_disclaimer,
        "summary": explanation.summary,
        "direction": explanation.direction.value,
        "confidence": explanation.confidence_score,
        "indicators_analysis": [
            {
                "name": ind.name,
                "value": ind.value,
                "interpretation": ind.interpretation,
                "alignment": ind.signal_alignment,
                "context": ind.context
            }
            for ind in explanation.indicator_explanations
        ],
        "strategy_votes_detail": [
            {
                "strategy": vote.strategy_name,
                "vote": vote.vote,
                "score": vote.score,
                "reasoning": vote.reasoning,
                "conditions_met": vote.conditions_met,
                "conditions_failed": vote.conditions_failed,
                "key_indicators": vote.key_indicators
            }
            for vote in explanation.strategy_votes
        ]
    }
    
    if output_format == "json":
        result["raw"] = explanation
    
    return result


def get_risk_warnings(indicators: Dict[str, float], 
                     signal_direction: str,
                     strategy_votes: List[Dict],
                     cfg: dict = None) -> List[Dict]:
    """
    Obtiene solo las advertencias de riesgo para una señal.
    
    Útil para mostrar alertas en el dashboard sin el análisis completo.
    
    Args:
        indicators: Diccionario con indicadores técnicos
        signal_direction: "LONG", "SHORT", o "NEUTRAL"
        strategy_votes: Lista de votos de estrategias
        cfg: Configuración (opcional)
    
    Returns:
        Lista de advertencias de riesgo ordenadas por severidad
    """
    if cfg is None:
        from mvpfx.config import get_cfg
        cfg = get_cfg()
    
    explainer = TradingExplainer(cfg)
    
    # Mapear dirección
    if signal_direction == "LONG":
        direction = SignalDirection.LONG
    elif signal_direction == "SHORT":
        direction = SignalDirection.SHORT
    else:
        direction = SignalDirection.NEUTRAL
    
    warnings = explainer.generate_risk_warnings(indicators, direction, strategy_votes)
    
    return [
        {
            "code": w.code,
            "level": w.level.value,
            "title": w.title,
            "description": w.description,
            "recommendation": w.recommendation,
            "indicator_values": w.indicator_values
        }
        for w in warnings
    ]


def explain_indicator(indicator_name: str, 
                     value: float, 
                     signal_direction: str,
                     all_indicators: Dict[str, float],
                     cfg: dict = None) -> Dict[str, str]:
    """
    Genera explicación contextual para un indicador específico.
    
    Args:
        indicator_name: Nombre del indicador (rsi, macd, adx, etc.)
        value: Valor actual del indicador
        signal_direction: Dirección de la señal ("LONG", "SHORT", "NEUTRAL")
        all_indicators: Todos los indicadores para contexto
        cfg: Configuración (opcional)
    
    Returns:
        Diccionario con interpretación, alineación y contexto
    """
    if cfg is None:
        from mvpfx.config import get_cfg
        cfg = get_cfg()
    
    explainer = TradingExplainer(cfg)
    
    # Mapear dirección
    if signal_direction == "LONG":
        direction = SignalDirection.LONG
    elif signal_direction == "SHORT":
        direction = SignalDirection.SHORT
    else:
        direction = SignalDirection.NEUTRAL
    
    explanation = explainer.explain_indicator(indicator_name, value, direction, all_indicators)
    
    return {
        "name": explanation.name,
        "value": explanation.value,
        "interpretation": explanation.interpretation,
        "alignment": explanation.signal_alignment,
        "context": explanation.context
    }


def explain_strategy_vote_detailed(strategy_name: str,
                                   vote: str,
                                   score: float,
                                   indicators: Dict[str, float],
                                   cfg: dict = None) -> Dict[str, Any]:
    """
    Genera explicación detallada del voto de una estrategia.
    
    Args:
        strategy_name: Nombre de la estrategia
        vote: Voto emitido ("LONG", "SHORT", "NEUTRAL")
        score: Score de confianza
        indicators: Indicadores técnicos
        cfg: Configuración (opcional)
    
    Returns:
        Diccionario con razón, condiciones cumplidas/falladas, indicadores clave
    """
    if cfg is None:
        from mvpfx.config import get_cfg
        cfg = get_cfg()
    
    explainer = TradingExplainer(cfg)
    
    explanation = explainer.explain_strategy_vote(strategy_name, vote, score, indicators, cfg)
    
    return {
        "strategy": explanation.strategy_name,
        "vote": explanation.vote,
        "score": explanation.score,
        "reasoning": explanation.reasoning,
        "conditions_met": explanation.conditions_met,
        "conditions_failed": explanation.conditions_failed,
        "key_indicators": explanation.key_indicators
    }


def clear_analysis_cache():
    """Limpia el caché de análisis."""
    global _analysis_cache
    _analysis_cache = {}
    return {"status": "cache cleared", "items_removed": len(_analysis_cache)}


if __name__ == "__main__":
    out = explain_trade("EMA+RSI+MACD", "long",
                        {"ema_fast":12,"ema_slow":26,"rsi":60,"macd":0.0004},
                        {"risk_pct":0.0075,"sl_atr_mult":1.5,"tp_atr_mult":2.0},
                        0.82)
    print(out["text"]); print(out["json"])

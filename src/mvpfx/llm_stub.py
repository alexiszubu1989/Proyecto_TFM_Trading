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
    
    # Usar Google Gemini para generar explicación
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
        # Fallback si falla la API
        text = (f"Se propone {signal} con confianza {rationale['confidence']}. "
                f"EMAs y MACD alineados; RSI en zona coherente. [Error LLM: {str(e)}]")
    
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
    
    # Construir el prompt completo
    prompt = f"""Eres un experto en análisis de comportamiento del mercado de acciones, forex y ETF. Tu misión es explicar de forma detallada, clara y sencilla las señales de trading generadas a través de backtesting.

Aquí está el historial de señales que necesitas analizar:

<signal_history>
{formatted_history}
</signal_history>

El activo que se está analizando es:
<asset_name>
{asset_name}
</asset_name>

El historial de señales contiene resultados de backtest donde se calculan indicadores técnicos y se evalúan 4 estrategias diferentes para determinar si se debe tomar una posición SHORT (venta) o LONG (compra) en el comportamiento histórico del activo.

Las 4 estrategias evaluadas son:
1. **EMA Crossover**: Cruces de medias móviles exponenciales (rápida/lenta) con filtros de RSI, MACD y régimen de tendencia.
2. **RSI Reversal**: Reversión a la media usando RSI, Stochastic y Williams %R para detectar zonas de sobrecompra/sobreventa.
3. **MACD Crossover**: Cruces de MACD con su línea de señal, confirmado por CCI, Momentum y ROC.
4. **Bollinger Breakout**: Rupturas de bandas de Bollinger con confirmación de ADX y direccionalidad (+DI/-DI).

Tu tarea es explicar CADA señal individual del historial de la siguiente manera:

Para cada señal, debes:
1. Describir las 4 estrategias que fueron configuradas y evaluadas
2. Explicar qué indicadores técnicos utilizó cada estrategia
3. Identificar cuál fue la mejor estrategia para esa señal específica
4. Explicar claramente por qué se tomó la decisión de SHORT o LONG
5. Justificar cómo la mejor estrategia impulsó esa decisión
6. Explicar la gestión de riesgo con el Stop Loss y Take Profit definidos

Antes de escribir tu explicación final, usa el espacio de <analisis> para:
- Revisar los datos de cada señal
- Identificar los valores de los indicadores técnicos
- Determinar qué estrategia tuvo mejor desempeño
- Organizar tu explicación de forma lógica

Tu explicación debe ser:
- En español
- Clara y sencilla (evita jerga técnica excesiva, o explícala cuando la uses)
- Contundente y directa
- Fácil de entender para alguien con conocimientos básicos de trading

Formato de respuesta:

Para cada señal en el historial, estructura tu respuesta así:

<analisis>
[Aquí analiza los datos de la señal, identifica estrategias, indicadores y la mejor opción]
</analisis>

<explicacion_señal>
[Aquí escribe tu explicación clara y detallada de la señal, incluyendo:
- Número o identificador de la señal
- Descripción de las 4 estrategias evaluadas
- Decisión tomada (SHORT o LONG)
- Mejor estrategia y por qué
- Justificación completa de la decisión
- Gestión de riesgo (SL/TP)]
</explicacion_señal>

Repite este formato para cada señal en el historial.

Comienza tu análisis ahora"""

    # Si no hay modelo configurado, usar respuesta por defecto
    if model is None:
        default_analysis = _generate_default_analysis(asset_name, signal_history)
        if use_cache:
            _analysis_cache[cache_key] = default_analysis
        return {"analysis": default_analysis, "cached": False, "error": "LLM no configurado - usando análisis básico"}
    
    # Llamar al LLM
    try:
        response = model.generate_content(prompt)
        analysis_text = response.text.strip()
        
        # Guardar en caché
        if use_cache:
            _analysis_cache[cache_key] = analysis_text
        
        return {"analysis": analysis_text, "cached": False}
    
    except Exception as e:
        # Fallback si falla la API
        error_msg = str(e)
        default_analysis = _generate_default_analysis(asset_name, signal_history)
        return {"analysis": default_analysis, "cached": False, "error": f"Error LLM: {error_msg}"}


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


def _generate_default_analysis(asset_name: str, signals: list) -> str:
    """Genera un análisis básico cuando el LLM no está disponible."""
    analysis_parts = []
    
    for i, sig in enumerate(signals, 1):
        signal_type = "LONG" if sig.get("signal") == 1 else "SHORT"
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
            risk_text = f"Stop Loss a {sl_pct:.2f}% y Take Profit a {tp_pct:.2f}%."
        
        part = f"""
### Señal #{i} - {signal_type} en {asset_name}

**Fecha:** {sig.get("timestamp", "N/A")}
**Precio:** {price:.5f}
**Confianza:** {sig.get("score", 0):.2%}

**Decisión:** Se generó una señal de **{signal_type}** basada en la votación de 4 estrategias:
- Votos LONG: {sig.get("long_votes", 0)}
- Votos SHORT: {sig.get("short_votes", 0)}
- Votos Neutral: {sig.get("neutral_votes", 0)}

**Estrategia dominante:** {winning_strategy}

**Gestión de Riesgo:** {risk_text if risk_text else "No definida"}

---"""
        analysis_parts.append(part)
    
    header = f"""# Análisis de Señales para {asset_name}

**Total de señales analizadas:** {len(signals)}

*Nota: Este es un análisis básico generado automáticamente. Para un análisis más detallado, configure la API de Google Gemini.*

"""
    
    return header + "\n".join(analysis_parts)


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

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
if GOOGLE_API_KEY and GOOGLE_API_KEY != "tu_api_key_aqui":
    genai.configure(api_key=GOOGLE_API_KEY)
    # Probar con diferentes nombres de modelo disponibles
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except:
        try:
            model = genai.GenerativeModel('gemini-pro')
        except:
            model = None
else:
    model = None

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

if __name__ == "__main__":
    out = explain_trade("EMA+RSI+MACD", "long",
                        {"ema_fast":12,"ema_slow":26,"rsi":60,"macd":0.0004},
                        {"risk_pct":0.0075,"sl_atr_mult":1.5,"tp_atr_mult":2.0},
                        0.82)
    print(out["text"]); print(out["json"])

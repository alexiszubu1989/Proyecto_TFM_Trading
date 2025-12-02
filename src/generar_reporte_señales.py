"""Script para generar reporte detallado de todas las señales con explicaciones IA"""
import json
from datetime import datetime
from mvpfx.config import get_cfg
from mvpfx.data import fetch_yfinance
from mvpfx.indicators import compute_all_indicators
from mvpfx.strategy import generate_signals
from mvpfx.llm_stub import explain_trade

cfg = get_cfg()

print("🔄 Descargando datos...")
df = fetch_yfinance(cfg["symbol"], cfg["timeframe"], cfg["warmup_bars"] + 200)
print(f"✅ {len(df)} barras descargadas")

print("\n🔄 Calculando indicadores...")
df = compute_all_indicators(df, cfg)

print("\n🔄 Aplicando warmup...")
df = df.iloc[cfg["warmup_bars"]:].copy()
print(f"✅ {len(df)} barras después de warmup")

print("\n🔄 Generando señales...")
df = generate_signals(df, cfg)

# Filtrar solo las señales
signals_df = df[df['signal'] != 0].copy()
total_signals = len(signals_df)

print(f"\n✅ {total_signals} señales encontradas")
print(f"   🔹 LONG: {(signals_df['signal'] == 1).sum()}")
print(f"   🔻 SHORT: {(signals_df['signal'] == -1).sum()}")

# Generar explicaciones para cada señal
print("\n🤖 Generando explicaciones con IA (Gemini)...")
print("=" * 80)

report = []
for i, (timestamp, row) in enumerate(signals_df.iterrows(), 1):
    signal_type = "LONG" if row['signal'] == 1 else "SHORT"
    action = "COMPRA" if row['signal'] == 1 else "VENTA"
    
    print(f"\n[{i}/{total_signals}] 📍 Señal {signal_type} - {timestamp}")
    print(f"Precio: ${row['close']:.2f}")
    print(f"RSI: {row['rsi']:.2f} | MACD: {row['macd']:.4f} | ATR: {row['atr']:.4f}")
    
    # Generar explicación IA
    try:
        explanation_result = explain_trade(
            strategy="EMA Cross (Ultra-Rápido 3/8)",
            signal="long" if row['signal'] == 1 else "short",
            indicators={
                "ema_fast": float(row['ema_fast']),
                "ema_slow": float(row['ema_slow']),
                "rsi": float(row['rsi']),
                "macd": float(row['macd']),
                "atr": float(row['atr'])
            },
            risk={
                "risk_pct": cfg["risk"]["risk_per_trade"],
                "sl_atr_mult": cfg["risk"]["atr_sl_mult"],
                "tp_atr_mult": cfg["risk"]["atr_tp_mult"]
            },
            confidence=0.75
        )
        explanation = explanation_result["text"]
        print(f"🤖 Explicación IA: {explanation[:150]}...")
        
        # Guardar en reporte
        report.append({
            "numero": i,
            "timestamp": str(timestamp),
            "tipo": signal_type,
            "accion": action,
            "precio": float(row['close']),
            "indicadores": {
                "ema_fast": float(row['ema_fast']),
                "ema_slow": float(row['ema_slow']),
                "rsi": float(row['rsi']),
                "macd": float(row['macd']),
                "atr": float(row['atr'])
            },
            "explicacion_ia": explanation
        })
    except Exception as e:
        print(f"⚠️ Error generando explicación: {e}")
        report.append({
            "numero": i,
            "timestamp": str(timestamp),
            "tipo": signal_type,
            "accion": action,
            "precio": float(row['close']),
            "indicadores": {
                "ema_fast": float(row['ema_fast']),
                "ema_slow": float(row['ema_slow']),
                "rsi": float(row['rsi']),
                "macd": float(row['macd']),
                "atr": float(row['atr'])
            },
            "explicacion_ia": "Error al generar explicación"
        })

# Guardar reporte JSON
output_file = "reporte_señales_completo.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "fecha_generacion": datetime.now().isoformat(),
        "simbolo": cfg["symbol"],
        "timeframe": cfg["timeframe"],
        "total_señales": total_signals,
        "señales_long": int((signals_df['signal'] == 1).sum()),
        "señales_short": int((signals_df['signal'] == -1).sum()),
        "configuracion": {
            "ema_fast": cfg["indicators"]["ema_fast"],
            "ema_slow": cfg["indicators"]["ema_slow"],
            "rsi_period": cfg["indicators"]["rsi_period"]
        },
        "señales": report
    }, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print(f"\n✅ Reporte completo generado: {output_file}")
print(f"📊 Total de señales con explicación IA: {len(report)}")

# Generar también un reporte HTML legible
html_output = "reporte_señales_completo.html"
with open(html_output, 'w', encoding='utf-8') as f:
    f.write(f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Señales - {cfg["symbol"]}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .signal-card {{
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 5px solid;
        }}
        .signal-card.long {{
            border-left-color: #27ae60;
        }}
        .signal-card.short {{
            border-left-color: #e74c3c;
        }}
        .signal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .signal-type {{
            font-size: 1.2em;
            font-weight: bold;
        }}
        .signal-type.long {{
            color: #27ae60;
        }}
        .signal-type.short {{
            color: #e74c3c;
        }}
        .indicators {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .indicator {{
            font-size: 0.9em;
        }}
        .indicator-label {{
            color: #7f8c8d;
            font-weight: bold;
        }}
        .explanation {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
            border-left: 3px solid #3498db;
        }}
        .explanation-title {{
            font-weight: bold;
            color: #2980b9;
            margin-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }}
        .stat-box {{
            text-align: center;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 5px;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <h1>📊 Reporte de Señales de Trading - {cfg["symbol"]}</h1>
    
    <div class="summary">
        <h2>Resumen General</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{total_signals}</div>
                <div class="stat-label">Total Señales</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color: #27ae60;">{(signals_df['signal'] == 1).sum()}</div>
                <div class="stat-label">Señales LONG</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color: #e74c3c;">{(signals_df['signal'] == -1).sum()}</div>
                <div class="stat-label">Señales SHORT</div>
            </div>
        </div>
        <p style="margin-top: 20px;">
            <strong>Timeframe:</strong> {cfg["timeframe"]} | 
            <strong>EMAs:</strong> {cfg["indicators"]["ema_fast"]}/{cfg["indicators"]["ema_slow"]} |
            <strong>Fecha:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
""")
    
    # Escribir cada señal
    for signal_data in report:
        signal_class = signal_data['tipo'].lower()
        f.write(f"""
    <div class="signal-card {signal_class}">
        <div class="signal-header">
            <span class="signal-type {signal_class}">
                {'🔹 ' if signal_data['tipo'] == 'LONG' else '🔻 '}
                Señal #{signal_data['numero']}: {signal_data['tipo']} ({signal_data['accion']})
            </span>
            <span style="color: #7f8c8d;">{signal_data['timestamp']}</span>
        </div>
        
        <div style="font-size: 1.3em; margin: 10px 0;">
            <strong>Precio: ${signal_data['precio']:.2f}</strong>
        </div>
        
        <div class="indicators">
            <div class="indicator">
                <span class="indicator-label">EMA Rápida:</span>
                <span>${signal_data['indicadores']['ema_fast']:.2f}</span>
            </div>
            <div class="indicator">
                <span class="indicator-label">EMA Lenta:</span>
                <span>${signal_data['indicadores']['ema_slow']:.2f}</span>
            </div>
            <div class="indicator">
                <span class="indicator-label">RSI:</span>
                <span>{signal_data['indicadores']['rsi']:.2f}</span>
            </div>
            <div class="indicator">
                <span class="indicator-label">MACD:</span>
                <span>{signal_data['indicadores']['macd']:.4f}</span>
            </div>
            <div class="indicator">
                <span class="indicator-label">ATR:</span>
                <span>{signal_data['indicadores']['atr']:.4f}</span>
            </div>
        </div>
        
        <div class="explanation">
            <div class="explanation-title">🤖 Explicación con IA (Google Gemini):</div>
            <div>{signal_data['explicacion_ia']}</div>
        </div>
    </div>
""")
    
    f.write("""
</body>
</html>
""")

print(f"✅ Reporte HTML generado: {html_output}")
print(f"\n🌐 Abre el archivo HTML en tu navegador para ver el reporte visual completo")

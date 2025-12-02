# 📊 Proyecto TFM Trading - Sistema Algorítmico de Trading

> **Trabajo Final de Máster (TFM)** - Sistema de trading algorítmico educativo con análisis técnico automatizado, generación de señales en tiempo real y visualización interactiva.

## 🎯 Propósito y Objetivo

Este proyecto implementa un **sistema completo de trading algorítmico** diseñado para:

### **Propósito Principal:**
- 🔬 **Investigación académica**: Demostrar la aplicación práctica de algoritmos de trading automatizado
- 📚 **Aprendizaje**: Proporcionar un framework educativo para entender estrategias técnicas
- 🧪 **Experimentación**: Permitir el backtesting y análisis de estrategias sin riesgo financiero

### **Objetivos Específicos:**
1. ✅ **Análisis Técnico Automatizado**: Calcular indicadores técnicos (EMAs, RSI, ATR, MACD) en tiempo real
2. ✅ **Generación de Señales**: Producir señales LONG/SHORT basadas en cruces de medias móviles y confirmación de indicadores
3. ✅ **Gestión de Riesgo**: Implementar stop-loss y take-profit dinámicos basados en volatilidad (ATR)
4. ✅ **Backtesting**: Evaluar estrategias con datos históricos y métricas de performance (CAGR, Sharpe, MaxDD)
5. ✅ **Visualización Interactiva**: Dashboard web con gráficos de velas, señales marcadas y estadísticas en tiempo real
6. ✅ **Integración de Datos**: Conexión con Yahoo Finance para datos de mercado reales

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Dashboard)                      │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │  Gráfico OHLC  │  │  Señal Activa│  │   Estadísticas  │ │
│  │  + Señales     │  │  Destacada   │  │   Trading       │ │
│  └────────────────┘  └──────────────┘  └─────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP REST
┌───────────────────────────▼─────────────────────────────────┐
│                   BACKEND API (FastAPI)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ /signals    │  │ /explanations│  │  Gestión Estado  │   │
│  │ Endpoint    │  │ IA Endpoint  │  │  del Servidor    │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   MOTOR DE ESTRATEGIA                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │  Indicadores │  │  Generación  │  │  Gestión de     │   │
│  │  Técnicos    │──▶  de Señales  │──▶  Riesgo (SL/TP)│   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   CAPA DE DATOS                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │  yfinance    │  │  Validación  │  │  Cache/         │   │
│  │  (Yahoo)     │──▶  de Datos    │──▶  Transformación│   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Cómo Ejecutar el Proyecto

### **1️⃣ Clonar el Repositorio**

```bash
git clone https://github.com/alexiszubu1989/Proyecto_TFM_Trading.git
cd Proyecto_TFM_Trading
```

### **2️⃣ Crear Entorno Virtual y Instalar Dependencias**

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### **3️⃣ Configurar Variables de Entorno**

```powershell
# Windows
Copy-Item .env.example .env

# Linux/Mac
cp .env.example .env
```

Edita `.env` si necesitas configurar API keys (opcional para funcionalidades avanzadas).

### **4️⃣ Orden de Ejecución (Paso a Paso)**

#### **Paso 1: Ejecutar Backtest (Análisis Histórico)**

```powershell
python src/mvpfx/backtest.py
```

**Qué hace:**
- ✅ Descarga datos históricos de Yahoo Finance (AAPL, 250 barras, M5)
- ✅ Calcula indicadores técnicos (EMA 3/8, RSI, ATR, MACD)
- ✅ Genera señales de trading basadas en la estrategia configurada
- ✅ Simula operaciones con gestión de riesgo
- ✅ Calcula métricas de performance: CAGR, Sharpe Ratio, Max Drawdown
- ✅ Genera `backtest_report.json` con resultados

**Salida esperada:**
```
✅ Backtest completado
   CAGR: -0.98%
   Sharpe: 0.16
   MaxDD: -3.4%
   Total Trades: 29
```

#### **Paso 2: Iniciar API REST (Servidor Backend)**

```powershell
python -m uvicorn mvpfx.api:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

**Qué hace:**
- ✅ Inicia servidor FastAPI en `http://127.0.0.1:8000`
- ✅ Expone endpoint `/signals` con datos en tiempo real
- ✅ Expone endpoint `/explanations` con análisis de IA
- ✅ Actualiza datos automáticamente cada 30 segundos
- ✅ Modo `--reload` para desarrollo (reinicia al detectar cambios)

**Verificar que funciona:**
```powershell
curl http://127.0.0.1:8000/signals
```

**Salida esperada:** JSON con 200 barras de datos OHLC + señales

#### **Paso 3: Abrir Dashboard (Visualización)**

1. Abrir navegador web (Chrome, Firefox, Edge)
2. Navegar a: `C:\Users\alexiszul\Documents\Proyecto_TFM\Proyecto_TFM_Trading\dashboard\index.html`
3. O hacer doble clic en el archivo `dashboard/index.html`

**Qué verás:**
- 📊 **Gráfico de velas (candlestick)** con señales LONG/SHORT marcadas
- 🎯 **Señal Activa** destacada con precio, SL, TP y R/R ratio
- 📈 **Estadísticas**: conteo de señales LONG/SHORT, rangos de precio, periodo
- 📋 **Historial scrollable** con todas las señales generadas
- 🔄 **Auto-refresh** cada 30 segundos

## ⚙️ Configuración del Sistema

### **Archivo Principal: `config.yml`**

```yaml
symbol: "AAPL"              # Símbolo a tradear (Apple Inc.)
timeframe: "M5"             # Temporalidad: 5 minutos
warmup_bars: 50             # Barras de calentamiento antes de generar señales

# Estrategia de EMAs (Medias Móviles Exponenciales)
ema_fast: 3                 # EMA rápida (ultra-sensible)
ema_slow: 8                 # EMA lenta (confirmación)

# Filtros de RSI (Desactivados para máxima generación)
rsi_long_min: 0             # Mínimo RSI para LONG (0 = sin filtro)
rsi_short_max: 100          # Máximo RSI para SHORT (100 = sin filtro)

# Gestión de Riesgo
risk_per_trade: 0.0075      # 0.75% de capital por operación
sl_atr_mult: 1.5            # Stop Loss = 1.5 × ATR
tp_atr_mult: 2.0            # Take Profit = 2.0 × ATR

# Fuente de Datos
data:
  source: "yfinance"        # Yahoo Finance
  bars: 250                 # Cantidad de barras a descargar
```

### **Modificar Estrategia:**

Para **cambiar de símbolo** (ejemplo: Tesla):
```yaml
symbol: "TSLA"
```

Para **cambiar temporalidad** (ejemplo: 1 hora):
```yaml
timeframe: "1h"
```

Para **aumentar señales** (EMAs más rápidas):
```yaml
ema_fast: 2
ema_slow: 5
```

Para **reducir señales** (EMAs más lentas):
```yaml
ema_fast: 12
ema_slow: 26
```

## 📁 Estructura del Proyecto

```
Proyecto_TFM_Trading/
├── 📄 config.yml                    # Configuración principal del sistema
├── 📄 requirements.txt              # Dependencias de Python
├── 📄 .env / .env.example          # Variables de entorno
├── 📄 README.md                     # Este archivo
│
├── 📁 dashboard/                    # Frontend del sistema
│   ├── index.html                  # Dashboard interactivo
│   └── Prueba4.css                 # Estilos visuales
│
├── 📁 src/mvpfx/                    # Código principal (backend)
│   ├── api.py                      # 🔌 REST API (FastAPI)
│   ├── backtest.py                 # 📊 Motor de backtesting
│   ├── data.py                     # 📥 Obtención de datos (yfinance)
│   ├── indicators.py               # 📈 Indicadores técnicos (EMA, RSI, ATR, MACD)
│   ├── strategy.py                 # 🎯 Lógica de generación de señales
│   ├── risk.py                     # 🛡️ Gestión de riesgo (SL/TP)
│   ├── config.py                   # ⚙️ Carga de configuración
│   ├── llm_stub.py                 # 🤖 Integración de IA
│   ├── broker_ib.py                # 🏦 Integración con brokers
│   └── logging_utils.py            # 📝 Sistema de logs
│
├── 📁 src/
│   └── generar_reporte_señales.py  # 📄 Generador de reportes HTML/JSON
│
└── 📁 tests/                        # Suite de pruebas automatizadas
    ├── test_api_smoke.py           # Tests de API
    ├── test_indicators.py          # Tests de indicadores
    ├── test_serialization.py       # Tests de serialización
    └── test_strategy_risk.py       # Tests de estrategia y riesgo
```

## 🧪 Ejecutar Pruebas

```powershell
pytest -v
```

## 📊 Métricas y Performance

El sistema calcula automáticamente:

- **CAGR** (Compound Annual Growth Rate): Retorno anualizado
- **Sharpe Ratio**: Relación riesgo/retorno
- **Max Drawdown**: Pérdida máxima desde pico
- **Win Rate**: Porcentaje de operaciones ganadoras
- **Total Trades**: Cantidad de operaciones ejecutadas

## 🛠️ Tecnologías Utilizadas

| Componente | Tecnología | Versión |
|------------|-----------|---------|
| **Backend** | Python | 3.11+ |
| **API** | FastAPI | 0.104+ |
| **Servidor** | Uvicorn | 0.24+ |
| **Datos** | yfinance | 0.2.40+ |
| **Análisis** | pandas, numpy | - |
| **Visualización** | Chart.js | 4.4.0 |
| **Testing** | pytest | - |

## ⚠️ Advertencia Legal

**Este proyecto es EXCLUSIVAMENTE educativo y académico.**

- ❌ NO constituye asesoría financiera
- ❌ NO garantiza rentabilidad
- ❌ Los resultados pasados NO predicen resultados futuros
- ✅ Usar ÚNICAMENTE en modo paper (simulación)
- ✅ Investigar y comprender los riesgos antes de operar con dinero real

## 📧 Contacto

**Autor:** Alexis Zuluaga  
**Repositorio:** [github.com/alexiszubu1989/Proyecto_TFM_Trading](https://github.com/alexiszubu1989/Proyecto_TFM_Trading)  
**Institución:** Trabajo Final de Máster (TFM)

---

**🎓 Desarrollado como parte del Trabajo Final de Máster en [Tu Universidad/Programa]**

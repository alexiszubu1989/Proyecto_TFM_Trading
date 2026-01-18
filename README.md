# Proyecto TFM Trading

Sistema de trading algoritmico para analisis tecnico. Genera senales LONG/SHORT mediante votacion de estrategias y ejecuta backtests con gestion de riesgo basada en ATR.

## Arquitectura

```
config.yml          Parametros del sistema
src/mvpfx/
  api.py            Endpoints REST (FastAPI)
  backtest.py       Motor de backtesting
  strategy.py       Generacion de senales
  indicators.py     EMA, RSI, MACD, ATR, Bollinger, Stochastic, ADX, CCI
  risk.py           Calculo de posicion y limites diarios
  data.py           Obtencion de datos (yfinance, IB)
  config.py         Carga de configuracion
  llm_stub.py       Integracion con Gemini (opcional)
  broker_ib.py      Conexion Interactive Brokers
dashboard/
  index.html        Interfaz web
```

## Logica de Senales

El sistema evalua 4 estrategias simultaneamente:

| Estrategia | Condicion LONG | Condicion SHORT |
|------------|----------------|-----------------|
| EMA Crossover | EMA rapida cruza hacia arriba EMA lenta | EMA rapida cruza hacia abajo EMA lenta |
| RSI Reversal | RSI sale de sobreventa (<30) + Stochastic confirma | RSI sale de sobrecompra (>70) + Stochastic confirma |
| MACD Crossover | MACD cruza hacia arriba linea de senal + ADX>20 | MACD cruza hacia abajo linea de senal + ADX>20 |
| Bollinger Breakout | Precio rompe banda superior + momentum positivo | Precio rompe banda inferior + momentum negativo |

Cada estrategia emite un voto (+1 LONG, -1 SHORT, 0 neutral). La senal final requiere `min_strategy_votes` votos en la misma direccion (default: 2 de 4).

Metodos de desempate disponibles: `score`, `priority`, `adx_trend`, `momentum`, `conservative`.

## Calculo de Riesgo (SL/TP)

```
Stop Loss  = Precio entrada +/- (ATR * atr_sl_mult)
Take Profit = Precio entrada +/- (ATR * atr_tp_mult)
```

Valores por defecto:
- `atr_sl_mult`: 1.5
- `atr_tp_mult`: 2.0
- Risk/Reward ratio: 1:1.33

**Calculo de posicion:**
```
risk_amount = equity * risk_per_trade
stop_distance = ATR * atr_sl_mult
units = floor(risk_amount / stop_distance)
```

Limites:
- `risk_per_trade`: 0.75% del capital
- `daily_loss_limit`: 3% del capital
- `max_trades_per_day`: 6

## Calculo de PnL

**LONG:**
```
PnL = (exit_price - entry_price) * units
```

**SHORT:**
```
PnL = (entry_price - exit_price) * units
```

El backtest simula spread y slippage configurables en `execution.simulate_spread` y `execution.simulate_slippage`.

## Configuracion

Editar `config.yml`:

```yaml
symbol: "AAPL"
timeframe: "M5"

indicators:
  ema_fast: 3
  ema_slow: 8
  rsi_period: 14
  atr_period: 14

strategy:
  combine_strategies: true
  min_strategy_votes: 2
  enabled_strategies:
    - "ema_crossover"
    - "rsi_reversal"
    - "macd_crossover"
    - "bollinger_breakout"

risk:
  capital: 10000.0
  risk_per_trade: 0.0075
  atr_sl_mult: 1.5
  atr_tp_mult: 2.0

data:
  source: "yfinance"    # yfinance | ib
  bars: 250
```

## Uso

**Backtest:**
```bash
python src/mvpfx/backtest.py --print --detailed
```

**API:**
```bash
python -m uvicorn mvpfx.api:app --app-dir src --host 127.0.0.1 --port 8000
```

**Endpoints:**
| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| GET | `/signals` | Senales de trading |
| GET | `/backtest/{ticker}` | Ejecutar backtest |
| POST | `/analysis` | Analisis LLM |
| GET | `/` | Dashboard |

## Metricas

- CAGR: Retorno anualizado
- Sharpe Ratio: Retorno ajustado por riesgo
- Sortino Ratio: Retorno ajustado por downside risk
- Max Drawdown: Perdida maxima desde pico
- Win Rate: Porcentaje de trades ganadores
- Profit Factor: Ganancias brutas / Perdidas brutas

## Fuentes de Datos

| Fuente | Requisitos | Uso |
|--------|------------|-----|
| yfinance | `pip install yfinance` | Datos historicos Yahoo Finance |
| ib | TWS/IB Gateway corriendo | Interactive Brokers (paper/live) |

## Tests

```bash
pytest -v
```

## Dependencias

- Python 3.11+
- FastAPI, Uvicorn
- pandas, numpy
- yfinance
- ib_insync (opcional)
- google-generativeai (opcional)

## Advertencia

Proyecto educativo. No constituye asesoria financiera.

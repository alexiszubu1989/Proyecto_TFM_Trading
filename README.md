# MVP EUR/USD (Paper)

> Educativo, no asesoría financiera. Paper por defecto.

## Cómo correr
1) `python -m venv .venv && source .venv/bin/activate`  
2) `pip install -r requirements.txt`  
3) `cp .env.example .env`  
4) `python src/mvpfx/backtest.py`  
5) `uvicorn mvpfx.api:app --app-dir src --reload`  
6) abrir `dashboard/index.html`

## Configuración
Editar `config.yml` (timeframe, ventanas, riesgo). Los ENV tienen prioridad para flags críticos (e.g. PAPER).

## Pruebas
`pytest -q`

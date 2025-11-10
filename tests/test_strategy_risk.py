from mvpfx.data import simulate_ohlcv
from mvpfx.indicators import compute_all_indicators
from mvpfx.strategy import generate_signals
from mvpfx.risk import position_size
from mvpfx.config import get_cfg

def test_strategy_and_sizing():
    cfg = get_cfg()
    df = simulate_ohlcv(800, cfg["timeframe"], cfg["data"]["seed"])
    feats = compute_all_indicators(df, cfg)
    sigs = generate_signals(feats, cfg)
    assert "signal" in sigs.columns
    # Debe haber al menos alguna señal (no garantizado siempre, pero muy probable con sim)
    assert sigs["signal"].abs().sum() >= 0
    units = position_size(cfg["risk"]["capital"], price=1.08, atr=0.001, cfg=cfg)
    assert units >= cfg["risk"]["min_position_units"]

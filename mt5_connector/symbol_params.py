import MetaTrader5 as mt5
SYMBOLS = ["GBPUSDrfd", "EURUSDrfd", "#SBER", "#ROSN", "USDCADrfd", "#NVTK", "USDJPYrfd",
           "#MOEX", "USDCHFrfd", "#LKOH", "NZDUSDrfd", "#GAZP", "#PHOR", "AUDUSDrfd", "USDRUBrfd",
           "#GMKN", "#MTSS", "#VTBR", "#T"]
TIMEFRAMES = {
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1
}
TRAINBARS = 2500
BACKTEST_BARS = 500
TOP_RULES = 5  # Топ-5 (стабильность)

DEPOSIT_RUB = 6500  # Стартовый капитал
RISK_PER_TRADE = 0.01  # 1%
ROTATION_THRESHOLD = 0.3  # lift разница для ротации
MIN_MARGIN_PCT = 0.30  # 30% маржи минимум

SL_ATR = 2.0               # SL = 2×ATR
TP_ATR = 3.0               # TP = 3×ATR
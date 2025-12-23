import MetaTrader5 as mt5
SYMBOLS = ["GBPUSDrfd", "EURUSDrfd", "#SBER", "#ROSN", "USDCADrfd", "#NVTK", "USDJPYrfd",
           "#MOEX", "USDCHFrfd", "#LKOH", "NZDUSDrfd", "#GAZP", "#PHOR", "AUDUSDrfd", "USDRUBrfd",
           "#GMKN", "#MTSS", "#VTBR", "#T"]
TIMEFRAMES = {
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1
}


DEPOSIT_RUB = 6500  # Стартовый капитал
RISK_PER_TRADE = 0.01  # 1%
ROTATION_THRESHOLD = 0.3  # lift разница для ротации
MIN_MARGIN_PCT = 0.30  # 30% маржи минимум
MAX_PYRAMID_PER_INSTRUMENT = 2    # Макс. 2 сделки НА ИНСТРУМЕНТ
RISK_TRADE_PER_CAPITAL = 1        # Макс. 1 "агрессивная" сделка на весь капитал

SL_ATR = 2.0               # SL = 2×ATR
TP_ATR = 3.0               # TP = 3×ATR
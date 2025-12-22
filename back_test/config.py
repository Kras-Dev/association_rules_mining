"""⚙️ Глобальные настройки торговой системы"""

# 💰 Капитал и риск
INITIAL_CAPITAL = 85.0
RISK_PER_TRADE = 0.01  # 1%

# 🛡️ Риск-менеджмент
MAX_PYRAMID_LEVELS = 3
SL_ATR_MULTIPLIER = 2.0
SL_CAP_PCT = 0.015  # 1.5%
TP_ATR_MULTIPLIER = 2.0


# 📊 Режимы выхода
EXIT_MODES = {
    "SIGNAL_TO_SIGNAL": "Держим от сигнала до противоположного",
    "ONE_CANDLE": "Входим → держим 1 свечу → выходим",
    "ATR_TP": "TP=2*ATR, SL=динамический"
}

# 🎯 Фильтры правил
MIN_CONFIDENCE = 0.70
MIN_RULES = 10
MIN_SIGNALS = 5

# 📈 Символы для теста С SL_MULTIPLIER
SL_MULTIPLIER = {
    '#': 1.2,      # Все акции
    'rfd': 1.2     # Все форекс
}

TEST_SYMBOLS = ["GBPUSDrfd", "EURUSDrfd", "#SBER", "#ROSN", "USDCADrfd", "#NVTK", "USDJPYrfd",
           "#MOEX", "USDCHFrfd", "#LKOH", "NZDUSDrfd", "#GAZP", "#PHOR", "AUDUSDrfd", "USDRUBrfd",
           "#GMKN", "#MTSS", "#VTBR", "#T"]
TEST_TIMEFRAMES = ["D1", "H4", "H1", "M15"]
CANDLES_BY_TF = {
    'M15': 20000,  # ~4 месяца
    'H1':  12000,  # ~1.5 года
    'H4':  6000,   # ~6 лет
    'D1':  3000    # ~12 лет
}

def get_candles(tf):
    return CANDLES_BY_TF.get(tf, 35000)

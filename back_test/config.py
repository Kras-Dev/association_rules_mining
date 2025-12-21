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

# 📈 Символы для теста
TEST_SYMBOLS = {
    "#GMKN": "D1",
    "#GMKN": "H4",
    "#GMKN": "H1",
    "#LKOH": "D1",
    "#LKOH": "H4",
    "#LKOH": "H1",
    "#SBER": "D1",
    "#SBER": "H4",
    "#SBER": "H1",
    "EURUSDrfd": "D1",
    "EURUSDrfd": "H4",
    "EURUSDrfd": "H1",
    "USDCADrfd": "D1",
    "USDCADrfd": "H4",
    "USDCADrfd": "H1",
    "USDJPYrfd": "D1",
    "USDJPYrfd": "H4",
    "USDJPYrfd": "H1",
    "#ROSN": "D1",
    "#ROSN": "H4",
    "#ROSN": "H1"
}
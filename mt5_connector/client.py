import logging
from typing import Optional, Any

import MetaTrader5 as mt5
import pandas as pd
from utils.config_manager import load_config

logger = logging.getLogger(__name__)

class MT5Client:
    """Клиент для MT5 с методами загрузки данных."""

    def __init__(self):
        self.config = load_config()
        if self.config is None:
            raise FileNotFoundError("[MT5Client]:❌ config.yaml не найден!")

        self.login = self.config.get('login')
        self.password = self.config.get('password')
        self.server = self.config.get('server')
        self.connected = False
        self.account_info = None

        if not mt5.initialize():
            raise ConnectionError(f"❌ MT5 initialize failed: {mt5.last_error()}")

    def connect(self) -> bool:
        if self.connected:
            return True

        if self.login:
            self.connected = mt5.login(self.login, password=self.password, server=self.server)
        else:
            self.connected = True  # демо-доступ

        if not self.connected:
            print(f"[MT5Client]:❌ Ошибка: {mt5.last_error()}")
            return False

        self.account_info = mt5.account_info()
        print("[MT5Client]:✅ MT5 подключён")
        return True

    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("[MT5Client]: MT5 отключён")

    def get_rates(self, symbol: str, timeframe: int, count: int = 100,
                  start_pos: int = 1) -> Optional[pd.DataFrame]:
        """OHLCV данные."""
        if not mt5.symbol_select(symbol, True):
            print(f"[MT5Data]: ❌ Символ {symbol} не выбран")
            return None

        rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
        if rates is None or len(rates) == 0:
            print(f"[MT5Data]: ❌ Нет данных {symbol}: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        #print(f"[MT5Client]: ✅ {len(df)} баров {symbol} период {self._mt5_var_name(timeframe)}")
        return df.dropna(subset=['time', 'close']).sort_values('time').reset_index(drop=True)

    def mt5_var_name(self, tf_value) -> str:
        """mt5.TIMEFRAME_D1 → 'TIMEFRAME_D1'"""
        mt5_vars = {v: k for k, v in vars(mt5).items() if k.startswith('TIMEFRAME_')}
        return mt5_vars.get(tf_value)

    def check_autotrading(self) -> bool:
        """ПРОВЕРКА AutoTrading ВКЛЮЧЁН?"""
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            return False

        trade_allowed = terminal_info.trade_allowed
        if not trade_allowed:
            logger.error("[MT5Client]: ❌ AutoTrading ОТКЛЮЧЁН! Включи зелёную кнопку!")
            return False

        logger.info("[MT5Client]: ✅ AutoTrading ВКЛЮЧЁН")
        return True

    def send_order(self, symbol: str, action: int, volume: float, price: float,
                   sl: float, tp: float, comment: str = "LiveTrader") -> bool:
        if not self.check_autotrading():
            return False

        if not mt5.symbol_select(symbol, True):
            logger.error(f"[MT5Client]: ❌ Символ {symbol} не выбран")
            return False

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"[MT5Client]: ❌ Нет info {symbol}")
            return False


        point = symbol_info.point
        broker_min_stop = symbol_info.trade_stops_level * point

        # 🔥 УНИВЕРСАЛЬНЫЙ МИНИМУМ по типу символа
        if symbol.startswith('#'):  # Акции
            MIN_STOP_PIPS = 50
            MIN_TP_PIPS = 75
        else:  # Forex
            MIN_STOP_PIPS = 20
            MIN_TP_PIPS = 30

        SAFE_STOP = max(broker_min_stop * 2, MIN_STOP_PIPS * point)
        SAFE_TP = max(broker_min_stop * 3, MIN_TP_PIPS * point)

        # 🔥 КОРРЕКЦИЯ SL/TP
        if action > 0:  # BUY
            if sl > 0:
                sl = price - max(SAFE_STOP, abs(price - sl))  # Берём большее!
            if tp > 0:
                tp = price + max(SAFE_TP, abs(tp - price))
        else:  # SELL
            if sl > 0:
                sl = price + max(SAFE_STOP, abs(sl - price))
            if tp > 0:
                tp = price - max(SAFE_TP, abs(tp - price))

        logger.info(f"[MT5Client]: {symbol} broker={broker_min_stop:.5f} safeSL={SAFE_STOP:.5f}")

        order_type = mt5.ORDER_TYPE_BUY if action > 0 else mt5.ORDER_TYPE_SELL
        price = self._normalize_price(symbol, price)
        sl = self._normalize_price(symbol, sl)
        tp = self._normalize_price(symbol, tp)

        logger.info(f"[MT5Client]: DEBUG {symbol} ACTION={action} ({'BUY' if action == 0 else 'SELL'})")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": f"{comment} v2.0",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"[MT5Client]: ❌ {symbol} {result.retcode} {result.comment}")
            return False

        logger.info(f"[MT5Client]: ✅ {symbol} {volume:.2f}л | {price:.5f} | SL:{sl:.5f} | ID:{result.order}")
        return True

    def close_position(self, position_ticket: int) -> bool:
        """Закрытие позиции по тикету"""
        positions = mt5.positions_get(ticket=position_ticket)
        if not positions:
            logger.warning(f"[MT5Client]: ❌ Позиция {position_ticket} не найдена")
            return False

        pos = positions[0]
        symbol = pos.symbol

        # Обратный ордер для закрытия
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": position_ticket,
            "price": mt5.symbol_info_tick(symbol).bid if pos.type == 0 else mt5.symbol_info_tick(symbol).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "LiveTrader CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[MT5Client]: ✅ Закрыта {symbol} #{position_ticket}")
            return True
        else:
            logger.error(f"[MT5Client]: ❌ Закрытие {symbol}: {result.retcode}")
            return False

    def _normalize_price(self, symbol: str, price: float) -> float:
        """Округление цены до tick_size"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            digits = symbol_info.digits
            return round(price, digits)
        return price

    def get_positions(self) -> list:
        """Все текущие позиции"""
        if not self.connected:
            return []
        return list(mt5.positions_get())  # ✅ list[namedtuple]

    def get_position(self, symbol: str) -> Optional[Any]:
        """Позиция по символу"""
        positions = mt5.positions_get(symbol=symbol)
        return positions[0] if positions else None  # ✅ namedtuple или None

    def get_position_by_ticket(self, ticket: int) -> Optional[Any]:
        """Позиция по тикету"""
        positions = mt5.positions_get(ticket=ticket)
        return positions[0] if positions else None


    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

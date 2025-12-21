"""💱 Модели сделок и позиций"""

from dataclasses import dataclass
from typing import Dict
import pandas as pd


@dataclass
class Trade:
    """Сделка"""
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    size: float
    pnl: float
    win: bool
    rule: str
    pyramid_level: int = 1


class PositionManager:
    """Управление позицией"""

    @staticmethod
    def create_long(entry_price: float, atr: float, size: float,
                    entry_time: pd.Timestamp, entry_idx: int, rule: str) -> Dict:
        """Создаёт LONG позицию"""
        sl_distance = min(atr * 2.0, entry_price * 0.015)
        sl_price = entry_price - sl_distance

        return {
            'type': 'LONG',
            'entry': entry_price,
            'sl': sl_price,
            'size': size,
            'pyramid_level': 1,
            'entry_time': entry_time,
            'entry_idx': entry_idx,
            'rule': rule
        }

    @staticmethod
    def create_short(entry_price: float, atr: float, size: float,
                     entry_time: pd.Timestamp, entry_idx: int, rule: str) -> Dict:
        """Создаёт SHORT позицию"""
        sl_distance = min(atr * 2.0, entry_price * 0.015)
        sl_price = entry_price + sl_distance

        return {
            'type': 'SHORT',
            'entry': entry_price,
            'sl': sl_price,
            'size': size,
            'pyramid_level': 1,
            'entry_time': entry_time,
            'entry_idx': entry_idx,
            'rule': rule
        }

    @staticmethod
    def pyramid(position: Dict) -> Dict:
        """Добавка к позиции"""
        add_size = position['size'] * 0.5
        position['size'] += add_size
        position['pyramid_level'] += 1
        return position

    @staticmethod
    def calculate_pnl(position: Dict, exit_price: float) -> float:
        """Расчёт PnL"""
        if position['type'] == 'LONG':
            return (exit_price - position['entry']) * position['size']
        return (position['entry'] - exit_price) * position['size']

    @staticmethod
    def should_exit_sl(position: Dict, current_price: float) -> bool:
        """SL пробой"""
        if position['type'] == 'LONG':
            return current_price <= position['sl']
        return current_price >= position['sl']

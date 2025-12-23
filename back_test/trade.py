"""💱 Модели сделок и позиций"""
from dataclasses import dataclass
from typing import Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)

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
    stop_loss: float = 0.0
    take_profit: float = 0.0
    pyramid_level: int = 1
    direction: str = ''

class PositionManager:
    """Управление позицией (без логов)"""

    @staticmethod
    def create_long(entry_price: float, atr: float, size: float,
                    entry_time: pd.Timestamp, entry_idx: int, rule: str) -> Dict:
        sl_distance = min(atr * 2.0, entry_price * 0.015)
        stop_loss_level = entry_price - sl_distance
        return {
            'type': 'LONG', 'entry': entry_price, 'sl': stop_loss_level,
            'size': size, 'pyramid_level': 1, 'entry_time': entry_time,
            'entry_idx': entry_idx, 'rule': rule
        }

    @staticmethod
    def create_short(entry_price: float, atr: float, size: float,
                     entry_time: pd.Timestamp, entry_idx: int, rule: str) -> Dict:
        sl_distance = min(atr * 2.0, entry_price * 0.015)
        stop_loss_level = entry_price + sl_distance
        return {
            'type': 'SHORT', 'entry': entry_price, 'sl': stop_loss_level,
            'size': size, 'pyramid_level': 1, 'entry_time': entry_time,
            'entry_idx': entry_idx, 'rule': rule
        }

    @staticmethod
    def pyramid(position: Dict) -> Dict:
        add_size = position['size'] * 0.5
        position['size'] += add_size
        position['pyramid_level'] += 1
        return position

    @staticmethod
    def calculate_pnl(position: Dict, exit_price: float) -> float:
        if position['type'] == 'LONG':
            return (exit_price - position['entry']) * position['size']
        return (position['entry'] - exit_price) * position['size']

    @staticmethod
    def should_exit_sl(position: Dict, current_price: float) -> bool:
        if position['type'] == 'LONG':
            return current_price <= position['sl']
        return current_price >= position['sl']

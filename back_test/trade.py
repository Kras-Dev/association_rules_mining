"""
Модуль, содержащий структуры данных для сделок (Trade)
и набор методов для управления позициями (PositionManager).
"""
from dataclasses import dataclass
from typing import Dict
import pandas as pd

from back_test.config import  SL_CAP_PCT
from utils.base_logger import BaseLogger


@dataclass
class Trade:
    """
    Класс данных (DTO) для хранения информации о завершенной торговой сделке.

    Attributes:
        entry_time (pd.Timestamp): Время входа в позицию.
        entry_price (float): Цена входа.
        exit_time (pd.Timestamp): Время выхода из позиции.
        exit_price (float): Цена выхода.
        size (float): Размер позиции (количество лотов/акций).
        pnl (float): Финансовый результат сделки.
        win (bool): True, если сделка прибыльная.
        rule (str): Имя правила, инициировавшего вход.
        stop_loss (float, optional): Уровень Stop Loss на момент входа.
        take_profit (float, optional): Уровень Take Profit на момент входа.
        pyramid_level (int, optional): Уровень пирамидинга (1 - базовый вход).
        direction (str, optional): Направление сделки ('LONG'/'SHORT').
    """
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

class PositionManager(BaseLogger):
    """
    Статические методы для управления логикой торговых позиций (открытие,
    пирамидинг, расчет PnL, проверка SL/TP).
    """
    def __init__(self, verbose=False):

        super().__init__(verbose=verbose)

    def create_long(self, entry_price: float, atr: float, size: float,
                    entry_time: pd.Timestamp, entry_idx: int, rule: str,
                    sl_mult: float) -> Dict:
        """
        Открытие длинной позиции (Long).

        Args:
            entry_price (float): Цена входа.
            atr (float): Текущий ATR.
            size (float): Размер позиции.
            entry_time (pd.Timestamp): Время.
            entry_idx (int): Индекс бара.
            rule (str): Имя правила.
            sl_mult (float): стоп-лосс мультипликатор

        Returns:
            Dict: Словарь, представляющий открытую позицию.
        """
        # --- Расчет Stop Loss ---
        sl_distance = atr * sl_mult
        max_sl_pct = entry_price * SL_CAP_PCT  # Максимум 1.5% от цены
        # Выбираем МЕНЬШУЮ из дистанций, чтобы не превысить лимит в 1.5%
        final_sl_distance = min(sl_distance, max_sl_pct)
        stop_loss_level = entry_price - final_sl_distance
        self._log_info(f"OPEN LONG: {entry_price:.2f}, SL: {stop_loss_level:.2f}")
        return {
            'type': 'LONG', 'entry': entry_price, 'sl': stop_loss_level,
            'size': size, 'pyramid_level': 1, 'entry_time': entry_time,
            'entry_idx': entry_idx, 'rule': rule
        }

    def create_short(self, entry_price: float, atr: float, size: float,
                     entry_time: pd.Timestamp, entry_idx: int, rule: str,
                     sl_mult: float) -> Dict:
        """
        Открытие короткой позиции (Short). Аналогично Long, но в обратную сторону.
        """
        # --- Расчет Stop Loss ---
        sl_distance = atr * sl_mult
        max_sl_pct = entry_price * SL_CAP_PCT
        # Выбираем МЕНЬШУЮ из дистанций
        final_sl_distance = min(sl_distance, max_sl_pct)
        stop_loss_level = entry_price + final_sl_distance
        self._log_info(f"OPEN SHORT: {entry_price:.2f}, SL: {stop_loss_level:.2f}")
        return {
            'type': 'SHORT', 'entry': entry_price, 'sl': stop_loss_level,
            'size': size, 'pyramid_level': 1, 'entry_time': entry_time,
            'entry_idx': entry_idx, 'rule': rule
        }

    def pyramid(self, position: Dict, current_price: float, multiplier: float = 0.5) -> Dict:
        """
        Увеличивает объем текущей позиции (пирамидинг).

        Логика:
        1. Рассчитывает объем добавки от ТЕКУЩЕГО общего объема.
        2. Увеличивает счетчик уровней.

        Args:
            position (Dict): Словарь текущей позиции.
            multiplier (float): Доля от текущего размера для добавки (0.5 = +50%).
            current_price: Цена добавления позиции

        Returns:
            Dict: Обновленный словарь позиции.
        """
        # Рассчитываем добавку
        add_size = position['size'] * multiplier
        old_size = position['size']

        # Обновляем средневзвешенную цену входа
        position['entry'] = ((position['entry'] * old_size) + (current_price * add_size)) / (old_size + add_size)
        position['size'] += add_size
        position['pyramid_level'] += 1
        self._log_info(f"Пирамидинг: уровень {position['pyramid_level']}, новый размер {position['size']}")

        return position

    def calculate_pnl(self, position: Dict, exit_price: float) -> float:
        """Рассчитывает PnL при закрытии позиции."""
        if position['type'] == 'LONG':
            # Прибыль = (Цена_выхода - Цена_входа) * Размер
            return (exit_price - position['entry']) * position['size']
        # Для Short:
        # Прибыль = (Цена_входа - Цена_выхода) * Размер
        return (position['entry'] - exit_price) * position['size']

    def should_exit_sl(self, position: Dict, current_price: float) -> bool:
        """Проверяет, сработал ли Stop Loss."""
        if position['type'] == 'LONG':
            return current_price <= position['sl']
        return current_price >= position['sl']

""" Основной класс для запуска симуляции торговых стратегий (бэктестинга)"""

from pathlib import Path

import numpy as np
import pandas as pd
import talib
from typing import Dict, Optional, cast
from tqdm import tqdm

from back_test.config import *
from back_test.trade import PositionManager, Trade
from back_test.metrics import MetricsCalculator
from utils.base_file_handler import BaseFileHandler

class Backtester(BaseFileHandler):
    """
    Класс Backtester симулирует торговые операции на исторических данных,
    используя набор правил ассоциации и управляя позициями.

    Attributes:
        rules (pd.DataFrame): Загруженные торговые правила.
        capital (float): Начальный/текущий капитал симуляции.
        trades (list[Trade]): Список совершенных сделок.
        position (Optional[Dict]): Текущая открытая позиция.
        exit_mode (str): Режим выхода из позиции ("SIGNAL_TO_SIGNAL", "ONE_CANDLE", "ATR_TP").
        symbol (str): Торгуемый инструмент (например, 'EURUSD').
        timeframe (str): Используемый таймфрейм (например, 'H1').
    """

    def __init__(self, symbol: str, verbose: bool = True, history_dir: Path = None):
        """
        Инициализация бэктестера.

        Args:
            symbol (str): Торгуемый инструмент.
            verbose (bool): Детальное логирование (по умолчанию False).
            history_dir (Path, optional): Путь к директории истории/моделей.
        """
        super().__init__(verbose, history_dir)
        self.rules:  Optional[pd.DataFrame]  = None
        self.capital = INITIAL_CAPITAL
        self.trades: list[Trade] = []
        self.pos_manager = PositionManager(verbose=self.verbose)
        self.position = None
        self.exit_mode = "SIGNAL_TO_SIGNAL"
        self.symbol = symbol
        self.timeframe = None
        self.total_sl_hits = 0
        self.equity_history = []
        self.current_sl_mode = None
        self.current_min_conf = None

    def load_rules(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Загружает предобученные торговые правила из файла кэша.

        Args:
            symbol (str): Инструмент.
            timeframe (str): Таймфрейм.

        Returns:
            pd.DataFrame: Отфильтрованные правила, готовые к использованию.
        """
        cache_file = self._load_cache(self._get_cache_path(symbol, timeframe))
        # --- Обработка ошибок загрузки кэша ---
        if not cache_file:
            self._log_error(f"❌ Нет файла кэша правил для {symbol} {timeframe}")
            return pd.DataFrame()

        top_rules_df = cache_file['top_rules']
        if top_rules_df.empty:
            self._log_warning(f"⚠️ Кэш файл пуст для {symbol} {timeframe}")
            return pd.DataFrame()
        # --- Фильтрация правил по минимальной уверенности (MIN_CONFIDENCE) ---
        rules = top_rules_df[top_rules_df['confidence'] > MIN_CONFIDENCE]
        self._log_info(f"✅ {len(rules)} правил >{MIN_CONFIDENCE:.0%} conf")
        return rules

    def get_active_rules(self, features_row: pd.Series) -> pd.DataFrame:
        """
        Определяет, какие из загруженных правил сработали на текущем баре.

        Args:
            features_row (pd. Series): Строка с бинарными признаками текущего бара.

        Returns:
            pd.DataFrame: Сработавшие правила.
        """
        matched_rules = []
        # Получаем только те признаки, которые РЕАЛЬНО равны 1 на этом баре
        active_features_on_bar = set(features_row[features_row == 1].index)

        for idx, rule in self.rules.iterrows():
            rule_name = rule['rule_name']
            # 1. Заменяем 'prev' и 'curr' на специальный разделитель, например '|'
            # 2. Убираем лишние подчеркивания вокруг них
            clean_name = rule_name.replace('_prev_', '|').replace('_curr_', '|')

            # 3. Разбиваем по разделителю и очищаем от пустых строк
            # Теперь из 'big_red_prev_big_green' мы получим ['big_red', 'big_green']
            needed_features = [p.strip('_') for p in clean_name.split('|') if p]

            # 4. Проверяем наличие полных имен фич в активных колонках
            if set(needed_features).issubset(active_features_on_bar):
                matched_rules.append(rule.to_dict())

        return pd.DataFrame(matched_rules) if matched_rules else pd.DataFrame()

    def _get_sl_multiplier(self) -> float:
        """Определяет множитель стоп-лосса в зависимости от типа инструмента."""
        if self.symbol.startswith('#'):
            return SL_MULTIPLIER['#']  #  акции
        return SL_MULTIPLIER['rfd']  # форекс

    def _check_pyramid(self, active_rules: pd.DataFrame, current_price: float):
        """
        Проверяет возможность добавления к открытой позиции (пирамидинг).
        """
        if len(active_rules) == 0 or self.position['pyramid_level'] >= MAX_PYRAMID_LEVELS:
            return
        # Ищем сигналы в ту же сторону, что и открытая позиция
        dir_rules = active_rules[active_rules['direction'] ==
                                 ('UP' if self.position['type'] == 'LONG' else 'DOWN')]
        if len(dir_rules) > 0:
            self.position = self.pos_manager.pyramid(self.position, current_price, multiplier=0.5)

    def _close_position(self, row: pd.Series, idx: int):
        """
        Физическое закрытие позиции и добавление сделки в список trades.
        """
        # Определяем итоговую цену выхода:
        # Если есть переопределение (сработал SL/TP), используем его.
        # Иначе используем цену закрытия бара (для выхода по сигналу/времени).
        has_override = 'exit_price_override' in self.position
        # 2. Извлекаем цену (теперь ключ удалится из self.position)
        final_exit_price = self.position.pop('exit_price_override', row['close'])
        if has_override and self.position['sl'] is not None:
            is_sl = False
            if self.position['type'] == 'LONG':
                is_sl = (final_exit_price <= self.position['sl'] + 1e-9)
            else:
                is_sl = (final_exit_price >= self.position['sl'] - 1e-9)
            if is_sl:
                self.total_sl_hits += 1

        # Расчет PnL (profit and loss)
        pnl = self.pos_manager.calculate_pnl(self.position, final_exit_price)

        trade = Trade(
            entry_time=self.position['entry_time'],
            entry_price=self.position['entry'],
            exit_time=cast(pd.Timestamp, row.name),
            exit_price=final_exit_price,
            size=self.position['size'],
            pnl=pnl,
            win=pnl > 0,
            rule=self.position['rule'],
            pyramid_level=self.position['pyramid_level'],
            stop_loss=self.position['sl']
        )

        self.trades.append(trade)
        self.capital += pnl

        self.position = None

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR"""
        return pd.Series(
            talib.ATR(df['high'], df['low'], df['close'], period),
            index=df.index
        )

    def reset(self):
        """Сброс состояния"""
        self.rules = None
        self.capital = INITIAL_CAPITAL
        self.trades = []
        self.position = None
        self.total_sl_hits = 0
        self.equity_history = []

    def _prepare_vectorized_rules(self, features_columns: pd.Index):
        """
        Превращает текстовые правила в бинарную матрицу NumPy для мгновенных расчетов.
        """
        if self.rules.empty:
            self.rule_matrix = np.array([])
            return

        # Создаем пустую матрицу: [кол-во правил x кол-во признаков]
        num_rules = len(self.rules)
        num_features = len(features_columns)
        self.rule_matrix = np.zeros((num_rules, num_features), dtype=np.int8)

        # Составляем карту соответствия: имя фичи -> индекс столбца
        feature_to_idx = {name: i for i, name in enumerate(features_columns)}

        for i, (idx, rule) in enumerate(self.rules.iterrows()):
            rule_name = rule['rule_name']
            # Используем вашу логику очистки имен
            clean_name = rule_name.replace('_prev_', '|').replace('_curr_', '|')
            needed_features = [p.strip('_') for p in clean_name.split('|') if p]

            for feat in needed_features:
                if feat in feature_to_idx:
                    self.rule_matrix[i, feature_to_idx[feat]] = 1

        # Считаем, сколько признаков должно совпасть для каждого правила
        self.rule_requirements = self.rule_matrix.sum(axis=1)

    def get_active_rules_fast(self, features_row_values: np.ndarray) -> pd.DataFrame:
        """
        Векторизованная проверка правил через NumPy.
        features_row_values: одномерный массив (0 и 1) текущего бара.
        """
        if self.rule_matrix.size == 0:
            return pd.DataFrame()

        # Магическая строка: перемножаем матрицу правил на вектор текущих фич
        # Результат: сколько признаков каждого правила сработало на этом баре
        matched_counts = np.dot(self.rule_matrix, features_row_values)

        # Сравниваем: если кол-во сработавших == кол-во требуемых, правило активно
        active_indices = np.where(matched_counts == self.rule_requirements)[0]

        if len(active_indices) == 0:
            return pd.DataFrame()

        return self.rules.iloc[active_indices]

    def run_backtest(self, df: pd.DataFrame, features: pd.DataFrame, symbol: str,
                     timeframe: str, exit_mode: str = "SIGNAL_TO_SIGNAL", use_sl=None,
                     rules_data: Dict = None) -> Dict:
        # 1. Синхронизация данных
        common_index = df.index.intersection(features.index)
        df = df.loc[common_index].copy()
        features = features.loc[common_index].copy()

        self.exit_mode, self.symbol, self.timeframe, self.current_sl_mode = exit_mode, symbol, timeframe, use_sl
        self.reset()

        data = rules_data if rules_data is not None else self.load_rules(symbol, timeframe)
        if data is None:
            return {'error': 'Нет правил (ни в памяти, ни в кэше)'}
        # Если загрузился словарь
        if isinstance(data, dict):
            # Поддерживаем оба ключа 'all_rules' (из памяти) и 'top_rules' (из кэша)
            self.rules = data.get('all_rules') if 'all_rules' in data else data.get('top_rules')
            self.current_min_conf = data.get('min_confidence', "N/A")
        else:
            self.rules = data
            self.current_min_conf = "N/A"

        rules_count = len(self.rules) if not self.rules.empty else 0
        if self.rules.empty: return {'error': 'Нет правил'}

        # 2. Подготовка векторизации
        self._prepare_vectorized_rules(features.columns)

        # Переводим всё в NumPy массивы (обращение к ним в сотни раз быстрее iloc)
        feat_values = features.values.astype(np.int8)
        atr_values = self.calculate_atr(df).values
        close_prices = df['close'].values
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        time_values = df['time'].values if 'time' in df.columns else df.index.values

        # МГНОВЕННЫЙ ПРЕДРАССЧЕТ СИГНАЛОВ (Матричное умножение)
        match_matrix = np.dot(feat_values, self.rule_matrix.T)
        signals_mask = (match_matrix == self.rule_requirements)

        # 3. ОСНОВНОЙ ЦИКЛ (Чистый NumPy)
        for i in range(200, len(df)):
            # Получаем правила через маску (быстрее, чем get_active_rules)
            active_idx = np.where(signals_mask[i])[0]
            active_rules = self.rules.iloc[active_idx] if active_idx.size > 0 else pd.DataFrame()

            # ВЫЗОВ ОПТИМИЗИРОВАННОГО ОБРАБОТЧИКА
            self._process_bar_optimized(
                close_prices[i], open_prices[i], high_prices[i], low_prices[i],
                time_values[i], active_rules, atr_values[i], i
            )

        # 4. ФИНАЛИЗАЦИЯ
        if self.position:
            # Для закрытия последнего бара создаем минимальный Series
            last_row = pd.Series({'close': close_prices[-1]}, name=time_values[-1])
            self._close_position(last_row, len(df) - 1)

        start_date = pd.to_datetime(time_values[0]).strftime('%d-%m-%Y')
        end_date = pd.to_datetime(time_values[-1]).strftime('%d-%m-%Y')
        period = f"[{start_date} → {end_date}]"

        calculator = MetricsCalculator(verbose=self.verbose)
        metrics = calculator.calculate(
            self.trades, INITIAL_CAPITAL, rules_count,
            sl_hits=self.total_sl_hits,
            equity_history=self.equity_history,
            use_sl=self.current_sl_mode
        )
        calculator.print_metrics(metrics, symbol, timeframe, exit_mode, period,  min_conf=self.current_min_conf)
        return metrics

    def _process_bar_optimized(self, close: float, open_p: float, high: float, low: float,
                               timestamp, active_rules: pd.DataFrame, atr: float, idx: int):
        # Вход
        if not self.position:
            if not active_rules.empty:
                self._check_entry_optimized(close, timestamp, active_rules, atr, idx)
        # Управление позицией
        else:
            # 1. Сначала проверяем выход (Stop Loss, TP, Сигналы)
            if self._check_exit_optimized(close, open_p, high, low, active_rules, atr, idx):
                fake_row = pd.Series({'close': close}, name=timestamp)
                self._close_position(fake_row, idx)
            else:
                # 2. Если не вышли — проверяем пирамидинг
                if not active_rules.empty:
                    self._check_pyramid(active_rules, close)

        # 3. Расчет эквити (Floating DD) на чистых числах
        unrealized = self.pos_manager.calculate_unrealized_pnl(self.position, close)
        self.equity_history.append(self.capital + unrealized)

    def _check_entry_optimized(self, close: float, timestamp, active_rules: pd.DataFrame, atr: float, idx: int):
        if 'direction' not in active_rules.columns: return

        buy_rules = active_rules[active_rules['direction'] == 'UP']
        sell_rules = active_rules[active_rules['direction'] == 'DOWN']

        sl_mult = self._get_sl_multiplier()
        risk_amount = self.capital * RISK_PER_TRADE

        # Защита от деления на 0 при отключенном стопе
        effective_sl_mult = sl_mult if sl_mult > 0 else 1.0
        size = risk_amount / (atr * SL_ATR_MULTIPLIER * effective_sl_mult)

        if not buy_rules.empty:
            rule = buy_rules.loc[buy_rules['lift'].idxmax()]
            self.position = self.pos_manager.create_long(
                close, atr, size, pd.Timestamp(timestamp), idx, rule['rule_name'], sl_mult)
        elif not sell_rules.empty:
            rule = sell_rules.loc[sell_rules['lift'].idxmax()]
            self.position = self.pos_manager.create_short(
                close, atr, size, pd.Timestamp(timestamp), idx, rule['rule_name'], sl_mult)

    def _check_exit_optimized(self, close: float, open_p: float, high: float, low: float,
                              active_rules: pd.DataFrame, atr: float, idx: int) -> bool:
        # Проверка Stop Loss
        if self.position['sl'] is not None:
            if self.position['type'] == 'LONG':
                if open_p <= self.position['sl']:
                    self.position['exit_price_override'] = open_p
                    return True
                if low <= self.position['sl']:
                    self.position['exit_price_override'] = self.position['sl']
                    return True
            else:  # SHORT
                if open_p >= self.position['sl']:
                    self.position['exit_price_override'] = open_p
                    return True
                if high >= self.position['sl']:
                    self.position['exit_price_override'] = self.position['sl']
                    return True

        # Сигналы выхода (SIGNAL_TO_SIGNAL)
        if self.exit_mode == "SIGNAL_TO_SIGNAL" and not active_rules.empty:
            opp_dir = 'DOWN' if self.position['type'] == 'LONG' else 'UP'
            if not active_rules[active_rules['direction'] == opp_dir].empty:
                return True

        # ONE_CANDLE
        if self.exit_mode == "ONE_CANDLE":
            return idx >= self.position['entry_idx'] + 1

        # ATR_TP
        if self.exit_mode == "ATR_TP":
            tp_dist = atr * TP_ATR_MULTIPLIER
            if self.position['type'] == 'LONG' and high >= self.position['entry'] + tp_dist:
                self.position['exit_price_override'] = self.position['entry'] + tp_dist
                return True
            if self.position['type'] == 'SHORT' and low <= self.position['entry'] - tp_dist:
                self.position['exit_price_override'] = self.position['entry'] - tp_dist
                return True

        return False


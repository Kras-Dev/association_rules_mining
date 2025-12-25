""" Основной класс для запуска симуляции торговых стратегий (бэктестинга)"""

from pathlib import Path
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

    def run_backtest(self, df: pd.DataFrame, features: pd.DataFrame, symbol: str,
                     timeframe: str, exit_mode: str = "SIGNAL_TO_SIGNAL",
                     ) -> Dict:
        """
        Запускает основной цикл бэктестинга.

        Args:
            df (pd.DataFrame): DataFrame с ценами (OHLCV).
            features (pd.DataFrame): DataFrame с предрассчитанными признаками.
            symbol (str): Торгуемый инструмент.
            timeframe (str): Таймфрейм.
            exit_mode (str, optional): Режим выхода из сделки.

                "SIGNAL_TO_SIGNAL". "ONE_CANDLE". "ATR_TP".

        Returns:
            Dict: Словарь с метриками производительности стратегии.
        """
        # СИНХРОНИЗАЦИЯ. Оставляем только те строки, которые есть в обоих датафреймах
        common_index = df.index.intersection(features.index)
        df = df.loc[common_index].copy()
        features = features.loc[common_index].copy()

        self.exit_mode = exit_mode
        self.reset()
        self.symbol = symbol
        self.timeframe = timeframe

        self._log_info(f"{symbol} {timeframe} | {exit_mode}")
        self.rules = self.load_rules(symbol, timeframe)
        rules_count = len(self.rules) if not self.rules.empty else 0
        # --- Проверка наличия правил/сигналов ---
        if self.rules.empty:
            self._log_warning(f"⚠️ Нет правил для {symbol} {timeframe}, тест пропущен.")
            return {'error': 'Нет правил'}

        self._log_info(f"Используем {features.shape[1]} готовых фич")

        # Быстрая проверка сигналов для оценки перспективности
        signal_count = 0
        limit = min(1000, len(df))
        for i in tqdm(range(limit), desc="Signals", miniters=100, disable=not self.verbose):
            active_rules = self.get_active_rules(features.iloc[i])
            if not active_rules.empty and 'direction' in active_rules.columns:
                signal_count += 1

        if signal_count == 0:
            self._log_warning(f"⚠️ Нет сигналов для теста {symbol} {timeframe}, тест пропущен.")
            return {'error': 'Нет сигналов'}

        # --- Основной цикл симуляции ---
        # Расчет ATR (Average True Range) для управления рисками
        atr = self.calculate_atr(df)
        desc = f"💹 Backtest {symbol[:6]}" if self.verbose else None
        with tqdm(total=len(df) - 200, desc=desc, miniters=500, leave=self.verbose, disable=not self.verbose) as pbar:
            for i in range(200, len(df)):
                self._process_bar(df.iloc[i], features.iloc[i], atr.iloc[i], i)
                if self.verbose:
                    pbar.update(1)
                    pbar.set_postfix({
                        'Capital': f"${self.capital:.0f}",
                        'Trades': len(self.trades),
                        'Pos': 'YES' if self.position else 'NO'
                    })
        # --- Финализация результатов ---
        if self.position:
            self._close_position(df.iloc[-1], len(df) - 1)
        # Форматирование периода теста
        start_date = df.iloc[0]['time'].strftime('%d-%m-%Y')
        end_date = df.iloc[-1]['time'].strftime('%d-%m-%Y')
        period = f"[{start_date} → {end_date}]"
        # Расчет и вывод метрик
        calculator = MetricsCalculator(verbose=self.verbose)
        metrics = calculator.calculate(self.trades, INITIAL_CAPITAL, rules_count,
                                       sl_hits=self.total_sl_hits, equity_history=self.equity_history)
        calculator.print_metrics(metrics, symbol, timeframe, exit_mode, period, rules_count)
        return metrics

    def _process_bar(self, row: pd.Series, features_row: pd.Series, atr: float, idx: int):
        """
        Обрабатывает один бар в цикле бэктеста.

        Args:
            row (pd.Series): Текущий бар цен.
            features_row (pd.Series): Текущие признаки.
            atr (float): Текущее значение ATR.
            idx (int): Индекс бара.
        """
        active_rules = self.get_active_rules(features_row)

        # Вход
        if not self.position:
            self._check_entry(row, active_rules, atr, idx)
        # --- Управление открытой позицией ---
        else:
            # 1. Сначала проверяем выход
            if self._check_exit(row, features_row, active_rules, atr, idx):
                self._close_position(row, idx)
            else:
                # 2. Если не вышли, тогда возможен пирамидинг
                self._check_pyramid(active_rules, row['close'])
                # После обработки бара, сохраняем текущий баланс + плавающий PnL

        current_unrealized_pnl = self.pos_manager.calculate_unrealized_pnl(self.position, row['close'])
        current_equity = self.capital + current_unrealized_pnl
        self.equity_history.append(current_equity)

    def _get_sl_multiplier(self) -> float:
        """Определяет множитель стоп-лосса в зависимости от типа инструмента."""
        if self.symbol.startswith('#'):
            return SL_MULTIPLIER['#']  # 1.5 акции
        return SL_MULTIPLIER['rfd']  # 1.2 форекс

    def _check_entry(self, row: pd.Series, active_rules: pd.DataFrame, atr: float, idx: int):
        """
        Проверяет условия для входа в позицию (Long/Short).
        """
        if active_rules.empty or len(active_rules) == 0:
            return
        # ПРОВЕРКА КОЛОНОК!
        if 'direction' not in active_rules.columns:
            self._log_error(f"⚠️ Нет колонки 'direction' в {len(active_rules)} правилах")
            return
        # --- Расчет размера позиции ---
        buy_rules = active_rules[active_rules['direction'] == 'UP']
        sell_rules = active_rules[active_rules['direction'] == 'DOWN']

        sl_mult = self._get_sl_multiplier()
        risk_amount = self.capital * RISK_PER_TRADE
        sl_mult_effective = sl_mult if sl_mult > 0 else 1.0
        denom = (atr * SL_ATR_MULTIPLIER * sl_mult_effective)
        size = risk_amount / denom if denom != 0 else 0
        # --- Вход в Long/Short по правилу с максимальным 'lift' (силой) ---
        if len(buy_rules) > 0:
            rule = buy_rules.loc[buy_rules['lift'].idxmax()]
            self.position = self.pos_manager.create_long(
                row['close'], atr, size, cast(pd.Timestamp, row.name), idx, rule['rule_name'], sl_mult)

        elif len(sell_rules) > 0:
            rule = sell_rules.loc[sell_rules['lift'].idxmax()]
            self.position = self.pos_manager.create_short(
                row['close'], atr, size, cast(pd.Timestamp, row.name), idx, rule['rule_name'], sl_mult)

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

    def _check_exit(self, row: pd.Series, features_row: pd.Series,
                    active_rules: pd.DataFrame, atr: float, idx: int) -> bool:
        """
        Проверяет условия выхода из позиции.

        Обрабатывает Stop Loss, Take Profit и выходы по противоположному сигналу.
        """
        # Если стопа нет, пропускаем блок проверок SL по теням
        if self.position['sl'] is not None:
            # --- Всегда проверяем Stop Loss (по теням свечи) для всех режимов---
            if self.position['type'] == 'LONG':
                # Обработка гэпа: если Open уже ниже SL, закрываем по Open
                if row['open'] <= self.position['sl']:
                    self.position['exit_price_override'] = row['open']
                    return True
                # Касание SL внутри бара: закрываем по цене SL
                if row['low'] <= self.position['sl']:
                    self.position['exit_price_override'] = self.position['sl']
                    return True
            else:  # SHORT
                # Обработка гэпа: если Open уже выше SL, закрываем по Open
                if row['open'] >= self.position['sl']:
                    self.position['exit_price_override'] = row['open']
                    return True
                # Касание SL внутри бара: закрываем по цене SL
                if row['high'] >= self.position['sl']:
                    self.position['exit_price_override'] = self.position['sl']
                    return True

        if active_rules.empty or 'direction' not in active_rules.columns:
            # Если нет правил выхода - проверяем только TP/ONE_CANDLE
            pass
        # --- Проверка выхода по противоположному сигналу (SIGNAL_TO_SIGNAL) ---
        else:
            if self.exit_mode == "SIGNAL_TO_SIGNAL":
                opp_rules = active_rules[active_rules['direction'] !=
                                         ('UP' if self.position['type'] == 'LONG' else 'DOWN')]
                if len(opp_rules) > 0:
                    return True
        # --- Проверка других режимов выхода (ONE_CANDLE, ATR_TP) ---
        # Остальные режимы (НЕ зависят от active_rules)
        if self.exit_mode == "ONE_CANDLE":
            # Выход через 1 бар после входа
            return idx >= self.position['entry_idx'] + 1

        elif self.exit_mode == "ATR_TP":
            # Выход по Take Profit, рассчитанному на базе ATR
            tp_dist = atr * TP_ATR_MULTIPLIER
            if self.position['type'] == 'LONG':
                if row['high'] >= self.position['entry'] + tp_dist:
                    # Устанавливаем цену выхода как уровень TP
                    self.position['exit_price_override'] = self.position['entry'] + tp_dist
                    return True

            if self.position['type'] == 'SHORT':
                if row['low'] <= self.position['entry'] - tp_dist:
                    # Устанавливаем цену выхода как уровень TP
                    self.position['exit_price_override'] = self.position['entry'] - tp_dist
                    return True

        return False

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

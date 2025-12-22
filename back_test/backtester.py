"""🎯 Основной бэктестер"""

import os
import pickle
import pandas as pd
import talib
from typing import Dict, Optional
from tqdm import tqdm
from association_miner.features_engineer import Features
from back_test.config import *
from back_test.trade import PositionManager, Trade
from back_test.metrics import MetricsCalculator


class Backtester:
    """🔥 Бэктестер стратегии паттернов"""

    def __init__(self, symbol: str):
        self.rules: pd.DataFrame = None
        self.capital = INITIAL_CAPITAL
        self.trades: list[Trade] = []
        self.position = None
        self.exit_mode = "SIGNAL_TO_SIGNAL"
        self.symbol = symbol
        self.timeframe = None

    def load_rules(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Загрузка правил"""
        cache_file = f"models/rules_{symbol}_{timeframe}.pkl"
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"[BackTester]: ❌ Нет {cache_file}")

        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)

        rules = cache['top_rules'][cache['top_rules']['confidence'] > MIN_CONFIDENCE]
        print(f"[BackTester]: ✅ {len(rules)} правил >{MIN_CONFIDENCE:.0%} conf")

        return rules

    def get_active_rules(self, features_row: pd.Series) -> pd.DataFrame:
        """🔍 АКТИВНЫЕ ПРАВИЛА - СОХРАНЯЕМ КОЛОНКИ!"""
        matched_rules = []

        for idx, rule in self.rules.iterrows():
            rule_name = rule['rule_name']
            matched_features = []

            # Матчинг по словам (минимум 2 совпадения)
            rule_words = rule_name.split('_')
            for word in rule_words:
                if word != 'prev' and features_row.get(word, 0) == 1:
                    matched_features.append(word)

            if len(matched_features) >= 2:
                # ✅ СОХРАНЯЕМ ВСЮ СТРОКУ С КОЛОНКАМИ!
                matched_rules.append(rule.to_dict())

        if matched_rules:
            df_result = pd.DataFrame(matched_rules)
            return df_result
        else:
            return pd.DataFrame()  # ✅ ПУСТОЙ DataFrame с колонками!

    def run_backtest(self, df: pd.DataFrame, symbol: str, timeframe: str,
                     exit_mode: str = "SIGNAL_TO_SIGNAL") -> Optional[Dict]:
        """🔥 Запуск бэктеста С ПРОВЕРКОЙ СИГНАЛОВ"""
        self.exit_mode = exit_mode
        self.reset()
        self.symbol = symbol
        self.timeframe = timeframe

        print(f"\n[BackTester]: {symbol} {timeframe} | {exit_mode}")
        print(f"[BackTester]:📊 {len(df)} свечей")

        # ✅ ШАГ 1: БЫСТРАЯ ПРОВЕРКА СИГНАЛОВ
        print("[BackTester]: Проверяем наличие сигналов...")

        # Загружаем правила и фичи ОДИН РАЗ
        self.rules = self.load_rules(symbol, timeframe)
        if self.rules.empty:
            print(f"[BackTester]: ❌ {symbol} {timeframe} | Нет правил >{MIN_CONFIDENCE:.0%} conf")
            return {'error': 'Нет правил'}

        features = Features(verbose=False).create_all_features(df)

        # ✅ БЫСТРАЯ СКАННИНГ (только проверка сигналов!)
        signal_count = 0
        for i in range(200, min(1000, len(df))):  # Проверяем только 800 баров!
            active_rules = self.get_active_rules(features.iloc[i])
            if not active_rules.empty and 'direction' in active_rules.columns:
                signal_count += 1

        print(f"[BackTester]: ✅ Найдено {signal_count} потенциальных сигналов")

        if signal_count == 0 :
            print(f"[BackTester]: ❌ {symbol} {timeframe} | {exit_mode} | Нет сделок (SKIP)")
            return {'error': 'Нет сигналов'}

        # ✅ ШАГ 2: ПОЛНЫЙ БЭКТЕСТ (только если есть сигналы!)

        atr = self.calculate_atr(df)
        with tqdm(total=len(df) - 200,
                  desc=f"{symbol} {timeframe} {exit_mode}",
                  position=0, leave=False,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            for i in range(200, len(df)):
                self._process_bar(df.iloc[i], features.iloc[i], atr.iloc[i], i)
                pbar.update(1)
                pbar.set_postfix({
                    'Capital': f"${self.capital:.0f}",
                    'Trades': len(self.trades),
                    'Pos': 'YES' if self.position else 'NO'
                }, refresh=False)

        # Закрываем позицию
        if self.position:
            self._close_position(df.iloc[-1], len(df) - 1)

        metrics = MetricsCalculator.calculate(self.trades, self.capital)
        metrics['capital'] = self.capital
        MetricsCalculator.print_metrics(metrics, symbol, timeframe, exit_mode)
        return metrics


    def _process_bar(self, row: pd.Series, features_row: pd.Series, atr: float, idx: int):
        """Обработка одного бара"""
        active_rules = self.get_active_rules(features_row)

        # Вход
        if not self.position:
            self._check_entry(row, active_rules, atr, idx)
        else:
            # Pyramid
            self._check_pyramid(active_rules)
            # Выход
            if self._check_exit(row, features_row, active_rules, atr, idx):
                self._close_position(row, idx)

    def _get_sl_multiplier(self) -> float:
        if self.symbol.startswith('#'):
            return SL_MULTIPLIER['#']  # 1.5 акции
        return SL_MULTIPLIER['rfd']  # 1.2 форекс

    def _check_entry(self, row: pd.Series, active_rules: pd.DataFrame, atr: float, idx: int):
        """Проверка входа"""

        if active_rules.empty or len(active_rules) == 0:
            return

        # ✅ ПРОВЕРКА КОЛОНОК!
        if 'direction' not in active_rules.columns:
            print(f"⚠️ Нет колонки 'direction' в {len(active_rules)} правилах")
            return
        buy_rules = active_rules[active_rules['direction'] == 'UP']
        sell_rules = active_rules[active_rules['direction'] == 'DOWN']

        sl_mult = self._get_sl_multiplier()

        risk_amount = self.capital * RISK_PER_TRADE
        size = risk_amount / (atr * SL_ATR_MULTIPLIER * sl_mult)

        if len(buy_rules) > 0:
            rule = buy_rules.loc[buy_rules['lift'].idxmax()]
            self.position = PositionManager.create_long(
                row['close'], atr, size, row.name, idx, rule['rule_name'])


        elif len(sell_rules) > 0:
            rule = sell_rules.loc[sell_rules['lift'].idxmax()]
            self.position = PositionManager.create_short(
                row['close'], atr, size, row.name, idx, rule['rule_name'])


    def _check_pyramid(self, active_rules: pd.DataFrame):
        """Проверка pyramid"""
        if len(active_rules) == 0 or self.position['pyramid_level'] >= MAX_PYRAMID_LEVELS:
            return

        dir_rules = active_rules[active_rules['direction'] ==
                                 ('UP' if self.position['type'] == 'LONG' else 'DOWN')]
        if len(dir_rules) > 0:
            self.position = PositionManager.pyramid(self.position)

    def _check_exit(self, row: pd.Series, features_row: pd.Series,
                    active_rules: pd.DataFrame, atr: float, idx: int) -> bool:
        """Проверка выхода - ПО ТЕНЯМ!"""

        # SL по ТЕНЯМ для ВСЕХ режимов
        if self.position['type'] == 'LONG':
            sl_hit = row['low'] <= self.position['sl']
        else:
            sl_hit = row['high'] >= self.position['sl']

        if sl_hit:
            return True

        if active_rules.empty or 'direction' not in active_rules.columns:
            # ✅ Если нет правил выхода - проверяем только TP/ONE_CANDLE
            pass
        else:
            if self.exit_mode == "SIGNAL_TO_SIGNAL":
                opp_rules = active_rules[active_rules['direction'] !=
                                         ('UP' if self.position['type'] == 'LONG' else 'DOWN')]
                if len(opp_rules) > 0:
                    return True

        # Остальные режимы (НЕ зависят от active_rules)
        if self.exit_mode == "ONE_CANDLE":
            return idx >= self.position['entry_idx'] + 1

        elif self.exit_mode == "ATR_TP":
            tp_dist = atr * TP_ATR_MULTIPLIER
            if self.position['type'] == 'LONG':
                return row['high'] >= self.position['entry'] + tp_dist
            return row['low'] <= self.position['entry'] - tp_dist

        return False

    def _close_position(self, row: pd.Series, idx: int):
        """Закрытие позиции"""
        pnl = PositionManager.calculate_pnl(self.position, row['close'])

        trade = Trade(
            entry_time=self.position['entry_time'],
            entry_price=self.position['entry'],
            exit_time=row.name,
            exit_price=row['close'],
            size=self.position['size'],
            pnl=pnl,
            win=pnl > 0,
            rule=self.position['rule'],
            pyramid_level=self.position['pyramid_level']
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

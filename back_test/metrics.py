"""Модуль расчёта торговых метрик и формирования отчётов"""
import pandas as pd
import numpy as np
from typing import Dict, List

from back_test.trade import Trade
from utils.base_logger import BaseLogger


class MetricsCalculator(BaseLogger):
    """
    Калькулятор торговых метрик.

    Выполняет математический анализ списка сделок, рассчитывает показатели
    эффективности (Profit Factor, Win Rate, Drawdown) и выводит отчёты.
    """

    def __init__(self, verbose: bool = False):
        """
        Args:
            verbose (bool): Если True, разрешает детальный вывод в консоль.
        """
        super().__init__(verbose)
        self.verbose = verbose


    def calculate(self, trades: List[Trade], initial_capital: float,
                  rules_count: int = 0,  sl_hits: int = 0) -> Dict:
        """
        Рассчитывает основные показатели эффективности стратегии.

        Args:
            trades (List[Trade]): Список объектов завершенных сделок.
            initial_capital (float): Стартовый капитал.
            rules_count (int): Количество правил, участвовавших в генерации сигналов.

        Returns:
            Dict: Словарь со всеми рассчитанными метриками.
        """
        # --- Базовая проверка на наличие данных ---
        if not trades:
            return {'error': 'Нет сделок', 'total_trades': 0, 'final_capital': initial_capital}
        # --- Подготовка данных. ---
        # Превращаем список объектов сделок в таблицу для быстрых расчетов
        trades_df = pd.DataFrame([t.__dict__ for t in trades])
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] < 0]

        # --- Расчет кривой эквити и просадок ---
        # Equity curve начинается со стартового капитала
        equity = np.cumsum([initial_capital] + [t.pnl for t in trades])
        peak = np.maximum.accumulate(equity)

        # Просадка в процентах от пика
        drawdown = (equity - peak) / peak * 100

        # --- Расчет средних показателей и соотношений ---
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0

        # Соотношение риск/прибыль (Risk/Reward)
        rr_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0

        # Общий финансовый результат
        # PnL% (profit and loss)
        pnl_pct = ((equity[-1] / initial_capital - 1) * 100)
        total_pnl = trades_df['pnl'].sum()

        # Максимальная просадка в денежном выражении (для Recovery Factor)
        max_dd_money = (peak - equity).max()
        recovery_factor = total_pnl / max_dd_money if max_dd_money > 0 else 0
        # --- Сборка итогового словаря метрик ---
        return {
            'total_trades': len(trades_df),
            'win_rate': len(wins) / len(trades_df) if len(trades_df) > 0 else 0,
            'profit_factor': wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else float('inf'),
            'total_pnl': trades_df['pnl'].sum(),
            'total_pnl_pct': round(pnl_pct, 2),
            'final_capital': equity[-1],
            'max_dd_pct': abs(drawdown.min()) if len(drawdown) > 0 else 0,
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'rr_ratio': round(rr_ratio, 2),
            'best_trade': trades_df['pnl'].max() if len(trades_df) > 0 else 0,
            'worst_trade': trades_df['pnl'].min() if len(trades_df) > 0 else 0,
            'rules_count': rules_count,
            'recovery_factor': round(recovery_factor, 2),
            'sl_hits': sl_hits,
        }

    def print_metrics(self, metrics: Dict, symbol: str, tf: str, mode: str, period: str="", rules_count: int = 0):
        """
        Выводит отчёт в консоль в человекочитаемом виде.

        Args:
            metrics (Dict): Результаты метода calculate.
            symbol (str): Тикер инструмента.
            tf (str): Таймфрейм.
            mode (str): Режим выхода.
            period (str): Строка временного диапазона теста.
            rules_count (int): Кол-во правил (используется как fallback).
        """
        # --- Обработка ошибок в метриках ---
        if 'error' in metrics:
            self._log_error(f"❌ {symbol} {tf} | {mode} | {metrics['error']}")
            return

        total_trades = metrics.get('total_trades', 0)
        actual_rules = metrics.get('rules_count', rules_count)

        # --- Валидация качества данных ---
        # Если сделки есть, а правил в кэше ноль — это подозрительный результат
        if total_trades > 0 and actual_rules == 0:
            self._log_warning(f"⚠️  ВНИМАНИЕ: {total_trades} сделок при 0 правил! Проверьте кэш.")
        # Если сигналов не было — краткий вывод
        if total_trades == 0:
            print(f"\n📊 {symbol} {tf} | {mode}")
            print("-" * 60)
            print("❌ NO SIGNALS (0 trades)")
            return

        # --- Формирование красивого отчёта ---
        rules = metrics.get('rules_count', rules_count)
        period_str = f" | {period}" if period.strip() else ""

        print(f"\n📊 {symbol} {tf} | {mode}{period_str} | правил: {rules}")
        print("-" * 60)
        # 💰 Final Capital: итоговый капитал (абсолют $) + % прироста от стартового
        print(f"💰 Final Capital:   ${metrics['final_capital']:.2f} ({metrics['total_pnl_pct']:.1f}%)")
        # 📈 Profit Factor: сумма профитов/сумма лоссов | RR: средний профит/средний лосс
        print(f"📈 Profit Factor:   {metrics['profit_factor']:.2f} | RR: {metrics['rr_ratio']:.2f}")
        # 🎯 Win Rate: % прибыльных сделок (кол-во всех сделок)
        print(f"🎯 Win Rate:        {metrics['win_rate'] * 100:.1f}% ({metrics['total_trades']} сделок)")
        # 📉 Max DD: максимальная просадка капитала (% от пика)
        print(f"📉 Max DD:          {metrics['max_dd_pct']:.1f}%")
        # ⭐ Best: самая прибыльная сделка ($) | 💥 Worst: самая убыточная сделка ($)
        print(f"⭐ Best:            ${metrics['best_trade']:.2f}")
        print(f"💥 Worst:           ${metrics['worst_trade']:.2f}")
        # 🛡️ Коэффициент восстановления (RF)
        print(f"🛡️ Recovery Factor: {metrics['recovery_factor']:.2f}")
        # 📊 Avg Win/Loss: средний профит выигрышей / средний лосс проигрышей
        print(f"📊 Avg Win/Loss:    ${metrics['avg_win']:.2f} / ${metrics['avg_loss']:.2f}")
        print(f" sl_hits:            {metrics['sl_hits']}")

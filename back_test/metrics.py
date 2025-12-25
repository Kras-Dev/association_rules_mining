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

    def calculate(self, trades: List[Trade], initial_capital: float,
                  rules_count: int = 0, sl_hits: int = 0, equity_history: List[float]=None,
                  use_sl: bool=None) -> Dict:
        """
        Рассчитывает основные показатели эффективности стратегии.

        Args:
            trades (List[Trade]): Список объектов завершенных сделок.
            initial_capital (float): Стартовый капитал.
            rules_count (int): Количество правил, участвовавших в генерации сигналов.
            sl_hits (int):
            equity_history (List[float]):
            use_sl (bool) :

        Returns:
            Dict: Словарь со всеми рассчитанными метриками.
        """
        # --- Базовая проверка ---
        if not trades:
            return {'error': 'Нет сделок', 'total_trades': 0, 'final_capital': initial_capital}

        # Превращаем сделки в DataFrame для расчетов
        trades_df = pd.DataFrame([t.__dict__ for t in trades])
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] < 0]

        # 1. РАСЧЕТ ПРОСАДОК (DRAWDOWN)
        # Floating DD: на основе поминутной/побарной истории эквити (самая точная)
        max_floating_dd = self._calculate_max_drawdown(equity_history) if equity_history else 0.0

        # Equity DD: только по точкам закрытия сделок (традиционная)
        equity_curve_closed = np.cumsum([initial_capital] + [t.pnl for t in trades])
        max_equity_dd = self._calculate_max_drawdown(equity_curve_closed.tolist())

        # 2. РАСЧЕТ ФИНАНСОВЫХ ПОКАЗАТЕЛЕЙ
        total_pnl = trades_df['pnl'].sum()
        final_capital = equity_curve_closed[-1]
        pnl_pct = ((final_capital / initial_capital - 1) * 100)

        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        rr_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0

        # 3. RECOVERY FACTOR (используем денежную просадку из Floating истории)
        # Берем историю эквити (если есть) или кривую по закрытым
        ref_equity = np.array(equity_history) if equity_history else equity_curve_closed
        max_dd_money = (np.maximum.accumulate(ref_equity) - ref_equity).max()
        recovery_factor = total_pnl / max_dd_money if max_dd_money > 0 else 0

        return {
            'total_trades': len(trades_df),
            'win_rate': len(wins) / len(trades_df) if len(trades_df) > 0 else 0,
            'profit_factor': wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else float('inf'),
            'total_pnl': round(total_pnl, 2),
            'total_pnl_pct': round(pnl_pct, 2),
            'final_capital': round(final_capital, 2),
            'max_floating_dd': round(max_floating_dd, 2),
            'max_equity_dd': round(max_equity_dd, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'rr_ratio': round(rr_ratio, 2),
            'best_trade': round(trades_df['pnl'].max(), 2),
            'worst_trade': round(trades_df['pnl'].min(), 2),
            'rules_count': rules_count,
            'recovery_factor': round(recovery_factor, 2),
            'sl_hits': sl_hits,
            'sl_enabled': use_sl
        }

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Вспомогательная функция для расчета максимальной просадки."""
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak != 0 else 0.0
            if drawdown > max_dd:
                max_dd = drawdown
        return max_dd * 100.0  # В процентах

    def print_metrics(self, metrics: Dict, symbol: str, tf: str, mode: str,
                      period: str=""):
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

        # Определяем текстовую метку для стоп-лосса
        sl_label = "WITH SL" if metrics.get('sl_enabled') is True else "NO SL"
        rules = metrics.get('rules_count')
        period_str = f" {period}" if period.strip() else ""

        # --- Валидация качества данных ---
        # Если сделки есть, а правил в кэше ноль — это подозрительный результат
        if total_trades > 0 and rules == 0:
            self._log_warning(f"⚠️  ВНИМАНИЕ: {total_trades} сделок при 0 правил! Проверьте кэш.")
        # Если сигналов не было — краткий вывод
        if total_trades == 0:
            print(f"\n📊 {symbol} {tf} | {mode} {sl_label} | {period_str} | правил: {rules}")
            print("-" * 60)
            print("❌ NO SIGNALS (0 trades)")
            return

        # --- Формирование красивого отчёта ---

        print(f"\n📊 {symbol} {tf} | {mode} {sl_label} | {period_str} | правил: {rules}")
        print("-" * 80)

        # Основные финансовые показатели
        print(
            f"💰 Final Capital:   ${metrics['final_capital']} ({metrics['total_pnl_pct']}%) — Итоговый капитал и чистая прибыль в %")
        print(
            f"📈 Profit Factor:   {metrics['profit_factor']:.2f} — Отношение общей прибыли к общему убытку (лучше > 1.5)")
        print(f"⚖️ RR Ratio:        {metrics['rr_ratio']} — Соотношение средний плюс / средний минус")
        print(
            f"🎯 Win Rate:        {metrics['win_rate']:.1%} (Всего сделок: {metrics['total_trades']}) — Процент прибыльных сделок")

        # Риски и просадки
        print(f"📉 Floating DD:     {metrics['max_floating_dd']}% — Худшая точка (плавающая просадка) за всю историю")
        print(f"📊 Equity DD:       {metrics['max_equity_dd']}% — Макс. просадка только по зафиксированным сделкам")

        # Сравнение сделок
        print(f"⭐ Best Trade:      ${metrics['best_trade']} — Самая прибыльная сделка")
        print(f"💥 Worst Trade:     ${metrics['worst_trade']} — Самая убыточная сделка")

        # Эффективность восстановления
        print(
            f"🛡️ Recovery Factor: {metrics['recovery_factor']} — Способность системы восстанавливаться после просадок (лучше > 1.0)")
        print(
            f"💵 Avg Win/Loss:    ${metrics['avg_win']} / ${metrics['avg_loss']} — Средний профит и средний лосс на сделку, {abs(metrics['avg_win']/metrics['avg_loss']):.2f}")

        # Статистика стопов
        print(f"🛑 SL Hits:         {metrics['sl_hits']} — Количество закрытий по Stop Loss")
        print("-" * 80)

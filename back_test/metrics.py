"""📊 Расчёт торговых метрик"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from back_test.trade import Trade

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Калькулятор метрик с verbose контролем"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log_info(self, message: str):
        if self.verbose:
            logger.info(message)

    def calculate(self, trades: List[Trade], initial_capital: float, rules_count: int = 0) -> Dict:
        """Основные метрики"""
        if not trades:
            return {'error': 'Нет сделок', 'total_trades': 0, 'final_capital': initial_capital}

        trades_df = pd.DataFrame([t.__dict__ for t in trades])
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] < 0]

        equity = np.cumsum([initial_capital] + [t.pnl for t in trades])
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100

        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        rr_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0

        # ✅ PnL%
        pnl_pct = ((equity[-1] / initial_capital - 1) * 100)

        return {
            'total_trades': len(trades_df),
            'win_rate': len(wins) / len(trades_df) if len(trades_df) > 0 else 0,
            'profit_factor': wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else float('inf'),
            'total_pnl': trades_df['pnl'].sum(),
            'total_pnl_pct': round(pnl_pct, 2),
            'final_capital': equity[-1],
            'max_dd_pct': abs(drawdown.min()) if len(drawdown) > 0 else 0,
            'avg_win': round(avg_win, 2),
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'rr_ratio': round(rr_ratio, 2),
            'best_trade': trades_df['pnl'].max() if len(trades_df) > 0 else 0,
            'worst_trade': trades_df['pnl'].min() if len(trades_df) > 0 else 0,
            'rules_count': rules_count,
        }

    def print_metrics(self, metrics: Dict, symbol: str, tf: str, mode: str, period: str="", rules_count: int = 0):
        """Красивый вывод (только если verbose=True)"""
        if 'error' in metrics:
            logger.error(f"❌ {symbol} {tf} | {mode} | {metrics['error']}")
            return

        total_trades = metrics.get('total_trades', 0)

        # 🔥 ФИКС: ПРОВЕРКА БАГА 0 правил + сделки
        actual_rules = metrics.get('rules_count', rules_count)
        if total_trades > 0 and actual_rules == 0:
            print(f"⚠️  {total_trades} сделок | ПРАВИЛ: {actual_rules} (кэш/анализ?)")

        if total_trades == 0 and self.verbose:
            print(f"\n📊 {symbol} {tf} | {mode}")
            print("-" * 60)
            print("❌ NO SIGNALS (0 trades)")
            return


        rules = metrics.get('rules_count', rules_count)

        period_str = f" | {period}" if period.strip() else ""

        print(f"\n📊 {symbol} {tf} | {mode}{period_str} | правил: {rules}")
        print("-" * 60)

        print(f"💰 Final Capital:  ${metrics['final_capital']:.2f} ({metrics['total_pnl_pct']:.1f}%)")
        print(f"📈 Profit Factor:  {metrics['profit_factor']:.2f} | RR: {metrics['rr_ratio']:.2f}")
        print(f"🎯 Win Rate:       {metrics['win_rate'] * 100:.1f}% ({metrics['total_trades']} сделок)")
        print(f"📉 Max DD:         {metrics['max_dd_pct']:.1f}%")
        print(f"⭐ Best:           ${metrics['best_trade']:.2f}")
        print(f"💥 Worst:          ${metrics['worst_trade']:.2f}")
        print(f"📊 Avg Win/Loss:   ${metrics['avg_win']:.2f} / ${metrics['avg_loss']:.2f}")

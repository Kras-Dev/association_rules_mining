"""📊 Расчёт торговых метрик"""

import pandas as pd
import numpy as np
from typing import Dict, List
from back_test.trade import Trade


class MetricsCalculator:
    """Калькулятор метрик"""

    @staticmethod
    def calculate(trades: List[Trade], initial_capital: float) -> Dict:
        """Основные метрики"""
        if not trades:
            return {'error': 'Нет сделок', 'total_trades': 0, 'final_capital': initial_capital}

        trades_df = pd.DataFrame([t.__dict__ for t in trades])
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] < 0]

        # 🔥 ИСПРАВЛЕНО: equity curve + final_capital
        equity = np.cumsum([initial_capital] + [t.pnl for t in trades])
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100

        return {
            'total_trades': len(trades_df),
            'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            'profit_factor': wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else float('inf'),
            'total_pnl': trades_df['pnl'].sum(),
            'final_capital': equity[-1],
            'max_dd_pct': abs(drawdown.min()) if len(drawdown) > 0 else 0,
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'best_trade': trades_df['pnl'].max() if len(trades_df) > 0 else 0,
            'worst_trade': trades_df['pnl'].min() if len(trades_df) > 0 else 0,
        }

    @staticmethod
    def print_metrics(metrics: Dict, symbol: str, tf: str, mode: str):
        """Красивый вывод"""
        if 'error' in metrics:
            print(f"❌ {symbol} {tf} | {mode} | {metrics['error']}")
            return
        total_trades = metrics.get('total_trades', 0)

        if total_trades == 0:
            print(f"\n📊 {symbol} {tf} | {mode}")
            print("-" * 60)
            print("❌ NO SIGNALS (0 trades)")

        print(f"\n📊 {symbol} {tf} | {mode}")
        print("-" * 60)
        print(f"💰 Final Capital:  ${metrics['final_capital']:.2f} (+{metrics['total_pnl']:.2f})")
        print(f"📈 Profit Factor:  {metrics['profit_factor']:.2f} (1.3-2.0 идеал) За каждый $1 убытка → {metrics['profit_factor']:.2f} прибыли!")
        print(f"🎯 Win Rate:       {metrics['win_rate']:.1f}% ({metrics['total_trades']} сделок)")
        print(f"📉 Max DD:         {metrics['max_dd_pct']:.1f}% (Max просадка капитала DD<15% = ПРИЕМЛЕМО)")
        print(f"⭐ Best:           ${metrics['best_trade']:.2f}")
        print(f"💥 Worst:          ${metrics['worst_trade']:.2f}")
        print(f"📊 Avg Win/Loss:   ${metrics['avg_win']:.2f} / ${metrics['avg_loss']:.2f}")
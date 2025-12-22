"""🚀 Backtest Runner с сохранением JSON!"""
import multiprocessing as mp
import shutil

from tqdm.contrib.concurrent import process_map
from association_miner.candle_miner import CandleMiner
from association_miner.features_engineer import Features
from mt5_connector.client import MT5Client
import MetaTrader5 as mt5
from back_test.backtester import Backtester
from back_test.config import TEST_SYMBOLS, TEST_TIMEFRAMES, get_candles
import json
import pandas as pd
from typing import List, Tuple, Dict, Any
from datetime import datetime
from pathlib import Path
from utils.logger import setup_logger


class BacktestRunner:
    """Полный запуск бэктестов + JSON сохранение"""

    def __init__(self, max_workers: int = None, verbose: bool = True):
        self.max_workers = max_workers or (mp.cpu_count() - 2 or 1)
        self.verbose = verbose
        self.results: List[Tuple[str, str, str, Dict[str, Any]]] = []
        self.exp_dir = self._create_history_dir()
        self.models_dir = self.exp_dir / "models"

    def _create_history_dir(self):
        timestamp = datetime.now().strftime("%d-%m-%Y")
        exp_dir = Path("history") / timestamp
        exp_dir.mkdir(parents=True, exist_ok=True)

        # ✅ Создаем папки ОДИН РАЗ
        (exp_dir / "models").mkdir(exist_ok=True)
        (exp_dir / "results").mkdir(exist_ok=True)
        log_dir = exp_dir / "logs"  # ✅ Готовый путь
        log_dir.mkdir(exist_ok=True)  # ✅ Создаем ОДИН РАЗ

        setup_logger("MAIN", log_dir)  # ✅ Правильно!

        # ✅ Cross-platform active (try/except для Linux)
        active_dir = Path("history/active")
        if active_dir.exists():
            shutil.rmtree(active_dir)
        active_dir.mkdir(exist_ok=True)

        # ✅ КОПИРУЕМ ТОЛЬКО models/ и results/ (БЕЗ logs/)
        for item in exp_dir.rglob("*"):
            if item.is_file() and "logs" not in str(item.relative_to(exp_dir)):
                rel_path = item.relative_to(exp_dir)
                dest = active_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)

        return exp_dir

    def backtest_single(self, args: Tuple[str, str, str]) -> Tuple[str, str, str, Dict[str, Any]]:
        """Один тест """
        symbol, tf, mode = args
        if self.verbose:
            print(f"[{mp.current_process().name}] {symbol} {tf}")

        try:
            with MT5Client() as client:
                tf_mt5 = getattr(mt5, f"TIMEFRAME_{tf}")
                candles_count = get_candles(tf)
                df_full = client.get_rates(symbol, tf_mt5, candles_count, 1)

                if len(df_full) < 1000:
                    return symbol, tf, mode, {'error': f'Мало данных ({len(df_full)})'}

                split_70 = int(len(df_full) * 0.7)
                train_df = df_full[:split_70]
                test_df = df_full[split_70:]

                miner = CandleMiner(min_confidence=0.7, min_support=10, verbose=False, history_dir=self.exp_dir)
                train_results = miner.smart_analyze(train_df, symbol, tf)

                feat_gen = Features(verbose=False)
                test_features = feat_gen.create_all_features(test_df)

                bt = Backtester(symbol, verbose=False, history_dir=self.exp_dir)
                metrics = bt.run_backtest(test_df, test_features, symbol, tf, mode, verbose=False)

                pnl = metrics.get('total_pnl', 0) if 'error' not in metrics else 0

                # ✅ ПЕРИОД + rules_count
                start_date = test_df.iloc[0]['time'].strftime('%d.%m.%y')
                end_date = test_df.iloc[-1]['time'].strftime('%d.%m.%y')
                rules_count = len(train_results['all_rules'])
                if self.verbose:
                    print(f"✅ [{mp.current_process().name}] {symbol} {tf} {start_date}-{end_date}: {pnl:.1f}%")

                metrics.update({
                    'period': f"{start_date}-{end_date}",
                    'rules_count': rules_count,
                    'test_date': datetime.now().strftime('%Y-%m-%d %H:%M')
                })

            return symbol, tf, mode, metrics

        except Exception as e:
            error_msg = str(e)[:80]
            if self.verbose:
                print(f"❌ [{mp.current_process().name}] {symbol} {tf}: {error_msg}")
            return symbol, tf, mode, {'error': error_msg}

    def run_parallel(self) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """🧪 Параллельный запуск всех тестов"""
        tasks = [(s, t, "SIGNAL_TO_SIGNAL") for s in TEST_SYMBOLS for t in TEST_TIMEFRAMES]

        print(f"🚀 {len(tasks)} тестов × {self.max_workers} ядер")
        print("=" * 80)

        results = process_map(
            self.backtest_single,
            tasks,
            max_workers=self.max_workers,
            chunksize=10,
            desc="🧪 Backtests"
        )

        self.results = results
        return results

    def save_results(self):
        """💾 СОХРАНЕНИЕ JSON + CSV"""
        results_db = []

        for symbol, tf, mode, metrics in self.results:

            result = {
                'symbol': symbol,
                'timeframe': tf,
                'mode': mode,
                'test_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                **metrics
            }
            results_db.append(result)

        # JSON
        json_path = self.exp_dir / "results" / "backtest_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_db, f, indent=2, ensure_ascii=False)

        # CSV
        csv_path = self.exp_dir / "results" / "backtest_results.csv"
        pd.DataFrame(results_db).to_csv(csv_path, index=False, encoding='utf-8')

        print(f"💾 Сохранено в: {self.exp_dir}/results/")

    def print_summary(self):
        """📊 КРАСИВАЯ ТАБЛИЦА + ТОП-15"""
        wins = profitable = 0
        total_pnl = 0
        top_results = []

        print("\n" + "=" * 80)
        print("🎯 РЕЗУЛЬТАТЫ:")
        print("=" * 80)

        for symbol, tf, mode, metrics in self.results:
            if 'error' not in metrics:
                pnl = metrics.get('total_pnl', 0)
                total_pnl += pnl
                wins += 1

                if pnl > 0:
                    profitable += 1
                    top_results.append((pnl, symbol, tf, metrics))

                    period = metrics.get('period', '')
                    print(
                        f"✅ {symbol:10s} {tf:3s}: +${pnl:6.1f} | WR:{metrics.get('win_rate', 0) * 100:4.1f}% | {period}")
                else:
                    print(f"➖ {symbol:10s} {tf:3s}:  ${pnl:6.1f}")
            else:
                print(f"❌ {symbol:10s} {tf:3s}: {metrics['error'][:40]}")

        # 🏆 ТОП-15
        top_results.sort(reverse=True)
        print("\n🏆 ТОП-15:")
        print("-" * 80)
        for i, (pnl, sym, tfm, metrics) in enumerate(top_results[:15], 1):
            wr = metrics.get('win_rate', 0) * 100
            rules = metrics.get('rules_count', 0)
            print(f"{i}. {sym:10s} {tfm:3s}: +${pnl:6.1f} | WR:{wr:4.1f}% | 📜{rules}")

        print("\n" + "=" * 80)
        print(f"🎯 ИТОГО: {wins}/{len(self.results)} тестов | {profitable} профитных")
        print(f"💰 СУММАРНЫЙ PnL: ${total_pnl:.1f}")
        print("=" * 80)

    def get_live_candidates(self, min_pnl_pct: float = 15.0, max_dd: float = 15.0,
                            min_trades: int = 100, min_rr: float = 1.2, min_pf: float = 1.1) -> List[Dict]:
        """🎯 ЛИДЕРЫ ДЛЯ LIVE ТРЕЙДИНГА (НОВЫЕ КРИТЕРИИ!)"""
        candidates = []

        for symbol, tf, mode, metrics in self.results:
            if 'error' in metrics:
                continue

            # ✅ PnL% (если нет в metrics - считаем)
            pnl_pct = metrics.get('total_pnl_pct', 0)
            if pnl_pct == 0:
                final_capital = metrics.get('final_capital', 10000)
                pnl_pct = ((final_capital / 10000 - 1) * 100)

            # ✅ RR Ratio (если нет в metrics - считаем)
            rr_ratio = metrics.get('rr_ratio', 0)
            if rr_ratio == 0:
                avg_win = metrics.get('avg_win', 0)
                avg_loss = metrics.get('avg_loss', 0)
                rr_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0

            # ✅ 5 КРИТЕРИЕВ ДЛЯ LIVE
            if (pnl_pct > min_pnl_pct and  # +15%+
                    metrics.get('max_dd_pct', 100) < max_dd and  # DD <15%
                    metrics.get('total_trades', 0) > min_trades and  # >100 сделок
                    rr_ratio > min_rr and  # RR >1.2
                    metrics.get('profit_factor', 0) > min_pf):  # PF >1.1

                candidates.append({
                    'symbol': symbol,
                    'timeframe': tf,
                    'pnl_pct': round(pnl_pct, 1),
                    'profit_factor': round(metrics.get('profit_factor', 0), 2),
                    'win_rate_pct': round(metrics.get('win_rate', 0) * 100, 1),
                    'trades': metrics.get('total_trades', 0),
                    'max_dd_pct': round(metrics.get('max_dd_pct', 0), 1),
                    'rr_ratio': round(rr_ratio, 2),
                    'avg_win': round(metrics.get('avg_win', 0), 2),
                    'avg_loss': round(metrics.get('avg_loss', 0), 2),
                    'rules_count': metrics.get('rules_count', 0),
                    'period': metrics.get('period', '')
                })

        # ✅ СОРТИРОВКА ПО ДОХОДНОСТИ
        candidates.sort(key=lambda x: x['pnl_pct'], reverse=True)

        # 🎯 КРАСИВЫЙ ВЫВОД
        print(f"\n🎯 LIVE КАНДИДАТЫ ({len(candidates)}):")
        print("-" * 90)
        print(f"{'#':<2} {'Символ':<10} {'TF':<4} {'PnL%':<6} {'PF':<5} {'RR':<5} {'DD%':<5} {'Сделок':<6} {'Правила'}")
        print("-" * 90)

        for i, c in enumerate(candidates[:15], 1):  # ТОП-15
            print(f"{i:<2} {c['symbol']:<10} {c['timeframe']:<4} "
                  f"{c['pnl_pct']:+6.1f}% {c['profit_factor']:<5.2f} "
                  f"{c['rr_ratio']:<5.2f} {c['max_dd_pct']:<5.1f}% "
                  f"{c['trades']:<6} {c['rules_count']}")

        print("-" * 90)

        # 📊 СТАТИСТИКА
        if candidates:
            top_pnl = max(c['pnl_pct'] for c in candidates)
            avg_pf = sum(c['profit_factor'] for c in candidates) / len(candidates)
            print(f"🏆 ЛИДЕР: +{top_pnl:.1f}% | Ср. PF: {avg_pf:.2f}")

        return candidates




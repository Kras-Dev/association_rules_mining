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

from utils.base_file_handler import BaseFileHandler


class BacktestRunner(BaseFileHandler):
    """
    Оркестратор системы тестирования стратегий.

    Класс управляет полным жизненным циклом бэктеста: от загрузки данных из MT5
    до параллельного запуска тестов, сохранения архива и синхронизации с live-ботом.

    Attributes:
        timestamp_dir (Path): Уникальная папка для текущей сессии бэктеста.
        max_workers (int): Количество используемых ядер процессора.
        results (List[Tuple]): Хранилище результатов всех завершенных тестов.
    """

    def __init__(self, max_workers: int = None, verbose: bool = False):
        """
        Инициализирует Runner, создает структуру папок и настраивает логи.

        Args:
            max_workers (int, optional): Лимит ядер. По умолчанию: (все ядра - 2).
            verbose (bool): Флаг детального логирования в консоль.
        """
        # Runner может создавать папку с датой и передавать её в супер-класс
        self.timestamp_dir = Path("history") / datetime.now().strftime("%d-%m-%Y")
        super().__init__(verbose, self.timestamp_dir)
        self.max_workers = max_workers or (mp.cpu_count() - 2 or 1)
        self.results: List[Tuple] = []
        self._prepare_logs()

    def _prepare_logs(self):
        """Создает служебную директорию для лог-файлов текущей сессии."""
        log_dir = self.exp_dir / "logs"
        log_dir.mkdir(exist_ok=True)

    def update_live_directory(self):
        """
        Синхронизирует результаты текущего прогона с рабочей папкой Live-бота.

        Метод полностью заменяет содержимое папки 'history/active' данными
        из текущей сессии (кроме логов), чтобы бот всегда использовал свежие правила.
        """
        active_dir = Path("history/active")
        try:
            # Очистка старой конфигурации лайва
            if active_dir.exists():
                shutil.rmtree(active_dir)
            # Клонирование текущей сессии в active (без логов)
            shutil.copytree(
                self.timestamp_dir,
                active_dir,
                ignore=shutil.ignore_patterns('logs*')
            )
            self._log_info(f"🚀 СИНХРОНИЗАЦИЯ: Папка {self.timestamp_dir} теперь ACTIVE для лайва")
        except Exception as e:
            self._log_error(f"❌ Ошибка обновления LIVE папки: {e}")

    def backtest_single(self, args: Tuple[str, str, str]) -> Tuple[str, str, str, Dict[str, Any]]:
        """
        Выполняет полный цикл теста для одного актива и таймфрейма.

        Процесс: загрузка данных -> генерация фич -> обучение Miner -> тест Backtester.

        Args:
            args (Tuple): Кортеж (symbol, timeframe, mode).

        Returns:
            Tuple: Данные об инструменте и словарь с метриками (или ошибкой).
        """
        symbol, tf, mode = args
        self._log_info(f"[{mp.current_process().name}] {symbol} {tf}")
        # Гарантируем, что miner и bt смотрят в self.exp_dir (папку сессии)
        shared_history_dir = self.exp_dir

        try:
            with MT5Client() as client:
                # --- ПОДГОТОВКА ДАННЫХ ---
                tf_mt5 = getattr(mt5, f"TIMEFRAME_{tf}")
                candles_count = get_candles(tf)
                df_full = client.get_rates(symbol, tf_mt5, candles_count, 1)

                if df_full is None or len(df_full) < 1000:
                    return symbol, tf, mode, {'error': 'Мало данных'}
                # ШАГ 1: Генерация фич (MA, индикаторы) на всей истории для корректности
                feat_gen = Features(verbose=False)
                df_with_all_features = feat_gen.create_all_features(df_full)
                # ШАГ 2: Разделение на обучение (70%) и тест (30%)
                split_70 = int(len(df_with_all_features) * 0.7)
                # Для Майнера(обучение) отдаем СЫРЫЕ цены (он сам вызовет генерацию фич для train куска)
                train_df = df_full.iloc[:split_70].copy()
                # Для Бэктестера отдаем ПРЕДРАССЧИТАННЫЕ фичи (для честных MA)
                test_df_prices = df_full.iloc[split_70:].copy()
                test_features = df_with_all_features.iloc[split_70:].copy()
                # --- ОБУЧЕНИЕ И ТЕСТ ---
                # ШАГ 3: Майнер анализирует паттерны на тренировочном куске
                miner = CandleMiner(min_confidence=0.7, min_support=10, verbose=False, history_dir=shared_history_dir)
                train_results = miner.smart_analyze(train_df, symbol, tf)
                # ШАГ 4: Бэктестер проверяет паттерны на тестовом (новом) куске
                bt = Backtester(symbol, verbose=False, history_dir=shared_history_dir)
                metrics = bt.run_backtest(test_df_prices, test_features, symbol, tf, mode)
                # Сбор финальных данных
                pnl = metrics.get('total_pnl', 0) if 'error' not in metrics else 0
                start_date = test_df_prices.iloc[0]['time'].strftime('%d.%m.%y')
                end_date = test_df_prices.iloc[-1]['time'].strftime('%d.%m.%y')

                msg = f"✅ [{mp.current_process().name}] {symbol} {tf} {start_date}-{end_date}: {pnl:.1f}%"
                self._log_info(msg)

                metrics.update({
                    'period': f"{start_date}-{end_date}",
                    'rules_count': len(train_results['all_rules']),
                    'test_date': datetime.now().strftime('%Y-%m-%d %H:%M')
                })

            return symbol, tf, mode, metrics

        except Exception as e:
            import traceback
            # Печатаем полный стек ошибки, чтобы видеть где именно падает
            self._log_error(f"❌ Ошибка в {symbol} {tf}: {traceback.format_exc()}")
            return symbol, tf, mode, {'error': str(e)}

    def run_parallel(self) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """
        Запускает пул процессов для выполнения всех запланированных тестов.

        Returns:
        List: Список кортежей с результатами по каждому инструменту.
        """
        tasks = [(s, t, "SIGNAL_TO_SIGNAL") for s in TEST_SYMBOLS for t in TEST_TIMEFRAMES]

        print(f"{self._get_context()}: {len(tasks)} тестов × {self.max_workers} ядер")
        print("=" * 80)
        # Параллельное выполнение с визуализацией прогресса
        results = process_map(
            self.backtest_single,
            tasks,
            max_workers=self.max_workers,
            chunksize=10,
            desc="Backtests",
            position=0,
        )

        self.results = results
        return results

    def save_results(self):
        """
        Сохраняет накопленные результаты в форматах JSON и CSV.

        После успешной записи инициирует обновление папки для Live-трейдинга.
        """
        if not self.results:
            self._log_warning("Нет результатов для сохранения")
            return
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

        # Сохранение в JSON (для парсинга системой)
        json_path = self.results_dir / "backtest_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_db, f, indent=2, ensure_ascii=False)
        # Сохранение в CSV (для анализа в Excel/Pandas)
        csv_path = self.results_dir / "backtest_results.csv"
        pd.DataFrame(results_db).to_csv(csv_path, index=False, encoding='utf-8')

        print(f"{self._get_context()}: 💾 Сохранено в: {self.exp_dir}/results/")
        # Финальный шаг: делаем этот прогон актуальным для бота
        self.update_live_directory()

    def print_summary(self):
        """Выводит в консоль сводную статистику и топ инструментов по доходности."""
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

        # Вывод лучших стратегий
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
                            min_trades: int = 49, min_rr: float = 1.2, min_pf: float = 1.1,
                            min_rf: float = 1.5) -> List[Dict]:
        """🎯 ЛИДЕРЫ ДЛЯ LIVE ТРЕЙДИНГА (НОВЫЕ КРИТЕРИИ!)"""
        candidates = []

        for symbol, tf, mode, metrics in self.results:
            if 'error' in metrics:
                continue

            # PnL% (если нет в metrics - считаем)
            pnl_pct = metrics.get('total_pnl_pct', 0)
            if pnl_pct == 0:
                final_capital = metrics.get('final_capital', 10000)
                pnl_pct = ((final_capital / 10000 - 1) * 100)

            # RR Ratio (если нет в metrics - считаем)
            rr_ratio = metrics.get('rr_ratio', 0)
            if rr_ratio == 0:
                avg_win = metrics.get('avg_win', 0)
                avg_loss = metrics.get('avg_loss', 0)
                rr_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0

            # 3. Достаем текущие DD и RF
            current_dd = metrics.get('max_dd_pct', 100)
            current_rf = metrics.get('recovery_factor', 0)

            # --- ЛОГИКА ДИНАМИЧЕСКОГО RF
            required_rf = min_rf  # По умолчанию 1.5

            if pnl_pct > 80 and current_dd < 30:
                required_rf = 3.0
            elif pnl_pct > 40 and current_dd < 20:
                required_rf = 2.0
            elif pnl_pct > 15 and current_dd < 15:
                required_rf = 1.5

            # ОБЪЕДИНЕННАЯ ПРОВЕРКА ВСЕХ 6 КРИТЕРИЕВ ДЛЯ LIVE ТРЕЙДИНГА
            if (pnl_pct > min_pnl_pct and  # (1) PnL > 15%
                    current_dd <= max_dd and  # (2) MaxDD <= 15%
                    metrics.get('total_trades', 0) > min_trades and  # (3) Trades > 49
                    rr_ratio > min_rr and  # (4) RR > 1.2
                    metrics.get('profit_factor', 0) > min_pf and  # (5) PF > 1.1
                    current_rf >= required_rf):  # (6) Динамический RF ✓

                candidates.append({
                    'symbol': symbol,
                    'timeframe': tf,
                    'pnl_pct': round(pnl_pct, 1),
                    'profit_factor': round(metrics.get('profit_factor', 0), 2),
                    'win_rate_pct': round(metrics.get('win_rate', 0) * 100, 1),
                    'trades': metrics.get('total_trades', 0),
                    'max_dd_pct': round(current_dd, 1),
                    'rr_ratio': round(rr_ratio, 2),
                    'avg_win': round(metrics.get('avg_win', 0), 2),
                    'avg_loss': round(metrics.get('avg_loss', 0), 2),
                    'recovery_factor': round(current_rf, 2),
                    'rules_count': metrics.get('rules_count', 0),
                    'period': metrics.get('period', '')
                })

        # СОРТИРОВКА ПО ДОХОДНОСТИ
        candidates.sort(key=lambda x: x['pnl_pct'], reverse=True)

        # КРАСИВЫЙ ВЫВОД
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

        # СТАТИСТИКА
        if candidates:
            top_pnl = max(c['pnl_pct'] for c in candidates)
            avg_pf = sum(c['profit_factor'] for c in candidates) / len(candidates)
            print(f"🏆 ЛИДЕР: +{top_pnl:.1f}% | Ср. PF: {avg_pf:.2f}")

        return candidates
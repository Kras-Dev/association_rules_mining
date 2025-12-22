import pandas as pd

from association_miner.candle_miner import CandleMiner
from association_miner.features_engineer import Features
from mt5_connector.client import MT5Client
import MetaTrader5 as mt5
from back_test.backtester import Backtester
from back_test.config import TEST_SYMBOLS, TEST_TIMEFRAMES, get_candles

SYMBOL = "#GMKN"
TIMEFRAME = mt5.TIMEFRAME_H4

def main():
    with MT5Client() as client:
        df = client.get_rates(SYMBOL, TIMEFRAME, 35000, 300)
        miner = CandleMiner(min_confidence=0.65, verbose=True, min_support=10)
        results = miner.smart_analyze(df, SYMBOL, client.mt5_var_name(TIMEFRAME))

        # 2. Разделение данных
        train_df = df[:24500]  # 70% IN-SAMPLE
        test_df = df[24500:35000]  # 30% OUT-OF-SAMPLE
        live_df = client.get_rates(SYMBOL, TIMEFRAME, 300, start_pos=1)  # LIVE

        # 3. Бэктест по фазам
        # results_train = backtest(train_df, rules)
        # results_test = backtest(test_df, rules)
        # results_live = backtest(live_df, rules)

        # miner.print_top_rules(results, top_n=90, symbol=SYMBOL,timeframe=client.mt5_var_name(TIMEFRAME))
        # with pd.option_context('display.max_rows', None,  # Печатать все строки
        #                        'display.max_columns', None,  # Печатать все колонки
        #                        'display.width', 1000,  # Не переносить строки на новую линию
        #                        'display.max_colwidth', None):
        #     print(results)


def b_test():
    with MT5Client() as client:
        for symbol in TEST_SYMBOLS:  # 6 символов
            for tf in TEST_TIMEFRAMES:  # 4 TF
                print(f"\n{'=' * 80}")
                print(f"🔥 {symbol} {tf}")
                print('=' * 80)

                # 1. Загружаем данные
                tf_mt5 = getattr(mt5, f"TIMEFRAME_{tf}")
                df_full = client.get_rates(symbol, tf_mt5, get_candles(tf), 1)
                if len(df_full) < 1000:
                    print(f"❌ Мало данных: {len(df_full)}")
                    continue

                # 2. Разделение
                total = len(df_full)
                train_end = int(total * 0.7)
                test_end = int(total * 0.9)

                train_df = df_full[300:train_end]  # TRAIN
                test_df = df_full[train_end:test_end]  # TEST
                live_df = df_full[test_end:]  # LIVE

                print(f"📊 train:{len(train_df)} | test:{len(test_df)} | live:{len(live_df)}")

                # 3. Майним правила на TRAIN
                miner = CandleMiner(min_confidence=0.7, min_support=10)
                train_results = miner.smart_analyze(train_df, symbol, tf)
                miner.print_top_rules(train_results, 10, symbol, tf)

                # 4. БЭКТЕСТ на TEST
                for mode in ["SIGNAL_TO_SIGNAL"]:  # Добавь другие позже
                    bt = Backtester(symbol)  # ✅ Новый Backtester!
                    metrics = bt.run_backtest(test_df, symbol, tf, mode)


if __name__ == "__main__":
    #main()
    b_test()
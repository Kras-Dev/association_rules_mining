import pandas as pd

from association_miner.candle_miner import CandleMiner
from association_miner.features_engineer import Features
from mt5_connector.client import MT5Client
import MetaTrader5 as mt5
from back_test.backtester import Backtester
from back_test.config import TEST_SYMBOLS

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
        for symbol, tf in TEST_SYMBOLS.items():
            print(f"\n{'=' * 80}")
            print(f"🔥 {symbol} {tf}")
            print('=' * 80)

            df_full = client.get_rates(symbol, getattr(mt5, f"TIMEFRAME_{tf}"), 35000, 1)
            print(f"📈 Всего: {len(df_full)} свечей")

            # ✅ ДИНАМИЧЕСКОЕ РАЗДЕЛЕНИЕ
            total = len(df_full)
            train_end = int(total * 0.7)
            test_end = int(total * 0.9)

            train_df = df_full[300:train_end]
            test_df = df_full[train_end:test_end]
            live_df = df_full[test_end:]

            print(f"📊 train: {len(train_df)} | test: {len(test_df)} | live: {len(live_df)}")

            # ФАЗА 1: НОВЫЕ правила на TRAIN
            miner = CandleMiner(min_confidence=0.65, min_support=10)
            train_results = miner.smart_analyze(train_df, symbol, tf)  # ✅ Ты прав!
            miner.print_top_rules(train_results, 10)

            # ФАЗА 2: DEBUG get_active_rules()
            feat_gen = Features(verbose=False)


            bt = Backtester()
            bt.rules = train_results['all_rules'][train_results['all_rules']['confidence'] > 0.70]



            # ФАЗА 3: Бэктест ТОЛЬКО если есть данные
            if len(test_df) > 1000:
                for mode in ["SIGNAL_TO_SIGNAL", "ONE_CANDLE", "ATR_TP"]:
                    bt.run_backtest(test_df, symbol, tf, mode)

if __name__ == "__main__":
    #main()
    b_test()
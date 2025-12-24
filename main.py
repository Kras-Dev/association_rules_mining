from back_test.back_test_runner import BacktestRunner
import multiprocessing as mp

import MetaTrader5 as mt5
from mt5_connector.client import MT5Client

if __name__ == "__main__":
    from association_miner.features_engineer import Features
    f = Features()
    with MT5Client() as client:
        # --- ПОДГОТОВКА ДАННЫХ ---
        tf_mt5 = getattr(mt5, f"TIMEFRAME_H1")
        df_full = client.get_rates("#SBER", tf_mt5, 1000, 1)
        f.test_features(df_full)


    mp.set_start_method('spawn', force=True)
    runner = BacktestRunner(max_workers=5)
    runner.run_parallel()
    runner.print_summary()
    runner.save_results()
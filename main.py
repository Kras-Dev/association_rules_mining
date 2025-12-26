

from back_test.back_test_runner import BacktestRunner
import multiprocessing as mp


if __name__ == "__main__":

    # from utils.base_file_handler import BaseFileHandler
    # from pathlib import Path
    # real_path = Path("history/26-12-2025")
    # bf = BaseFileHandler(history_dir=real_path)
    # f_path = bf._get_cache_path("#SBER", "D1")
    # f_cache = bf._load_cache(f_path)
    # print(f_cache)


    mp.set_start_method('spawn', force=True)
    runner = BacktestRunner(max_workers=4)
    runner.run_parallel()
    runner.print_summary()
    runner.save_results()
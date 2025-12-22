from back_test.back_test_runner import BacktestRunner
import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    runner = BacktestRunner(max_workers=5)
    runner.run_parallel()
    runner.print_summary()
    runner.save_results()
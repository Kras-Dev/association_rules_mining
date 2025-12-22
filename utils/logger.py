import logging
import sys
from pathlib import Path
import os


def setup_logger(process_id: str = "MAIN", log_dir=None):
    # ✅ ОБЯЗАТЕЛЬНЫЙ log_dir!
    if not log_dir:
        raise ValueError("log_dir required for all loggers!")

    log_file = f'{log_dir}/backtest_{process_id}.log'

    logger = logging.getLogger(process_id)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)-7s | %(name)-12s | [%(processName)s] | %(message)s'))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)-7s | %(name)-12s | [%(processName)s] | %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.getLogger('tqdm').setLevel(logging.WARNING)
    return logger

import pickle
from pathlib import Path
from typing import Optional, Dict

from utils.base_logger import BaseLogger


class BaseFileHandler(BaseLogger):
    """Базовый класс для работы с путями и кэшированием"""
    def __init__(self, verbose: bool = True, history_dir: Optional[Path] = None):
        super().__init__(verbose)

        # Если history_dir не передан, используем дефолт
        self.exp_dir = history_dir or Path("history/active")
        self.models_dir = self.exp_dir / "models"
        self.results_dir = self.exp_dir / "results"

        # Создаем структуру папок один раз здесь
        for folder in [self.models_dir, self.results_dir]:
            folder.mkdir(parents=True, exist_ok=True)


    def _get_cache_path(self, symbol: str, tf: str) -> Path:
        return self.models_dir / f"rules_{symbol}_{tf}.pkl"

    def _save_pickle(self, path: Path, data: Dict):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def _load_pickle(self, path: Path) -> Optional[Dict]:
        if path.exists():
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

from pathlib import Path
from association_miner.features_engineer import Features
from tqdm import tqdm
import pandas as pd
from typing import Dict, Optional, Tuple

from utils.base_file_handler import BaseFileHandler


class CandleMiner(BaseFileHandler):
    """
    CandleMiner: Класс для поиска статистически значимых свечных паттернов.
    Находит правила (features), которые предсказывают движение цены с уверенностью (confidence) 60%+.
    """

    def __init__(self, min_confidence: float = 0.60, min_support: int = 10, verbose: bool = False,
                 history_dir: Path = None):
        """
        Инициализация майнера правил.

        Args:
            min_confidence (float): Минимальная точность правила (от 0 до 1).
            min_support (int): Минимальное количество вхождений паттерна в истории.
            verbose (bool): Флаг детального логирования.
            history_dir (Path): Путь к директории с историческими данными/моделями.
        """
        super().__init__(verbose, history_dir)
        self.min_confidence = min_confidence
        self.min_support = min_support

    def save_rules(self, results: Dict, symbol: str, tf: str, min_confidence: float = 0.70) -> str:
        """
        Фильтрация и сохранение найденных правил в кэш (pickle).

        Args:
            results (Dict): Результаты анализа из find_strong_rules.
            symbol (str): Торговый инструмент.
            tf (str): Таймфрейм.
            min_confidence (float): Порог точности для сохранения в "топ".

        Returns:
            str: Путь к сохраненному файлу кэша.
        """
        cache_file = self._get_cache_path(symbol, tf)
        # Фильтруем правила: оставляем только самые надежные для продакшена
        high_conf_rules = results['all_rules'][results['all_rules']['confidence'] >= min_confidence]
        rules_count = len(high_conf_rules)

        cache = {
            'top_rules': high_conf_rules,
            'base_prob_up': results['base_prob_up'],
            'base_prob_down': results['base_prob_down'],
            'symbol': symbol, 'tf': tf,
            'timestamp': pd.Timestamp.now(),
            'total_features': len(results['all_features'].columns)
        }
        self._save_pickle(cache_file, cache)
        self._log_info(f"[CandleMiner]: 💾 Сохранено: {cache_file} ({rules_count}/{len(results['all_rules'])} "
                       f">{min_confidence:.0%} conf правил)")
        return str(cache_file)

    def load_rules(self, symbol: str, tf: str) -> Optional[Dict]:
        """
        Загрузка предобученных правил из кэша.
        """
        cache_path = self._get_cache_path(symbol, tf)
        data = self._load_pickle(cache_path)
        if data:
            self._log_info(f"📂 Загружено: {cache_path} ({len(data['top_rules'])} правил)")
            return data
        self._log_warning(f"❌ Кэш не найден для {symbol} {tf} по пути: {cache_path}")
        return None

    def find_strong_rules(self, features: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Основной алгоритм поиска ассоциативных правил.
        Вычисляет Confidence (точность) и Lift (превышение над базовой вероятностью).
        """
        buy_conditions, sell_conditions = [], []
        # Отбираем только бинарные признаки (0 или 1), исключая целевые переменные
        binary_features = [col for col in features.select_dtypes(include=['int64']).columns
                           if col not in ['next_up', 'next_down']]

        mean_up = features['next_up'].mean()
        mean_down = features['next_down'].mean()

        self._log_info(f"Тестируем {len(binary_features)} признаков...")

        for feature in tqdm(binary_features, desc="Rules", unit="feature", disable=not self.verbose):
            # Считаем, сколько раз встретился данный признак
            feature_series = features[feature]
            total = feature_series.sum()
            # Проверка на минимальную поддержку (Support)
            if total < self.min_support:
                continue

            # --- Анализ для LONG (UP) ---
            buy_hits = features[(features[feature] == 1) & (features['next_up'] == 1)].shape[0]
            buy_conf = buy_hits / total
            buy_lift = buy_conf / features['next_up'].mean() if features['next_up'].mean() > 0 else 1.0

            buy_conditions.append({'rule_name': feature, 'confidence': buy_conf,
                                       'support': total, 'lift': buy_lift,
                                       'direction': 'UP'})
            # --- Анализ для SHORT (DOWN) ---
            sell_hits = features[(features[feature] == 1) & (features['next_down'] == 1)].shape[0]
            sell_conf = sell_hits / total
            sell_lift = sell_conf / features['next_down'].mean() if features['next_down'].mean() > 0 else 1.0

            sell_conditions.append({'rule_name': feature, 'confidence': sell_conf,
                                        'support': total, 'lift': sell_lift,
                                        'direction': 'DOWN'})
                # Сборка результатов в DataFrame
        buy_rules = pd.DataFrame(buy_conditions)
        sell_rules = pd.DataFrame(sell_conditions)

        # Сортировка по силе (Lift)
        if not buy_rules.empty:
            buy_rules = buy_rules.sort_values('lift', ascending=False).reset_index(drop=True)
        if not sell_rules.empty:
            sell_rules = sell_rules.sort_values('lift', ascending=False).reset_index(drop=True)

        all_rules = pd.concat([buy_rules, sell_rules], ignore_index=True)
        if not all_rules.empty:
            all_rules = all_rules.sort_values('lift', ascending=False).reset_index(drop=True)

        self._log_info(f"Результат: {len(buy_rules)} BUY, {len(sell_rules)} SELL правил.")
        return buy_rules, sell_rules, all_rules

    def smart_analyze(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """
        Умный анализ: возвращает закэшированные правила или запускает новый поиск.
        """
        # 1. Пробуем загрузить существующую "модель"
        cached = self.load_rules(symbol, timeframe)
        if cached:
            self._log_info(f"КЭШ АКТУАЛЕН ({len(df)} свечей)")
            return {
                'all_rules': cached['top_rules'],
                'base_prob_up': cached['base_prob_up'],
                'base_prob_down': cached['base_prob_down'],
                'symbol': symbol, 'tf': timeframe, 'from_cache': True
            }
        # 2. Если кэша нет - запускаем Feature Engineering и Mining
        self._log_info(f"ПОЛНЫЙ АНАЛИЗ {symbol} {timeframe}")

        feat_gen = Features(verbose=self.verbose)
        all_features = feat_gen.create_all_features(df)
        # 3. Поиск закономерностей
        buy_rules, sell_rules, all_rules = self.find_strong_rules(all_features)
        # Фильтруем для "сильных" правил (тех, что пойдут в результаты)
        strong_rules = all_rules[all_rules['confidence'] >= self.min_confidence] \
                    if not all_rules.empty else pd.DataFrame()

        base_prob_up = all_features['next_up'].mean()
        base_prob_down = all_features['next_down'].mean()
        if not strong_rules.empty:
            # 4. Формирование финального словаря результатов
            results = {
                'all_features': all_features, 'buy_rules': buy_rules, 'sell_rules': sell_rules,
                'all_rules': strong_rules, 'base_prob_up': base_prob_up, 'base_prob_down': base_prob_down,
                'symbol': symbol, 'tf_name': timeframe
            }
            # 5. Автоматическое сохранение результатов
            self.save_rules(results, symbol, timeframe, min_confidence=0.70)
        else:
            # Находим лучшие показатели среди всех попыток, даже если они слабые
            max_conf = all_rules['confidence'].max() if not all_rules.empty else 0
            max_lift = all_rules['lift'].max() if not all_rules.empty else 0
            msg = f"⚠️ Для {symbol} {timeframe} сильных правил не найдено."
            if max_conf > 0:
                msg += f" (Лучший Conf: {max_conf:.2%}, Lift: {max_lift:.2f})"

            self._log_warning(f"{msg}. Кэш не создан.")
        return {'all_rules': pd.DataFrame(), 'error': 'No strong rules'}

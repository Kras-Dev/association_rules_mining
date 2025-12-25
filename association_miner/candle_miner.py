from pathlib import Path
from association_miner.features_engineer import Features
from tqdm import tqdm
import pandas as pd
from typing import Dict, Optional, Tuple, Any

from back_test.config import SL_MULTIPLIER, ARM_CONFIG
from utils.base_file_handler import BaseFileHandler


class CandleMiner(BaseFileHandler):
    """
    CandleMiner: Класс для поиска статистически значимых свечных паттернов.
    Находит правила (features), которые предсказывают движение цены с уверенностью (confidence) 60%+.
    """

    def __init__(self, min_confidence: float = 0.68, min_support: int = 22, verbose: bool = False,
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

    def save_rules(self, results: Dict, symbol: str, tf: str, min_confidence: float = None) -> str:
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
        if min_confidence is None:
            min_confidence = self.min_confidence

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
            'total_features': results.get('total_features', 0),
            'min_confidence': results.get('min_confidence', -1),
        }
        self._save_cache(cache_file, cache)
        self._log_info(f"[CandleMiner]: 💾 Сохранено: {cache_file} ({rules_count}/{len(results['all_rules'])} "
                       f">{min_confidence:.0%} conf правил)")
        return str(cache_file)

    def load_rules(self, symbol: str, tf: str) -> Optional[Dict]:
        """
        Загрузка предобученных правил из кэша.
        """
        cache_path = self._get_cache_path(symbol, tf)
        data = self._load_cache(cache_path)
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
        binary_features = [col for col in features.columns
                           if features[col].nunique() <= 2 and
                           col not in ['next_up', 'next_down']]

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
        Умный анализ: возвращает закэшированные правила или запускает новый поиск
        с динамической фильтрацией признаков для оптимизации скорости.
        """
        # Поиск настроек (берем значение или дефолт)
        config = ARM_CONFIG.get(timeframe, {
            'min_support': self.min_support,
            'min_confidence': self.min_confidence
        })
        current_supp = config['min_support']
        current_conf = config['min_confidence']

        # 1. Пробуем загрузить существующую "модель" из кэша
        cached = self._load_cache(self._get_cache_path(symbol, timeframe))
        if cached:
            self._log_info(f"✅ КЭШ АКТУАЛЕН: {symbol} {timeframe} ({len(df)} свечей)")
            return {
                'all_rules': cached['top_rules'],
                'base_prob_up': cached['base_prob_up'],
                'base_prob_down': cached['base_prob_down'],
                'symbol': symbol,
                'tf': timeframe,
                'from_cache': True,
                'min_confidence': cached['min_confidence'],

            }

        # 2. Если кэша нет - запускаем Feature Engineering
        self._log_info(f"🔍 ПОЛНЫЙ АНАЛИЗ {symbol} {timeframe} (Старт майнинга)")

        feat_gen = Features(verbose=self.verbose)
        all_features = feat_gen.create_all_features(df)

        if all_features.empty:
            self._log_error("No features generated")
            return {'all_rules': pd.DataFrame(), 'error': 'No features generated'}

        # --- ДИНАМИЧЕСКАЯ ФИЛЬТРАЦИЯ ПРИЗНАКОВ (Оптимизация 2025) ---
        # Считаем, какой % от всей истории составляет твой min_support (21)
        # На D1 (3000 св.) это ~0.7%, на H4 (6000 св.) это ~0.35%
        total_rows = len(all_features)
        dynamic_support_pct = current_supp / total_rows

        # Ограничиваем снизу (не меньше 0.1%), чтобы не перегружать Apriori на M15/H1
        effective_support_pct = max(0.001, dynamic_support_pct)
        min_support_count = total_rows * effective_support_pct

        initial_feat_count = all_features.shape[1]

        # Отбираем колонки, где количество "единиц" (сигналов) >= порога
        # Исключаем целевые колонки из фильтрации, чтобы они не пропали
        targets = ['next_up', 'next_down']
        cols_to_check = [c for c in all_features.columns if c not in targets]

        # Быстрый подсчет сумм по колонкам
        feat_sums = all_features[cols_to_check].sum()
        keep_cols = feat_sums[feat_sums >= min_support_count].index.tolist()

        # Формируем оптимизированный набор данных для поиска правил
        all_features_filtered = all_features[keep_cols + targets]

        self._log_info(f"Оптимизация: Фичи {initial_feat_count} -> {len(keep_cols) + 2} "
                       f"(Порог: {effective_support_pct:.2%} или {int(min_support_count)} баров)")

        # 3. Поиск закономерностей (на облегченных данных)
        buy_rules, sell_rules, all_rules = self.find_strong_rules(all_features_filtered)

        # 4. Фильтруем для "сильных" правил согласно установленному min_confidence (0.63)
        if not all_rules.empty:
            strong_rules = all_rules[all_rules['confidence'] >= current_conf]
        else:
            strong_rules = pd.DataFrame()

        base_prob_up = all_features['next_up'].mean()
        base_prob_down = all_features['next_down'].mean()

        if not strong_rules.empty:
            # 5. Формирование финального словаря результатов
            results = {
                'all_rules': strong_rules,
                'base_prob_up': base_prob_up,
                'base_prob_down': base_prob_down,
                'symbol': symbol,
                'tf': timeframe,
                'total_features': len(keep_cols),
                'min_confidence': current_conf,
            }
            # 6. Сохранение (используем порог из конфига класса)
            self.save_rules(results, symbol, timeframe, min_confidence=current_conf)
            self._log_info(f"💾 Найдено и сохранено {len(strong_rules)} правил.")

            return {**results, 'from_cache': False}
        else:
            # Логика обработки отсутствия правил
            max_conf = all_rules['confidence'].max() if not all_rules.empty else 0
            self._log_warning(f"⚠️ Сильных правил (>{self.min_confidence}) не найдено. "
                              f"Лучший результат: {max_conf:.2%}")
            return {'all_rules': pd.DataFrame(), 'error': 'No strong rules', 'from_cache': False}

    def get_dynamic_params(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        ДИНАМИЧЕСКАЯ ЛОГИКА: conf + supp + SL_MULTIPLIER по TF/инструменту

        """
        current_conf = 0.68  # Базовый
        current_supp = 22  # Базовый
        sl_mult_key = symbol[:1]  # '#' или 'r'

        # 🔥 TF-ЛОГИКА (по приоритету качества сигналов)
        if 'M15' in timeframe:
            current_supp = 35
            current_conf = 0.65  # ✅ ROSN/SBER +35/+23%
        elif 'M30' in timeframe:
            current_supp = 35
            current_conf = 0.67  # ✅ USDCAD/MOEX +10/+13%
        elif 'H1' in timeframe:
            current_supp = 25 if symbol.startswith('#') else 28
            current_conf = 0.70 if symbol.startswith('#') else 0.68  # MOEX H1 +19%
        elif 'H4' in timeframe:
            if symbol.startswith('#'):  # Акции
                current_conf = 0.70
                current_supp = 25
            else:  # Форекс
                if symbol in ['USDCADrfd', 'EURUSDrfd']:
                    current_conf = 0.70
                    current_supp = 22  # EUR H4 Calmar 2.15
                else:  # GBPUSD, USDJPY
                    current_conf = 0.68
                    current_supp = 28
        elif 'D1' in timeframe:
            current_supp = 20
            current_conf = 0.72  # 🏆 MOEX D1 Calmar 5.32!

        # 🔥 SL_MULTIPLIER по TF (жестче = меньше шума)
        if 'M15' in timeframe or 'M30' in timeframe:
            SL_MULTIPLIER[sl_mult_key] = 2.2  # Шум → жесткий SL
        elif 'D1' in timeframe:
            SL_MULTIPLIER[sl_mult_key] = 1.8  # Чистый сигнал → мягкий SL
        elif 'H4' in timeframe:
            SL_MULTIPLIER[sl_mult_key] = 1.9 if symbol.startswith('#') else 2.0
        else:  # H1
            SL_MULTIPLIER[sl_mult_key] = 2.0

        return {
            'min_confidence': current_conf,
            'min_support': current_supp,
            'sl_multiplier_key': sl_mult_key,
            'sl_multiplier': SL_MULTIPLIER[sl_mult_key]
        }

import logging
import os
import pickle
import pandas as pd
from typing import Dict, Optional, Tuple
from association_miner.features_engineer import Features  # импорт внутри метода

logger = logging.getLogger(__name__)


class CandleMiner:
    """
    🔥 КИЛЛЕР: Находит свечные паттерны с confidence 60%+ и lift >1.0
    Использует Features для генерации признаков
    """

    def __init__(self, min_confidence: float = 0.60, min_support: int = 20, verbose: bool = False):
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.verbose = verbose

    def _log_features(self, features: pd.DataFrame, stage: str = "features") -> None:
        """Логгер количества бинарных признаков"""
        if not self.verbose:
            return

        binary_cols = features.select_dtypes(include=['int64']).columns.tolist()
        logger.info(f"[CandleMiner]: ✅ {len(binary_cols)} бинарных {stage}!")

    def save_rules(self, results: Dict, symbol: str, tf: str):
        """💾 Сохраняет ТОП-100 правил"""
        cache_file = f"rules_{symbol}_{tf}.pkl"
        top_rules = results['all_rules'].head(100)

        cache = {
            'top_rules': top_rules,
            'base_prob_up': results['base_prob_up'],
            'base_prob_down': results['base_prob_down'],
            'symbol': symbol,
            'tf': tf,
            'timestamp': pd.Timestamp.now()
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        print(f"[CandleMiner]: 💾 Правила сохранены: {cache_file} ({len(top_rules)} правил)")

    def load_rules(self, symbol: str, tf: str) -> Optional[Dict]:
        """📂 Загружает готовые правила"""
        cache_file = f"rules_{symbol}_{tf}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            print(f"[CandleMiner]: 📂 Загружено из кэша: {cache_file}")
            return cache
        return None

    def _log_rules(self, buy_rules: pd.DataFrame, sell_rules: pd.DataFrame) -> None:
        """📊 Логгер найденных правил"""
        if not self.verbose:
            return

        logger.info(f"[CandleMiner]: НАЙДЕНО: {len(buy_rules)} BUY, {len(sell_rules)} SELL правил")
        logger.info(
            f"[CandleMiner]: ТОП BUY: {buy_rules.head(1)['confidence'].iloc[0]:.1%} ({buy_rules.head(1)['feature'].iloc[0]})")
        logger.info(
            f"[CandleMiner]: ТОП SELL: {sell_rules.head(1)['confidence'].iloc[0]:.1%} ({sell_rules.head(1)['feature'].iloc[0]})")


    def find_strong_rules(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Находит сильные правила (confidence > min_confidence, support > min_support).

        Args:
            features: pd.DataFrame с бинарными фичами + next_up/next_down

        Returns:
            (buy_rules, sell_rules, all_rules) — отсортированные по lift
        """
        buy_conditions, sell_conditions = [], []

        # Все бинарные фичи (исключая target)
        binary_features = [col for col in features.select_dtypes(include=['int64']).columns
                           if col not in ['next_up', 'next_down']]

        if self.verbose:
            logger.info(f"[CandleMiner]: Тестируем {len(binary_features)} признаков...")

        for feature in binary_features:
            total = features[features[feature] == 1].shape[0]
            if total < self.min_support:
                continue

            # BUY: feature → next_up
            buy_hits = features[(features[feature] == 1) & (features['next_up'] == 1)].shape[0]
            buy_conf = buy_hits / total
            buy_lift = buy_conf / features['next_up'].mean() if features['next_up'].mean() > 0 else 1.0

            # SELL: feature → next_down
            sell_hits = features[(features[feature] == 1) & (features['next_down'] == 1)].shape[0]
            sell_conf = sell_hits / total
            sell_lift = sell_conf / features['next_down'].mean() if features['next_down'].mean() > 0 else 1.0

            if buy_conf > self.min_confidence:
                buy_conditions.append({
                    'feature': feature, 'confidence': buy_conf, 'support': total,
                    'lift': buy_lift, 'direction': 'UP'
                })

            if sell_conf > self.min_confidence:
                sell_conditions.append({
                    'feature': feature, 'confidence': sell_conf, 'support': total,
                    'lift': sell_lift, 'direction': 'DOWN'
                })

        buy_rules = pd.DataFrame(buy_conditions).sort_values('lift', ascending=False).reset_index(drop=True)
        sell_rules = pd.DataFrame(sell_conditions).sort_values('lift', ascending=False).reset_index(drop=True)
        all_rules = pd.concat([buy_rules, sell_rules]).sort_values('lift', ascending=False)

        self._log_rules(buy_rules, sell_rules)
        return buy_rules, sell_rules, all_rules

    def print_top_rules(self, results: Dict, top_n: int = 20, symbol: str=None, timeframe: str=None) -> None:
        """
        Красиво выводит ТОП-N лучших правил.

        Args:
            results: результат analyze()
            top_n: количество топ-правил для вывода
            symbol: символ инструмента
            timeframe: таймфрейм
        """
        all_rules = results['all_rules']
        if all_rules.empty:
            print("[CandleMiner]: ❌ Нет сильных правил (confidence > {:.0%})".format(self.min_confidence))
            return

        print("\n" + "=" * 80)
        print(f"[CandleMiner]: ТОП-{top_n} СВЕЧНЫХ ПАТТЕРНОВ (conf > {self.min_confidence:.0%}) {symbol} {timeframe}")
        print("=" * 80)

        top = all_rules.head(top_n)
        for i, (_, rule) in enumerate(top.iterrows(), 1):
            emoji = "🟢" if rule['direction'] == 'UP' else "🔴"
            print(
                f"{i:2d}. {emoji} {rule['feature']:<40}",
                f"{rule['confidence']:.1%} когда этот паттерн сработал,",
                f"сила сигнала(lift)={rule['lift']:.2f} ({int(rule['support'])} случая)"
            )

        print("=" * 80)

    def analyze(self, df: pd.DataFrame, symbol: Optional[str] = None, tf_name: Optional[str] = None) -> Dict:
        """
        Полный анализ: Features → Rules → Результаты.

        Args:
            df: DataFrame с OHLCV
            symbol: название инструмента (для логов)
            tf_name: таймфрейм (для логов)

        Returns:
            Dict с результатами анализа
        """
        print(f"[CandleMiner]: Анализ {symbol} {tf_name} ({len(df)} свечей)...")
        if symbol and tf_name:
            logger.info(f"[CandleMiner]: Анализ {symbol} {tf_name}...")

        # Генерация всех фич
        feat_gen = Features(verbose=self.verbose)
        all_features = feat_gen.create_all_features(df)

        self._log_features(all_features, "ИТОГОВЫХ")

        # Поиск правил
        print("[CandleMiner]: Поиск сильных правил...")
        buy_rules, sell_rules, all_rules = self.find_strong_rules(all_features)

        # Базовые вероятности
        base_prob_up = all_features['next_up'].mean()
        base_prob_down = all_features['next_down'].mean()

        return {
            'all_features': all_features,
            'buy_rules': buy_rules,
            'sell_rules': sell_rules,
            'all_rules': all_rules,
            'base_prob_up': base_prob_up,
            'base_prob_down': base_prob_down,
            'symbol': symbol,
            'tf_name': tf_name
        }


# ПРИМЕР ИСПОЛЬЗОВАНИЯ
"""
miner = CandleMiner(min_confidence=0.60, verbose=True)
results = miner.analyze(df, "EURUSD", "M5")
miner.print_top_rules(results, top_n=20)
"""

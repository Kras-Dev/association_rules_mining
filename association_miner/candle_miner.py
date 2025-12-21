import logging
import os
import pickle
from tqdm import tqdm
import pandas as pd
from typing import Dict, Optional, Tuple
from association_miner.features_engineer import Features

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

    def save_rules(self, results: Dict, symbol: str, tf: str, top_rules: int=100) -> str:
        """💾 Сохраняет ТОП-100 правил в rules_models/"""
        os.makedirs("models", exist_ok=True)
        cache_file = f"models/rules_{symbol}_{tf}.pkl"
        top_rules = results['all_rules'].head(top_rules)

        cache = {
            'top_rules': top_rules,
            'base_prob_up': results['base_prob_up'],
            'base_prob_down': results['base_prob_down'],
            'symbol': symbol,
            'tf': tf,
            'timestamp': pd.Timestamp.now(),
            'total_features': len(results['all_features'].columns)
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        print(f"[CandleMiner]: 💾 Сохранено: {cache_file} ({len(top_rules)} правил)")
        return cache_file

    def load_rules(self, symbol: str, tf: str) -> Optional[Dict]:
        """📂 Загружает правила из rules_models/"""
        cache_file = f"models/rules_{symbol}_{tf}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            print(f"[CandleMiner]: 📂 Загружено: {cache_file} ({len(cache['top_rules'])} правил)")
            return cache
        print(f"[CandleMiner]: ❌ Кэш не найден: {cache_file}")
        return None

    def _log_rules(self, buy_rules: pd.DataFrame, sell_rules: pd.DataFrame) -> None:
        """📊 Логгер найденных правил"""
        if not self.verbose:
            return

        logger.info(f"[CandleMiner]: НАЙДЕНО: {len(buy_rules)} BUY, {len(sell_rules)} SELL правил")
        logger.info(
            f"[CandleMiner]: ТОП BUY: {buy_rules.head(1)['confidence'].iloc[0]:.1%} ({buy_rules.head(1)['rule_name'].iloc[0]})")
        logger.info(
            f"[CandleMiner]: ТОП SELL: {sell_rules.head(1)['confidence'].iloc[0]:.1%} ({sell_rules.head(1)['rule_name'].iloc[0]})")


    def find_strong_rules(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Находит сильные правила (confidence > min_confidence, support > min_support).

        Args:
            features: pd.DataFrame с бинарными фичами + next_up/next_down

        Returns:
            (buy_rules, sell_rules, all_rules) — отсортированные по lift
        """
        buy_conditions, sell_conditions = [], []

        # TQDМ по ВСЕМ ФИЧАМ
        binary_features = [col for col in features.select_dtypes(include=['int64']).columns
                           if col not in ['next_up', 'next_down']]

        if self.verbose:
            logger.info(f"[CandleMiner]: Тестируем {len(binary_features)} признаков...")

        # 🔥 ПРОГРЕСС-БАР ПО ФИЧАМ
        print(f"🔍 Анализ {len(binary_features)} фич...")
        for feature in tqdm(binary_features, desc="Rules", unit="feature"):
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
                    'rule_name': feature, 'confidence': buy_conf, 'support': total,
                    'lift': buy_lift, 'direction': 'UP'
                })

            if sell_conf > self.min_confidence:
                sell_conditions.append({
                    'rule_name': feature, 'confidence': sell_conf, 'support': total,
                    'lift': sell_lift, 'direction': 'DOWN'
                })

        buy_rules = pd.DataFrame(buy_conditions).sort_values('lift', ascending=False).reset_index(drop=True)
        sell_rules = pd.DataFrame(sell_conditions).sort_values('lift', ascending=False).reset_index(drop=True)
        all_rules = pd.concat([buy_rules, sell_rules], ignore_index=True) \
            .sort_values('lift', ascending=False) \
            .reset_index(drop=True)

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
                f"{i:2d}. {emoji} {rule['rule_name']:<40}",
                f"{rule['confidence']:.1%} когда этот паттерн сработал,",
                f"сила сигнала(lift)={rule['lift']:.2f} ({int(rule['support'])} случая)"
            )

        print("=" * 80)

    def analyze(self, df: pd.DataFrame, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> Dict:
        """
        Полный анализ: Features → Rules → Результаты.

        Args:
            df: DataFrame с OHLCV
            symbol: название инструмента (для логов)
            timeframe: таймфрейм (для логов)

        Returns:
            Dict с результатами анализа
        """
        print(f"[CandleMiner]: Анализ {symbol} {timeframe} ({len(df)} свечей)...")
        if symbol and timeframe:
            logger.info(f"[CandleMiner]: Анализ {symbol} {timeframe}...")

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
            'tf_name': timeframe
        }

    def smart_analyze(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """УМНЫЙ анализ: кэш ИЛИ полный пересчёт"""
        cached = self.load_rules(symbol, timeframe)
        if cached:
            print(f"[CandleMiner]: КЭШ АКТУАЛЕН ({len(df)} свечей)")
            return {
                'all_rules': cached['top_rules'],
                'base_prob_up': cached['base_prob_up'],
                'base_prob_down': cached['base_prob_down'],
                'symbol': symbol,
                'tf': timeframe,
                'from_cache': True
            }

        print(f"[CandleMiner]: 🔥 ПОЛНЫЙ АНАЛИЗ ({len(df)} свечей)")
        results = self.analyze(df, symbol, timeframe)
        self.save_rules(results, symbol, timeframe)
        return results


# ПРИМЕР ИСПОЛЬЗОВАНИЯ
"""
miner = CandleMiner(min_confidence=0.60, verbose=True)
results = miner.analyze(df, "EURUSD", "M5")
miner.print_top_rules(results, top_n=20)
"""

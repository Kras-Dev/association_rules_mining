import logging
import os
import pickle
from pathlib import Path
from association_miner.features_engineer import Features
from tqdm import tqdm
import pandas as pd
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class CandleMiner:
    """🔥 Находит свечные паттерны confidence 60%+"""

    def __init__(self, min_confidence: float = 0.60, min_support: int = 10, verbose: bool = True,
                 history_dir: Path = None):
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.verbose = verbose
        self.exp_dir = history_dir or Path("history/active")
        self.models_dir = self.exp_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _log_info(self, message: str):
        if self.verbose:
            logger.info(message)

    def save_rules(self, results: Dict, symbol: str, tf: str, min_confidence: float = 0.70) -> str:
        """💾 Сохранение в history/active/models/"""
        cache_file = self.models_dir / f"rules_{symbol}_{tf}.pkl"
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
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        self._log_info(f"[CandleMiner]: 💾 Сохранено: {cache_file} ({rules_count}/{len(results['all_rules'])} "
                       f">{min_confidence:.0%} conf правил)")
        return str(cache_file)

    def load_rules(self, symbol: str, tf: str) -> Optional[Dict]:
        """📂 Загрузка из history/active/models/"""
        cache_file = self.models_dir / f"rules_{symbol}_{tf}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            self._log_info(f"[CandleMiner]: 📂 Загружено: {cache_file} ({len(cache['top_rules'])} правил)")
            return cache
        logger.warning(f"[CandleMiner]: ❌ Кэш не найден: {cache_file}")
        return None

    def find_strong_rules(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        buy_conditions, sell_conditions = [], []

        binary_features = [col for col in features.select_dtypes(include=['int64']).columns
                           if col not in ['next_up', 'next_down']]

        self._log_info(f"[CandleMiner]: Тестируем {len(binary_features)} признаков...")

        for feature in tqdm(binary_features, desc="Rules", unit="feature", disable=not self.verbose):
            total = features[features[feature] == 1].shape[0]
            if total < self.min_support:
                continue

            # BUY
            buy_hits = features[(features[feature] == 1) & (features['next_up'] == 1)].shape[0]
            buy_conf = buy_hits / total
            buy_lift = buy_conf / features['next_up'].mean() if features['next_up'].mean() > 0 else 1.0

            # SELL
            sell_hits = features[(features[feature] == 1) & (features['next_down'] == 1)].shape[0]
            sell_conf = sell_hits / total
            sell_lift = sell_conf / features['next_down'].mean() if features['next_down'].mean() > 0 else 1.0

            if buy_conf > self.min_confidence:
                buy_conditions.append({'rule_name': feature, 'confidence': buy_conf, 'support': total, 'lift': buy_lift,
                                       'direction': 'UP'})
            if sell_conf > self.min_confidence:
                sell_conditions.append(
                    {'rule_name': feature, 'confidence': sell_conf, 'support': total, 'lift': sell_lift,
                     'direction': 'DOWN'})

        buy_rules = pd.DataFrame(buy_conditions).sort_values('lift', ascending=False).reset_index(drop=True)
        sell_rules = pd.DataFrame(sell_conditions).sort_values('lift', ascending=False).reset_index(drop=True)
        all_rules = pd.concat([buy_rules, sell_rules], ignore_index=True).sort_values('lift',
                             ascending=False).reset_index(drop=True)
        if self.verbose:
            logger.info(f"[CandleMiner]: НАЙДЕНО: {len(buy_rules)} BUY, {len(sell_rules)} SELL")
        return buy_rules, sell_rules, all_rules

    def smart_analyze(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Кэш ИЛИ полный анализ"""
        cached = self.load_rules(symbol, timeframe)
        if cached:
            self._log_info(f"[CandleMiner]: КЭШ АКТУАЛЕН ({len(df)} свечей)")
            return {
                'all_rules': cached['top_rules'],
                'base_prob_up': cached['base_prob_up'],
                'base_prob_down': cached['base_prob_down'],
                'symbol': symbol, 'tf': timeframe, 'from_cache': True
            }

        self._log_info(f"[CandleMiner]: 🔥 ПОЛНЫЙ АНАЛИЗ {symbol} {timeframe}")

        feat_gen = Features(verbose=self.verbose)
        all_features = feat_gen.create_all_features(df)

        buy_rules, sell_rules, all_rules = self.find_strong_rules(all_features)
        base_prob_up = all_features['next_up'].mean()
        base_prob_down = all_features['next_down'].mean()

        results = {
            'all_features': all_features, 'buy_rules': buy_rules, 'sell_rules': sell_rules,
            'all_rules': all_rules, 'base_prob_up': base_prob_up, 'base_prob_down': base_prob_down,
            'symbol': symbol, 'tf_name': timeframe
        }

        self.save_rules(results, symbol, timeframe, min_confidence=0.70)
        return results

import logging

import numpy as np
import pandas as pd
import talib
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Features:
    """🔥 Генератор свечных признаков: base → volume → sequences → target"""
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _log_features(self, features: pd.DataFrame, stage: str = "features") -> None:
        """Логгер количества бинарных признаков"""
        if not self.verbose:
            return

        binary_cols = features.select_dtypes(include=['int64']).columns.tolist()
        logger.info(f"[Features]: ✅ {len(binary_cols)} бинарных {stage}!")
        logger.debug(f"[Features]: 📊 Бинарные фичи: {binary_cols[:10]}...")  # первые 10

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Рассчитывает ATR (Average True Range) для сравнения волатильности.

        Args:
            df: DataFrame с колонками ['high', 'low', 'close']
            period: период для ATR (по умолчанию 14)

        Returns:
            pd.Series с ATR значениями
        """
        high, low, close = df['high'], df['low'], df['close']
        atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
        return pd.Series(atr, index=df.index).bfill()

    def create_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создаёт базовые свечные признаки (OHLC геометрия + паттерны).

        Args:
            df: DataFrame с колонками ['open', 'high', 'low', 'close', 'tick_volume']

        Returns:
            pd.DataFrame с бинарными признаками
        """
        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        features = pd.DataFrame(index=df.index)

        # БАЗОВЫЕ РАЗМЕРЫ
        features['body_size'] = abs(c - o)
        features['upper_shadow'] = h - np.maximum(c, o)
        features['lower_shadow'] = np.minimum(c, o) - l
        features['total_range'] = h - l

        # % ОТ РЕЙНДЖА (float)
        features['body_pct'] = features['body_size'] / features['total_range'].replace(0, np.nan)
        features['upper_shadow_pct'] = features['upper_shadow'] / features['total_range'].replace(0, np.nan)
        features['lower_shadow_pct'] = features['lower_shadow'] / features['total_range'].replace(0, np.nan)

        # ПОЗИЦИЯ CLOSE (бинарные)
        features['close_top_30'] = (c >= h - (h - l) * 0.3).astype(int)
        features['close_bottom_30'] = (c <= l + (h - l) * 0.3).astype(int)
        features['close_middle'] = ((features['close_top_30'] == 0) & (features['close_bottom_30'] == 0)).astype(int)

        # СВЕЧНЫЕ ПАТТЕРНЫ (классика)
        features['doji'] = (features['body_pct'] < 0.1).astype(int)
        features['marubozu'] = (features['body_pct'] > 0.9).astype(int)
        features['hammer'] = ((features['lower_shadow_pct'] > 0.6) & (features['body_pct'] < 0.3)).astype(int)
        features['shooting_star'] = ((features['upper_shadow_pct'] > 0.6) & (features['body_pct'] < 0.3)).astype(int)
        features['spinning_top'] = ((features['body_pct'] < 0.2) &
                                    (features['upper_shadow_pct'] > 0.2) &
                                    (features['lower_shadow_pct'] > 0.2)).astype(int)
        features['small_body'] = (features['body_pct'] < 0.3).astype(int)

        # НАПРАВЛЕНИЕ + ТЕНИ
        features['bullish'] = (c > o).astype(int)
        features['bearish'] = (c < o).astype(int)
        features['upper_shadow_long'] = (features['upper_shadow_pct'] > 0.4).astype(int)
        features['lower_shadow_long'] = (features['lower_shadow_pct'] > 0.4).astype(int)

        # ATR СРАВНЕНИЕ
        atr = self.calculate_atr(df, 14)
        features['atr_high'] = (features['total_range'] > atr * 1.5).astype(int)
        features['atr_low'] = (features['total_range'] < atr * 0.5).astype(int)

        # БОЛЬШИЕ ТЕЛА
        features['big_green'] = ((features['bullish'] == 1) & (features['body_pct'] > 0.6)).astype(int)
        features['big_red'] = ((features['bearish'] == 1) & (features['body_pct'] > 0.6)).astype(int)

        self._log_features(features, "БАЗОВЫХ свечных")
        return features

    def add_volume_combos(self, candle_features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет volume фичи + ДИНАМИЧЕСКИЕ комбинации паттерн×volume.

        Args:
            candle_features: результат create_candle_features()
            df: исходный DataFrame (нужен tick_volume)

        Returns:
            pd.DataFrame с volume комбинациями
        """
        features = candle_features.copy()
        tv = df['tick_volume']

        # VOLUME БАЗОВЫЕ
        features['vol_variation'] = (tv - tv.shift(1)).fillna(0)
        features['vol_spike'] = (features['vol_variation'] > tv.shift(1) * 1.9).fillna(0).astype(int)
        features['vol_drop'] = (features['vol_variation'] < -tv.shift(1) * 1.9).fillna(0).astype(int)
        features['vol_up'] = (features['vol_variation'] > tv.shift(1) * 0.2).fillna(0).astype(int)
        features['vol_down'] = (features['vol_variation'] < -tv.shift(1) * 0.2).fillna(0).astype(int)

        # ДИНАМИЧЕСКИ: ВСЕ binary × vol_types
        binary_patterns = candle_features.select_dtypes(include=['int64']).columns.tolist()
        vol_types = ['vol_spike', 'vol_drop', 'vol_up', 'vol_down']

        for pattern in binary_patterns:
            for vol_type in vol_types:
                new_feature = f'{pattern}_{vol_type}'
                features[new_feature] = (features[pattern] & features[vol_type]).astype(int)

        self._log_features(features, "VOLUME_КОМБО")
        return features

    def add_equal_extremes(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔥 Равные экстремумы: H(n-1)≈H(n), L(n-1)≈L(n) — уровни поддержки/сопротивления
        """
        extremes_features = pd.DataFrame(index=df.index)
        h, l = df['high'], df['low']
        h1, l1 = h.shift(1), l.shift(1)

        # 🔥 АДАПТИВНАЯ ТОЧНОСТЬ: ATR * 0.05
        atr = self.calculate_atr(df, 14)
        tolerance_dynamic = atr * 0.05

        # Равные экстремумы
        extremes_features['equal_high'] = (abs(h - h1) < tolerance_dynamic).astype(int)
        extremes_features['equal_low'] = (abs(l - l1) < tolerance_dynamic).astype(int)

        # Volume комбинации (БЕЗОПАСНО)
        vol_types = ['vol_spike', 'vol_drop', 'vol_up', 'vol_down']
        for extreme in ['equal_high', 'equal_low']:
            for vol in vol_types:
                extremes_features[f'{extreme}_{vol}'] = (
                        extremes_features[extreme] & features[vol].fillna(0).astype(int)
                ).astype(int)

        # 🔥 ИСПРАВЛЕНО: fillna(0).astype(int) ПЕРЕД &
        extremes_features['equal_high_prev_bullish'] = (
                extremes_features['equal_high'] &
                features['bullish'].shift(1).fillna(0).astype(int)
        ).astype(int)

        extremes_features['equal_low_prev_bearish'] = (
                extremes_features['equal_low'] &
                features['bearish'].shift(1).fillna(0).astype(int)
        ).astype(int)

        # В add_equal_extremes ДОБАВЬ мощные комбинации:
        extremes_features['equal_high_rejection'] = (
                extremes_features['equal_high'] &
                features['upper_shadow_long'] &
                features['bearish']
        ).astype(int)  # Отбой от сопротивления!

        extremes_features['equal_low_bounce'] = (
                extremes_features['equal_low'] &
                features['lower_shadow_long'] &
                features['bullish']
        ).astype(int)  # Отбой от поддержки!
        print(f"equal_high: {(extremes_features['equal_high'] == 1).sum()} случаев")
        print(f"equal_low: {(extremes_features['equal_low'] == 1).sum()} случаев")
        return extremes_features

    def add_sequences(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет ДИНАМИЧЕСКИЕ последовательности + классические паттерны.

        Args:
            features: результат add_volume_combos()
            df: исходный DataFrame (OHLC)

        Returns:
            pd.DataFrame с последовательностями
        """
        seq_features = features.copy()
        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)

        # Base patterns
        binary_cols = features.select_dtypes(include=['int64']).columns.tolist()
        base_patterns = [col for col in binary_cols
                         if not col.startswith('vol_')
                         and col not in ['next_up', 'next_down']
                         and (features[col] == 1).sum() > 45]

        logger.info(f"🔄 Генерируем {len(base_patterns)}^2 = {len(base_patterns) ** 2} последовательностей...")

        sequence_columns = []

        # ДИНАМИЧЕСКИЕ последовательности
        for pat1 in base_patterns:
            for pat2 in base_patterns:
                seq_col = (
                        features[pat1].shift(1).fillna(0).astype(int) &
                        features[pat2].fillna(0).astype(int)
                )
                sequence_columns.append(seq_col.rename(f'{pat1}_prev_{pat2}'))

        # CLASSIC PATTERNS
        sequence_columns.extend([
            ((l < l1) & (features['vol_spike'] == 1) & (c > l * 1.002)).astype(int).rename('exhaustion_min'),
            ((h > h1) & (features['vol_spike'] == 1) & (c < h * 0.998)).astype(int).rename('exhaustion_max'),
            ((c1 < o1) & (c > o) & (o < c1) & (c > o1)).astype(int).rename('bullish_engulfing'),
            ((c1 > o1) & (c < o) & (o > c1) & (c < o1)).astype(int).rename('bearish_engulfing'),
            ((h < h1) & (l > l1)).astype(int).rename('inside_bar'),
            ((h > h1) & (l < l1)).astype(int).rename('outside_bar'),
            (h > h1).astype(int).rename('higher_high'),
            (l > l1).astype(int).rename('higher_low'),
            (h < h1).astype(int).rename('lower_high'),
            (l < l1).astype(int).rename('lower_low')
        ])

        # EQUAL EXTREMES
        equal_extremes = self.add_equal_extremes(features, df)
        for col in equal_extremes.columns:
            sequence_columns.append(equal_extremes[col])

        # 🔥 ОДИН pd.concat
        new_seq_df = pd.concat(sequence_columns, axis=1)

        # ✅ Объединяем БЕЗ дублей
        result = pd.concat([seq_features, new_seq_df], axis=1)
        result = result.loc[:, ~result.columns.duplicated()]  # убираем дубли

        self._log_features(result, "ПОСЛЕДОВАТЕЛЬНОСТЕЙ + EQUAL_EXTREMES")
        return result

    def create_target(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет целевые переменные для обучения.

        Args:
            df: исходный DataFrame
            features: фичи с последовательностями

        Returns:
            features с next_up/next_down
        """
        features['next_up'] = (df['close'].shift(-1) > df['close']).astype(int)
        features['next_down'] = (df['close'].shift(-1) < df['close']).astype(int)
        self._log_features(features, "FINALS с target")
        return features

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Полный пайплайн: base → volume → sequences → target.

        Returns:
            pd.DataFrame с  признаками
        """
        logger.info("[Features]: Запуск полного пайплайна Features...")

        print("[Features]: Генерация фич...")

        print("[Features]: 1/4 Базовые свечи...", end=" ")
        base = self.create_candle_features(df)
        print(f"[Features]: ✅ {len(base.select_dtypes('int64').columns)} фич")

        print("[Features]: 2/4 Volume комбо...", end=" ")
        vol_combos = self.add_volume_combos(base, df)
        print(f"[Features]: ✅ +{len(vol_combos.columns) - len(base.columns)} фич")

        print("[Features]: 3/4 Последовательности...", end=" ")
        sequences = self.add_sequences(vol_combos, df)
        print(f"[Features]: ✅ +{len(sequences.columns) - len(vol_combos.columns)} фич")

        print("[Features]: 4/4 Target...", end=" ")
        final = self.create_target(df, sequences)
        print("[Features]: ✅ ГОТОВО!")

        print(f"[Features]: ИТОГО: {len(final.select_dtypes('int64').columns)} бинарных фич")
        return final

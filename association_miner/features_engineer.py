import numpy as np
import pandas as pd
import talib

from utils.base_logger import BaseLogger


class Features(BaseLogger):
    """Генератор свечных признаков: base → volume → sequences → target"""

    def __init__(self, verbose: bool = False):
        """verbose=True → INFO логи | ERROR всегда активны"""
        super().__init__(verbose)



    def _log_features(self, features: pd.DataFrame, stage: str = "features"):
        """Логгер количества фич (только INFO)"""
        if not self.verbose:
            return
        binary_cols = features.select_dtypes(include=['int64']).columns.tolist()
        self._log_info(f" ✅ {len(binary_cols)} бинарных {stage}!")

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df['high'], df['low'], df['close']
        atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
        return pd.Series(atr, index=df.index).bfill()

    def create_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        features = pd.DataFrame(index=df.index)

        # БАЗОВЫЕ РАЗМЕРЫ
        features['body_size'] = abs(c - o)
        features['upper_shadow'] = h - np.maximum(c, o)
        features['lower_shadow'] = np.minimum(c, o) - l
        features['total_range'] = h - l

        # % ОТ РЕЙНДЖА
        features['body_pct'] = features['body_size'] / features['total_range'].replace(0, np.nan)
        features['upper_shadow_pct'] = features['upper_shadow'] / features['total_range'].replace(0, np.nan)
        features['lower_shadow_pct'] = features['lower_shadow'] / features['total_range'].replace(0, np.nan)

        # ПОЗИЦИЯ CLOSE
        features['close_top_30'] = (c >= h - (h - l) * 0.3).astype(int)
        features['close_bottom_30'] = (c <= l + (h - l) * 0.3).astype(int)
        features['close_middle'] = ((features['close_top_30'] == 0) & (features['close_bottom_30'] == 0)).astype(int)

        # СВЕЧНЫЕ ПАТТЕРНЫ
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

        # 🔥 ORDER BLOCKS (SMC)
        h1, l1, c1 = h.shift(1), l.shift(1), c.shift(1)
        o1 = o.shift(1)

        # ✅ BULLISH OB: Медвежья свеча + сильный рост
        features['is_bullish_ob'] = (
                (c1 < o1) &  # Предыдущая свеча медвежья (красная)
                (c > h1 * 1.002)  # Текущая пробила high предыдущей
        ).astype(int)

        # ✅ BEARISH OB: Бычья свеча + сильное падение
        features['is_bearish_ob'] = (
                (c1 > o1) &  # Предыдущая свеча бычья (зеленая)
                (c < l1 * 0.998)  # Текущая пробила low предыдущей
        ).astype(int)

        self._log_features(features, "БАЗОВЫХ свечных")
        return features

    def add_volume_combos(self, candle_features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
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

    def add_sequences(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        seq_features = features.copy()
        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)

        # Base patterns для последовательностей
        binary_cols = features.select_dtypes(include=['int64']).columns.tolist()
        base_patterns = [col for col in binary_cols
                         if not col.startswith('vol_')
                         and col not in ['next_up', 'next_down']
                         and (features[col] == 1).sum() > 45]

        self._log_info(f"🔄 Генерируем {len(base_patterns)}^2 = {len(base_patterns) ** 2} последовательностей...")

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

        new_seq_df = pd.concat(sequence_columns, axis=1)
        result = pd.concat([seq_features, new_seq_df], axis=1)
        result = result.loc[:, ~result.columns.duplicated()]

        self._log_features(result, "ПОСЛЕДОВАТЕЛЬНОСТЕЙ")
        return result

    def add_equal_extremes(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """🔥 Равные экстремумы (двойные/тройные вершины-днища)"""
        h, l, c = df['high'], df['low'], df['close']

        # Shifted highs/lows
        h1, l1 = h.shift(1), l.shift(1)
        h2, l2 = h.shift(2), l.shift(2)
        h3, l3 = h.shift(3), l.shift(3)

        result = pd.DataFrame(index=features.index)

        # Двойные вершины (highs в пределах 0.05%)
        result['double_top'] = (
                (h >= h1 * 0.9995) & (h1 >= h2 * 0.999) &
                (c < h * 0.998)  # Закрытие ниже вершины
        ).astype(int)

        # Двойные днища
        result['double_bottom'] = (
                (l <= l1 * 1.0005) & (l1 <= l2 * 1.001) &
                (c > l * 1.002)  # Закрытие выше дна
        ).astype(int)

        # Тройные
        result['triple_top'] = (
                (h >= h1 * 0.9995) & (h >= h2 * 0.999) & (h >= h3 * 0.998)
        ).astype(int)
        result['triple_bottom'] = (
                (l <= l1 * 1.0005) & (l <= l2 * 1.001) & (l <= l3 * 1.002)
        ).astype(int)

        # С volume спайком
        if 'vol_spike' in features.columns:
            result['double_top_vol'] = (result['double_top'] & features['vol_spike']).astype(int)
            result['double_bottom_vol'] = (result['double_bottom'] & features['vol_spike']).astype(int)

        self._log_features(result, "EQUAL_EXTREMES")
        return result

    def add_trend_ma(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        ma21 = df['close'].rolling(21).mean()
        ma50 = df['close'].rolling(50).mean()
        ma200 = df['close'].rolling(200).mean()

        features['ma_bull_21_50'] = (ma21 > ma50).astype(int)
        features['ma_bear_21_50'] = (ma21 < ma50).astype(int)
        features['ma_bull_all'] = ((ma21 > ma50) & (ma50 > ma200)).astype(int)
        features['ma_bear_all'] = ((ma21 < ma50) & (ma50 < ma200)).astype(int)

        features['price_above_all_ma'] = ((df['close'] > ma21) & (df['close'] > ma50) & (df['close'] > ma200)).astype(
            int)
        features['price_below_all_ma'] = ((df['close'] < ma21) & (df['close'] < ma50) & (df['close'] < ma200)).astype(
            int)

        features['bearish_below_all_ma'] = (features['bearish'] & features['price_below_all_ma']).astype(int)
        features['bullish_above_all_ma'] = (features['bullish'] & features['price_above_all_ma']).astype(int)

        return features

    def create_target(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        features['next_up'] = (df['close'].shift(-1) > df['close']).astype(int)
        features['next_down'] = (df['close'].shift(-1) < df['close']).astype(int)
        self._log_features(features, "FINALS с target")
        return features

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ Полный пайплайн БЕЗ СПАМА"""
        self._log_info("[Features]: Запуск пайплайна...")

        # 1. Базовые свечи
        self._log_info("[Features]: 1/5 Базовые свечи...")
        base = self.create_candle_features(df)

        # 2. Volume
        self._log_info("[Features]: 2/5 Volume комбо...")
        vol_combos = self.add_volume_combos(base, df)

        # 3. MA
        self._log_info("[Features]: 3/5 Трендовые MA...")
        trend_features = self.add_trend_ma(df, vol_combos)

        # 4. Sequences
        self._log_info("[Features]: 4/5 Последовательности...")
        sequences = self.add_sequences(trend_features, df)

        # 5. Target
        self._log_info("[Features]: 5/5 Target...")
        final = self.create_target(df, sequences)

        return final

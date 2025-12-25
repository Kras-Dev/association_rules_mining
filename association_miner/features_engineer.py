import numpy as np
import pandas as pd
import talib
from utils.base_logger import BaseLogger


class Features(BaseLogger):
    """Генератор свечных признаков: base → volume → sequences → target"""

    def __init__(self, verbose: bool = False):
        """verbose=True → INFO логи | ERROR всегда активны"""
        super().__init__(verbose)
        self.warmup_period = 200
        self._atr_cache = None

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        cache_key = (id(df), period)
        if self._atr_cache and cache_key == self._atr_cache['key']:
            return self._atr_cache['atr']

        atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        self._atr_cache = {'key': cache_key, 'atr': pd.Series(atr, index=df.index)}
        return self._atr_cache['atr']

    def create_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        atr = self.calculate_atr(df)
        features = pd.DataFrame(index=df.index)

        # БАЗОВЫЕ РАЗМЕРЫ
        features['body_size'] = abs(c - o)
        features['upper_shadow'] = h - np.maximum(c, o)
        features['lower_shadow'] = np.minimum(c, o) - l
        features['total_range'] = h - l

        # ГАП-ФИЛЬТР
        prev_close = c.shift(1).fillna(c)
        gap_up = (o > prev_close * 1.002).astype(float)  # Gap >0.2%
        gap_down = (o < prev_close * 0.998).astype(float)  # Gap >0.2%
        big_gap_up = (o > prev_close * 1.005).astype(float)  # Gap >0.5%
        big_gap_down = (o < prev_close * 0.995).astype(float)  # Gap >0.5%

        # Гап + объем
        features['gap_up'] = gap_up
        features['gap_down'] = gap_down
        features['big_gap_up'] = big_gap_up
        features['big_gap_down'] = big_gap_down

        # % ОТ РЕЙНДЖА
        total_range_safe = features['total_range'].replace(0, np.nan)
        features['body_pct'] = features['body_size'] / total_range_safe
        features['upper_shadow_pct'] = features['upper_shadow'] / total_range_safe
        features['lower_shadow_pct'] = features['lower_shadow'] / total_range_safe

        # ПОЗИЦИЯ CLOSE
        range_val = h - l
        range_val_safe = range_val.replace(0, np.nan)
        features['close_top_30'] = (c >= h - range_val * 0.3).astype(float)
        features['close_bottom_30'] = (c <= l + range_val * 0.3).astype(float)
        features['close_middle'] = ((features['close_top_30'] == 0) & (features['close_bottom_30'] == 0)).astype(float)


        # СВЕЧНЫЕ ПАТТЕРНЫ
        features['doji'] = (features['body_pct'] <  0.1).astype(float)
        features['marubozu'] = (features['body_pct'] > 0.9).astype(float)
        features['hammer'] = ((features['lower_shadow_pct'] > 0.6) & (features['body_pct'] < 0.3)).astype(float)
        features['shooting_star'] = ((features['upper_shadow_pct'] > 0.6) & (features['body_pct'] < 0.3)).astype(float)
        features['spinning_top'] = ((features['body_pct'] < 0.2) &
                                    (features['upper_shadow_pct'] > 0.2) &
                                    (features['lower_shadow_pct'] > 0.2)).astype(float)
        features['small_body'] = (features['body_pct'] < 0.3).astype(float)

        # НАПРАВЛЕНИЕ + ТЕНИ
        features['bullish'] = (c > o).astype(float)
        features['bearish'] = (c < o).astype(float)
        features['upper_shadow_long'] = (features['upper_shadow_pct'] > 0.4).astype(float)
        features['lower_shadow_long'] = (features['lower_shadow_pct'] > 0.4).astype(float)

        # ATR СРАВНЕНИЕ
        features['atr_high'] = (features['total_range'] > atr * 1.5).astype(float)
        features['atr_low'] = (features['total_range'] < atr * 0.5).astype(float)

        # БОЛЬШИЕ ТЕЛА
        features['big_green'] = (((features['bullish'] == 1) & (features['body_pct'] > 0.6))
                                .astype(float))
        features['big_red'] = (((features['bearish'] == 1) & (features['body_pct'] > 0.6))
                                .astype(float))

        # ORDER BLOCKS (SMC)
        # BULLISH OB: Медвежья свеча + сильный рост
        features['is_bullish_ob'] = (
                (c.shift(1).fillna(0) < o.shift(1).fillna(999)) &
                (c > h.shift(1).fillna(0)) &
                (features['total_range'] > atr)
        ).astype(float)

        features['is_bearish_ob'] = (
                (c.shift(1).fillna(999) > o.shift(1).fillna(0)) &
                (c < l.shift(1).fillna(999)) &
                (features['total_range'] > atr)
        ).astype(float)

        self._log_debug(f"Базовых свечных признаков: {len(features)}")
        return features

    def add_volume_combos(self, candle_features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        features = candle_features.copy()
        tv = df['tick_volume']

        # VOLUME БАЗОВЫЕ
        features['vol_variation'] = (tv - tv.shift(1)).fillna(0)
        features['vol_spike'] = (features['vol_variation'] > tv.shift(1) * 1.9).fillna(0).astype(float)
        features['vol_drop'] = (features['vol_variation'] < -tv.shift(1) * 1.9).fillna(0).astype(float)
        features['vol_up'] = (features['vol_variation'] > tv.shift(1) * 0.2).fillna(0).astype(float)
        features['vol_down'] = (features['vol_variation'] < -tv.shift(1) * 0.2).fillna(0).astype(float)

        # ДИНАМИЧЕСКИ: ВСЕ binary × vol_types (Векторизация вместо Parallel)
        binary_patterns = [col for col in candle_features.columns if candle_features[col].nunique() <= 2]
        vol_types = ['vol_spike', 'vol_drop', 'vol_up', 'vol_down']

        new_combo_cols = {}
        for pattern in binary_patterns:
            for vol_type in vol_types:
                new_feature_name = f'{pattern}_{vol_type}'
                # Умножение float 0.0 и 1.0 дает тот же результат, что и AND,
                # но не боится NaN и работает быстрее
                new_combo_cols[new_feature_name] = features[pattern] * features[vol_type]

        # Добавляем новые колонки БЕЗ приведения к uint8 здесь
        combos_df = pd.DataFrame(new_combo_cols, index=features.index)
        features = pd.concat([features, combos_df], axis=1)

        self._log_debug(f"Volumes признаки: {len(features)}")
        return features

    def add_sequences(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        seq_features = features.copy()
        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)

        # 1. Фильтрация качественных базовых паттернов
        binary_cols = [col for col in features.columns if features[col].nunique() <= 2]
        sums = features[binary_cols].sum()
        total_rows = len(features)

        base_patterns = [col for col in binary_cols if
                         not col.startswith('vol_') and
                         col not in ['next_up', 'next_down'] and
                         20 < sums[col] < (total_rows * 0.85)]

        vol_patterns = [col for col in binary_cols if col.startswith('vol_') and sums[col] > 15]

        self._log_info(f"🔄 Генерируем {len(base_patterns)}^2 = {len(base_patterns) ** 2} последовательностей...")

        # Словарь для быстрой сборки новых колонок
        new_cols_dict = {}

        # 2. Динамические последовательности (Candle + Candle)
        for p1 in base_patterns:
            p1_shifted = features[p1].shift(1).fillna(0)
            for p2 in base_patterns:
                new_cols_dict[f'{p1}_prev_{p2}'] = p1_shifted * features[p2]

        # 3. Объемные последовательности (ВЫНЕСЕНО ИЗ ВНУТРЕННЕГО ЦИКЛА)
        for candle in base_patterns:
            candle_shifted = features[candle].shift(1).fillna(0)
            for vol in vol_patterns:
                vol_shifted = features[vol].shift(1).fillna(0)
                # Текущая свеча + предыдущий volume
                new_cols_dict[f'{candle}_prev_{vol}'] = features[candle] * vol_shifted
                # Предыдущая свеча + текущий volume
                new_cols_dict[f'{vol}_curr_{candle}'] = vol_shifted * features[candle]

        # Создаем DataFrame из словаря
        seq_df = pd.DataFrame(new_cols_dict, index=features.index)

        # 4. УМНЫЙ ОТСЕВ: Удаляем последовательности, которые встречаются слишком редко (меньше 10 раз)
        # Это не даст им пройти порог Confidence > 70% и сэкономит память
        seq_sums = seq_df.sum()
        valid_cols = seq_sums[seq_sums >= 10].index
        seq_df = seq_df[valid_cols]
        self._log_info(f"✅ После фильтрации по Support осталось {len(seq_df.columns)} последовательностей")

        # 5. CLASSIC PATTERNS (Сборка в список)
        classic_columns = [
            ((l < l1) & (features['vol_spike'] == 1) & (c > l * 1.002)).astype(float).rename('exhaustion_min'),
            ((h > h1) & (features['vol_spike'] == 1) & (c < h * 0.998)).astype(float).rename('exhaustion_max'),
            ((c1 < o1) & (c > o) & (o < c1) & (c > o1)).astype(float).rename('bullish_engulfing'),
            ((c1 > o1) & (c < o) & (o > c1) & (c < o1)).astype(float).rename('bearish_engulfing'),
            ((h < h1) & (l > l1)).astype(float).rename('inside_bar'),
            ((h > h1) & (l < l1)).astype(float).rename('outside_bar'),
            (h > h1).astype(float).rename('higher_high'),
            (l > l1).astype(float).rename('higher_low'),
            (h < h1).astype(float).rename('lower_high'),
            (l < l1).astype(float).rename('lower_low')
        ]

        # 6. EQUAL EXTREMES
        eq_df = self.add_equal_extremes(features, df)

        # ФИНАЛЬНАЯ КОНКАТЕНАЦИЯ
        # Собираем всё: базовые + отфильтрованные секвенции + классика + экстремумы
        result = pd.concat([
            seq_features,
            seq_df,
            pd.concat(classic_columns, axis=1),
            eq_df
        ], axis=1)

        # Удаляем дубликаты (на всякий случай)
        result = result.loc[:, ~result.columns.duplicated(keep='last')]

        self._log_debug(f"Последовательности: {len(result.columns)} колонок")
        return result

    def add_equal_extremes(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Равные экстремумы (двойные/тройные вершины-днища) с динамическим порогом ATR"""
        h, l, c = df['high'], df['low'], df['close']

        h1, l1 = h.shift(1).fillna(h), l.shift(1).fillna(l)
        h2, l2 = h.shift(2).fillna(h), l.shift(2).fillna(l)
        h3, l3 = h.shift(3).fillna(h), l.shift(3).fillna(l)

        result = pd.DataFrame(index=features.index)
        atr = self.calculate_atr(df)
        threshold = atr * 0.1 # Динамический порог в пунктах

        # Двойные вершины (highs в пределах ATR*0.1)
        result['double_top'] = (
                (abs(h - h1) < threshold) &
                (abs(h1 - h2) < threshold) &
                (c < h - threshold) # Закрытие ниже вершины на величину порога
        ).astype(float)

        # Двойные днища
        result['double_bottom'] = (
                (abs(l - l1) < threshold) &
                (abs(l1 - l2) < threshold) &
                (c > l + threshold) # Закрытие выше дна на величину порога
        ).astype(float)

        # Тройные
        result['triple_top'] = (
                (abs(h - h1) < threshold) &
                (abs(h - h2) < threshold) &
                (abs(h - h3) < threshold)
        ).astype(float)

        result['triple_bottom'] = (
                (abs(l - l1) < threshold) &
                (abs(l - l2) < threshold) &
                (abs(l - l3) < threshold)
        ).astype(float)

        # С volume спайком
        if 'vol_spike' in features.columns:
            result['double_top_vol'] = (result['double_top'] * features['vol_spike']).astype(float)
            result['double_bottom_vol'] = (result['double_bottom'] * features['vol_spike']).astype(float)

        result['eq_highs'] = (abs(h - h1) < threshold).astype(float)
        result['eq_lows'] = (abs(l - l1) < threshold).astype(float)
        result['liquidity_sweep_high'] = (result['eq_highs'].shift(1).fillna(0) * (h > h1)
                                       .astype(float) * (c < h1).astype(float))

        self._log_debug(f"EQUAL_EXTREMES: {len(result)}")
        return result

    def add_trend_ma(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        ma21 = df['close'].rolling(21).mean()
        ma50 = df['close'].rolling(50).mean()
        ma200 = df['close'].rolling(200).mean()

        features['ma_bull_21_50'] = (ma21 > ma50).astype(float)
        features['ma_bear_21_50'] = (ma21 < ma50).astype(float)
        features['ma_bull_all'] = ((ma21 > ma50) & (ma50 > ma200)).astype(float)
        features['ma_bear_all'] = ((ma21 < ma50) & (ma50 < ma200)).astype(float)

        features['price_above_all_ma'] = ((df['close'] > ma21) & (df['close'] > ma50) & (df['close'] > ma200)).astype(
            float)
        features['price_below_all_ma'] = ((df['close'] < ma21) & (df['close'] < ma50) & (df['close'] < ma200)).astype(
            float)

        features['bearish_below_all_ma'] = (features['bearish'] * features['price_below_all_ma']).astype(float)
        features['bullish_above_all_ma'] = (features['bullish'] * features['price_above_all_ma']).astype(float)
        # Дистанция до MA (если цена слишком далеко от MA200 - риск разворота)
        features['overextended_high'] = (df['close'] > ma200 + (self.calculate_atr(df) * 5)).astype(float)
        # Расчет ADX (обычно период 14)
        c, h, l = df['close'], df['high'], df['low']
        adx = talib.ADX(h.values, l.values, c.values, timeperiod=14)
        features['adx_no_trend'] = (adx < 20).astype(float)
        features['adx_strong_trend'] = (adx > 25).astype(float)


        self._log_debug(f"Trend & ma: {len(features)}")
        return features

    def add_volume_vsa(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """VSA (Volume Spread Analysis) признаки."""
        v, c, o = df['tick_volume'], df['close'], df['open']
        v_ma = v.rolling(20).mean()
        features['vol_climax'] = (v > v_ma * 2.0).astype(float)
        features['vol_low'] = (v < v_ma * 0.5).astype(float)
        features['effort_no_result_bull'] = ((c > o) & (v < v.shift(1).fillna(v))).astype(float)
        features['effort_no_result_bear'] = ((c < o) & (v < v.shift(1).fillna(v))).astype(float)

        self._log_debug(f"VSA: {len(features)}")
        return features


    # def create_target(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    #     # Таргеты на 1 шаг вперед. Look-ahead bias отсутствует, если последняя строка удаляется позже.
    #     features['next_up'] = (df['close'].shift(-1) > df['close']).astype(float)
    #     features['next_down'] = (df['close'].shift(-1) < df['close']).astype(float)
    #
    #     self._log_debug(f"FINALS с target:{len(features)}")
    #     return features
    def create_target(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        atr = self.calculate_atr(df)  # Используем ваш метод ATR

        # Сдвиг на -1 (следующая свеча)
        diff = df['close'].shift(-1) - df['close']

        # Цель: движение вверх больше, чем 0.2 * ATR (фильтр шума)
        features['next_up'] = (diff > (atr * 0.2)).astype(float)
        features['next_down'] = (diff < -(atr * 0.2)).astype(float)

        return features

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Полный пайплайн"""
        self._log_debug("Запуск пайплайна...")
        self._log_debug(f"Генерация признаков для {len(df)} баров...")
        # 1. Базовые свечи
        self._log_info("1/6 Базовые свечи...")
        base = self.create_candle_features(df)

        # 2. Volume
        self._log_info("2/6 Volume комбо...")
        vol_combos = self.add_volume_combos(base, df)

        # 3. MA тренды
        self._log_info("3/6 Трендовые MA...")
        trend_features = self.add_trend_ma(df, vol_combos)

        # 4. VSA
        self._log_info("4/6 VSA Volume...")
        vsa_features = self.add_volume_vsa(df, trend_features)

        # 5. Sequences
        self._log_info("5/6 Последовательности...")
        sequences = self.add_sequences(vsa_features, df)

        # 6. Target + финальная очистка
        self._log_info("6/6 Target + cleanup...")
        final = self.create_target(df, sequences)

        # ФИНАЛЬНАЯ БИНАРИЗАЦИЯ И СЖАТИЕ (Защита 2025)
        final = final.select_dtypes(include=['number', 'bool']).replace([np.inf, -np.inf], np.nan).fillna(0)
        final_uint8 = (final > 0.5).astype(np.uint8)

        # Отрезаем прогрев. Теперь в result строк меньше, чем в исходном df на 200 штук.
        result = final_uint8.iloc[self.warmup_period:]
        self._log_info(f"✅ Готово: {result.shape[1]} фич")
        return result

    def test_features(self, df: pd.DataFrame):
        features = self.create_all_features(df)
        assert not features.empty and features.isna().sum().sum() == 0, "Ошибка данных!"
        assert all(features[c].dtype == np.uint8 for c in features.columns), "Тип не uint8!"
        self._log_info("💎 Тест пройден!")
        return features
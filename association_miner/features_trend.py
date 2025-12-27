import numpy as np
import pandas as pd
import talib
from utils.base_logger import BaseLogger


class FeaturesTrend(BaseLogger):
    """
    Генератор признаков для трендовой стратегии:
    - тренд по МА + ADX
    - откат к MA21
    - continuation + entry_long/short с фильтрами
    - таргет на N баров по ATR
    """

    def __init__(self, verbose: bool = False, warmup_period: int = 200):
        super().__init__(verbose)
        self.warmup_period = warmup_period
        self._atr_cache = None

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        cache_key = (id(df), period)
        if self._atr_cache and cache_key == self._atr_cache["key"]:
            return self._atr_cache["atr"]

        atr = talib.ATR(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            timeperiod=period,
        )
        atr = pd.Series(atr, index=df.index)
        self._atr_cache = {"key": cache_key, "atr": atr}
        return atr

    def create_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        atr = self.calculate_atr(df)

        f = pd.DataFrame(index=df.index)

        # размеры
        f["body_size"] = (c - o).abs()
        f["upper_shadow"] = h - np.maximum(c, o)
        f["lower_shadow"] = np.minimum(c, o) - l
        f["total_range"] = (h - l).replace(0, np.nan)

        f["body_pct"] = f["body_size"] / f["total_range"]
        f["upper_shadow_pct"] = f["upper_shadow"] / f["total_range"]
        f["lower_shadow_pct"] = f["lower_shadow"] / f["total_range"]

        # направление
        f["bullish"] = (c > o).astype(float)
        f["bearish"] = (c < o).astype(float)

        # позиция close
        f["close_top_30"] = (c >= h - (h - l) * 0.3).astype(float)
        f["close_bottom_30"] = (c <= l + (h - l) * 0.3).astype(float)

        # базовые паттерны
        f["doji"] = (f["body_pct"] < 0.1).astype(float)
        f["spinning_top"] = (
                (f["body_pct"] < 0.2)
                & (f["upper_shadow_pct"] > 0.2)
                & (f["lower_shadow_pct"] > 0.2)
        ).astype(float)
        f["hammer"] = (
                (f["lower_shadow_pct"] > 0.6) & (f["body_pct"] < 0.3)
        ).astype(float)
        f["shooting_star"] = (
                (f["upper_shadow_pct"] > 0.6) & (f["body_pct"] < 0.3)
        ).astype(float)

        # большие тела
        f["big_green"] = (
                (f["bullish"] == 1) & (f["body_pct"] > 0.6)
        ).astype(float)
        f["big_red"] = (
                (f["bearish"] == 1) & (f["body_pct"] > 0.6)
        ).astype(float)

        # volume (добавил базовый vol_spike)
        v = df.get("tick_volume", pd.Series(0, index=df.index))
        v_ma = v.rolling(20).mean()
        f["vol_spike"] = (v > v_ma * 2.0).astype(float)

        # ATR
        f["atr"] = atr
        f["atr_high"] = (f["total_range"] > atr * 1.5).astype(float)
        f["atr_low"] = (f["total_range"] < atr * 0.5).astype(float)

        self._log_debug(f"Базовых свечных признаков: {len(f.columns)}")
        return f

    def add_trend_ma(self, df: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        ma21 = c.rolling(21).mean()
        ma50 = c.rolling(50).mean()
        ma200 = c.rolling(200).mean()

        f["ma21"] = ma21
        f["ma50"] = ma50
        f["ma200"] = ma200

        f["trend_up"] = ((ma21 > ma50) & (ma50 > ma200)).astype(float)
        f["trend_down"] = ((ma21 < ma50) & (ma50 < ma200)).astype(float)

        # цена относительно МА
        f["price_above_ma21"] = (c > ma21).astype(float)
        f["price_above_ma50"] = (c > ma50).astype(float)
        f["price_above_ma200"] = (c > ma200).astype(float)
        f["price_below_ma21"] = (c < ma21).astype(float)
        f["price_below_ma50"] = (c < ma50).astype(float)
        f["price_below_ma200"] = (c < ma200).astype(float)

        # серии подряд
        for n in [3, 5]:
            f[f"trend_up_run_{n}"] = (
                f["trend_up"].rolling(n).sum().eq(n).astype(float)
            )
            f[f"trend_down_run_{n}"] = (
                f["trend_down"].rolling(n).sum().eq(n).astype(float)
            )
            f[f"above_ma21_run_{n}"] = (
                f["price_above_ma21"].rolling(n).sum().eq(n).astype(float)
            )
            f[f"below_ma21_run_{n}"] = (
                f["price_below_ma21"].rolling(n).sum().eq(n).astype(float)
            )

        # ADX
        adx = talib.ADX(
            df["high"].values,
            df["low"].values,
            c.values,
            timeperiod=14,
        )
        adx = pd.Series(adx, index=df.index)
        f["adx_strong"] = (adx > 25).astype(float)
        f["adx_weak"] = (adx < 20).astype(float)

        # перекупленность/перепроданность
        f["overextended_high"] = (c > ma200 + f["atr"] * 5).astype(float)
        f["overextended_low"] = (c < ma200 - f["atr"] * 5).astype(float)

        self._log_debug(f"Trend+MA: {len(f.columns)}")
        return f

    def add_pullback_continuation(self, df: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
        o, c, l, h = df["open"], df["close"], df["low"], df["high"]
        ma21 = f["ma21"]
        atr = f["atr"]

        # расстояние до MA21
        dist_to_ma21 = (c - ma21).abs()
        f["dist_to_ma21_atr"] = dist_to_ma21 / atr.replace(0, np.nan)

        # предыдущая цена относительно MA21
        close_above_ma21_prev = f["price_above_ma21"].shift(1).fillna(0)
        close_below_ma21_prev = f["price_below_ma21"].shift(1).fillna(0)

        # ✅ ИСПРАВЛЕННЫЕ откаты
        f["pullback_to_ma21_up"] = (
                (f["trend_up_run_3"] == 1)  # тренд вверх
                & (close_above_ma21_prev == 1)  # была над MA21
                & (dist_to_ma21 <= atr * 0.5)  # подошла близко
        ).astype(float)

        f["pullback_to_ma21_down"] = (
                (f["trend_down_run_3"] == 1)  # тренд вниз
                & (close_below_ma21_prev == 1)  # была под MA21
                & (dist_to_ma21 <= atr * 0.5)  # подошла близко
        ).astype(float)

        # базовые continuation
        f["continuation_long"] = (
                (f["pullback_to_ma21_up"].shift(1) == 1)
                & (f["bullish"] == 1)
                & (f["body_pct"] > 0.5)
                & (f["close_top_30"] == 1)
        ).astype(float)

        f["continuation_short"] = (
                (f["pullback_to_ma21_down"].shift(1) == 1)
                & (f["bearish"] == 1)
                & (f["body_pct"] > 0.5)
                & (f["close_bottom_30"] == 1)
        ).astype(float)

        # сетапы
        f["setup_long_trend_pullback"] = (
                (f["trend_up_run_3"] == 1) & (f["pullback_to_ma21_up"] == 1)
        ).astype(float)
        f["setup_short_trend_pullback"] = (
                (f["trend_down_run_3"] == 1) & (f["pullback_to_ma21_down"] == 1)
        ).astype(float)

        # 🔥 УМНЫЕ ENTRY С ФИЛЬТРАМИ (против mean reversion)
        f["entry_long"] = (
                (f["continuation_long"] == 1)  # базовый сетап
                & (f["adx_strong"] == 1)  # сильный тренд
                & (f["vol_spike"].shift(1).fillna(0) == 0)  # откат БЕЗ объёма
                & (f["price_above_ma50"] == 1)  # выше MA50
                & (f["overextended_high"] == 0)  # не перекуплено
        ).astype(float)

        f["entry_short"] = (
                (f["continuation_short"] == 1)
                & (f["adx_strong"] == 1)
                & (f["vol_spike"].shift(1).fillna(0) == 0)
                & (f["price_below_ma50"] == 1)
                & (f["overextended_low"] == 0)
        ).astype(float)

        # контртренд версии (по ARM)
        f["entry_reversal_long"] = (
                (f["continuation_long"] == 1) & (f["big_red"] == 1)
        ).astype(float)
        f["entry_reversal_short"] = (
                (f["continuation_short"] == 1) & (f["big_green"] == 1)
        ).astype(float)

        self._log_debug(f"Entry signals: long={f['entry_long'].sum()}, short={f['entry_short'].sum()}")
        return f

    def create_target(self, df: pd.DataFrame, f: pd.DataFrame,
                      horizon: int = 3, atr_mult: float = 0.5) -> pd.DataFrame:
        c = df["close"]
        atr = f["atr"]

        future_max = c.shift(-1).rolling(horizon, min_periods=1).max()
        future_min = c.shift(-1).rolling(horizon, min_periods=1).min()

        up_move = future_max - c
        down_move = c - future_min

        f["next_up"] = (up_move > atr * atr_mult).astype(float)
        f["next_down"] = (down_move > atr * atr_mult).astype(float)

        self._log_debug(
            f"Target(h={horizon}, k={atr_mult}): up={f['next_up'].sum()}, "
            f"down={f['next_down'].sum()}"
        )
        return f

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log_debug("🚀 Трендовый пайплайн...")

        f = self.create_candle_features(df)
        f = self.add_trend_ma(df, f)
        f = self.add_pullback_continuation(df, f)
        f = self.create_target(df, f, horizon=3, atr_mult=0.5)

        # зачистка
        f = f.iloc[self.warmup_period:]
        f = f.dropna()

        # только бинарные для ARM
        binary_cols = []
        for col in f.columns:
            if f[col].dtype in [float, int]:
                uniq = f[col].dropna().unique()
                if len(uniq) <= 3 and set(uniq).issubset({0., 1., 0, 1}):
                    binary_cols.append(col)

        result = f[binary_cols].astype(np.uint8)

        self._log_info(f"✅ {len(result.columns)} бинарных фич | entry_long={result['entry_long'].sum()}")
        return result

    def test_features(self, df: pd.DataFrame):
        f = self.create_all_features(df)
        assert not f.empty, "Пустые фичи!"
        assert f.isna().sum().sum() == 0, "NaN!"
        assert all(f[c].dtype == np.uint8 for c in f.columns), "Не uint8!"
        self._log_info("💎 Тест OK!")
        return f

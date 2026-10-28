"""Causal adaptive regime engine for research and paper trading.

Ranks candidate signals using only prior-bar performance. This is research tooling,
not a prediction guarantee or live execution engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtest import signal_series

DEFAULT_CANDIDATES = [
    "EMA crossover", "RSI mean reversion", "Donchian breakout",
    "MACD trend", "Bollinger reversion", "ATR trend filter",
    "EMA + RSI confirmation",
]


def _candidate_positions(df, market, fee, slippage, candidates):
    close = df["close"]
    bar_returns = close.pct_change().fillna(0.0)
    positions, pnl = {}, {}
    for name in candidates:
        # Signal calculated at close t, position is held starting at t+1.
        raw = signal_series(df, name).shift(1).fillna(0).clip(-1, 1)
        if market == "spot":
            raw = raw.clip(0, 1)
        change = raw.diff().abs().fillna(raw.abs())
        returns = raw * bar_returns - change * (fee + slippage)
        positions[name] = raw
        pnl[name] = returns
    return pd.DataFrame(positions, index=df.index), pd.DataFrame(pnl, index=df.index)


def adaptive_regime_backtest(
    df: pd.DataFrame,
    initial_equity=1000.0,
    market="spot",
    fee=0.001,
    slippage=0.0005,
    lookback=240,
    min_score=0.0,
    risk_off_vol_percentile=0.98,
    candidates=None,
    position_size_pct=100.0,
):
    """Walk forward by bar; exposure is capped by position_size_pct."""
    if not isinstance(df, pd.DataFrame) or len(df) < max(300, int(lookback) + 50):
        raise ValueError("Для адаптивного теста нужно минимум 300 свечей и желательно 1 000+.")
    if market not in {"spot", "linear"}:
        raise ValueError("Рынок должен быть spot или linear.")
    if not np.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("Начальный капитал должен быть положительным.")
    if not 0 <= float(position_size_pct) <= 100:
        raise ValueError("Размер позиции должен быть от 0 до 100%.")
    if fee < 0 or slippage < 0:
        raise ValueError("Комиссия и проскальзывание не могут быть отрицательными.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Индекс данных должен быть DatetimeIndex.")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("Свечи должны быть отсортированы по времени без дубликатов.")

    lookback = max(80, min(int(lookback), len(df) - 50))
    candidates = candidates or DEFAULT_CANDIDATES
    pos_frame, pnl_frame = _candidate_positions(df, market, fee, slippage, candidates)

    # At timestamp t, score includes candidate returns only through t-1.
    min_periods = max(60, lookback // 2)
    mean = pnl_frame.rolling(lookback, min_periods=min_periods).mean().shift(1)
    std = pnl_frame.rolling(lookback, min_periods=min_periods).std(ddof=1).shift(1)
    score = (mean / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    has_score = score.notna().any(axis=1)
    chosen = score.fillna(-np.inf).idxmax(axis=1).where(has_score, "NO_EDGE")
    best_score = score.max(axis=1)

    ret = df["close"].pct_change().fillna(0.0)
    vol = ret.rolling(30, min_periods=20).std()
    vol_cut = vol.rolling(max(lookback, 80), min_periods=60).quantile(
        float(risk_off_vol_percentile)
    ).shift(1)
    extreme_vol = (vol > vol_cut).shift(1).fillna(False).astype(bool)

    position = pd.Series(0.0, index=df.index, name="Position")
    for name in candidates:
        mask = chosen.eq(name) & (best_score >= float(min_score)) & ~extreme_vol
        position.loc[mask] = pos_frame.loc[mask, name]

    if market == "spot":
        position = position.clip(0, 1)
    # IMPORTANT: cap exposure according to the UI setting.
    position = position * (float(position_size_pct) / 100.0)

    # Changes in exposure, including switching to/from flat, incur costs.
    changes = position.diff().abs().fillna(position.abs())
    costs = changes * (float(fee) + float(slippage))
    net = (position * ret - costs).clip(lower=-0.999)
    equity = float(initial_equity) * (1.0 + net).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1.0) * 100.0

    seconds = df.index.to_series().diff().dt.total_seconds().median()
    ppy = 365.25 * 24 * 3600 / seconds if pd.notna(seconds) and seconds > 0 else 365.25
    std_net = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    sharpe = float(net.mean() / std_net * np.sqrt(ppy)) if std_net > 0 else 0.0
    years = max(len(net) / ppy, 1.0 / ppy)
    end_equity = float(equity.iloc[-1])
    cagr = ((end_equity / float(initial_equity)) ** (1.0 / years) - 1.0) * 100 if end_equity > 0 else -100.0

    metrics = {
        "Начальный капитал": round(float(initial_equity), 2),
        "Конечный капитал": round(end_equity, 2),
        "Доходность, %": round((end_equity / float(initial_equity) - 1.0) * 100, 3),
        "CAGR, %": round(cagr, 3),
        "Макс. просадка, %": round(float(drawdown.min()), 3),
        "Sharpe": round(sharpe, 3),
        "Время в рынке, %": round(float((position.abs() > 0).mean() * 100), 2),
        "Средняя экспозиция, x": round(float(position.abs().mean()), 3),
        "Доля времени без сигнала / risk-off, %": round(float((position == 0).mean() * 100), 2),
        "Смен выбора кандидата": int((chosen != chosen.shift(1)).sum() - 1),
    }

    regimes = pd.Series("Sideways / unclear", index=df.index, name="Режим")
    ema_fast = df["close"].ewm(span=20, adjust=False).mean()
    ema_slow = df["close"].ewm(span=100, adjust=False).mean()
    trend = ema_fast / ema_slow - 1.0
    regimes.loc[trend > 0.015] = "Uptrend"
    regimes.loc[trend < -0.015] = "Downtrend"
    regimes.loc[extreme_vol] = "High volatility / risk-off"

    details = pd.DataFrame({
        "close": df["close"],
        "regime": regimes,
        "selected_candidate": chosen,
        "selection_score": best_score,
        "position": position,
        "bar_return_net": net,
        "equity": equity,
        "drawdown_pct": drawdown,
        "realized_volatility": vol,
        "risk_off": extreme_vol,
    })

    # Comparable spot buy-and-hold reference: same configured exposure and entry cost.
    benchmark_net = ret * (float(position_size_pct) / 100.0)
    if len(benchmark_net):
        benchmark_net.iloc[0] -= (float(fee) + float(slippage)) * (float(position_size_pct) / 100.0)
    benchmark_net = benchmark_net.clip(lower=-0.999)
    benchmark_equity = float(initial_equity) * (1.0 + benchmark_net).cumprod()
    bh_return = (float(benchmark_equity.iloc[-1]) / float(initial_equity) - 1.0) * 100.0
    metrics["Buy & Hold доходность, %"] = round(bh_return, 3)
    metrics["Отставание / опережение Buy & Hold, п.п."] = round(
        metrics["Доходность, %"] - bh_return, 3
    )

    return {
        "metrics": metrics,
        "details": details,
        "equity": equity,
        "benchmark_equity": benchmark_equity.rename("Buy & Hold"),
        "candidate_counts": chosen.value_counts().rename_axis("Кандидат").reset_index(name="Свечей выбрано"),
        "verdict": _verdict(metrics),
    }


def _verdict(metrics):
    if (
        metrics["Доходность, %"] > 0
        and metrics["Макс. просадка, %"] > -15
        and metrics["Sharpe"] > 1
    ):
        return "Предварительный сигнал: только независимая проверка и paper trading, не реальные ордера."
    if metrics["Доходность, %"] > 0:
        return "Результат неубедителен по сочетанию доходности, просадки и Sharpe. Продолжать только исследование."
    return "Преимущество на этой выборке не обнаружено. Не торговать по этой модели."

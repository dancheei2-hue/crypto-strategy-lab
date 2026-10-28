"""Causal adaptive regime engine for research and paper trading.

At each bar, ranks a small family of signal generators using only prior-bar
returns, then applies the selected signal to the current bar. A risk gate can
stay flat when trailing performance is weak or realized volatility is extreme.
This is not a prediction guarantee or an execution engine.
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
        raw = signal_series(df, name).shift(1).fillna(0).clip(-1, 1)
        if market == "spot":
            raw = raw.clip(0, 1)
        change = raw.diff().abs().fillna(raw.abs())
        returns = raw * bar_returns - change * (fee + slippage)
        positions[name] = raw
        pnl[name] = returns
    return pd.DataFrame(positions, index=df.index), pd.DataFrame(pnl, index=df.index)


def adaptive_regime_backtest(df: pd.DataFrame, initial_equity=1000.0, market="spot",
                             fee=0.001, slippage=0.0005, lookback=240,
                             min_score=0.0, risk_off_vol_percentile=0.98,
                             candidates=None):
    """Select candidates from trailing performance without looking ahead.

    `lookback` is in bars. Ranking at t uses candidate PnL only through t-1.
    The model is deliberately simple and inspectable; no future bars are used.
    """
    if not isinstance(df, pd.DataFrame) or len(df) < max(300, int(lookback) + 50):
        raise ValueError("Для адаптивного теста нужно минимум 300 свечей и желательно 1 000+.")
    if market not in {"spot", "linear"}:
        raise ValueError("Рынок должен быть spot или linear.")
    lookback = max(80, min(int(lookback), len(df) - 50))
    candidates = candidates or DEFAULT_CANDIDATES
    pos_frame, pnl_frame = _candidate_positions(df, market, fee, slippage, candidates)

    # Use only history strictly before current bar. Score balances average return
    # against variability; this is a selection heuristic, not a calibrated forecast.
    mean = pnl_frame.rolling(lookback, min_periods=max(60, lookback // 2)).mean().shift(1)
    std = pnl_frame.rolling(lookback, min_periods=max(60, lookback // 2)).std(ddof=1).shift(1)
    score = (mean / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    has_score = score.notna().any(axis=1)
    chosen = score.fillna(-np.inf).idxmax(axis=1).where(has_score, "NO_EDGE")
    best_score = score.max(axis=1)

    # Volatility risk gate is causal. Compare current trailing vol with its own
    # trailing empirical distribution, shifted so current return cannot leak in.
    ret = df["close"].pct_change().fillna(0)
    vol = ret.rolling(30, min_periods=20).std()
    vol_cut = vol.rolling(max(lookback, 80), min_periods=60).quantile(risk_off_vol_percentile).shift(1)
    # Shift the risk gate: position for bar t must not depend on bar t's return.
    extreme_vol = (vol > vol_cut).shift(1).astype("boolean").fillna(False).astype(bool)

    position = pd.Series(0.0, index=df.index, name="Position")
    for name in candidates:
        mask = chosen.eq(name) & (best_score >= float(min_score)) & ~extreme_vol
        position.loc[mask] = pos_frame.loc[mask, name]
    if market == "spot":
        position = position.clip(0, 1)

    # Account for switching between candidates: position changes incur costs.
    changes = position.diff().abs().fillna(position.abs())
    costs = changes * (fee + slippage)
    gross = position * ret
    net = (gross - costs).clip(lower=-0.999)
    equity = float(initial_equity) * (1 + net).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1) * 100
    std_net = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    seconds = df.index.to_series().diff().dt.total_seconds().median()
    periods_per_year = 365.25 * 24 * 3600 / seconds if pd.notna(seconds) and seconds > 0 else 365.25
    sharpe = float(net.mean() / std_net * np.sqrt(periods_per_year)) if std_net > 0 else 0.0
    years = max(len(net) / periods_per_year, 1 / periods_per_year)
    cagr = ((float(equity.iloc[-1]) / initial_equity) ** (1 / years) - 1) * 100 if equity.iloc[-1] > 0 else -100.0
    metrics = {
        "Начальный капитал": round(float(initial_equity), 2),
        "Конечный капитал": round(float(equity.iloc[-1]), 2),
        "Доходность, %": round((float(equity.iloc[-1]) / initial_equity - 1) * 100, 3),
        "CAGR, %": round(cagr, 3),
        "Макс. просадка, %": round(float(drawdown.min()), 3),
        "Sharpe": round(sharpe, 3),
        "Время в рынке, %": round(float((position.abs() > 0).mean() * 100), 2),
        "Доля времени без сигнала / risk-off, %": round(float((position == 0).mean() * 100), 2),
        "Смен выбора кандидата": int((chosen != chosen.shift(1)).sum() - 1),
    }
    regimes = pd.Series("Sideways / unclear", index=df.index, name="Режим")
    ema_fast = df["close"].ewm(span=20, adjust=False).mean()
    ema_slow = df["close"].ewm(span=100, adjust=False).mean()
    trend = ema_fast / ema_slow - 1
    regimes.loc[trend > 0.015] = "Uptrend"
    regimes.loc[trend < -0.015] = "Downtrend"
    regimes.loc[extreme_vol] = "High volatility / risk-off"
    details = pd.DataFrame({"close": df["close"], "regime": regimes, "selected_candidate": chosen,
                            "selection_score": best_score, "position": position,
                            "bar_return_net": net, "equity": equity, "drawdown_pct": drawdown,
                            "realized_volatility": vol, "risk_off": extreme_vol})
    benchmark_net = ret - (ret.ne(0).astype(float) * 0)  # gross buy-and-hold reference, no trading costs assumed
    benchmark_equity = float(initial_equity) * (1 + benchmark_net).cumprod()
    metrics["Buy & Hold доходность, %"] = round((float(benchmark_equity.iloc[-1]) / initial_equity - 1) * 100, 3)
    metrics["Отставание / опережение Buy & Hold, п.п."] = round(metrics["Доходность, %"] - metrics["Buy & Hold доходность, %"], 3)
    return {"metrics": metrics, "details": details, "equity": equity,
            "benchmark_equity": benchmark_equity.rename("Buy & Hold"),
            "candidate_counts": chosen.value_counts().rename_axis("Кандидат").reset_index(name="Свечей выбрано"),
            "verdict": _verdict(metrics)}


def _verdict(m):
    if m["Доходность, %"] > 0 and m["Макс. просадка, %"] > -15 and m["Sharpe"] > 1:
        return "Предварительно интересный результат. Нужны независимые новые данные и paper trading; это не допуск к реальным ордерам."
    if m["Доходность, %"] > 0:
        return "Результат недостаточно убедителен по сочетанию доходности, просадки и Sharpe. Продолжать только исследование."
    return "На этой выборке преимущество не обнаружено. Рекомендация: не торговать по этой модели."

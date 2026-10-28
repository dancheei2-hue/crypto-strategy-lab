"""Conservative long-only spot strategy; signals use only information available at prior close."""
from __future__ import annotations
import numpy as np
import pandas as pd


def conservative_backtest(df: pd.DataFrame, initial_equity: float = 1000.0,
                          fee: float = 0.001, slippage: float = 0.0005,
                          max_exposure: float = 0.50, target_vol: float = 0.15,
                          max_annual_vol: float = 1.00,
                          trend_period: int = 200, momentum_period: int = 28,
                          vol_period: int = 30) -> dict:
    """Daily long/cash model. No leverage, no shorting, no intrabar stop assumptions.

    Entry condition: close above trend EMA and positive momentum over momentum_period.
    Exposure is volatility-scaled from lagged realized volatility and capped.
    Exposure is zero when trend/momentum filters fail or estimated annual volatility
    exceeds max_annual_vol. All position changes pay fee + slippage.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Индекс данных должен быть DatetimeIndex")
    if len(df) < max(trend_period + 10, momentum_period + 10, vol_period + 10):
        raise ValueError(f"Нужно не менее {max(trend_period + 10, momentum_period + 10, vol_period + 10)} свечей")
    if not 0 < max_exposure <= 1:
        raise ValueError("max_exposure должен быть в диапазоне (0, 1]")
    close = df["close"].astype(float)
    if (close <= 0).any() or close.isna().any():
        raise ValueError("Цены закрытия должны быть положительными и без пропусков")

    ema = close.ewm(span=trend_period, adjust=False, min_periods=trend_period).mean()
    momentum = close.pct_change(momentum_period)
    returns = close.pct_change().fillna(0.0)
    # Use prior bars only for signal and volatility estimate; position applies next bar.
    vol = returns.rolling(vol_period, min_periods=vol_period).std(ddof=1) * np.sqrt(365)
    eligible = ((close > ema) & (momentum > 0)).astype(float)
    eligible = eligible.shift(1).fillna(0.0)
    lagged_vol = vol.shift(1)
    scale = (target_vol / lagged_vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    scale = scale.fillna(0.0).clip(lower=0.0, upper=max_exposure)
    risk_ok = (lagged_vol <= max_annual_vol).fillna(False).astype(float)
    exposure = eligible * scale * risk_ok
    exposure = exposure.clip(0.0, max_exposure)

    turnover = exposure.diff().abs().fillna(exposure.abs())
    costs = turnover * (fee + slippage)
    net_returns = exposure * returns - costs
    equity = initial_equity * (1.0 + net_returns).cumprod()
    benchmark = initial_equity * (1.0 + returns - returns.abs() * 0.0).cumprod()
    # Benchmark also pays a one-time entry cost, approximated conservatively at start.
    if len(benchmark):
        benchmark.iloc[0] *= (1.0 - fee - slippage)

    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    elapsed_days = max((df.index[-1] - df.index[0]).total_seconds() / 86400.0, 1.0)
    years = elapsed_days / 365.25
    cagr = ((float(equity.iloc[-1]) / initial_equity) ** (1 / years) - 1) * 100 if years > 0 and equity.iloc[-1] > 0 else -100.0
    daily_vol = float(net_returns.std(ddof=1)) if len(net_returns) > 1 else 0.0
    sharpe = float(net_returns.mean() / daily_vol * np.sqrt(365)) if daily_vol > 0 else 0.0
    downside = net_returns[net_returns < 0]
    downside_vol = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(net_returns.mean() / downside_vol * np.sqrt(365)) if downside_vol > 0 else 0.0
    changes = int((turnover > 1e-9).sum())
    metrics = {
        "Начальный капитал, USDT": round(float(initial_equity), 2),
        "Конечный капитал, USDT": round(float(equity.iloc[-1]), 2),
        "Доходность, %": round((float(equity.iloc[-1]) / initial_equity - 1) * 100, 2),
        "CAGR, %": round(cagr, 2),
        "Макс. просадка, %": round(float(drawdown.min() * 100), 2),
        "Sharpe (приближённый)": round(sharpe, 2),
        "Sortino (приближённый)": round(sortino, 2),
        "Время в рынке, %": round(float((exposure > 0).mean() * 100), 2),
        "Средняя экспозиция, %": round(float(exposure.mean() * 100), 2),
        "Изменений позиции": changes,
        "Buy & Hold доходность, %": round((float(benchmark.iloc[-1]) / initial_equity - 1) * 100, 2),
        "Buy & Hold макс. просадка, %": round(float((benchmark / benchmark.cummax() - 1).min() * 100), 2),
    }
    details = pd.DataFrame({
        "close": close, "ema_trend": ema, "momentum": momentum,
        "estimated_annual_vol": vol, "eligible_signal_lagged": eligible,
        "exposure": exposure, "turnover": turnover, "costs_return": costs,
        "strategy_return": net_returns, "equity": equity, "buy_hold_equity": benchmark,
        "drawdown_pct": drawdown * 100,
    }, index=df.index)
    if metrics["Доходность, %"] > 0 and metrics["Макс. просадка, %"] > -20 and metrics["Sharpe (приближённый)"] >= 0.7:
        verdict = "Предварительно прошла фильтры риска на этом периоде. Это не подтверждение будущей доходности — нужна независимая walk-forward-проверка и paper trading."
    else:
        verdict = "Стратегия не прошла консервативные фильтры на этом периоде. Не запускать реальную торговлю; сохранять деньги вне рынка предпочтительнее, чем принуждать алгоритм к сделкам."
    return {"metrics": metrics, "details": details, "verdict": verdict}

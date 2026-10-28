"""Automatic strategy discovery with expanding-window selection and untouched OOS folds.
This is a research aid, not a profit guarantee or a live trading system.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from backtest import STRATEGIES, run_backtest


def candidate_configs(market: str):
    configs = []
    # Compact, predeclared search space; not an unlimited parameter sweep.
    for fast, slow in [(8, 21), (12, 26), (20, 50), (30, 80)]:
        configs.append(("EMA crossover", {"ema_fast": fast, "ema_slow": slow}))
    for entry, exit_level in [(20, 50), (25, 55), (30, 60), (35, 65)]:
        configs.append(("RSI mean reversion", {"rsi_entry": entry, "rsi_exit": exit_level}))
    for entry, exit_ in [(15, 7), (20, 10), (30, 12), (40, 15)]:
        configs.append(("Donchian breakout", {"donchian_entry": entry, "donchian_exit": exit_}))
    for name in ["MACD trend", "Bollinger reversion", "ATR trend filter", "EMA + RSI confirmation"]:
        configs.append((name, {}))
    return configs


def _score(metrics: dict) -> float:
    """Prefer risk-adjusted returns, penalize drawdown and undersampled strategies."""
    trades = int(metrics.get("Сделок", 0) or 0)
    ret = float(metrics.get("Доходность, %", 0) or 0)
    dd = abs(float(metrics.get("Макс. просадка, %", 0) or 0))
    sharpe = float(metrics.get("Sharpe", 0) or 0)
    pf = metrics.get("Profit factor", 0)
    pf = 3.0 if pf == float("inf") else float(pf or 0)
    if trades < 8:
        sample_penalty = (8 - trades) * 0.8
    else:
        sample_penalty = 0.0
    # Do not let raw return dominate. Negative returns are strongly discouraged.
    return sharpe * 1.5 + np.clip(ret, -50, 50) * 0.035 + min(pf, 3) * 0.25 - dd * 0.10 - sample_penalty


def automatic_research(df: pd.DataFrame, initial_equity=1000.0, market="spot", fee=0.001,
                       slippage=0.0005, leverage=1.0, funding_8h=0.0001,
                       position_size_pct=50.0, folds=5, min_train_fraction=0.5):
    """At each fold, select a candidate on past data only, then evaluate on next unseen block."""
    if len(df) < 300:
        raise ValueError("Для автоматического исследования нужно минимум 300 свечей; желательно 1 000+.")
    folds = int(np.clip(folds, 3, 8))
    warm = max(200, int(len(df) * min_train_fraction))
    remaining = len(df) - warm
    if remaining < folds * 20:
        raise ValueError("Недостаточно данных для выбранного числа проверочных отрезков.")
    chunk = remaining // folds
    configs = candidate_configs(market)
    fold_rows, selected_rows, curves = [], [], {}
    all_oos_returns = []
    for fold in range(folds):
        start = warm + fold * chunk
        end = warm + (fold + 1) * chunk if fold < folds - 1 else len(df)
        train = df.iloc[:start]
        # Selection uses training data only. OOS segment is never used to rank candidates.
        ranked = []
        for strategy, params in configs:
            try:
                out = run_backtest(train, strategy, initial_equity, market, fee, slippage,
                                   leverage, funding_8h, params=params,
                                   position_size_pct=position_size_pct)
                metrics = out["metrics"]
                ranked.append((_score(metrics), strategy, params, metrics))
            except (ValueError, KeyError, ZeroDivisionError, FloatingPointError):
                continue
        if not ranked:
            raise ValueError(f"Не удалось оценить ни одной конфигурации на fold {fold+1}.")
        ranked.sort(key=lambda item: item[0], reverse=True)
        _, strategy, params, train_metrics = ranked[0]
        # Compute causally on all available bars so indicators have warm-up; only score OOS bars.
        prefix_out = run_backtest(df.iloc[:end], strategy, initial_equity, market, fee, slippage,
                                  leverage, funding_8h, params=params,
                                  position_size_pct=position_size_pct)
        segment_returns = prefix_out["returns"].iloc[start:end].copy()
        if segment_returns.empty:
            continue
        oos_equity = initial_equity * (1 + segment_returns).cumprod()
        dd = ((oos_equity / oos_equity.cummax()) - 1).min() * 100
        fold_return = (float(oos_equity.iloc[-1]) / initial_equity - 1) * 100
        seconds = df.index.to_series().diff().dt.total_seconds().median()
        ppy = 365.25 * 24 * 3600 / seconds if pd.notna(seconds) and seconds > 0 else 365.25
        std = float(segment_returns.std(ddof=1)) if len(segment_returns) > 1 else 0.0
        sharpe = float(segment_returns.mean() / std * np.sqrt(ppy)) if std > 0 else 0.0
        active_pct = float((prefix_out["position"].iloc[start:end].abs() > 0).mean() * 100)
        fold_rows.append({"Отрезок": fold + 1, "Начало OOS": str(df.index[start]), "Конец OOS": str(df.index[end-1]),
                          "Выбранная стратегия": strategy, "Параметры": str(params),
                          "Доходность OOS, %": round(fold_return, 3), "Просадка OOS, %": round(float(dd), 3),
                          "Sharpe OOS": round(sharpe, 3), "Время в рынке, %": round(active_pct, 2),
                          "Доходность train, %": train_metrics.get("Доходность, %", 0),
                          "Сделок train": train_metrics.get("Сделок", 0),
                          "Кандидатов оценено": len(ranked)})
        selected_rows.append({"Отрезок": fold + 1, "Стратегия": strategy, "Параметры": params,
                              "Train score": round(float(ranked[0][0]), 4)})
        curves[f"OOS fold {fold+1}"] = oos_equity.rename(f"OOS fold {fold+1}")
        all_oos_returns.append(segment_returns)
    table = pd.DataFrame(fold_rows)
    if table.empty:
        raise ValueError("Автоматическое исследование не сформировало OOS-результатов.")
    positive = float((table["Доходность OOS, %"] > 0).mean() * 100)
    median_ret = float(table["Доходность OOS, %"].median())
    median_dd = float(table["Просадка OOS, %"].median())
    stable_winners = table["Выбранная стратегия"].value_counts().rename_axis("Стратегия").reset_index(name="Выбрана раз")
    if positive >= 70 and median_ret > 0 and median_dd > -20:
        verdict = "Есть предварительный сигнал преимущества — требуется paper trading и повторная проверка на новых данных."
    elif positive >= 50 and median_ret > 0:
        verdict = "Результат неоднозначный: преимущества недостаточно для автоматического допуска к торговле."
    else:
        verdict = "Надёжное преимущество не обнаружено. Рекомендация системы: не торговать этой выборкой."
    return {"folds": table, "selected": pd.DataFrame(selected_rows), "curves": curves,
            "summary": {"Положительных OOS-отрезков, %": round(positive, 1),
                        "Медианная доходность OOS, %": round(median_ret, 3),
                        "Медианная просадка OOS, %": round(median_dd, 3),
                        "Всего свечей": len(df), "Кандидатов в поиске": len(configs), "Отрезков OOS": len(table)},
            "stable_winners": stable_winners, "verdict": verdict}

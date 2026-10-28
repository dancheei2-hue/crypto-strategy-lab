"""Conservative walk-forward strategy research. Allows NO TRADE; no profit guarantee."""
from __future__ import annotations
import numpy as np
import pandas as pd
from backtest import run_backtest


def candidate_configs(market: str):
    configs = []
    for fast, slow in [(8, 21), (12, 26), (20, 50), (30, 80)]:
        configs.append(("EMA crossover", {"ema_fast": fast, "ema_slow": slow}))
    for entry, exit_level in [(20, 50), (25, 55), (30, 60), (35, 65)]:
        configs.append(("RSI mean reversion", {"rsi_entry": entry, "rsi_exit": exit_level}))
    for entry, exit_level in [(15, 7), (20, 10), (30, 12), (40, 15)]:
        configs.append(("Donchian breakout", {"donchian_entry": entry, "donchian_exit": exit_level}))
    for name in ["MACD trend", "Bollinger reversion", "ATR trend filter", "EMA + RSI confirmation"]:
        configs.append((name, {}))
    return configs


def _num(metrics, key, default=0.0):
    try:
        value = float(metrics.get(key, default))
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _rank_score(metrics):
    ret = _num(metrics, "Доходность, %")
    dd = abs(_num(metrics, "Макс. просадка, %"))
    sharpe = _num(metrics, "Sharpe")
    trades = int(_num(metrics, "Сделок"))
    pf = _num(metrics, "Profit factor")
    if not np.isfinite(pf):
        pf = 3.0
    return sharpe * 1.5 + np.clip(ret, -30, 30) * 0.04 + min(pf, 3) * 0.2 - dd * 0.12 - max(0, 8-trades) * 0.75


def _periods_per_year(index):
    seconds = index.to_series().diff().dt.total_seconds().median()
    return 365.25 * 24 * 3600 / seconds if pd.notna(seconds) and seconds > 0 else 365.25


def automatic_research(df: pd.DataFrame, initial_equity=1000.0, market="spot", fee=0.001,
                       slippage=0.0005, leverage=1.0, funding_8h=0.0001,
                       position_size_pct=50.0, folds=5, min_train_fraction=0.5):
    """Choose on past data only; evaluate next unseen block with indicator warm-up preserved."""
    if not isinstance(df, pd.DataFrame) or len(df) < 300:
        raise ValueError("Для автоматического исследования нужно минимум 300 свечей; желательно 1 000+.")
    if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("Данные должны быть отсортированы по времени и не содержать дубликатов.")
    if market not in {"spot", "linear"}:
        raise ValueError("Рынок должен быть spot или linear.")
    if not 0 <= float(position_size_pct) <= 100:
        raise ValueError("Размер позиции должен быть от 0 до 100%.")

    folds = int(np.clip(folds, 3, 8))
    warm = max(200, int(len(df) * float(min_train_fraction)))
    remaining = len(df) - warm
    if remaining < folds * 20:
        raise ValueError("Недостаточно данных для выбранного числа проверочных отрезков.")
    chunk = remaining // folds
    configs = candidate_configs(market)
    rows, selected_rows, curves = [], [], {}
    ppy = _periods_per_year(df.index)

    for fold in range(folds):
        start = warm + fold * chunk
        end = warm + (fold + 1) * chunk if fold < folds - 1 else len(df)
        train = df.iloc[:start]
        benchmark = run_backtest(train, "Buy & Hold benchmark", initial_equity, market, fee,
                                 slippage, leverage, funding_8h,
                                 position_size_pct=position_size_pct, allow_short=False)
        benchmark_return = _num(benchmark["metrics"], "Доходность, %")
        ranked = []
        for strategy, params in configs:
            try:
                result = run_backtest(train, strategy, initial_equity, market, fee, slippage,
                                      leverage, funding_8h, params=params,
                                      position_size_pct=position_size_pct,
                                      allow_short=(market != "spot"))
                m = result["metrics"]
                ret = _num(m, "Доходность, %")
                dd = _num(m, "Макс. просадка, %")
                sharpe = _num(m, "Sharpe")
                trades = int(_num(m, "Сделок"))
                # A candidate must earn its place; do not force a winner.
                eligible = ret > 0 and sharpe > 0 and dd > -30 and trades >= 5
                if market == "spot":
                    eligible = eligible and ret > benchmark_return
                if eligible:
                    ranked.append((_rank_score(m), strategy, params, m))
            except (ValueError, KeyError, ZeroDivisionError, FloatingPointError):
                continue

        ranked.sort(key=lambda x: x[0], reverse=True)
        if ranked:
            score, strategy, params, train_metrics = ranked[0]
            prefix = run_backtest(df.iloc[:end], strategy, initial_equity, market, fee, slippage,
                                  leverage, funding_8h, params=params,
                                  position_size_pct=position_size_pct,
                                  allow_short=(market != "spot"))
            segment = prefix["returns"].iloc[start:end].copy()
            active = prefix["position"].iloc[start:end].abs() > 0
            train_ret = _num(train_metrics, "Доходность, %")
            train_trades = int(_num(train_metrics, "Сделок"))
            score_out = round(float(score), 4)
            params_text = str(params)
        else:
            strategy, params = "NO TRADE", {}
            segment = pd.Series(0.0, index=df.index[start:end])
            active = pd.Series(False, index=segment.index)
            train_ret, train_trades, score_out, params_text = 0.0, 0, None, "{}"

        if segment.empty:
            continue
        equity = float(initial_equity) * (1 + segment).cumprod()
        dd = float((equity / equity.cummax() - 1).min() * 100)
        fold_return = (float(equity.iloc[-1]) / float(initial_equity) - 1) * 100
        std = float(segment.std(ddof=1)) if len(segment) > 1 else 0.0
        sharpe = float(segment.mean() / std * np.sqrt(ppy)) if std > 0 else 0.0
        rows.append({
            "Отрезок": fold + 1, "Начало OOS": str(df.index[start]), "Конец OOS": str(df.index[end-1]),
            "Выбранная стратегия": strategy, "Параметры": params_text,
            "Доходность OOS, %": round(fold_return, 3), "Просадка OOS, %": round(dd, 3),
            "Sharpe OOS": round(sharpe, 3), "Время в рынке, %": round(float(active.mean()*100), 2),
            "Доходность train, %": round(train_ret, 3), "Сделок train": train_trades,
            "Кандидатов, прошедших фильтр": len(ranked), "Buy & Hold train, %": round(benchmark_return, 3)
        })
        selected_rows.append({"Отрезок": fold+1, "Стратегия": strategy, "Параметры": params, "Train score": score_out})
        curves[f"OOS fold {fold+1}"] = equity.rename(f"OOS fold {fold+1}")

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("Автоматическое исследование не сформировало OOS-результатов.")
    positive = float((table["Доходность OOS, %"] > 0).mean() * 100)
    median_ret = float(table["Доходность OOS, %"].median())
    median_dd = float(table["Просадка OOS, %"].median())
    no_trade_pct = float((table["Выбранная стратегия"] == "NO TRADE").mean() * 100)
    if no_trade_pct == 100:
        verdict = "Ни один кандидат не прошёл фильтр на обучающих данных. Рекомендация: не торговать."
    elif positive >= 80 and median_ret > 0 and median_dd > -15 and no_trade_pct == 0:
        verdict = "Есть предварительный сигнал преимущества. Нужны paper trading и новые данные; прибыль не гарантируется."
    elif positive >= 60 and median_ret > 0:
        verdict = "Результат неоднозначный. Преимущество недостаточно подтверждено — только paper trading."
    else:
        verdict = "Надёжное преимущество не подтверждено на отложенных данных. Не включать реальную торговлю."
    stable = table["Выбранная стратегия"].value_counts().rename_axis("Стратегия").reset_index(name="Выбрана раз")
    return {
        "folds": table, "selected": pd.DataFrame(selected_rows), "curves": curves,
        "summary": {"Положительных OOS-отрезков, %": round(positive, 1),
                    "Медианная доходность OOS, %": round(median_ret, 3),
                    "Медианная просадка OOS, %": round(median_dd, 3),
                    "Всего свечей": len(df), "Кандидатов в поиске": len(configs),
                    "Отрезков OOS": len(table), "Отрезков без торговли, %": round(no_trade_pct, 1)},
        "stable_winners": stable, "verdict": verdict
    }

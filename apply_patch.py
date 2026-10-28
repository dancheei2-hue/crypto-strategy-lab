from pathlib import Path
import shutil
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
regime_path = ROOT / "regime_engine.py"
app_path = ROOT / "app.py"

def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: ожидался 1 фрагмент, найдено {count}. Файл не будет изменён.")
    return text.replace(old, new, 1)

def backup(path):
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))

if not regime_path.exists() or not app_path.exists():
    raise SystemExit("Не найдены regime_engine.py и app.py. Запусти скрипт из корня crypto-strategy-lab.")

regime = regime_path.read_text(encoding="utf-8")
app = app_path.read_text(encoding="utf-8")

# 1. Add position sizing to adaptive model; preserve backward compatibility.
regime = replace_once(
    regime,
    '                             min_score=0.0, risk_off_vol_percentile=0.98,\n'
    '                             candidates=None):',
    '                             min_score=0.0, risk_off_vol_percentile=0.98,\n'
    '                             candidates=None, position_size_pct=100.0):',
    "adaptive signature",
)
regime = replace_once(
    regime,
    '    if market == "spot":\n'
    '        position = position.clip(0, 1)\n'
    '\n'
    '    # Account for switching between candidates: position changes incur costs.\n',
    '    if market == "spot":\n'
    '        position = position.clip(0, 1)\n'
    '    # Apply the requested capital exposure before calculating costs and returns.\n'
    '    size_pct = max(0.0, min(float(position_size_pct), 100.0)) / 100.0\n'
    '    position = position * size_pct\n'
    '\n'
    '    # Account for switching between candidates: position changes incur costs.\n',
    "adaptive position sizing",
)
# Make benchmark explicit: long spot exposure, with entry cost, instead of a free gross benchmark.
regime = replace_once(
    regime,
    '    benchmark_net = ret - (ret.ne(0).astype(float) * 0)  # gross buy-and-hold reference, no trading costs assumed\n'
    '    benchmark_equity = float(initial_equity) * (1 + benchmark_net).cumprod()\n',
    '    # Buy-and-hold benchmark includes one entry fee/slippage charge; no exit charge.\n'
    '    benchmark_net = ret.copy()\n'
    '    if len(benchmark_net):\n'
    '        benchmark_net.iloc[0] = benchmark_net.iloc[0] - float(fee) - float(slippage)\n'
    '    benchmark_equity = float(initial_equity) * (1 + benchmark_net.clip(lower=-0.999)).cumprod()\n',
    "benchmark entry cost",
)

# 2. Pass the configured position size from the UI into adaptive backtest.
app = replace_once(
    app,
    '                    lookback=lookback, min_score=min_score, risk_off_vol_percentile=vol_percentile)\n',
    '                    lookback=lookback, min_score=min_score, risk_off_vol_percentile=vol_percentile,\n'
    '                    position_size_pct=ctx["position_size"])\n',
    "adaptive UI position size",
)

# 3. Fix manual holdout: compute indicators on the full prefix, then score only the holdout.
old = (
'                    split = int(len(data) * (1 - ctx["holdout_pct"] / 100))\n'
'                    test = data.iloc[split:]\n'
'                    oos = run_backtest(test, strategy, ctx["initial"], ctx["market"], ctx["fee"],\n'
'                                       ctx["slippage"], 1.0, 0.0, params=params,\n'
'                                       position_size_pct=ctx["position_size"])\n'
)
new = (
'                    split = int(len(data) * (1 - ctx["holdout_pct"] / 100))\n'
'                    # Compute indicators on the full series to retain warm-up history;\n'
'                    # only the untouched final segment contributes to OOS performance.\n'
'                    oos_returns = full["returns"].iloc[split:].copy()\n'
'                    oos_equity = ctx["initial"] * (1 + oos_returns).cumprod()\n'
'                    oos_drawdown = (oos_equity / oos_equity.cummax() - 1) * 100\n'
'                    seconds = data.index.to_series().diff().dt.total_seconds().median()\n'
'                    periods_per_year = 365.25 * 24 * 3600 / seconds if pd.notna(seconds) and seconds > 0 else 365.25\n'
'                    std_oos = float(oos_returns.std(ddof=1)) if len(oos_returns) > 1 else 0.0\n'
'                    sharpe_oos = float(oos_returns.mean() / std_oos * periods_per_year ** 0.5) if std_oos > 0 else 0.0\n'
'                    # Count closed trades in the holdout; an open position is not a closed trade.\n'
'                    oos_pos = full["position"].iloc[split:]\n'
'                    closed_oos = int(((oos_pos.shift(1).fillna(0) != 0) & (oos_pos == 0)).sum())\n'
'                    oos = {"returns": oos_returns, "equity": oos_equity,\n'
'                           "trades": full["trades"],\n'
'                           "metrics": {"Начальный капитал": round(float(ctx["initial"]), 4),\n'
'                                       "Конечный капитал": round(float(oos_equity.iloc[-1]), 4) if len(oos_equity) else round(float(ctx["initial"]), 4),\n'
'                                       "Доходность, %": round((float(oos_equity.iloc[-1]) / float(ctx["initial"]) - 1) * 100, 3) if len(oos_equity) else 0.0,\n'
'                                       "Макс. просадка, %": round(float(oos_drawdown.min()), 3) if len(oos_drawdown) else 0.0,\n'
'                                       "Sharpe": round(sharpe_oos, 3), "Сделок": closed_oos}}\n'
'                    test = data.iloc[split:]\n'
)
app = replace_once(app, old, new, "manual holdout indicator warm-up")

backup(regime_path)
backup(app_path)
regime_path.write_text(regime, encoding="utf-8")
app_path.write_text(app, encoding="utf-8")
print("Исправления применены:")
print("- адаптивная модель учитывает размер позиции из интерфейса;")
print("- benchmark учитывает стоимость входа;")
print("- OOS-тест использует историю до начала holdout для разогрева индикаторов;")
print("Созданы резервные копии *.bak. Перед запуском проверь git diff и тесты.")

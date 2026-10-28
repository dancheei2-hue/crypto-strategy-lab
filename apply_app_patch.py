from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
app_path = root / "app.py"
regime_path = root / "regime_engine.py"
if not app_path.exists() or not regime_path.exists():
    raise SystemExit("Запусти скрипт в папке репозитория: не найдены app.py и regime_engine.py")

app = app_path.read_text(encoding="utf-8")
old_call = '                    lookback=lookback, min_score=min_score, risk_off_vol_percentile=vol_percentile)'
new_call = ('                    lookback=lookback, min_score=min_score, risk_off_vol_percentile=vol_percentile,\n'
            '                    position_size_pct=ctx["position_size"])')
if old_call in app:
    app_new = app.replace(old_call, new_call, 1)
elif new_call in app:
    app_new = app
else:
    raise SystemExit("Не найден ожидаемый вызов adaptive_regime_backtest в app.py. Файл не изменён.")

shutil.copy2(app_path, app_path.with_suffix(".py.bak"))
shutil.copy2(regime_path, regime_path.with_suffix(".py.bak"))
app_path.write_text(app_new, encoding="utf-8")
print("app.py обновлён; резервные копии созданы. Замените regime_engine.py файлом из архива.")

import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from backtest import fetch_ohlcv, run_backtest, STRATEGIES

st.set_page_config(page_title="Crypto Strategy Lab", page_icon="📈", layout="wide")
st.title("Crypto Strategy Lab")
st.caption("Лаборатория криптостратегий: котировки, бэктест, сравнение стратегий, риск и out-of-sample. Исторический результат не гарантирует будущую прибыль.")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.header("1. Рынок и данные")
    exchange_id = st.selectbox("Биржа котировок", ["binance", "bybit", "bingx"], format_func=lambda x: {"binance": "Binance", "bybit": "Bybit", "bingx": "BingX"}[x])
    market = st.selectbox("Рынок", ["spot", "linear"], format_func=lambda x: "Спот" if x == "spot" else "USDT perpetual / фьючерсы")
    symbol = st.text_input("Торговая пара", "BTC/USDT", help="Примеры: BTC/USDT, ETH/USDT, SOL/USDT. Доступность зависит от выбранной биржи и рынка.")
    timeframe = st.selectbox("Таймфрейм", ["5m", "15m", "30m", "1h", "2h", "4h", "1d"], index=3)
    months = st.select_slider("Глубина истории", options=[1, 3, 6, 12, 18, 24], value=12, format_func=lambda x: f"{x} мес.")
    st.divider()
    st.header("2. Модель исполнения")
    initial = st.number_input("Стартовый капитал, USDT", min_value=10.0, max_value=1_000_000.0, value=1000.0, step=100.0)
    fee_pct = st.number_input("Комиссия за сторону, %", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
    slippage_pct = st.number_input("Проскальзывание за сторону, %", min_value=0.0, max_value=2.0, value=0.05, step=0.01)
    position_size = st.slider("Размер позиции, % капитала", 1, 100, 100)
    leverage = st.slider("Плечо (только модель фьючерсов)", 1.0, 5.0, 1.0, 0.5, disabled=(market == "spot"))
    funding_pct_8h = st.number_input("Допущение funding за 8 часов, %", min_value=0.0, max_value=1.0, value=0.01, step=0.01, disabled=(market == "spot"))
    holdout_pct = st.slider("Out-of-sample часть истории, %", 10, 50, 25)
    st.caption("Funding задан пользователем и не является фактической историей ставок. Ликвидации, биржевые минимумы и точный размер лота не моделируются.")

st.header("3. Выбор стратегий")
selected = st.multiselect("Стратегии для сравнения", list(STRATEGIES.keys()), default=["EMA crossover", "RSI mean reversion", "Donchian breakout", "Buy & Hold benchmark"])
with st.expander("Параметры стратегий", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        ema_fast = st.number_input("EMA fast", 2, 100, 12)
        ema_slow = st.number_input("EMA slow", 3, 300, 26)
        rsi_entry = st.number_input("RSI вход ниже", 5, 45, 30)
    with c2:
        rsi_exit = st.number_input("RSI выход выше", 45, 90, 55)
        donchian_entry = st.number_input("Donchian вход, бары", 5, 100, 20)
        donchian_exit = st.number_input("Donchian выход, бары", 2, 50, 10)
    with c3:
        st.write("Используются индикаторы EMA, RSI, MACD, Bollinger Bands и Donchian. Их настройки для расширенных стратегий пока фиксированы.")
        st.info("Для честного сравнения избегайте подгонки параметров под весь период истории.")
params = {"ema_fast": int(ema_fast), "ema_slow": int(ema_slow), "rsi_entry": float(rsi_entry), "rsi_exit": float(rsi_exit),
          "donchian_entry": int(donchian_entry), "donchian_exit": int(donchian_exit)}

run = st.button("Загрузить котировки и выполнить анализ", type="primary", use_container_width=True)
if run:
    if not selected:
        st.error("Выберите хотя бы одну стратегию.")
    elif ema_fast >= ema_slow:
        st.error("EMA fast должна быть меньше EMA slow.")
    else:
        try:
            with st.spinner(f"Загружаем публичные свечи {exchange_id.capitalize()}…"):
                data = fetch_ohlcv(symbol, timeframe, months, market, exchange_id)
            if len(data) < 200:
                st.error(f"Получено только {len(data)} свечей. Для анализа нужно минимум 200; увеличьте историю или выберите другой таймфрейм/пару.")
                st.stop()
            split = int(len(data) * (1 - holdout_pct / 100))
            train = data.iloc[:split].copy()
            test = data.iloc[split:].copy()
            rows, curves, trades_out, detail = [], {}, {}, {}
            full_out = {}
            for name in selected:
                # Full-sample display is diagnostic only; OOS is the holdout segment.
                full = run_backtest(data, name, initial, market, fee_pct / 100, slippage_pct / 100, leverage,
                                    funding_pct_8h / 100, params=params, position_size_pct=position_size)
                tr = run_backtest(train, name, initial, market, fee_pct / 100, slippage_pct / 100, leverage,
                                  funding_pct_8h / 100, params=params, position_size_pct=position_size)
                # Keep full history for indicators then evaluate only holdout via slicing the full result.
                # The simple isolated OOS run below remains visible as a conservative basic holdout estimate.
                os = run_backtest(test, name, initial, market, fee_pct / 100, slippage_pct / 100, leverage,
                                  funding_pct_8h / 100, params=params, position_size_pct=position_size)
                rows.extend([{"Стратегия": name, "Период": "Вся история (диагностика)", **full["metrics"]},
                             {"Стратегия": name, "Период": "Train", **tr["metrics"]},
                             {"Стратегия": name, "Период": "OOS (отдельный сегмент)", **os["metrics"]}])
                curves[f"{name} — весь период"] = full["equity"].rename(f"{name} — весь период")
                curves[f"{name} — OOS"] = os["equity"].rename(f"{name} — OOS")
                trades_out[f"{name} — весь период"] = full["trades"]
                trades_out[f"{name} — OOS"] = os["trades"]
                detail[name] = full
                full_out[name] = full
            summary = pd.DataFrame(rows)
            # Use Streamlit state so the report remains available after changing controls.
            st.session_state.last_result = {"summary": summary, "curves": curves, "trades": trades_out, "detail": detail,
                                            "data": data, "train": train, "test": test, "exchange": exchange_id,
                                            "market": market, "symbol": symbol, "timeframe": timeframe,
                                            "months": months, "params": params, "initial": initial,
                                            "settings": {"fee_pct": fee_pct, "slippage_pct": slippage_pct,
                                                          "leverage": leverage, "position_size": position_size,
                                                          "funding_pct_8h": funding_pct_8h}}
        except Exception as e:
            st.error(f"Не удалось выполнить анализ: {e}")
            st.info("Проверьте пару и рынок. Для проверки используйте BTC/USDT, Spot, 1h и 3–6 месяцев. Ограничения доступа к API могут зависеть от региона сервера.")

res = st.session_state.last_result
if res:
    st.success(f"Загружено {len(res['data']):,} свечей: {res['data'].index.min()} — {res['data'].index.max()} (UTC)")
    st.caption(f"Источник: публичные OHLCV • {res['exchange'].capitalize()} • {res['symbol']} • {res['timeframe']} • {res['market']}. Незакрытая последняя свеча исключается.")
    summary = res["summary"]
    st.subheader("Сводное сравнение")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    full_rows = summary[summary["Период"] == "Вся история (диагностика)"].copy()
    if not full_rows.empty:
        st.subheader("Ключевые показатели по всей истории")
        cols = st.columns(min(4, len(full_rows)))
        for i, (_, row) in enumerate(full_rows.iterrows()):
            with cols[i % len(cols)]:
                st.metric(str(row["Стратегия"]), f"{row['Доходность, %']:.2f}%", f"DD {row['Макс. просадка, %']:.2f}%")
    st.subheader("Кривая капитала")
    chart_df = pd.concat(list(res["curves"].values()), axis=1)
    st.line_chart(chart_df, use_container_width=True, height=380)
    st.subheader("Цена и границы выборки")
    st.line_chart(res["data"]["close"].rename("Close"), use_container_width=True, height=220)
    st.caption(f"Train: до {res['train'].index.max()} UTC. OOS: с {res['test'].index.min()} UTC. Отдельный OOS-запуск в этой версии перезапускает индикаторы на начале OOS, поэтому считайте его предварительной оценкой, не walk-forward-валидацией.")
    st.subheader("Сделки")
    which = st.selectbox("Стратегия и период", list(res["trades"].keys()))
    trades = res["trades"][which]
    if trades.empty:
        st.info("Сделок на выбранном участке нет.")
    else:
        st.dataframe(trades, use_container_width=True, hide_index=True)
    st.subheader("Скачать отчёты")
    st.download_button("Сводная таблица CSV", summary.to_csv(index=False).encode("utf-8-sig"), "strategy_metrics.csv", "text/csv")
    all_trades = []
    for label, frame in res["trades"].items():
        cp = frame.copy(); cp.insert(0, "Набор", label); all_trades.append(cp)
    trade_csv = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    st.download_button("Все сделки CSV", trade_csv.to_csv(index=False).encode("utf-8-sig"), "strategy_trades.csv", "text/csv")
    csv_buf = io.StringIO()
    res["data"].to_csv(csv_buf)
    st.download_button("Исходные свечи CSV", csv_buf.getvalue().encode("utf-8-sig"), "ohlcv.csv", "text/csv")
    st.download_button("Настройки эксперимента JSON", json.dumps({"exchange": res["exchange"], "market": res["market"], "symbol": res["symbol"], "timeframe": res["timeframe"], "months": res["months"], "params": res["params"], "settings": res["settings"], "selected_strategies": list(res["detail"].keys()), "generated_at_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2).encode("utf-8"), "experiment_config.json", "application/json")

st.divider()
st.header("Paper trading и автоматизация")
st.write("Этот интерфейс пока исследовательский: он не отправляет ордера, не держит постоянный процесс между перезапусками Streamlit и не хранит API-ключи. Следующий безопасный этап — симуляция paper trading на свежих закрытых свечах с журналом сигналов и позиций. Реальную автоторговлю следует добавлять отдельно с лимитами риска, защитой ключей, журналом действий и аварийной остановкой.")
st.warning("Результаты — модель, а не финансовая рекомендация. В текущей версии не моделируются ликвидации, минимальные размеры ордеров, точные комиссии биржи и реальная историческая ставка funding.")

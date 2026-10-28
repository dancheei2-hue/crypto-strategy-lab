import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from backtest import fetch_ohlcv, run_backtest, signal_series, STRATEGIES
from auto_research import automatic_research
from regime_engine import adaptive_regime_backtest
from paper import get_account, reset_account, get_events, process_closed_candle

st.set_page_config(page_title="Crypto Strategy Lab", page_icon="📈", layout="wide")

st.title("Crypto Strategy Lab")
st.caption("Понятная проверка торговых идей. Сначала исследование и симуляция — без реальных ордеров.")

# Keep the last loaded dataset and results available across Streamlit reruns.
if "csl_data" not in st.session_state:
    st.session_state.csl_data = None
if "csl_result" not in st.session_state:
    st.session_state.csl_result = None
if "csl_auto" not in st.session_state:
    st.session_state.csl_auto = None
if "csl_adaptive" not in st.session_state:
    st.session_state.csl_adaptive = None

with st.sidebar:
    st.header("Рынок")
    exchange = st.selectbox("Биржа котировок", ["binance", "bybit", "bingx"],
                            format_func=lambda x: {"binance": "Binance", "bybit": "Bybit", "bingx": "BingX"}[x])
    market_label = st.radio("Тип рынка", ["Спот", "Фьючерсы (только тест)"], index=0)
    market = "spot" if market_label == "Спот" else "linear"
    symbol = st.text_input("Торговая пара", "BTC/USDT").strip().upper()
    timeframe = st.selectbox("Таймфрейм", ["15m", "30m", "1h", "2h", "4h", "1d"], index=5)
    months = st.select_slider("История", options=[3, 6, 12, 18, 24], value=12,
                              format_func=lambda x: f"{x} мес.")

    with st.expander("Издержки и капитал", expanded=False):
        initial = st.number_input("Начальный капитал, USDT", min_value=10.0, max_value=1000000.0,
                                  value=1000.0, step=100.0)
        fee_pct = st.number_input("Комиссия за сторону, %", min_value=0.0, max_value=1.0,
                                  value=0.10, step=0.01)
        slippage_pct = st.number_input("Проскальзывание за сторону, %", min_value=0.0, max_value=2.0,
                                       value=0.05, step=0.01)
        position_size = st.slider("Размер позиции, % капитала", 5, 100, 50, step=5)
        holdout_pct = st.slider("История для финальной проверки, %", 10, 40, 25)
    load_clicked = st.button("Загрузить рынок", type="primary", use_container_width=True)
    st.caption("По умолчанию — спот без плеча. Доступность бирж зависит от сети и региона.")

if load_clicked:
    try:
        with st.spinner("Загружаем свечи и проверяем данные…"):
            data = fetch_ohlcv(symbol, timeframe, months, market, exchange)
        if data is None or len(data) < 200:
            st.error(f"Получено только {0 if data is None else len(data)} свечей. Нужно минимум 200. Попробуйте увеличить историю или выбрать другой таймфрейм.")
        else:
            st.session_state.csl_data = data
            st.session_state.csl_result = None
            st.session_state.csl_auto = None
            st.session_state.csl_adaptive = None
            st.session_state.csl_context = {
                "exchange": exchange, "market": market, "symbol": symbol,
                "timeframe": timeframe, "months": months, "initial": initial,
                "fee": fee_pct / 100, "slippage": slippage_pct / 100,
                "position_size": position_size, "holdout_pct": holdout_pct,
            }
            st.success(f"Загружено {len(data):,} свечей")
    except Exception as exc:
        st.error(f"Не удалось загрузить котировки: {exc}")
        st.info("Для первой проверки используйте Binance, Спот, BTC/USDT, 1d и 6–12 месяцев.")

data = st.session_state.csl_data
ctx = st.session_state.get("csl_context")

if data is None or ctx is None:
    st.subheader("Начните с проверки рынка")
    st.write("1. Слева выберите биржу и пару. 2. Нажмите **«Загрузить рынок»**. 3. Запустите проверку или автоматический поиск.")
    st.info("Если стратегия не показывает преимущество после комиссий, приложение должно рекомендовать не торговать — это нормальный результат исследования.")
    st.stop()

st.success(f"{ctx['symbol']} · {ctx['timeframe']} · {ctx['exchange'].capitalize()} · {len(data):,} свечей")
st.caption(f"Период: {data.index.min()} — {data.index.max()} UTC. Используются публичные котировки; последняя незакрытая свеча исключается.")

home, auto_tab, adaptive_tab, paper_tab, data_tab = st.tabs([
    "Обзор", "Найти стратегию", "Адаптивная модель", "Paper trading", "Данные и отчёты"
])

with home:
    st.subheader("Проверить торговую гипотезу")
    st.write("Выберите один подход для контрольного теста. Для автоматического выбора откройте вкладку **«Найти стратегию»**.")
    preferred = ["EMA + RSI confirmation", "ATR trend filter", "MACD trend", "Buy & Hold benchmark"]
    default_strategy = "EMA + RSI confirmation" if "EMA + RSI confirmation" in STRATEGIES else list(STRATEGIES)[0]
    strategy = st.selectbox("Модель для проверки", list(STRATEGIES.keys()),
                            index=list(STRATEGIES.keys()).index(default_strategy))
    with st.expander("Дополнительные настройки стратегии", expanded=False):
        ema_fast = st.number_input("Быстрая EMA", min_value=2, max_value=100, value=12)
        ema_slow = st.number_input("Медленная EMA", min_value=3, max_value=300, value=26)
        rsi_entry = st.number_input("RSI: вход ниже", min_value=5, max_value=45, value=30)
        rsi_exit = st.number_input("RSI: выход выше", min_value=45, max_value=90, value=55)
        donchian_entry = st.number_input("Donchian: вход, свечей", min_value=5, max_value=100, value=20)
        donchian_exit = st.number_input("Donchian: выход, свечей", min_value=2, max_value=50, value=10)
    params = {"ema_fast": int(ema_fast), "ema_slow": int(ema_slow), "rsi_entry": float(rsi_entry),
              "rsi_exit": float(rsi_exit), "donchian_entry": int(donchian_entry),
              "donchian_exit": int(donchian_exit)}
    if st.button("Проверить стратегию", type="primary", key="run_one"):
        if ema_fast >= ema_slow:
            st.error("Быстрая EMA должна быть меньше медленной.")
        else:
            try:
                with st.spinner("Проводим бэктест с учётом издержек…"):
                    full = run_backtest(data, strategy, ctx["initial"], ctx["market"], ctx["fee"],
                                        ctx["slippage"], 1.0, 0.0, params=params,
                                        position_size_pct=ctx["position_size"])
                    split = int(len(data) * (1 - ctx["holdout_pct"] / 100))
                    test = data.iloc[split:]
                    oos = run_backtest(test, strategy, ctx["initial"], ctx["market"], ctx["fee"],
                                       ctx["slippage"], 1.0, 0.0, params=params,
                                       position_size_pct=ctx["position_size"])
                st.session_state.csl_result = {"strategy": strategy, "full": full, "oos": oos,
                                                "params": params, "test_start": str(test.index.min())}
            except Exception as exc:
                st.error(f"Бэктест не выполнен: {exc}")
    result = st.session_state.csl_result
    if result:
        m = result["full"]["metrics"]
        o = result["oos"]["metrics"]
        st.divider()
        st.subheader("Результат")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Доходность", f"{m.get('Доходность, %', 0):.2f}%")
        c2.metric("Макс. просадка", f"{m.get('Макс. просадка, %', 0):.2f}%")
        c3.metric("Sharpe", f"{m.get('Sharpe', 0):.2f}")
        c4.metric("Сделок", str(m.get("Сделок", 0)))
        o1, o2, o3 = st.columns(3)
        o1.metric("Финальная проверка", f"{o.get('Доходность, %', 0):.2f}%")
        o2.metric("Просадка проверки", f"{o.get('Макс. просадка, %', 0):.2f}%")
        o3.metric("Сделок на проверке", str(o.get("Сделок", 0)))
        if o.get("Доходность, %", 0) <= 0:
            st.warning("На отдельном проверочном отрезке прибыльность не подтверждена. Не включайте эту модель в реальную торговлю.")
        elif o.get("Макс. просадка, %", 0) < -15:
            st.warning("Проверочный отрезок положительный, но просадка высокая для консервативного подхода.")
        else:
            st.info("Результат выглядит достойным дальнейшей проверки, но не доказывает будущую прибыльность. Нужен paper trading.")
        chart = pd.concat([result["full"]["equity"].rename("Вся история"),
                           result["oos"]["equity"].rename("Отдельный OOS-сегмент")], axis=1)
        st.line_chart(chart, use_container_width=True)
        with st.expander("Подробные метрики и сделки", expanded=False):
            st.dataframe(pd.DataFrame([{"Период": "Вся история", **m}, {"Период": "Проверочный сегмент", **o}]),
                         use_container_width=True, hide_index=True)
            trades = result["full"]["trades"]
            if not trades.empty:
                st.dataframe(trades, use_container_width=True, hide_index=True)
                st.download_button("Скачать сделки CSV", trades.to_csv(index=False).encode("utf-8-sig"), "trades.csv")

with auto_tab:
    st.subheader("Пусть алгоритм проверит варианты за вас")
    st.write("Алгоритм выбирает кандидата только по прошлым данным, затем проверяет его на следующем временном отрезке. Он может прийти к выводу, что торгового преимущества нет.")
    auto_folds = st.slider("Количество проверочных отрезков", 3, 8, 5, key="simple_auto_folds")
    auto_size = st.slider("Размер позиции для теста, % капитала", 10, 75, min(50, ctx["position_size"]), step=5, key="simple_auto_size")
    if st.button("Найти стратегию автоматически", type="primary", key="simple_auto_run"):
        try:
            with st.spinner("Обучаем на прошлом и проверяем на следующих периодах…"):
                st.session_state.csl_auto = automatic_research(
                    data, ctx["initial"], ctx["market"], ctx["fee"], ctx["slippage"], 1.0, 0.0,
                    position_size_pct=auto_size, folds=auto_folds)
        except Exception as exc:
            st.error(f"Автоматический поиск не выполнен: {exc}")
    ar = st.session_state.csl_auto
    if ar:
        st.warning(ar["verdict"])
        s = ar["summary"]
        a, b, c = st.columns(3)
        a.metric("Положительных периодов", f"{s['Положительных OOS-отрезков, %']:.1f}%")
        b.metric("Медианная доходность", f"{s['Медианная доходность OOS, %']:.2f}%")
        c.metric("Медианная просадка", f"{s['Медианная просадка OOS, %']:.2f}%")
        st.dataframe(ar["folds"], use_container_width=True, hide_index=True)
        st.line_chart(pd.concat(list(ar["curves"].values()), axis=1), use_container_width=True)
        st.download_button("Скачать отчёт CSV", ar["folds"].to_csv(index=False).encode("utf-8-sig"), "auto_research.csv")

with adaptive_tab:
    st.subheader("Адаптивная модель")
    st.write("На каждой свече модель оценивает предыдущие результаты кандидатов и может оставаться вне рынка. Это исследовательская гипотеза, не гарантированно прибыльный бот.")
    with st.expander("Настройки адаптивной модели", expanded=False):
        lookback = st.slider("Окно оценки, свечей", 80, 1000, 240, step=40, key="simple_lookback")
        min_score = st.slider("Минимальная оценка сигнала", -0.10, 0.20, 0.0, step=0.01, key="simple_min_score")
        vol_percentile = st.slider("Порог отключения при волатильности", 0.80, 0.995, 0.98, step=0.005, key="simple_vol")
    if st.button("Проверить адаптивную модель", key="simple_adaptive_run"):
        try:
            with st.spinner("Проверяем модель на исторических данных…"):
                st.session_state.csl_adaptive = adaptive_regime_backtest(
                    data, ctx["initial"], ctx["market"], ctx["fee"], ctx["slippage"],
                    lookback=lookback, min_score=min_score, risk_off_vol_percentile=vol_percentile,
                    position_size_pct=ctx["position_size"])
        except Exception as exc:
            st.error(f"Проверка не выполнена: {exc}")
    ad = st.session_state.csl_adaptive
    if ad:
        st.warning(ad["verdict"])
        metrics = ad["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Доходность", f"{metrics.get('Доходность, %', 0):.2f}%")
        c2.metric("Макс. просадка", f"{metrics.get('Макс. просадка, %', 0):.2f}%")
        c3.metric("Sharpe", f"{metrics.get('Sharpe', 0):.2f}")
        st.line_chart(pd.concat([ad["equity"], ad["benchmark_equity"]], axis=1), use_container_width=True)
        with st.expander("Детали по свечам", expanded=False):
            st.dataframe(ad["details"].tail(500), use_container_width=True, hide_index=True)
            st.download_button("Скачать детали CSV", ad["details"].to_csv().encode("utf-8-sig"), "adaptive_details.csv")

with paper_tab:
    st.subheader("Paper trading — только виртуальные сделки")
    st.write("Симуляция работает по кнопке. Она не отправляет ордера на биржу и не является круглосуточным ботом.")
    paper_strategy = st.selectbox("Стратегия для виртуального счёта", list(STRATEGIES.keys()),
                                  index=list(STRATEGIES.keys()).index("EMA + RSI confirmation") if "EMA + RSI confirmation" in STRATEGIES else 0,
                                  key="simple_paper_strategy")
    cash = st.number_input("Начальный виртуальный баланс, USDT", min_value=10.0, max_value=1000000.0,
                           value=1000.0, step=100.0, key="simple_paper_cash")
    a, b = st.columns(2)
    with a:
        if st.button("Создать / сбросить счёт", key="simple_paper_reset"):
            reset_account(cash)
            st.success("Виртуальный счёт создан. Предыдущая история удалена.")
    with b:
        if st.button("Обработать последнюю закрытую свечу", type="primary", key="simple_paper_process"):
            if ctx["market"] != "spot":
                st.error("Paper trading доступен только для спота.")
            elif get_account() is None:
                st.error("Сначала создайте виртуальный счёт.")
            else:
                try:
                    sig = signal_series(data, paper_strategy, {})
                    outcome = process_closed_candle(data.index[-1], ctx["symbol"], paper_strategy,
                        float(data["close"].iloc[-1]), float(sig.iloc[-1]), ctx["fee"], ctx["slippage"])
                    st.info(outcome.get("message", outcome.get("status", "Готово")))
                except Exception as exc:
                    st.error(f"Не удалось обработать свечу: {exc}")
    account = get_account()
    if account:
        qty = float(account.get("quantity") or 0)
        cash_now = float(account.get("cash") or 0)
        equity_now = cash_now + qty * float(data["close"].iloc[-1])
        x, y, z = st.columns(3)
        x.metric("Доступный USDT", f"{cash_now:.2f}")
        y.metric("Количество актива", f"{qty:.6f}")
        z.metric("Виртуальный капитал", f"{equity_now:.2f} USDT")
    events = get_events(500)
    if not events.empty:
        with st.expander("Журнал виртуальных операций", expanded=True):
            st.dataframe(events, use_container_width=True, hide_index=True)
            st.download_button("Скачать журнал CSV", events.to_csv(index=False).encode("utf-8-sig"), "paper_journal.csv")

with data_tab:
    st.subheader("Данные и отчёты")
    st.write("Исходные данные и результаты можно скачать для независимой проверки.")
    st.download_button("Скачать свечи CSV", data.to_csv().encode("utf-8-sig"), "ohlcv.csv")
    config = {**ctx, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    st.download_button("Скачать настройки JSON", json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8"), "experiment_settings.json", "application/json")
    st.dataframe(data.tail(100), use_container_width=True)
    with st.expander("Что означают показатели", expanded=False):
        st.markdown("""
        - **Доходность** — изменение капитала за тестовый период после моделируемых издержек.
        - **Макс. просадка** — наибольшее падение капитала от предыдущего максимума.
        - **Sharpe** — доходность относительно разброса результатов; сам по себе не доказывает преимущество.
        - **OOS** — участок истории, который не использовался для настройки выбранной модели в этом простом тесте.
        - **Paper trading** — виртуальные сделки без реального исполнения.
        """)

st.divider()
st.caption("Результаты тестирования не гарантируют будущую прибыль. Не включайте реальные ордера без независимой проверки и длительного paper trading.")

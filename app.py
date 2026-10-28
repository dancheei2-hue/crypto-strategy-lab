import streamlit as st
import pandas as pd
import numpy as np
from backtest import fetch_ohlcv, run_backtest, STRATEGIES

st.set_page_config(page_title="Crypto Strategy Lab", layout="wide")
st.title("Crypto Strategy Lab")
st.caption("Публичные котировки бирж • исторический бэктест • результаты не гарантируют будущую прибыль")

with st.sidebar:
    st.header("Настройки")
    exchange_id = st.selectbox("Биржа котировок", ["bingx", "binance", "bybit"], format_func=lambda x: {"bingx": "BingX", "binance": "Binance", "bybit": "Bybit"}[x], index=0)
    market = st.selectbox("Рынок", ["spot", "linear"], format_func=lambda x: "Спот" if x == "spot" else "USDT-фьючерсы")
    symbol = st.text_input("Торговая пара", "BTC/USDT")
    timeframe = st.selectbox("Таймфрейм", ["5m", "15m", "30m", "1h", "2h", "4h", "1d"], index=3)
    months = st.selectbox("История", [3, 6, 12], index=2, format_func=lambda x: f"{x} месяцев")
    initial = st.number_input("Стартовый капитал (USDT)", min_value=10.0, value=100.0, step=10.0)
    fee_pct = st.number_input("Комиссия за сторону (%)", min_value=0.0, value=0.10, step=0.01)
    slippage_pct = st.number_input("Проскальзывание за сторону (%)", min_value=0.0, value=0.05, step=0.01)
    funding_pct_8h = st.number_input("Допущение funding за 8 ч. (%) для фьючерсов", min_value=0.0, value=0.01, step=0.01)
    leverage = st.number_input("Плечо (фьючерсы; только модель)", min_value=1.0, max_value=3.0, value=1.0, step=0.5)
    holdout_pct = st.slider("Последняя часть истории для out-of-sample (%)", 10, 40, 25)
    st.caption("Комиссия и проскальзывание применяются при каждом изменении позиции. Funding — упрощённое допущение, не фактическая история ставок.")

strategies = st.multiselect("Стратегии для сравнения", list(STRATEGIES.keys()), default=list(STRATEGIES.keys()))
run = st.button("Загрузить котировки и выполнить бэктест", type="primary")

if run:
    try:
        with st.spinner(f"Получаем публичные свечи {exchange_id.capitalize()}…"):
            df = fetch_ohlcv(symbol, timeframe, months, market, exchange_id)
        if df.empty or len(df) < 200:
            st.error(f"Недостаточно свечей: {len(df)}. Проверьте пару, рынок и таймфрейм.")
            st.stop()
        st.success(f"Получено {len(df):,} свечей: {df.index.min()} — {df.index.max()} (UTC)")
        st.caption(f"Источник: публичные OHLCV {exchange_id.capitalize()} через CCXT. Последняя незакрытая свеча исключается.")
        split = int(len(df) * (1 - holdout_pct / 100))
        train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
        summary = []
        curves = {}
        trade_frames = {}
        for name in strategies:
            params = {}
            a = run_backtest(train, name, initial, market, fee_pct/100, slippage_pct/100, leverage, funding_pct_8h/100)
            b = run_backtest(test, name, initial, market, fee_pct/100, slippage_pct/100, leverage, funding_pct_8h/100)
            summary.append({"Стратегия": name, "Период": "Train", **a["metrics"]})
            summary.append({"Стратегия": name, "Период": "Out-of-sample", **b["metrics"]})
            curves[f"{name} — train"] = a["equity"]
            curves[f"{name} — OOS"] = b["equity"]
            trade_frames[f"{name} — train"] = a["trades"]
            trade_frames[f"{name} — OOS"] = b["trades"]
        result = pd.DataFrame(summary)
        st.subheader("Результаты")
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.subheader("Кривая капитала")
        for name, curve in curves.items():
            st.line_chart(curve.rename(name), height=180)
        st.subheader("Сделки")
        chosen = st.selectbox("Таблица сделок", list(trade_frames.keys()))
        st.dataframe(trade_frames[chosen], use_container_width=True, hide_index=True)
        st.download_button("Скачать метрики CSV", result.to_csv(index=False).encode(), "metrics.csv", "text/csv")
        for key, frame in trade_frames.items():
            st.download_button(f"Скачать сделки: {key}", frame.to_csv(index=False).encode(), f"trades_{key.replace(' ', '_').replace('—','-')}.csv", "text/csv")
    except Exception as e:
        st.error(f"Не удалось выполнить бэктест: {e}")
        st.info("Попробуйте выбрать другую биржу, проверьте пару (например BTC/USDT), рынок и доступность API из региона размещения приложения. Доступность Binance также зависит от сети и региона.")

st.divider()
st.subheader("Paper trading — безопасный режим")
st.write("Этот MVP не отправляет реальные ордера и не просит API-ключи. Для paper trading используйте выбранную стратегию и новые закрытые свечи; отдельный постоянный исполнитель пока не включён.")
st.warning("Не используйте результаты как гарантию доходности. Малый капитал особенно чувствителен к комиссиям, минимальному размеру ордера и проскальзыванию.")

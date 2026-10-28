from __future__ import annotations
import time
import numpy as np
import pandas as pd

STRATEGIES = {
    "EMA crossover": "ema",
    "RSI mean reversion": "rsi",
    "Donchian breakout": "donchian",
}

def fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", months=12, market="spot", exchange_id="bybit"):
    """Download public OHLCV candles from Bybit or Binance using CCXT."""
    import ccxt

    supported = {"bybit": ccxt.bybit, "binance": ccxt.binance}
    if exchange_id not in supported:
        raise ValueError(f"Неподдерживаемая биржа: {exchange_id}")

    options = {"defaultType": "swap" if market == "linear" and exchange_id == "bybit" else
               "future" if market == "linear" and exchange_id == "binance" else "spot"}
    exchange = supported[exchange_id]({"enableRateLimit": True, "options": options})
    # Binance documents this host specifically for unauthenticated public spot market data.
    # It may work when api.binance.com is blocked by the hosting region; it is not a guarantee.
    if exchange_id == "binance" and market == "spot":
        exchange.urls["api"]["public"] = "https://data-api.binance.vision/api/v3"
    exchange.load_markets()

    requested_symbol = symbol.strip().upper().replace(" ", "")
    if market == "linear":
        # CCXT unified symbols for USDT perpetuals commonly use BTC/USDT:USDT.
        base = requested_symbol.split(":")[0]
        if exchange_id == "binance" and ":" not in requested_symbol:
            requested_symbol = base + ":USDT"
        elif exchange_id == "bybit" and ":" not in requested_symbol:
            requested_symbol = base + ":USDT"

    if requested_symbol not in exchange.markets:
        alternatives = [requested_symbol.split(":")[0], requested_symbol.replace(":USDT", "")]
        found = next((candidate for candidate in alternatives if candidate in exchange.markets), None)
        if found:
            requested_symbol = found
        else:
            raise ValueError(f"Пара {symbol} не найдена на {exchange_id} для рынка {market}. Для USDT-фьючерсов попробуйте BTC/USDT.")

    since = exchange.milliseconds() - int(months * 30.4375 * 24 * 60 * 60 * 1000)
    all_rows = []
    limit = 1000
    while since < exchange.milliseconds():
        rows = exchange.fetch_ohlcv(requested_symbol, timeframe=timeframe, since=since, limit=limit)
        if not rows:
            break
        all_rows.extend(rows)
        next_since = rows[-1][0] + 1
        if next_since <= since:
            break
        since = next_since
        if len(rows) < 2:
            break
        time.sleep(max(getattr(exchange, "rateLimit", 0), 0) / 1000)
    if not all_rows:
        raise ValueError(f"{exchange_id} не вернула исторические свечи для {requested_symbol}.")
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    # Never use the current, potentially incomplete candle.
    if len(df) > 1:
        df = df.iloc[:-1]
    return df.astype(float)

# Backwards-compatible alias for existing scripts/tests.
def fetch_bybit_ohlcv(symbol="BTC/USDT", timeframe="1h", months=12, market="spot"):
    return fetch_ohlcv(symbol, timeframe, months, market, "bybit")

def signal_series(df, strategy):
    close = df["close"]
    if strategy == "EMA crossover":
        fast = close.ewm(span=12, adjust=False).mean()
        slow = close.ewm(span=26, adjust=False).mean()
        raw = np.where(fast > slow, 1, 0)
        return pd.Series(raw, index=df.index, dtype=float)
    if strategy == "RSI mean reversion":
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        # Hold long after oversold; exit after RSI returns above 55.
        out = []
        pos = 0
        for value in rsi:
            if np.isnan(value):
                out.append(pos); continue
            if pos == 0 and value < 30: pos = 1
            elif pos == 1 and value > 55: pos = 0
            out.append(pos)
        return pd.Series(out, index=df.index, dtype=float)
    if strategy == "Donchian breakout":
        upper = df["high"].rolling(20).max().shift(1)
        lower = df["low"].rolling(10).min().shift(1)
        out = []
        pos = 0
        for i in range(len(df)):
            if pos == 0 and close.iloc[i] > upper.iloc[i]: pos = 1
            elif pos == 1 and close.iloc[i] < lower.iloc[i]: pos = 0
            out.append(pos)
        return pd.Series(out, index=df.index, dtype=float)
    raise ValueError(f"Неизвестная стратегия: {strategy}")

def _metrics(equity, trades):
    if len(equity) == 0:
        return {}
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    peak = equity.cummax()
    dd = (equity / peak - 1).min() * 100
    pnl = trades["net_pnl"] if len(trades) else pd.Series(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    return {
        "Начальный капитал": round(start, 4),
        "Конечный капитал": round(end, 4),
        "Доходность, %": round((end / start - 1) * 100, 3) if start else 0,
        "Макс. просадка, %": round(float(dd), 3),
        "Сделок": int(len(trades)),
        "Прибыльных сделок, %": round((len(wins) / len(trades) * 100), 2) if len(trades) else 0,
        "Profit factor": round(float(gross_profit / gross_loss), 3) if gross_loss else (float("inf") if gross_profit else 0),
        "Средний PnL сделки": round(float(pnl.mean()), 5) if len(pnl) else 0,
    }

def run_backtest(df, strategy, initial_equity=100.0, market="spot", fee=0.001,
                 slippage=0.0005, leverage=1.0, funding_8h=0.0001):
    """
    Signals are shifted one candle to avoid using a close to trade at that same close.
    Simple single-position model; spot is long-only, linear futures may be long/short
    for EMA strategy (other baseline strategies are long/flat). Funding is an
    explicit approximation applied per 8h held, not actual historical funding.
    """
    if len(df) < 50:
        raise ValueError("Для устойчивого расчёта требуется минимум 50 свечей.")
    df = df.copy()
    signal = signal_series(df, strategy).shift(1).fillna(0)
    if market == "linear" and strategy == "EMA crossover":
        # EMA cross below slow trend -> short. This simple variant is long/short.
        fast = df["close"].ewm(span=12, adjust=False).mean()
        slow = df["close"].ewm(span=26, adjust=False).mean()
        signal = pd.Series(np.where(fast.shift(1) > slow.shift(1), 1, -1), index=df.index, dtype=float)
    returns = df["close"].pct_change().fillna(0)
    position = signal.clip(-1, 1)
    if market == "spot":
        position = position.clip(0, 1)
    # Exposure return applied on the following candle; transaction costs on position changes.
    change = position.diff().abs().fillna(position.abs())
    cost = change * (fee + slippage)
    gross = position * returns * (leverage if market == "linear" else 1.0)
    if market == "linear":
        # Simplified funding assumption, proportional to absolute exposure and bars held.
        seconds = df.index.to_series().diff().dt.total_seconds().median()
        bars_per_8h = max(1, 8 * 3600 / seconds) if pd.notna(seconds) and seconds > 0 else 8
        funding = position.abs() * funding_8h / bars_per_8h
    else:
        funding = 0.0
    net_returns = gross - cost - funding
    equity = initial_equity * (1 + net_returns).cumprod()
    # Extract position transitions and calculate approximate trade PnL from equity differences.
    trades = []
    active = None
    prev_pos = 0
    prev_equity = initial_equity
    for i, (ts, pos) in enumerate(position.items()):
        eq = float(equity.iloc[i])
        if prev_pos == 0 and pos != 0:
            active = {"entry_time": ts, "side": "long" if pos > 0 else "short",
                      "entry_price": float(df["close"].iloc[i]), "entry_equity": eq}
        elif prev_pos != 0 and (pos == 0 or np.sign(pos) != np.sign(prev_pos)):
            if active:
                trades.append({"entry_time": active["entry_time"], "exit_time": ts,
                               "side": active["side"], "entry_price": active["entry_price"],
                               "exit_price": float(df["close"].iloc[i]),
                               "net_pnl": round(eq - active["entry_equity"], 6)})
            active = None
            if pos != 0:
                active = {"entry_time": ts, "side": "long" if pos > 0 else "short",
                          "entry_price": float(df["close"].iloc[i]), "entry_equity": eq}
        prev_pos = pos
        prev_equity = eq
    trades_df = pd.DataFrame(trades, columns=["entry_time", "exit_time", "side", "entry_price", "exit_price", "net_pnl"])
    return {"metrics": _metrics(equity, trades_df), "equity": equity.rename("Equity"), "trades": trades_df}

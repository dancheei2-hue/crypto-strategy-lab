from __future__ import annotations

import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd

TIMEFRAMES = {"5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400}
STRATEGIES = {
    "EMA crossover": "ema",
    "RSI mean reversion": "rsi",
    "Donchian breakout": "donchian",
    "MACD trend": "macd",
    "Bollinger reversion": "bollinger",
    "ATR trend filter": "atr_trend",
    "EMA + RSI confirmation": "ema_rsi",
    "Buy & Hold benchmark": "buy_hold",
}


def _http_json(url: str, timeout: int = 25):
    req = Request(url, headers={"User-Agent": "CryptoStrategyLab/2.0", "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            import json
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Ошибка HTTP-запроса котировок: {exc}") from exc


def _normalise_rows(rows):
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], name="timestamp"))
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    return df.astype(float)


def _fetch_binance_spot(symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    pair = symbol.strip().upper().replace("/", "").replace(" ", "")
    rows_all, cursor = [], since_ms
    while cursor < int(time.time() * 1000):
        query = urlencode({"symbol": pair, "interval": timeframe, "startTime": cursor, "limit": 1000})
        data = _http_json("https://data-api.binance.vision/api/v3/klines?" + query)
        if not isinstance(data, list):
            raise ValueError(f"Binance вернула неожиданный ответ: {data}")
        if not data:
            break
        rows_all.extend([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in data])
        nxt = int(data[-1][0]) + 1
        if nxt <= cursor or len(data) < 2:
            break
        cursor = nxt
        time.sleep(0.12)
    df = _normalise_rows(rows_all)
    if len(df) > 0:
        df = df.iloc[:-1]  # ignore last, likely unfinished candle
    return df


def _fetch_bingx(symbol: str, timeframe: str, months: int, market: str) -> pd.DataFrame:
    pair = symbol.strip().upper().replace("/", "-").replace("_", "-").replace(" ", "")
    since_ms = int(time.time() * 1000) - int(months * 30.4375 * 24 * 60 * 60 * 1000)
    interval = timeframe  # BingX expects 1h rather than 60min
    all_rows, cursor = [], since_ms
    endpoint = ("https://open-api.bingx.com/openApi/spot/v2/market/kline" if market == "spot"
                else "https://open-api.bingx.com/openApi/swap/v3/quote/klines")
    while cursor < int(time.time() * 1000):
        query = urlencode({"symbol": pair, "interval": interval, "startTime": cursor,
                           "endTime": int(time.time() * 1000), "limit": 1000})
        payload = _http_json(endpoint + "?" + query)
        code = str(payload.get("code", "0"))
        if code not in ("0", "200", "None"):
            raise ValueError(f"BingX API: code={code}; {payload.get('msg', payload)}")
        data = payload.get("data", [])
        if isinstance(data, dict):
            data = data.get("data", data.get("rows", []))
        if not data:
            break
        page_rows = []
        for r in data:
            if isinstance(r, dict):
                ts = r.get("time", r.get("timestamp", r.get("openTime")))
                if ts is None: continue
                if float(ts) < 10**11: ts = int(float(ts) * 1000)
                page_rows.append([int(float(ts)), r.get("open"), r.get("high"), r.get("low"), r.get("close"), r.get("volume", r.get("vol", 0))])
            elif isinstance(r, (list, tuple)) and len(r) >= 6:
                page_rows.append([r[0], *r[1:6]])
        if not page_rows:
            break
        page_rows.sort(key=lambda r: int(r[0]))
        all_rows.extend(page_rows)
        nxt = int(page_rows[-1][0]) + 1
        if nxt <= cursor or len(page_rows) < 2:
            break
        cursor = nxt
        time.sleep(0.12)
    df = _normalise_rows(all_rows)
    if len(df) > 0:
        df = df.loc[~df.index.duplicated(keep="last")]
        if len(df) > 1: df = df.iloc[:-1]
    return df


def fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", months=12, market="spot", exchange_id="bybit"):
    """Download public OHLCV candles. API credentials are not needed."""
    import ccxt
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Неподдерживаемый таймфрейм: {timeframe}")
    if market not in {"spot", "linear"}:
        raise ValueError("Рынок должен быть spot или linear")
    since_ms = int(time.time() * 1000) - int(months * 30.4375 * 24 * 60 * 60 * 1000)
    if exchange_id == "binance" and market == "spot":
        df = _fetch_binance_spot(symbol, timeframe, since_ms)
    elif exchange_id == "bingx":
        df = _fetch_bingx(symbol, timeframe, months, market)
    else:
        supported = {"bybit": ccxt.bybit, "binance": ccxt.binance}
        if exchange_id not in supported:
            raise ValueError(f"Неподдерживаемая биржа: {exchange_id}")
        options = {"defaultType": "future" if market == "linear" and exchange_id == "binance" else
                   "swap" if market == "linear" else "spot"}
        exchange = supported[exchange_id]({"enableRateLimit": True, "options": options})
        exchange.load_markets()
        requested_symbol = symbol.strip().upper().replace(" ", "")
        if market == "linear" and ":" not in requested_symbol:
            requested_symbol = requested_symbol.split(":")[0] + ":USDT"
        if requested_symbol not in exchange.markets:
            alts = [requested_symbol.split(":")[0], requested_symbol.replace(":USDT", "")]
            found = next((candidate for candidate in alts if candidate in exchange.markets), None)
            if found: requested_symbol = found
            else: raise ValueError(f"Пара {symbol} не найдена на {exchange_id} для рынка {market}.")
        all_rows, cursor = [], since_ms
        while cursor < exchange.milliseconds():
            rows = exchange.fetch_ohlcv(requested_symbol, timeframe=timeframe, since=cursor, limit=1000)
            if not rows: break
            all_rows.extend(rows)
            nxt = int(rows[-1][0]) + 1
            if nxt <= cursor or len(rows) < 2: break
            cursor = nxt
            time.sleep(max(getattr(exchange, "rateLimit", 0), 0) / 1000)
        df = _normalise_rows(all_rows)
        if len(df) > 0 and len(df) > 1: df = df.iloc[:-1]
    if df.empty:
        raise ValueError(f"Биржа не вернула свечи для {symbol}, {timeframe}, {market}. Попробуйте BTC/USDT и 1h/15m.")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_bybit_ohlcv(symbol="BTC/USDT", timeframe="1h", months=12, market="spot"):
    return fetch_ohlcv(symbol, timeframe, months, market, "bybit")


def _indicators(df):
    close, high, low = df["close"], df["high"], df["low"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    mid = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    return {"rsi": rsi, "ema_fast": ema_fast, "ema_slow": ema_slow, "macd": macd,
            "macd_signal": macd_signal, "bb_upper": mid + 2*std, "bb_lower": mid-2*std,
            "atr": atr, "donchian_upper": high.rolling(20).max().shift(1),
            "donchian_lower": low.rolling(10).min().shift(1)}


def signal_series(df, strategy, params=None):
    """Return desired exposure signals: -1 short, 0 flat, +1 long."""
    p = params or {}
    c = df["close"]
    ind = _indicators(df)
    if strategy == "EMA crossover":
        fast = c.ewm(span=int(p.get("ema_fast", 12)), adjust=False).mean()
        slow = c.ewm(span=int(p.get("ema_slow", 26)), adjust=False).mean()
        return pd.Series(np.where(fast > slow, 1.0, 0.0), index=df.index)
    if strategy == "RSI mean reversion":
        entry, exit_level = float(p.get("rsi_entry", 30)), float(p.get("rsi_exit", 55))
        result, pos = [], 0
        for v in ind["rsi"]:
            if pd.notna(v):
                if pos == 0 and v < entry: pos = 1
                elif pos == 1 and v > exit_level: pos = 0
            result.append(pos)
        return pd.Series(result, index=df.index, dtype=float)
    if strategy == "Donchian breakout":
        upper = df["high"].rolling(int(p.get("donchian_entry", 20))).max().shift(1)
        lower = df["low"].rolling(int(p.get("donchian_exit", 10))).min().shift(1)
        result, pos = [], 0
        for i, close in enumerate(c):
            if pos == 0 and pd.notna(upper.iloc[i]) and close > upper.iloc[i]: pos = 1
            elif pos == 1 and pd.notna(lower.iloc[i]) and close < lower.iloc[i]: pos = 0
            result.append(pos)
        return pd.Series(result, index=df.index, dtype=float)
    if strategy == "MACD trend":
        return pd.Series(np.where(ind["macd"] > ind["macd_signal"], 1.0, 0.0), index=df.index)
    if strategy == "Bollinger reversion":
        result, pos = [], 0
        for i, close in enumerate(c):
            lo, mid = ind["bb_lower"].iloc[i], c.rolling(20).mean().iloc[i]
            if pos == 0 and pd.notna(lo) and close < lo: pos = 1
            elif pos == 1 and pd.notna(mid) and close > mid: pos = 0
            result.append(pos)
        return pd.Series(result, index=df.index, dtype=float)
    if strategy == "ATR trend filter":
        trend = c.ewm(span=50, adjust=False).mean()
        return pd.Series(np.where((c > trend) & (c.diff() > 0), 1.0, 0.0), index=df.index)
    if strategy == "EMA + RSI confirmation":
        return pd.Series(np.where((ind["ema_fast"] > ind["ema_slow"]) & (ind["rsi"] > 50), 1.0, 0.0), index=df.index)
    if strategy == "Buy & Hold benchmark":
        return pd.Series(1.0, index=df.index)
    raise ValueError(f"Неизвестная стратегия: {strategy}")


def _metrics(equity, trades, net_returns, periods_per_year):
    if len(equity) == 0: return {}
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    peak = equity.cummax()
    dd_series = (equity / peak - 1) * 100
    pnl = trades["net_pnl"] if len(trades) else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_profit, gross_loss = wins.sum(), abs(losses.sum())
    total_return = (end / start - 1) if start else 0
    years = max(len(equity) / max(periods_per_year, 1), 1 / max(periods_per_year, 1))
    cagr = ((end / start) ** (1 / years) - 1) * 100 if start > 0 and end > 0 else -100.0
    std = float(net_returns.std(ddof=1)) if len(net_returns) > 1 else 0.0
    sharpe = float(net_returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0
    downside = net_returns[net_returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float(net_returns.mean() / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0.0
    return {
        "Начальный капитал": round(start, 4), "Конечный капитал": round(end, 4),
        "Доходность, %": round(total_return * 100, 3), "CAGR, %": round(cagr, 3),
        "Макс. просадка, %": round(float(dd_series.min()), 3), "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3), "Сделок": int(len(trades)),
        "Прибыльных сделок, %": round(len(wins)/len(trades)*100, 2) if len(trades) else 0,
        "Profit factor": round(float(gross_profit/gross_loss), 3) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0),
        "Средний PnL сделки": round(float(pnl.mean()), 5) if len(pnl) else 0,
        "Лучший trade": round(float(pnl.max()), 5) if len(pnl) else 0,
        "Худший trade": round(float(pnl.min()), 5) if len(pnl) else 0,
        "Время в рынке, %": round(float((equity.index.to_series().notna()).sum()*0 + 100), 2) if False else None,
    }


def run_backtest(df, strategy, initial_equity=100.0, market="spot", fee=0.001,
                 slippage=0.0005, leverage=1.0, funding_8h=0.0001, params=None,
                 position_size_pct=100.0, allow_short=True):
    """Vectorised close-to-close backtest with one-bar-lagged signal and transaction costs."""
    if len(df) < 50: raise ValueError("Для расчёта требуется минимум 50 свечей.")
    if initial_equity <= 0: raise ValueError("Стартовый капитал должен быть больше нуля.")
    df = df.copy().sort_index()
    if not set(["open", "high", "low", "close", "volume"]).issubset(df.columns):
        raise ValueError("OHLCV должны содержать open/high/low/close/volume")
    if df["close"].isna().any() or (df["close"] <= 0).any(): raise ValueError("В данных есть некорректные цены закрытия.")
    raw = signal_series(df, strategy, params).shift(1).fillna(0).clip(-1, 1)
    if market == "spot" or not allow_short: raw = raw.clip(0, 1)
    position = raw * max(0.0, min(float(position_size_pct), 100.0)) / 100.0
    if market == "linear": position = position * float(leverage)
    returns = df["close"].pct_change().fillna(0.0)
    changes = position.diff().abs().fillna(position.abs())
    transaction_cost = changes * (float(fee) + float(slippage))
    if market == "linear":
        seconds = df.index.to_series().diff().dt.total_seconds().median()
        bars_per_8h = max(1.0, 28800 / seconds) if pd.notna(seconds) and seconds > 0 else 8.0
        funding = (position.abs() * float(funding_8h) / bars_per_8h)
    else: funding = pd.Series(0.0, index=df.index)
    net_returns = (position * returns) - transaction_cost - funding
    # Avoid an impossible negative equity after a catastrophic leveraged bar.
    net_returns = net_returns.clip(lower=-0.999)
    equity = float(initial_equity) * (1 + net_returns).cumprod()
    equity.name = "Equity"
    active = None
    prev_pos = 0.0
    trades = []
    for i, (ts, pos) in enumerate(position.items()):
        price = float(df["close"].iloc[i])
        eq = float(equity.iloc[i])
        if prev_pos == 0 and pos != 0:
            active = {"entry_time": ts, "side": "long" if pos > 0 else "short", "entry_price": price, "entry_equity": eq}
        elif prev_pos != 0 and (pos == 0 or np.sign(pos) != np.sign(prev_pos)):
            if active:
                trades.append({"entry_time": active["entry_time"], "exit_time": ts, "side": active["side"],
                               "entry_price": active["entry_price"], "exit_price": price,
                               "net_pnl": round(eq - active["entry_equity"], 6),
                               "return_pct": round((eq/active["entry_equity"]-1)*100, 4) if active["entry_equity"] else 0})
            active = None
            if pos != 0: active = {"entry_time": ts, "side": "long" if pos > 0 else "short", "entry_price": price, "entry_equity": eq}
        prev_pos = pos
    trades_df = pd.DataFrame(trades, columns=["entry_time", "exit_time", "side", "entry_price", "exit_price", "net_pnl", "return_pct"])
    seconds = df.index.to_series().diff().dt.total_seconds().median()
    ppy = (365.25 * 24 * 3600 / seconds) if pd.notna(seconds) and seconds > 0 else 365.25
    metrics = _metrics(equity, trades_df, net_returns, ppy)
    metrics["Время в рынке, %"] = round(float((position.abs() > 0).mean() * 100), 2)
    metrics["Средняя экспозиция, x"] = round(float(position.abs().mean()), 3)
    metrics["Средний бар, %"] = round(float(net_returns.mean() * 100), 5)
    return {"metrics": metrics, "equity": equity, "trades": trades_df, "position": position, "returns": net_returns}

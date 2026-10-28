"""Simple local SQLite paper-trading journal. No real exchange orders are ever sent."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).with_name("paper_trading.db")


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS accounts (
        id INTEGER PRIMARY KEY CHECK (id = 1), initial_cash REAL NOT NULL,
        cash REAL NOT NULL, quantity REAL NOT NULL DEFAULT 0,
        entry_price REAL, last_processed TEXT, updated_at TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL, strategy TEXT NOT NULL, action TEXT NOT NULL,
        price REAL NOT NULL, quantity REAL NOT NULL, cash_after REAL NOT NULL,
        equity_after REAL NOT NULL, note TEXT
    )""")
    conn.commit()
    return conn


def get_account():
    with _connect() as c:
        row = c.execute("SELECT * FROM accounts WHERE id=1").fetchone()
        return dict(row) if row else None


def reset_account(initial_cash: float):
    with _connect() as c:
        c.execute("DELETE FROM accounts")
        c.execute("DELETE FROM events")
        c.execute("INSERT INTO accounts(id,initial_cash,cash,quantity,entry_price,last_processed,updated_at) VALUES(1,?,?,0,NULL,NULL,datetime('now'))", (float(initial_cash), float(initial_cash)))


def get_events(limit=500):
    with _connect() as c:
        rows = c.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return pd.DataFrame([dict(r) for r in rows])


def process_closed_candle(timestamp, symbol, strategy, price, target_position, fee_rate=0.001, slippage_rate=0.0005):
    """Process a single new candle, target_position 0/1. Idempotent per candle timestamp."""
    ts = pd.Timestamp(timestamp).isoformat()
    price = float(price)
    target = 1 if float(target_position) > 0 else 0
    with _connect() as c:
        c.execute("BEGIN IMMEDIATE")
        a = c.execute("SELECT * FROM accounts WHERE id=1").fetchone()
        if not a:
            raise ValueError("Сначала создайте paper-счёт.")
        if a["last_processed"] == ts:
            return {"status": "duplicate", "message": "Эта свеча уже обработана.", "account": dict(a)}
        cash, qty = float(a["cash"]), float(a["quantity"])
        action, note = "HOLD", "Сигнал не изменил позицию"
        # Long-only spot simulation: use all available cash; no leverage or shorts.
        if target and qty <= 0 and cash > 0:
            effective = price * (1 + float(slippage_rate))
            spend = cash / (1 + float(fee_rate))
            qty = spend / effective
            fee = spend * float(fee_rate)
            cash = max(0.0, cash - spend - fee)
            action, note = "BUY", "Виртуальная покупка; реальные ордера не отправлялись"
        elif not target and qty > 0:
            effective = price * (1 - float(slippage_rate))
            gross = qty * effective
            fee = gross * float(fee_rate)
            cash += gross - fee
            qty = 0.0
            action, note = "SELL", "Виртуальная продажа; реальные ордера не отправлялись"
        equity = cash + qty * price
        c.execute("UPDATE accounts SET cash=?, quantity=?, entry_price=CASE WHEN ? > 0 THEN COALESCE(entry_price, ?) ELSE NULL END, last_processed=?, updated_at=datetime('now') WHERE id=1", (cash, qty, qty, price, ts))
        c.execute("INSERT INTO events(timestamp,symbol,strategy,action,price,quantity,cash_after,equity_after,note) VALUES(?,?,?,?,?,?,?,?,?)", (ts, symbol, strategy, action, price, qty, cash, equity, note))
        c.commit()
        a2 = c.execute("SELECT * FROM accounts WHERE id=1").fetchone()
        return {"status": "processed", "message": note, "account": dict(a2), "equity": equity, "action": action}

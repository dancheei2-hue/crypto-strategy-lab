# Crypto Strategy Lab

Russian-language crypto strategy research MVP with selectable public market data from Binance or Bybit.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app lets you select Binance or Bybit and downloads public OHLCV candles via CCXT. No API keys are required for public candles. It excludes the latest candle because it may still be open. Availability depends on the hosting region and exchange restrictions; Binance may also be blocked in some regions.

## Included baseline strategies
- EMA crossover (12/26)
- RSI mean reversion (14; entry below 30, exit above 55)
- Donchian breakout (20-bar breakout, 10-bar exit)

## Important model limitations
- These are baseline research implementations, not proven profitable systems.
- Signals are delayed by one candle to avoid trading on the same candle close that creates the signal.
- Fee and slippage inputs are explicit, editable assumptions.
- Linear-futures funding is an approximation based on the user-entered rate, not historical funding data.
- This MVP does not submit real orders and does not collect API secrets.
- Trade-by-trade PnL is an approximation from changes in equity around position transitions. Review before relying on results for decisions.
- The initial train/OOS split evaluates each segment separately; indicators in the OOS slice are initialized on the OOS data, so this is a basic holdout report rather than a production-grade walk-forward framework.
- Liquidation, exchange-specific minimum order sizes, and exact fee tiers are not modeled yet.
- Small balances can be materially affected by minimum order sizes and fees.

## Tests
```bash
python -m unittest discover -s tests -v
```

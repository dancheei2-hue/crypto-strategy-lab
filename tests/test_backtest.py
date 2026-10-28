import unittest
import numpy as np
import pandas as pd
from backtest import signal_series, run_backtest, STRATEGIES

class BacktestTests(unittest.TestCase):
    def sample(self, n=500):
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        t = np.arange(n)
        close = 100 + np.linspace(0, 25, n) + 3*np.sin(t/8) + 0.8*np.sin(t/2.9)
        return pd.DataFrame({"open": close, "high": close*1.012, "low": close*.988,
                             "close": close, "volume": np.ones(n)}, index=idx)

    def test_all_strategy_signals_are_valid(self):
        df = self.sample()
        for strategy in STRATEGIES:
            s = signal_series(df, strategy)
            self.assertEqual(len(s), len(df), strategy)
            self.assertTrue(s.dropna().isin([-1, 0, 1]).all(), strategy)

    def test_backtest_outputs_equity_and_metrics(self):
        df = self.sample()
        out = run_backtest(df, "EMA crossover", 100, "spot", .001, .0005)
        self.assertEqual(len(out["equity"]), len(df))
        self.assertIn("Доходность, %", out["metrics"])
        self.assertIn("Макс. просадка, %", out["metrics"])
        self.assertIn("Sharpe", out["metrics"])
        self.assertTrue(np.isfinite(out["equity"]).all())
        self.assertGreater(out["equity"].iloc[-1], 0)

    def test_spot_never_short(self):
        out = run_backtest(self.sample(), "EMA crossover", 100, "spot")
        self.assertGreaterEqual(out["position"].min(), 0)

    def test_futures_can_go_short_for_ema(self):
        out = run_backtest(self.sample(), "EMA crossover", 100, "linear")
        self.assertGreaterEqual(out["position"].min(), -5)

    def test_invalid_capital(self):
        with self.assertRaises(ValueError):
            run_backtest(self.sample(), "EMA crossover", 0)

if __name__ == "__main__":
    unittest.main()

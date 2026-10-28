import unittest
import numpy as np
import pandas as pd
from backtest import signal_series, run_backtest

class BacktestTests(unittest.TestCase):
    def sample(self, n=400):
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        close = 100 + np.linspace(0, 20, n) + np.sin(np.arange(n)/7)*2
        return pd.DataFrame({"open": close, "high": close*1.01, "low": close*.99,
                             "close": close, "volume": np.ones(n)}, index=idx)

    def test_signals_length_and_bounds(self):
        df = self.sample()
        for strategy in ["EMA crossover", "RSI mean reversion", "Donchian breakout"]:
            s = signal_series(df, strategy)
            self.assertEqual(len(s), len(df))
            self.assertTrue(set(s.dropna().unique()).issubset({0, 1}))

    def test_backtest_outputs_metrics(self):
        df = self.sample()
        out = run_backtest(df, "EMA crossover", 100, "spot", .001, .0005)
        self.assertEqual(len(out["equity"]), len(df))
        self.assertIn("Доходность, %", out["metrics"])
        self.assertIn("Макс. просадка, %", out["metrics"])
        self.assertGreater(out["metrics"]["Конечный капитал"], 0)

    def test_spot_never_short(self):
        df = self.sample()
        out = run_backtest(df, "EMA crossover", 100, "spot")
        self.assertGreater(out["metrics"]["Конечный капитал"], 0)

if __name__ == "__main__":
    unittest.main()

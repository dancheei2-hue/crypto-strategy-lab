import unittest
import numpy as np
import pandas as pd
from auto_research import automatic_research

class AutoResearchTests(unittest.TestCase):
    def sample(self, n=600):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        close = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.006, n)))
        open_ = np.r_[close[0], close[:-1]]
        return pd.DataFrame({"open": open_, "high": np.maximum(open_, close)*1.003,
            "low": np.minimum(open_, close)*0.997, "close": close,
            "volume": rng.uniform(1, 100, n)}, index=idx)

    def test_auto_research_returns_oos_folds(self):
        result = automatic_research(self.sample(), folds=3)
        self.assertEqual(len(result["folds"]), 3)
        self.assertIn("Доходность OOS, %", result["folds"].columns)
        self.assertIn("verdict", result)
        self.assertGreater(result["summary"]["Кандидатов в поиске"], 1)

if __name__ == "__main__":
    unittest.main()

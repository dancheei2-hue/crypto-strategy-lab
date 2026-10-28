import unittest
import numpy as np
import pandas as pd
from regime_engine import adaptive_regime_backtest

class RegimeEngineTests(unittest.TestCase):
    def sample(self, n=700):
        rng = np.random.default_rng(10)
        # Regime shifts: trend up, noisy sideways, trend down.
        drift = np.r_[np.full(n//3, .0008), np.zeros(n//3), np.full(n-2*(n//3), -.0005)]
        returns = drift + rng.normal(0, .004, n)
        close = 100 * np.exp(np.cumsum(returns))
        idx = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
        op = np.r_[close[0], close[:-1]]
        return pd.DataFrame({'open':op,'high':np.maximum(op,close)*1.002,'low':np.minimum(op,close)*.998,'close':close,'volume':rng.uniform(10,100,n)}, index=idx)

    def test_adaptive_engine_outputs_metrics_and_causal_details(self):
        out = adaptive_regime_backtest(self.sample(), lookback=100)
        self.assertIn('Sharpe', out['metrics'])
        self.assertEqual(len(out['details']), 700)
        self.assertIn('selected_candidate', out['details'])
        self.assertIn('verdict', out)
        self.assertTrue((out['details']['equity'] > 0).all())

    def test_spot_positions_not_short(self):
        out = adaptive_regime_backtest(self.sample(), market='spot', lookback=100)
        self.assertGreaterEqual(out['details']['position'].min(), 0)

if __name__ == '__main__':
    unittest.main()

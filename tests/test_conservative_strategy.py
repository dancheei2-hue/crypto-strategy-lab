import unittest
import numpy as np
import pandas as pd
from conservative_strategy import conservative_backtest

class ConservativeStrategyTests(unittest.TestCase):
    def sample(self, n=600):
        idx = pd.date_range('2020-01-01', periods=n, freq='D', tz='UTC')
        rng = np.random.default_rng(7)
        returns = rng.normal(0.0007, 0.018, n)
        close = 100 * np.cumprod(1 + returns)
        return pd.DataFrame({'open': close, 'high': close*1.02, 'low': close*.98,
                             'close': close, 'volume': np.ones(n)}, index=idx)

    def test_returns_metrics_and_detail(self):
        out = conservative_backtest(self.sample(), 1000, .001, .0005)
        self.assertIn('Макс. просадка, %', out['metrics'])
        self.assertIn('buy_hold_equity', out['details'])
        self.assertTrue((out['details']['exposure'] >= 0).all())
        self.assertTrue((out['details']['exposure'] <= .5).all())

    def test_no_leverage_exposure_cap(self):
        out = conservative_backtest(self.sample(), max_exposure=.25)
        self.assertLessEqual(float(out['details']['exposure'].max()), .25 + 1e-9)

    def test_short_history_rejected(self):
        with self.assertRaises(ValueError):
            conservative_backtest(self.sample(100))

if __name__ == '__main__':
    unittest.main()

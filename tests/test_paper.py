import tempfile
import unittest
from pathlib import Path
import pandas as pd
import paper

class PaperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_path = paper.DB_PATH
        paper.DB_PATH = Path(self.tmp.name) / "test.db"
        paper.reset_account(1000)

    def tearDown(self):
        paper.DB_PATH = self.old_path
        self.tmp.cleanup()

    def test_buy_and_duplicate_candle(self):
        ts = pd.Timestamp("2026-01-01T00:00:00Z")
        first = paper.process_closed_candle(ts, "BTC/USDT", "EMA crossover", 100, 1)
        self.assertEqual(first["action"], "BUY")
        second = paper.process_closed_candle(ts, "BTC/USDT", "EMA crossover", 100, 1)
        self.assertEqual(second["status"], "duplicate")
        self.assertEqual(len(paper.get_events()), 1)

    def test_sell_closes_position(self):
        ts1 = pd.Timestamp("2026-01-01T00:00:00Z")
        ts2 = pd.Timestamp("2026-01-01T01:00:00Z")
        paper.process_closed_candle(ts1, "BTC/USDT", "EMA crossover", 100, 1)
        out = paper.process_closed_candle(ts2, "BTC/USDT", "EMA crossover", 110, 0)
        self.assertEqual(out["action"], "SELL")
        self.assertAlmostEqual(out["account"]["quantity"], 0)

if __name__ == "__main__":
    unittest.main()

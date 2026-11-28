import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tabs.tab_multibagger_base import detect_multibagger_base


class TestMultibaggerBase(unittest.TestCase):
    def _make_cupid_style_df(self):
        """Synthetic stock with prior run (80 -> 150) + 25d tight shelf (140-152) on dry volume + breakout."""
        dates = pd.bdate_range(end="2026-11-27", periods=220)
        base1 = np.linspace(75.0, 95.0, 140)
        run1 = np.linspace(95.0, 150.0, 54)
        # 25-day tight horizontal shelf (142 to 150, ~5.6% range)
        shelf = 146.0 + 3.0 * np.sin(np.linspace(0, 4 * np.pi, 25))
        closes = np.concatenate([base1, run1, shelf, np.array([152.5])])

        highs = closes * 1.01
        lows = closes * 0.99
        opens = closes * 0.995
        opens[-1] = 149.0

        vols = np.full(220, 250000.0)
        vols[-26:-1] = 100000.0  # Dry volume inside shelf (0.4x)
        vols[-1] = 450000.0      # Breakout volume surge

        return pd.DataFrame({
            "Date": dates,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": vols,
        })

    def test_detects_high_tight_flag_breakout(self):
        df = self._make_cupid_style_df()
        res = detect_multibagger_base("TESTCUPID", df, max_box_range_pct=20.0, min_box_days=15)
        self.assertIsNotNone(res)
        self.assertTrue(res["is_breakout"])
        self.assertIn("Flag", res["setup_style"])
        self.assertLessEqual(res["box_range_pct"], 20.0)
        self.assertLess(res["vol_dry_ratio"], 0.8)
        self.assertGreater(res["target_price"], res["cmp"])

    def test_rejects_downtrend_stock(self):
        df = self._make_cupid_style_df()
        df["Close"] = np.linspace(200.0, 100.0, 220)
        df["High"] = df["Close"] * 1.01
        df["Low"] = df["Close"] * 0.99
        res = detect_multibagger_base("BEAR", df)
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tabs.tab_swing_finalists import score_swing_candidate


class TestSwingFinalists(unittest.TestCase):
    def _make_vcp_pullback_df(self):
        """Build a 220-day synthetic Stage-2 momentum stock that ran up and then formed a tight dry-vol VCP pullback."""
        dates = pd.bdate_range(end="2026-11-27", periods=220)
        # Base 100 -> steady uptrend to 125 over 190 days, sharp 20d runup to 165, then 10d quiet pullback to 152
        close = np.linspace(100.0, 125.0, 190)
        runup = np.linspace(125.0, 165.0, 20)
        # Quiet pullback (~-8% off 165 high -> 152, still +10%+ on 1M and +25%+ on 3M)
        pullback = np.linspace(163.0, 152.0, 10)
        closes = np.concatenate([close, runup, pullback])

        # Prior 20d wide range (4%), last 5d tight range (1%)
        spreads = np.full(220, 0.03)
        spreads[-25:-5] = 0.045
        spreads[-5:] = 0.010

        highs = closes * (1 + spreads / 2)
        lows = closes * (1 - spreads / 2)
        opens = closes * 0.998

        # Volume: 200k avg, last 5d dry volume 80k (0.4x)
        vols = np.full(220, 200000.0)
        vols[-5:] = 80000.0

        return pd.DataFrame({
            "Date": dates,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": vols,
        })

    def test_qualifies_vcp_pullback_leader(self):
        df = self._make_vcp_pullback_df()
        res = score_swing_candidate("TESTVCP", df, min_1m_pct=2.0, min_3m_pct=10.0)
        self.assertIsNotNone(res)
        self.assertEqual(res["symbol"], "TESTVCP")
        self.assertGreater(res["ret_3m"], 10.0)
        self.assertLess(res["pullback_pct"], -1.5)
        self.assertLess(res["contraction_ratio"], 0.75)
        self.assertLess(res["vol_dry_ratio"], 0.85)
        self.assertGreater(res["target_price"], res["cmp"])
        self.assertLess(res["stop_loss"], res["cmp"])

    def test_rejects_downtrend_stock(self):
        df = self._make_vcp_pullback_df()
        df["Close"] = np.linspace(200.0, 100.0, 220)
        df["High"] = df["Close"] * 1.01
        df["Low"] = df["Close"] * 0.99
        res = score_swing_candidate("BEAR", df)
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()

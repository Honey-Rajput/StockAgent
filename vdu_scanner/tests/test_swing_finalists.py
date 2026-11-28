import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tabs.tab_swing_finalists import score_swing_candidate


class TestSwingFinalists(unittest.TestCase):
    def _make_manas_arora_vcp_df(self):
        """Synthetic stock with >=35% run (100 -> 165), <=25% off 1M high (~152), VCP contraction, and gap-up off recent low."""
        dates = pd.bdate_range(end="2026-11-27", periods=220)
        close = np.linspace(100.0, 115.0, 180)
        runup = np.linspace(115.0, 165.0, 25)
        pullback = np.linspace(163.0, 148.0, 14)
        closes = np.concatenate([close, runup, pullback, np.array([153.5])])

        spreads = np.full(220, 0.03)
        spreads[-26:-6] = 0.045
        spreads[-6:-1] = 0.010
        spreads[-1] = 0.02

        highs = closes * (1 + spreads / 2)
        lows = closes * (1 - spreads / 2)
        opens = closes * 0.998
        # Last day opens gap-up (+2.5% above prev close 148.0 -> 151.7) and closes higher at 153.5
        opens[-1] = 151.7

        vols = np.full(220, 200000.0)
        vols[-6:-1] = 80000.0

        return pd.DataFrame({
            "Date": dates,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": vols,
        })

    def test_qualifies_manas_arora_vcp_gap_up(self):
        df = self._make_manas_arora_vcp_df()
        res = score_swing_candidate("TESTMA", df, min_run_pct=35.0, max_dd_1m_pct=25.0, require_gap_up=True)
        self.assertIsNotNone(res)
        self.assertEqual(res["symbol"], "TESTMA")
        self.assertGreaterEqual(res["prior_run_pct"], 35.0)
        self.assertGreaterEqual(res["dd_1m_high_pct"], -25.0)
        self.assertLess(res["contraction_ratio"], 0.95)
        self.assertGreaterEqual(res["gap_up_pct"], 0.5)
        self.assertGreater(res["target_price"], res["cmp"])
        self.assertLess(res["stop_loss"], res["cmp"])

    def test_rejects_downtrend_stock(self):
        df = self._make_manas_arora_vcp_df()
        df["Close"] = np.linspace(200.0, 100.0, 220)
        df["High"] = df["Close"] * 1.01
        df["Low"] = df["Close"] * 0.99
        res = score_swing_candidate("BEAR", df)
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()

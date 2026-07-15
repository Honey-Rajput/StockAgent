import numpy as np
import pandas as pd
import yfinance as yf


# ==========================================
# 1. CONFIG (mirrors Pine "inputs")
# ==========================================

class VCPConfig:
    def __init__(
        self,
        ma50_len: int = 50,
        ma150_len: int = 150,
        ma200_len: int = 200,
        vcp_lookback: int = 5,
        vcp_thresh: float = 2.5,   # % range considered a "squeeze"
        benchmark: str = "^NSEI",  # swap SPY -> Nifty 50 by default for NSE use
        pressure_window: int = 20,
        risk_low: float = 15.0,    # <=15% above 50MA -> Low Risk
        risk_caution: float = 25.0,  # <=25% -> Caution, else Extended
        rpr_good: float = 80.0,
        rpr_warn: float = 70.0,
        contraction_lookback: int = 3,   # bars to confirm range is tightening
        vol_dryup_ratio: float = 0.75,   # volume must fall below 75% of its 20d avg
        breakout_vol_mult: float = 1.5,  # volume must expand to 1.5x 20d avg on breakout
    ):
        self.ma50_len = ma50_len
        self.ma150_len = ma150_len
        self.ma200_len = ma200_len
        self.vcp_lookback = vcp_lookback
        self.vcp_thresh = vcp_thresh
        self.benchmark = benchmark
        self.pressure_window = pressure_window
        self.risk_low = risk_low
        self.risk_caution = risk_caution
        self.rpr_good = rpr_good
        self.rpr_warn = rpr_warn
        self.contraction_lookback = contraction_lookback
        self.vol_dryup_ratio = vol_dryup_ratio
        self.breakout_vol_mult = breakout_vol_mult


# ==========================================
# 2. HELPERS
# ==========================================

def _roc(series: pd.Series, length: int) -> pd.Series:
    """Rate of change over `length` bars, same as Pine's roc() closure."""
    return (series - series.shift(length)) / series.shift(length) * 100


def _fetch_history(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ==========================================
# 3. CORE ANALYZER
# ==========================================

class MinerviniVCPAnalyzer:
    """
    Usage:
        analyzer = MinerviniVCPAnalyzer("SIEMENS.NS", VCPConfig())
        result = analyzer.run()
        print(result)  # dict of latest-bar readings, table-row style
        df = analyzer.df  # full dataframe with all computed columns
    """

    def __init__(self, symbol: str, df: pd.DataFrame = None, benchmark_df: pd.DataFrame = None, config: VCPConfig = None, period: str = "2y", interval: str = "1d"):
        self.symbol = symbol
        self.cfg = config if config is not None else VCPConfig()
        self.period = period
        self.interval = interval
        self.df: pd.DataFrame = df
        self.benchmark_df: pd.DataFrame = benchmark_df

    # ---- calculations -------------------------------------------------

    def _load_data(self):
        if self.df is not None and not self.df.empty:
            return
        df = _fetch_history(self.symbol, self.period, self.interval)
        if df.empty:
            raise ValueError(f"No data returned for {self.symbol}")
        self.df = df

    def _moving_averages(self):
        df = self.df
        c = self.cfg
        df["ma50"] = df["Close"].rolling(c.ma50_len).mean()
        df["ma150"] = df["Close"].rolling(c.ma150_len).mean()
        df["ma200"] = df["Close"].rolling(c.ma200_len).mean()

    def _trend_template(self):
        df = self.df
        close = df["Close"]

        high_52wk = df["High"].rolling(252).max()
        low_52wk = df["Low"].rolling(252).min()

        c1 = (close > df["ma150"]) & (close > df["ma200"])
        c2 = df["ma150"] > df["ma200"]
        c3 = df["ma200"] > df["ma200"].shift(20)          # 200MA trending up
        c4 = (df["ma50"] > df["ma150"]) & (df["ma50"] > df["ma200"])
        c5 = close > df["ma50"]
        c6 = close > low_52wk * 1.25                       # 25% above 52wk low
        c7 = close > high_52wk * 0.75                       # within 25% of 52wk high

        df["tpr_pass"] = c1 & c2 & c3 & c4 & c5 & c6 & c7
        df["tpr_txt"] = np.where(df["tpr_pass"], "PASSED", "WAIT")

    def _buy_risk(self):
        df = self.df
        c = self.cfg
        dist_50 = (df["Close"] - df["ma50"]) / df["ma50"] * 100
        df["dist_50"] = dist_50

        def status(x):
            if pd.isna(x):
                return None
            if x < 0:
                return "Broken"
            if x <= c.risk_low:
                return "Low Risk"
            if x <= c.risk_caution:
                return "Caution"
            return "Extended"

        df["risk_status"] = dist_50.apply(status)

    def _pressure(self):
        df = self.df
        c = self.cfg
        up_vol = np.where(df["Close"] > df["Open"], df["Volume"], 0)
        down_vol = np.where(df["Close"] <= df["Open"], df["Volume"], 0)

        buy_vol = pd.Series(up_vol, index=df.index).rolling(c.pressure_window).sum()
        sell_vol = pd.Series(down_vol, index=df.index).rolling(c.pressure_window).sum()

        df["pressure_val"] = buy_vol > sell_vol
        df["pressure_txt"] = np.where(df["pressure_val"], "Buying", "Selling")

    def _rpr(self):
        """Relative Price Strength proxy vs benchmark, weighted ROC blend."""
        df = self.df
        c = self.cfg
        if self.benchmark_df is not None and not self.benchmark_df.empty:
            bm = self.benchmark_df
        else:
            bm = _fetch_history(c.benchmark, self.period, self.interval)
        
        if bm.empty:
            df["rpr_proxy"] = 50.0  # fallback
            return
            
        if isinstance(bm.columns, pd.MultiIndex):
            bm.columns = bm.columns.get_level_values(0)
        bm_close = bm["Close"].reindex(df.index).ffill()

        close = df["Close"]

        sym_roc3 = _roc(close, 63)
        sym_roc6 = _roc(close, 126)
        sym_roc9 = _roc(close, 189)
        sym_roc12 = _roc(close, 252)

        bm_roc3 = _roc(bm_close, 63)
        bm_roc6 = _roc(bm_close, 126)
        bm_roc9 = _roc(bm_close, 189)
        bm_roc12 = _roc(bm_close, 252)

        rs_raw = sym_roc3 * 0.4 + sym_roc6 * 0.2 + sym_roc9 * 0.2 + sym_roc12 * 0.2
        bm_raw = bm_roc3 * 0.4 + bm_roc6 * 0.2 + bm_roc9 * 0.2 + bm_roc12 * 0.2

        rpr_calc = 50 + rs_raw - bm_raw
        df["rpr_proxy"] = rpr_calc.clip(lower=1, upper=99)

    def _vcp(self):
        df = self.df
        c = self.cfg
        vcp_high = df["Close"].rolling(c.vcp_lookback).max()
        vcp_low = df["Close"].rolling(c.vcp_lookback).min()
        vcp_range_pct = (vcp_high - vcp_low) / df["Close"] * 100

        df["vcp_range_pct"] = vcp_range_pct
        df["vcp_trigger"] = vcp_range_pct < c.vcp_thresh
        df["vcp_txt"] = np.where(df["vcp_trigger"], "SQUEEZE", "Normal")

    def _entry_signals(self):
        """
        Two-stage entry logic on top of the VCP squeeze:

        1. Early Entry Zone - the setup is still forming. Range is getting
           tighter bar-over-bar AND volume is drying up. Fires before the
           breakout, while price is still coiling inside the base.

        2. Breakout Confirmed - the trigger. Close breaks above the recent
           contraction high, on volume expansion, while TPR still passes.
        """
        df = self.df
        c = self.cfg

        vol_avg20 = df["Volume"].rolling(c.pressure_window).mean()

        range_contracting = df["vcp_range_pct"] < df["vcp_range_pct"].shift(c.contraction_lookback)
        vol_dryup = df["Volume"] < (vol_avg20 * c.vol_dryup_ratio)

        df["early_entry_zone"] = df["vcp_trigger"] & range_contracting & vol_dryup

        contraction_high = df["Close"].shift(1).rolling(c.vcp_lookback).max()
        vol_expansion = df["Volume"] > (vol_avg20 * c.breakout_vol_mult)
        breaking_out = df["Close"] > contraction_high

        df["breakout_confirmed"] = breaking_out & vol_expansion & df["tpr_pass"]

        conditions = [df["breakout_confirmed"], df["early_entry_zone"]]
        choices = ["BREAKOUT", "EARLY ENTRY"]
        df["entry_signal"] = np.select(conditions, choices, default="-")

    # ---- public API -----------------------------------------------------

    def run(self) -> dict:
        """Fetch, compute everything, and return the latest-bar summary
        (mirrors the Pine script's table output row-for-row)."""
        self._load_data()
        self._moving_averages()
        self._trend_template()
        self._buy_risk()
        self._pressure()
        self._rpr()
        self._vcp()
        self._entry_signals()

        last = self.df.iloc[-1]
        return {
            "symbol": self.symbol,
            "date": self.df.index[-1].strftime("%Y-%m-%d"),
            "close": round(float(last["Close"]), 2),
            "Pressure": last["pressure_txt"],
            "Risk (50d)": last["risk_status"],
            "Trend (TPR)": last["tpr_txt"],
            "RS Rating": round(float(last["rpr_proxy"]), 1) if pd.notna(last["rpr_proxy"]) else None,
            "VCP (5d)": last["vcp_txt"],
            "VCP range %": round(float(last["vcp_range_pct"]), 2) if pd.notna(last["vcp_range_pct"]) else None,
            "Entry Signal": last["entry_signal"],
        }


# ==========================================
# 4. BATCH SCAN HELPER (for your multi-tab scanner)
# ==========================================

def scan_symbols(symbols: list[str], config: VCPConfig = None) -> pd.DataFrame:
    """Run the analyzer across a watchlist and return one row per symbol.
    Errors on individual symbols are captured, not raised, so one bad
    ticker doesn't kill the whole scan."""
    rows = []
    for sym in symbols:
        try:
            result = MinerviniVCPAnalyzer(sym, config).run()
        except Exception as e:
            result = {"symbol": sym, "error": str(e)}
        rows.append(result)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick smoke test
    cfg = VCPConfig(benchmark="^NSEI")
    out = MinerviniVCPAnalyzer("RELIANCE.NS", cfg).run()
    for k, v in out.items():
        print(f"{k}: {v}")
"""
tab_atlas_momentum.py
Dedicated Chartink-Atlas-style 1M % & 3M % Momentum Leaderboard across NSE Top-1000.
"""

import pandas as pd
import streamlit as st

from data_fetcher import get_top1000_nse_symbols, get_stock_sector
from local_cache_manager import bulk_get_cached_ohlcv
from breadth_ledger import get_stock_cluster_tag
from trading_rules import evaluate_manas_trend_rules
from utils import get_day_change_badge_html


def _get_nifty_3m_return() -> float:
    try:
        import yfinance as yf
        nsei = yf.download("^NSEI", period="6mo", interval="1d", progress=False, timeout=10)
        if nsei is not None and len(nsei) >= 64:
            c = nsei["Close"].squeeze().dropna()
            if len(c) >= 64 and float(c.iloc[-64]) > 0:
                return ((float(c.iloc[-1]) - float(c.iloc[-64])) / float(c.iloc[-64])) * 100.0
    except Exception:
        pass
    return 0.0


def compute_atlas_momentum_rows(min_1m: float = 0.0, min_3m: float = 0.0) -> list[dict]:
    """Compute 1M%, 3M%, 6M%, 1M high drawdown, RS vs NIFTY, and day change % for NSE Top-1000."""
    symbols = [s.strip().upper().replace(".NS", "") for s in get_top1000_nse_symbols() if s.strip()]
    bulk_data = bulk_get_cached_ohlcv(symbols, "1d")
    bench_3m = _get_nifty_3m_return()

    rows = []
    for sym in symbols:
        df = bulk_data.get(sym)
        if df is None or len(df) < 65:
            continue
        df = df.sort_values("Date").reset_index(drop=True)
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        vol = df["Volume"].astype(float)

        cmp_val = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        if cmp_val < 10.0 or prev_close <= 0:
            continue

        avg_vol_20 = float(vol.iloc[-20:].mean())
        if avg_vol_20 < 20000:
            continue

        c_1m = float(close.iloc[-22]) if len(close) >= 22 else float(close.iloc[0])
        c_3m = float(close.iloc[-64]) if len(close) >= 64 else float(close.iloc[0])
        c_6m = float(close.iloc[-127]) if len(close) >= 127 else float(close.iloc[0])
        if c_1m <= 0 or c_3m <= 0 or c_6m <= 0:
            continue

        ret_1m = round(((cmp_val - c_1m) / c_1m) * 100.0, 2)
        ret_3m = round(((cmp_val - c_3m) / c_3m) * 100.0, 2)
        ret_6m = round(((cmp_val - c_6m) / c_6m) * 100.0, 2)

        if ret_1m < min_1m or ret_3m < min_3m:
            continue

        day_chg = round(((cmp_val - prev_close) / prev_close) * 100.0, 2)
        high_1m = float(high.iloc[-21:].max())
        off_1m_high = round(((cmp_val - high_1m) / high_1m) * 100.0, 2) if high_1m > 0 else 0.0
        vol_ratio = round(float(vol.iloc[-1]) / avg_vol_20, 2) if avg_vol_20 > 0 else 1.0
        m_rules = evaluate_manas_trend_rules(df, benchmark_3m_ret=bench_3m)

        rows.append({
            "symbol": sym,
            "sector": get_stock_sector(sym),
            "cluster_tag": get_stock_cluster_tag(sym) or "",
            "cmp": round(cmp_val, 2),
            "day_change_pct": day_chg,
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "ret_6m": ret_6m,
            "off_1m_high": off_1m_high,
            "vol_ratio": vol_ratio,
            "days_above_50ma": m_rules["days_above_50ma"],
            "closes_below_50ma": m_rules["closes_below_50ma"],
            "rs_vs_nifty": m_rules["rs_vs_nifty"],
            "extension_label": m_rules["extension_label"],
            "trail_ma_label": m_rules["trail_ma_label"],
        })

    rows.sort(key=lambda x: (x["ret_3m"], x["ret_1m"]), reverse=True)
    return rows


def render():
    st.markdown("### 📊 1M & 3M Momentum Atlas (Chartink Style)")
    st.markdown(
        "<p style='font-size:0.9rem; color:#94a3b8; margin-top:-6px;'>"
        "Dedicated leaderboard tracking <b>1-Month % Change</b>, <b>3-Month % Change</b>, <b>6-Month % Change</b>, "
        "and <b>Drawdown from 1M High</b> across the NSE Top-1000 universe."
        "</p>",
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4, f5 = st.columns([2, 1.5, 1.5, 1.5, 1.5])
    with f1:
        search_q = st.text_input("🔍 Search Stock / Sector:", "", placeholder="e.g. PHARMA, SAILIFE...", key="atlas_search").strip().upper()
    with f2:
        min_1m = st.number_input("Min 1M %:", value=5.0, step=5.0, key="atlas_min_1m")
    with f3:
        min_3m = st.number_input("Min 3M %:", value=15.0, step=5.0, key="atlas_min_3m")
    with f4:
        sort_by = st.selectbox("Sort By:", ["3M % ↓", "1M % ↓", "6M % ↓", "Day Chg % ↓", "Nearest 1M High"], key="atlas_sort")
    with f5:
        top_n = st.selectbox("Show Rows:", [50, 100, 250, 500], index=0, key="atlas_topn")

    only_est_trend = st.checkbox(
        "🛡️ Established Trend Only: Never / barely closed below 50 SMA in last 50 days (≤ 2 closes below 50MA)",
        value=False,
        key="atlas_est_50ma",
    )

    if st.button("🔄 Refresh Atlas Data", key="atlas_refresh_btn") or "atlas_rows_cache" not in st.session_state:
        with st.spinner("Computing 1M% & 3M% momentum across NSE Top-1000..."):
            st.session_state["atlas_rows_cache"] = compute_atlas_momentum_rows(min_1m=-999.0, min_3m=-999.0)

    all_rows = st.session_state.get("atlas_rows_cache", [])
    filtered = [
        r for r in all_rows
        if r["ret_1m"] >= min_1m and r["ret_3m"] >= min_3m
        and (not only_est_trend or r.get("closes_below_50ma", 50) <= 2)
        and (not search_q or search_q in r["symbol"] or search_q in r["sector"].upper())
    ]

    if sort_by == "1M % ↓":
        filtered.sort(key=lambda x: x["ret_1m"], reverse=True)
    elif sort_by == "6M % ↓":
        filtered.sort(key=lambda x: x["ret_6m"], reverse=True)
    elif sort_by == "Day Chg % ↓":
        filtered.sort(key=lambda x: x["day_change_pct"], reverse=True)
    elif sort_by == "Nearest 1M High":
        filtered.sort(key=lambda x: x["off_1m_high"], reverse=True)
    else:
        filtered.sort(key=lambda x: x["ret_3m"], reverse=True)

    disp = filtered[:top_n]
    st.caption(f"Showing **{len(disp)}** of **{len(filtered)}** matching stocks (from {len(all_rows)} scanned).")

    if not disp:
        st.info("No stocks match the current 1M% / 3M% filters.")
        return

    c_dl1, c_dl2, _ = st.columns([1.5, 2, 5])
    with c_dl1:
        st.download_button(
            "📥 Export CSV",
            pd.DataFrame(disp).to_csv(index=False).encode("utf-8"),
            file_name="atlas_1m_3m_momentum.csv",
            mime="text/csv",
            key="atlas_csv_btn",
        )
    with c_dl2:
        tv_str = ",".join([f"NSE:{r['symbol']}" for r in disp[:100]])
        st.download_button(
            "📋 TradingView List (.txt)",
            tv_str.encode("utf-8"),
            file_name="atlas_1m_3m_tv.txt",
            mime="text/plain",
            key="atlas_tv_btn",
        )

    rows_html = []
    for idx, r in enumerate(disp, 1):
        sym = r["symbol"]
        cl_tag = (
            f'<span style="margin-left:5px;padding:1px 5px;background:rgba(251,191,36,0.16);color:#fbbf24;'
            f'border:1px solid rgba(251,191,36,0.4);border-radius:3px;font-size:0.68rem;font-weight:700;">✦ {r["cluster_tag"]}</span>'
            if r["cluster_tag"] else ""
        )
        c1m = "#00e676" if r["ret_1m"] >= 20 else ("#38bdf8" if r["ret_1m"] >= 0 else "#ef4444")
        c3m = "#00e676" if r["ret_3m"] >= 35 else ("#22d3ee" if r["ret_3m"] >= 0 else "#ef4444")
        c6m = "#00e676" if r["ret_6m"] >= 50 else ("#cbd5e1" if r["ret_6m"] >= 0 else "#ef4444")
        rows_html.append(
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<td style="padding:8px;text-align:center;color:#64748b;font-weight:600;">{idx}</td>'
            f'<td style="padding:8px;"><a href="https://in.tradingview.com/chart/?symbol=NSE:{sym}" target="_blank" '
            f'style="color:#38bdf8;font-weight:800;text-decoration:none;">{sym} ↗</a>{cl_tag}'
            f'<div style="font-size:0.71rem;color:#64748b;">{r["sector"]}</div></td>'
            f'<td style="padding:8px;text-align:right;font-weight:700;color:#f8fafc;">₹{r["cmp"]:,.2f}</td>'
            f'<td style="padding:8px;text-align:center;">{get_day_change_badge_html(r["day_change_pct"])}</td>'
            f'<td style="padding:8px;text-align:center;font-weight:800;color:{c1m};">{r["ret_1m"]:+.2f}%</td>'
            f'<td style="padding:8px;text-align:center;font-weight:800;color:{c3m};">{r["ret_3m"]:+.2f}%</td>'
            f'<td style="padding:8px;text-align:center;font-weight:700;color:{c6m};">{r["ret_6m"]:+.2f}%</td>'
            f'<td style="padding:8px;text-align:center;color:#fca5a5;">{r["off_1m_high"]:.2f}%</td>'
            f'<td style="padding:8px;text-align:center;color:#00e676;font-weight:700;">+{r.get("rs_vs_nifty", 0):.1f}%</td>'
            f'<td style="padding:8px;text-align:center;color:#00e676;font-weight:700;">{r.get("days_above_50ma", 0)}/50</td>'
            f'<td style="padding:8px;text-align:center;font-size:0.76rem;">{r.get("extension_label", "—")}</td>'
            f'<td style="padding:8px;text-align:center;color:#94a3b8;">{r["vol_ratio"]:.2f}x</td>'
            f'</tr>'
        )

    st.markdown(
        '<div style="overflow-x:auto;border:1px solid rgba(255,255,255,0.1);border-radius:10px;background:rgba(9,13,22,0.92);">'
        '<table style="width:100%;border-collapse:collapse;font-size:0.84rem;">'
        '<thead><tr style="background:rgba(255,255,255,0.04);border-bottom:1px solid rgba(255,255,255,0.12);'
        'color:#94a3b8;text-transform:uppercase;font-size:0.72rem;">'
        '<th style="padding:10px 8px;text-align:center;">#</th>'
        '<th style="padding:10px 8px;text-align:left;">Stock</th>'
        '<th style="padding:10px 8px;text-align:right;">CMP</th>'
        '<th style="padding:10px 8px;text-align:center;">Day Chg %</th>'
        '<th style="padding:10px 8px;text-align:center;color:#38bdf8;">1M %</th>'
        '<th style="padding:10px 8px;text-align:center;color:#22d3ee;">3M %</th>'
        '<th style="padding:10px 8px;text-align:center;">6M %</th>'
        '<th style="padding:10px 8px;text-align:center;">Off 1M High</th>'
        '<th style="padding:10px 8px;text-align:center;color:#00e676;">RS vs NIFTY</th>'
        '<th style="padding:10px 8px;text-align:center;">50MA (50d)</th>'
        '<th style="padding:10px 8px;text-align:center;">50MA Ext</th>'
        '<th style="padding:10px 8px;text-align:center;">Vol Ratio</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table></div>',
        unsafe_allow_html=True,
    )

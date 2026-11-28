"""
tab_swing_finalists.py
Top-10 Swing Trade Finalists (Manas Arora / Chartink Atlas 1M% & 3M% Momentum + VCP Pullback).
Selects up to 10 high-conviction swing candidates that showed strong 1M/3M leadership
and have now pulled back quietly on contracting volatility (VCP) and dry volume.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import get_company_name
from data_fetcher import get_top1000_nse_symbols, get_stock_sector
from local_cache_manager import bulk_get_cached_ohlcv
from breadth_ledger import get_stock_cluster_tag
from utils import get_day_change_badge_html
import watchlist


def score_swing_candidate(sym: str, df: pd.DataFrame, min_1m_pct: float = 4.0, min_3m_pct: float = 10.0) -> dict | None:
    """
    Evaluate a single stock's daily OHLCV DataFrame for Manas Arora / VCP swing readiness:
      1. Prior momentum leadership (1M% and 3M% change).
      2. Healthy Stage-2 trend (CMP > 50 SMA * 0.98 and CMP > 200 SMA, 50 SMA > 200 SMA).
      3. Low / resting pullback from 60-day high (not extended, -2% to -22% off high, near 10/21 EMA).
      4. VCP volatility contraction (5d range < 20d range) + Dry Volume (5d avg vol < 20d avg vol).
    Returns a scored dict or None if the stock does not qualify.
    """
    if df is None or len(df) < 130:
        return None

    df = df.sort_values("Date").reset_index(drop=True)
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)

    cmp_val = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    if cmp_val < 20.0 or prev_close <= 0:
        return None

    # Liquidity filter: 20d avg traded value >= 2 Cr (or avg vol >= 50,000)
    avg_vol_20 = float(vol.iloc[-20:].mean())
    avg_vol_5 = float(vol.iloc[-5:].mean())
    if avg_vol_20 < 50000 or (cmp_val * avg_vol_20) < 1e7:
        return None

    day_chg_pct = ((cmp_val - prev_close) / prev_close) * 100.0

    # 1M (21d), 3M (63d), 6M (126d) percentage returns
    c_1m = float(close.iloc[-22]) if len(close) >= 22 else float(close.iloc[0])
    c_3m = float(close.iloc[-64]) if len(close) >= 64 else float(close.iloc[0])
    c_6m = float(close.iloc[-127]) if len(close) >= 127 else float(close.iloc[0])
    if c_1m <= 0 or c_3m <= 0 or c_6m <= 0:
        return None

    ret_1m = ((cmp_val - c_1m) / c_1m) * 100.0
    ret_3m = ((cmp_val - c_3m) / c_3m) * 100.0
    ret_6m = ((cmp_val - c_6m) / c_6m) * 100.0

    # Must have prior momentum leadership (1M% and 3M%)
    if ret_1m < min_1m_pct or ret_3m < min_3m_pct:
        return None

    # Moving averages
    ema10 = float(close.ewm(span=10, adjust=False).mean().iloc[-1])
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.rolling(120).mean().iloc[-1])

    # Stage 2 / uptrend structure
    if cmp_val < sma200 or sma50 < sma200 * 0.99:
        return None
    if cmp_val < sma50 * 0.96:
        return None

    # Pullback from 60-day high (should be resting / pulled back, NOT chasing +15% extended)
    high_60 = float(high.iloc[-60:].max())
    if high_60 <= 0:
        return None
    pullback_pct = ((cmp_val - high_60) / high_60) * 100.0  # negative value e.g. -6.5%

    # Must be in a constructive pullback / tight base (-22% <= pullback <= -1.5%)
    # and not wildly extended above 21 EMA (within -5% to +6% of 21 EMA)
    dist_ema21_pct = ((cmp_val - ema21) / ema21) * 100.0 if ema21 > 0 else 0.0
    if pullback_pct > -1.5 or pullback_pct < -22.0:
        return None
    if dist_ema21_pct > 6.5 or dist_ema21_pct < -5.5:
        return None

    # VCP Volatility Contraction: compare recent 5-day avg daily range % vs prior 20-day avg daily range %
    daily_range_pct = ((high - low) / close.replace(0, np.nan)) * 100.0
    range_5d = float(daily_range_pct.iloc[-5:].mean())
    range_20d = float(daily_range_pct.iloc[-25:-5].mean()) if len(daily_range_pct) >= 25 else float(daily_range_pct.iloc[-20:].mean())
    if range_20d <= 0 or np.isnan(range_5d) or np.isnan(range_20d):
        return None
    contraction_ratio = range_5d / range_20d  # < 1.0 means tightening!

    # Require genuine tightening (VCP contraction <= 0.95)
    if contraction_ratio > 0.95:
        return None

    # Dry Volume on pullback (5d avg vol / 20d avg vol)
    vol_dry_ratio = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else 1.0
    if vol_dry_ratio > 1.05:
        return None

    # Composite Swing Readiness Score (0 - 100)
    # 1. Momentum (1M + 3M leadership): up to 35 pts
    momo_pts = min(20.0, max(0.0, (ret_3m - 10.0) * 0.45)) + min(15.0, max(0.0, (ret_1m - 4.0) * 0.6))
    # 2. VCP Tightness: up to 25 pts (lower contraction_ratio = tighter base)
    vcp_pts = min(25.0, max(0.0, (1.0 - contraction_ratio) * 50.0))
    # 3. Dry Volume: up to 20 pts (lower vol_dry_ratio = quieter pullback)
    dry_pts = min(20.0, max(0.0, (1.05 - vol_dry_ratio) * 35.0))
    # 4. Pullback Sweet Spot (-3% to -12% from high & close to 21 EMA): up to 10 pts
    pb_abs = abs(pullback_pct)
    pb_pts = 10.0 if 3.0 <= pb_abs <= 12.0 else 6.0
    # 5. Institutional Cluster Boost: up to 10 pts
    cluster_tag = get_stock_cluster_tag(sym)
    cluster_pts = 10.0 if cluster_tag else 0.0

    total_score = round(momo_pts + vcp_pts + dry_pts + pb_pts + cluster_pts, 1)

    # Trade Levels: Buy around CMP / 10-21 EMA, SL below recent 10d swing low, Target at 2.5R+ or prior high+
    swing_low_10 = float(low.iloc[-10:].min())
    stop_loss = round(min(swing_low_10 * 0.99, ema21 * 0.975), 2)
    if stop_loss >= cmp_val * 0.985:
        stop_loss = round(cmp_val * 0.95, 2)
    risk = cmp_val - stop_loss
    target_price = round(max(high_60 * 1.08, cmp_val + risk * 2.5), 2)
    rr_ratio = round((target_price - cmp_val) / risk, 1) if risk > 0 else 0.0

    # Setup label
    if contraction_ratio <= 0.65 and vol_dry_ratio <= 0.70:
        verdict = "🎯 Prime VCP Coil"
    elif abs(dist_ema21_pct) <= 2.5 and vol_dry_ratio <= 0.85:
        verdict = "⚡ Tight 21 EMA Pullback"
    else:
        verdict = "🔥 Dry-Vol Base"

    return {
        "symbol": sym,
        "company_name": get_company_name(sym),
        "sector": get_stock_sector(sym),
        "cluster_tag": cluster_tag or "",
        "cmp": round(cmp_val, 2),
        "day_change_pct": round(day_chg_pct, 2),
        "ret_1m": round(ret_1m, 2),
        "ret_3m": round(ret_3m, 2),
        "ret_6m": round(ret_6m, 2),
        "pullback_pct": round(pullback_pct, 2),
        "dist_ema21_pct": round(dist_ema21_pct, 2),
        "range_5d": round(range_5d, 2),
        "contraction_ratio": round(contraction_ratio, 2),
        "vol_dry_ratio": round(vol_dry_ratio, 2),
        "ema10": round(ema10, 2),
        "ema21": round(ema21, 2),
        "sma50": round(sma50, 2),
        "high_60": round(high_60, 2),
        "stop_loss": stop_loss,
        "target_price": target_price,
        "rr_ratio": rr_ratio,
        "swing_score": total_score,
        "verdict": verdict,
    }


def find_top10_swing_finalists(min_1m_pct: float = 4.0, min_3m_pct: float = 10.0, max_results: int = 10) -> list[dict]:
    """Scan Top 1000 NSE stocks from local 1d OHLCV cache and return top N swing finalists."""
    symbols = [s.strip().upper().replace(".NS", "") for s in get_top1000_nse_symbols() if str(s).strip()]
    bulk_data = bulk_get_cached_ohlcv(symbols, "1d")

    candidates = []
    for sym in symbols:
        df = bulk_data.get(sym)
        if df is None or df.empty:
            continue
        scored = score_swing_candidate(sym, df, min_1m_pct=min_1m_pct, min_3m_pct=min_3m_pct)
        if scored:
            candidates.append(scored)

    candidates.sort(key=lambda x: x["swing_score"], reverse=True)
    return candidates[:max_results]


def _render_chart(sym: str, candidate: dict):
    """Render interactive Plotly Candlestick + 10/21/50 EMA + Volume chart for a finalist."""
    bulk = bulk_get_cached_ohlcv([sym], "1d")
    df = bulk.get(sym)
    if df is None or df.empty:
        st.warning(f"No historical chart data cached for {sym}.")
        return

    df = df.sort_values("Date").tail(120).copy()
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )
    fig.add_trace(
        go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name=sym, increasing_line_color="#00e676", decreasing_line_color="#ef4444",
        ),
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA10"], name="10 EMA", line=dict(color="#38bdf8", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA21"], name="21 EMA", line=dict(color="#f59e0b", width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="50 SMA", line=dict(color="#a855f7", width=1.6, dash="dot")), row=1, col=1)

    fig.add_hline(y=candidate["stop_loss"], line_dash="dash", line_color="#ef4444", annotation_text=f"SL ₹{candidate['stop_loss']}", row=1, col=1)
    fig.add_hline(y=candidate["target_price"], line_dash="dash", line_color="#00e676", annotation_text=f"Target ₹{candidate['target_price']}", row=1, col=1)

    vol_colors = ["#00e676" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color=vol_colors, opacity=0.75), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=460,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.5)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render():
    st.markdown("### 🏆 Top 10 Swing Finalists (Manas Arora / Atlas 1M% & 3M% + VCP Pullback)")
    st.markdown(
        "<p style='font-size:0.9rem; color:#94a3b8; margin-top:-6px;'>"
        "Filters the NSE Top-1000 universe for <b>prior 1M &amp; 3M momentum leaders</b> that have now <b>pulled back quietly</b> "
        "near their 10/21 EMA on <b>contracting volatility (VCP)</b> and <b>dry volume</b> — capped at the <b>Top 10 actionable swing finalists</b>."
        "</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        min_1m = st.slider("Min 1M % Change", min_value=0.0, max_value=25.0, value=4.0, step=1.0, key="sf_min_1m")
    with c2:
        min_3m = st.slider("Min 3M % Change", min_value=0.0, max_value=50.0, value=10.0, step=2.0, key="sf_min_3m")
    with c3:
        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
        refresh_clicked = st.button("🔄 Refresh Top 10 Finalists", key="sf_refresh_btn", use_container_width=True)

    cache_key = f"swing_finalists_{min_1m}_{min_3m}"
    if refresh_clicked or cache_key not in st.session_state:
        with st.spinner("Scanning cached 1M/3M momentum leaders & VCP contractions..."):
            st.session_state[cache_key] = find_top10_swing_finalists(min_1m_pct=min_1m, min_3m_pct=min_3m, max_results=10)

    finalists = st.session_state.get(cache_key, [])

    if not finalists:
        st.info("No stocks currently meet all 1M%/3M% + VCP pullback criteria. Try lowering the Min 1M% or Min 3M% sliders above.")
        return

    # Summary KPIs
    k1, k2, k3, k4 = st.columns(4)
    avg_1m = sum(f["ret_1m"] for f in finalists) / len(finalists)
    avg_3m = sum(f["ret_3m"] for f in finalists) / len(finalists)
    avg_vcp = sum(f["contraction_ratio"] for f in finalists) / len(finalists)
    k1.metric("Finalists Selected", f"{len(finalists)} / 10")
    k2.metric("Avg 1M Return", f"{avg_1m:+.1f}%")
    k3.metric("Avg 3M Return", f"{avg_3m:+.1f}%")
    k4.metric("Avg VCP Contraction", f"{avg_vcp:.2f}x", help="< 0.75x = tight VCP contraction")

    # Export / Copy helpers
    exp_df = pd.DataFrame(finalists)
    e1, e2, _ = st.columns([1.5, 1.8, 5])
    with e1:
        st.download_button(
            "📥 Export CSV",
            exp_df.to_csv(index=False).encode("utf-8"),
            file_name="top10_swing_finalists.csv",
            mime="text/csv",
            key="sf_csv_dl",
        )
    with e2:
        tv_txt = ",".join([f"NSE:{f['symbol']}" for f in finalists])
        st.download_button(
            "📋 TradingView Watchlist (.txt)",
            tv_txt.encode("utf-8"),
            file_name="top10_swing_finalists_tv.txt",
            mime="text/plain",
            key="sf_tv_dl",
        )

    # Render Atlas-style HTML table for the Top 10 Finalists
    rows_html = []
    for idx, r in enumerate(finalists, 1):
        sym = r["symbol"]
        cluster_badge = (
            f'<span style="display:inline-block;margin-left:4px;padding:1px 5px;background:rgba(251,191,36,0.18);'
            f'color:#fbbf24;border:1px solid rgba(251,191,36,0.4);border-radius:3px;font-size:0.68rem;font-weight:700;">'
            f'✦ {r["cluster_tag"]}</span>' if r["cluster_tag"] else ""
        )
        day_badge = get_day_change_badge_html(r["day_change_pct"])
        m1_color = "#00e676" if r["ret_1m"] >= 15 else "#38bdf8"
        m3_color = "#00e676" if r["ret_3m"] >= 30 else "#22d3ee"

        rows_html.append(
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.06);">'
            f'<td style="padding:10px 8px;text-align:center;font-weight:800;color:#fbbf24;">#{idx}</td>'
            f'<td style="padding:10px 8px;">'
            f'<a href="https://in.tradingview.com/chart/?symbol=NSE:{sym}" target="_blank" style="color:#38bdf8;font-weight:800;text-decoration:none;">{sym} ↗</a>'
            f'{cluster_badge}<div style="font-size:0.72rem;color:#64748b;">{r["sector"]}</div></td>'
            f'<td style="padding:10px 8px;text-align:right;font-weight:700;color:#f8fafc;">₹{r["cmp"]:,.2f}</td>'
            f'<td style="padding:10px 8px;text-align:center;">{day_badge}</td>'
            f'<td style="padding:10px 8px;text-align:center;font-weight:800;color:{m1_color};">{r["ret_1m"]:+.1f}%</td>'
            f'<td style="padding:10px 8px;text-align:center;font-weight:800;color:{m3_color};">{r["ret_3m"]:+.1f}%</td>'
            f'<td style="padding:10px 8px;text-align:center;color:#cbd5e1;">{r["ret_6m"]:+.1f}%</td>'
            f'<td style="padding:10px 8px;text-align:center;color:#fca5a5;">{r["pullback_pct"]:.1f}%</td>'
            f'<td style="padding:10px 8px;text-align:center;color:#22d3ee;font-weight:700;">{r["contraction_ratio"]:.2f}x</td>'
            f'<td style="padding:10px 8px;text-align:center;color:#38bdf8;font-weight:700;">{r["vol_dry_ratio"]:.2f}x</td>'
            f'<td style="padding:10px 8px;text-align:right;color:#ef4444;font-weight:600;">₹{r["stop_loss"]:,.2f}</td>'
            f'<td style="padding:10px 8px;text-align:right;color:#00e676;font-weight:600;">₹{r["target_price"]:,.2f}'
            f'<div style="font-size:0.68rem;color:#6ee7b7;">1:{r["rr_ratio"]} R:R</div></td>'
            f'<td style="padding:10px 8px;text-align:center;"><span style="background:rgba(0,230,118,0.15);color:#00e676;'
            f'border:1px solid rgba(0,230,118,0.4);padding:2px 7px;border-radius:4px;font-weight:800;">{r["swing_score"]}</span></td>'
            f'<td style="padding:10px 8px;font-size:0.78rem;font-weight:600;color:#e2e8f0;">{r["verdict"]}</td>'
            f'</tr>'
        )

    table_html = (
        '<div style="overflow-x:auto;border:1px solid rgba(255,255,255,0.1);border-radius:10px;background:rgba(9,13,22,0.92);margin-top:8px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:0.84rem;">'
        '<thead><tr style="background:rgba(255,255,255,0.04);border-bottom:1px solid rgba(255,255,255,0.12);'
        'color:#94a3b8;text-transform:uppercase;font-size:0.72rem;letter-spacing:0.4px;">'
        '<th style="padding:10px 8px;text-align:center;">Rank</th>'
        '<th style="padding:10px 8px;text-align:left;">Stock</th>'
        '<th style="padding:10px 8px;text-align:right;">CMP</th>'
        '<th style="padding:10px 8px;text-align:center;">Day Chg %</th>'
        '<th style="padding:10px 8px;text-align:center;color:#38bdf8;">1M %</th>'
        '<th style="padding:10px 8px;text-align:center;color:#22d3ee;">3M %</th>'
        '<th style="padding:10px 8px;text-align:center;">6M %</th>'
        '<th style="padding:10px 8px;text-align:center;">Off High</th>'
        '<th style="padding:10px 8px;text-align:center;">VCP Tight</th>'
        '<th style="padding:10px 8px;text-align:center;">Vol Dry</th>'
        '<th style="padding:10px 8px;text-align:right;">Stop Loss</th>'
        '<th style="padding:10px 8px;text-align:right;">Target</th>'
        '<th style="padding:10px 8px;text-align:center;">Score</th>'
        '<th style="padding:10px 8px;text-align:left;">Setup</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    # Interactive Chart & Watchlist Add for any of the 10 Finalists
    st.markdown("---")
    st.markdown("#### 📈 Finalist Chart & Quick Watchlist")
    sel_col, wl_col = st.columns([3, 1])
    with sel_col:
        sel_sym = st.selectbox(
            "Select Finalist to Inspect VCP Structure:",
            [f["symbol"] for f in finalists],
            key="sf_chart_select",
        )
    selected_cand = next((f for f in finalists if f["symbol"] == sel_sym), finalists[0])
    with wl_col:
        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
        if st.button(f"➕ Add {sel_sym} to Watchlist", key="sf_add_wl", use_container_width=True):
            watchlist.add_to_watchlist(
                sel_sym,
                selected_cand["cmp"],
                selected_cand["stop_loss"],
                selected_cand["target_price"],
                f"Top-10 Swing Finalist ({selected_cand['verdict']} | 1M: {selected_cand['ret_1m']:+.1f}%, 3M: {selected_cand['ret_3m']:+.1f}%)",
            )
            st.success(f"Added {sel_sym} to Watchlist!")

    _render_chart(sel_sym, selected_cand)

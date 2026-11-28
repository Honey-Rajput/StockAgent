"""
tab_multibagger_base.py
2x / Multibagger Base & Tight Shelf Scanner (GREAVESCOT / WABAG / WELSPUNLIV / CUPID style):
Detects two high-conviction base-breakout & coil structures:
  1. Launchpad Flat Base (WELSPUNLIV / CUPID Box 1 / GREAVESCOT): 20-40d tight consolidation
     near 20/50/200 SMA on dry volume, coiled at box high or breaking out on volume.
  2. High-Tight Flag / Mid-Trend VCP Shelf (CUPID Box 2 / WABAG): >=35% prior run followed by
     a 10-25d tight shelf (<=20% range) on dry volume, ready for the second explosive leg.
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import get_company_name
from data_fetcher import get_top1500_nse_symbols, get_stock_sector
from local_cache_manager import bulk_get_cached_ohlcv
from breadth_ledger import get_stock_cluster_tag
from trading_rules import evaluate_manas_trend_rules
from utils import get_day_change_badge_html
import watchlist


def detect_multibagger_base(
    sym: str,
    df: pd.DataFrame,
    max_box_range_pct: float = 22.0,
    min_box_days: int = 15,
) -> dict | None:
    """
    Evaluate a stock for a 2x Launchpad Base or High-Tight-Flag Shelf.
    Returns scored dict with box coordinates (box_start_idx, box_low, box_high) or None.
    """
    if df is None or len(df) < 130:
        return None

    df = df.sort_values("Date").reset_index(drop=True)
    open_s = df["Open"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    cmp_val = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    if cmp_val < 20.0 or prev_close <= 0:
        return None

    avg_vol_50 = float(vol.iloc[-50:].mean())
    if avg_vol_50 < 40000 or (cmp_val * avg_vol_50) < 1e7:
        return None

    day_chg_pct = ((cmp_val - prev_close) / prev_close) * 100.0

    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.rolling(120).mean().iloc[-1])
    # Must be in constructive Stage-2 / above 200 SMA and near/above 50 SMA
    if cmp_val < sma200 or cmp_val < sma50 * 0.96:
        return None

    # Search best tight base window ending yesterday (15 to 35 trading days)
    best_box = None
    for win in (20, 25, 30, 15):
        if win < min_box_days or len(df) < win + 25:
            continue
        box_slice = df.iloc[-(win + 1):-1]
        b_high = float(box_slice["High"].max())
        b_low = float(box_slice["Low"].min())
        if b_low <= 0:
            continue
        b_range_pct = ((b_high - b_low) / b_low) * 100.0
        if b_range_pct <= max_box_range_pct:
            # Check last 5 days inside box are even tighter
            last5_h = float(df["High"].iloc[-6:-1].max())
            last5_l = float(df["Low"].iloc[-6:-1].min())
            last5_pct = ((last5_h - last5_l) / last5_l) * 100.0 if last5_l > 0 else b_range_pct
            if best_box is None or b_range_pct < best_box["box_range_pct"]:
                best_box = {
                    "box_days": win,
                    "box_high": b_high,
                    "box_low": b_low,
                    "box_range_pct": b_range_pct,
                    "last5_range_pct": last5_pct,
                    "box_start_date": str(box_slice["Date"].iloc[0])[:10],
                    "box_end_date": str(box_slice["Date"].iloc[-1])[:10],
                    "box_vol_avg": float(box_slice["Volume"].mean()),
                }

    if best_box is None:
        return None

    b_high = best_box["box_high"]
    b_low = best_box["box_low"]
    b_days = best_box["box_days"]

    # Price must be either breaking out above box_high (up to +8%) or coiled near top of box (within -6% of box_high)
    dist_to_pivot_pct = ((cmp_val - b_high) / b_high) * 100.0
    if dist_to_pivot_pct < -6.5 or dist_to_pivot_pct > 8.5:
        return None

    # Volume dry-up inside the base vs prior 50d average (sellers exhausted)
    vol_dry_ratio = best_box["box_vol_avg"] / avg_vol_50 if avg_vol_50 > 0 else 1.0
    if vol_dry_ratio > 1.05:
        return None

    today_vol_ratio = float(vol.iloc[-1]) / best_box["box_vol_avg"] if best_box["box_vol_avg"] > 0 else 1.0

    # Prior 6M run-up before the base
    pre_base_low = float(low.iloc[-126:-(b_days + 1)].min()) if len(df) >= 126 else float(low.iloc[:-(b_days + 1)].min())
    prior_run_pct = ((b_high - pre_base_low) / pre_base_low) * 100.0 if pre_base_low > 0 else 0.0

    # Classify setup style matching user's 4 examples
    is_breakout = cmp_val >= b_high * 0.995 and today_vol_ratio >= 1.25 and cmp_val > float(open_s.iloc[-1])
    if prior_run_pct >= 35.0:
        setup_style = "🚩 High-Tight Flag (2nd Leg)" if not is_breakout else "🚀 Flag Breakout (2nd Leg)"
    else:
        setup_style = "🚀 Launchpad Breakout" if is_breakout else "📦 Tight Shelf Coil"

    # Score (0 - 100)
    tight_pts = min(30.0, max(0.0, (max_box_range_pct - best_box["box_range_pct"]) * 1.8))
    dry_pts = min(25.0, max(0.0, (1.05 - vol_dry_ratio) * 40.0))
    brk_pts = 25.0 if is_breakout else max(0.0, 15.0 - abs(dist_to_pivot_pct) * 2.0)
    run_pts = min(10.0, max(0.0, prior_run_pct * 0.15))
    cluster_tag = get_stock_cluster_tag(sym)
    cluster_pts = 10.0 if cluster_tag else 0.0
    score = round(tight_pts + dry_pts + brk_pts + run_pts + cluster_pts, 1)

    stop_loss = round(max(b_low * 0.985, sma50 * 0.97), 2)
    if stop_loss >= cmp_val * 0.98:
        stop_loss = round(cmp_val * 0.93, 2)
    risk = cmp_val - stop_loss
    target_price = round(max(cmp_val + risk * 3.0, cmp_val * 1.30), 2)
    m_rules = evaluate_manas_trend_rules(df, base_pivot=b_high)

    return {
        "symbol": sym,
        "company_name": get_company_name(sym),
        "sector": get_stock_sector(sym),
        "cluster_tag": cluster_tag or "",
        "cmp": round(cmp_val, 2),
        "day_change_pct": round(day_chg_pct, 2),
        "setup_style": setup_style,
        "is_breakout": is_breakout,
        "box_days": b_days,
        "box_range_pct": round(best_box["box_range_pct"], 2),
        "last5_range_pct": round(best_box["last5_range_pct"], 2),
        "box_high": round(b_high, 2),
        "box_low": round(b_low, 2),
        "box_start_date": best_box["box_start_date"],
        "box_end_date": best_box["box_end_date"],
        "dist_to_pivot_pct": round(dist_to_pivot_pct, 2),
        "vol_dry_ratio": round(vol_dry_ratio, 2),
        "today_vol_ratio": round(today_vol_ratio, 2),
        "prior_run_pct": round(prior_run_pct, 1),
        "stop_loss": stop_loss,
        "target_price": target_price,
        "score": score,
        "days_above_50ma": m_rules["days_above_50ma"],
        "bar_timing_label": m_rules["bar_timing_label"],
        "trail_ma_label": m_rules["trail_ma_label"],
    }


def scan_multibagger_bases(
    max_box_range_pct: float = 22.0,
    min_box_days: int = 15,
    max_price: float = 1000.0,
    max_results: int = 30,
) -> list[dict]:
    """Scan NSE Top-1000 from local 1d cache for 2x Launchpad & High-Tight-Flag bases."""
    symbols = [s.strip().upper().replace(".NS", "") for s in get_top1500_nse_symbols() if s.strip()]
    bulk_data = bulk_get_cached_ohlcv(symbols, "1d")

    hits = []
    for sym in symbols:
        df = bulk_data.get(sym)
        if df is None or df.empty:
            continue
        res = detect_multibagger_base(sym, df, max_box_range_pct=max_box_range_pct, min_box_days=min_box_days)
        if res and (max_price <= 0 or res["cmp"] <= max_price):
            hits.append(res)

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:max_results]


def _render_base_chart(sym: str, item: dict):
    """Render Plotly Candlestick + 50/200 SMA + highlighted Red Base Box + Volume."""
    bulk = bulk_get_cached_ohlcv([sym], "1d")
    df = bulk.get(sym)
    if df is None or df.empty:
        st.warning(f"No chart data cached for {sym}.")
        return

    df = df.sort_values("Date").tail(140).copy()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean() if len(df) >= 200 else df["Close"].rolling(100).mean()

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
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="50 SMA", line=dict(color="#f59e0b", width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA200"], name="200 SMA", line=dict(color="#38bdf8", width=1.5, dash="dot")), row=1, col=1)

    # Highlight the tight consolidation box (just like the red boxes in your examples)
    fig.add_shape(
        type="rect",
        x0=item["box_start_date"], x1=item["box_end_date"],
        y0=item["box_low"], y1=item["box_high"],
        line=dict(color="#ef4444", width=2),
        fillcolor="rgba(239, 68, 68, 0.10)",
        row=1, col=1,
    )
    fig.add_hline(y=item["box_high"], line_dash="dash", line_color="#22d3ee", annotation_text=f"Pivot ₹{item['box_high']}", row=1, col=1)
    fig.add_hline(y=item["stop_loss"], line_dash="dash", line_color="#ef4444", annotation_text=f"SL ₹{item['stop_loss']}", row=1, col=1)

    vol_colors = ["#00e676" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color=vol_colors, opacity=0.75), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=470,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.5)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render():
    st.markdown("### 💎 2x Multibagger Base & Tight Shelf Scanner")
    st.markdown(
        "<p style='font-size:0.9rem; color:#94a3b8; margin-top:-6px;'>"
        "Finds <b>GREAVESCOT / WABAG / WELSPUNLIV / CUPID</b> style multi-week <b>tight horizontal bases &amp; high-tight flags</b> "
        "on <b>ultra-dry volume</b> that are coiled at the pivot or breaking out for a major move."
        "</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns([1.8, 1.3, 1.3, 1.2, 1.8, 1.3])
    with c1:
        max_range = st.slider("Max Base Box Range (%)", min_value=10.0, max_value=30.0, value=20.0, step=2.0, key="mb_max_range")
    with c2:
        min_days = st.selectbox("Min Base Duration:", [15, 20, 25, 30], index=0, key="mb_min_days")
    with c3:
        max_price = st.number_input("Max Price (₹, 0=All):", min_value=0, value=1000, step=100, key="mb_max_price")
    with c4:
        max_res = st.selectbox("Show Top:", [20, 30, 50, 100], index=1, key="mb_max_res")
    with c5:
        style_filter = st.selectbox("Setup Filter:", ["All Setups", "Breakouts Only", "High-Tight Flags (2nd Leg)"], key="mb_style_filter")
    with c6:
        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
        refresh_btn = st.button("🔄 Scan Bases", key="mb_refresh_btn", use_container_width=True)

    cache_key = f"mb_cache_{max_range}_{min_days}_{max_price}_{max_res}"
    if refresh_btn or cache_key not in st.session_state:
        with st.spinner("Scanning NSE Top-1000 for tight multibagger bases & dry-volume shelves..."):
            st.session_state[cache_key] = scan_multibagger_bases(
                max_box_range_pct=max_range, min_box_days=min_days, max_price=float(max_price), max_results=int(max_res)
            )

    results = st.session_state.get(cache_key, [])
    if style_filter == "Breakouts Only":
        filtered = [r for r in results if r["is_breakout"]]
    elif style_filter == "High-Tight Flags (2nd Leg)":
        filtered = [r for r in results if "Flag" in r["setup_style"]]
    else:
        filtered = results

    if not filtered:
        st.info("No bases currently match the selected filters. Try widening Max Base Box Range or switching Setup Filter to 'All Setups'.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Bases Found", len(filtered))
    k2.metric("Active Breakouts", sum(1 for r in filtered if r["is_breakout"]))
    k3.metric("Avg Box Range", f"{sum(r['box_range_pct'] for r in filtered) / len(filtered):.1f}%")
    k4.metric("Avg Base Vol Dry", f"{sum(r['vol_dry_ratio'] for r in filtered) / len(filtered):.2f}x")

    rows_html = []
    for idx, r in enumerate(filtered, 1):
        sym = r["symbol"]
        cl_tag = (
            f'<span style="margin-left:4px;padding:1px 5px;background:rgba(251,191,36,0.18);color:#fbbf24;'
            f'border:1px solid rgba(251,191,36,0.4);border-radius:3px;font-size:0.68rem;font-weight:700;">✦ {r["cluster_tag"]}</span>'
            if r["cluster_tag"] else ""
        )
        rows_html.append(
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.06);">'
            f'<td style="padding:9px 8px;text-align:center;color:#fbbf24;font-weight:800;">#{idx}</td>'
            f'<td style="padding:9px 8px;"><a href="https://in.tradingview.com/chart/?symbol=NSE:{sym}" target="_blank" '
            f'style="color:#38bdf8;font-weight:800;text-decoration:none;">{sym} ↗</a>{cl_tag}'
            f'<div style="font-size:0.71rem;color:#64748b;">{r["sector"]}</div></td>'
            f'<td style="padding:9px 8px;text-align:right;font-weight:700;color:#f8fafc;">₹{r["cmp"]:,.2f}</td>'
            f'<td style="padding:9px 8px;text-align:center;">{get_day_change_badge_html(r["day_change_pct"])}</td>'
            f'<td style="padding:9px 8px;font-weight:700;color:#e2e8f0;">{r["setup_style"]}</td>'
            f'<td style="padding:9px 8px;text-align:center;color:#cbd5e1;">{r["box_days"]}d ({r["box_range_pct"]:.1f}%)</td>'
            f'<td style="padding:9px 8px;text-align:center;color:#22d3ee;font-weight:700;">{r["vol_dry_ratio"]:.2f}x</td>'
            f'<td style="padding:9px 8px;text-align:center;color:#38bdf8;font-weight:700;">{r["today_vol_ratio"]:.2f}x</td>'
            f'<td style="padding:9px 8px;text-align:center;color:#00e676;font-weight:700;">+{r["prior_run_pct"]:.1f}%</td>'
            f'<td style="padding:9px 8px;text-align:right;color:#ef4444;font-weight:600;">₹{r["stop_loss"]:,.2f}</td>'
            f'<td style="padding:9px 8px;font-size:0.78rem;color:#22d3ee;">{r.get("trail_ma_label", "🛡️ 20 SMA")}</td>'
            f'<td style="padding:9px 8px;text-align:center;color:#00e676;font-weight:700;">{r.get("days_above_50ma", 0)}/50</td>'
            f'<td style="padding:9px 8px;text-align:center;font-size:0.76rem;">{r.get("bar_timing_label", "—")}</td>'
            f'<td style="padding:9px 8px;text-align:center;"><span style="background:rgba(0,230,118,0.15);color:#00e676;'
            f'border:1px solid rgba(0,230,118,0.4);padding:2px 7px;border-radius:4px;font-weight:800;">{r["score"]}</span></td>'
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
        '<th style="padding:10px 8px;text-align:left;">Setup Style</th>'
        '<th style="padding:10px 8px;text-align:center;">Base Box</th>'
        '<th style="padding:10px 8px;text-align:center;">Base Vol Dry</th>'
        '<th style="padding:10px 8px;text-align:center;">Today Vol</th>'
        '<th style="padding:10px 8px;text-align:center;">Prior Run</th>'
        '<th style="padding:10px 8px;text-align:right;">Initial SL</th>'
        '<th style="padding:10px 8px;text-align:left;">Trail Stop (Ride Trend)</th>'
        '<th style="padding:10px 8px;text-align:center;">50MA (50d)</th>'
        '<th style="padding:10px 8px;text-align:center;">Bar Timing</th>'
        '<th style="padding:10px 8px;text-align:center;">Score</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### 📦 Base Box Chart Inspector (Red Box = Detected Tight Shelf)")
    sel_col, wl_col = st.columns([3, 1])
    with sel_col:
        sel_sym = st.selectbox("Select Stock to View Base Box:", [r["symbol"] for r in filtered], key="mb_chart_sel")
    sel_item = next((r for r in filtered if r["symbol"] == sel_sym), filtered[0])
    with wl_col:
        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
        if st.button(f"➕ Add {sel_sym} to Watchlist", key="mb_add_wl", use_container_width=True):
            watchlist.add_to_watchlist(
                sel_sym, sel_item["cmp"], sel_item["stop_loss"], sel_item["target_price"],
                f"2x Base ({sel_item['setup_style']} | {sel_item['box_days']}d box {sel_item['box_range_pct']:.1f}%)",
            )
            st.success(f"Added {sel_sym} to Watchlist!")

    _render_base_chart(sel_sym, sel_item)

"""
tab_swing_finalists.py
Manas Arora Style Swing Trade Finalists (Max 10):
  1. Minimum prior run-up of >= 35% (swing low to swing high over 3M-6M).
  2. Not deep down from its 1-Month high (drawdown <= 25%, i.e. within 0% to -25% of 1M high).
  3. Forming a VCP pattern (volatility contraction + dry volume during the base).
  4. Breaking its recent low / VCP base by opening gap-up (recent 1-5 sessions) and ready for the next leg.
"""

import pandas as pd
import numpy as np
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


def score_swing_candidate(
    sym: str,
    df: pd.DataFrame,
    min_run_pct: float = 35.0,
    max_dd_1m_pct: float = 25.0,
    require_gap_up: bool = True,
    benchmark_3m_ret: float | None = None,
) -> dict | None:
    """
    Evaluate a stock for the Manas Arora VCP + Gap-Up Reversal setup:
      - Prior run-up >= min_run_pct (default 35%) from 6M/3M low to recent high.
      - Drawdown from 1M (21d) high <= max_dd_1m_pct (default 25%).
      - VCP contraction (5d range < 20d range) + dry volume on base.
      - Breaking recent low / base via gap-up open (within last 5 sessions) with bullish close.
    """
    if df is None or len(df) < 130:
        return None

    df = df.sort_values("Date").reset_index(drop=True)
    open_s = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    cmp_val = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    if cmp_val < 20.0 or prev_close <= 0:
        return None

    avg_vol_20 = float(vol.iloc[-20:].mean())
    avg_vol_5 = float(vol.iloc[-5:].mean())
    if avg_vol_20 < 50000 or (cmp_val * avg_vol_20) < 1e7:
        return None

    day_chg_pct = ((cmp_val - prev_close) / prev_close) * 100.0

    c_1m = float(close.iloc[-22]) if len(close) >= 22 else float(close.iloc[0])
    c_3m = float(close.iloc[-64]) if len(close) >= 64 else float(close.iloc[0])
    c_6m = float(close.iloc[-127]) if len(close) >= 127 else float(close.iloc[0])
    ret_1m = ((cmp_val - c_1m) / c_1m) * 100.0 if c_1m > 0 else 0.0
    ret_3m = ((cmp_val - c_3m) / c_3m) * 100.0 if c_3m > 0 else 0.0
    ret_6m = ((cmp_val - c_6m) / c_6m) * 100.0 if c_6m > 0 else 0.0

    # Rule 1: Minimum prior run of >= 35% (low over 126d to high over 63d)
    low_126 = float(low.iloc[-126:].min())
    high_63 = float(high.iloc[-63:].max())
    if low_126 <= 0 or high_63 <= 0:
        return None
    prior_run_pct = ((high_63 - low_126) / low_126) * 100.0
    if prior_run_pct < min_run_pct:
        return None

    # Rule 2: Not deep down from 1-Month (21d) high — max drawdown <= 25%
    high_1m = float(high.iloc[-21:].max())
    if high_1m <= 0:
        return None
    dd_from_1m_high_pct = ((cmp_val - high_1m) / high_1m) * 100.0
    if dd_from_1m_high_pct < -abs(max_dd_1m_pct):
        return None

    # Moving averages (Stage 2 / constructive structure)
    ema10 = float(close.ewm(span=10, adjust=False).mean().iloc[-1])
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.rolling(120).mean().iloc[-1])
    if cmp_val < sma200 or cmp_val < sma50 * 0.94:
        return None

    # Rule 3: VCP Volatility Contraction (recent 5d range contracting vs prior 20d range)
    daily_range_pct = ((high - low) / close.replace(0, np.nan)) * 100.0
    range_5d = float(daily_range_pct.iloc[-6:-1].mean()) if len(daily_range_pct) >= 6 else float(daily_range_pct.iloc[-5:].mean())
    range_20d = float(daily_range_pct.iloc[-26:-6].mean()) if len(daily_range_pct) >= 26 else float(daily_range_pct.iloc[-20:].mean())
    if range_20d <= 0 or np.isnan(range_5d) or np.isnan(range_20d):
        return None
    contraction_ratio = range_5d / range_20d
    if contraction_ratio > 0.95:
        return None

    vol_dry_ratio = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else 1.0

    # Rule 4: Breaking its recent low by opening gap-up (check last 5 sessions)
    recent_low_15 = float(low.iloc[-15:-1].min()) if len(low) >= 15 else float(low.iloc[:-1].min())
    best_gap_pct = 0.0
    gap_days_ago = None
    for offset in range(1, 6):
        idx = -offset
        if abs(idx) >= len(df):
            break
        o_i = float(open_s.iloc[idx])
        c_i = float(close.iloc[idx])
        prev_c_i = float(close.iloc[idx - 1])
        if prev_c_i > 0:
            g_pct = ((o_i - prev_c_i) / prev_c_i) * 100.0
            if g_pct >= 0.5 and c_i >= o_i * 0.985 and c_i > prev_c_i and cmp_val >= recent_low_15 * 1.005:
                if g_pct > best_gap_pct:
                    best_gap_pct = g_pct
                    gap_days_ago = offset - 1

    has_gap_up_break = best_gap_pct >= 0.5
    if require_gap_up and not has_gap_up_break:
        return None

    run_pts = min(25.0, max(0.0, (prior_run_pct - 35.0) * 0.35 + 10.0))
    vcp_pts = min(25.0, max(0.0, (1.0 - contraction_ratio) * 50.0))
    gap_pts = min(25.0, best_gap_pct * 5.0 + (8.0 if gap_days_ago == 0 else 4.0)) if has_gap_up_break else 0.0
    dry_pts = min(15.0, max(0.0, (1.1 - vol_dry_ratio) * 25.0))
    cluster_tag = get_stock_cluster_tag(sym)
    cluster_pts = 10.0 if cluster_tag else 0.0
    total_score = round(run_pts + vcp_pts + gap_pts + dry_pts + cluster_pts, 1)

    stop_loss = round(min(recent_low_15 * 0.99, ema21 * 0.975), 2)
    if stop_loss >= cmp_val * 0.985:
        stop_loss = round(cmp_val * 0.95, 2)
    risk = cmp_val - stop_loss
    target_price = round(max(high_1m * 1.10, cmp_val + risk * 2.5), 2)
    rr_ratio = round((target_price - cmp_val) / risk, 1) if risk > 0 else 0.0

    if has_gap_up_break and gap_days_ago == 0:
        verdict = f"🚀 Today Gap-Up +{best_gap_pct:.1f}% VCP"
    elif has_gap_up_break:
        verdict = f"⚡ Gap-Up +{best_gap_pct:.1f}% ({gap_days_ago}d ago)"
    else:
        verdict = "🎯 VCP Coil Ready"

    m_rules = evaluate_manas_trend_rules(df, base_pivot=recent_low_15, benchmark_3m_ret=benchmark_3m_ret)

    return {
        "symbol": sym,
        "company_name": get_company_name(sym),
        "sector": get_stock_sector(sym),
        "cluster_tag": cluster_tag or "",
        "cmp": round(cmp_val, 2),
        "day_change_pct": round(day_chg_pct, 2),
        "prior_run_pct": round(prior_run_pct, 1),
        "ret_1m": round(ret_1m, 2),
        "ret_3m": round(ret_3m, 2),
        "ret_6m": round(ret_6m, 2),
        "dd_1m_high_pct": round(dd_from_1m_high_pct, 2),
        "contraction_ratio": round(contraction_ratio, 2),
        "vol_dry_ratio": round(vol_dry_ratio, 2),
        "gap_up_pct": round(best_gap_pct, 2),
        "gap_days_ago": gap_days_ago,
        "recent_low": round(recent_low_15, 2),
        "high_1m": round(high_1m, 2),
        "ema10": round(ema10, 2),
        "ema21": round(ema21, 2),
        "sma50": round(sma50, 2),
        "stop_loss": stop_loss,
        "target_price": target_price,
        "rr_ratio": rr_ratio,
        "swing_score": total_score,
        "verdict": verdict,
        "days_above_50ma": m_rules["days_above_50ma"],
        "closes_below_50ma": m_rules["closes_below_50ma"],
        "bar_timing_label": m_rules["bar_timing_label"],
        "trail_ma_label": m_rules["trail_ma_label"],
        "pyramid_label": m_rules["pyramid_label"],
        "rs_vs_nifty": m_rules["rs_vs_nifty"],
        "extension_label": m_rules["extension_label"],
    }


def _get_nifty_3m_return() -> float:
    """Fetch NIFTY 50 (^NSEI) 3M return once for Relative Strength calculation."""
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


def find_top10_swing_finalists(
    min_run_pct: float = 35.0,
    max_dd_1m_pct: float = 25.0,
    require_gap_up: bool = True,
    max_price: float = 1000.0,
    max_results: int = 20,
) -> list[dict]:
    """Scan Top 1000 NSE stocks from local 1d OHLCV cache and return top N Manas Arora VCP finalists."""
    symbols = [s.strip().upper().replace(".NS", "") for s in get_top1500_nse_symbols() if s.strip()]
    bulk_data = bulk_get_cached_ohlcv(symbols, "1d")
    bench_3m = _get_nifty_3m_return()

    candidates = []
    for sym in symbols:
        df = bulk_data.get(sym)
        if df is None or df.empty:
            continue
        scored = score_swing_candidate(
            sym,
            df,
            min_run_pct=min_run_pct,
            max_dd_1m_pct=max_dd_1m_pct,
            require_gap_up=require_gap_up,
            benchmark_3m_ret=bench_3m,
        )
        if scored and (max_price <= 0 or scored["cmp"] <= max_price):
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
    st.markdown("### 🚀 Manas Arora Style VCP + Gap-Up Finalists (Top 20)")
    st.markdown(
        "<p style='font-size:0.9rem; color:#94a3b8; margin-top:-6px;'>"
        "Selects up to <b>20 finalist swing stocks</b> with a <b>≥35% prior run-up</b>, holding within <b>≤25% of their 1-Month high</b>, "
        "forming a tight <b>VCP contraction</b>, and <b>breaking their recent low with a Gap-Up open</b> ready for the next move."
        "</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns([1.8, 1.8, 1.3, 1.2, 1.8, 1.3])
    with c1:
        min_run = st.slider("Min Prior Run-Up (%)", min_value=20.0, max_value=80.0, value=35.0, step=5.0, key="ma_min_run")
    with c2:
        max_dd = st.slider("Max Drop from 1M High (%)", min_value=5.0, max_value=35.0, value=25.0, step=1.0, key="ma_max_dd")
    with c3:
        max_price = st.number_input("Max Price (₹, 0=All):", min_value=0, value=1000, step=100, key="ma_max_price")
    with c4:
        max_res = st.selectbox("Show Top:", [10, 20, 30, 50], index=1, key="ma_max_res")
    with c5:
        req_gap = st.checkbox("Require Gap-Up Break Off Recent Low", value=True, key="ma_req_gap")
    with c6:
        refresh_clicked = st.button("🔄 Refresh", key="sf_refresh_btn", use_container_width=True)

    cache_key = f"ma_vcp_{min_run}_{max_dd}_{req_gap}_{max_price}_{max_res}"
    if refresh_clicked or cache_key not in st.session_state:
        with st.spinner("Scanning for ≥35% runners forming VCP + Gap-Up reversal..."):
            st.session_state[cache_key] = find_top10_swing_finalists(
                min_run_pct=min_run, max_dd_1m_pct=max_dd, require_gap_up=req_gap, max_price=float(max_price), max_results=int(max_res)
            )

    finalists = st.session_state.get(cache_key, [])

    if not finalists:
        st.info("No stocks currently meet all criteria. Try unchecking 'Require Gap-Up Break Off Recent Low' or adjusting the sliders.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Finalists Selected", f"{len(finalists)} / {max_res}")
    k2.metric("Avg Prior Run", f"+{sum(f['prior_run_pct'] for f in finalists) / len(finalists):.1f}%")
    k3.metric("Avg Off 1M High", f"{sum(f['dd_1m_high_pct'] for f in finalists) / len(finalists):.1f}%")
    k4.metric("Avg VCP Tightness", f"{sum(f['contraction_ratio'] for f in finalists) / len(finalists):.2f}x")

    exp_df = pd.DataFrame(finalists)
    e1, e2, _ = st.columns([1.5, 1.8, 5])
    with e1:
        st.download_button(
            "📥 Export CSV",
            exp_df.to_csv(index=False).encode("utf-8"),
            file_name="manas_arora_top10_vcp.csv",
            mime="text/csv",
            key="sf_csv_dl",
        )
    with e2:
        tv_txt = ",".join([f"NSE:{f['symbol']}" for f in finalists])
        st.download_button(
            "📋 TradingView Watchlist (.txt)",
            tv_txt.encode("utf-8"),
            file_name="manas_arora_top10_tv.txt",
            mime="text/plain",
            key="sf_tv_dl",
        )

    rows_html = []
    for idx, r in enumerate(finalists, 1):
        sym = r["symbol"]
        cluster_badge = (
            f'<span style="display:inline-block;margin-left:4px;padding:1px 5px;background:rgba(251,191,36,0.18);'
            f'color:#fbbf24;border:1px solid rgba(251,191,36,0.4);border-radius:3px;font-size:0.68rem;font-weight:700;">'
            f'✦ {r["cluster_tag"]}</span>' if r["cluster_tag"] else ""
        )
        rows_html.append(
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.06);">'
            f'<td style="padding:10px 8px;text-align:center;font-weight:800;color:#fbbf24;">#{idx}</td>'
            f'<td style="padding:10px 8px;">'
            f'<a href="https://in.tradingview.com/chart/?symbol=NSE:{sym}" target="_blank" style="color:#38bdf8;font-weight:800;text-decoration:none;">{sym} ↗</a>'
            f'{cluster_badge}<div style="font-size:0.72rem;color:#64748b;">{r["sector"]}</div></td>'
            f'<td style="padding:10px 8px;text-align:right;font-weight:700;color:#f8fafc;">₹{r["cmp"]:,.2f}</td>'
            f'<td style="padding:10px 8px;text-align:center;">{get_day_change_badge_html(r["day_change_pct"])}</td>'
            f'<td style="padding:10px 8px;text-align:center;font-weight:800;color:#00e676;">+{r["prior_run_pct"]:.1f}%</td>'
            f'<td style="padding:10px 8px;text-align:center;color:#fca5a5;font-weight:700;">{r["dd_1m_high_pct"]:.1f}%</td>'
            f'<td style="padding:10px 8px;text-align:center;color:#22d3ee;font-weight:700;">{r["contraction_ratio"]:.2f}x</td>'
            f'<td style="padding:10px 8px;text-align:center;color:#38bdf8;font-weight:800;">+{r["gap_up_pct"]:.2f}%</td>'
            f'<td style="padding:10px 8px;text-align:right;color:#ef4444;font-weight:600;">₹{r["stop_loss"]:,.2f}</td>'
            f'<td style="padding:10px 8px;font-size:0.78rem;color:#22d3ee;">{r.get("trail_ma_label", "🛡️ 20 SMA")}</td>'
            f'<td style="padding:10px 8px;text-align:center;color:#00e676;font-weight:700;">+{r.get("rs_vs_nifty", 0):.1f}%</td>'
            f'<td style="padding:10px 8px;text-align:center;font-size:0.76rem;color:#00e676;font-weight:700;">{r.get("days_above_50ma", 0)}/50</td>'
            f'<td style="padding:10px 8px;text-align:center;font-size:0.76rem;">{r.get("extension_label", "—")}</td>'
            f'<td style="padding:10px 8px;text-align:center;font-size:0.76rem;">{r.get("bar_timing_label", "—")}</td>'
            f'<td style="padding:10px 8px;text-align:center;font-size:0.76rem;">{r.get("pyramid_label", "✅")}</td>'
            f'<td style="padding:10px 8px;text-align:center;"><span style="background:rgba(0,230,118,0.15);color:#00e676;'
            f'border:1px solid rgba(0,230,118,0.4);padding:2px 7px;border-radius:4px;font-weight:800;">{r["swing_score"]}</span></td>'
            f'<td style="padding:10px 8px;font-size:0.78rem;font-weight:600;color:#e2e8f0;">{r["verdict"]}</td>'
            f'</tr>'
        )

    st.markdown(
        '<div style="overflow-x:auto;border:1px solid rgba(255,255,255,0.1);border-radius:10px;background:rgba(9,13,22,0.92);margin-top:8px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:0.84rem;">'
        '<thead><tr style="background:rgba(255,255,255,0.04);border-bottom:1px solid rgba(255,255,255,0.12);'
        'color:#94a3b8;text-transform:uppercase;font-size:0.72rem;letter-spacing:0.4px;">'
        '<th style="padding:10px 8px;text-align:center;">Rank</th>'
        '<th style="padding:10px 8px;text-align:left;">Stock</th>'
        '<th style="padding:10px 8px;text-align:right;">CMP</th>'
        '<th style="padding:10px 8px;text-align:center;">Day Chg %</th>'
        '<th style="padding:10px 8px;text-align:center;color:#00e676;">Prior Run</th>'
        '<th style="padding:10px 8px;text-align:center;">Off 1M High</th>'
        '<th style="padding:10px 8px;text-align:center;">VCP Tight</th>'
        '<th style="padding:10px 8px;text-align:center;color:#38bdf8;">Gap-Up %</th>'
        '<th style="padding:10px 8px;text-align:right;">Initial SL</th>'
        '<th style="padding:10px 8px;text-align:left;">Trail Stop (No Cap)</th>'
        '<th style="padding:10px 8px;text-align:center;color:#00e676;">RS vs NIFTY</th>'
        '<th style="padding:10px 8px;text-align:center;">50MA (50d)</th>'
        '<th style="padding:10px 8px;text-align:center;">50MA Ext</th>'
        '<th style="padding:10px 8px;text-align:center;">Bar Timing</th>'
        '<th style="padding:10px 8px;text-align:center;">Pyramid</th>'
        '<th style="padding:10px 8px;text-align:center;">Score</th>'
        '<th style="padding:10px 8px;text-align:left;">Setup</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### 🛡️ Risk Management & Position Sizing + Finalist Chart")
    r_col1, r_col2, r_col3, r_col4 = st.columns([2, 1.5, 1.5, 1.5])
    with r_col1:
        sel_sym = st.selectbox(
            "Select Finalist to Inspect:",
            [f["symbol"] for f in finalists],
            key="sf_chart_select",
        )
    selected_cand = next((f for f in finalists if f["symbol"] == sel_sym), finalists[0])
    with r_col2:
        acct_cap = st.number_input("Trading Capital (₹):", min_value=10000, value=500000, step=50000, key="sf_cap")
    with r_col3:
        risk_pct = st.number_input("Risk per Trade (%):", min_value=0.25, max_value=5.0, value=1.0, step=0.25, key="sf_risk_pct")
    with r_col4:
        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
        if st.button(f"➕ Add {sel_sym} to Watchlist", key="sf_add_wl", use_container_width=True):
            watchlist.add_to_watchlist(
                sel_sym,
                selected_cand["cmp"],
                selected_cand["stop_loss"],
                selected_cand["target_price"],
                f"Manas Arora VCP ({selected_cand['verdict']} | Run: +{selected_cand['prior_run_pct']:.1f}%, Off 1M High: {selected_cand['dd_1m_high_pct']:.1f}%)",
            )
            st.success(f"Added {sel_sym} to Watchlist!")

    risk_rupees = acct_cap * (risk_pct / 100.0)
    risk_per_sh = max(0.01, selected_cand["cmp"] - selected_cand["stop_loss"])
    total_qty = int(risk_rupees / risk_per_sh)
    pilot_qty = max(1, int(total_qty * 0.5)) if total_qty > 0 else 0
    pos_val = total_qty * selected_cand["cmp"]
    st.info(
        f"**{sel_sym} Position Plan:** Max Risk **₹{risk_rupees:,.0f}** ({risk_pct}% of ₹{acct_cap:,.0f}) | "
        f"Risk/Share **₹{risk_per_sh:,.2f}** (CMP ₹{selected_cand['cmp']:,.2f} → SL ₹{selected_cand['stop_loss']:,.2f}) | "
        f"**Total Qty: {total_qty:,} shares** (₹{pos_val:,.0f}) → Start with **Pilot 50%: {pilot_qty:,} shares**, "
        f"pyramid remaining after 1–2 days of positive feedback."
    )

    _render_chart(sel_sym, selected_cand)

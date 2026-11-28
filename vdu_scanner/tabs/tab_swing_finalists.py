"""
tab_swing_finalists.py
Complete Manas Arora-Inspired Swing Trading Playbook:
  Step 1: Market Suitability Gate (Nifty > 50 DMA, 50 DMA rising 20d, Monthly Thrust, Fresh Highs > Lows)
  Step 2: Momentum Scanner (RS, 50/200 DMA, 200 DMA rising 3M, <=15% off 52W High, >50% of 52W range, Liquidity)
  Step 3: VCP Detector (Prior rally, 3-10d consolidation, last 2-3 candles <=3%, 3d vol < 70% of 20d, near 21 EMA, Inside Bar)
  Step 4: Strong Start Plan (0.2-2.0% gap zone, inside-bar high, 0.1% above trigger)
  Step 5: Dual-Limit Position Sizing (min(Risk Qty at 0.5%, Max 20% Allocation Qty) * Market Size Factor)
  Step 6: Trailing Exit Rules (Initial SL, 20 DMA -> 10 DMA when rising above 20 DMA, 5%+ surge trim)
  Step 8: Trading Journal (log & export trades with R-multiples)
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import get_company_name
from data_fetcher import get_top1500_nse_symbols, get_stock_sector
from local_cache_manager import bulk_get_cached_ohlcv
from breadth_ledger import get_stock_cluster_tag, get_market_breadth_summary
from trading_rules import evaluate_manas_trend_rules
from utils import get_day_change_badge_html
import watchlist

JOURNAL_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "manas_trading_journal.csv")


def get_market_suitability() -> dict:
    """
    Step 1: Evaluate the 4 Market Suitability conditions:
      1. Nifty 50 closing above 50 DMA
      2. Nifty 50 DMA > Nifty 50 DMA 20 sessions ago
      3. Market breadth monthly thrust positive (stocks > 50 DMA >= 45% or Nifty > 0)
      4. Fresh 52W Highs > Fresh 52W Lows
    Returns dict with condition booleans, score (0-4), size_multiplier (1.0 / 0.5 / 0.0), and badge.
    """
    nifty_above_50 = False
    dma50_rising = False
    nifty_cmp = 0.0
    nifty_50dma = 0.0
    nifty_3m_ret = 0.0

    try:
        import yfinance as yf
        nsei = yf.download("^NSEI", period="6mo", interval="1d", progress=False, timeout=10)
        if nsei is not None and len(nsei) >= 75:
            c = nsei["Close"].squeeze().dropna().astype(float)
            sma50 = c.rolling(50).mean()
            nifty_cmp = float(c.iloc[-1])
            nifty_50dma = float(sma50.iloc[-1])
            sma50_20ago = float(sma50.iloc[-21]) if len(sma50.dropna()) >= 21 else nifty_50dma
            nifty_above_50 = nifty_cmp > nifty_50dma
            dma50_rising = nifty_50dma > sma50_20ago
            if len(c) >= 64 and float(c.iloc[-64]) > 0:
                nifty_3m_ret = ((nifty_cmp - float(c.iloc[-64])) / float(c.iloc[-64])) * 100.0
    except Exception:
        pass

    breadth = get_market_breadth_summary()
    highs_52w = int(breadth.get("highs_52w", 0))
    lows_52w = int(breadth.get("lows_52w", 0))
    fresh_highs_gt_lows = highs_52w > lows_52w
    # Monthly thrust positive: >= 45% stocks above 50 DMA or net 4% surge > drop
    m50 = float(breadth.get("stocks_above_50dma", 0.0))
    surge_net = int(breadth.get("surge_up_4pct", 0)) - int(breadth.get("drop_dn_4pct", 0))
    monthly_thrust_pos = (m50 >= 45.0) or (surge_net > 0)

    score = sum([nifty_above_50, dma50_rising, monthly_thrust_pos, fresh_highs_gt_lows])
    if score == 4:
        size_mult = 1.0
        decision = "🟢 Normal Position Sizing (4/4 Positive)"
    elif score >= 2:
        size_mult = 0.5
        decision = f"🟡 Half Position Sizing ({score}/4 Positive)"
    else:
        size_mult = 0.0
        decision = f"🔴 No Fresh Trades ({score}/4 Positive)"

    return {
        "nifty_above_50": nifty_above_50,
        "dma50_rising": dma50_rising,
        "monthly_thrust_pos": monthly_thrust_pos,
        "fresh_highs_gt_lows": fresh_highs_gt_lows,
        "nifty_cmp": round(nifty_cmp, 2),
        "nifty_50dma": round(nifty_50dma, 2),
        "nifty_3m_ret": round(nifty_3m_ret, 2),
        "highs_52w": highs_52w,
        "lows_52w": lows_52w,
        "m50": m50,
        "score": score,
        "size_multiplier": size_mult,
        "decision": decision,
    }


def score_swing_candidate(
    sym: str,
    df: pd.DataFrame,
    min_run_pct: float = 35.0,
    max_dd_1m_pct: float = 25.0,
    require_gap_up: bool = True,
    benchmark_3m_ret: float | None = None,
    strict_playbook: bool = False,
) -> dict | None:
    """
    Evaluate a stock for the Manas Arora VCP + Strong Start setup.
    Supports both flexible mode and Strict Playbook mode (Step 2 & 3 exact rules).
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
    avg_vol_3 = float(vol.iloc[-3:].mean())
    avg_traded_val_cr = (cmp_val * avg_vol_20) / 1e7
    min_val_cr = 10.0 if strict_playbook else 1.0
    if avg_vol_20 < 50000 or avg_traded_val_cr < min_val_cr:
        return None

    day_chg_pct = ((cmp_val - prev_close) / prev_close) * 100.0

    c_1m = float(close.iloc[-22]) if len(close) >= 22 else float(close.iloc[0])
    c_3m = float(close.iloc[-64]) if len(close) >= 64 else float(close.iloc[0])
    c_6m = float(close.iloc[-127]) if len(close) >= 127 else float(close.iloc[0])
    ret_1m = ((cmp_val - c_1m) / c_1m) * 100.0 if c_1m > 0 else 0.0
    ret_3m = ((cmp_val - c_3m) / c_3m) * 100.0 if c_3m > 0 else 0.0
    ret_6m = ((cmp_val - c_6m) / c_6m) * 100.0 if c_6m > 0 else 0.0

    # Moving averages (Step 2)
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200_s = close.rolling(200).mean() if len(close) >= 200 else close.rolling(120).mean()
    sma200 = float(sma200_s.iloc[-1])
    sma200_3m_ago = float(sma200_s.iloc[-64]) if len(sma200_s.dropna()) >= 64 else sma200

    if cmp_val < sma200 or cmp_val < sma50 * (1.0 if strict_playbook else 0.94):
        return None
    if strict_playbook and (sma50 <= sma200 or sma200 <= sma200_3m_ago):
        return None

    # 52W High / Low checks (Step 2)
    win_52w = min(252, len(df))
    high_52w = float(high.iloc[-win_52w:].max())
    low_52w = float(low.iloc[-win_52w:].min())
    dist_52w_high_pct = ((cmp_val - high_52w) / high_52w) * 100.0 if high_52w > 0 else 0.0
    range_52w = high_52w - low_52w
    pos_in_52w_pct = ((cmp_val - low_52w) / range_52w) * 100.0 if range_52w > 0 else 50.0
    if strict_playbook and (dist_52w_high_pct < -15.0 or pos_in_52w_pct < 50.0):
        return None

    # Step 3: Prior momentum >= min_run_pct (35% default, 50% in strict playbook)
    eff_min_run = max(min_run_pct, 50.0) if strict_playbook else min_run_pct
    low_126 = float(low.iloc[-126:].min())
    high_63 = float(high.iloc[-63:].max())
    if low_126 <= 0 or high_63 <= 0:
        return None
    prior_run_pct = ((high_63 - low_126) / low_126) * 100.0
    if prior_run_pct < eff_min_run:
        return None

    # Not deep down from 1-Month high (<= 25%)
    high_1m = float(high.iloc[-21:].max())
    dd_from_1m_high_pct = ((cmp_val - high_1m) / high_1m) * 100.0 if high_1m > 0 else 0.0
    if dd_from_1m_high_pct < -abs(max_dd_1m_pct):
        return None

    # Step 3: VCP Contraction + Last 2-3 candles range <= 3% + 3d vol < 70% of 20d avg
    daily_range_pct = ((high - low) / close.replace(0, np.nan)) * 100.0
    last3_range_pct = float(daily_range_pct.iloc[-3:].mean())
    range_5d = float(daily_range_pct.iloc[-6:-1].mean()) if len(daily_range_pct) >= 6 else float(daily_range_pct.iloc[-5:].mean())
    range_20d = float(daily_range_pct.iloc[-26:-6].mean()) if len(daily_range_pct) >= 26 else float(daily_range_pct.iloc[-20:].mean())
    if range_20d <= 0 or np.isnan(range_5d) or np.isnan(range_20d):
        return None
    contraction_ratio = range_5d / range_20d
    if contraction_ratio > 0.95:
        return None

    vol_dry_3d_ratio = avg_vol_3 / avg_vol_20 if avg_vol_20 > 0 else 1.0
    if strict_playbook and (last3_range_pct > 3.2 or vol_dry_3d_ratio > 0.75):
        return None

    # Inside Bar detection on latest or prior candle
    is_inside_bar = (float(high.iloc[-1]) <= float(high.iloc[-2])) and (float(low.iloc[-1]) >= float(low.iloc[-2]))
    inside_bar_high = float(high.iloc[-1]) if is_inside_bar else float(high.iloc[-2])

    # Gap-Up reversal off recent low (last 5 sessions)
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

    # Step 4: Strong Start next-session trigger levels
    ss_gap_low = round(cmp_val * 1.002, 2)
    ss_gap_high = round(cmp_val * 1.020, 2)
    entry_trigger = round(inside_bar_high * 1.001, 2)

    stop_loss = round(min(recent_low_15 * 0.99, ema21 * 0.975), 2)
    if stop_loss >= cmp_val * 0.985:
        stop_loss = round(cmp_val * 0.95, 2)
    sl_dist_pct = round(((cmp_val - stop_loss) / cmp_val) * 100.0, 2)
    risk = cmp_val - stop_loss
    target_price = round(max(high_1m * 1.10, cmp_val + risk * 2.5), 2)

    m_rules = evaluate_manas_trend_rules(df, base_pivot=recent_low_15, benchmark_3m_ret=benchmark_3m_ret)

    # Composite Score (0 - 100)
    run_pts = min(25.0, max(0.0, (prior_run_pct - 35.0) * 0.35 + 10.0))
    vcp_pts = min(25.0, max(0.0, (1.0 - contraction_ratio) * 50.0))
    gap_pts = min(20.0, best_gap_pct * 4.5 + (6.0 if gap_days_ago == 0 else 3.0)) if has_gap_up_break else 0.0
    dry_pts = min(15.0, max(0.0, (1.1 - vol_dry_3d_ratio) * 25.0))
    ib_pts = 5.0 if is_inside_bar else 0.0
    cluster_tag = get_stock_cluster_tag(sym)
    cluster_pts = 10.0 if cluster_tag else 0.0
    total_score = round(run_pts + vcp_pts + gap_pts + dry_pts + ib_pts + cluster_pts, 1)

    if has_gap_up_break and gap_days_ago == 0:
        verdict = f"🚀 Gap-Up +{best_gap_pct:.1f}% VCP"
    elif is_inside_bar:
        verdict = "📦 Inside Bar Coil (Strong Start Ready)"
    elif has_gap_up_break:
        verdict = f"⚡ Gap-Up +{best_gap_pct:.1f}% ({gap_days_ago}d ago)"
    else:
        verdict = "🎯 VCP Coil Ready"

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
        "dist_52w_high_pct": round(dist_52w_high_pct, 1),
        "avg_traded_val_cr": round(avg_traded_val_cr, 1),
        "contraction_ratio": round(contraction_ratio, 2),
        "last3_range_pct": round(last3_range_pct, 2),
        "vol_dry_ratio": round(vol_dry_3d_ratio, 2),
        "is_inside_bar": is_inside_bar,
        "inside_bar_high": round(inside_bar_high, 2),
        "ss_gap_low": ss_gap_low,
        "ss_gap_high": ss_gap_high,
        "entry_trigger": entry_trigger,
        "gap_up_pct": round(best_gap_pct, 2),
        "gap_days_ago": gap_days_ago,
        "recent_low": round(recent_low_15, 2),
        "high_1m": round(high_1m, 2),
        "stop_loss": stop_loss,
        "sl_dist_pct": sl_dist_pct,
        "target_price": target_price,
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


def find_top10_swing_finalists(
    min_run_pct: float = 35.0,
    max_dd_1m_pct: float = 25.0,
    require_gap_up: bool = True,
    max_price: float = 1000.0,
    strict_playbook: bool = False,
    benchmark_3m_ret: float | None = None,
    max_results: int = 20,
) -> list[dict]:
    """Scan Top 1500 NSE stocks from local 1d OHLCV cache and return top N Manas Arora VCP finalists."""
    symbols = [s.strip().upper().replace(".NS", "") for s in get_top1500_nse_symbols() if s.strip()]
    bulk_data = bulk_get_cached_ohlcv(symbols, "1d")
    if benchmark_3m_ret is None:
        benchmark_3m_ret = get_market_suitability().get("nifty_3m_ret", 0.0)

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
            benchmark_3m_ret=benchmark_3m_ret,
            strict_playbook=strict_playbook,
        )
        if scored and (max_price <= 0 or scored["cmp"] <= max_price):
            candidates.append(scored)

    candidates.sort(key=lambda x: x["swing_score"], reverse=True)
    return candidates[:max_results]


def _render_chart(sym: str, candidate: dict):
    """Render interactive Plotly Candlestick + 10/21/50 EMA/SMA + Volume chart for a finalist."""
    bulk = bulk_get_cached_ohlcv([sym], "1d")
    df = bulk.get(sym)
    if df is None or df.empty:
        st.warning(f"No historical chart data cached for {sym}.")
        return

    df = df.sort_values("Date").tail(120).copy()
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()
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
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], name="20 SMA (Trail)", line=dict(color="#22d3ee", width=1.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA21"], name="21 EMA", line=dict(color="#f59e0b", width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], name="50 SMA", line=dict(color="#a855f7", width=1.6, dash="dot")), row=1, col=1)

    fig.add_hline(y=candidate["entry_trigger"], line_dash="dot", line_color="#38bdf8", annotation_text=f"Trigger ₹{candidate['entry_trigger']}", row=1, col=1)
    fig.add_hline(y=candidate["stop_loss"], line_dash="dash", line_color="#ef4444", annotation_text=f"SL ₹{candidate['stop_loss']}", row=1, col=1)

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


def _load_journal() -> pd.DataFrame:
    cols = ["Date", "Stock", "Setup", "RS", "Prior Rally %", "Contraction", "Entry", "Stop Loss", "Qty", "Exit", "P&L", "Return (R)"]
    if os.path.exists(JOURNAL_CSV_PATH):
        try:
            return pd.read_csv(JOURNAL_CSV_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=cols)


def render():
    st.markdown("### 🚀 Manas Arora Swing Playbook & VCP Finalists (Top 20)")

    # ── Step 1: Market Suitability Gate ──────────────────────────────────────
    if "ma_market_gate" not in st.session_state:
        st.session_state["ma_market_gate"] = get_market_suitability()
    mg = st.session_state["ma_market_gate"]

    gate_color = "#00e676" if mg["size_multiplier"] == 1.0 else ("#fbbf24" if mg["size_multiplier"] == 0.5 else "#ef4444")
    st.markdown(
        f'<div style="background:rgba(15,23,42,0.85);border:1px solid {gate_color};border-radius:10px;padding:12px 16px;margin-bottom:14px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
        f'<div><span style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;">Step 1 — Market Suitability Gate</span>'
        f'<h4 style="margin:2px 0 0 0;color:{gate_color};">{mg["decision"]}</h4></div>'
        f'<div style="font-size:0.82rem;color:#cbd5e1;">'
        f'{"✅" if mg["nifty_above_50"] else "❌"} Nifty &gt; 50 DMA ({mg["nifty_cmp"]:,.0f} vs {mg["nifty_50dma"]:,.0f}) &nbsp;|&nbsp; '
        f'{"✅" if mg["dma50_rising"] else "❌"} 50 DMA Rising (20d) &nbsp;|&nbsp; '
        f'{"✅" if mg["monthly_thrust_pos"] else "❌"} Monthly Thrust ({mg["m50"]:.1f}% &gt; 50MA) &nbsp;|&nbsp; '
        f'{"✅" if mg["fresh_highs_gt_lows"] else "❌"} 52W Highs/Lows ({mg["highs_52w"]}/{mg["lows_52w"]})'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )

    # ── Step 2 & 3 Controls ──────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns([1.6, 1.6, 1.2, 1.1, 2.0, 1.3])
    with c1:
        min_run = st.slider("Min Prior Run (%)", min_value=20.0, max_value=80.0, value=35.0, step=5.0, key="ma_min_run")
    with c2:
        max_dd = st.slider("Max Off 1M High (%)", min_value=5.0, max_value=35.0, value=25.0, step=1.0, key="ma_max_dd")
    with c3:
        max_price = st.number_input("Max Price (₹):", min_value=0, value=1000, step=100, key="ma_max_price")
    with c4:
        max_res = st.selectbox("Show Top:", [10, 20, 30, 50], index=1, key="ma_max_res")
    with c5:
        strict_mode = st.checkbox("Strict Playbook (50% Run, ₹10Cr, <70% Vol)", value=False, key="ma_strict")
        req_gap = st.checkbox("Require Gap-Up Break Off Low", value=False, key="ma_req_gap")
    with c6:
        refresh_clicked = st.button("🔄 Refresh", key="sf_refresh_btn", use_container_width=True)
        if refresh_clicked:
            st.session_state["ma_market_gate"] = get_market_suitability()

    cache_key = f"ma_vcp_{min_run}_{max_dd}_{req_gap}_{max_price}_{max_res}_{strict_mode}"
    if refresh_clicked or cache_key not in st.session_state:
        with st.spinner("Scanning 1,500 NSE stocks for Manas Arora VCP + Strong Start setups..."):
            st.session_state[cache_key] = find_top10_swing_finalists(
                min_run_pct=min_run,
                max_dd_1m_pct=max_dd,
                require_gap_up=req_gap,
                max_price=float(max_price),
                strict_playbook=strict_mode,
                benchmark_3m_ret=mg["nifty_3m_ret"],
                max_results=int(max_res),
            )

    finalists = st.session_state.get(cache_key, [])
    if not finalists:
        st.info("No stocks currently meet all selected criteria. Try unchecking 'Strict Playbook' or 'Require Gap-Up Break Off Low'.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Finalists Selected", f"{len(finalists)} / {max_res}")
    k2.metric("Avg Prior Run", f"+{sum(f['prior_run_pct'] for f in finalists) / len(finalists):.1f}%")
    k3.metric("Inside-Bar Coils", sum(1 for f in finalists if f["is_inside_bar"]))
    k4.metric("Avg 3d Vol Dry", f"{sum(f['vol_dry_ratio'] for f in finalists) / len(finalists):.2f}x")

    exp_df = pd.DataFrame(finalists)
    e1, e2, _ = st.columns([1.5, 1.8, 5])
    with e1:
        st.download_button("📥 Export CSV", exp_df.to_csv(index=False).encode("utf-8"), file_name="manas_arora_playbook_finalists.csv", mime="text/csv", key="sf_csv_dl")
    with e2:
        tv_txt = ",".join([f"NSE:{f['symbol']}" for f in finalists])
        st.download_button("📋 TradingView (.txt)", tv_txt.encode("utf-8"), file_name="manas_arora_tv.txt", mime="text/plain", key="sf_tv_dl")

    rows_html = []
    for idx, r in enumerate(finalists, 1):
        sym = r["symbol"]
        cl_tag = (
            f'<span style="margin-left:4px;padding:1px 5px;background:rgba(251,191,36,0.18);color:#fbbf24;'
            f'border:1px solid rgba(251,191,36,0.4);border-radius:3px;font-size:0.68rem;font-weight:700;">✦ {r["cluster_tag"]}</span>'
            if r["cluster_tag"] else ""
        )
        ib_tag = ' <span style="background:rgba(56,189,248,0.2);color:#38bdf8;padding:1px 4px;border-radius:3px;font-size:0.67rem;font-weight:700;">IB</span>' if r["is_inside_bar"] else ""
        sl_warn = " ⚠️>2%" if r["sl_dist_pct"] > 5.0 else ""
        rows_html.append(
            f'<tr style="border-bottom:1px solid rgba(255,255,255,0.06);">'
            f'<td style="padding:9px 6px;text-align:center;font-weight:800;color:#fbbf24;">#{idx}</td>'
            f'<td style="padding:9px 6px;"><a href="https://in.tradingview.com/chart/?symbol=NSE:{sym}" target="_blank" style="color:#38bdf8;font-weight:800;text-decoration:none;">{sym} ↗</a>{ib_tag}{cl_tag}'
            f'<div style="font-size:0.7rem;color:#64748b;">{r["sector"]} · ₹{r["avg_traded_val_cr"]}Cr/d</div></td>'
            f'<td style="padding:9px 6px;text-align:right;font-weight:700;color:#f8fafc;">₹{r["cmp"]:,.2f}</td>'
            f'<td style="padding:9px 6px;text-align:center;">{get_day_change_badge_html(r["day_change_pct"])}</td>'
            f'<td style="padding:9px 6px;text-align:center;font-weight:800;color:#00e676;">+{r["prior_run_pct"]:.0f}%</td>'
            f'<td style="padding:9px 6px;text-align:center;color:#fca5a5;">{r["dd_1m_high_pct"]:.1f}%</td>'
            f'<td style="padding:9px 6px;text-align:center;color:#22d3ee;font-weight:700;">{r["last3_range_pct"]:.1f}% / {r["vol_dry_ratio"]:.2f}x</td>'
            f'<td style="padding:9px 6px;text-align:right;color:#38bdf8;font-weight:700;">₹{r["entry_trigger"]:,.2f}'
            f'<div style="font-size:0.66rem;color:#94a3b8;">Gap ₹{r["ss_gap_low"]:,.0f}–{r["ss_gap_high"]:,.0f}</div></td>'
            f'<td style="padding:9px 6px;text-align:right;color:#ef4444;font-weight:600;">₹{r["stop_loss"]:,.2f}'
            f'<div style="font-size:0.66rem;color:#fca5a5;">-{r["sl_dist_pct"]:.1f}%{sl_warn}</div></td>'
            f'<td style="padding:9px 6px;font-size:0.75rem;color:#22d3ee;">{r["trail_ma_label"]}</td>'
            f'<td style="padding:9px 6px;text-align:center;color:#00e676;font-weight:700;">+{r["rs_vs_nifty"]:.1f}%</td>'
            f'<td style="padding:9px 6px;text-align:center;font-size:0.75rem;color:#00e676;font-weight:700;">{r["days_above_50ma"]}/50</td>'
            f'<td style="padding:9px 6px;text-align:center;font-size:0.74rem;">{r["bar_timing_label"]}</td>'
            f'<td style="padding:9px 6px;text-align:center;"><span style="background:rgba(0,230,118,0.15);color:#00e676;border:1px solid rgba(0,230,118,0.4);padding:2px 6px;border-radius:4px;font-weight:800;">{r["swing_score"]}</span></td>'
            f'</tr>'
        )

    st.markdown(
        '<div style="overflow-x:auto;border:1px solid rgba(255,255,255,0.1);border-radius:10px;background:rgba(9,13,22,0.92);margin-top:8px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:0.82rem;">'
        '<thead><tr style="background:rgba(255,255,255,0.04);border-bottom:1px solid rgba(255,255,255,0.12);color:#94a3b8;text-transform:uppercase;font-size:0.7rem;">'
        '<th style="padding:9px 6px;text-align:center;">#</th>'
        '<th style="padding:9px 6px;text-align:left;">Stock</th>'
        '<th style="padding:9px 6px;text-align:right;">CMP</th>'
        '<th style="padding:9px 6px;text-align:center;">Day %</th>'
        '<th style="padding:9px 6px;text-align:center;color:#00e676;">Run</th>'
        '<th style="padding:9px 6px;text-align:center;">Off 1M</th>'
        '<th style="padding:9px 6px;text-align:center;">3d Rng / Vol</th>'
        '<th style="padding:9px 6px;text-align:right;color:#38bdf8;">Strong Start</th>'
        '<th style="padding:9px 6px;text-align:right;">Initial SL</th>'
        '<th style="padding:9px 6px;text-align:left;">Trail Stop</th>'
        '<th style="padding:9px 6px;text-align:center;color:#00e676;">RS vs NIFTY</th>'
        '<th style="padding:9px 6px;text-align:center;">50MA (50d)</th>'
        '<th style="padding:9px 6px;text-align:center;">Bar</th>'
        '<th style="padding:9px 6px;text-align:center;">Score</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table></div>',
        unsafe_allow_html=True,
    )

    # ── Step 5: Dual-Limit Position Sizing Calculator + Chart ────────────────
    st.markdown("---")
    st.markdown("#### 🛡️ Step 5 — Dual-Limit Position Sizing & Strong Start Plan")
    r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns([1.8, 1.4, 1.1, 1.1, 1.5])
    with r_col1:
        sel_sym = st.selectbox("Select Finalist:", [f["symbol"] for f in finalists], key="sf_chart_select")
    selected_cand = next((f for f in finalists if f["symbol"] == sel_sym), finalists[0])
    with r_col2:
        acct_cap = st.number_input("Capital (₹):", min_value=10000, value=100000, step=25000, key="sf_cap")
    with r_col3:
        risk_pct = st.number_input("Risk % (0.5%):", min_value=0.25, max_value=3.0, value=0.5, step=0.25, key="sf_risk_pct")
    with r_col4:
        max_alloc_pct = st.number_input("Max Alloc %:", min_value=5.0, max_value=50.0, value=20.0, step=5.0, key="sf_alloc_pct")
    with r_col5:
        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
        if st.button(f"➕ Watchlist ({sel_sym})", key="sf_add_wl", use_container_width=True):
            watchlist.add_to_watchlist(
                sel_sym, selected_cand["entry_trigger"], selected_cand["stop_loss"], selected_cand["target_price"],
                f"Manas Playbook | Strong Start ₹{selected_cand['entry_trigger']} | 3dVol {selected_cand['vol_dry_ratio']}x",
            )
            st.success(f"Added {sel_sym} to Watchlist!")

    # Dual-limit math: min(risk_qty, alloc_qty) * market_size_multiplier
    entry_px = selected_cand["entry_trigger"]
    sl_px = selected_cand["stop_loss"]
    risk_per_sh = max(0.01, entry_px - sl_px)
    eff_risk_rupees = acct_cap * (risk_pct / 100.0) * (mg["size_multiplier"] if mg["size_multiplier"] > 0 else 0.5)
    risk_qty = int(eff_risk_rupees / risk_per_sh)
    alloc_limit_rupees = acct_cap * (max_alloc_pct / 100.0) * (mg["size_multiplier"] if mg["size_multiplier"] > 0 else 0.5)
    alloc_qty = int(alloc_limit_rupees / entry_px) if entry_px > 0 else 0
    final_qty = min(risk_qty, alloc_qty)
    actual_invest = final_qty * entry_px
    actual_risk = final_qty * risk_per_sh

    st.info(
        f"**{sel_sym} Playbook Plan** ({mg['decision']}): "
        f"9:15–9:30 AM Strong Start Gap Zone **₹{selected_cand['ss_gap_low']:,.2f}–₹{selected_cand['ss_gap_high']:,.2f}** → "
        f"Trigger **₹{entry_px:,.2f}** | SL **₹{sl_px:,.2f}** (Risk/sh ₹{risk_per_sh:,.2f}) | "
        f"Risk Qty: **{risk_qty}**, Max-Alloc Qty ({max_alloc_pct:.0f}%): **{alloc_qty}** → "
        f"**Final Qty: {final_qty:,} shares** (₹{actual_invest:,.0f} invested, Actual Risk **₹{actual_risk:,.0f}**)."
    )

    _render_chart(sel_sym, selected_cand)

    # ── Step 8: Trading Journal ──────────────────────────────────────────────
    with st.expander("📓 Step 8 — Trading Journal & Daily Checklist", expanded=False):
        j1, j2, j3 = st.columns([2, 2, 2])
        with j1:
            if st.button(f"📝 Log {sel_sym} Setup to Journal", key="sf_log_journal"):
                jdf = _load_journal()
                new_row = {
                    "Date": pd.Timestamp.today().strftime("%Y-%m-%d"),
                    "Stock": sel_sym,
                    "Setup": selected_cand["verdict"],
                    "RS": selected_cand["rs_vs_nifty"],
                    "Prior Rally %": selected_cand["prior_run_pct"],
                    "Contraction": f"{selected_cand['last3_range_pct']}%",
                    "Entry": entry_px,
                    "Stop Loss": sl_px,
                    "Qty": final_qty,
                    "Exit": "",
                    "P&L": "",
                    "Return (R)": "",
                }
                jdf = pd.concat([jdf, pd.DataFrame([new_row])], ignore_index=True)
                jdf.to_csv(JOURNAL_CSV_PATH, index=False)
                st.success(f"Logged {sel_sym} ({final_qty} qty @ ₹{entry_px}) to journal!")
        jdf_cur = _load_journal()
        if not jdf_cur.empty:
            st.dataframe(jdf_cur, use_container_width=True, hide_index=True)
            st.download_button("📥 Download Journal CSV", jdf_cur.to_csv(index=False).encode("utf-8"), file_name="manas_trading_journal.csv", mime="text/csv", key="sf_j_dl")

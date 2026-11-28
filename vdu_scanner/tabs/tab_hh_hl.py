import json
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from config import IST_TIMEZONE
import database
from scanner import scan_hh_hl_uptrend
from trading_rules import render_10_rules_execution_card

def get_market_date(for_display=False):
    today = datetime.now(IST_TIMEZONE)
    if today.isoweekday() == 7:
        target_date = (today - timedelta(days=2)).strftime('%Y-%m-%d')
    elif today.isoweekday() == 6:
        target_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        target_date = today.strftime('%Y-%m-%d')

    if for_display:
        if not database.has_scanned_today(target_date):
            avail = database.get_available_scan_dates()
            if avail:
                return avail[0]
    return target_date


def render_swing_chart(symbol: str, df: pd.DataFrame, swing_highs: list, swing_lows: list, last_sh: float, last_sl: float):
    """Renders a Plotly candlestick chart with marked Swing Highs and Swing Lows."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    chart_df = df.tail(80).copy()
    if chart_df.empty:
        st.warning("Insufficient historical data to render chart.")
        return

    # Flatten MultiIndex columns if present (from yfinance multi-column downloads)
    if isinstance(chart_df.columns, pd.MultiIndex):
        if 'Close' in chart_df.columns.get_level_values(0):
            chart_df.columns = chart_df.columns.get_level_values(0)
        elif 'Close' in chart_df.columns.get_level_values(1):
            chart_df.columns = chart_df.columns.get_level_values(1)
        else:
            chart_df.columns = chart_df.columns.get_level_values(-1)
    chart_df = chart_df.loc[:, ~chart_df.columns.duplicated()]

    # Make index tz-naive for consistent comparisons
    try:
        if hasattr(chart_df.index, 'tz') and chart_df.index.tz is not None:
            chart_df.index = chart_df.index.tz_localize(None)
    except Exception:
        pass

    close_s = pd.Series(chart_df['Close'].squeeze(), dtype=float)
    open_s = pd.Series(chart_df['Open'].squeeze(), dtype=float)
    high_s = pd.Series(chart_df['High'].squeeze(), dtype=float)
    low_s = pd.Series(chart_df['Low'].squeeze(), dtype=float)
    vol_s = pd.Series(chart_df['Volume'].squeeze(), dtype=float)

    # Calculate 20 & 50 SMA for visual trend
    sma20 = close_s.rolling(20).mean()
    sma50 = close_s.rolling(50).mean()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25]
    )

    # 1. Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=open_s,
            high=high_s,
            low=low_s,
            close=close_s,
            name=f"{symbol} Price",
            increasing_line_color='#00e676',
            decreasing_line_color='#ef4444'
        ),
        row=1, col=1
    )

    # 2. Moving Averages
    fig.add_trace(
        go.Scatter(
            x=chart_df.index, y=sma20,
            mode='lines', line=dict(color='#38bdf8', width=1.5),
            name="20 SMA"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df.index, y=sma50,
            mode='lines', line=dict(color='#fbbf24', width=1.5),
            name="50 SMA"
        ),
        row=1, col=1
    )

    # 3. Mark Swing Highs (▲ HH)
    sh_dates = []
    sh_prices = []
    for sh in swing_highs:
        try:
            d = pd.to_datetime(sh.get('date'))
            if hasattr(d, 'tz') and d.tz is not None:
                d = d.tz_localize(None)
            if d in chart_df.index or (len(chart_df.index) > 0 and d >= chart_df.index[0]):
                sh_dates.append(d)
                sh_prices.append(float(sh.get('price', 0.0)))
        except Exception:
            pass

    if sh_dates:
        fig.add_trace(
            go.Scatter(
                x=sh_dates, y=sh_prices,
                mode='markers+text',
                marker=dict(symbol='triangle-down', size=12, color='#f43f5e'),
                text=[f"HH ₹{p:.1f}" for p in sh_prices],
                textposition="top center",
                textfont=dict(size=10, color='#fda4af'),
                name="Swing High (HH)"
            ),
            row=1, col=1
        )

    # 4. Mark Swing Lows (▼ HL)
    sl_dates = []
    sl_prices = []
    for sl in swing_lows:
        try:
            d = pd.to_datetime(sl.get('date'))
            if hasattr(d, 'tz') and d.tz is not None:
                d = d.tz_localize(None)
            if d in chart_df.index or (len(chart_df.index) > 0 and d >= chart_df.index[0]):
                sl_dates.append(d)
                sl_prices.append(float(sl.get('price', 0.0)))
        except Exception:
            pass

    if sl_dates:
        fig.add_trace(
            go.Scatter(
                x=sl_dates, y=sl_prices,
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=12, color='#10b981'),
                text=[f"HL ₹{p:.1f}" for p in sl_prices],
                textposition="bottom center",
                textfont=dict(size=10, color='#6ee7b7'),
                name="Swing Low (HL)"
            ),
            row=1, col=1
        )

    # 5. Resistance line at latest Swing High
    if last_sh > 0:
        fig.add_hline(
            y=last_sh, line_dash="dash", line_color="rgba(244, 63, 94, 0.6)",
            annotation_text=f"Resistance (Swing High) ₹{last_sh:.2f}",
            annotation_position="top right", row=1, col=1
        )

    # 6. Support line at latest Swing Low
    if last_sl > 0:
        fig.add_hline(
            y=last_sl, line_dash="dash", line_color="rgba(16, 185, 129, 0.6)",
            annotation_text=f"Support (Higher Low) ₹{last_sl:.2f}",
            annotation_position="bottom right", row=1, col=1
        )

    # 7. Volume Bars
    vol_colors = ['#00e676' if float(c) >= float(o) else '#ef4444' for c, o in zip(close_s.values, open_s.values)]
    fig.add_trace(
        go.Bar(
            x=chart_df.index, y=vol_s,
            marker_color=vol_colors, name="Volume", opacity=0.6
        ),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(10,15,30,0.8)",
        plot_bgcolor="rgba(10,15,30,0.8)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render():
    st.markdown("### 📈 Higher High & Higher Low (HH/HL) Uptrend Scanner")
    st.markdown(
        "<p style='font-size:0.92rem; color:#94a3b8; margin-top:-6px;'>"
        "Detects stocks in a confirmed <b>Dow Theory Uptrend</b> forming successive <b>Higher Highs (HH)</b> and "
        "<b>Higher Lows (HL)</b>, identifying institutional accumulation, momentum breakouts, and low-risk dip entries."
        "</p>",
        unsafe_allow_html=True
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 1. EDUCATIONAL GUIDE EXPANDER
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("📖 Dow Theory & Market Structure (Higher Highs & Higher Lows) Explained", expanded=False):
        st.markdown("""
### 🧠 What is a Higher High / Higher Low Structure?
In classical **Dow Theory** and modern **Price Action / Smart Money Concepts (SMC)**:
- **Higher High (HH)**: Each swing high peak forms at a price higher than the previous peak ($SH_n > SH_{n-1}$), showing sustained aggressive demand and buyers willing to push through resistance.
- **Higher Low (HL)**: Each pullback trough bottoms at a price higher than the previous trough ($SL_n > SL_{n-1}$), showing that institutional buyers step in at progressively higher prices to absorb sell orders.
- **Trend Invalidation (CHoCH / Structure Break)**: If the price breaks below the most recent Higher Low ($CMP < SL_n$), the uptrend structure is **invalidated**. The stop-loss for any swing trade should be placed right below the latest Higher Low.

---

### 🎯 4 Actionable Trading Setup Types

| Setup Badge | Market Structure Condition | Actionable Trading Strategy |
| :--- | :--- | :--- |
| <span style="color:#f43f5e; font-weight:bold;">🔥 Fresh HH Breakout</span> | Price is currently trading at or breaking above the latest Swing High | High-momentum breakout entry with volume expansion; target next resistance level. |
| <span style="color:#38bdf8; font-weight:bold;">⚡ Consolidating near Peak</span> | Price is within 2% below the latest Swing High, building energy | Pre-breakout watch; set price alert for breakout above Swing High. |
| <span style="color:#10b981; font-weight:bold;">🛡️ Higher Low Pullback</span> | Price has pulled back near the Higher Low support (within 4%) and is bouncing | **Lowest risk / highest reward dip entry**; tight stop loss right below Higher Low. |
| <span style="color:#818cf8; font-weight:bold;">📈 Confirmed HH+HL Uptrend</span> | 2+ consecutive rising peaks & troughs, moving upward in trend channel | Ride trend with trailing stop-loss below the ascending Higher Lows. |
""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. DATA HYDRATION (Session State / Database Cache)
    # ─────────────────────────────────────────────────────────────────────────
    if 'hh_hl_results' not in st.session_state:
        st.session_state.hh_hl_results = None
    if 'hh_hl_weekly_results' not in st.session_state:
        st.session_state.hh_hl_weekly_results = None

    # Pick up background scan status if available
    bg_status = st.session_state.get('ALL_TAB_SCAN_STATUS', {})
    if not st.session_state.hh_hl_results and bg_status.get("hh_hl_results"):
        st.session_state.hh_hl_results = bg_status["hh_hl_results"]
    if not st.session_state.hh_hl_weekly_results and bg_status.get("hh_hl_weekly_results"):
        st.session_state.hh_hl_weekly_results = bg_status["hh_hl_weekly_results"]

    # Auto-load from DB cache if still missing
    if not st.session_state.hh_hl_results:
        try:
            all_latest = database.get_all_latest_scan_dates()
            hh_date = all_latest.get('scanned_hh_hl') or database.get_best_scan_date()
            if hh_date:
                cached_daily = database.get_cached_hh_hl(hh_date)
                if cached_daily:
                    st.session_state.hh_hl_results = cached_daily
                cached_weekly = database.get_cached_hh_hl_weekly(hh_date)
                if cached_weekly:
                    st.session_state.hh_hl_weekly_results = cached_weekly
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # 3. SCAN CONTROLS (Run Real-Time Scan)
    # ─────────────────────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 2, 2])
    with ctrl_col1:
        timeframe_sel = st.selectbox(
            "📅 Analysis Timeframe:",
            ["Daily (Swing Trading)", "Weekly (Positional / Macro)"],
            index=0,
            key="hh_hl_timeframe_selector"
        )
    with ctrl_col2:
        universe_sel = st.selectbox(
            "🌐 Universe:",
            ["Top 1000 NSE Stocks", "NIFTY 500", "NIFTY 100", "NIFTY 50", "Watchlist"],
            index=0,
            key="hh_hl_universe_selector"
        )
    with ctrl_col3:
        st.write("")
        st.write("")
        run_scan_btn = st.button("🔍 Run HH/HL Scan", type="primary", use_container_width=True, key="run_hh_hl_scan_btn")

    if run_scan_btn:
        with st.spinner(f"Running HH/HL Uptrend Scan on {universe_sel} ({timeframe_sel})..."):
            import yfinance as yf
            from data_fetcher import get_index_stocks, get_top1000_nse_symbols

            if "Top 1000" in universe_sel:
                raw_symbols = get_top1000_nse_symbols()
            elif "NIFTY 500" in universe_sel:
                raw_symbols = get_index_stocks("NIFTY 500")
            elif "NIFTY 100" in universe_sel:
                raw_symbols = get_index_stocks("NIFTY 100")
            elif "NIFTY 50" in universe_sel:
                raw_symbols = get_index_stocks("NIFTY 50")
            else:
                raw_symbols = get_index_stocks("WATCHLIST")

            all_syms = [s if s.endswith('.NS') else f"{s}.NS" for s in raw_symbols if str(s).strip()]
            period_str = "1y" if "Daily" in timeframe_sel else "2y"
            interval_str = "1d" if "Daily" in timeframe_sel else "1wk"

            progress_bar = st.progress(0.0)
            status_box = st.empty()

            results_list = []
            chunk_size = 50
            chunks = [all_syms[i:i + chunk_size] for i in range(0, len(all_syms), chunk_size)]

            for c_idx, chunk in enumerate(chunks):
                status_box.text(f"Scanning batch {c_idx + 1}/{len(chunks)} ({len(chunk)} stocks)...")
                try:
                    data = yf.download(chunk, period=period_str, interval=interval_str, progress=False, timeout=18)
                    for sym in chunk:
                        clean_sym = sym.replace('.NS', '')
                        try:
                            if isinstance(data.columns, pd.MultiIndex):
                                all_tkrs = data.columns.get_level_values(1).unique().tolist()
                                matched_t = next((t for t in all_tkrs if t.upper() == sym.upper()), None)
                                if not matched_t:
                                    continue
                                df = data.xs(matched_t, axis=1, level=1).copy()
                            else:
                                if len(chunk) == 1:
                                    df = data.copy()
                                else:
                                    continue
                            df = df.dropna(subset=['Close'])
                            if len(df) >= 50:
                                res = scan_hh_hl_uptrend(clean_sym, df)
                                if res:
                                    results_list.append(res)
                        except Exception:
                            pass
                except Exception:
                    pass
                progress_bar.progress((c_idx + 1) / len(chunks))

            today_str = get_market_date()
            if "Daily" in timeframe_sel:
                st.session_state.hh_hl_results = results_list
                try:
                    database.save_hh_hl_only(today_str, results_list)
                except Exception:
                    pass
            else:
                st.session_state.hh_hl_weekly_results = results_list
                try:
                    database.save_hh_hl_weekly_only(today_str, results_list)
                except Exception:
                    pass

            status_box.text(f"✅ Scan Complete! Found {len(results_list)} uptrend stocks making Higher Highs & Higher Lows.")
            st.rerun()

    # Active dataset
    active_data = st.session_state.hh_hl_results if "Daily" in timeframe_sel else st.session_state.hh_hl_weekly_results
    if not active_data:
        st.info("💡 No HH/HL scan data loaded yet. Click **'🔍 Run HH/HL Scan'** above or run the scanner from the sidebar.")
        return

    # ── Baseline quality filter applied to ALL cached data ─────────────────────
    # Ensures KPI counts reflect quality-gated universe even when loading old DB rows.
    # Mirrors scanner Gate 1 (CMP > 50 SMA), Gate 2 (ADX ≥ 20), Gate 3 (weekly OR ADX ≥ 25), Gate 4 (score ≥ 40)
    active_data = [
        r for r in active_data
        if r.get('trend_score', 0) >= 40
        and r.get('adx', 0) >= 20
        and float(r.get('cmp', 0)) >= float(r.get('sma50', 0)) * 0.995  # small tolerance for rounding
        and (r.get('weekly_confirmed', False) or r.get('adx', 0) >= 25)
    ]
    if not active_data:
        st.info("💡 No high-quality HH/HL setups found in cached data. Run a fresh scan to get updated results.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 4. KPI SUMMARY METRIC CARDS
    # ─────────────────────────────────────────────────────────────────────────
    total_count = len(active_data)
    breakout_count = sum(1 for r in active_data if "Breakout" in r.get('setup_tag', ''))
    consolidating_count = sum(1 for r in active_data if "Consolidating" in r.get('setup_tag', ''))
    pullback_count = sum(1 for r in active_data if "Pullback" in r.get('setup_tag', '') or "Testing" in r.get('setup_tag', ''))
    pure_trend_count = sum(1 for r in active_data if "Confirmed" in r.get('setup_tag', ''))
    avg_score = round(sum(r.get('trend_score', 0) for r in active_data) / max(total_count, 1), 1)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #38bdf8;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">Total Uptrend</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#38bdf8;">{total_count}</h3><span style="font-size:0.68rem; color:#64748b;">Quality HH/HL</span></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #f43f5e;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">🔥 HH Breakout</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#f43f5e;">{breakout_count}</h3><span style="font-size:0.68rem; color:#64748b;">At/Above Peak</span></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #38bdf8;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">⚡ Near Peak</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#38bdf8;">{consolidating_count}</h3><span style="font-size:0.68rem; color:#64748b;">Within 2% High</span></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #10b981;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">🛡️ HL Pullback</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#10b981;">{pullback_count}</h3><span style="font-size:0.68rem; color:#64748b;">Dip Buy Setup</span></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #818cf8;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">📈 Pure Uptrend</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#818cf8;">{pure_trend_count}</h3><span style="font-size:0.68rem; color:#64748b;">Ascending Waves</span></div>', unsafe_allow_html=True)
    k6.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #fbbf24;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">⭐ Avg Score</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#fbbf24;">{avg_score}</h3><span style="font-size:0.68rem; color:#64748b;">Out of 100</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. FILTERING & SORTING CONTROLS
    # ─────────────────────────────────────────────────────────────────────────
    f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns([2.5, 2, 2, 2, 2.5])
    with f_col1:
        quality_tier = st.selectbox(
            "⭐ Quality Tier:",
            [
                "⭐⭐⭐ Institutional (Score≥75 + Weekly)",
                "⭐⭐ Strong (Score≥55 + ADX≥20)",
                "⭐ All Uptrends",
            ],
            index=1,
            key="hh_hl_quality_tier"
        )
    with f_col2:
        setup_filter = st.selectbox(
            "Filter by Setup:",
            [
                f"All Setups ({total_count})",
                f"🔥 Fresh HH Breakout ({breakout_count})",
                f"⚡ Consolidating near Peak ({consolidating_count})",
                f"🛡️ HL Pullback / Support Test ({pullback_count})",
                f"📈 Confirmed Pure Uptrend ({pure_trend_count})"
            ],
            key="hh_hl_setup_filter"
        )
    with f_col3:
        min_cycles = st.selectbox(
            "Min HH/HL Cycles:",
            ["All Cycles (≥1)", "2+ Cycles", "3+ Cycles", "4+ Cycles"],
            index=0,
            key="hh_hl_min_cycles_sel"
        )
    with f_col4:
        sort_by = st.selectbox(
            "Sort By:",
            [
                "Trend Score ↓",
                "Uptrend Days ↓ (Established)",
                "Uptrend Days ↑ (Fresh/Early)",
                "Day Change % ↓",
                "Volume Ratio ↓",
                "CMP ↑",
                "Distance to High ↑"
            ],
            index=0,
            key="hh_hl_sort_by_sel"
        )
    with f_col5:
        search_sym = st.text_input("🔍 Search Symbol:", "", placeholder="e.g. RELIANCE, INFY...", key="hh_hl_sym_search").strip().upper()

    # Apply filters
    filtered = []
    for r in active_data:
        tag = r.get('setup_tag', '')

        # ── Quality Tier gate (applied first) ──────────────────────────────
        ts   = r.get('trend_score', 0)
        adxv = r.get('adx', 0)
        wkly = r.get('weekly_confirmed', False)
        if "Institutional" in quality_tier:
            if ts < 75 or not wkly:
                continue
        elif "Strong" in quality_tier:
            if ts < 55 or adxv < 20:
                continue
        # "All Uptrends" → no extra gate

        if "Breakout" in setup_filter and "Breakout" not in tag:
            continue
        if "near Peak" in setup_filter and "Consolidating" not in tag:
            continue
        if "HL Pullback" in setup_filter and "Pullback" not in tag and "Testing" not in tag:
            continue
        if "Pure Uptrend" in setup_filter and "Confirmed" not in tag:
            continue

        hh_cnt = r.get('hh_count', 0)
        hl_cnt = r.get('hl_count', 0)
        if "2+ Cycles" in min_cycles and (hh_cnt < 2 or hl_cnt < 2):
            continue
        if "3+ Cycles" in min_cycles and (hh_cnt < 3 or hl_cnt < 3):
            continue
        if "4+ Cycles" in min_cycles and (hh_cnt < 4 or hl_cnt < 4):
            continue

        if search_sym and search_sym not in r.get('symbol', '').upper():
            continue
        _max_cmp_hh = float(st.session_state.get("global_max_cmp", 1000))
        if _max_cmp_hh > 0 and float(r.get('cmp') or 0.0) > _max_cmp_hh:
            continue

        filtered.append(r)

    # Sort
    if sort_by == "Trend Score ↓":
        filtered.sort(key=lambda x: x.get('trend_score', 0), reverse=True)
    elif sort_by == "Uptrend Days ↓ (Established)":
        filtered.sort(key=lambda x: x.get('uptrend_days', 0), reverse=True)
    elif sort_by == "Uptrend Days ↑ (Fresh/Early)":
        filtered.sort(key=lambda x: x.get('uptrend_days', 9999))
    elif sort_by == "Day Change % ↓":
        filtered.sort(key=lambda x: x.get('day_change_pct', 0), reverse=True)
    elif sort_by == "Volume Ratio ↓":
        filtered.sort(key=lambda x: x.get('vol_ratio', 1.0), reverse=True)
    elif sort_by == "CMP ↑":
        filtered.sort(key=lambda x: x.get('cmp', 0))
    elif sort_by == "Distance to High ↑":
        filtered.sort(key=lambda x: abs(x.get('dist_to_sh_pct', 0)))

    st.markdown(f"<p style='color:#94a3b8; font-size:0.85rem;'>Showing <b>{len(filtered)}</b> matching stocks</p>", unsafe_allow_html=True)

    if not filtered:
        st.info("No stocks match the selected filter criteria.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 6. RESULTS TABLE DISPLAY
    # ─────────────────────────────────────────────────────────────────────────
    df_disp = pd.DataFrame(filtered)
    # Add clickable TradingView URL
    df_disp['tv_url'] = df_disp['symbol'].apply(lambda s: f"https://in.tradingview.com/chart/?symbol=NSE:{s}")
    df_disp['weekly_icon'] = df_disp['weekly_confirmed'].apply(lambda x: "✅ Weekly HH+HL" if x else "—")

    # Scan streak from database
    scan_streaks = database.get_hh_hl_scan_streaks() if hasattr(database, 'get_hh_hl_scan_streaks') else {}
    df_disp['scan_streak'] = df_disp['symbol'].map(scan_streaks).fillna(1).astype(int)
    df_disp['scan_streak_badge'] = df_disp['scan_streak'].apply(
        lambda s: f"🔥 {s}d streak" if s >= 3 else (f"🟢 {s}d" if s >= 2 else "1d")
    )

    # Ensure numeric and string columns
    for col in ['entry', 'stop_loss', 'target1', 'target2', 'risk_pct', 'adx', 'uptrend_days', 'uptrend_bars']:
        if col not in df_disp.columns:
            df_disp[col] = 0.0

    if 'uptrend_start_date' not in df_disp.columns:
        df_disp['uptrend_start_date'] = ''

    df_disp['uptrend_age_str'] = df_disp.apply(
        lambda r: f"🔥 {int(r['uptrend_days'])}d" if r['uptrend_days'] >= 30
        else (f"⚡ {int(r['uptrend_days'])}d" if r['uptrend_days'] > 0 else "—"),
        axis=1
    )

    st.dataframe(
        df_disp[[
            'tv_url', 'company_name', 'setup_tag', 'cmp', 'day_change_pct',
            'uptrend_age_str', 'uptrend_start_date', 'scan_streak_badge',
            'entry', 'stop_loss', 'target1', 'target2', 'risk_pct',
            'hh_count', 'hl_count', 'last_swing_high', 'last_swing_low',
            'trend_score', 'vol_ratio', 'adx', 'weekly_icon', 'ma_alignment'
        ]],
        use_container_width=True,
        column_config={
            "tv_url": st.column_config.LinkColumn(
                "Stock (TradingView)",
                display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)",
                width="medium"
            ),
            "company_name": st.column_config.TextColumn("Company", width="medium"),
            "setup_tag": st.column_config.TextColumn("Setup State", width="medium"),
            "cmp": st.column_config.NumberColumn("CMP", format="₹ %.2f", width="small"),
            "day_change_pct": st.column_config.NumberColumn("Chg %", format="%.2f %%", width="small"),
            "uptrend_age_str": st.column_config.TextColumn(
                "Uptrend Age",
                help="Total calendar days since the origin Higher Low of this ascending wave formed",
                width="small"
            ),
            "uptrend_start_date": st.column_config.TextColumn(
                "Trend Since",
                help="Date when the ascending wave started",
                width="small"
            ),
            "scan_streak_badge": st.column_config.TextColumn(
                "Scan Streak",
                help="Consecutive daily scans this stock appeared in pure uptrend",
                width="small"
            ),
            "entry": st.column_config.NumberColumn("Entry Trigger", format="₹ %.2f", width="small"),
            "stop_loss": st.column_config.NumberColumn("Stop Loss", format="₹ %.2f", width="small"),
            "target1": st.column_config.NumberColumn("Target 1 (2R)", format="₹ %.2f", width="small"),
            "target2": st.column_config.NumberColumn("Target 2 (3R)", format="₹ %.2f", width="small"),
            "risk_pct": st.column_config.NumberColumn("Risk %", format="%.1f %%", width="small"),
            "hh_count": st.column_config.NumberColumn("HHs", width="small"),
            "hl_count": st.column_config.NumberColumn("HLs", width="small"),
            "last_swing_high": st.column_config.NumberColumn("Swing High", format="₹ %.2f", width="small"),
            "last_swing_low": st.column_config.NumberColumn("Swing Low", format="₹ %.2f", width="small"),
            "trend_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d", width="small"),
            "vol_ratio": st.column_config.NumberColumn("Vol Ratio", format="%.2fx", width="small"),
            "adx": st.column_config.NumberColumn("ADX", format="%.1f", width="small"),
            "weekly_icon": st.column_config.TextColumn("Weekly Trend", width="small"),
            "ma_alignment": st.column_config.TextColumn("MA Stack", width="medium")
        },
        hide_index=True,
        height=min(450, 45 + len(filtered) * 38)
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 7. INTERACTIVE SWING CHART INSPECTOR
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 Interactive Higher High / Higher Low Chart Inspector")
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.85rem; margin-top:-4px;'>"
        "Select any stock from the list to visualize its candlestick chart with marked <b>Higher High peaks (▲ HH)</b>, "
        "<b>Higher Low troughs (▼ HL)</b>, 20 & 50 SMAs, and resistance/support lines."
        "</p>",
        unsafe_allow_html=True
    )

    stock_options = [r['symbol'] for r in filtered]
    sel_chart_col1, sel_chart_col2 = st.columns([2, 5])
    with sel_chart_col1:
        sel_symbol = st.selectbox("Select Stock to Inspect:", stock_options, key="hh_hl_chart_sym_sel")

    stock_row = next((r for r in filtered if r['symbol'] == sel_symbol), None)

    if stock_row:
        # Display detail metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CMP", f"₹{stock_row['cmp']:,.2f}", f"{stock_row['day_change_pct']:+.2f}%")
        c2.metric("Latest Swing High (Resistance)", f"₹{stock_row['last_swing_high']:,.2f}", f"{stock_row['dist_to_sh_pct']:+.1f}%")
        c3.metric("Latest Swing Low (Support / SL)", f"₹{stock_row['last_swing_low']:,.2f}", f"+{stock_row['dist_to_sl_pct']:.1f}% cushion")
        c4.metric("Dow Trend Score", f"{stock_row['trend_score']}/100", f"{stock_row['hh_count']} HH / {stock_row['hl_count']} HL")

        u_days = stock_row.get('uptrend_days') or 0
        u_bars = stock_row.get('uptrend_bars') or 0
        u_date = stock_row.get('uptrend_start_date') or ''
        if u_days > 0 and u_date:
            st.caption(f"🗓️ **Pure Uptrend Duration**: **{u_days} calendar days** ({u_bars} trading sessions) — ascending Higher Low sequence active since base pivot on **{u_date}**.")

        # Fetch recent historical data to render chart
        import yfinance as yf
        try:
            sym_ns = sel_symbol if sel_symbol.endswith('.NS') else f"{sel_symbol}.NS"
            interval = "1d" if "Daily" in timeframe_sel else "1wk"
            hist_df = yf.download(sym_ns, period="8mo", interval=interval, progress=False, timeout=12)
            if hist_df is not None and not hist_df.empty:
                if isinstance(hist_df.columns, pd.MultiIndex):
                    if 'Close' in hist_df.columns.get_level_values(0):
                        hist_df.columns = hist_df.columns.get_level_values(0)
                    elif 'Close' in hist_df.columns.get_level_values(1):
                        hist_df.columns = hist_df.columns.get_level_values(1)
                    else:
                        hist_df.columns = hist_df.columns.get_level_values(-1)
                hist_df = hist_df.loc[:, ~hist_df.columns.duplicated()]

                # Volume bar coloring in render_swing_chart
                sh_list = json.loads(stock_row.get('swing_highs_json', '[]'))
                sl_list = json.loads(stock_row.get('swing_lows_json', '[]'))
                render_swing_chart(
                    sel_symbol, hist_df, sh_list, sl_list,
                    float(stock_row.get('last_swing_high') or 0.0),
                    float(stock_row.get('last_swing_low') or 0.0)
                )
                # Render 10 Rules & Strategy A Execution Blueprint
                render_10_rules_execution_card(stock_row, df=hist_df)
            else:
                st.warning("Could not download recent candle data for chart.")
        except Exception as _ce:
            st.warning(f"Error fetching chart: {_ce}")

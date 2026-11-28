import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

from config import IST_TIMEZONE
import database
from scanner import scan_wt_cross


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


def render_wavetrend_chart(symbol: str, df: pd.DataFrame, oversold_thresh: float = -40.0):
    """Renders Plotly candlestick chart with 20/50/200 SMAs and WaveTrend (WT1 & WT2) subplot."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    chart_df = df.tail(120).copy()
    if chart_df.empty or len(chart_df) < 25:
        st.warning("Insufficient historical data to render chart.")
        return

    # Flatten MultiIndex columns if present (from yfinance multi-column downloads)
    if isinstance(chart_df.columns, pd.MultiIndex):
        chart_df.columns = chart_df.columns.get_level_values(0)

    # Make index tz-naive for consistent comparisons
    try:
        if hasattr(chart_df.index, 'tz') and chart_df.index.tz is not None:
            chart_df.index = chart_df.index.tz_localize(None)
    except Exception:
        pass

    # 1. Moving Averages
    sma20 = chart_df['Close'].rolling(20).mean()
    sma50 = chart_df['Close'].rolling(50).mean()
    sma200 = chart_df['Close'].rolling(200).mean() if len(df) >= 200 else pd.Series(dtype=float)

    # 2. WaveTrend calculation
    ap = (chart_df['High'] + chart_df['Low'] + chart_df['Close']) / 3.0
    esa = ap.ewm(span=10, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=10, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d + 1e-10)
    wt1 = ci.ewm(span=21, adjust=False).mean()
    wt2 = wt1.rolling(window=4).mean()

    # Find crossover points (green dots)
    cross_dates = []
    cross_wt1 = []
    for i in range(1, len(wt1)):
        if pd.notna(wt1.iloc[i]) and pd.notna(wt2.iloc[i]) and pd.notna(wt1.iloc[i-1]) and pd.notna(wt2.iloc[i-1]):
            if wt1.iloc[i-1] <= wt2.iloc[i-1] and wt1.iloc[i] > wt2.iloc[i] and wt1.iloc[i] <= (oversold_thresh + 10):
                cross_dates.append(chart_df.index[i])
                cross_wt1.append(wt1.iloc[i])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.35],
        subplot_titles=[f"{symbol} Price & SMAs", "🌊 WaveTrend with Crosses (LazyBear)"]
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name="Price",
            increasing_line_color='#00e676',
            decreasing_line_color='#ef4444'
        ),
        row=1, col=1
    )

    # SMAs
    fig.add_trace(go.Scatter(x=chart_df.index, y=sma20, mode='lines', line=dict(color='#38bdf8', width=1.3), name="20 SMA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=sma50, mode='lines', line=dict(color='#fbbf24', width=1.5), name="50 SMA"), row=1, col=1)
    if len(sma200) > 0 and pd.notna(sma200.iloc[-1]):
        fig.add_trace(go.Scatter(x=chart_df.index, y=sma200, mode='lines', line=dict(color='#f43f5e', width=1.5), name="200 SMA"), row=1, col=1)

    # WaveTrend lines
    fig.add_trace(go.Scatter(x=chart_df.index, y=wt1, mode='lines', line=dict(color='#00e676', width=1.8), name="WT1 (Fast)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=wt2, mode='lines', line=dict(color='#ef4444', width=1.4), name="WT2 (Signal)"), row=2, col=1)

    # Threshold horizontal reference lines
    fig.add_hline(y=oversold_thresh, line_dash="dash", line_color="rgba(0, 230, 118, 0.6)", annotation_text=f"Oversold ({oversold_thresh})", row=2, col=1)
    fig.add_hline(y=-60.0, line_dash="dot", line_color="rgba(239, 68, 68, 0.5)", annotation_text="Extreme (-60)", row=2, col=1)
    fig.add_hline(y=60.0, line_dash="dash", line_color="rgba(244, 63, 94, 0.5)", annotation_text="Overbought (+60)", row=2, col=1)
    fig.add_hline(y=0.0, line_dash="solid", line_color="rgba(148, 163, 184, 0.2)", row=2, col=1)

    # Green dot buy signals
    if cross_dates:
        fig.add_trace(
            go.Scatter(
                x=cross_dates, y=cross_wt1,
                mode='markers',
                marker=dict(symbol='circle', size=9, color='#00e676', line=dict(width=1.5, color='#ffffff')),
                name="🟢 Buy Cross"
            ),
            row=2, col=1
        )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(10,15,30,0.8)",
        plot_bgcolor="rgba(10,15,30,0.8)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render():
    st.markdown("### 🌊 WaveTrend Oversold Scanner [LazyBear WT Cross]")
    st.markdown(
        "<p style='font-size:0.92rem; color:#94a3b8; margin-top:-6px;'>"
        "Detects deeply oversold mean-reversion candidates using <b>LazyBear's WaveTrend with Crosses</b>. "
        "Flags high-probability turning points where <b>WT1 crosses above WT2 (🟢 Green Dot)</b> in the oversold zone."
        "</p>",
        unsafe_allow_html=True
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 1. EDUCATIONAL GUIDE EXPANDER
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("📖 WaveTrend Oscillator & Trading Strategy Explained", expanded=False):
        st.markdown("""
### 🧠 How WaveTrend with Crosses Works
The **WaveTrend oscillator** (popularized by LazyBear) is an institutional-grade momentum oscillator:
- **WT1 (Green line)**: The smoothed trend oscillator based on typical price deviation ($HLC3$) from its 10-period EMA.
- **WT2 (Red line)**: A 4-period SMA of WT1, acting as a signal trigger line.
- **Oversold Zone (WT1 ≤ -40)**: Indicates exhaustion of intense selling pressure.
- **Extreme Oversold (WT1 ≤ -60)**: Severe panic-selling stretch; prime rubber-band rebound territory.
- **🟢 Buy Signal (Green Dot)**: Triggered when **WT1 crosses ABOVE WT2** from inside or near the oversold zone.
- **📈 Bullish Divergence**: Price prints a lower low while WaveTrend prints a higher low — strong institutional accumulation sign.
""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. SCAN CONTROLS
    # ─────────────────────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2.5, 2.5, 2, 2.5])
    with ctrl_col1:
        wt_timeframe = st.selectbox(
            "📅 Timeframe:",
            ["Daily (Swing Trading)", "Weekly (Positional)", "Monthly (Macro)", "1 Hour (Intraday)", "15 Min (Scalp)"],
            index=0,
            key="wt_tab_timeframe_selector"
        )
    with ctrl_col2:
        wt_universe = st.selectbox(
            "🌐 Universe:",
            ["Top 1000 NSE Stocks", "NIFTY 500", "NIFTY 100", "NIFTY 50", "Watchlist"],
            index=0,
            key="wt_tab_universe_selector"
        )
    with ctrl_col3:
        default_thresh = -20.0 if ("Weekly" in wt_timeframe or "Monthly" in wt_timeframe) else -40.0
        wt_oversold_threshold = st.number_input(
            "📉 Oversold Cutoff:",
            min_value=-90.0,
            max_value=0.0,
            value=default_thresh,
            step=5.0,
            key=f"wt_oversold_threshold_input_{wt_timeframe[:4]}",
            help="Stocks with WT1 below this value are considered oversold. Default: -40 for daily, -20 for weekly."
        )
    with ctrl_col4:
        st.write("")
        st.write("")
        run_wt_btn = st.button("🌊 Run WaveTrend Scan", type="primary", use_container_width=True, key="run_wt_scan_btn")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. SCAN EXECUTION
    # ─────────────────────────────────────────────────────────────────────────
    if run_wt_btn:
        with st.spinner(f"Running WaveTrend scan on {wt_universe} ({wt_timeframe}, WT1 ≤ {wt_oversold_threshold})..."):
            from data_fetcher import get_top1000_nse_symbols, get_index_stocks
            from local_cache_manager import bulk_get_cached_ohlcv, resample_ohlcv
            import yfinance as yf

            if "Top 1000" in wt_universe:
                raw_syms = get_top1000_nse_symbols()
            elif "NIFTY 500" in wt_universe:
                raw_syms = get_index_stocks("NIFTY 500")
            elif "NIFTY 100" in wt_universe:
                raw_syms = get_index_stocks("NIFTY 100")
            elif "NIFTY 50" in wt_universe:
                raw_syms = get_index_stocks("NIFTY 50")
            else:
                raw_syms = get_index_stocks("WATCHLIST")

            clean_syms = [s.strip().upper().replace('.NS', '') for s in raw_syms if str(s).strip()]

            progress_bar = st.progress(0.0)
            status_box = st.empty()
            results = []

            # Determine execution strategy
            if "Daily" in wt_timeframe:
                status_box.text("Loading daily data from Master Cache...")
                bulk_data = bulk_get_cached_ohlcv(clean_syms, "1d")
                total = len(clean_syms)
                for idx, sym in enumerate(clean_syms):
                    df = bulk_data.get(sym)
                    if df is not None and len(df) >= 40:
                        res = scan_wt_cross(sym, df, wt_oversold_threshold=float(wt_oversold_threshold))
                        if res:
                            res['timeframe'] = "Daily"
                            results.append(res)
                    if idx % 50 == 0 or idx == total - 1:
                        progress_bar.progress((idx + 1) / max(total, 1))

                # Save daily results to database
                today_str = get_market_date()
                st.session_state.wt_results = results
                try:
                    database.save_wt_cross_only(today_str, results)
                except Exception:
                    pass

            elif "Weekly" in wt_timeframe or "Monthly" in wt_timeframe:
                status_box.text(f"Resampling master cache data to {wt_timeframe}...")
                bulk_data = bulk_get_cached_ohlcv(clean_syms, "1d")
                freq = '1W-MON' if "Weekly" in wt_timeframe else '1ME'
                min_b = 25 if "Weekly" in wt_timeframe else 15
                total = len(clean_syms)
                for idx, sym in enumerate(clean_syms):
                    df = bulk_data.get(sym)
                    if df is not None and len(df) >= 50:
                        try:
                            eval_df = resample_ohlcv(df, freq)
                            if eval_df is not None and len(eval_df) >= min_b:
                                res = scan_wt_cross(sym, eval_df, wt_oversold_threshold=float(wt_oversold_threshold), min_bars=min_b)
                                if res:
                                    res['timeframe'] = "Weekly" if "Weekly" in wt_timeframe else "Monthly"
                                    results.append(res)
                        except Exception:
                            pass
                    if idx % 50 == 0 or idx == total - 1:
                        progress_bar.progress((idx + 1) / max(total, 1))

            else:
                # Intraday (1 Hour or 15 Min)
                interval = "60m" if "1 Hour" in wt_timeframe else "15m"
                period = "60d" if "1 Hour" in wt_timeframe else "30d"
                chunk_size = 40
                chunks = [clean_syms[i:i + chunk_size] for i in range(0, len(clean_syms), chunk_size)]
                for c_idx, chunk in enumerate(chunks):
                    status_box.text(f"Fetching intraday data batch {c_idx+1}/{len(chunks)} ({len(chunk)} stocks)...")
                    chunk_ns = [f"{s}.NS" for s in chunk]
                    try:
                        data = yf.download(tickers=chunk_ns, period=period, interval=interval, progress=False, timeout=18)
                        for sym in chunk:
                            sym_ns = f"{sym}.NS"
                            try:
                                if isinstance(data.columns, pd.MultiIndex):
                                    all_t = data.columns.get_level_values(1).unique().tolist()
                                    matched = next((t for t in all_t if t.upper() == sym_ns.upper()), None)
                                    if not matched:
                                        continue
                                    s_df = data.xs(matched, axis=1, level=1).dropna(subset=['Close']).copy()
                                else:
                                    s_df = data.dropna(subset=['Close']).copy() if len(chunk) == 1 else None
                                if s_df is not None and len(s_df) >= 30:
                                    res = scan_wt_cross(sym, s_df, wt_oversold_threshold=float(wt_oversold_threshold), min_bars=25)
                                    if res:
                                        res['timeframe'] = wt_timeframe
                                        results.append(res)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    progress_bar.progress((c_idx + 1) / len(chunks))

            tf_key = f"{wt_timeframe[:4]}_{wt_oversold_threshold}"
            if 'wt_results_by_tf' not in st.session_state:
                st.session_state.wt_results_by_tf = {}
            st.session_state.wt_results_by_tf[tf_key] = results

            status_box.text(f"✅ Scan Complete! Found {len(results)} oversold stocks (WT1 ≤ {wt_oversold_threshold}).")
            st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # 4. DATA HYDRATION
    # ─────────────────────────────────────────────────────────────────────────
    if 'wt_results' not in st.session_state:
        st.session_state.wt_results = None
    if 'wt_results_by_tf' not in st.session_state:
        st.session_state.wt_results_by_tf = {}

    # Check background scan status
    bg_status = st.session_state.get('ALL_TAB_SCAN_STATUS', {})
    if not st.session_state.wt_results and bg_status.get("wt_results"):
        st.session_state.wt_results = bg_status["wt_results"]

    # Auto-load from database cache for Daily
    if not st.session_state.wt_results:
        try:
            latest_dates = database.get_all_latest_scan_dates()
            wt_date = latest_dates.get('scanned_wt_cross') or database.get_best_scan_date()
            if wt_date:
                cached = database.get_cached_wt_cross(wt_date)
                if cached:
                    st.session_state.wt_results = cached
        except Exception:
            pass

    # Active dataset
    tf_key = f"{wt_timeframe[:4]}_{wt_oversold_threshold}"
    active_data = None
    if tf_key in st.session_state.wt_results_by_tf:
        active_data = st.session_state.wt_results_by_tf[tf_key]
    elif "Daily" in wt_timeframe and st.session_state.wt_results:
        # Dynamically filter daily results by current threshold
        active_data = [r for r in st.session_state.wt_results if float(r.get('wt_value', 0)) <= float(wt_oversold_threshold)]

    if active_data is None:
        if bg_status.get("is_running", False):
            st.info("🔄 Background all-tab scan is running. WaveTrend results will appear here once ready.")
        else:
            st.info("💡 No WaveTrend data loaded yet. Click **'🌊 Run WaveTrend Scan'** above to run the scan.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 5. KPI METRICS
    # ─────────────────────────────────────────────────────────────────────────
    total_count = len(active_data)
    buy_count = sum(1 for r in active_data if r.get('buy_signal', False))
    div_count = sum(1 for r in active_data if r.get('bullish_divergence', False))
    above_200_count = sum(1 for r in active_data if r.get('above_200sma', False))
    deepest_val = round(min((float(r.get('wt_value', 0)) for r in active_data), default=0.0), 1)
    avg_val = round(sum(float(r.get('wt_value', 0)) for r in active_data) / max(total_count, 1), 1)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #38bdf8;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">Oversold Stocks</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#38bdf8;">{total_count}</h3><span style="font-size:0.68rem; color:#64748b;">WT1 ≤ {wt_oversold_threshold}</span></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #00e676;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">🟢 Buy Signals</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#00e676;">{buy_count}</h3><span style="font-size:0.68rem; color:#64748b;">Green Dot Cross</span></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #fbbf24;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">📈 Bullish Div</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#fbbf24;">{div_count}</h3><span style="font-size:0.68rem; color:#64748b;">Divergence</span></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #10b981;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">🛡️ Above 200 SMA</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#10b981;">{above_200_count}</h3><span style="font-size:0.68rem; color:#64748b;">Uptrend Bounce</span></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #ef4444;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">Deepest WT1</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#ef4444;">{deepest_val}</h3><span style="font-size:0.68rem; color:#64748b;">Most Oversold</span></div>', unsafe_allow_html=True)
    k6.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #38bdf8;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">Avg WT1</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#38bdf8;">{avg_val}</h3><span style="font-size:0.68rem; color:#64748b;">Average Score</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. FILTERING & SORTING CONTROLS
    # ─────────────────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns([3, 2, 3])
    with f1:
        signal_filter = st.selectbox(
            "Filter Setups:",
            [
                f"All Oversold ({total_count})",
                f"🟢 Buy Signals Only ({buy_count})",
                f"📈 Bullish Divergence Only ({div_count})",
                f"🛡️ Above 200 DMA Only ({above_200_count})",
                "📈 Above 20 SMA Only",
                "🛡️ Above 50 SMA Only"
            ],
            key="wt_signal_filter_sel"
        )
    with f2:
        sort_by = st.selectbox(
            "Sort By:",
            ["WT1 ↑ (Deepest Oversold)", "WT Diff ↑ (Strongest Cross)", "Day Change % ↓", "CMP ↑"],
            key="wt_sort_by_sel"
        )
    with f3:
        search_sym = st.text_input("🔍 Search Symbol:", "", placeholder="e.g. RELIANCE, INFY...", key="wt_sym_search").strip().upper()

    # Apply filters
    filtered = []
    for r in active_data:
        if "Buy Signals" in signal_filter and not r.get('buy_signal', False):
            continue
        if "Divergence" in signal_filter and not r.get('bullish_divergence', False):
            continue
        if "Above 200 DMA" in signal_filter and not r.get('above_200sma', False):
            continue
        if "Above 20 SMA" in signal_filter and not r.get('above_20sma', False):
            continue
        if "Above 50 SMA" in signal_filter and not r.get('above_50sma', False):
            continue
        if search_sym and search_sym not in str(r.get('symbol', '')).upper():
            continue
        _max_cmp_wt = float(st.session_state.get("global_max_cmp", 1000))
        if _max_cmp_wt > 0 and float(r.get('cmp') or 0.0) > _max_cmp_wt:
            continue
        filtered.append(r)

    # Sort
    if sort_by == "WT1 ↑ (Deepest Oversold)":
        filtered.sort(key=lambda x: float(x.get('wt_value', 0.0)))
    elif sort_by == "WT Diff ↑ (Strongest Cross)":
        filtered.sort(key=lambda x: float(x.get('wt_diff', 0.0)), reverse=True)
    elif sort_by == "Day Change % ↓":
        filtered.sort(key=lambda x: float(x.get('day_change_pct', 0.0)), reverse=True)
    elif sort_by == "CMP ↑":
        filtered.sort(key=lambda x: float(x.get('cmp', 0.0)))

    st.markdown(f"<p style='color:#94a3b8; font-size:0.85rem;'>Showing <b>{len(filtered)}</b> matching stocks</p>", unsafe_allow_html=True)

    if not filtered:
        st.info("No stocks match the selected filter criteria.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 7. RESULTS TABLE
    # ─────────────────────────────────────────────────────────────────────────
    df_disp = pd.DataFrame(filtered)
    df_disp['tv_url'] = df_disp['symbol'].apply(lambda s: f"https://in.tradingview.com/chart/?symbol=NSE:{str(s).replace('.NS', '')}")

    # Guard against missing columns from old cached DB rows
    for _col, _default in [
        ('bullish_divergence', False),
        ('buy_signal', False),
        ('above_200sma', False),
        ('above_50sma', False),
        ('above_20sma', False),
        ('company_name', ''),
        ('wt2_value', 0.0),
        ('wt_diff', 0.0),
        ('buy_price', 0.0),
        ('exit_price', 0.0),
        ('target_price', 0.0),
        ('confidence', 'N/A'),
    ]:
        if _col not in df_disp.columns:
            df_disp[_col] = _default

    df_disp['signal_badge'] = df_disp['buy_signal'].apply(
        lambda b: "🟢 BUY SIGNAL" if b else "⚪ Oversold"
    )
    df_disp['div_badge'] = df_disp['bullish_divergence'].apply(
        lambda d: "📈 Bullish Div" if d else "—"
    )
    df_disp['trend_context'] = df_disp.apply(
        lambda r: "🛡️ Above 200 DMA" if r.get('above_200sma')
        else ("📈 Above 50 SMA" if r.get('above_50sma')
              else ("Above 20 SMA" if r.get('above_20sma') else "Below 20/50/200")),
        axis=1
    )

    st.dataframe(
        df_disp[[
            'tv_url', 'company_name', 'cmp', 'day_change_pct', 'wt_value', 'wt2_value', 'wt_diff',
            'signal_badge', 'div_badge', 'trend_context', 'buy_price', 'exit_price', 'target_price', 'confidence'
        ]],
        use_container_width=True,
        column_config={
            "tv_url": st.column_config.LinkColumn(
                "Stock (TradingView)",
                display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)",
                width="medium"
            ),
            "company_name": st.column_config.TextColumn("Company", width="medium"),
            "cmp": st.column_config.NumberColumn("CMP", format="₹ %.2f", width="small"),
            "day_change_pct": st.column_config.NumberColumn("Chg %", format="%.2f %%", width="small"),
            "wt_value": st.column_config.NumberColumn("WT1", format="%.1f", help="Fast WaveTrend line (Green)", width="small"),
            "wt2_value": st.column_config.NumberColumn("WT2", format="%.1f", help="Signal line 4-SMA (Red)", width="small"),
            "wt_diff": st.column_config.NumberColumn("Diff (WT1-WT2)", format="%+.1f", width="small"),
            "signal_badge": st.column_config.TextColumn("Signal", width="small"),
            "div_badge": st.column_config.TextColumn("Divergence", width="small"),
            "trend_context": st.column_config.TextColumn("Trend Status", width="small"),
            "buy_price": st.column_config.NumberColumn("Buy Trigger", format="₹ %.2f", width="small"),
            "exit_price": st.column_config.NumberColumn("Stop Loss", format="₹ %.2f", width="small"),
            "target_price": st.column_config.NumberColumn("Target (2R)", format="₹ %.2f", width="small"),
            "confidence": st.column_config.TextColumn("Confidence", width="small")
        },
        hide_index=True,
        height=min(450, 45 + len(filtered) * 38)
    )

    # CSV Download Button
    csv_bytes = df_disp.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "📥 Download WaveTrend Results (CSV)",
        data=csv_bytes,
        file_name=f"wavetrend_{wt_timeframe[:4]}_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="wt_csv_download_btn"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 8. INTERACTIVE WAVETREND CHART INSPECTOR
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 Interactive WaveTrend Chart Inspector")
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.85rem; margin-top:-4px;'>"
        "Select any stock from the table to visualize its candlesticks, 20/50/200 SMAs, and WaveTrend oscillator sub-plot."
        "</p>",
        unsafe_allow_html=True
    )

    stock_options = [r['symbol'] for r in filtered]
    sel_chart_col1, _ = st.columns([2, 5])
    with sel_chart_col1:
        sel_symbol = st.selectbox("Select Stock to Inspect:", stock_options, key="wt_chart_sym_sel")

    stock_row = next((r for r in filtered if r['symbol'] == sel_symbol), None)
    if stock_row:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CMP", f"₹{stock_row['cmp']:,.2f}", f"{stock_row['day_change_pct']:+.2f}%")
        c2.metric("WaveTrend (WT1)", f"{stock_row['wt_value']:.1f}", f"WT2: {stock_row['wt2_value']:.1f}")
        c3.metric("Cross Status", "🟢 BUY SIGNAL" if stock_row['buy_signal'] else "Oversold", f"Diff: {stock_row['wt_diff']:+.1f}")
        c4.metric("Risk / Reward", f"SL: ₹{stock_row.get('exit_price', 0):,.2f}", f"Target: ₹{stock_row.get('target_price', 0):,.2f}")

        import yfinance as yf
        try:
            sym_ns = sel_symbol if sel_symbol.endswith('.NS') else f"{sel_symbol}.NS"
            interval = "1d" if "Daily" in wt_timeframe else ("1wk" if "Weekly" in wt_timeframe else "1d")
            hist_df = yf.download(sym_ns, period="8mo", interval=interval, progress=False, timeout=12)
            if not hist_df.empty:
                if isinstance(hist_df.columns, pd.MultiIndex):
                    hist_df.columns = hist_df.columns.get_level_values(0)
                render_wavetrend_chart(sel_symbol, hist_df, float(wt_oversold_threshold))
            else:
                st.warning("Could not download recent candle data for chart.")
        except Exception as _ce:
            st.warning(f"Error fetching chart: {_ce}")

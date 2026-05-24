# app.py
import streamlit as st
import pandas as pd
from datetime import datetime
import os
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from config import IST_TIMEZONE, get_company_name, DRY_ZONE_MIN_DAYS, DRY_ZONE_MAX_DAYS, MIN_VOLUME_RATIO, MIN_PRICE_CHANGE
from data_fetcher import fetch_ohlcv, get_index_stocks
from scanner import scan_stock, scan_coiled_spring

import watchlist
from utils import inject_premium_css, get_signal_badge_html, get_day_change_badge_html
import database
import ai_detector


# --- Page Configurations ---
st.set_page_config(
    page_title="Volume Surge Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern Outfit typography, glassmorphism card layouts and custom color styles
inject_premium_css()

# Initialize PostgreSQL database schema (Neon) on app load
database.init_db()


# --- Initialize Session State ---
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'total_scanned' not in st.session_state:
    st.session_state.total_scanned = 0
if 'failed_count' not in st.session_state:
    st.session_state.failed_count = 0
if 'last_scanned' not in st.session_state:
    st.session_state.last_scanned = None
if 'confirm_clear' not in st.session_state:
    st.session_state.confirm_clear = False
if 'coiled_results' not in st.session_state:
    st.session_state.coiled_results = None

# --- Automatic Daily Database Cache Loader ---
try:
    today_ist_str = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")
    cached_log = database.has_scanned_today(today_ist_str)
    if cached_log and st.session_state.scan_results is None:
        st.session_state.scan_results = database.get_cached_breakouts(today_ist_str)
        st.session_state.coiled_results = database.get_cached_squeezes(today_ist_str)
        st.session_state.total_scanned = cached_log['total_scanned']
        st.session_state.failed_count = 0
        st.session_state.last_scanned = today_ist_str + " (Loaded from DB Cache)"
except Exception as cache_err:
    print(f"Error loading daily database scan cache on boot: {cache_err}")


# --- HEADER SECTION ---
st.markdown('<h1 class="gradient-title">📈 Volume Surge Scanner</h1>', unsafe_allow_html=True)
st.markdown('<p class="gradient-subtitle">Scan NSE-listed stocks for institutional Volume Dry-Up (VDU) breakouts & build a high-conviction swing trading watchlist.</p>', unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown('### ⚙️ Scan Universe')
st.sidebar.markdown(
    "<div style='padding:12px; background:rgba(41,182,246,0.06); border:1px solid rgba(41,182,246,0.15); border-radius:10px; margin-bottom: 15px;'>"
    "<span style='color:#94a3b8; font-size:0.85rem;'>Active Universe:</span><br>"
    "<b style='font-size:1.15rem; color:#29b6f6;'>All NSE Listed Equities</b><br>"
    "<span style='color:#ffa000; font-size:0.85rem; font-weight:600;'>⚡ Filters: Price > ₹200 | Market Cap > ₹3000 Cr</span>"
    "</div>", 
    unsafe_allow_html=True
)


st.sidebar.markdown('---')
st.sidebar.markdown('### 🔍 VDU Strategy Filters')

# Algorithmic parameter sliders
min_vol_ratio = st.sidebar.slider(
    "Min Volume Ratio",
    min_value=2.0,
    max_value=10.0,
    value=float(MIN_VOLUME_RATIO),
    step=0.5,
    help="Breakout day volume compared to dry average volume (e.g., 2.0 = 2x surge)"
)

min_price_chg = st.sidebar.slider(
    "Min Price Change %",
    min_value=1.5,
    max_value=10.0,
    value=float(MIN_PRICE_CHANGE),
    step=0.5,
    help="Minimum price percentage increase on the breakout day (Close vs Open)"
)

dry_zone_range = st.sidebar.slider(
    "Dry Zone Range (Trading Days)",
    min_value=0,
    max_value=150,
    value=(15, 60),
    step=5,
    help="Configure the minimum and maximum duration of the dry zone consolidation period (up to 150 days)"
)

max_dry_spikes = st.sidebar.slider(
    "Max Spikes in Dry Zone",
    min_value=0,
    max_value=5,
    value=2,
    step=1,
    help="Allows up to this many unusual volume surge days inside the dry zone window to filter market noise"
)

min_signal_str = st.sidebar.slider(
    "Min Signal Strength Score",
    min_value=0,
    max_value=100,
    value=30,
    step=5,
    help="Filter stocks based on overall calculated algorithmic rating"
)

above_50dma_only = st.sidebar.checkbox(
    "Above 50 DMA Only",
    value=False,
    help="If checked, only lists breakout stocks trading above their 50-day Simple Moving Average"
)

force_fresh_scan = st.sidebar.checkbox(
    "Force Fresh Scan (Bypass Cache)",
    value=False,
    help="If checked, bypasses today's database cache and runs a fresh market scan on all listed NSE stocks"
)

st.sidebar.markdown('---')


# --- RUN SCAN ACTION ---
if st.sidebar.button("🔍 Run Scanner", use_container_width=True):
    # Check database cache first if Force Fresh Scan is False
    today_ist_str = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")
    cached_log = database.has_scanned_today(today_ist_str)
    
    if cached_log and not force_fresh_scan:
        st.sidebar.info("⚡ Today's scan is already cached in database!")
        with st.spinner("Loading cached scan results from Neon PostgreSQL..."):
            st.session_state.scan_results = database.get_cached_breakouts(today_ist_str)
            st.session_state.coiled_results = database.get_cached_squeezes(today_ist_str)
            st.session_state.total_scanned = cached_log['total_scanned']
            st.session_state.failed_count = 0
            st.session_state.last_scanned = today_ist_str + " (Loaded from DB Cache)"
            st.toast("⚡ Today's scan loaded instantly from Neon PostgreSQL!", icon="🟢")
            st.rerun()

    # Automatically query ALL listed equity shares on the National Stock Exchange of India (NSE)
    raw_symbols = get_index_stocks("ALL NSE")
        
    if not raw_symbols:
        st.sidebar.error("❌ No symbols found to scan.")
    else:
        # Step A: Perform high-speed parallel bulk download of today's quotes to filter Price > 200 instantly
        all_tickers_ns = []
        for s in raw_symbols:
            formatted = s.strip().upper()
            if not formatted.endswith(".NS"):
                formatted = f"{formatted}.NS"
            all_tickers_ns.append(formatted)
            
        close_price_map = {}
        with st.spinner("Downloading real-time quotes for all NSE Listed Equities in parallel chunks..."):
            import time
            chunk_size = 300
            ticker_chunks = [all_tickers_ns[i:i + chunk_size] for i in range(0, len(all_tickers_ns), chunk_size)]
            
            for idx, chunk in enumerate(ticker_chunks):
                retries = 0
                max_retries = 3
                backoff = 2.0
                while retries <= max_retries:
                    try:
                        # Fetch quote chunk
                        quotes_df = yf.download(tickers=chunk, period="1d", progress=False)
                        if not quotes_df.empty:
                            if isinstance(quotes_df.columns, pd.MultiIndex):
                                close_series = quotes_df['Close'].iloc[-1]
                            else:
                                close_series = pd.Series({chunk[0]: quotes_df['Close'].iloc[-1]})
                                
                            # Map prices back to plain symbols
                            for k, v in close_series.items():
                                clean_k = k.replace(".NS", "").upper()
                                if not pd.isna(v) and v > 0:
                                    close_price_map[clean_k] = float(v)
                            # Successfully loaded chunk
                            break
                        else:
                            raise ValueError("Empty DataFrame returned")
                    except Exception as chunk_ex:
                        retries += 1
                        if retries > max_retries:
                            print(f"Error downloading quote chunk {idx+1}/{len(ticker_chunks)} after {max_retries} retries: {chunk_ex}")
                            break
                        print(f"Rate limited or quote download failed for chunk {idx+1}/{len(ticker_chunks)}. Retrying in {backoff}s... (Error: {chunk_ex})")
                        time.sleep(backoff)
                        backoff *= 2.0
                        
                # Short cooldown between successful chunks to keep Yahoo Finance happy
                time.sleep(1.0)
                
        # Fast filter Price > 200 (reduces scanning load immensely by removing penny and low-priced stocks)
        scan_symbols = [s for s in raw_symbols if close_price_map.get(s.strip().upper(), 0.0) > 200.0]
        
        n_stocks = len(scan_symbols)
        failed_count = 0
        flagged_list = []
        coiled_list = []
        
        # Unpack manual dry constraints from the sidebar range slider
        min_dry = dry_zone_range[0]
        max_dry = dry_zone_range[1]
            
        # UI Scanner Feedback
        prog_bar = st.progress(0)
        status_box = st.empty()
        
        with st.spinner(f"Scanning {n_stocks} active NSE listed equities (Price > ₹200)..."):
            for i, sym in enumerate(scan_symbols):
                # Update text status and progress bar
                status_box.text(f"Scanning: {sym} ({i+1}/{n_stocks})")
                prog_bar.progress((i + 1) / n_stocks)
                
                # Fetch clean data
                df = fetch_ohlcv(sym)
                if df is None or len(df) < 5:
                    failed_count += 1
                    continue
                    
                # Fast price double check
                today_close_val = df['Close'].iloc[-1]
                if today_close_val <= 200.0:
                    continue

                    
                # Scan breakouts (passing max_dry_spikes)
                scan_res = scan_stock(
                    symbol=sym,
                    df=df,
                    min_dry_days=min_dry,
                    max_dry_days=max_dry,
                    min_volume_ratio=min_vol_ratio,
                    min_price_change=min_price_chg,
                    max_dry_spikes=max_dry_spikes
                )
                
                if scan_res is not None:
                    # Lazy market cap filter for matching breakouts (keeps scan extremely fast!)
                    formatted_sym = sym.strip().upper()
                    if not formatted_sym.endswith(".NS"):
                        formatted_sym = f"{formatted_sym}.NS"
                        
                    try:
                        ticker_obj = yf.Ticker(formatted_sym)
                        mcap = ticker_obj.fast_info.get("market_cap", 0)
                        if mcap <= 0:
                            mcap = ticker_obj.info.get("marketCap", 0)
                    except Exception:
                        mcap = 3000 * 1e7  # Fallback to pass if API is rate limited
                        
                    mcap_crores = mcap / 1e7
                    
                    # Hard filter: Market Cap >= 3000 Crore
                    if mcap_crores >= 3000.0:
                        scan_res['market_cap_cr'] = mcap_crores
                        if scan_res['signal_strength'] >= min_signal_str:
                            if not above_50dma_only or scan_res['above_50dma']:
                                flagged_list.append(scan_res)
                            
                # Scan coiled spring VCP setups
                coiled_res = scan_coiled_spring(sym, df)
                if coiled_res is not None:
                    # Lazy market cap filter for matching VCP contractions
                    formatted_sym = sym.strip().upper()
                    if not formatted_sym.endswith(".NS"):
                        formatted_sym = f"{formatted_sym}.NS"
                        
                    try:
                        ticker_obj = yf.Ticker(formatted_sym)
                        mcap = ticker_obj.fast_info.get("market_cap", 0)
                        if mcap <= 0:
                            mcap = ticker_obj.info.get("marketCap", 0)
                    except Exception:
                        mcap = 3000 * 1e7
                        
                    mcap_crores = mcap / 1e7
                    
                    # Hard filter: Market Cap >= 3000 Crore
                    if mcap_crores >= 3000.0:
                        coiled_res['market_cap_cr'] = mcap_crores
                        if coiled_res['squeeze_score'] >= min_signal_str:
                            coiled_list.append(coiled_res)
                            
        # Clean progress assets
        prog_bar.empty()
        status_box.empty()
        
        # Cache results in state to allow seamless widget interactions
        st.session_state.scan_results = flagged_list
        st.session_state.coiled_results = coiled_list
        st.session_state.total_scanned = n_stocks
        st.session_state.failed_count = failed_count
        st.session_state.last_scanned = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d %I:%M:%S %p")
        
        # Save to database cache daily
        try:
            today_ist_str = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")
            database.save_scan_results(
                date_str=today_ist_str,
                breakouts=flagged_list,
                squeezes=coiled_list,
                total_scanned=n_stocks
            )
            st.toast("💾 Today's scan results cached in Neon PostgreSQL!", icon="✅")
        except Exception as db_err:
            print(f"Failed to cache daily scan results to database: {db_err}")
        
        # Highlight large failure rate
        if n_stocks > 0 and (failed_count / n_stocks) > 0.20:
            st.sidebar.warning(f"⚠️ Failed to fetch {failed_count}/{n_stocks} symbols ({failed_count/n_stocks*100:.1f}%). Check internet connection.")


# Display Last Scanned Timestamp
if st.session_state.last_scanned:
    st.sidebar.markdown(f"<p style='text-align: center; font-size: 0.85rem; color: #94a3b8; margin-top: 10px;'>⏱️ Last Scan: <b>{st.session_state.last_scanned}</b></p>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<p style='text-align: center; font-size: 0.85rem; color: #64748b; margin-top: 10px;'>⚠️ Click 'Run Scanner' to start</p>", unsafe_allow_html=True)


# --- MAIN INTERFACE TABS ---
tab_scan, tab_detail, tab_watchlist, tab_ai, tab_coiled = st.tabs([
    "📊 Scanner Results", 
    "📈 Stock Detail", 
    "📋 My Watchlist",
    "🤖 AI Chart Pattern Detector",
    "🌀 Coiled Spring Squeeze"
])



# Get scan cache
scan_data = st.session_state.scan_results

# ==============================================================================
# TAB 1: SCANNER RESULTS
# ==============================================================================
with tab_scan:
    # 1. Premium Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    
    if scan_data:
        total_scanned = st.session_state.total_scanned
        flagged_count = len(scan_data)
        top_score = max(r['signal_strength'] for r in scan_data)
        avg_vol_ratio = sum(r['volume_ratio'] for r in scan_data) / flagged_count
    else:
        total_scanned = st.session_state.total_scanned or 0
        flagged_count = 0
        top_score = 0.0
        avg_vol_ratio = 0.0
        
    m1.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Total Stocks Scanned</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{total_scanned}</h3></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="glass-card metric-glow-green"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Breakouts Identified</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#00e676;">{flagged_count}</h3></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="glass-card metric-glow-amber"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Highest Signal Score</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#ffa000;">{top_score:.1f} <span style="font-size: 1.1rem; color: #94a3b8;">pts</span></h3></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg Volume Ratio</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{avg_vol_ratio:.2f}x</h3></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. Main Scan Table
    if scan_data is None:
        st.info("💡 Get started by configuring your universe in the sidebar and clicking '**Run Scanner**'.")
    elif len(scan_data) == 0:
        st.info("ℹ️ No VDU breakouts found today matching these criteria. Try lowering the thresholds in the sidebar (e.g. Min Volume Ratio or Min Price Change) and re-running.")
    else:
        # Sort results descending by score
        sorted_scan = sorted(scan_data, key=lambda x: x['signal_strength'], reverse=True)
        
        st.markdown("### 🔥 Flagged Breakouts")
        
        # Draw dynamic header columns
        h_cols = st.columns([1.2, 2.0, 1.0, 1.0, 1.2, 1.2, 1.0, 0.9, 0.9, 1.8, 0.8])
        h_cols[0].markdown("**Symbol**")
        h_cols[1].markdown("**Company Name**")
        h_cols[2].markdown("**CMP (₹)**")
        h_cols[3].markdown("**Day Chg%**")
        h_cols[4].markdown("**Today Vol**")
        h_cols[5].markdown("**Dry Avg Vol**")
        h_cols[6].markdown("**Vol Ratio**")
        h_cols[7].markdown("**Dry Days**")
        h_cols[8].markdown("**Spikes**")
        h_cols[9].markdown("**Signal Score**")
        h_cols[10].markdown("**Action**")
        st.markdown("<hr style='margin: 8px 0; border-color: rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
        
        # Display each row
        for r in sorted_scan:
            is_high = r['signal_strength'] >= 70.0
            
            # Draw row container column
            r_cols = st.columns([1.2, 2.0, 1.0, 1.0, 1.2, 1.2, 1.0, 0.9, 0.9, 1.8, 0.8])
            
            # Gold highlights for premium scoring
            if is_high:
                sym_txt = f"<span style='color: #ffa000; font-weight: bold;'>🌟 {r['symbol']}</span>"
            else:
                sym_txt = f"**{r['symbol']}**"
                
            r_cols[0].markdown(sym_txt, unsafe_allow_html=True)
            r_cols[1].markdown(f"<span style='font-size:0.9rem; color:#94a3b8;'>{r['company_name']}</span>", unsafe_allow_html=True)
            r_cols[2].markdown(f"₹{r['cmp']:.2f}")
            
            chg_badge = get_day_change_badge_html(r['day_change_pct'])
            r_cols[3].markdown(chg_badge, unsafe_allow_html=True)
            
            r_cols[4].markdown(f"{r['today_volume']:,}")
            r_cols[5].markdown(f"{int(r['dry_avg_vol']):,}")
            r_cols[6].markdown(f"<b>{r['volume_ratio']:.2f}x</b>", unsafe_allow_html=True)
            r_cols[7].markdown(f"{r['dry_days_count']}d")
            
            # Render Dry Spikes count!
            spikes_badge = f"<span class='custom-badge badge-red' style='font-weight:600;'>{r['dry_spikes']}</span>" if r['dry_spikes'] > 0 else f"<span class='custom-badge badge-grey'>{r['dry_spikes']}</span>"
            r_cols[8].markdown(spikes_badge, unsafe_allow_html=True)
            
            score_badge = get_signal_badge_html(r['signal_strength'])
            r_cols[9].markdown(score_badge, unsafe_allow_html=True)
            
            # Render "Add" button with customized CSS keying
            add_clicked = r_cols[10].button(
                "➕ Add", 
                key=f"add_{r['symbol']}", 
                use_container_width=True
            )
            
            if add_clicked:
                added = watchlist.add_stock(
                    symbol=r['symbol'],
                    entry_price=r['cmp'],
                    signal_strength=r['signal_strength'],
                    company_name=r['company_name']
                )
                if added:
                    st.toast(f"✅ Added {r['symbol']} to Watchlist!", icon="📈")
                else:
                    st.toast(f"⚠️ {r['symbol']} is already in Watchlist.", icon="👀")
                    
            st.markdown("<hr style='margin: 4px 0; border-color: rgba(255,255,255,0.03);'>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Download Results Option
        export_rows = []
        for r in sorted_scan:
            export_rows.append({
                "Symbol": r['symbol'],
                "Company Name": r['company_name'],
                "CMP (₹)": r['cmp'],
                "Day Change %": r['day_change_pct'],
                "Today Volume": r['today_volume'],
                "Dry Avg Volume": r['dry_avg_vol'],
                "Volume Ratio": r['volume_ratio'],
                "Dry Days": r['dry_days_count'],
                "Dry Spikes": r['dry_spikes'],
                "Market Cap (Cr)": round(r.get('market_cap_cr', 3000.0), 1),
                "Signal Strength": r['signal_strength'],
                "Above 50 DMA": r['above_50dma'],
                "Dry Start Date": r['dry_start_date'].strftime("%Y-%m-%d"),
                "Dry End Date": r['dry_end_date'].strftime("%Y-%m-%d"),
            })
        export_df = pd.DataFrame(export_rows)
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Scan Results (CSV)",
            data=csv_data,
            file_name=f"vdu_scan_results_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


# ==============================================================================
# TAB 2: STOCK DETAIL
# ==============================================================================
with tab_detail:
    if not scan_data:
        st.info("💡 No scan results available. Run a scanner from the sidebar first to view detailed analysis.")
    else:
        # Dropdown to choose from flagged list
        symbols_flagged = [r['symbol'] for r in scan_data]
        selected_sym = st.selectbox(
            "Select Flagged Stock for Detailed Charting",
            options=symbols_flagged,
            index=0,
            help="Choose a stock from current scan output"
        )
        
        # Resolve selected details dictionary
        detail_data = next((r for r in scan_data if r['symbol'] == selected_sym), None)
        
        if detail_data:
            # Lazy-load historical OHLCV data for charting if loaded from daily database cache
            if 'df' not in detail_data or detail_data['df'] is None or detail_data['df'].empty:
                with st.spinner(f"Lazy-loading historical candle data for {selected_sym}..."):
                    detail_data['df'] = fetch_ohlcv(selected_sym)
            
            df = detail_data['df']
            dry_start_date = detail_data['dry_start_date']
            dry_end_date = detail_data['dry_end_date']
            today_date = df['Date'].iloc[-1]
            
            # A. Dual subplot layout
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(f"📈 {selected_sym} Candlestick Chart & 50 DMA", f"📊 Volume Analysis")
            )
            
            # Top Candlestick trace
            fig.add_trace(
                go.Candlestick(
                    x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="Price",
                    increasing_line_color="#00e676",
                    decreasing_line_color="#ef4444"
                ),
                row=1, col=1
            )
            
            # Top 50 DMA trace
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['MA50'],
                    name="50 DMA",
                    line=dict(color="#ab47bc", width=2, dash="dash"),
                    mode="lines"
                ),
                row=1, col=1
            )
            
            # Bottom volume color builder
            bar_colors = []
            for _, row in df.iterrows():
                row_date = row['Date']
                if row_date == today_date:
                    bar_colors.append("#00e676") # Breakout surge
                elif dry_start_date <= row_date <= dry_end_date:
                    bar_colors.append("#475569") # Dry volume zone
                else:
                    bar_colors.append("#1e3a8a") # Normal blue volume
                    
            fig.add_trace(
                go.Bar(
                    x=df['Date'],
                    y=df['Volume'],
                    name="Volume",
                    marker_color=bar_colors,
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Shade the dry zone region on the candlestick subplot
            fig.add_vrect(
                x0=dry_start_date,
                x1=dry_end_date,
                fillcolor="rgba(255, 160, 0, 0.08)",
                opacity=0.6,
                layer="below",
                line_width=1,
                line_color="rgba(255,160,0,0.15)",
                annotation_text="📭 Dry Zone (Consolidation)",
                annotation_position="top left",
                annotation_font=dict(color="#ffa000", size=11, family="Outfit"),
                row=1, col=1
            )
            
            # Draw breakout arrow annotation on today's price action
            fig.add_annotation(
                x=today_date,
                y=detail_data['cmp'],
                text="🚀 Breakout",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor="#00e676",
                ax=-50,
                ay=-40,
                font=dict(color="#00e676", size=12, family="Outfit", weight="bold"),
                bgcolor="rgba(0, 230, 118, 0.08)",
                bordercolor="rgba(0,230,118,0.3)",
                borderwidth=1,
                borderpad=4,
                row=1, col=1
            )
            
            # Visual templates update
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="#090d16",
                paper_bgcolor="#090d16",
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis=dict(
                    rangeslider=dict(visible=False),
                    gridcolor="rgba(255,255,255,0.04)"
                ),
                xaxis2=dict(
                    gridcolor="rgba(255,255,255,0.04)"
                ),
                yaxis=dict(
                    gridcolor="rgba(255,255,255,0.04)",
                    title="Price (₹)"
                ),
                yaxis2=dict(
                    gridcolor="rgba(255,255,255,0.04)",
                    title="Volume"
                ),
                font=dict(family="Outfit, sans-serif"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # B. 3-column detailed metric cards
            c1, c2, c3 = st.columns(3)
            
            # Column 1
            c1.markdown(f"""
            <div class="glass-card">
                <h4 style="margin-top:0; color:#29b6f6; font-size:1.1rem; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:8px;">📈 Price Action Details</h4>
                <div style="margin: 12px 0;"><span style="color:#94a3b8; font-size:0.9rem;">Current Price:</span><br><b style="font-size:1.3rem;">₹{detail_data['cmp']:.2f}</b></div>
                <div style="margin: 12px 0;"><span style="color:#94a3b8; font-size:0.9rem;">Price Change today:</span><br>{get_day_change_badge_html(detail_data['day_change_pct'])}</div>
                <div style="margin: 12px 0;"><span style="color:#94a3b8; font-size:0.9rem;">120d Period High / Low:</span><br><b>₹{detail_data['high_52w']:.2f}</b> / <b>₹{detail_data['low_52w']:.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Column 2
            c2.markdown(f"""
            <div class="glass-card">
                <h4 style="margin-top:0; color:#00e676; font-size:1.1rem; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:8px;">📭 Dry Zone Volume Metrics</h4>
                <div style="margin: 12px 0;"><span style="color:#94a3b8; font-size:0.9rem;">Volume Ratio:</span><br><b style="font-size:1.3rem; color:#00e676;">{detail_data['volume_ratio']:.2f}x</b> (vs Dry Average)</div>
                <div style="margin: 12px 0;"><span style="color:#94a3b8; font-size:0.9rem;">Dry zone Duration:</span><br><b>{detail_data['dry_days_count']}</b> trading days</div>
                <div style="margin: 12px 0;"><span style="color:#94a3b8; font-size:0.9rem;">Dry average / today's volume:</span><br><b>{int(detail_data['dry_avg_vol']):,}</b> / <b>{detail_data['today_volume']:,}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Column 3: Custom Plotly Gauge Chart for strength
            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=detail_data['signal_strength'],
                    title={'text': "Signal Score Rating", 'font': {'size': 15, 'color': '#ffa000', 'family': 'Outfit'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                        'bar': {'color': "#ffa000"},
                        'bgcolor': "rgba(255,255,255,0.03)",
                        'borderwidth': 1,
                        'bordercolor': "rgba(255,255,255,0.08)",
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(148, 163, 184, 0.08)'},
                            {'range': [50, 70], 'color': 'rgba(41, 182, 246, 0.12)'},
                            {'range': [70, 100], 'color': 'rgba(255, 160, 0, 0.16)'}
                        ]
                    }
                )
            )
            gauge_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#e2e8f0", 'family': "Outfit"},
                height=180,
                margin=dict(l=15, r=15, t=30, b=10)
            )
            
            with c3:
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # DMA Flag badge
                dma_status = detail_data['above_50dma']
                dma_badge = '<span class="custom-badge badge-green">▲ ABOVE 50 DMA</span>' if dma_status else '<span class="custom-badge badge-red">▼ BELOW 50 DMA</span>'
                
                st.markdown(
                    f"""
                    <div style='text-align:center; padding:12px; background:rgba(17, 24, 39, 0.4); border-radius:10px; border:1px solid rgba(255,255,255,0.05); margin-top:-10px;'>
                        <b>DMA Trend Filter:</b><br>{dma_badge}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

# ==============================================================================
# TAB 3: WATCHLIST
# ==============================================================================
with tab_watchlist:
    st.markdown("### 📋 My Watchlist Monitor")
    
    # Read persistent DB
    w_df = watchlist.load_watchlist()
    
    if w_df.empty:
        st.info("ℹ️ Your watchlist is currently empty. Run scans on index universes or paste custom tickers to build your watchlist!")
    else:
        # A. SINGLE BATCH YFINANCE PRICE DOWNLOAD
        tickers_list = [f"{s}.NS" for s in w_df['symbol'].unique()]
        cmp_dict = {}
        
        with st.spinner("Fetching real-time quotes for watchlisted assets..."):
            try:
                # Fetch only 1 day to query CMP
                prices_df = yf.download(tickers=tickers_list, period="1d", progress=False)
                if not prices_df.empty:
                    # Clean columns if response is MultiIndexed
                    if isinstance(prices_df.columns, pd.MultiIndex):
                        close_prices = prices_df['Close'].iloc[-1]
                    else:
                        close_prices = {tickers_list[0]: prices_df['Close'].iloc[-1]}
                        
                    # Build lookup maps
                    if isinstance(close_prices, pd.Series):
                        for k, v in close_prices.items():
                            clean_k = k.replace(".NS", "").upper()
                            cmp_dict[clean_k] = float(v)
                    else:
                        clean_key = tickers_list[0].replace(".NS", "").upper()
                        cmp_dict[clean_key] = float(close_prices)
            except Exception as quote_ex:
                st.warning("⚠️ Could not fetch real-time quotes. Using historical entry price for watchlist CMP.")
                
        # B. BUILD WATCHLIST VIEW DATA
        display_rows = []
        for idx, row in w_df.iterrows():
            sym = row['symbol'].upper()
            entry = float(row['entry_price'])
            
            # Fetch CMP or fall back to entry
            cmp_val = cmp_dict.get(sym, entry)
            if pd.isna(cmp_val) or cmp_val <= 0:
                cmp_val = entry
                
            pnl_val = ((cmp_val - entry) / entry * 100)
            
            display_rows.append({
                "symbol": sym,
                "company_name": row['company_name'],
                "added_date": row['added_date'],
                "entry_price": entry,
                "signal_strength_at_add": float(row['signal_strength_at_add']),
                "CMP (₹)": round(cmp_val, 2),
                "PnL %": round(pnl_val, 2),
                "tag": row['tag'],
                "notes": str(row['notes']) if not pd.isna(row['notes']) else ""
            })
            
        display_df = pd.DataFrame(display_rows)
        
        # C. INTERACTIVE DATA EDITOR (Auto-saves Tag and Notes)
        st.markdown("<p style='font-size:0.85rem; color:#94a3b8;'>✏️ You can edit the <b>Tag</b> dropdowns or write custom text in <b>Notes</b> cells. Changes persist immediately.</p>", unsafe_allow_html=True)
        
        # Define table configs
        config_table = {
            "symbol": st.column_config.TextColumn("Symbol", disabled=True),
            "company_name": st.column_config.TextColumn("Company Name", disabled=True),
            "added_date": st.column_config.TextColumn("Added Date", disabled=True),
            "entry_price": st.column_config.NumberColumn("Entry Price (₹)", disabled=True, format="₹%.2f"),
            "signal_strength_at_add": st.column_config.NumberColumn("Original Signal", disabled=True, format="%.1f pts"),
            "CMP (₹)": st.column_config.NumberColumn("Current Price (₹)", disabled=True, format="₹%.2f"),
            "PnL %": st.column_config.NumberColumn("Unrealized PnL %", disabled=True, format="%.2f%%"),
            "tag": st.column_config.SelectboxColumn("Tag Status", options=["Watching 👀", "Ready to Buy 🟢", "Tracking 📍", "Avoid 🔴"]),
            "notes": st.column_config.TextColumn("Notes (Click to Edit)")
        }
        
        edited_table = st.data_editor(
            display_df,
            column_config=config_table,
            use_container_width=True,
            hide_index=True,
            key="watchlist_editor_grid"
        )
        
        # Check cell changes
        if not edited_table.equals(display_df):
            # Map back to standard CSV columns
            save_df = edited_table[['symbol', 'company_name', 'added_date', 'entry_price', 'signal_strength_at_add', 'tag', 'notes']].copy()
            watchlist.save_watchlist(save_df)
            st.toast("💾 Watchlist auto-saved successfully!")
            st.rerun()
            
        st.markdown("---")
        
        # D. MANAGEMENT CONTROLS PANEL
        st.markdown("### ⚙️ Watchlist Controls")
        
        col_c1, col_c2 = st.columns(2)
        
        # 1. Removal widget
        with col_c1:
            st.markdown("#### ❌ Delete Ticker")
            c_del1, c_del2 = st.columns([2, 1])
            ticker_to_delete = c_del1.selectbox(
                "Choose stock to remove:", 
                options=[""] + list(display_df['symbol'].unique()), 
                key="del_box"
            )
            
            if ticker_to_delete:
                del_clicked = c_del2.button("Remove Ticker", type="secondary", key="del_action", use_container_width=True)
                if del_clicked:
                    watchlist.remove_stock(ticker_to_delete)
                    st.toast(f"Removed {ticker_to_delete} from your watchlist.")
                    st.rerun()
                    
        # 2. Export and Clear watchlist
        with col_c2:
            st.markdown("#### 📂 Operations")
            
            # Export CSV
            watchlist_csv_bytes = watchlist.export_csv()
            st.download_button(
                label="📥 Export Watchlist CSV",
                data=watchlist_csv_bytes,
                file_name=f"vdu_watchlist_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_watchlist"
            )
            
            # Clear all database
            clear_btn = st.button("🗑️ Clear Entire Watchlist", type="secondary", use_container_width=True, key="clear_watchlist_btn")
            if clear_btn:
                st.session_state.confirm_clear = True
                
            if st.session_state.confirm_clear:
                st.markdown("<p style='color:#ef4444; font-weight:600;'>⚠️ Are you absolutely sure? This deletes watchlist.csv entries forever.</p>", unsafe_allow_html=True)
                col_yes, col_no = st.columns(2)
                
                if col_yes.button("Yes, Clear All", type="primary", use_container_width=True, key="clr_yes"):
                    # Clear CSV
                    empty_df = pd.DataFrame(columns=watchlist.COLUMNS)
                    watchlist.save_watchlist(empty_df)
                    st.session_state.confirm_clear = False
                    st.toast("🗑️ Watchlist fully cleared.")
                    st.rerun()
                    
                if col_no.button("Cancel", use_container_width=True, key="clr_no"):
                    st.session_state.confirm_clear = False
                    st.rerun()

# ==============================================================================
# TAB 4: AI CHART PATTERN DETECTOR
# ==============================================================================
with tab_ai:
    st.markdown("### 🤖 Technical Chart Pattern Recognition with AI")
    st.markdown("<p style='font-size:0.9rem; color:#94a3b8;'>Inspect daily candle charts with Llama-3 technical analyst AI and save/cache findings in Neon PostgreSQL database.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Fetch available symbols for analyzer
    w_db = watchlist.load_watchlist()
    available_tickers = []
    if not w_db.empty:
        available_tickers.extend(list(w_db['symbol'].unique()))
    if st.session_state.scan_results:
        available_tickers.extend([r['symbol'] for r in st.session_state.scan_results])
    
    # Unique sorted values
    available_tickers = list(set([s.upper() for s in available_tickers]))
    available_tickers.sort()

    col_s1, col_s2 = st.columns([3, 1])
    
    # Initialize selector defaults from session state if set by the dashboard load click
    default_sel_idx = 0
    if st.session_state.get("ai_stock_selection_box") == "Custom Ticker (Type Manual)":
        default_sel_idx = len(available_tickers) + 1 if available_tickers else 1
        
    ai_selection = col_s1.selectbox(
        "Select Stock to Analyze:",
        options=[""] + available_tickers + ["Custom Ticker (Type Manual)"],
        index=default_sel_idx,
        key="ai_stock_selection_box"
    )

    custom_ai_sym = ""
    if ai_selection == "Custom Ticker (Type Manual)":
        default_val = st.session_state.get("ai_custom_sym_input", "")
        custom_ai_sym = col_s2.text_input(
            "Enter Ticker Name (e.g. INFIBEAM):", 
            value=default_val,
            key="ai_custom_sym_input"
        ).strip().upper()

    ticker_to_analyze = custom_ai_sym if ai_selection == "Custom Ticker (Type Manual)" else ai_selection

    if ticker_to_analyze:
        st.markdown(f"#### 🔍 Ready to Analyze: **{ticker_to_analyze}**")
        
        # Action button to trigger scan
        btn_analyze = st.button("🤖 Analyze Pattern with AI", key="run_ai_analysis_btn")
        
        # Get today's date in IST
        today_date_str = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")
        
        # Check cache first (always check cache automatically to show today's output immediately!)
        cached_result = database.get_pattern_by_date(ticker_to_analyze, today_date_str)
        
        if cached_result or btn_analyze:
            # We either load from cache or run live!
            analysis_dict = None
            loaded_from_db = False
            
            if cached_result:
                analysis_dict = cached_result
                loaded_from_db = True
            elif btn_analyze:
                # Run live scan
                with st.spinner(f"Downloading historical data & querying AI Technical Analyst for {ticker_to_analyze}..."):
                    df_historical = fetch_ohlcv(ticker_to_analyze)
                    if df_historical is None or df_historical.empty:
                        st.error(f"❌ Failed to download historical data for {ticker_to_analyze} via yfinance.")
                    else:
                        analysis_dict = ai_detector.detect_chart_pattern(ticker_to_analyze, df_historical)
                        
                        if analysis_dict and analysis_dict.get("pattern_name") != "Error":
                            # Create small snapshot string of last 5 days close prices
                            subset_5d = df_historical.iloc[-5:]
                            snap_list = [f"{row['Date'].strftime('%m-%d')}:{row['Close']:.0f}" for _, row in subset_5d.iterrows()]
                            snap_str = ",".join(snap_list)
                            
                            # Cache in Postgres Neon db
                            database.save_pattern(
                                symbol=ticker_to_analyze,
                                pattern_name=analysis_dict['pattern_name'],
                                confidence=analysis_dict['confidence'],
                                direction=analysis_dict['direction'],
                                analysis_text=analysis_dict['analysis_text'],
                                price_data_snapshot=snap_str,
                                date_str=today_date_str
                            )
                            st.toast(f"💾 Analysis cached in Neon PostgreSQL for today!", icon="✅")
            
            if analysis_dict:
                if analysis_dict.get("pattern_name") == "Error":
                    st.error(f"❌ Analysis failed: {analysis_dict['analysis_text']}")
                else:
                    # Display results beautifully
                    if loaded_from_db:
                        st.markdown("<p style='color: #00e676; font-size: 0.85rem; font-weight: 600; margin-bottom: 15px;'>⚡ Cache Hit: Loaded instantly from PostgreSQL Database (Neon)</p>", unsafe_allow_html=True)
                    else:
                        st.markdown("<p style='color: #29b6f6; font-size: 0.85rem; font-weight: 600; margin-bottom: 15px;'>🤖 Live Analysis: Computed via Groq Llama-3 AI Technical Analyst</p>", unsafe_allow_html=True)
                    
                    # Columns for pattern metrics
                    c_det1, c_det2 = st.columns([1, 2])
                    
                    with c_det1:
                        # Color coding direction
                        d_val = analysis_dict['direction'].strip().capitalize()
                        if d_val == "Bullish":
                            dir_badge_html = '<span class="custom-badge badge-green">▲ Bullish</span>'
                        elif d_val == "Bearish":
                            dir_badge_html = '<span class="custom-badge badge-red">▼ Bearish</span>'
                        else:
                            dir_badge_html = '<span class="custom-badge badge-blue">■ Neutral</span>'
                            
                        # Color coding confidence
                        c_val = analysis_dict['confidence'].strip().capitalize()
                        if c_val == "High":
                            conf_badge_html = '<span class="custom-badge badge-amber">★ High Confidence</span>'
                        elif c_val == "Medium":
                            conf_badge_html = '<span class="custom-badge badge-blue">☆ Medium Confidence</span>'
                        else:
                            conf_badge_html = '<span class="custom-badge badge-grey">☆ Low/None</span>'
                            
                        st.markdown(f"""
                        <div class="glass-card">
                            <h4 style="margin-top:0; color:#29b6f6;">AI Assessment</h4>
                            <div style="margin: 14px 0;"><span style="color:#94a3b8; font-size:0.85rem;">Pattern Detected:</span><br><b style="font-size:1.25rem; color:#ffa000;">{analysis_dict['pattern_name']}</b></div>
                            <div style="margin: 14px 0;"><span style="color:#94a3b8; font-size:0.85rem;">Market Direction:</span><br>{dir_badge_html}</div>
                            <div style="margin: 14px 0;"><span style="color:#94a3b8; font-size:0.85rem;">Model Confidence:</span><br>{conf_badge_html}</div>
                            <div style="margin: 10px 0; font-size: 0.85rem; color:#64748b;">Scan Date: {analysis_dict['analyzed_date']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with c_det2:
                        st.markdown(f"""
                        <div class="glass-card" style="height: 100%;">
                            <h4 style="margin-top:0; color:#ffa000;">Technical Analyst Remarks</h4>
                            <p style="font-size: 1.05rem; line-height: 1.6; color: #e2e8f0; margin-top: 15px;">
                                "{analysis_dict['analysis_text']}"
                            </p>
                            <br>
                            <div style="padding: 10px; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid rgba(255,255,255,0.04); font-size:0.85rem; color:#94a3b8;">
                                💡 <b>Technical Tip:</b> Technical patterns provide high-probability outcomes when aligned with volume. Always verify breakout levels before initiating trades.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Candlestick chart for the last 30 trading days
                    # Load historical data for plotting
                    df_chart = fetch_ohlcv(ticker_to_analyze)
                    if df_chart is not None and not df_chart.empty:
                        df_chart_30d = df_chart.iloc[-30:].copy()
                        
                        fig_ai = go.Figure(
                            data=[
                                go.Candlestick(
                                    x=df_chart_30d['Date'],
                                    open=df_chart_30d['Open'],
                                    high=df_chart_30d['High'],
                                    low=df_chart_30d['Low'],
                                    close=df_chart_30d['Close'],
                                    increasing_line_color="#00e676",
                                    decreasing_line_color="#ef4444",
                                    name="Price"
                                )
                            ]
                        )
                        fig_ai.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="#090d16",
                            paper_bgcolor="#090d16",
                            margin=dict(l=30, r=30, t=30, b=30),
                            xaxis=dict(
                                rangeslider=dict(visible=False),
                                gridcolor="rgba(255,255,255,0.04)"
                            ),
                            yaxis=dict(
                                gridcolor="rgba(255,255,255,0.04)",
                                title="Price (₹)"
                            ),
                            font=dict(family="Outfit, sans-serif"),
                            height=350,
                            title={
                                'text': f"🔍 Last 30 Trading Days Price History for {ticker_to_analyze}",
                                'font': {'size': 14, 'family': 'Outfit', 'color': '#29b6f6'}
                            }
                        )
                        st.plotly_chart(fig_ai, use_container_width=True)

    # ==========================================================================
    # BATCH AI DASHBOARD FOR FLAGGED STOCKS
    # ==========================================================================
    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)
    st.markdown("### 📊 Scanned Breakouts & Squeezes AI Pattern Dashboard")
    st.markdown("<p style='font-size:0.9rem; color:#94a3b8; margin-top:-10px;'>Batch-analyze classical chart patterns recognized by AI for all breakout and contraction setups flagged in today's scans.</p>", unsafe_allow_html=True)
    
    # Collate active flagged stocks from scanner results
    active_flagged_symbols = []
    symbol_origins = {}
    
    if st.session_state.scan_results:
        for r in st.session_state.scan_results:
            sym = r['symbol'].upper()
            active_flagged_symbols.append(sym)
            symbol_origins[sym] = "📊 Breakout"
            
    if st.session_state.coiled_results:
        for r in st.session_state.coiled_results:
            sym = r['symbol'].upper()
            if sym not in symbol_origins:
                active_flagged_symbols.append(sym)
                symbol_origins[sym] = "🌀 VCP Coiled"
                
    active_flagged_symbols = list(set(active_flagged_symbols))
    active_flagged_symbols.sort()
    
    if not active_flagged_symbols:
        st.info("💡 Run a market scan first from the sidebar to find breakout or contraction setups and dynamically batch-analyze them with AI here!")
    else:
        # Load cached patterns from database for all active flagged symbols
        today_str = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")
        
        flagged_db_records = {}
        for s in active_flagged_symbols:
            rec = database.get_pattern_by_date(s, today_str)
            if rec:
                flagged_db_records[s] = rec
                
        # Count stats
        scanned_count = len(flagged_db_records)
        unscanned_count = len(active_flagged_symbols) - scanned_count
        
        # Display small dashboard summary
        d_c1, d_c2, d_c3 = st.columns(3)
        d_c1.markdown(f'<div class="glass-card"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Flagged Candidates</p><h3 style="font-size:1.6rem; margin:5px 0 0 0; color:#29b6f6;">{len(active_flagged_symbols)}</h3></div>', unsafe_allow_html=True)
        d_c2.markdown(f'<div class="glass-card"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">AI Analyzed Today</p><h3 style="font-size:1.6rem; margin:5px 0 0 0; color:#00e676;">{scanned_count}</h3></div>', unsafe_allow_html=True)
        d_c3.markdown(f'<div class="glass-card"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Pending AI Scan</p><h3 style="font-size:1.6rem; margin:5px 0 0 0; color:#ffa000;">{unscanned_count}</h3></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Batch Scan Control Button
        if unscanned_count > 0:
            btn_batch_scan = st.button(f"🤖 Run AI Scan on {unscanned_count} Pending Stocks", key="batch_ai_scan_action_btn", use_container_width=True)
            if btn_batch_scan:
                prog_ai = st.progress(0)
                status_ai = st.empty()
                
                scanned_ok = 0
                for idx, sym in enumerate(active_flagged_symbols):
                    if sym not in flagged_db_records:
                        status_ai.text(f"Running AI Technical Analysis on {sym} ({idx+1}/{len(active_flagged_symbols)})...")
                        prog_ai.progress((idx + 1) / len(active_flagged_symbols))
                        
                        df_hist = fetch_ohlcv(sym)
                        if df_hist is not None and not df_hist.empty:
                            ans_dict = ai_detector.detect_chart_pattern(sym, df_hist)
                            if ans_dict and ans_dict.get("pattern_name") != "Error":
                                subset_5d = df_hist.iloc[-5:]
                                snap_list = [f"{row['Date'].strftime('%m-%d')}:{row['Close']:.0f}" for _, row in subset_5d.iterrows()]
                                snap_str = ",".join(snap_list)
                                
                                database.save_pattern(
                                    symbol=sym,
                                    pattern_name=ans_dict['pattern_name'],
                                    confidence=ans_dict['confidence'],
                                    direction=ans_dict['direction'],
                                    analysis_text=ans_dict['analysis_text'],
                                    price_data_snapshot=snap_str,
                                    date_str=today_str
                                )
                                scanned_ok += 1
                                
                status_ai.empty()
                prog_ai.empty()
                st.toast(f"✅ Successfully scanned & cached {scanned_ok} stocks in Neon PostgreSQL!", icon="🤖")
                st.rerun()
                
        # Interactive filters for the dashboard list
        st.markdown("#### 🔍 Filter Patterns Identified")
        f_cols = st.columns(3)
        
        unique_patterns = ["All"]
        for s, rec in flagged_db_records.items():
            pat = rec['pattern_name'].strip()
            if pat not in unique_patterns and pat != "None" and pat != "Error":
                unique_patterns.append(pat)
                
        filter_pattern = f_cols[0].selectbox("Filter by Pattern Shape:", options=unique_patterns, key="dash_filter_pat")
        filter_direction = f_cols[1].selectbox("Filter by AI Direction:", options=["All", "Bullish", "Bearish", "Neutral"], key="dash_filter_dir")
        filter_status = f_cols[2].selectbox("Filter by Analysis Status:", options=["All", "AI Scanned Only", "Not Scanned Only"], key="dash_filter_status")
        
        # Display Flagged Stocks list
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 AI Chart Pattern Summary")
        
        tb_cols = st.columns([1.2, 1.2, 2.0, 1.2, 1.2, 2.2, 1.0])
        tb_cols[0].markdown("**Symbol**")
        tb_cols[1].markdown("**Scanner Type**")
        tb_cols[2].markdown("**Pattern Shape**")
        tb_cols[3].markdown("**Direction**")
        tb_cols[4].markdown("**Confidence**")
        tb_cols[5].markdown("**AI Technical Remarks**")
        tb_cols[6].markdown("**Actions**")
        st.markdown("<hr style='margin: 8px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
        
        displayed_rows = 0
        for sym in active_flagged_symbols:
            rec = flagged_db_records.get(sym)
            
            # Apply filters
            if filter_status == "AI Scanned Only" and not rec:
                continue
            if filter_status == "Not Scanned Only" and rec:
                continue
                
            if rec:
                pat_name = rec['pattern_name'].strip()
                dir_val = rec['direction'].strip().capitalize()
                conf_val = rec['confidence'].strip().capitalize()
                text_val = rec['analysis_text']
                
                if filter_pattern != "All" and pat_name != filter_pattern:
                    continue
                if filter_direction != "All" and dir_val != filter_direction:
                    continue
            else:
                pat_name = "None/Pending"
                dir_val = "Pending"
                conf_val = "Pending"
                text_val = "Stock has not been analyzed by AI technical analyst yet. Click batch scan above to compute."
                
                if filter_pattern != "All":
                    continue
                if filter_direction != "All":
                    continue
                    
            displayed_rows += 1
            
            row_cols = st.columns([1.2, 1.2, 2.0, 1.2, 1.2, 2.2, 1.0])
            
            # Symbol & Origin styling
            row_cols[0].markdown(f"**{sym}**")
            
            origin = symbol_origins.get(sym, "📊 Breakout")
            origin_color = "#29b6f6" if "Breakout" in origin else "#ab47bc"
            row_cols[1].markdown(f"<span style='color:{origin_color}; font-weight:600;'>{origin}</span>", unsafe_allow_html=True)
            
            # Pattern Shape
            if rec:
                row_cols[2].markdown(f"<b style='color:#ffa000;'>{pat_name}</b>", unsafe_allow_html=True)
                
                # Direction badge
                if dir_val == "Bullish":
                    d_badge = '<span class="custom-badge badge-green">▲ Bullish</span>'
                elif dir_val == "Bearish":
                    d_badge = '<span class="custom-badge badge-red">▼ Bearish</span>'
                else:
                    d_badge = '<span class="custom-badge badge-blue">■ Neutral</span>'
                    
                # Confidence badge
                if conf_val == "High":
                    c_badge = '<span class="custom-badge badge-amber">★ High</span>'
                elif conf_val == "Medium":
                    c_badge = '<span class="custom-badge badge-blue">☆ Medium</span>'
                else:
                    c_badge = '<span class="custom-badge badge-grey">☆ Low</span>'
            else:
                row_cols[2].markdown("<span style='color:#64748b;'>⏳ Not Scanned</span>", unsafe_allow_html=True)
                d_badge = '<span class="custom-badge badge-grey">⏳ Pending</span>'
                c_badge = '<span class="custom-badge badge-grey">⏳ Pending</span>'
                
            row_cols[3].markdown(d_badge, unsafe_allow_html=True)
            row_cols[4].markdown(c_badge, unsafe_allow_html=True)
            
            # Shortened remarks snippet
            remarks_snippet = text_val[:80] + "..." if len(text_val) > 80 else text_val
            row_cols[5].markdown(f"<span style='font-size:0.85rem; color:#94a3b8;'>\"{remarks_snippet}\"</span>", unsafe_allow_html=True)
            
            # Action button to select this ticker inside selector
            action_key = f"dash_load_{sym}_{displayed_rows}"
            if row_cols[6].button("🔍 View", key=action_key, use_container_width=True):
                st.session_state.ai_stock_selection_box = "Custom Ticker (Type Manual)"
                st.session_state.ai_custom_sym_input = sym
                st.toast(f"🔍 Loading detailed charts & AI context for {sym}...")
                st.rerun()
                
            st.markdown("<hr style='margin: 4px 0; border-color: rgba(255,255,255,0.03);'>", unsafe_allow_html=True)
            
        if displayed_rows == 0:
            st.info("ℹ️ No stocks match the active filters in this dashboard.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 📋 Recent AI Patterns Scanned")
    st.markdown("<p style='font-size:0.85rem; color:#94a3b8; margin-top:-10px;'>A real-time dashboard of technical patterns identified by other scans saved on Neon PostgreSQL.</p>", unsafe_allow_html=True)
    
    recent_records = database.get_recent_patterns(limit=10)
    if not recent_records:
        st.info("ℹ️ No technical patterns have been analyzed or saved in the database yet. Select a stock above and run the AI scanner to cache the first result!")
    else:
        # Sort and build dashboard columns
        head_cols = st.columns([1.5, 2.5, 1.5, 1.5, 2.0, 1.5])
        head_cols[0].markdown("**Symbol**")
        head_cols[1].markdown("**Pattern Identified**")
        head_cols[2].markdown("**Direction**")
        head_cols[3].markdown("**Confidence**")
        head_cols[4].markdown("**Analyzed Date**")
        head_cols[5].markdown("**Fetch Cache**")
        st.markdown("<hr style='margin: 8px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
        
        for idx, rec in enumerate(recent_records):
            row_cols = st.columns([1.5, 2.5, 1.5, 1.5, 2.0, 1.5])
            row_cols[0].markdown(f"**{rec['symbol']}**")
            row_cols[1].markdown(f"<span style='color:#ffa000; font-weight:500;'>{rec['pattern_name']}</span>", unsafe_allow_html=True)
            
            # Direction styling
            d_lower = rec['direction'].strip().lower()
            if d_lower == "bullish":
                d_badge = '<span class="custom-badge badge-green">▲ Bullish</span>'
            elif d_lower == "bearish":
                d_badge = '<span class="custom-badge badge-red">▼ Bearish</span>'
            else:
                d_badge = '<span class="custom-badge badge-blue">■ Neutral</span>'
                
            # Confidence styling
            c_lower = rec['confidence'].strip().lower()
            if c_lower == "high":
                c_badge = '<span class="custom-badge badge-amber">★ High</span>'
            elif c_lower == "medium":
                c_badge = '<span class="custom-badge badge-blue">☆ Medium</span>'
            else:
                c_badge = '<span class="custom-badge badge-grey">☆ Low</span>'
                
            row_cols[2].markdown(d_badge, unsafe_allow_html=True)
            row_cols[3].markdown(c_badge, unsafe_allow_html=True)
            row_cols[4].markdown(f"<span style='font-size:0.85rem; color:#94a3b8;'>{rec['analyzed_date']}</span>", unsafe_allow_html=True)
            
            # Action button to load this symbol's cached analysis
            if row_cols[5].button("⚡ Load", key=f"load_rec_{rec['symbol']}_{idx}", use_container_width=True):
                # Set session state options to trigger the analysis box for this symbol
                st.session_state.ai_stock_selection_box = "Custom Ticker (Type Manual)"
                st.toast(f"Loading cached analysis for {rec['symbol']}!")
                st.session_state.ai_custom_sym_input = rec['symbol']
                st.rerun()
                
            st.markdown("<hr style='margin: 4px 0; border-color: rgba(255,255,255,0.03);'>", unsafe_allow_html=True)

# ==============================================================================
# TAB 5: COILED SPRING SQUEEZE
# ==============================================================================
with tab_coiled:
    st.markdown("### 🌀 Volatility Contraction Pattern (VCP) Squeeze")
    st.markdown("<p style='font-size:0.9rem; color:#94a3b8;'>Scan for coiled springs in final contraction (VCP) setups—price compressing tightly with drying volume *before* breakout.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    coiled_data = st.session_state.coiled_results
    
    # 1. Metrics row
    c_m1, c_m2, c_m3 = st.columns(3)
    
    if coiled_data:
        coiled_count = len(coiled_data)
        min_range = min(r['range_5d'] for r in coiled_data)
        avg_squeeze = sum(r['squeeze_score'] for r in coiled_data) / coiled_count
    else:
        coiled_count = 0
        min_range = 0.0
        avg_squeeze = 0.0
        
    c_m1.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Coiled Setups Found</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{coiled_count}</h3></div>', unsafe_allow_html=True)
    c_m2.markdown(f'<div class="glass-card metric-glow-green"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Tightest 5d Price Range</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#00e676;">{min_range:.2f}%</h3></div>', unsafe_allow_html=True)
    c_m3.markdown(f'<div class="glass-card metric-glow-amber"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg Squeeze Rating</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#ffa000;">{avg_squeeze:.1f} <span style="font-size: 1.1rem; color: #94a3b8;">pts</span></h3></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. Results table
    if coiled_data is None:
        st.info("💡 Run the scanner from the sidebar to identify pre-breakout coiled spring setups.")
    elif len(coiled_data) == 0:
        st.info("ℹ️ No coiled spring (VCP) setups found today matching these filters. Price ranges might not be tight enough yet (must be <= 4.0% over the last 5 days with dried volume).")
    else:
        # Sort results descending by score
        sorted_coiled = sorted(coiled_data, key=lambda x: x['squeeze_score'], reverse=True)
        
        st.markdown("### 🌀 Active Final Contraction Squeezes")
        
        # Table headers
        ch_cols = st.columns([1.5, 2.5, 1.2, 1.5, 1.5, 1.5, 2.0, 1.0])
        ch_cols[0].markdown("**Symbol**")
        ch_cols[1].markdown("**Company Name**")
        ch_cols[2].markdown("**CMP (₹)**")
        ch_cols[3].markdown("**5-Day Range %**")
        ch_cols[4].markdown("**Pre-Range %**")
        ch_cols[5].markdown("**Vol Ratio**")
        ch_cols[6].markdown("**Squeeze Score**")
        ch_cols[7].markdown("**Action**")
        st.markdown("<hr style='margin: 8px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
        
        for r in sorted_coiled:
            is_strong = r['squeeze_score'] >= 70.0
            
            # Row columns
            cr_cols = st.columns([1.5, 2.5, 1.2, 1.5, 1.5, 1.5, 2.0, 1.0])
            
            # Formatting
            if is_strong:
                sym_txt = f"<span style='color: #ab47bc; font-weight: bold;'>🌀 {r['symbol']}</span>"
            else:
                sym_txt = f"**{r['symbol']}**"
                
            cr_cols[0].markdown(sym_txt, unsafe_allow_html=True)
            cr_cols[1].markdown(f"<span style='font-size:0.9rem; color:#94a3b8;'>{r['company_name']}</span>", unsafe_allow_html=True)
            cr_cols[2].markdown(f"₹{r['cmp']:.2f}")
            
            # Range highlighted in bright green/cyan to signify high tightness
            range_badge = f"<span class='custom-badge badge-green' style='font-weight: 600;'>{r['range_5d']:.2f}%</span>"
            cr_cols[3].markdown(range_badge, unsafe_allow_html=True)
            cr_cols[4].markdown(f"{r['range_prev']:.2f}%")
            
            cr_cols[5].markdown(f"<b>{r['vol_ratio']:.2f}x</b>", unsafe_allow_html=True)
            
            # Squeeze Score representation
            if r['squeeze_score'] >= 70.0:
                score_badge = f'<span class="custom-badge badge-amber">🌀 Coiled ({r["squeeze_score"]} pts)</span>'
            elif r['squeeze_score'] >= 50.0:
                score_badge = f'<span class="custom-badge badge-blue">📈 Tight ({r["squeeze_score"]} pts)</span>'
            else:
                score_badge = f'<span class="custom-badge badge-grey">⏳ Compressing ({r["squeeze_score"]} pts)</span>'
                
            cr_cols[6].markdown(score_badge, unsafe_allow_html=True)
            
            # Add to Watchlist button
            add_coiled = cr_cols[7].button(
                "➕ Add",
                key=f"add_coiled_{r['symbol']}",
                use_container_width=True
            )
            
            if add_coiled:
                added = watchlist.add_stock(
                    symbol=r['symbol'],
                    entry_price=r['cmp'],
                    signal_strength=r['squeeze_score'],
                    company_name=r['company_name']
                )
                if added:
                    st.toast(f"✅ Added {r['symbol']} Squeeze setup to Watchlist!", icon="🌀")
                else:
                    st.toast(f"⚠️ {r['symbol']} is already in Watchlist.", icon="👀")
                    
            st.markdown("<hr style='margin: 4px 0; border-color: rgba(255,255,255,0.03);'>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Download Coiled Results Option
        export_coiled = []
        for r in sorted_coiled:
            export_coiled.append({
                "Symbol": r['symbol'],
                "Company Name": r['company_name'],
                "CMP (₹)": r['cmp'],
                "5-Day Range %": r['range_5d'],
                "Previous Range %": r['range_prev'],
                "Volume Ratio": r['vol_ratio'],
                "Squeeze Score": r['squeeze_score'],
                "Above 20 EMA": r['above_20ema']
            })
        export_c_df = pd.DataFrame(export_coiled)
        csv_c_data = export_c_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Coiled Squeezes (CSV)",
            data=csv_c_data,
            file_name=f"coiled_squeezes_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )



import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import database
from config import IST_TIMEZONE

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


def render():
    is_admin = st.session_state.get('is_admin', False)
    st.markdown("### 📦 Flat Base Squeeze")
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.95rem; margin-top:-8px;'>"
        "Identifies stocks coiled in an <b>ultra-tight horizontal consolidation channel</b> with "
        "<b>Green VPA</b>, <b>strictly stacked moving averages (10 ≥ 21 ≥ 50 SMA)</b>, and compressed volatility "
        "<i>(Modeled after textbook base-breakout structures like Cohance Lifesciences)</i>."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Initialize session state for flat base
    if "flat_base_results" not in st.session_state:
        st.session_state.flat_base_results = []
        
    # Auto-load cached results on first visit
    if not st.session_state.flat_base_results:
        try:
            today_str = get_market_date()
            cached = database.get_cached_flat_base(today_str)
            if cached:
                st.session_state.flat_base_results = cached
        except Exception as e:
            print(f"Error loading cached flat base: {e}")
            
    # Parameter Controls in an Expander
    with st.expander("⚙️ Scanner Filter Parameters", expanded=False):
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            max_gap = st.slider("Max Compression Gap (%)", min_value=1.5, max_value=6.0, value=3.5, step=0.25,
                                help="Maximum spread percentage between 10, 21, and 50 SMAs. Cohance is ~2.5%.")
        with p_col2:
            max_base = st.slider("Max 30-Day Channel Range (%)", min_value=8.0, max_value=25.0, value=16.0, step=1.0,
                                 help="Max (High - Low) / Low percentage over last 30 trading days. Ensures a flat horizontal base.")
        with p_col3:
            strict_stack = st.checkbox("Strict Bullish Stack (10 ≥ 21 ≥ 50)", value=True,
                                       help="Ensures moving averages are stacked in proper ascending bullish order.")
    
    col_btn, col_info = st.columns([3, 7])
    with col_btn:
        run_scan_btn = st.button("🚀 Run Flat Base Squeeze Scan", width="stretch")
        
    if run_scan_btn:
        st.session_state.flat_base_results = []
        with st.spinner("Scanning for Flat Base Squeeze setups across all NSE equities..."):
            try:
                from data_fetcher import get_all_nse_symbols
                from scanner import scan_flat_base_squeeze
                from local_cache_manager import bulk_get_cached_ohlcv
                
                raw_symbols = get_all_nse_symbols()
                symbols_to_scan = [s.strip().upper().replace('.NS', '') for s in raw_symbols if str(s).strip()]
                
                prog = st.progress(0)
                status = st.empty()
                status.text("Fetching market data from cache...")
                
                bulk_cached = bulk_get_cached_ohlcv(symbols_to_scan, "1d")
                status.text("Evaluating flat base channel and VPA compression...")
                prog.progress(0.7)
                
                results = []
                for sym, df in bulk_cached.items():
                    res = scan_flat_base_squeeze(
                        sym, df,
                        max_gap_pct=max_gap,
                        max_base_range_pct=max_base,
                        strict_bullish_alignment=strict_stack
                    )
                    if res is not None:
                        results.append(res)
                        
                prog.progress(1.0)
                prog.empty()
                status.empty()
                
                st.session_state.flat_base_results = results
                
                # Save to database
                today_str = get_market_date()
                database.save_flat_base_only(today_str, results)
                st.success(f"Scan complete! Found {len(results)} high-conviction flat base setups.")
                
            except Exception as e:
                st.error(f"Error executing scan: {e}")

    results = st.session_state.get('flat_base_results', [])
    if results:
        df_res = pd.DataFrame(results)
        
        # Add TradingView URL formatting
        df_res['tv_link'] = df_res['symbol'].apply(
            lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}"
        )
        
        # Sort by Compression Gap ascending
        if 'ma_gap_pct' in df_res.columns:
            df_res = df_res.sort_values('ma_gap_pct', ascending=True)
            
        st.markdown(f"#### 🎯 Matching Setups ({len(results)} stocks)")
        
        display_cols = [
            'tv_link', 'company_name', 'cmp', 'day_change_pct',
            'ma_gap_pct', 'base_range_pct', 'sma10', 'sma21', 'sma50',
            'dist_to_200_pct', 'rsi', 'cci', 'buy_price', 'target_price', 'exit_price'
        ]
        display_cols = [c for c in display_cols if c in df_res.columns]
        
        st.dataframe(
            df_res[display_cols],
            use_container_width=True,
            column_config={
                "tv_link": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)", width="small"),
                "company_name": st.column_config.TextColumn("Company", width="medium"),
                "cmp": st.column_config.NumberColumn("CMP (₹)", format="%.2f", width="small"),
                "day_change_pct": st.column_config.NumberColumn("Chg %", format="%.2f", width="small"),
                "ma_gap_pct": st.column_config.NumberColumn("MA Gap %", format="%.2f%%", width="small", help="Max distance % between 10, 21, and 50 SMAs."),
                "base_range_pct": st.column_config.NumberColumn("30d Base %", format="%.2f%%", width="small", help="Price swing range over the last 30 trading days (narrower = tighter base)."),
                "sma10": st.column_config.NumberColumn("10 SMA", format="%.2f", width="small"),
                "sma21": st.column_config.NumberColumn("21 SMA", format="%.2f", width="small"),
                "sma50": st.column_config.NumberColumn("50 SMA", format="%.2f", width="small"),
                "dist_to_200_pct": st.column_config.NumberColumn("Dist to 200 SMA", format="%.2f%%", width="small"),
                "rsi": st.column_config.NumberColumn("RSI", format="%.1f", width="small"),
                "cci": st.column_config.NumberColumn("CCI", format="%.1f", width="small"),
                "buy_price": st.column_config.NumberColumn("Buy Point", format="%.2f", width="small"),
                "target_price": st.column_config.NumberColumn("Target", format="%.2f", width="small"),
                "exit_price": st.column_config.NumberColumn("Stop Loss", format="%.2f", width="small"),
            },
            hide_index=True
        )
    else:
        st.info("ℹ️ No stocks currently matching the Flat Base Squeeze criteria. Click **Run Flat Base Squeeze Scan** to evaluate active NSE equities.")

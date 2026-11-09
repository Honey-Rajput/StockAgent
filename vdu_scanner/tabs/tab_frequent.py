import threading
import concurrent.futures
import re
import json
import time
import streamlit as st
import pandas as pd
import database

from collections import defaultdict
import streamlit as st
import pandas as pd
import database

def render():
    st.markdown("### 🚨 High-Conviction Confluence & Consistent Alerts")
    st.write("Aggregates stocks triggering across multiple scanner strategies today, alongside multi-day frequent alert flyers.")
    
    # ── Section 1: Live Active Session Confluence Alerts ──────────────────────
    live_alerts = defaultdict(lambda: {
        'symbol': '',
        'cmp': 0.0,
        'strategies': set(),
        'max_score': 0.0,
        'rsi': 0.0,
        'cci': 0.0
    })
    
    # 1. VDU Breakouts
    for r in st.session_state.get('scan_results', []):
        sym = str(r.get('symbol', '')).strip().upper().replace('.NS', '')
        if sym:
            live_alerts[sym]['symbol'] = sym
            live_alerts[sym]['cmp'] = float(r.get('cmp') or 0.0)
            live_alerts[sym]['strategies'].add('VDU Breakout')
            live_alerts[sym]['max_score'] = max(live_alerts[sym]['max_score'], float(r.get('signal_strength') or 0.0))
            if r.get('rsi'): live_alerts[sym]['rsi'] = float(r.get('rsi'))
            if r.get('cci'): live_alerts[sym]['cci'] = float(r.get('cci'))
            
    # 2. Minervini Stage 2 Setups
    for r in st.session_state.get('minervini_results', []):
        sym = str(r.get('symbol', '')).strip().upper().replace('.NS', '')
        if sym:
            live_alerts[sym]['symbol'] = sym
            if not live_alerts[sym]['cmp']: live_alerts[sym]['cmp'] = float(r.get('cmp') or 0.0)
            live_alerts[sym]['strategies'].add('Minervini Stage 2')
            
    # 3. WaveTrend Oversold Buy
    for r in st.session_state.get('wt_results', []):
        sym = str(r.get('symbol', '')).strip().upper().replace('.NS', '')
        if sym:
            live_alerts[sym]['symbol'] = sym
            if not live_alerts[sym]['cmp']: live_alerts[sym]['cmp'] = float(r.get('cmp') or 0.0)
            live_alerts[sym]['strategies'].add('WaveTrend Buy')
            
    # 4. Gap-Up Setups
    for r in st.session_state.get('gapup_results', []):
        sym = str(r.get('symbol', '')).strip().upper().replace('.NS', '')
        if sym:
            live_alerts[sym]['symbol'] = sym
            if not live_alerts[sym]['cmp']: live_alerts[sym]['cmp'] = float(r.get('cmp') or 0.0)
            live_alerts[sym]['strategies'].add('Gap-Up')
            
    # 5. 9/21 EMA Support
    for r in st.session_state.get('ema_support_results', []):
        sym = str(r.get('symbol', '')).strip().upper().replace('.NS', '')
        if sym:
            live_alerts[sym]['symbol'] = sym
            if not live_alerts[sym]['cmp']: live_alerts[sym]['cmp'] = float(r.get('cmp') or 0.0)
            live_alerts[sym]['strategies'].add('9/21 EMA Support')
            if r.get('score'): live_alerts[sym]['max_score'] = max(live_alerts[sym]['max_score'], float(r.get('score') or 0.0))

    # 6. Volume Profile Buy Zone
    for r in st.session_state.get('vp_results', []):
        sym = str(r.get('symbol', '')).strip().upper().replace('.NS', '')
        d = r.get('daily') or {}
        if isinstance(d, dict) and d.get('zone') == '✅ Can Buy (Near Support)':
            if sym:
                live_alerts[sym]['symbol'] = sym
                if not live_alerts[sym]['cmp']: live_alerts[sym]['cmp'] = float(r.get('cmp') or 0.0)
                live_alerts[sym]['strategies'].add('VP Buy Zone')

    tv_col_config = {
        "Symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)"),
        "CMP": st.column_config.NumberColumn("CMP (₹)", format="%.2f"),
        "Score": st.column_config.NumberColumn("Score", format="%.1f"),
    }

    if live_alerts:
        # Sort by confluence (strategies count desc), then score desc
        live_list = sorted(live_alerts.values(), key=lambda x: (len(x['strategies']), x['max_score']), reverse=True)
        
        # Confluence metrics
        confluence_2plus = [x for x in live_list if len(x['strategies']) >= 2]
        
        st.subheader(f"⚡ Today's Live Alerts ({len(live_list)} Stocks Active)")
        st.caption("Stocks actively meeting scanner criteria in today's session. High-confluence stocks (matching 2+ strategies) represent the strongest institutional setups.")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Active Alerts", len(live_list))
        m2.metric("🔥 High Confluence (2+ Strategies)", len(confluence_2plus))
        m3.metric("Top Confluence Count", max(len(x['strategies']) for x in live_list) if live_list else 0)
        
        live_rows = []
        for rank, x in enumerate(live_list, 1):
            strat_count = len(x['strategies'])
            badge = f"🔥 {strat_count} Strategies" if strat_count >= 2 else "1 Strategy"
            live_rows.append({
                'Rank': rank,
                'Symbol': f"https://in.tradingview.com/chart/?symbol=NSE:{x['symbol']}",
                'CMP': x['cmp'],
                'Confluence': badge,
                'Triggered Strategies': " | ".join(sorted(x['strategies'])),
                'Score': x['max_score']
            })
            
        df_live = pd.DataFrame(live_rows)
        st.dataframe(df_live, width="stretch", hide_index=True, column_config=tv_col_config, height=min(450, 45 + len(df_live) * 35))
        st.markdown("---")

    # ── Section 2: Historical Multi-Day Database Alerts ───────────────────────
    st.subheader("🏆 Multi-Day Frequent Alerts (Database Archive)")
    st.caption("Tracks stocks repeatedly flagged across completed daily scanner runs.")
    
    col1, col2 = st.columns([2, 5])
    with col1:
        lookback_days = st.slider("Lookback Window (Days)", min_value=3, max_value=30, value=15, step=1,
                                 help="Number of recent distinct scan dates to search across.")
    
    with st.spinner(f"Aggregating alerts for the last {lookback_days} scan days..."):
        frequent_stocks = database.get_frequent_stocks(days_lookback=lookback_days)
        
    if not frequent_stocks:
        st.info(f"No multi-day repeated stocks found in the database for the last {lookback_days} days. As daily scans complete, frequent stocks will appear here.")
        return
        
    # Convert to DataFrame
    df = pd.DataFrame(frequent_stocks)
    
    # Process Strategy tags into nice labels
    def format_strategies(strategies_str):
        if not strategies_str:
            return ""
        strats = set([s.strip() for s in strategies_str.split(',')])
        return " | ".join(sorted(strats))
        
    df['strategies'] = df['strategies'].apply(format_strategies)
    
    # Split into New Today and Repeated
    new_today_df = df[df.get('is_new_today', pd.Series([False]*len(df))) == True]
    repeated_df = df[df.get('is_new_today', pd.Series([True]*len(df))) != True]
    
    def apply_tv_links(disp_df):
        disp_df['Symbol'] = disp_df['Symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
        return disp_df
        
    # --- NEW TODAY SECTION ---
    if len(new_today_df) > 0:
        st.subheader(f"🆕 New in Archive — First Time Flagged ({len(new_today_df)} stocks)")
        st.caption("These stocks just appeared in the historical archive for the first time.")
        
        new_display = pd.DataFrame()
        new_display['Symbol'] = new_today_df['symbol'].astype(str)
        new_display['Score'] = pd.to_numeric(new_today_df['max_score'], errors='coerce').fillna(0).astype(float).round(1)
        new_display['RSI'] = pd.to_numeric(new_today_df['rsi'], errors='coerce').fillna(0).astype(float).round(2)
        new_display['CCI'] = pd.to_numeric(new_today_df['cci'], errors='coerce').fillna(0).astype(float).round(2)
        new_display['Strategies'] = new_today_df['strategies'].astype(str)
        new_display['Total Hits'] = pd.to_numeric(new_today_df['total_appearances'], errors='coerce').fillna(0).astype(int)
        
        new_display = apply_tv_links(new_display)
        
        st.dataframe(
            new_display,
            width="stretch",
            hide_index=True,
            column_config=tv_col_config,
            height=min(400, 35 + len(new_display) * 35)
        )
        
        st.write("---")
    
    # --- REPEATED SECTION ---
    if len(repeated_df) > 0:
        repeated_df = repeated_df.sort_values(by=['days_appeared', 'total_appearances'], ascending=[False, False])
        
        rep_display = pd.DataFrame()
        rep_display['Symbol'] = repeated_df['symbol'].astype(str)
        rep_display['Consistency %'] = (pd.to_numeric(repeated_df['days_appeared'], errors='coerce').fillna(0) / lookback_days * 100).round(1).astype(str) + "%"
        rep_display['Days Appeared'] = pd.to_numeric(repeated_df['days_appeared'], errors='coerce').fillna(0).astype(int)
        rep_display['Score'] = pd.to_numeric(repeated_df['max_score'], errors='coerce').fillna(0).astype(float).round(1)
        rep_display['First Alert'] = repeated_df['first_seen_date'].astype(str)
        rep_display['Most Recent'] = repeated_df['last_seen_date'].astype(str)
        rep_display['Triggered Strategies'] = repeated_df['strategies'].astype(str)
        
        rep_display = apply_tv_links(rep_display)
        
        st.subheader(f"🔁 Repeated Multi-Day Alerts (Last {lookback_days} Scans) — {len(repeated_df)} stocks")
        
        st.dataframe(
            rep_display,
            width="stretch",
            hide_index=True,
            column_config=tv_col_config,
            height=600
        )
    
    # CSV Download — full data
    full_display = pd.DataFrame()
    full_display['Symbol'] = df['symbol'].astype(str)
    full_display['New Today'] = df.get('is_new_today', False).apply(lambda x: '🆕 Yes' if x else 'No').astype(str)
    full_display['Consistency %'] = (pd.to_numeric(df['days_appeared'], errors='coerce').fillna(0) / lookback_days * 100).round(1).astype(str) + "%"
    full_display['Days Appeared'] = pd.to_numeric(df['days_appeared'], errors='coerce').fillna(0).astype(int)
    full_display['Score'] = pd.to_numeric(df['max_score'], errors='coerce').fillna(0).astype(float).round(1)
    full_display['First Alert'] = df['first_seen_date'].astype(str)
    full_display['Most Recent'] = df['last_seen_date'].astype(str)
    full_display['Triggered Strategies'] = df['strategies'].astype(str)
    
    csv_data = full_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download All Alerts (CSV)",
        data=csv_data,
        file_name="all_alerts.csv",
        mime="text/csv",
        key="dl_all_alerts_csv"
    )
    
    st.write("---")
    st.caption(f"Showing {len(new_today_df)} new today + {len(repeated_df)} repeated = {len(df)} total stocks across the last {lookback_days} scans.")

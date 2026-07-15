import streamlit as st
import pandas as pd
import database

def render():
    st.header("🏆 Consistent Alerts (Frequent Flyers)")
    st.write("Tracks stocks that have been frequently flagged across multiple scanner strategies over recent days. **🆕 New Today** stocks are shown first — catch moves on Day 1!")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        lookback_days = st.slider("Lookback Window (Days)", min_value=3, max_value=30, value=15, step=1,
                                 help="Number of recent distinct scan dates to search across.")
    
    with st.spinner(f"Aggregating alerts for the last {lookback_days} scan days..."):
        frequent_stocks = database.get_frequent_stocks(days_lookback=lookback_days)
        
    if not frequent_stocks:
        st.info(f"No stocks found in the last {lookback_days} scans.")
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
    
    # Define a helper function to add TV links
    def apply_tv_links(disp_df):
        disp_df['Symbol'] = disp_df['Symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
        return disp_df
        
    tv_col_config = {
        "Symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)")
    }
    
    # --- NEW TODAY SECTION ---
    if len(new_today_df) > 0:
        st.subheader(f"🆕 New Today — First Day Alerts ({len(new_today_df)} stocks)")
        st.caption("These stocks just appeared in the scanner for the first time. Catch the move early!")
        
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
        rep_display['RSI'] = pd.to_numeric(repeated_df['rsi'], errors='coerce').fillna(0).astype(float).round(2)
        rep_display['CCI'] = pd.to_numeric(repeated_df['cci'], errors='coerce').fillna(0).astype(float).round(2)
        rep_display['First Alert'] = repeated_df['first_seen_date'].astype(str)
        rep_display['Most Recent'] = repeated_df['last_seen_date'].astype(str)
        rep_display['Triggered Strategies'] = repeated_df['strategies'].astype(str)
        
        rep_display = apply_tv_links(rep_display)
        
        st.subheader(f"🔁 Repeated Alerts (Last {lookback_days} Scans) — {len(repeated_df)} stocks")
        
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
    full_display['RSI'] = pd.to_numeric(df['rsi'], errors='coerce').fillna(0).astype(float).round(2)
    full_display['CCI'] = pd.to_numeric(df['cci'], errors='coerce').fillna(0).astype(float).round(2)
    full_display['First Alert'] = df['first_seen_date'].astype(str)
    full_display['Most Recent'] = df['last_seen_date'].astype(str)
    full_display['Triggered Strategies'] = df['strategies'].astype(str)
    
    csv_data = full_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download All Alerts (CSV)",
        data=csv_data,
        file_name="all_alerts.csv",
        mime="text/csv",
        width="content"
    )
    
    st.write("---")
    st.caption(f"Showing {len(new_today_df)} new today + {len(repeated_df)} repeated = {len(df)} total stocks across the last {lookback_days} scans.")

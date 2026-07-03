import streamlit as st
import pandas as pd
import database

def render():
    st.header("🏆 Consistent Alerts (Frequent Flyers)")
    st.write("Tracks stocks that have been frequently flagged across multiple scanner strategies over recent days.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        lookback_days = st.slider("Lookback Window (Days)", min_value=3, max_value=30, value=15, step=1,
                                 help="Number of recent distinct scan dates to search across.")
    
    with st.spinner(f"Aggregating alerts for the last {lookback_days} scan days..."):
        frequent_stocks = database.get_frequent_stocks(days_lookback=lookback_days)
        
    if not frequent_stocks:
        st.info(f"No stocks found appearing on multiple days in the last {lookback_days} scans.")
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
    
    # Sort
    df = df.sort_values(by=['days_appeared', 'total_appearances'], ascending=[False, False])
    
    # Format for display
    display_df = pd.DataFrame()
    display_df['Symbol'] = df['symbol']
    display_df['Consistency %'] = (df['days_appeared'] / lookback_days * 100).round(1).astype(str) + "%"
    display_df['Days Appeared'] = df['days_appeared']
    display_df['Score'] = df['max_score'].fillna(0).round(1)
    display_df['RSI'] = df['rsi'].fillna(0).round(2)
    display_df['CCI'] = df['cci'].fillna(0).round(2)
    display_df['First Alert'] = df['first_seen_date']
    display_df['Most Recent'] = df['last_seen_date']
    display_df['Triggered Strategies'] = df['strategies']
    
    st.subheader(f"Top Repeated Alerts (Last {lookback_days} Scans)")
    
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        height=600
    )
    
    # CSV Download
    csv_data = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Consistent Alerts (CSV)",
        data=csv_data,
        file_name="consistent_alerts.csv",
        mime="text/csv",
        width="content"
    )
    
    st.write("---")
    st.caption(f"Showing {len(df)} stocks that appeared on more than 1 day out of the last {lookback_days} scans.")

import streamlit as st
import pandas as pd
from datetime import datetime
from breadth_ledger import fetch_breadth_data, get_market_breadth_summary, get_leading_clusters


def render():
    st.markdown("### 🌐 Institutional Market Breadth & Sector Clusters")
    st.markdown(
        "<p style='font-size:0.9rem; color:#94a3b8; margin-top:-6px;'>"
        "Real-time broad market participation, advance/decline dynamics, and institutional industry group clustering "
        "powered by <b>The Breadth Ledger</b> (<a href='https://markets.manasarora.com/' target='_blank' style='color:#38bdf8;'>markets.manasarora.com</a>)."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    col_refresh, _ = st.columns([1, 4])
    if col_refresh.button("🔄 Refresh Breadth Data", key="refresh_breadth_ledger_btn"):
        fetch_breadth_data(force_refresh=True)
        st.success("Refreshed latest market breadth data!")

    summary = get_market_breadth_summary()
    clusters = get_leading_clusters()
    raw_data = fetch_breadth_data()

    # 1. Market Regime & Key Metrics Cards
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <span style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Current Market Breadth Regime</span>
                <h3 style="margin: 2px 0 0 0; font-size: 1.35rem; color: #f8fafc;">{summary['regime_badge']}</h3>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.8rem; color: #94a3b8;">Liquid NSE Universe: <b>{summary['universe_size']} stocks</b></span>
                <div style="font-size: 0.85rem; color: #cbd5e1;">Stocks > 50 DMA: <b style="color: #38bdf8; font-size: 1.05rem;">{summary['stocks_above_50dma']}%</b></div>
            </div>
        </div>
        <p style="margin: 8px 0 0 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">{summary['regime_desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    # 2. Key Metrics Row
    m1, m2, m3, m4 = st.columns(4)

    adv_color = "#00e676" if summary["up_stocks"] >= summary["down_stocks"] else "#ef4444"
    m1.markdown(f"""
    <div class="glass-card" style="padding: 12px 14px; text-align: center;">
        <span style="font-size: 0.78rem; color: #94a3b8;">Advance / Decline</span>
        <div style="font-size: 1.3rem; font-weight: bold; margin-top: 4px; color: {adv_color};">
            {summary['up_stocks']} <span style="font-size: 0.9rem; color: #94a3b8;">/</span> {summary['down_stocks']}
        </div>
        <span style="font-size: 0.75rem; color: #94a3b8;">{summary['up_pct']}% Advancing</span>
    </div>
    """, unsafe_allow_html=True)

    m2.markdown(f"""
    <div class="glass-card" style="padding: 12px 14px; text-align: center;">
        <span style="font-size: 0.78rem; color: #94a3b8;">52-Week Highs / Lows</span>
        <div style="font-size: 1.3rem; font-weight: bold; margin-top: 4px; color: #00e676;">
            {summary['highs_52w']} <span style="font-size: 0.9rem; color: #94a3b8;">/</span> <span style="color: #ef4444;">{summary['lows_52w']}</span>
        </div>
        <span style="font-size: 0.75rem; color: #94a3b8;">Net: +{summary['highs_52w'] - summary['lows_52w']}</span>
    </div>
    """, unsafe_allow_html=True)

    m3.markdown(f"""
    <div class="glass-card" style="padding: 12px 14px; text-align: center;">
        <span style="font-size: 0.78rem; color: #94a3b8;">Momentum Thrust (>4%)</span>
        <div style="font-size: 1.3rem; font-weight: bold; margin-top: 4px; color: #38bdf8;">
            +{summary['surge_up_4pct']} <span style="font-size: 0.9rem; color: #94a3b8;">/</span> <span style="color: #f87171;">-{summary['drop_dn_4pct']}</span>
        </div>
        <span style="font-size: 0.75rem; color: #94a3b8;">Expansion vs Contraction</span>
    </div>
    """, unsafe_allow_html=True)

    vix_chg = summary['vix_chg']
    vix_color = "#ef4444" if vix_chg > 0 else "#00e676"
    m4.markdown(f"""
    <div class="glass-card" style="padding: 12px 14px; text-align: center;">
        <span style="font-size: 0.78rem; color: #94a3b8;">Indices & Volatility</span>
        <div style="font-size: 0.92rem; font-weight: bold; margin-top: 6px;">
            NIFTY: <span style="color: {'#00e676' if summary['nifty_chg'] >= 0 else '#ef4444'};">{summary['nifty_chg']:+.2f}%</span> | VIX: <span style="color: {vix_color};">{vix_chg:+.2f}%</span>
        </div>
        <span style="font-size: 0.75rem; color: #94a3b8;">Mid: {summary['midcap_chg']:+.2f}% | Small: {summary['smallcap_chg']:+.2f}%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. Where Fresh Highs Cluster (Leading Industry Groups)
    st.markdown("### 🎯 Where Fresh Highs Cluster (Institutional Leading Sectors)")
    st.markdown(
        "<p style='font-size:0.85rem; color:#94a3b8; margin-top:-6px;'>"
        "Institutional accumulation creates sector clusters. When multiple stocks in the same industry make fresh 1-Month, 3-Month, "
        "and 6-Month highs simultaneously, breakouts from these groups have the highest odds of sustained momentum."
        "</p>",
        unsafe_allow_html=True
    )

    if clusters:
        cluster_rows = []
        for rank, c in enumerate(clusters[:18], 1):
            cluster_rows.append({
                "Rank": rank,
                "Industry Group": c["industry"],
                "Total Stocks": c["total_stocks"],
                "1M Highs": c["highs_1m"],
                "3M Highs": c["highs_3m"],
                "6M Highs": c["highs_6m"],
                "Breadth Score": c["score"],
                "Leading Tickers": ", ".join(c.get("leading_stocks", [])[:6])
            })
        df_clusters = pd.DataFrame(cluster_rows)
        st.dataframe(
            df_clusters,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("#", width="small"),
                "Industry Group": st.column_config.TextColumn("Industry Group", width="medium"),
                "Total Stocks": st.column_config.NumberColumn("Total", width="small"),
                "1M Highs": st.column_config.NumberColumn("1M Highs", width="small"),
                "3M Highs": st.column_config.NumberColumn("3M Highs", width="small"),
                "6M Highs": st.column_config.NumberColumn("6M Highs", width="small"),
                "Breadth Score": st.column_config.NumberColumn("Score", width="small"),
                "Leading Tickers": st.column_config.TextColumn("Leading Institutional Stocks", width="large"),
            }
        )
    else:
        st.info("No industry cluster data currently available.")

    st.markdown("---")

    # 4. Rolling Breadth History
    recent_days = raw_data.get("recentDays", [])
    if recent_days:
        st.markdown("### 📜 Rolling Session Breadth History")
        st.markdown(
            "<p style='font-size:0.85rem; color:#94a3b8; margin-top:-6px;'>"
            "Track daily shifts in market participation, advance/decline volume, and percentage of stocks above 50 DMA."
            "</p>",
            unsafe_allow_html=True
        )

        hist_rows = []
        for d in reversed(recent_days):
            date_raw = str(d.get("date", ""))
            date_fmt = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}" if len(date_raw) == 8 else date_raw
            hist_rows.append({
                "Date": date_fmt,
                "Advancing": d.get("upN", 0),
                "Declining": d.get("dnN", 0),
                "52W Highs": d.get("hiN", 0),
                "52W Lows": d.get("loN", 0),
                "1M Highs": d.get("h1m", 0),
                "1M Lows": d.get("l1m", 0),
                "Gainers >25% (1M)": d.get("g25", 0),
                ">50 DMA (%)": f"{d.get('m50', 0)}%",
                ">5 DMA (%)": f"{d.get('m5', 0)}%"
            })
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

    # 5. Position Sizing Rules Box
    with st.expander("💡 How to Trade with Breadth Ledger (Position Sizing & Exposure Matrix)", expanded=False):
        st.markdown("""
        **Institutional Rule of Market Breadth (Minervini & Breadth Ledger Framework):**

        * **🟢 Green Regime (>60% stocks > 50 DMA)**:
          * **Market Action:** Broad expansion rally. Breakouts from VDU, VCP, and Stage 2 have maximum follow-through.
          * **Exposure:** 100% portfolio allocation; full position sizes (10-20% per position).
        
        * **🟡 Yellow Regime (40% - 60% stocks > 50 DMA)**:
          * **Market Action:** Selective market. Choppy index with stock-specific rotation.
          * **Exposure:** 50% - 70% portfolio allocation. Only trade stocks belonging to the **Top Leading Industry Clusters**.
        
        * **🔴 Red Regime (<40% stocks > 50 DMA)**:
          * **Market Action:** Market in distribution or correction. High rate of false breakouts.
          * **Exposure:** 0% - 30% allocation. Tighten trailing stops, hold cash, and wait for a Breadth Thrust reversal.
        """)

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
        table_rows = []
        for rank, c in enumerate(clusters[:20], 1):
            stock_links = []
            for sym in c.get("leading_stocks", []):
                clean_sym = str(sym).strip().upper().replace(".NS", "")
                if clean_sym:
                    link_html = (
                        f'<a href="https://in.tradingview.com/chart/?symbol=NSE:{clean_sym}" target="_blank" rel="noopener noreferrer" '
                        f'style="display: inline-block; margin: 2px 3px; padding: 3px 8px; background: rgba(56, 189, 248, 0.12); '
                        f'color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 4px; font-weight: 600; '
                        f'font-size: 0.78rem; text-decoration: none; transition: all 0.15s ease;" '
                        f'title="Open {clean_sym} on TradingView">{clean_sym} ↗</a>'
                    )
                    stock_links.append(link_html)
            stocks_cell = " ".join(stock_links) if stock_links else '<span style="color: #64748b;">—</span>'

            row_html = f"""
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05); transition: background 0.15s ease;">
                <td style="padding: 10px 12px; text-align: center; color: #64748b; font-weight: 600;">{rank}</td>
                <td style="padding: 10px 12px; font-weight: 600; color: #f1f5f9;">{c['industry']}</td>
                <td style="padding: 10px 12px; text-align: center; color: #94a3b8;">{c['total_stocks']}</td>
                <td style="padding: 10px 12px; text-align: center; font-weight: bold; color: #38bdf8;">{c['highs_1m']}</td>
                <td style="padding: 10px 12px; text-align: center; font-weight: bold; color: #22d3ee;">{c['highs_3m']}</td>
                <td style="padding: 10px 12px; text-align: center; font-weight: bold; color: #00e676;">{c['highs_6m']}</td>
                <td style="padding: 10px 12px; text-align: center; font-weight: bold; color: #ffa000;">{c['score']}</td>
                <td style="padding: 10px 12px;">{stocks_cell}</td>
            </tr>
            """
            table_rows.append(row_html)

        tbody_html = "".join(table_rows)

        st.markdown(f"""
        <div style="overflow-x: auto; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; background: rgba(9, 13, 22, 0.6); margin-top: 10px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);">
            <table style="width: 100%; border-collapse: collapse; font-family: 'Segoe UI', system-ui, sans-serif; font-size: 0.85rem; color: #cbd5e1;">
                <thead>
                    <tr style="background: rgba(255, 255, 255, 0.03); border-bottom: 1px solid rgba(255, 255, 255, 0.1); color: #94a3b8; text-transform: uppercase; font-size: 0.74rem; letter-spacing: 0.5px;">
                        <th style="padding: 10px 12px; text-align: center; width: 45px;">#</th>
                        <th style="padding: 10px 12px; text-align: left; width: 220px;">Industry Group</th>
                        <th style="padding: 10px 12px; text-align: center; width: 65px;">Total</th>
                        <th style="padding: 10px 12px; text-align: center; width: 80px; color: #38bdf8;">1M Highs</th>
                        <th style="padding: 10px 12px; text-align: center; width: 80px; color: #22d3ee;">3M Highs</th>
                        <th style="padding: 10px 12px; text-align: center; width: 80px; color: #00e676;">6M Highs</th>
                        <th style="padding: 10px 12px; text-align: center; width: 70px; color: #ffa000;">Score</th>
                        <th style="padding: 10px 12px; text-align: left;">Leading Institutional Stocks <span style="color:#38bdf8; font-weight:normal; font-size:0.7rem;">(Click ticker to open TradingView chart)</span></th>
                    </tr>
                </thead>
                <tbody>
                    {tbody_html}
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
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

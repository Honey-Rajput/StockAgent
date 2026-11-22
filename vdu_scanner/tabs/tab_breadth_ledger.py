import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from breadth_ledger import (
    fetch_breadth_data,
    get_market_breadth_summary,
    get_leading_clusters,
    get_breadth_stock_picks,
    track_cluster_changes,
)


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

    # 3. Fresh Highs Churn Tracker (Added / Deleted Stocks Tracking)
    churn = track_cluster_changes()
    new_symbols_set = churn.get("new_symbols_set", set())
    new_reasons_map = churn.get("new_reasons_map", {})
    added_list = churn.get("added", [])
    removed_list = churn.get("removed", [])
    has_comp = churn.get("has_comparison", False)
    net_chg = churn.get("net_change", 0)
    prev_d = churn.get("prev_date", "")
    curr_d = churn.get("curr_date", "")
    tot_leaders = churn.get("total_leaders", len(clusters))

    st.markdown("### 🔄 Sector & Fresh Highs Churn Tracker")
    st.markdown(
        "<p style='font-size:0.85rem; color:#94a3b8; margin-top:-6px;'>"
        "Tracks stocks <b>newly entering</b> or <b>dropping out of</b> the institutional Fresh Highs clusters between consecutive market sessions. "
        "A cluster expanding with new leaders signals high-conviction institutional accumulation."
        "</p>",
        unsafe_allow_html=True
    )

    ch_c1, ch_c2, ch_c3, ch_c4 = st.columns(4)
    net_badge_color = "#00e676" if net_chg > 0 else ("#ef4444" if net_chg < 0 else "#38bdf8")
    net_sign = "+" if net_chg > 0 else ""
    net_desc = "Expanding Breadth" if net_chg > 0 else ("Narrowing Breadth" if net_chg < 0 else "Neutral")

    ch_c1.markdown(
        f'<div style="background:rgba(0,230,118,0.06); border:1px solid rgba(0,230,118,0.25); border-radius:8px; padding:10px 14px;">'
        f'<span style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">🆕 Newly Added to Highs</span>'
        f'<h3 style="margin:2px 0 0 0; font-size:1.4rem; color:#00e676;">+{len(added_list)}</h3>'
        f'<span style="font-size:0.72rem; color:#6ee7b7;">Fresh Momentum Leaders</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    ch_c2.markdown(
        f'<div style="background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.25); border-radius:8px; padding:10px 14px;">'
        f'<span style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">🔻 Dropped from Highs</span>'
        f'<h3 style="margin:2px 0 0 0; font-size:1.4rem; color:#ef4444;">-{len(removed_list)}</h3>'
        f'<span style="font-size:0.72rem; color:#fca5a5;">Lost Highs Status</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    ch_c3.markdown(
        f'<div style="background:rgba(56,189,248,0.06); border:1px solid {net_badge_color}; border-radius:8px; padding:10px 14px;">'
        f'<span style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">📊 Net Churn Velocity</span>'
        f'<h3 style="margin:2px 0 0 0; font-size:1.4rem; color:{net_badge_color};">{net_sign}{net_chg}</h3>'
        f'<span style="font-size:0.72rem; color:#94a3b8;">{net_desc}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    ch_c4.markdown(
        f'<div style="background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:10px 14px;">'
        f'<span style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">🎯 Total Active Highs</span>'
        f'<h3 style="margin:2px 0 0 0; font-size:1.4rem; color:#38bdf8;">{tot_leaders}</h3>'
        f'<span style="font-size:0.72rem; color:#94a3b8;">Across {len(clusters)} Groups</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    if has_comp:
        st.markdown(
            f"<div style='margin-top:10px; font-size:0.78rem; color:#64748b;'>"
            f"Comparing session <b>{curr_d}</b> vs prior session <b>{prev_d}</b>"
            f"</div>",
            unsafe_allow_html=True
        )

        col_add, col_rem = st.columns(2)
        with col_add:
            if added_list:
                added_cards = []
                for item in added_list:
                    s = item["symbol"]
                    ind = item.get("industry", "")
                    reason = item.get("reason", "Fresh cycle high breakout.")
                    added_cards.append(
                        f'<div style="background:rgba(0,230,118,0.05); border:1px solid rgba(0,230,118,0.25); border-radius:6px; padding:8px 12px; margin-bottom:8px;">'
                        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">'
                        f'<a href="https://in.tradingview.com/chart/?symbol=NSE:{s}" target="_blank" rel="noopener noreferrer" '
                        f'style="font-weight:700; font-size:0.88rem; color:#00e676; text-decoration:none;" title="Open {s} on TradingView">+ {s} ↗</a>'
                        f'<span style="font-size:0.72rem; color:#86efac; background:rgba(0,230,118,0.12); padding:2px 8px; border-radius:4px; font-weight:600;">{ind}</span>'
                        f'</div>'
                        f'<div style="font-size:0.78rem; color:#cbd5e1; line-height:1.35;">'
                        f'💡 <b style="color:#6ee7b7;">Why Added:</b> {reason}'
                        f'</div>'
                        f'</div>'
                    )
                st.markdown(
                    f"<div style='margin-top:6px;'>"
                    f"<span style='font-size:0.82rem; font-weight:700; color:#00e676;'>🟢 Newly Added Stocks ({len(added_list)}):</span>"
                    f"<div style='margin-top:6px;'>{''.join(added_cards)}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.info("No newly added stocks compared to the previous session.")

        with col_rem:
            if removed_list:
                removed_cards = []
                for item in removed_list:
                    s = item["symbol"]
                    ind = item.get("industry", "")
                    reason = item.get("reason", "Momentum cooled below fresh-high threshold.")
                    removed_cards.append(
                        f'<div style="background:rgba(239,68,68,0.05); border:1px solid rgba(239,68,68,0.25); border-radius:6px; padding:8px 12px; margin-bottom:8px;">'
                        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">'
                        f'<span style="font-weight:700; font-size:0.88rem; color:#f87171;">- {s}</span>'
                        f'<span style="font-size:0.72rem; color:#fca5a5; background:rgba(239,68,68,0.12); padding:2px 8px; border-radius:4px; font-weight:600;">{ind}</span>'
                        f'</div>'
                        f'<div style="font-size:0.78rem; color:#cbd5e1; line-height:1.35;">'
                        f'⚠️ <b style="color:#fca5a5;">Why Dropped:</b> {reason}'
                        f'</div>'
                        f'</div>'
                    )
                st.markdown(
                    f"<div style='margin-top:6px;'>"
                    f"<span style='font-size:0.82rem; font-weight:700; color:#ef4444;'>🔴 Dropped from Highs ({len(removed_list)}):</span>"
                    f"<div style='margin-top:6px;'>{''.join(removed_cards)}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.caption("No stocks dropped from the fresh highs list.")
    else:
        st.markdown(
            f"<div style='margin-top:8px; padding:8px 12px; background:rgba(56,189,248,0.05); border:1px solid rgba(56,189,248,0.2); border-radius:6px; font-size:0.8rem; color:#94a3b8;'>"
            f"ℹ️ <b>Baseline Established:</b> Currently tracking <b>{tot_leaders} active fresh-high stocks</b> across <b>{len(clusters)} industry groups</b>. "
            f"Stocks added or dropped will be tracked here each day as market breadth evolves."
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 4. Where Fresh Highs Cluster (Leading Industry Groups)
    st.markdown("### 🎯 Where Fresh Highs Cluster (Institutional Leading Sectors)")
    st.markdown(
        "<p style='font-size:0.85rem; color:#94a3b8; margin-top:-6px;'>"
        "Institutional accumulation creates sector clusters. When multiple stocks in the same industry make fresh 1-Month, 3-Month, "
        "and 6-Month highs simultaneously, breakouts from these groups have the highest odds of sustained momentum."
        "</p>",
        unsafe_allow_html=True
    )

    if clusters:
        # Search & Filter Row
        cf1, cf2, cf3, cf4 = st.columns([3, 2, 2.5, 2.5])
        with cf1:
            search_cluster = st.text_input(
                "🔍 Search Industry or Stock:",
                "",
                placeholder="e.g. Banks, DMART, Auto, Hospital...",
                key="breadth_cluster_search"
            ).strip().upper()
        with cf2:
            min_highs_filter = st.selectbox(
                "Min Fresh Highs:",
                ["All Groups", "≥2 Highs", "≥4 Highs", "≥6 Highs"],
                index=0,
                key="breadth_cluster_min_highs"
            )
        with cf3:
            hide_zero_stocks = st.checkbox(
                "Hide groups with 0 fresh highs",
                value=False,
                key="breadth_cluster_hide_zero",
                help="Hide industry groups where no stock is currently making fresh highs"
            )
        with cf4:
            row_limit_sel = st.selectbox(
                "Show Groups:",
                [f"All Groups ({len(clusters)})", "Top 20", "Top 10"],
                index=0,
                key="breadth_cluster_row_limit"
            )

        # Apply Filters
        filtered_clusters = []
        for c in clusters:
            ind_upper = str(c["industry"]).upper()
            stocks_upper = [str(s).upper() for s in c.get("leading_stocks", [])]

            # 1. Search query
            if search_cluster:
                match_ind = search_cluster in ind_upper
                match_stock = any(search_cluster in s for s in stocks_upper)
                if not (match_ind or match_stock):
                    continue

            # 2. Min highs filter
            h1 = c.get("highs_1m", 0)
            if "≥2" in min_highs_filter and h1 < 2:
                continue
            elif "≥4" in min_highs_filter and h1 < 4:
                continue
            elif "≥6" in min_highs_filter and h1 < 6:
                continue

            # 3. Hide zero stocks
            if hide_zero_stocks and (len(c.get("leading_stocks", [])) == 0 or h1 == 0):
                continue

            filtered_clusters.append(c)

        # 4. Limit row count
        if "Top 10" in row_limit_sel:
            display_clusters = filtered_clusters[:10]
        elif "Top 20" in row_limit_sel:
            display_clusters = filtered_clusters[:20]
        else:
            display_clusters = filtered_clusters

        st.caption(f"Showing **{len(display_clusters)}** of **{len(clusters)}** industry groups.")

        table_rows = []
        for rank, c in enumerate(display_clusters, 1):
            stock_links = []
            for sym in c.get("leading_stocks", []):
                clean_sym = str(sym).strip().upper().replace(".NS", "")
                if clean_sym:
                    # New badge if newly added
                    is_new = clean_sym in new_symbols_set
                    reason_tip = f"Why Added: {new_reasons_map.get(clean_sym, 'Fresh High breakout')}" if is_new else f"Open {clean_sym} on TradingView"
                    new_tag = f'<span style="display:inline-block;margin-left:3px;padding:1px 4px;background:#00e676;color:#052e16;border-radius:3px;font-size:0.65rem;font-weight:800;" title="{reason_tip}">NEW</span>' if is_new else ''
                    link_bg = "rgba(0,230,118,0.15)" if is_new else "rgba(56,189,248,0.12)"
                    link_border = "rgba(0,230,118,0.4)" if is_new else "rgba(56,189,248,0.3)"
                    link_color = "#00e676" if is_new else "#38bdf8"

                    link_html = (
                        '<a href="https://in.tradingview.com/chart/?symbol=NSE:' + clean_sym + '" '
                        'target="_blank" rel="noopener noreferrer" '
                        'style="display:inline-block;margin:2px 3px;padding:3px 8px;'
                        'background:' + link_bg + ';color:' + link_color + ';'
                        'border:1px solid ' + link_border + ';border-radius:4px;'
                        'font-weight:600;font-size:0.78rem;text-decoration:none;" '
                        'title="' + reason_tip + '">' + clean_sym + ' &#8599;' + new_tag + '</a>'
                    )
                    stock_links.append(link_html)

            stocks_cell = " ".join(stock_links) if stock_links else '<span style="color:#64748b;font-style:italic;">No stocks at fresh highs</span>'

            table_rows.append(
                '<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">'
                '<td style="padding:10px 12px;text-align:center;color:#64748b;font-weight:600;">' + str(rank) + '</td>'
                '<td style="padding:10px 12px;font-weight:600;color:#f1f5f9;">' + str(c["industry"]) + '</td>'
                '<td style="padding:10px 12px;text-align:center;color:#94a3b8;">' + str(c["total_stocks"]) + '</td>'
                '<td style="padding:10px 12px;text-align:center;font-weight:bold;color:#38bdf8;">' + str(c["highs_1m"]) + '</td>'
                '<td style="padding:10px 12px;text-align:center;font-weight:bold;color:#22d3ee;">' + str(c["highs_3m"]) + '</td>'
                '<td style="padding:10px 12px;text-align:center;font-weight:bold;color:#00e676;">' + str(c["highs_6m"]) + '</td>'
                '<td style="padding:10px 12px;text-align:center;font-weight:bold;color:#ffa000;">' + str(c["score"]) + '</td>'
                '<td style="padding:10px 12px;">' + stocks_cell + '</td>'
                '</tr>'
            )

        tbody_html = "".join(table_rows)
        cluster_table_html = (
            '<div style="font-family:\'Segoe UI\',system-ui,sans-serif;color:#cbd5e1;">'
            '<div style="overflow-x:auto;border:1px solid rgba(255,255,255,0.08);border-radius:10px;'
            'background:rgba(9,13,22,0.9);box-shadow:0 4px 20px rgba(0,0,0,0.25);">'
            '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
            '<thead><tr style="background:rgba(255,255,255,0.03);border-bottom:1px solid rgba(255,255,255,0.1);'
            'color:#94a3b8;text-transform:uppercase;font-size:0.74rem;letter-spacing:0.5px;">'
            '<th style="padding:10px 12px;text-align:center;width:45px;">#</th>'
            '<th style="padding:10px 12px;text-align:left;width:220px;">Industry Group</th>'
            '<th style="padding:10px 12px;text-align:center;width:65px;">Total</th>'
            '<th style="padding:10px 12px;text-align:center;width:80px;color:#38bdf8;">1M Highs</th>'
            '<th style="padding:10px 12px;text-align:center;width:80px;color:#22d3ee;">3M Highs</th>'
            '<th style="padding:10px 12px;text-align:center;width:80px;color:#00e676;">6M Highs</th>'
            '<th style="padding:10px 12px;text-align:center;width:70px;color:#ffa000;">Score</th>'
            '<th style="padding:10px 12px;text-align:left;">Leading Institutional Stocks '
            '<span style="color:#38bdf8;font-weight:normal;font-size:0.7rem;">'
            '(Click ticker to open TradingView chart)</span></th>'
            '</tr></thead>'
            '<tbody>' + tbody_html + '</tbody>'
            '</table></div></div>'
        )
        components.html(cluster_table_html, height=min(120 + len(display_clusters) * 52, 2200), scrolling=True)
    else:
        st.info("No industry cluster data currently available.")

    st.markdown("---")

    # 4. Breadth-Powered Stock Recommendations
    st.markdown("### 🧭 Breadth-Powered Stock Recommendations")
    st.markdown(
        "<p style='font-size:0.85rem; color:#94a3b8; margin-top:-6px;'>"
        "Stocks ranked by <b>multi-timeframe momentum</b> (5-day to 250-day) from The Breadth Ledger universe. "
        "Stocks in <b>Top-10 leading industry clusters</b> receive a boost. "
        "Use the regime above to calibrate your exposure."
        "</p>",
        unsafe_allow_html=True
    )

    regime_m50 = summary["stocks_above_50dma"]
    if regime_m50 >= 60.0:
        regime_note = "🟢 <b>AGGRESSIVE:</b> Broad market supports full-sized positions. All picks below are actionable."
        regime_note_color = "#00e676"
    elif regime_m50 >= 40.0:
        regime_note = "🟡 <b>SELECTIVE:</b> Trade only <b>Top Leading Cluster</b> stocks (shown with cluster badge). Keep positions smaller."
        regime_note_color = "#fbbf24"
    else:
        regime_note = "🔴 <b>DEFENSIVE:</b> Avoid new longs. The picks below are watchlist candidates only — wait for breadth improvement."
        regime_note_color = "#ef4444"

    st.markdown(
        f"<div style='margin: -6px 0 14px 0; padding: 8px 14px; border-left: 3px solid {regime_note_color}; "
        f"background: rgba(0,0,0,0.2); border-radius: 4px; font-size:0.85rem; color:#cbd5e1;'>"
        f"{regime_note}</div>",
        unsafe_allow_html=True
    )

    picks = get_breadth_stock_picks(min_momo_score=28, top_n=40)

    if picks:
        # Controls row
        col_filter, col_sort = st.columns([2, 2])
        with col_filter:
            only_leading = st.checkbox(
                "🌟 Only Leading Cluster Stocks",
                value=(regime_m50 < 60.0),
                key="picks_only_leading",
                help="Show only stocks from the Top-10 leading industry clusters"
            )
        with col_sort:
            sort_by = st.selectbox(
                "Sort by",
                ["Final Score (default)", "Momentum Score", "Cluster Rank"],
                key="picks_sort_by",
                label_visibility="collapsed"
            )

        filtered = [p for p in picks if (not only_leading or p["in_leading_cluster"])]

        if sort_by == "Momentum Score":
            filtered.sort(key=lambda x: -x["momo_score"])
        elif sort_by == "Cluster Rank":
            filtered.sort(key=lambda x: (x["cluster_rank"], -x["momo_score"]))

        # Build HTML table rows
        pick_rows_html = []
        for idx, p in enumerate(filtered, 1):
            sym = p["symbol"]
            tv_url = f"https://in.tradingview.com/chart/?symbol=NSE:{sym}"
            sym_link = (
                f'<a href="{tv_url}" target="_blank" rel="noopener noreferrer" '
                f'style="font-weight:700; font-size:0.92rem; color:#38bdf8; text-decoration:none;" '
                f'title="Open {sym} chart on TradingView">{sym} ↗</a>'
            )

            # Cluster badge
            if p["in_leading_cluster"]:
                cl_badge = (
                    f'<span style="display:inline-block; padding:2px 7px; margin-left:5px; '
                    f'background:rgba(250,191,36,0.15); color:#fbbf24; border:1px solid rgba(250,191,36,0.4); '
                    f'border-radius:4px; font-size:0.72rem; font-weight:600;">★ #{p["cluster_rank"]}</span>'
                )
            else:
                cl_badge = ""

            # Momentum bar: 9 boxes, green if tf present, grey if not
            lb_all = [5, 10, 20, 30, 60, 90, 125, 180, 250]
            pos_set = set(p["positive_tfs"])
            bar_cells = "".join(
                f'<span title="{tf}d" style="display:inline-block;width:9px;height:14px;margin:0 1px;'
                f'border-radius:2px;background:{"#22c55e" if tf in pos_set else "#334155"};"></span>'
                for tf in lb_all
            )
            momo_bar = f'<div style="display:inline-flex;align-items:center;gap:1px;">{bar_cells}</div>'

            # Score pill
            score_pct = int((p["momo_score"] / 45) * 100)
            score_color = "#00e676" if score_pct >= 90 else ("#4ade80" if score_pct >= 70 else "#fbbf24")
            score_pill = (
                f'<span style="font-weight:700;color:{score_color}">{p["momo_score"]}</span>'
                f'<span style="color:#475569;font-size:0.74rem;">/45</span>'
                f'{"&nbsp;<span style=\\'font-size:0.72rem;color:#fbbf24;\\'>+"+str(p["cluster_boost"])+"✦</span>" if p["cluster_boost"] else ""}'
            )

            row_bg = "rgba(250,191,36,0.04)" if p["in_leading_cluster"] else "rgba(0,0,0,0)"
            pick_rows_html.append(
                '<tr style="border-bottom:1px solid rgba(255,255,255,0.05);background:' + row_bg + ';">'
                '<td style="padding:9px 10px;text-align:center;color:#475569;font-size:0.8rem;">' + str(idx) + '</td>'
                '<td style="padding:9px 10px;">' + sym_link + cl_badge + '</td>'
                '<td style="padding:9px 10px;color:#94a3b8;font-size:0.82rem;">' + p["industry"] + '</td>'
                '<td style="padding:9px 10px;text-align:center;">' + momo_bar + '</td>'
                '<td style="padding:9px 10px;text-align:center;">' + score_pill + '</td>'
                '<td style="padding:9px 10px;font-size:0.82rem;color:' + p["trend_color"] + ';font-weight:600;">' + p["trend_label"] + '</td>'
                '</tr>'
            )

        tbody = "".join(pick_rows_html)
        picks_html = (
            '<div style="font-family:\'Segoe UI\',system-ui,sans-serif;color:#cbd5e1;">'
            '<div style="overflow-x:auto;border:1px solid rgba(255,255,255,0.08);border-radius:10px;'
            'background:rgba(9,13,22,0.9);margin-top:8px;box-shadow:0 4px 20px rgba(0,0,0,0.25);">'
            '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
            '<thead><tr style="background:rgba(255,255,255,0.03);border-bottom:1px solid rgba(255,255,255,0.1);'
            'color:#94a3b8;text-transform:uppercase;font-size:0.73rem;letter-spacing:0.5px;">'
            '<th style="padding:10px;text-align:center;width:36px;">#</th>'
            '<th style="padding:10px;text-align:left;width:150px;">Stock</th>'
            '<th style="padding:10px;text-align:left;">Industry</th>'
            '<th style="padding:10px;text-align:center;width:120px;">Momentum Bars'
            '<div style="font-size:0.65rem;color:#475569;font-weight:normal;margin-top:1px;">5d·10d·20d·30d·60d·90d·125d·180d·250d</div></th>'
            '<th style="padding:10px;text-align:center;width:90px;">Score</th>'
            '<th style="padding:10px;text-align:left;width:180px;">Trend</th>'
            '</tr></thead>'
            '<tbody>' + tbody + '</tbody>'
            '</table></div>'
            '<p style="font-size:0.72rem;color:#475569;margin-top:6px;padding:0 4px;">'
            '&#10022; Cluster boost applied for stocks in Top-10 leading industry groups. '
            'Green bars = positive momentum for that timeframe. Click stock name to open TradingView chart. '
            '<b>Not financial advice — do your own due diligence.</b></p></div>'
        )
        components.html(picks_html, height=min(120 + len(filtered) * 46, 1800), scrolling=True)
    else:
        st.info("No stock picks available at this time. The market breadth data may still be loading.")

    st.markdown("---")

    # 6. Rolling Breadth History
    recent_days = raw_data.get("recentDays", [])
    if recent_days:
        st.markdown("### 📜 Rolling Session Breadth History")
        st.markdown(
            "<p style='font-size:0.85rem; color:#94a3b8; margin-top:-6px;'>"
            "Track daily shifts in market participation, advance/decline volume, and percentage of stocks above 50 DMA."
            "</p>",
            unsafe_allow_html=True
        )

        def _date_sort_key(d):
            raw = str(d.get("date", "")).strip().lower()
            if raw == "live":
                return "99999999"  # Live session always on top
            return raw

        sorted_days = sorted(recent_days, key=_date_sort_key, reverse=True)

        hist_rows = []
        for d in sorted_days:
            date_raw = str(d.get("date", "")).strip()
            if date_raw.lower() == "live":
                date_fmt = "Live"
            elif len(date_raw) == 8 and date_raw.isdigit():
                date_fmt = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"
            else:
                date_fmt = date_raw

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

    # 7. Position Sizing Rules Box
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

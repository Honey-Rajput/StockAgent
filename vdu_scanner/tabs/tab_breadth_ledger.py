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

        # Build JS cluster data map for modal (keyed by row-id)
        cluster_js_data = {}
        for rank_pre, c_pre in enumerate(display_clusters, 1):
            rid_pre = f"row-{rank_pre}"
            cs = c_pre.get("cluster_stocks", [])
            if not cs:
                cs = [
                    {
                        "symbol": str(s).strip().upper().replace(".NS", ""),
                        "flag": 1,
                        "tier_code": "1m",
                        "tier_name": "1M High",
                        "in_1m": True,
                        "in_3m": False,
                        "in_6m": False,
                    }
                    for s in c_pre.get("leading_stocks", [])
                ]
            stocks_6m = [s["symbol"] for s in cs if s.get("in_6m")]
            stocks_3m_only = [s["symbol"] for s in cs if s.get("in_3m") and not s.get("in_6m")]
            stocks_1m_only = [s["symbol"] for s in cs if s.get("in_1m") and not s.get("in_3m") and not s.get("in_6m")]
            cluster_js_data[rid_pre] = {
                "industry": str(c_pre.get("industry", "")),
                "total": c_pre.get("total_stocks", 0),
                "h1m": c_pre.get("highs_1m", 0),
                "h3m": c_pre.get("highs_3m", 0),
                "h6m": c_pre.get("highs_6m", 0),
                "s6m": stocks_6m,
                "s3m": stocks_3m_only,
                "s1m": stocks_1m_only,
            }

        import json as _json
        cluster_js_json = _json.dumps(cluster_js_data)

        table_rows = []
        for rank, c in enumerate(display_clusters, 1):
            row_id = f"row-{rank}"
            drawer_id = f"drawer-row-{rank}"
            chevron_id = f"chevron-row-{rank}"
            industry = str(c.get("industry", "Unknown"))
            tot = c.get("total_stocks", 0)
            h1 = c.get("highs_1m", 0)
            h3 = c.get("highs_3m", 0)
            h6 = c.get("highs_6m", 0)
            score = c.get("score", 0)

            pct_1m = round((h1 / tot * 100), 1) if tot > 0 else 0
            pct_3m = round((h3 / tot * 100), 1) if tot > 0 else 0
            pct_6m = round((h6 / tot * 100), 1) if tot > 0 else 0

            cluster_stocks = c.get("cluster_stocks", [])
            stocks_no_high = c.get("stocks_no_high", [])
            leading_stocks = c.get("leading_stocks", [])

            # Fallback if cluster_stocks empty
            if not cluster_stocks:
                cluster_stocks = [
                    {
                        "symbol": str(s).strip().upper().replace(".NS", ""),
                        "flag": 1,
                        "tier_code": "1m",
                        "tier_name": "1M High",
                        "in_1m": True,
                        "in_3m": False,
                        "in_6m": False,
                    }
                    for s in leading_stocks
                ]

            # Build stock chips for the main cell
            stock_chips = []
            for item in cluster_stocks:
                sym = item["symbol"]
                flag = item["flag"]
                is_no_high = flag == 0
                is_new = sym in new_symbols_set
                reason_tip = f"Why Added: {new_reasons_map.get(sym, 'Fresh High breakout')}" if is_new else f"{sym} ({item['tier_name']}) - Click to open chart"

                new_tag = f'<span style="display:inline-block;margin-left:3px;padding:1px 3px;background:#00e676;color:#052e16;border-radius:2px;font-size:0.62rem;font-weight:800;" title="{reason_tip}">NEW</span>' if is_new else ''

                tier_badge = ""
                if flag >= 3:
                    tier_badge = '<span style="font-size:0.62rem;background:#00e676;color:#042f2e;padding:0 3px;border-radius:2px;margin-left:3px;font-weight:800;">6M</span>'
                elif flag == 2:
                    tier_badge = '<span style="font-size:0.62rem;background:#22d3ee;color:#083344;padding:0 3px;border-radius:2px;margin-left:3px;font-weight:800;">3M</span>'
                elif flag == 1:
                    tier_badge = '<span style="font-size:0.62rem;background:#38bdf8;color:#082f49;padding:0 3px;border-radius:2px;margin-left:3px;font-weight:800;">1M</span>'

                display_style = "display:none;" if is_no_high else "display:inline-block;"
                tier_class = f"tier-{item['tier_code']}"

                chip_html = (
                    f'<a href="https://in.tradingview.com/chart/?symbol=NSE:{sym}" target="_blank" rel="noopener noreferrer" '
                    f'class="stock-chip {tier_class}" '
                    f'data-symbol="{sym}" '
                    f'data-tier="{item["tier_code"]}" '
                    f'data-in1m="{str(item["in_1m"]).lower()}" '
                    f'data-in3m="{str(item["in_3m"]).lower()}" '
                    f'data-in6m="{str(item["in_6m"]).lower()}" '
                    f'data-no-high="{str(is_no_high).lower()}" '
                    f'style="{display_style}" '
                    f'title="{reason_tip}">{sym} &#8599;{tier_badge}{new_tag}</a>'
                )
                stock_chips.append(chip_html)

            # Build drawer chips matching Image 1
            drawer_chips = []
            for item in cluster_stocks:
                sym = item["symbol"]
                flag = item["flag"]
                is_no_high = flag == 0

                if flag >= 3:
                    d_bg = "#00e676"
                    d_color = "#042f2e"
                    d_border = "#00e676"
                elif flag == 2:
                    d_bg = "#22d3ee"
                    d_color = "#083344"
                    d_border = "#22d3ee"
                elif flag == 1:
                    d_bg = "rgba(56,189,248,0.2)"
                    d_color = "#38bdf8"
                    d_border = "#38bdf8"
                else:
                    d_bg = "#1e293b"
                    d_color = "#64748b"
                    d_border = "#334155"

                d_chip = (
                    f'<a href="https://in.tradingview.com/chart/?symbol=NSE:{sym}" target="_blank" rel="noopener noreferrer" '
                    f'class="drawer-stock-chip" '
                    f'data-in1m="{str(item["in_1m"]).lower()}" '
                    f'data-in3m="{str(item["in_3m"]).lower()}" '
                    f'data-in6m="{str(item["in_6m"]).lower()}" '
                    f'data-no-high="{str(is_no_high).lower()}" '
                    f'style="display:inline-block; margin:2px 3px; padding:3px 9px; background:{d_bg}; color:{d_color}; '
                    f'border:1px solid {d_border}; border-radius:4px; font-weight:700; font-size:0.78rem; text-decoration:none;" '
                    f'title="{sym} ({item["tier_name"]}) - Open Chart">{sym}</a>'
                )
                drawer_chips.append(d_chip)

            # Main Row HTML
            table_rows.append(
                f'<tr id="{row_id}" class="cluster-row">'
                f'<td style="text-align:center;color:#64748b;font-weight:600;">{rank}</td>'
                f'<td>'
                f'<div style="cursor:pointer; display:flex; align-items:center; gap:6px;" onclick="toggleDrawer(\'{drawer_id}\', \'{chevron_id}\')" title="Click to expand {industry} details">'
                f'<span id="{chevron_id}" style="display:inline-block; font-size:0.75rem; color:#38bdf8; transition:transform 0.2s ease;">▶</span>'
                f'<span style="font-weight:700; color:#f1f5f9; font-size:0.86rem;">{industry}</span>'
                f'</div>'
                f'</td>'
                f'<td style="text-align:center;">'
                f'<span class="tier-num-pill pill-tot" onclick="toggleRowTierNumber(this, \'{row_id}\', \'all\')" title="Click to view all {tot} stocks in {industry}">{tot}</span>'
                f'</td>'
                f'<td style="text-align:center;">'
                f'<span class="tier-num-pill pill-1m" onclick="toggleRowTierNumber(this, \'{row_id}\', \'1m\')" title="Click to filter {h1} stocks at 1M Highs in {industry}">{h1}</span>'
                f'</td>'
                f'<td style="text-align:center;">'
                f'<span class="tier-num-pill pill-3m" onclick="toggleRowTierNumber(this, \'{row_id}\', \'3m\')" title="Click to filter {h3} stocks at 3M Highs in {industry}">{h3}</span>'
                f'</td>'
                f'<td style="text-align:center;">'
                f'<span class="tier-num-pill pill-6m" onclick="toggleRowTierNumber(this, \'{row_id}\', \'6m\')" title="Click to filter {h6} stocks at 6M Highs in {industry}">{h6}</span>'
                f'</td>'
                f'<td style="text-align:center;font-weight:bold;color:#ffa000;">{score}</td>'
                f'<td>'
                f'<div style="display:flex; align-items:center; gap:4px; margin-bottom:5px; flex-wrap:wrap;">'
                f'<button class="row-filter-btn active-row-btn" onclick="filterRow(\'{row_id}\', \'active\', this)">All Highs ({len(leading_stocks)})</button>'
                f'<button class="row-filter-btn btn-6m" onclick="filterRow(\'{row_id}\', \'6m\', this)">6M ({h6})</button>'
                f'<button class="row-filter-btn btn-3m" onclick="filterRow(\'{row_id}\', \'3m\', this)">3M ({h3})</button>'
                f'<button class="row-filter-btn btn-1m" onclick="filterRow(\'{row_id}\', \'1m\', this)">1M ({h1})</button>'
                f'{"<button class=\'row-filter-btn btn-none\' onclick=\"filterRow(\'" + row_id + "\', \'none\', this)\">No High (" + str(len(stocks_no_high)) + ")</button>" if stocks_no_high else ""}'
                f'<span id="status-{row_id}" style="font-size:0.7rem; color:#94a3b8; margin-left:auto;"></span>'
                f'</div>'
                f'<div id="chips-{row_id}" class="stocks-chips-wrap">{" ".join(stock_chips)}</div>'
                f'</td>'
                f'</tr>'
            )

            # Drawer Row HTML matching Image 1
            drawer_text_list = " · ".join(leading_stocks[:10]) + (" ..." if len(leading_stocks) > 10 else "")
            table_rows.append(
                f'<tr id="{drawer_id}" class="cluster-drawer" style="display:none; background:rgba(15,23,42,0.95); border-bottom:1px solid rgba(56,189,248,0.25);">'
                f'<td colspan="8" style="padding:14px 18px;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:10px;">'
                f'<div style="font-size:0.92rem; font-weight:700; color:#f8fafc;">'
                f'▾ {industry} <span style="color:#64748b; font-weight:normal; font-size:0.8rem;">· {tot} stocks</span>'
                f'</div>'
                f'<div style="display:flex; align-items:center; gap:10px;">'
                f'<div style="width:200px; height:8px; background:#1e293b; border-radius:4px; overflow:hidden; display:flex;">'
                f'<div style="width:{pct_6m}%; height:100%; background:#00e676;" title="{h6} at 6M High"></div>'
                f'<div style="width:{max(0, pct_3m - pct_6m)}%; height:100%; background:#22d3ee;" title="{h3} at 3M High"></div>'
                f'<div style="width:{max(0, pct_1m - pct_3m)}%; height:100%; background:#38bdf8;" title="{h1} at 1M High"></div>'
                f'</div>'
                f'<span style="font-size:0.8rem; font-weight:600; color:#22d3ee;">{h3} of {tot} at 3M highs ({pct_3m}%)</span>'
                f'</div>'
                f'</div>'
                f'<div style="font-size:0.78rem; color:#94a3b8; margin-bottom:10px; letter-spacing:0.3px;">{drawer_text_list}</div>'
                f'<div id="drawer-chips-{row_id}" style="display:flex; flex-wrap:wrap; gap:4px; margin-bottom:10px;">{" ".join(drawer_chips)}</div>'
                f'<div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">'
                f'<span style="font-size:0.72rem; color:#64748b; text-transform:uppercase;">Filter:</span>'
                f'<button class="drawer-btn" onclick="filterDrawerChips(\'drawer-chips-{row_id}\', \'6m\', this)">6M high</button>'
                f'<button class="drawer-btn active-drawer-btn" onclick="filterDrawerChips(\'drawer-chips-{row_id}\', \'3m\', this)">3M high</button>'
                f'<button class="drawer-btn" onclick="filterDrawerChips(\'drawer-chips-{row_id}\', \'1m\', this)">1M high</button>'
                f'<button class="drawer-btn" onclick="filterDrawerChips(\'drawer-chips-{row_id}\', \'none\', this)">no fresh high</button>'
                f'<button class="drawer-btn" onclick="filterDrawerChips(\'drawer-chips-{row_id}\', \'all\', this)">all stocks</button>'
                f'</div>'
                f'</td>'
                f'</tr>'
            )

        tbody_html = "".join(table_rows)

        cluster_table_html = f"""
        <!-- Modal Overlay -->
        <div id="breadth-modal-overlay" onclick="closeBreadthModal(event)" style="
            display:none; position:fixed; top:0; left:0; width:100%; height:100%; z-index:9999;
            background:rgba(0,0,0,0.75); backdrop-filter:blur(3px); align-items:center; justify-content:center;">
          <div id="breadth-modal-box" onclick="event.stopPropagation()" style="
              background:#0d1526; border:1px solid rgba(56,189,248,0.4); border-radius:14px;
              padding:24px 28px; max-width:680px; width:92%; max-height:82vh; overflow-y:auto;
              box-shadow:0 8px 40px rgba(0,0,0,0.6);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
              <div>
                <h3 id="modal-industry-title" style="margin:0; color:#f8fafc; font-size:1.1rem; font-weight:700;"></h3>
                <span id="modal-industry-subtitle" style="font-size:0.78rem; color:#64748b;"></span>
              </div>
              <button onclick="closeBreadthModal(null)" style="
                  background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15);
                  color:#94a3b8; border-radius:6px; padding:4px 10px; cursor:pointer; font-size:1rem;">✕</button>
            </div>
            <!-- Tab buttons inside modal -->
            <div style="display:flex; gap:8px; margin-bottom:14px; flex-wrap:wrap;">
              <button class="modal-tab-btn active-modal-tab" id="modal-tab-6m" onclick="switchModalTab('6m', this)">
                🟢 6M Highs <span id="modal-cnt-6m" style="font-size:0.75rem; color:#6ee7b7; margin-left:4px;"></span>
              </button>
              <button class="modal-tab-btn" id="modal-tab-3m" onclick="switchModalTab('3m', this)">
                🔷 3M High Only <span id="modal-cnt-3m" style="font-size:0.75rem; color:#67e8f9; margin-left:4px;"></span>
              </button>
              <button class="modal-tab-btn" id="modal-tab-1m" onclick="switchModalTab('1m', this)">
                ⚡ 1M High Only <span id="modal-cnt-1m" style="font-size:0.75rem; color:#93c5fd; margin-left:4px;"></span>
              </button>
              <button class="modal-tab-btn" id="modal-tab-all" onclick="switchModalTab('all', this)">
                👥 All Highs
              </button>
            </div>
            <div id="modal-stocks-panel" style="display:flex; flex-wrap:wrap; gap:6px; min-height:60px;"></div>
            <p style="font-size:0.72rem; color:#475569; margin-top:14px; margin-bottom:0;">
              Click any ticker to open its TradingView chart. 🟢 = 6M &nbsp; 🔷 = 3M only &nbsp; ⚡ = 1M only
            </p>
          </div>
        </div>

        <style>
        .cluster-wrap {{ font-family:'Segoe UI',system-ui,sans-serif; color:#cbd5e1; }}
        .cluster-table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
        .cluster-table th {{ padding:10px 12px; font-size:0.74rem; letter-spacing:0.5px; text-transform:uppercase; border-bottom:1px solid rgba(255,255,255,0.1); background:rgba(255,255,255,0.02); }}
        .cluster-table td {{ padding:10px 12px; border-bottom:1px solid rgba(255,255,255,0.05); vertical-align:middle; }}

        .tier-num-pill {{
            display:inline-block; min-width:32px; padding:3px 8px; border-radius:6px; font-weight:700;
            text-align:center; cursor:pointer; user-select:none; transition:all 0.15s ease-in-out;
        }}
        .tier-num-pill:hover {{ transform:scale(1.15); filter:brightness(1.3); }}

        .pill-1m {{ color:#38bdf8; background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3); }}
        .pill-3m {{ color:#22d3ee; background:rgba(34,211,238,0.1); border:1px solid rgba(34,211,238,0.3); }}
        .pill-6m {{ color:#00e676; background:rgba(0,230,118,0.1); border:1px solid rgba(0,230,118,0.3); }}
        .pill-tot {{ color:#cbd5e1; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.15); }}

        .pill-1m.active-pill {{ background:#38bdf8; color:#04233a; font-weight:800; border-color:#38bdf8; box-shadow:0 0 12px rgba(56,189,248,0.7); }}
        .pill-3m.active-pill {{ background:#22d3ee; color:#042f2e; font-weight:800; border-color:#22d3ee; box-shadow:0 0 12px rgba(34,211,238,0.7); }}
        .pill-6m.active-pill {{ background:#00e676; color:#052e16; font-weight:800; border-color:#00e676; box-shadow:0 0 12px rgba(0,230,118,0.7); }}
        .pill-tot.active-pill {{ background:#a855f7; color:#ffffff; font-weight:800; border-color:#a855f7; box-shadow:0 0 12px rgba(168,85,247,0.7); }}

        .row-filter-btn {{
            background:rgba(255,255,255,0.04); color:#94a3b8; border:1px solid rgba(255,255,255,0.1);
            border-radius:4px; font-size:0.68rem; padding:2px 7px; cursor:pointer; transition:all 0.15s;
        }}
        .row-filter-btn:hover {{ background:rgba(255,255,255,0.1); color:#f1f5f9; }}
        .row-filter-btn.active-row-btn {{ background:rgba(56,189,248,0.2); color:#38bdf8; border-color:rgba(56,189,248,0.5); font-weight:700; }}
        .row-filter-btn.btn-6m.active-row-btn {{ background:rgba(0,230,118,0.2); color:#00e676; border-color:rgba(0,230,118,0.5); font-weight:700; }}
        .row-filter-btn.btn-3m.active-row-btn {{ background:rgba(34,211,238,0.2); color:#22d3ee; border-color:rgba(34,211,238,0.5); font-weight:700; }}
        .row-filter-btn.btn-1m.active-row-btn {{ background:rgba(56,189,248,0.2); color:#38bdf8; border-color:rgba(56,189,248,0.5); font-weight:700; }}

        .global-filter-btn {{
            background:rgba(255,255,255,0.05); color:#cbd5e1; border:1px solid rgba(255,255,255,0.12);
            border-radius:6px; font-size:0.75rem; padding:4px 10px; cursor:pointer; transition:all 0.15s; font-weight:600;
        }}
        .global-filter-btn:hover {{ background:rgba(255,255,255,0.12); color:#ffffff; }}
        .global-filter-btn.active-global-btn {{ background:#38bdf8; color:#04233a; border-color:#38bdf8; font-weight:800; box-shadow:0 0 10px rgba(56,189,248,0.4); }}

        .drawer-btn {{
            background:rgba(255,255,255,0.05); color:#94a3b8; border:1px solid rgba(255,255,255,0.15);
            border-radius:4px; font-size:0.72rem; padding:3px 8px; cursor:pointer; transition:all 0.15s; font-weight:600;
        }}
        .drawer-btn:hover {{ background:rgba(255,255,255,0.12); color:#f8fafc; }}
        .drawer-btn.active-drawer-btn {{ background:#22d3ee; color:#042f2e; border-color:#22d3ee; font-weight:800; }}

        .stock-chip {{
            display:inline-block; margin:2px 3px; padding:3px 7px; border-radius:4px;
            font-size:0.77rem; font-weight:600; text-decoration:none; transition:all 0.15s;
        }}
        .stock-chip:hover {{ transform:scale(1.05); filter:brightness(1.15); }}
        .stock-chip.tier-6m {{ background:rgba(0,230,118,0.15); color:#00e676; border:1px solid rgba(0,230,118,0.4); }}
        .stock-chip.tier-3m {{ background:rgba(34,211,238,0.13); color:#22d3ee; border:1px solid rgba(34,211,238,0.35); }}
        .stock-chip.tier-1m {{ background:rgba(56,189,248,0.12); color:#38bdf8; border:1px solid rgba(56,189,248,0.3); }}
        .stock-chip.tier-none {{ background:rgba(30,41,59,0.5); color:#94a3b8; border:1px solid rgba(71,85,105,0.4); }}

        .modal-tab-btn {{
            background:rgba(255,255,255,0.05); color:#94a3b8; border:1px solid rgba(255,255,255,0.12);
            border-radius:6px; font-size:0.78rem; padding:5px 12px; cursor:pointer; transition:all 0.15s; font-weight:600;
        }}
        .modal-tab-btn:hover {{ background:rgba(255,255,255,0.12); color:#f8fafc; }}
        .modal-tab-btn.active-modal-tab {{ background:#38bdf8; color:#04233a; border-color:#38bdf8; font-weight:800; box-shadow:0 0 10px rgba(56,189,248,0.4); }}
        .modal-chip {{
            display:inline-block; padding:5px 11px; border-radius:5px;
            font-size:0.82rem; font-weight:700; text-decoration:none; transition:all 0.15s;
        }}
        .modal-chip:hover {{ transform:scale(1.06); filter:brightness(1.2); }}
        .modal-chip-6m {{ background:rgba(0,230,118,0.18); color:#00e676; border:1px solid rgba(0,230,118,0.5); }}
        .modal-chip-3m {{ background:rgba(34,211,238,0.15); color:#22d3ee; border:1px solid rgba(34,211,238,0.4); }}
        .modal-chip-1m {{ background:rgba(56,189,248,0.14); color:#38bdf8; border:1px solid rgba(56,189,248,0.4); }}
        </style>

        <div class="cluster-wrap">
          <!-- Global Filter Bar -->
          <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; background:rgba(15,23,42,0.8); padding:10px 16px; border-radius:8px; border:1px solid rgba(255,255,255,0.08); flex-wrap:wrap; gap:8px;">
            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
              <span style="font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.5px;">Global Timeframe Filter:</span>
              <button class="global-filter-btn active-global-btn" onclick="filterAllRows('active', this)">🌟 All Active Highs</button>
              <button class="global-filter-btn" onclick="filterAllRows('6m', this)">🟢 6M Highs Only</button>
              <button class="global-filter-btn" onclick="filterAllRows('3m', this)">🔷 3M Highs Only</button>
              <button class="global-filter-btn" onclick="filterAllRows('1m', this)">⚡ 1M Highs Only</button>
              <button class="global-filter-btn" onclick="filterAllRows('all', this)">👥 All Universe Stocks</button>
            </div>
            <div style="font-size:0.74rem; color:#64748b;">
              💡 <i>Tip: Click any number in <b>1M HIGHS</b>, <b>3M HIGHS</b>, or <b>6M HIGHS</b> columns to filter that group instantly.</i>
            </div>
          </div>

          <div style="overflow-x:auto; border:1px solid rgba(255,255,255,0.08); border-radius:10px; background:rgba(9,13,22,0.9); box-shadow:0 4px 20px rgba(0,0,0,0.25);">
            <table class="cluster-table">
              <thead>
                <tr>
                  <th style="text-align:center;width:45px;">#</th>
                  <th style="text-align:left;width:220px;">Industry Group</th>
                  <th style="text-align:center;width:65px;cursor:pointer;" onclick="filterAllRows('all')" title="Click to view all stocks across all groups">Total ⇲</th>
                  <th style="text-align:center;width:80px;color:#38bdf8;cursor:pointer;" onclick="filterAllRows('1m')" title="Click to filter all rows to 1M Highs">1M Highs ⇲</th>
                  <th style="text-align:center;width:80px;color:#22d3ee;cursor:pointer;" onclick="filterAllRows('3m')" title="Click to filter all rows to 3M Highs">3M Highs ⇲</th>
                  <th style="text-align:center;width:80px;color:#00e676;cursor:pointer;" onclick="filterAllRows('6m')" title="Click to filter all rows to 6M Highs">6M Highs ⇲</th>
                  <th style="text-align:center;width:70px;color:#ffa000;">Score</th>
                  <th style="text-align:left;">Leading Institutional Stocks <span style="color:#38bdf8;font-weight:normal;font-size:0.7rem;">(Click ticker to open TradingView chart)</span></th>
                </tr>
              </thead>
              <tbody>{tbody_html}</tbody>
            </table>
          </div>
        </div>

        <script>
        // ── Cluster data injected from Python ──
        const CLUSTER_DATA = {cluster_js_json};

        // ── Modal helpers ──
        let _modalCurrentData = null;
        let _modalCurrentTab = '6m';

        function openBreadthModal(rowId, startTab) {{
          const data = CLUSTER_DATA[rowId];
          if (!data) return;
          _modalCurrentData = data;
          _modalCurrentTab = startTab || '6m';

          document.getElementById('modal-industry-title').textContent = data.industry;
          document.getElementById('modal-industry-subtitle').textContent =
            data.total + ' stocks total · ' + data.h6m + ' at 6M · ' + data.h3m + ' at 3M · ' + data.h1m + ' at 1M high';
          document.getElementById('modal-cnt-6m').textContent = '(' + data.s6m.length + ')';
          document.getElementById('modal-cnt-3m').textContent = '(' + data.s3m.length + ')';
          document.getElementById('modal-cnt-1m').textContent = '(' + data.s1m.length + ')';

          const overlay = document.getElementById('breadth-modal-overlay');
          overlay.style.display = 'flex';
          switchModalTab(_modalCurrentTab, document.getElementById('modal-tab-' + _modalCurrentTab));
        }}

        function closeBreadthModal(evt) {{
          if (evt && evt.target !== document.getElementById('breadth-modal-overlay')) return;
          document.getElementById('breadth-modal-overlay').style.display = 'none';
        }}

        function switchModalTab(tab, btn) {{
          _modalCurrentTab = tab;
          document.querySelectorAll('.modal-tab-btn').forEach(b => b.classList.remove('active-modal-tab'));
          if (btn) btn.classList.add('active-modal-tab');

          const panel = document.getElementById('modal-stocks-panel');
          if (!panel || !_modalCurrentData) return;
          panel.innerHTML = '';

          let stocks = [];
          let chipClass = 'modal-chip-6m';
          let emptyMsg = '';
          const d = _modalCurrentData;

          if (tab === '6m') {{
            stocks = d.s6m;
            chipClass = 'modal-chip-6m';
            emptyMsg = 'No stocks at 6-Month High in this group.';
          }} else if (tab === '3m') {{
            stocks = d.s3m;
            chipClass = 'modal-chip-3m';
            emptyMsg = 'No stocks at 3M High only (not yet 6M) in this group.';
          }} else if (tab === '1m') {{
            stocks = d.s1m;
            chipClass = 'modal-chip-1m';
            emptyMsg = 'No stocks at 1M High only in this group.';
          }} else {{
            // All highs: show all three tiers combined, color-coded
            const all6 = (d.s6m || []).map(s => [s, 'modal-chip-6m', '🟢']);
            const all3 = (d.s3m || []).map(s => [s, 'modal-chip-3m', '🔷']);
            const all1 = (d.s1m || []).map(s => [s, 'modal-chip-1m', '⚡']);
            const combined = [...all6, ...all3, ...all1];
            if (combined.length === 0) {{
              panel.innerHTML = '<span style="color:#64748b;font-size:0.82rem;">No fresh high stocks in this group.</span>';
              return;
            }}
            combined.forEach(([sym, cls, icon]) => {{
              const a = document.createElement('a');
              a.href = 'https://in.tradingview.com/chart/?symbol=NSE:' + sym;
              a.target = '_blank';
              a.rel = 'noopener noreferrer';
              a.className = 'modal-chip ' + cls;
              a.textContent = icon + ' ' + sym;
              panel.appendChild(a);
            }});
            return;
          }}

          if (!stocks || stocks.length === 0) {{
            panel.innerHTML = '<span style="color:#64748b;font-size:0.82rem;">' + emptyMsg + '</span>';
            return;
          }}
          stocks.forEach(sym => {{
            const a = document.createElement('a');
            a.href = 'https://in.tradingview.com/chart/?symbol=NSE:' + sym;
            a.target = '_blank';
            a.rel = 'noopener noreferrer';
            a.className = 'modal-chip ' + chipClass;
            a.textContent = sym + ' ↗';
            panel.appendChild(a);
          }});
        }}

        // ── Existing row-level filtering ──
        function filterRow(rowId, tier, btnElement) {{
          const container = document.getElementById('chips-' + rowId);
          const statusEl = document.getElementById('status-' + rowId);
          if (!container) return;

          const rowTr = document.getElementById(rowId);
          if (rowTr) {{
            rowTr.querySelectorAll('.row-filter-btn').forEach(b => b.classList.remove('active-row-btn'));
            rowTr.querySelectorAll('.tier-num-pill').forEach(b => b.classList.remove('active-pill'));
          }}
          if (btnElement) {{
            if (btnElement.classList.contains('tier-num-pill')) {{
              btnElement.classList.add('active-pill');
            }} else {{
              btnElement.classList.add('active-row-btn');
            }}
          }}

          const chips = container.querySelectorAll('.stock-chip');
          let count = 0;
          chips.forEach(chip => {{
            let show = false;
            if (tier === 'all') {{
              show = true;
            }} else if (tier === 'active') {{
              show = chip.getAttribute('data-no-high') !== 'true';
            }} else if (tier === '6m') {{
              show = chip.getAttribute('data-in6m') === 'true';
            }} else if (tier === '3m') {{
              show = chip.getAttribute('data-in3m') === 'true';
            }} else if (tier === '1m') {{
              show = chip.getAttribute('data-in1m') === 'true';
            }} else if (tier === 'none') {{
              show = chip.getAttribute('data-no-high') === 'true';
            }}

            if (show) {{
              chip.style.display = 'inline-block';
              count++;
            }} else {{
              chip.style.display = 'none';
            }}
          }});

          if (statusEl) {{
            const labels = {{
              '6m': '6M Highs',
              '3m': '3M Highs',
              '1m': '1M Highs',
              'active': 'All Fresh Highs',
              'all': 'All Universe',
              'none': 'No Fresh Highs'
            }};
            statusEl.innerHTML = `<span style="color:#38bdf8;font-weight:bold;">${{count}}</span> ${{labels[tier] || tier}}`;
          }}
        }}

        function toggleRowTierNumber(el, rowId, tier) {{
          // Open modal showing 6M/3M/1M breakdown for this row
          const startTab = (tier === '1m' || tier === '3m' || tier === '6m') ? tier : '6m';
          openBreadthModal(rowId, startTab);
        }}

        function filterAllRows(tier, headerBtn) {{
          document.querySelectorAll('.global-filter-btn').forEach(b => b.classList.remove('active-global-btn'));
          if (headerBtn) headerBtn.classList.add('active-global-btn');

          document.querySelectorAll('.cluster-row').forEach(row => {{
            const rowId = row.id;
            filterRow(rowId, tier, null);
          }});
        }}

        function toggleDrawer(drawerId, chevronId) {{
          const d = document.getElementById(drawerId);
          const ch = document.getElementById(chevronId);
          if (!d) return;
          if (d.style.display === 'none' || d.style.display === '') {{
            d.style.display = 'table-row';
            if (ch) ch.style.transform = 'rotate(90deg)';
          }} else {{
            d.style.display = 'none';
            if (ch) ch.style.transform = 'rotate(0deg)';
          }}
        }}

        function filterDrawerChips(containerId, tier, btn) {{
          const container = document.getElementById(containerId);
          if (!container) return;
          const parent = btn.parentElement;
          if (parent) {{
            parent.querySelectorAll('.drawer-btn').forEach(b => b.classList.remove('active-drawer-btn'));
            btn.classList.add('active-drawer-btn');
          }}
          const chips = container.querySelectorAll('.drawer-stock-chip');
          chips.forEach(chip => {{
            let match = false;
            if (tier === 'all') match = true;
            else if (tier === '6m') match = chip.getAttribute('data-in6m') === 'true';
            else if (tier === '3m') match = chip.getAttribute('data-in3m') === 'true';
            else if (tier === '1m') match = chip.getAttribute('data-in1m') === 'true';
            else if (tier === 'none') match = chip.getAttribute('data-no-high') === 'true';

            if (match) {{
              chip.style.opacity = '1';
              chip.style.filter = 'none';
              chip.style.borderColor = (tier === '6m' ? '#00e676' : (tier === '3m' ? '#22d3ee' : '#38bdf8'));
            }} else {{
              chip.style.opacity = '0.35';
              chip.style.filter = 'grayscale(70%)';
            }}
          }});
        }}
        </script>
        """
        components.html(cluster_table_html, height=min(180 + len(display_clusters) * 90, 3600), scrolling=True)
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
        regime_note = "🟢 <b>AGGRESSIVE (Rule #4 - Follow The Trend):</b> Broad market supports full-sized positions. All picks below are actionable."
        regime_note_color = "#00e676"
    elif regime_m50 >= 40.0:
        regime_note = "🟡 <b>SELECTIVE (Rule #1 - Start Small):</b> Trade only <b>Top Leading Cluster</b> stocks. Begin with pilot positions (33-50%) and pyramid on confirmation."
        regime_note_color = "#fbbf24"
    else:
        regime_note = "🔴 <b>DEFENSIVE (Rule #9 - Don't Force Trades):</b> Avoid new longs. Missing an opportunity is better than taking a poor trade. Picks below are watchlist candidates only."
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

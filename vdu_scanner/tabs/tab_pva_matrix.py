import streamlit as st
import pandas as pd
from datetime import datetime

from config import IST_TIMEZONE
from utils import get_day_change_badge_html
import database
from export_pva_matrix import classify_pva, generate_pva_excel_bytes

def trend_badge(t_val):
    if t_val == 1:
        return "<span style='color:#00e676; font-weight:bold;'>▲ Up</span>"
    elif t_val == -1:
        return "<span style='color:#ef4444; font-weight:bold;'>▼ Dn</span>"
    return "<span style='color:#fbbf24; font-weight:bold;'>◼ Neu</span>"

def blast_tier_color(tier):
    if "BLAST" in tier:
        return "#ff4444"
    elif "Strong" in tier:
        return "#ff9900"
    elif "Moderate" in tier:
        return "#29b6f6"
    return "#64748b"

def render():
    st.markdown("### 🔬 6-Condition Price + Volume Matrix (Wyckoff / PVA Analysis)")
    st.markdown(
        "<p style='font-size:0.92rem; color:#94a3b8; margin-top:-6px;'>"
        "Classifies NSE stocks into <b>6 distinct Price + Volume regimes</b> with automated trading interpretations, "
        "detecting accumulation, momentum breakouts, selling exhaustion, and distribution."
        "</p>",
        unsafe_allow_html=True
    )

    # Retrieve data from session state or database cache
    vpa_data = st.session_state.get('vpa_results')
    if not vpa_data:
        try:
            all_latest = database.get_all_latest_scan_dates()
            latest_vpa_d = all_latest.get('scanned_vpa')
            if latest_vpa_d:
                cached = database.get_cached_vpa(latest_vpa_d)
                if cached:
                    vpa_data = cached
                    st.session_state.vpa_results = cached
                    st.session_state['vpa_results_date'] = latest_vpa_d
        except Exception as _e:
            print(f"Error hydrating PVA matrix from cache: {_e}")

        if not vpa_data:
            dates = database.get_available_scan_dates()
            if dates:
                for d in dates:
                    cached = database.get_cached_vpa(d)
                    if cached:
                        vpa_data = cached
                        st.session_state.vpa_results = cached
                        st.session_state['vpa_results_date'] = d
                        break

    if not vpa_data:
        st.info("ℹ️ No Price + Volume scan data available in session. Please run a scan from the VPA tab or load historical sessions from History.")
        return

    # Data enrichment: ensure PVA classification and blast scores
    try:
        from tabs.tab_vpa import compute_blast_score
    except Exception:
        compute_blast_score = lambda r: {'blast_score': 0, 'blast_tier': '', 'blast_reasons': []}

    for r in vpa_data:
        if 'vol_ratio_5d' not in r:
            r['vol_ratio_5d'] = r.get('vol_ratio_5d') or r.get('vol_ratio') or 1.0
        if 'blast_score' not in r:
            r['near_breakout'] = r.get('near_breakout', False)
            r.update(compute_blast_score(r))
        if 'pva' not in r:
            r['pva'] = classify_pva(r.get('day_change_pct', 0.0), r.get('vol_ratio_5d', 1.0), r.get('vol_rising'))

    # ─────────────────────────────────────────────────────────────────────────
    # 1. REFERENCE EXPLANATION EXPANDER
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("📖 View 6 Market Conditions Reference Matrix & Trading Interpretations", expanded=False):
        st.markdown("""
| Volume Movement | Price Movement | Automated Interpretation | Signal | Actionable Bias |
| :--- | :--- | :--- | :--- | :--- |
| **Increasing** | **Increasing** | Strong buying interest; possible accumulation or breakout | <span style="color:#00e676; font-weight:bold;">🟢 Bullish</span> | Look for momentum long entries / add to winners |
| **Increasing** | **Stable** | Buyers and sellers competing; possible accumulation | <span style="color:#38bdf8; font-weight:bold;">🔵 Watch</span> | Prime absorption zone; set alert on upside range break |
| **Decreasing** | **Increasing** | Price rising with weaker participation; rally may lack strength | <span style="color:#fbbf24; font-weight:bold;">🟡 Caution</span> | Tighten trailing stops; avoid chasing extended entries |
| **Decreasing** | **Stable** | Low participation; consolidation or lack of interest | <span style="color:#94a3b8; font-weight:bold;">⚪ Neutral</span> | Sideways churn / drift; wait for volume trigger |
| **Decreasing** | **Decreasing** | Selling pressure may be weakening; possible selling exhaustion | <span style="color:#c084fc; font-weight:bold;">🟣 Watch reversal</span> | Watch for bottom reversal / Wyckoff spring setup |
| **Increasing** | **Decreasing** | Strong selling pressure; possible distribution or breakdown | <span style="color:#ef4444; font-weight:bold;">🔴 Bearish</span> | High breakdown risk; exit long positions or short |
""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. 6 LIVE KPI METRIC CARDS
    # ─────────────────────────────────────────────────────────────────────────
    pva_counts = {
        'Bullish': 0,
        'Watch': 0,
        'Caution': 0,
        'Neutral': 0,
        'Watch reversal': 0,
        'Bearish': 0,
    }
    for r in vpa_data:
        pva = r.get('pva') or classify_pva(r.get('day_change_pct', 0.0), r.get('vol_ratio_5d', 1.0), r.get('vol_rising'))
        sig = pva['signal']
        pva_counts[sig] = pva_counts.get(sig, 0) + 1

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #00e676;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">🟢 Bullish</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#00e676;">{pva_counts.get("Bullish", 0)}</h3><span style="font-size:0.68rem; color:#64748b;">Vol ▲ Price ▲</span></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #38bdf8;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">🔵 Watch</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#38bdf8;">{pva_counts.get("Watch", 0)}</h3><span style="font-size:0.68rem; color:#64748b;">Vol ▲ Price ◼</span></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #fbbf24;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">🟡 Caution</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#fbbf24;">{pva_counts.get("Caution", 0)}</h3><span style="font-size:0.68rem; color:#64748b;">Vol ▼ Price ▲</span></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #94a3b8;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">⚪ Neutral</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#94a3b8;">{pva_counts.get("Neutral", 0)}</h3><span style="font-size:0.68rem; color:#64748b;">Vol ▼ Price ◼</span></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #c084fc;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">🟣 Reversal</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#c084fc;">{pva_counts.get("Watch reversal", 0)}</h3><span style="font-size:0.68rem; color:#64748b;">Vol ▼ Price ▼</span></div>', unsafe_allow_html=True)
    k6.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #ef4444;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">🔴 Bearish</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#ef4444;">{pva_counts.get("Bearish", 0)}</h3><span style="font-size:0.68rem; color:#64748b;">Vol ▲ Price ▼</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. INTERACTIVE CONTROLS
    # ─────────────────────────────────────────────────────────────────────────
    f_col1, f_col2, f_col3, f_col4 = st.columns([3, 2, 2, 3])
    with f_col1:
        signal_filter_options = [
            f"All ({len(vpa_data)})",
            f"🟢 Bullish ({pva_counts.get('Bullish', 0)})",
            f"🔵 Watch ({pva_counts.get('Watch', 0)})",
            f"🟡 Caution ({pva_counts.get('Caution', 0)})",
            f"⚪ Neutral ({pva_counts.get('Neutral', 0)})",
            f"🟣 Watch reversal ({pva_counts.get('Watch reversal', 0)})",
            f"🔴 Bearish ({pva_counts.get('Bearish', 0)})"
        ]
        chosen_filter = st.selectbox("Filter by Market Regime:", signal_filter_options, key="tab_pva_cond_sel")
    with f_col2:
        trend_tf = st.selectbox(
            "📅 Minor/Mid/Major Timeframe:",
            ["Daily", "Weekly", "Monthly", "All (D/W/M)"],
            index=0,  # Default to Daily!
            key="tab_pva_tf_sel"
        )
    with f_col3:
        pva_sort = st.selectbox("Sort Matrix By:", ["Volume Ratio ↓", "Blast Score ↓", "Change % ↓", "CMP ↑"], key="tab_pva_sort_sel")
    with f_col4:
        pva_search = st.text_input("🔍 Search Stock Symbol:", "", key="tab_pva_sym_search", placeholder="e.g. RELIANCE, TCS...")

    # Filter logic
    selected_sig_name = None
    if "Bullish" in chosen_filter:
        selected_sig_name = "Bullish"
    elif "Watch reversal" in chosen_filter:
        selected_sig_name = "Watch reversal"
    elif "Watch" in chosen_filter:
        selected_sig_name = "Watch"
    elif "Caution" in chosen_filter:
        selected_sig_name = "Caution"
    elif "Neutral" in chosen_filter:
        selected_sig_name = "Neutral"
    elif "Bearish" in chosen_filter:
        selected_sig_name = "Bearish"

    filtered_stocks = []
    for r in vpa_data:
        pva = r.get('pva') or classify_pva(r.get('day_change_pct', 0.0), r.get('vol_ratio_5d', 1.0), r.get('vol_rising'))
        if selected_sig_name and pva['signal'] != selected_sig_name:
            continue
        if pva_search and pva_search.strip().upper() not in r.get('symbol', '').upper():
            continue
        filtered_stocks.append(r)

    # Sort
    if pva_sort == "Volume Ratio ↓":
        filtered_stocks.sort(key=lambda x: -float(x.get('vol_ratio_5d') or x.get('vol_ratio') or 1.0))
    elif pva_sort == "Blast Score ↓":
        filtered_stocks.sort(key=lambda x: -x.get('blast_score', 0))
    elif pva_sort == "Change % ↓":
        filtered_stocks.sort(key=lambda x: -x.get('day_change_pct', 0.0))
    elif pva_sort == "CMP ↑":
        filtered_stocks.sort(key=lambda x: x.get('cmp', 0.0))

    # Configure 3 Trend columns based on chosen timeframe
    if trend_tf == "Daily":
        th_minor, th_mid, th_major = "Minor (Daily)", "Mid (Daily)", "Major (Daily)"
    elif trend_tf == "Weekly":
        th_minor, th_mid, th_major = "Minor (Weekly)", "Mid (Weekly)", "Major (Weekly)"
    elif trend_tf == "Monthly":
        th_minor, th_mid, th_major = "Minor (Monthly)", "Mid (Monthly)", "Major (Monthly)"
    else:
        th_minor, th_mid, th_major = "Minor (D/W/M)", "Mid (D/W/M)", "Major (D/W/M)"

    # ─────────────────────────────────────────────────────────────────────────
    # 4. RENDER TABLE
    # ─────────────────────────────────────────────────────────────────────────
    pva_rows = []
    for idx, r in enumerate(filtered_stocks[:200], start=1):
        pva = r.get('pva') or classify_pva(r.get('day_change_pct', 0.0), r.get('vol_ratio_5d', 1.0), r.get('vol_rising'))
        d = r.get('daily', {})
        w = r.get('weekly', {})
        m = r.get('monthly', {})
        sym = r.get('symbol', '')
        tv_link = f"https://in.tradingview.com/chart/?symbol=NSE:{sym.replace('.NS','')}"

        vr = float(r.get('vol_ratio_5d') or r.get('vol_ratio') or 1.0)
        vol_mov_badge = "<span style='color:#00e676; font-weight:600;'>▲ Increasing</span>" if pva['vol_movement'] == "Increasing" else "<span style='color:#94a3b8; font-weight:600;'>▼ Decreasing</span>"
        if pva['price_movement'] == "Increasing":
            price_mov_badge = "<span style='color:#00e676; font-weight:600;'>▲ Increasing</span>"
        elif pva['price_movement'] == "Decreasing":
            price_mov_badge = "<span style='color:#ef4444; font-weight:600;'>▼ Decreasing</span>"
        else:
            price_mov_badge = "<span style='color:#38bdf8; font-weight:600;'>◼ Stable</span>"

        sig_badge = f"<span style='background:rgba(255,255,255,0.06); color:{pva['color']}; border:1px solid {pva['color']}; padding:3px 8px; border-radius:4px; font-weight:bold;'>{pva['signal']}</span>"
        blast_badge = f"<span style='color:{blast_tier_color(r.get('blast_tier',''))}; font-weight:bold;'>{r.get('blast_score', 0)} ({r.get('blast_tier','')})</span>"

        if trend_tf == "Daily":
            c_minor, c_mid, c_major = trend_badge(d.get('minor', 0)), trend_badge(d.get('mid', 0)), trend_badge(d.get('major', 0))
        elif trend_tf == "Weekly":
            c_minor, c_mid, c_major = trend_badge(w.get('minor', 0)), trend_badge(w.get('mid', 0)), trend_badge(w.get('major', 0))
        elif trend_tf == "Monthly":
            c_minor, c_mid, c_major = trend_badge(m.get('minor', 0)), trend_badge(m.get('mid', 0)), trend_badge(m.get('major', 0))
        else:
            c_minor = f"{trend_badge(d.get('minor',0))} / {trend_badge(w.get('minor',0))} / {trend_badge(m.get('minor',0))}"
            c_mid = f"{trend_badge(d.get('mid',0))} / {trend_badge(w.get('mid',0))} / {trend_badge(m.get('mid',0))}"
            c_major = f"{trend_badge(d.get('major',0))} / {trend_badge(w.get('major',0))} / {trend_badge(m.get('major',0))}"

        pva_rows.append(f"""<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:9px 8px; color:#94a3b8; font-weight:bold;">#{idx}</td>
<td style="padding:9px 8px;"><a href="{tv_link}" target="_blank" style="color:#00e5ff; font-weight:bold; text-decoration:none;">{sym}</a></td>
<td style="padding:9px 8px; color:#e2e8f0;">₹{r.get('cmp', 0.0):,.2f}</td>
<td style="padding:9px 8px;">{get_day_change_badge_html(r.get('day_change_pct', 0.0))}</td>
<td style="padding:9px 8px; color:{'#00e676' if vr>=1.5 else '#38bdf8' if vr>=1.0 else '#94a3b8'}; font-weight:bold;">{vr:.2f}x</td>
<td style="padding:9px 8px;">{vol_mov_badge}</td>
<td style="padding:9px 8px;">{price_mov_badge}</td>
<td style="padding:9px 8px;">{sig_badge}</td>
<td style="padding:9px 8px; font-size:0.83rem; color:#cbd5e1; max-width:320px;">{pva['interpretation']}</td>
<td style="padding:9px 8px; text-align:center;">{c_minor}</td>
<td style="padding:9px 8px; text-align:center;">{c_mid}</td>
<td style="padding:9px 8px; text-align:center;">{c_major}</td>
<td style="padding:9px 8px;">{blast_badge}</td>
</tr>""")

    pva_table_html = f"""<div style="overflow-x:auto; margin-top:8px;">
<table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
<thead>
<tr style="background:rgba(56,189,248,0.08); border-bottom:2px solid rgba(56,189,248,0.3);">
<th style="padding:10px 8px; color:#38bdf8;">#</th>
<th style="padding:10px 8px; color:#38bdf8;">Symbol</th>
<th style="padding:10px 8px; color:#38bdf8;">CMP</th>
<th style="padding:10px 8px; color:#38bdf8;">Chg%</th>
<th style="padding:10px 8px; color:#38bdf8;">Vol Ratio</th>
<th style="padding:10px 8px; color:#38bdf8;">Volume Movement</th>
<th style="padding:10px 8px; color:#38bdf8;">Price Movement</th>
<th style="padding:10px 8px; color:#38bdf8;">Signal</th>
<th style="padding:10px 8px; color:#38bdf8;">Automatic Interpretation</th>
<th style="padding:10px 8px; color:#38bdf8; text-align:center;">{th_minor}</th>
<th style="padding:10px 8px; color:#38bdf8; text-align:center;">{th_mid}</th>
<th style="padding:10px 8px; color:#38bdf8; text-align:center;">{th_major}</th>
<th style="padding:10px 8px; color:#38bdf8;">Blast Score</th>
</tr>
</thead>
<tbody>
{"".join(pva_rows)}
</tbody>
</table>
</div>"""
    st.markdown(pva_table_html, unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.78rem; color:#64748b; margin-top:6px;'>Showing {len(filtered_stocks[:200])} of {len(filtered_stocks)} matching stocks</p>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. DUAL DOWNLOAD BUTTONS (EXCEL & CSV)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    dl_c1, dl_c2 = st.columns(2)
    with dl_c1:
        try:
            excel_bytes = generate_pva_excel_bytes(vpa_data)
            st.download_button(
                label="📥 Download 6-Condition Matrix (Excel .xlsx)",
                data=excel_bytes,
                file_name=f"price_volume_matrix_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Includes both the Price_Volume_Matrix sheet and the Market_Conditions_Legend sheet, ready for Google Sheets or Microsoft Excel.",
                key="pva_matrix_export_excel_btn"
            )
        except Exception as e:
            st.error(f"Excel export error: {e}")

    with dl_c2:
        csv_rows = []
        for r in vpa_data:
            pva = r.get('pva') or classify_pva(r.get('day_change_pct', 0.0), r.get('vol_ratio_5d', 1.0), r.get('vol_rising'))
            d = r.get('daily', {})
            w = r.get('weekly', {})
            m = r.get('monthly', {})
            def _tstr(val):
                return "Up" if val == 1 else ("Down" if val == -1 else "Neutral")
            csv_rows.append({
                'Symbol': r.get('symbol', ''),
                'Company Name': r.get('company_name', ''),
                'CMP': r.get('cmp', 0.0),
                'Change %': round(r.get('day_change_pct', 0.0), 2),
                'Volume Ratio': round(float(r.get('vol_ratio_5d') or r.get('vol_ratio') or 1.0), 2),
                'Volume Movement': pva['vol_movement'],
                'Price Movement': pva['price_movement'],
                'Signal': pva['signal'],
                'Automatic Interpretation': pva['interpretation'],
                'Daily Minor': _tstr(d.get('minor', 0)),
                'Daily Mid': _tstr(d.get('mid', 0)),
                'Daily Major': _tstr(d.get('major', 0)),
                'Weekly Minor': _tstr(w.get('minor', 0)),
                'Weekly Mid': _tstr(w.get('mid', 0)),
                'Weekly Major': _tstr(w.get('major', 0)),
                'Monthly Minor': _tstr(m.get('minor', 0)),
                'Monthly Mid': _tstr(m.get('mid', 0)),
                'Monthly Major': _tstr(m.get('major', 0)),
                'Blast Score': r.get('blast_score', 0),
                'Blast Tier': r.get('blast_tier', '')
            })
        st.download_button(
            label="📥 Download Matrix for Google Sheets (CSV)",
            data=pd.DataFrame(csv_rows).to_csv(index=False).encode('utf-8-sig'),
            file_name=f"price_volume_matrix_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Plain CSV compatible with Google Sheets import",
            key="pva_matrix_export_csv_btn"
        )


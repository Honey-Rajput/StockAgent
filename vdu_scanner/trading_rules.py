"""
Trading Rules & Strategy Module
Implements:
1. The 10 Most Important Trading Lessons (Position Sizing, Pyramiding, Trailing SL, Scale-out, Chasing Guard)
2. Strategy A: Entry Timing – The 2–3 Candle Strategy (Tight range + dry volume confirmation above broken resistance)
"""

from typing import Any
import pandas as pd


def detect_2_3_candle_pause(df: pd.DataFrame | None, pivot_price: float | None = None) -> dict[str, Any]:
    """
    Analyzes recent candles to detect Strategy A:
    The 2–3 Candle Strategy (tight price consolidation with dry volume right above support/pivot).
    """
    if df is None or len(df) < 5:
        return {
            "is_2_3_pause": False,
            "avg_range_pct": 0.0,
            "vol_ratio_3d": 1.0,
            "summary": "Insufficient candle history.",
        }

    try:
        # Standardize and deduplicate columns
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            elif 'Close' in df.columns.get_level_values(1):
                df.columns = df.columns.get_level_values(1)
            else:
                df.columns = df.columns.get_level_values(-1)
        df = df.loc[:, ~df.columns.duplicated()]

        cols = {str(c).lower(): c for c in df.columns}
        c_high = cols.get("high", "High")
        c_low = cols.get("low", "Low")
        c_close = cols.get("close", "Close")
        c_vol = cols.get("volume", "Volume")

        s_close = pd.Series(df[c_close].squeeze(), dtype=float)
        s_high = pd.Series(df[c_high].squeeze(), dtype=float)
        s_low = pd.Series(df[c_low].squeeze(), dtype=float)
        s_vol = pd.Series(df[c_vol].squeeze(), dtype=float)

        tail_3_close = s_close.iloc[-3:].values
        tail_3_high = s_high.iloc[-3:].values
        tail_3_low = s_low.iloc[-3:].values
        tail_3_vol = s_vol.iloc[-3:].values

        # Daily ranges as % of close
        ranges = [(h - l) / c for h, l, c in zip(tail_3_high, tail_3_low, tail_3_close) if c > 0]
        avg_range_pct = (sum(ranges) / len(ranges)) * 100 if ranges else 0.0

        # Volume comparison vs 20-period baseline
        vol_20_series = s_vol.rolling(20).mean().dropna()
        vol_20 = float(vol_20_series.iloc[-1]) if len(vol_20_series) > 0 else float(s_vol.mean())
        avg_vol_3d = float(sum(tail_3_vol) / len(tail_3_vol)) if len(tail_3_vol) > 0 else 0.0
        vol_ratio_3d = (avg_vol_3d / vol_20) if vol_20 > 0 else 1.0

        # Tightness check: average daily range <= 3.2%
        is_tight = avg_range_pct <= 3.2

        # Volume dryness check: volume <= 1.15x of 20 SMA (contracting supply)
        is_vol_dry = vol_ratio_3d <= 1.15

        # Price holding check: last close >= support/pivot if provided
        cmp = float(tail_3_close[-1]) if len(tail_3_close) > 0 else 0.0
        holds_pivot = True
        pivot_val = float(pivot_price) if (pivot_price is not None and not pd.isna(pivot_price)) else 0.0
        if pivot_val > 0:
            holds_pivot = cmp >= (pivot_val * 0.985)  # within 1.5% or above pivot

        is_pause = bool(is_tight and is_vol_dry and holds_pivot)

        if is_pause:
            summary = f"Confirmed 2–3 candle pause: tight range ({avg_range_pct:.1f}%) with dry volume ({vol_ratio_3d:.2f}x of 20d avg)."
        elif is_tight and not is_vol_dry:
            summary = f"Tight price action ({avg_range_pct:.1f}%), but volume slightly elevated ({vol_ratio_3d:.2f}x)."
        elif not is_tight and is_vol_dry:
            summary = f"Volume drying up ({vol_ratio_3d:.2f}x), but daily range wide ({avg_range_pct:.1f}%)."
        else:
            summary = f"Normal swing action: range {avg_range_pct:.1f}%, volume {vol_ratio_3d:.2f}x."

        return {
            "is_2_3_pause": is_pause,
            "avg_range_pct": round(avg_range_pct, 1),
            "vol_ratio_3d": round(vol_ratio_3d, 2),
            "summary": summary,
        }
    except Exception as e:
        return {
            "is_2_3_pause": False,
            "avg_range_pct": 0.0,
            "vol_ratio_3d": 1.0,
            "summary": f"Could not compute candle metrics: {e}",
        }


def evaluate_entry_rules(r: dict[str, Any], df: pd.DataFrame | None = None) -> dict[str, Any]:
    """
    Evaluates the 10 trading rules & 2-3 candle strategy for any scanned stock.
    Returns badges, guidance, risk levels, and pyramiding metrics.
    """
    try:
        cmp = float(r.get("cmp") or 0.0)
        buy = float(r.get("buy_price") or cmp or 0.0)
        sl = float(r.get("exit_price") or 0.0)
        target = float(r.get("target_price") or 0.0)
        day_chg = float(r.get("day_change_pct") or r.get("pct_change_today") or 0.0)

        # Distance from low-risk buy pivot
        if buy > 0 and cmp > 0:
            dist_pct = ((cmp - buy) / buy) * 100.0
        else:
            dist_pct = 0.0

        # Candle pause check
        stock_df = df if df is not None else r.get("df")
        pause_data = detect_2_3_candle_pause(stock_df, pivot_price=buy)
        is_2_3_pause = pause_data.get("is_2_3_pause", False)

        # Status classification based on rules
        if dist_pct > 6.0:
            status_code = "extended"
            status_label = "🔴 Extended (>6%)"
            status_desc = "Rule #3: Avoid late entries. Stock has moved substantially past pivot; do not chase. Wait for a 2–3 candle pullback."
            badge_color = "#ef4444"
            badge_bg = "rgba(239, 68, 68, 0.12)"
            badge_border = "rgba(239, 68, 68, 0.4)"
        elif is_2_3_pause or (0.0 <= dist_pct <= 3.5 and "breakout" in str(r.get("setup_type", "")).lower() and abs(day_chg) <= 2.5):
            status_code = "2_3_pause"
            status_label = "🟢 2–3 Candle Pause"
            status_desc = "Strategy A: Prime entry setup. 2–3 candles of tight consolidation with dry volume right above pivot support."
            badge_color = "#00e676"
            badge_bg = "rgba(0, 230, 118, 0.15)"
            badge_border = "rgba(0, 230, 118, 0.5)"
        elif dist_pct < -0.5:
            status_code = "pre_breakout"
            status_label = "⏳ Pre-Breakout"
            status_desc = "Consolidating below resistance. Accumulate near support or wait for volume surge confirmation."
            badge_color = "#ffa000"
            badge_bg = "rgba(255, 160, 0, 0.12)"
            badge_border = "rgba(255, 160, 0, 0.4)"
        elif 0.0 <= dist_pct <= 2.5 and day_chg >= 2.0:
            status_code = "day1_breakout"
            status_label = "⚡ Day 1 Breakout"
            status_desc = "Rule #2: Observe previous 2–3 candles to confirm strength before taking full position."
            badge_color = "#38bdf8"
            badge_bg = "rgba(56, 189, 248, 0.12)"
            badge_border = "rgba(56, 189, 248, 0.4)"
        else:
            status_code = "actionable"
            status_label = "✅ Low-Risk Buy"
            status_desc = "Within optimal low-risk buy zone with defined risk parameters."
            badge_color = "#22d3ee"
            badge_bg = "rgba(34, 211, 238, 0.12)"
            badge_border = "rgba(34, 211, 238, 0.4)"

        # Money Management Rules (Rules 1, 5, 6, 7, 10)
        initial_sl = sl if sl > 0 else round(cmp * 0.95, 2)
        risk_per_share = max(0.01, round(cmp - initial_sl, 2))
        risk_pct = round((risk_per_share / cmp * 100), 1) if cmp > 0 else 5.0

        # Rule 5: Trailing SL: If price advances > 6% from entry, trail SL to breakeven (buy price)
        if cmp >= (buy * 1.06):
            trailing_sl = round(buy, 2)
            trailing_note = f"Trailed to breakeven (₹{buy:,.2f}) after +6% gain"
        elif cmp >= (buy * 1.03):
            trailing_sl = round(initial_sl + (risk_per_share * 0.5), 2)
            trailing_note = f"Trailed upward to ₹{trailing_sl:,.2f} (+0.5R cushion)"
        else:
            trailing_sl = initial_sl
            trailing_note = f"Initial protective SL at ₹{initial_sl:,.2f} (-{risk_pct}%)"

        # Rule 7: Pyramiding Add Level (when price clears current pause high by +2.5%)
        pyramid_trigger = round(max(cmp, buy) * 1.025, 2)

        # Rule 6: Scale out gradually (T1 at 2R or +8%, T2 at +15% / Target)
        target_1 = round(cmp + (risk_per_share * 2), 2) if risk_per_share > 0 else round(cmp * 1.08, 2)
        target_2 = target if target > cmp else round(cmp * 1.15, 2)

        badge_html = (
            f'<span class="custom-badge" style="background:{badge_bg}; color:{badge_color}; '
            f'border:1px solid {badge_border}; font-size:0.75rem; font-weight:700; padding:2px 7px; '
            f'border-radius:4px; display:inline-block; white-space:nowrap;" '
            f'title="{status_desc}">{status_label}</span>'
        )

        return {
            "status_code": status_code,
            "status_label": status_label,
            "status_desc": status_desc,
            "badge_color": badge_color,
            "badge_html": badge_html,
            "dist_from_pivot_pct": round(dist_pct, 1),
            "is_extended": dist_pct > 6.0,
            "is_2_3_pause": is_2_3_pause,
            "pause_metrics": pause_data,
            # Execution Plan
            "pilot_size_pct": "33% – 50%",
            "pyramid_trigger": pyramid_trigger,
            "initial_sl": initial_sl,
            "trailing_sl": trailing_sl,
            "trailing_note": trailing_note,
            "target_1_partial": target_1,
            "target_2_runner": target_2,
            "risk_per_share": risk_per_share,
            "risk_pct": risk_pct,
        }
    except Exception as e:
        return {
            "status_code": "unknown",
            "status_label": "⚪ Normal",
            "status_desc": f"Rule check error: {e}",
            "badge_color": "#94a3b8",
            "badge_html": '<span class="custom-badge" style="color:#94a3b8;">⚪ Normal</span>',
            "dist_from_pivot_pct": 0.0,
            "is_extended": False,
            "is_2_3_pause": False,
            "pause_metrics": {},
            "pilot_size_pct": "33% – 50%",
            "pyramid_trigger": 0.0,
            "initial_sl": 0.0,
            "trailing_sl": 0.0,
            "trailing_note": "Normal SL",
            "target_1_partial": 0.0,
            "target_2_runner": 0.0,
            "risk_per_share": 0.0,
            "risk_pct": 0.0,
        }


def render_10_rules_execution_card(detail_data: dict[str, Any], df: pd.DataFrame | None = None) -> None:
    """
    Renders an interactive 10 Rules & Strategy A Execution Blueprint in Streamlit (e.g. for tab_detail.py).
    """
    import streamlit as st

    sym = detail_data.get("symbol", "STOCK")
    cmp = float(detail_data.get("cmp") or 0.0)
    rule_eval = evaluate_entry_rules(detail_data, df=df)
    pause_info = rule_eval.get("pause_metrics", {})

    st.markdown("---")
    st.markdown("### 🎯 10 Rules Execution Blueprint & Strategy A (2–3 Candle Strategy)")
    st.markdown(
        "<p style='font-size:0.85rem; color:#94a3b8; margin-top:-6px;'>"
        "Actionable entry, money management, and risk controls derived from the <b>10 Core Principles</b> and <b>Strategy A</b>."
        "</p>",
        unsafe_allow_html=True
    )

    # 4-Column Quick Overview
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:10px 12px;">'
            f'<span style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase;">Entry Timing Status</span><br>'
            f'<div style="margin-top:4px;">{rule_eval.get("badge_html", "")}</div>'
            f'<span style="font-size:0.75rem; color:#cbd5e1; display:block; margin-top:4px;">{rule_eval.get("dist_from_pivot_pct", 0.0):+.1f}% from pivot</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    with rc2:
        st.markdown(
            f'<div style="background:rgba(56,189,248,0.05); border:1px solid rgba(56,189,248,0.2); border-radius:8px; padding:10px 12px;">'
            f'<span style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase;">Rule #1: Pilot Size</span><br>'
            f'<span style="font-size:1.15rem; font-weight:700; color:#38bdf8;">{rule_eval.get("pilot_size_pct")}</span><br>'
            f'<span style="font-size:0.72rem; color:#94a3b8;">Start small, add on strength</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    with rc3:
        st.markdown(
            f'<div style="background:rgba(0,230,118,0.05); border:1px solid rgba(0,230,118,0.2); border-radius:8px; padding:10px 12px;">'
            f'<span style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase;">Rule #7: Pyramid Add</span><br>'
            f'<span style="font-size:1.15rem; font-weight:700; color:#00e676;">₹{rule_eval.get("pyramid_trigger"):,.2f}</span><br>'
            f'<span style="font-size:0.72rem; color:#94a3b8;">+2.5% strength confirmation</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    with rc4:
        st.markdown(
            f'<div style="background:rgba(239,68,68,0.05); border:1px solid rgba(239,68,68,0.2); border-radius:8px; padding:10px 12px;">'
            f'<span style="font-size:0.72rem; color:#94a3b8; text-transform:uppercase;">Rule #5: Trailing SL</span><br>'
            f'<span style="font-size:1.15rem; font-weight:700; color:#ef4444;">₹{rule_eval.get("trailing_sl"):,.2f}</span><br>'
            f'<span style="font-size:0.72rem; color:#94a3b8;">{rule_eval.get("trailing_note")}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Strategy A & 2-3 Candle Diagnostic Card
    is_pause = rule_eval.get("is_2_3_pause", False)
    stat_box_border = "#00e676" if is_pause else ("#ef4444" if rule_eval.get("is_extended") else "#38bdf8")
    st.markdown(
        f'<div style="margin-top:12px; background:rgba(15,23,42,0.6); border-left:4px solid {stat_box_border}; '
        f'border-radius:6px; padding:12px 16px; font-size:0.86rem; color:#cbd5e1;">'
        f'<b>Strategy A Checklist (The 2–3 Candle Rule):</b> {pause_info.get("summary", rule_eval.get("status_desc", ""))}<br>'
        f'<span style="font-size:0.78rem; color:#94a3b8; margin-top:4px; display:inline-block;">'
        f'• 3-Day Avg Range: <b>{pause_info.get("avg_range_pct", 0.0)}%</b> (Threshold: &le;3.2% for tight base) | '
        f'• 3-Day Volume: <b>{pause_info.get("vol_ratio_3d", 1.0)}x</b> of 20-day average'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Interactive Position Sizer & Scale-Out Calculator (Rules 1, 6, 10)
    with st.expander(f"🧮 Rule #10: Interactive Position Sizer & Multi-Tier Scale-Out (for {sym})", expanded=True):
        col_risk, col_t1, col_t2 = st.columns([2.5, 2.5, 3])
        with col_risk:
            risk_input = st.number_input(
                "Max Loss Capital to Risk (₹)",
                min_value=500,
                max_value=200000,
                value=5000,
                step=500,
                key=f"{sym}_rules_max_risk_input"
            )

        risk_per_sh = rule_eval.get("risk_per_share", 1.0)
        tot_shares = int(risk_input / risk_per_sh) if risk_per_sh > 0 else 0
        pilot_shares = int(tot_shares * 0.4)
        pyramid_shares = tot_shares - pilot_shares
        tot_invest = tot_shares * cmp
        pilot_invest = pilot_shares * cmp

        with col_t1:
            st.markdown(
                f'<div style="font-size:0.8rem; color:#94a3b8; line-height:1.5;">'
                f'<b>Total Shares:</b> <span style="color:#f8fafc; font-size:0.95rem;">{tot_shares:,}</span> (₹{tot_invest:,.0f})<br>'
                f'<b>• Tranche 1 (Pilot 40%):</b> <span style="color:#38bdf8;">{pilot_shares:,} shares</span> (₹{pilot_invest:,.0f})<br>'
                f'<b>• Tranche 2 (Pyramid 60%):</b> <span style="color:#00e676;">{pyramid_shares:,} shares</span> at ₹{rule_eval.get("pyramid_trigger"):,.2f}'
                f'</div>',
                unsafe_allow_html=True
            )

        with col_t2:
            t1 = rule_eval.get("target_1_partial", 0.0)
            t2 = rule_eval.get("target_2_runner", 0.0)
            st.markdown(
                f'<div style="font-size:0.8rem; color:#94a3b8; line-height:1.5;">'
                f'<b>Rule #6 Multi-Tier Scale Out:</b><br>'
                f'• <b>Target 1 (Book 50%):</b> <span style="color:#00e676; font-weight:700;">₹{t1:,.2f}</span> (+2R / +{round(((t1-cmp)/cmp*100),1) if cmp>0 else 0}%)<br>'
                f'• <b>Target 2 (Book 25%):</b> <span style="color:#22d3ee; font-weight:700;">₹{t2:,.2f}</span><br>'
                f'• <b>Runner (Hold 25%):</b> Trail with 20 SMA / 9 EMA until trend breakdown'
                f'</div>',
                unsafe_allow_html=True
            )


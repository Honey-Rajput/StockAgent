import streamlit as st
import pandas as pd

from datetime import datetime, timedelta

from config import IST_TIMEZONE

import database


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
    st.markdown("### 🎯 Minervini Ultimate +VCP")
    st.markdown("Scans for stocks passing the Minervini Trend Template with Volatility Contraction Pattern (VCP) squeeze.")

    # Auto-load cached results from DB on first page visit or when results are empty
    if not st.session_state.get('vcp_minervini_results'):
        try:
            import database
            vcp_d = database.get_all_latest_scan_dates().get("scanned_vcp_minervini") or database.get_best_scan_date()
            if vcp_d:
                cached = database.get_cached_vcp_minervini(vcp_d)
                if cached:
                    st.session_state.vcp_minervini_results = cached
            if not st.session_state.get('vcp_minervini_results'):
                for check_date in database.get_available_scan_dates():
                    cached = database.get_cached_vcp_minervini(check_date)
                    if cached:
                        st.session_state.vcp_minervini_results = cached
                        break
        except Exception:
            pass
    col_v1, col_v2, col_v3 = st.columns(3)
    vcp_thresh = col_v1.number_input("Max VCP Range (%)", value=2.5, step=0.5, help="Maximum percentage range over lookback to be considered a squeeze")
    vcp_lookback = col_v2.number_input("VCP Lookback (Bars)", value=5, step=1)
    risk_low = col_v3.number_input("Max Low Risk (%)", value=15.0, step=1.0, help="Maximum distance above 50SMA to be considered Low Risk")

    st.markdown("---")
    
    col_v_btn, col_v_note = st.columns([1, 2])
    run_vcp_btn = col_v_btn.button("🔍 Run VCP+Minervini Scan", type="primary", use_container_width=True, key="run_vcp_scan_btn")
    
    if run_vcp_btn:
        with st.spinner("Running Minervini VCP scan across Top 1500 NSE stocks..."):
            from vcp_minervini import VCPConfig, MinerviniVCPAnalyzer
            from data_fetcher import get_top1500_nse_symbols, get_stock_sector
            from local_cache_manager import bulk_get_cached_ohlcv
            import yfinance as yf

            raw_symbols = get_top1500_nse_symbols()
            clean_syms = [s.strip().upper().replace('.NS', '') for s in raw_symbols if s.strip()]

            cfg = VCPConfig(
                vcp_thresh=vcp_thresh,
                vcp_lookback=int(vcp_lookback),
                risk_low=risk_low
            )

            try:
                benchmark_df = yf.download("^NSEI", period="2y", interval="1d", progress=False, threads=False, timeout=15)
            except Exception:
                benchmark_df = None

            progress_bar = st.progress(0, text="Loading local OHLCV cache...")
            bulk_cached = bulk_get_cached_ohlcv(clean_syms, "1d")

            vcp_results = []
            total = len(clean_syms)
            for idx, clean_s in enumerate(clean_syms):
                if idx % 100 == 0 or idx == total - 1:
                    progress_bar.progress((idx + 1) / max(1, total), text=f"Analyzing {idx+1}/{total} ({len(vcp_results)} VCP setups)...")
                try:
                    t_df = bulk_cached.get(clean_s)
                    if t_df is None or len(t_df) < 250:
                        continue
                    t_df = t_df.copy()
                    if 'Date' in t_df.columns:
                        t_df['Date'] = pd.to_datetime(t_df['Date'])
                        t_df = t_df.set_index('Date').sort_index()

                    sym = f"{clean_s}.NS"
                    analyzer = MinerviniVCPAnalyzer(sym, cfg, benchmark_df=benchmark_df)
                    analyzer.df = t_df
                    analyzer._moving_averages()
                    analyzer._trend_template()
                    analyzer._buy_risk()
                    analyzer._pressure()
                    analyzer._rpr()
                    analyzer._vcp()
                    analyzer._entry_signals()

                    last = analyzer.df.iloc[-1]
                    tpr_txt = str(last["tpr_txt"])
                    entry_sig = str(last["entry_signal"])
                    has_squeeze = any(
                        str(last.get(k, "")) == "SQUEEZE"
                        for k in ("vcp_txt", "vcp10_txt", "vcp15_txt")
                    )
                    # Only keep real setups: PASSED Minervini Trend Template OR active Squeeze / Entry Signal
                    if ("PASSED" not in tpr_txt) and (not has_squeeze) and (entry_sig not in ("BREAKOUT", "EARLY ENTRY")):
                        continue

                    result = {
                        "symbol": sym,
                        "date": analyzer.df.index[-1].strftime("%Y-%m-%d"),
                        "close": round(float(last["Close"]), 2),
                        "Pressure": last["pressure_txt"],
                        "Risk (50d)": last["risk_status"],
                        "Trend (TPR)": tpr_txt,
                        "RS Proxy": round(float(last["rpr_proxy"]), 1) if pd.notna(last["rpr_proxy"]) else None,
                        "VCP (5d)": last["vcp_txt"],
                        "VCP range %": round(float(last["vcp_range_pct"]), 2) if pd.notna(last["vcp_range_pct"]) else None,
                        "VCP (10d)": last.get("vcp10_txt", "Normal"),
                        "VCP 10d range %": round(float(last["vcp10_range_pct"]), 2) if "vcp10_range_pct" in last and pd.notna(last["vcp10_range_pct"]) else None,
                        "VCP (15d)": last.get("vcp15_txt", "Normal"),
                        "VCP 15d range %": round(float(last["vcp15_range_pct"]), 2) if "vcp15_range_pct" in last and pd.notna(last["vcp15_range_pct"]) else None,
                        "Entry Signal": entry_sig,
                        "Sector": get_stock_sector(sym),
                    }
                    vcp_results.append(result)
                except Exception:
                    pass

            progress_bar.empty()
            
            if vcp_results:
                # Add score and rank
                vcp_df = pd.DataFrame(vcp_results)
                rs_proxy = pd.to_numeric(vcp_df.get('RS Proxy', 50), errors='coerce').fillna(50)
                vcp_range = pd.to_numeric(vcp_df.get('VCP range %', 100), errors='coerce').fillna(100)
                vcp_df['Score'] = rs_proxy - (vcp_range * 5)
                vcp_df = vcp_df.sort_values(by='Score', ascending=False)
                vcp_df.insert(0, 'Rank', range(1, len(vcp_df) + 1))
                st.session_state.vcp_minervini_results = vcp_df.to_dict('records')
                
                # Save to database
                try:
                    today_str = get_market_date()
                    database.save_vcp_minervini_scan(today_str, st.session_state.vcp_minervini_results)
                except Exception as e:
                    print(f"Error saving VCP+Minervini scan: {e}")
            else:
                st.session_state.vcp_minervini_results = []
            
            st.rerun()

    # Display results
    if st.session_state.get('vcp_minervini_results'):
        v_df = pd.DataFrame(st.session_state.vcp_minervini_results)
        
        vcp_count = len(v_df)
        st.success(f"Found {vcp_count} VCP+Minervini setups!")
        
        # CSV Download
        col_btn, _ = st.columns([2, 8])
        with col_btn:
            vcp_csv = v_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="⬇️ Download CSV",
                data=vcp_csv,
                file_name="vcp_minervini_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        col1, col2 = st.columns(2)
        with col1:
            show_buyable = st.checkbox("Show Buyable Only (Buying Pressure, Low Risk, PASSED Trend)", value=False)
        with col2:
            show_squeeze = st.checkbox("Show 'Squeeze' / Entry Signals Only", value=False)

        # Reorder columns for display
        display_cols = ['Rank', 'Score', 'symbol', 'Sector', 'close', 'Entry Signal', 'Trend (TPR)', 
                       'Pressure', 'Risk (50d)', 'RS Proxy', 'VCP (5d)', 'VCP range %', 'VCP (10d)', 'VCP 10d range %', 'VCP (15d)', 'VCP 15d range %', 'date']
        available_cols = [c for c in display_cols if c in v_df.columns]
        v_df = v_df[available_cols]
        
        if show_buyable:
            if 'Pressure' in v_df.columns:
                v_df = v_df[v_df['Pressure'].str.contains('Buying', case=False, na=False)]
            if 'Risk (50d)' in v_df.columns:
                v_df = v_df[v_df['Risk (50d)'].str.contains('Low Risk', case=False, na=False)]
            if 'Trend (TPR)' in v_df.columns:
                v_df = v_df[v_df['Trend (TPR)'].str.contains('PASSED', case=False, na=False)]
                
        if show_squeeze:
            if 'Entry Signal' in v_df.columns:
                mask = v_df['Entry Signal'].isin(["BREAKOUT", "EARLY ENTRY"])
                if 'VCP (5d)' in v_df.columns: mask = mask | (v_df['VCP (5d)'] == 'SQUEEZE')
                if 'VCP (10d)' in v_df.columns: mask = mask | (v_df['VCP (10d)'] == 'SQUEEZE')
                if 'VCP (15d)' in v_df.columns: mask = mask | (v_df['VCP (15d)'] == 'SQUEEZE')
                v_df = v_df[mask]
        
        _max_cmp_vcp = float(st.session_state.get("global_max_cmp", 0))
        if _max_cmp_vcp > 0 and 'close' in v_df.columns:
            v_df = v_df[v_df['close'].astype(float) <= _max_cmp_vcp]

        if 'symbol' in v_df.columns:
            v_df['symbol'] = v_df['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
            
        st.dataframe(v_df, use_container_width=True, column_config={
            "symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)"),
            "date": st.column_config.TextColumn("Date"),
            "close": st.column_config.NumberColumn("CMP", format="%.2f"),
            "Score": st.column_config.NumberColumn("Score", format="%.1f"),
            "RS Proxy": st.column_config.NumberColumn("RS Proxy (vs Nifty)", format="%.1f"),
            "VCP range %": st.column_config.NumberColumn("VCP Range 5d %", format="%.2f"),
            "VCP 10d range %": st.column_config.NumberColumn("VCP Range 10d %", format="%.2f"),
            "VCP 15d range %": st.column_config.NumberColumn("VCP Range 15d %", format="%.2f"),
            "Entry Signal": st.column_config.TextColumn("Entry Signal"),
            "Trend (TPR)": st.column_config.TextColumn("Trend Template"),
        })
    else:
        st.info("💡 Click 'Run VCP+Minervini Scan' to find setups.")

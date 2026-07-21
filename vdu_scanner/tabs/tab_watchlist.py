import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import watchlist
from utils import get_signal_badge_html
from ui_components import render_quick_trade_board, render_trading_setup_card
from data_fetcher import fetch_ohlcv
from scanner import compute_rich_analysis
from config import IST_TIMEZONE

def render():
    st.markdown("### 📋 My Watchlist Monitor")
    
    # Read persistent DB
    w_df = watchlist.load_watchlist()
    
    if w_df.empty:
        st.info("ℹ️ Your watchlist is currently empty. Run scans on index universes or paste custom tickers to build your watchlist!")
    else:
        # A. SINGLE BATCH YFINANCE PRICE DOWNLOAD
        tickers_list = [f"{s}.NS" for s in w_df['symbol'].unique()]
        cmp_dict = {}
        
        with st.spinner("Fetching real-time quotes for watchlisted assets..."):
            try:
                # yfinance 1.x: auto_adjust=True is default, auto_adjust=False is deprecated
                prices_df = yf.download(tickers=tickers_list, period="1d", progress=False, threads=False, timeout=15)
                if not prices_df.empty:
                    # yfinance 1.x multi-ticker: MultiIndex (price_type, ticker)
                    if isinstance(prices_df.columns, pd.MultiIndex):
                        close_prices = prices_df['Close'].iloc[-1]  # Series with .NS ticker index
                    else:
                        close_prices = prices_df['Close'].iloc[-1]  # scalar for single ticker
                        close_prices = {tickers_list[0]: close_prices}

                    # Build lookup maps (strip .NS from keys)
                    if isinstance(close_prices, pd.Series):
                        for k, v in close_prices.items():
                            clean_k = str(k).replace(".NS", "").upper()
                            if not pd.isna(v) and float(v) > 0:
                                cmp_dict[clean_k] = float(v)
                    elif isinstance(close_prices, dict):
                        for k, v in close_prices.items():
                            clean_k = str(k).replace(".NS", "").upper()
                            if v and not pd.isna(v):
                                cmp_dict[clean_k] = float(v)
            except Exception as quote_ex:
                st.warning("⚠️ Could not fetch real-time quotes. Using historical entry price for watchlist CMP.")

        # B. BUILD WATCHLIST VIEW DATA
        display_rows = []
        for idx, row in w_df.iterrows():
            sym = row['symbol'].upper()
            entry = float(row['entry_price'])

            # Fetch CMP or fall back to entry
            cmp_val = cmp_dict.get(sym, entry)
            if pd.isna(cmp_val) or cmp_val <= 0:
                cmp_val = entry

            pnl_val = ((cmp_val - entry) / entry * 100)

            display_rows.append({
                "symbol": sym,
                "company_name": row['company_name'],
                "added_date": row['added_date'],
                "entry_price": entry,
                "signal_strength_at_add": float(row['signal_strength_at_add']),
                "CMP (₹)": round(cmp_val, 2),
                "PnL %": round(pnl_val, 2),
                "tag": row['tag'],
                "notes": str(row['notes']) if not pd.isna(row['notes']) else ""
            })

        display_df = pd.DataFrame(display_rows)

        # C. INTERACTIVE DATA EDITOR (Auto-saves Tag and Notes)
        st.markdown("<p style='font-size:0.85rem; color:#94a3b8;'>✏️ You can edit the <b>Tag</b> dropdowns or write custom text in <b>Notes</b> cells. Changes persist immediately.</p>", unsafe_allow_html=True)

        # Define table configs
        config_table = {
            "symbol": st.column_config.TextColumn("Symbol", disabled=True),
            "added_date": st.column_config.TextColumn("Added Date", disabled=True),
            "entry_price": st.column_config.NumberColumn("Entry Price (₹)", disabled=True, format="₹%.2f"),
            "signal_strength_at_add": st.column_config.NumberColumn("Original Signal", disabled=True, format="%.1f pts"),
            "CMP (₹)": st.column_config.NumberColumn("Current Price (₹)", disabled=True, format="₹%.2f"),
            "PnL %": st.column_config.NumberColumn("Unrealized PnL %", disabled=True, format="%.2f%%"),
            "tag": st.column_config.SelectboxColumn("Tag Status", options=["Watching 👀", "Ready to Buy 🟢", "Tracking 📍", "Avoid 🔴"]),
            "notes": st.column_config.TextColumn("Notes (Click to Edit)")
        }

        edited_table = st.data_editor(
            display_df,
            column_config=config_table,
            width="stretch",
            hide_index=True,
            key="watchlist_editor_grid"
        )

        # Check cell changes
        if not edited_table.equals(display_df):
            # Map back to standard CSV columns
            save_df = edited_table[['symbol', 'company_name', 'added_date', 'entry_price', 'signal_strength_at_add', 'tag', 'notes']].copy()
            watchlist.save_watchlist(save_df)
            st.toast("💾 Watchlist auto-saved successfully!")
            st.rerun()

        st.markdown("---")

        # D. MANAGEMENT CONTROLS PANEL
        st.markdown("### ⚙️ Watchlist Controls")

        col_c1, col_c2 = st.columns(2)

        # 1. Removal widget
        with col_c1:
            st.markdown("#### ❌ Delete Ticker")
            c_del1, c_del2 = st.columns([2, 1])
            ticker_to_delete = c_del1.selectbox(
                "Choose stock to remove:",
                options=[""] + list(display_df['symbol'].unique()),
                key="del_box"
            )

            if ticker_to_delete:
                del_clicked = c_del2.button("Remove Ticker", type="secondary", key="del_action", width="stretch")
                if del_clicked:
                    watchlist.remove_stock(ticker_to_delete)
                    st.toast(f"Removed {ticker_to_delete} from your watchlist.")
                    st.rerun()

        # 2. Export and Clear watchlist
        with col_c2:
            st.markdown("#### 📂 Operations")

            # Export CSV
            watchlist_csv_bytes = watchlist.export_csv()
            st.download_button(
                label="📥 Export Watchlist CSV",
                data=watchlist_csv_bytes,
                file_name=f"vdu_watchlist_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d')}.csv",
                mime="text/csv",
                width="stretch",
                key="dl_watchlist"
            )

            # Clear all database
            clear_btn = st.button("🗑️ Clear Entire Watchlist", type="secondary", width="stretch", key="clear_watchlist_btn")
            if clear_btn:
                st.session_state.confirm_clear = True

            if st.session_state.confirm_clear:
                st.markdown("<p style='color:#ef4444; font-weight:600;'>⚠️ Are you absolutely sure? This deletes watchlist.csv entries forever.</p>", unsafe_allow_html=True)
                col_yes, col_no = st.columns(2)

                if col_yes.button("Yes, Clear All", type="primary", width="stretch", key="clr_yes"):
                    # Clear CSV
                    empty_df = pd.DataFrame(columns=watchlist.COLUMNS)
                    watchlist.save_watchlist(empty_df)
                    st.session_state.confirm_clear = False
                    st.toast("🗑️ Watchlist fully cleared.")
                    st.rerun()

                if col_no.button("Cancel", width="stretch", key="clr_no"):
                    st.session_state.confirm_clear = False
                    st.rerun()

        # Watchlist Technical Assessment inspector panel
        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)
        st.markdown("### 🎯 Watchlist Technical Assessment")
        st.markdown("<p style='font-size:0.9rem; color:#94a3b8; margin-top:-10px;'>Select any stock from your watchlist to inspect its real-time indicators and buying checklist.</p>", unsafe_allow_html=True)

        watch_symbols = list(display_df['symbol'].unique())
        selected_watch_sym = st.selectbox(
            "Select Stock to Inspect:",
            options=[""] + watch_symbols,
            key="watch_inspect_select"
        )

        if selected_watch_sym:
            # Fetch historical data and compute rich indicators
            with st.spinner(f"Loading technical indicators for {selected_watch_sym}..."):
                df_w = fetch_ohlcv(selected_watch_sym)
                if df_w is not None and not df_w.empty:
                    rich_payload = compute_rich_analysis(df_w, selected_watch_sym, "Watchlist Assessment", "Monitor key support levels for active trade setups.")
                    watch_item = next((r for r in display_rows if r['symbol'] == selected_watch_sym), None)
                    cmp_val = watch_item['CMP (₹)'] if watch_item else df_w['Close'].iloc[-1]

                    dummy_w = {
                        "symbol": selected_watch_sym,
                        "cmp": cmp_val,
                        "buy_price": watch_item['entry_price'] if watch_item else cmp_val,
                        "exit_price": cmp_val * 0.93,
                        "target_price": cmp_val * 1.15,
                        "confidence": "Medium-High",
                        "recommendation": rich_payload
                    }
                    render_trading_setup_card(dummy_w, "watchlist_tab_setup", 0)

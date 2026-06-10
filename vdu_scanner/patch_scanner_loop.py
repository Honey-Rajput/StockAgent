import re
import os

with open("app.py", "r", encoding="utf-8") as f:
    code = f.read()

process_func = """
        def process_single_symbol(sym, df, open_price_map, close_price_map, high_price_map, low_price_map, volume_map,
                                  min_dry, max_dry, min_vol_ratio, min_price_chg, min_dry_spikes,
                                  min_signal_str, above_50dma_only, above_200dma_only, vcp_max_tightness):
            from datetime import datetime
            import pandas as pd
            import pytz
            IST_TIMEZONE = pytz.timezone('Asia/Kolkata')
            
            res = {
                "failed": False,
                "gapup": None,
                "above_ma": None,
                "support_ma": None,
                "crossover_ma": None,
                "minervini": None,
                "flagged": None,
                "coiled": None,
                "wt": None,
                "vcs": None,
                "structural_vcp": None,
                "vpa": None
            }
            if df is None or len(df) < 5:
                res["failed"] = True
                return res
                
            df = df.sort_values('Date').reset_index(drop=True)
            last_df_date = df['Date'].iloc[-1].date()
            today_date = datetime.now(IST_TIMEZONE).date()
            
            if last_df_date < today_date:
                sym_clean = sym.strip().upper()
                if sym_clean in open_price_map and sym_clean in close_price_map:
                    new_row = {
                        'Date': pd.to_datetime(today_date),
                        'Open': open_price_map[sym_clean],
                        'High': high_price_map.get(sym_clean, close_price_map[sym_clean]),
                        'Low': low_price_map.get(sym_clean, close_price_map[sym_clean]),
                        'Close': close_price_map[sym_clean],
                        'Volume': volume_map.get(sym_clean, 0)
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
            today_close_val = df['Close'].iloc[-1]
            if today_close_val <= 200.0:
                res["failed"] = True
                return res
                
            today_open_val = float(df['Open'].iloc[-1])
            today_close_val = float(df['Close'].iloc[-1])
            yesterday_close_val = float(df['Close'].iloc[-2]) if len(df) >= 2 else today_open_val
            if today_open_val > yesterday_close_val and today_close_val > yesterday_close_val and today_close_val >= (today_open_val * 0.97):
                gap_pct = (today_open_val - yesterday_close_val) / yesterday_close_val * 100
                if gap_pct >= 8.0:
                    target_multiplier = 1.04; target_pct_str = "+4.0%"
                elif gap_pct >= 5.0:
                    target_multiplier = 1.06; target_pct_str = "+6.0%"
                else:
                    target_multiplier = 1.10; target_pct_str = "+10.0%"
                    
                gap_buy_price = round(today_close_val, 2)
                gap_exit_price = round(today_open_val * 0.98, 2) 
                gap_target_price = round(today_close_val * target_multiplier, 2) 
                gap_confidence = "High (Gap-Up Momentum)" if gap_pct > 3.0 else "Medium (Gap-Up)"
                base_gap_rec = (f"Bullish gap-up breakout of {gap_pct:.2f}% on strong momentum. Buy near ₹{gap_buy_price:.2f} "
                                f"with a stop loss below today's open price at ₹{gap_exit_price:.2f} "
                                f"targeting dynamic swing target ₹{gap_target_price:.2f} ({target_pct_str}).")
                gap_recommendation = compute_rich_analysis(df, sym, "Gap-Up", base_gap_rec)
                res["gapup"] = {
                    "symbol": sym.strip().upper(), "company_name": get_company_name(sym),
                    "prev_close": yesterday_close_val, "open_price": today_open_val, "cmp": today_close_val,
                    "gap_pct": round(gap_pct, 2), "volume": int(df['Volume'].iloc[-1]),
                    "day_change_pct": round(((today_close_val - yesterday_close_val) / yesterday_close_val * 100), 2),
                    "buy_price": gap_buy_price, "exit_price": gap_exit_price, "target_price": gap_target_price,
                    "confidence": gap_confidence, "recommendation": gap_recommendation
                }
                
            df_ma = df.copy()
            df_ma['SMA20'] = df_ma['Close'].rolling(window=20).mean()
            df_ma['SMA50'] = df_ma['Close'].rolling(window=50).mean()
            df_ma['SMA65'] = df_ma['Close'].rolling(window=65).mean()
            df_ma['SMA150'] = df_ma['Close'].rolling(window=150).mean()
            df_ma['SMA200'] = df_ma['Close'].rolling(window=200).mean()
            
            if len(df_ma) >= 200:
                today_row = df_ma.iloc[-1]; yesterday_row = df_ma.iloc[-2]
                c_val = float(today_row['Close']); l_val = float(today_row['Low'])
                sma20 = float(today_row['SMA20']); sma50 = float(today_row['SMA50'])
                sma65 = float(today_row['SMA65']); sma150 = float(today_row['SMA150'])
                sma200 = float(today_row['SMA200'])
                
                if c_val > sma20 and c_val > sma50:
                    above_buy_price = round(today_close_val, 2)
                    above_exit_price = round(sma50 * 0.97, 2) 
                    above_target_price = round(today_close_val * 1.12, 2) 
                    above_confidence = "High (Uptrend)" if sma20 > sma50 and sma50 > sma200 else "Medium-High (Uptrend)"
                    base_above_rec = (f"Strong medium-term uptrend. Close above 20 SMA & 50 SMA. Buy near ₹{above_buy_price:.2f} "
                                      f"with stop below 50 SMA support at ₹{above_exit_price:.2f} targeting momentum target ₹{above_target_price:.2f}.")
                    res["above_ma"] = {
                        "symbol": sym.strip().upper(), "company_name": get_company_name(sym), "cmp": today_close_val,
                        "day_change_pct": round(((today_close_val - yesterday_row['Close']) / yesterday_row['Close'] * 100), 2),
                        "dist_20sma_pct": round((today_close_val - sma20) / sma20 * 100, 2),
                        "dist_50sma_pct": round((today_close_val - sma50) / sma50 * 100, 2),
                        "setup_type": "above_ma", "buy_price": above_buy_price, "exit_price": above_exit_price,
                        "target_price": above_target_price, "confidence": above_confidence,
                        "recommendation": compute_rich_analysis(df_ma, sym, "Above 20/50 SMA", base_above_rec)
                    }
                    
                yesterday_l = float(yesterday_row['Low']); yesterday_sma65 = float(yesterday_row['SMA65'])
                tested_today = l_val <= sma65 * 1.01; tested_yesterday = yesterday_l <= yesterday_sma65 * 1.01
                o_val = float(today_row['Open']); yesterday_c = float(yesterday_row['Close'])
                is_green_candle = c_val > o_val; is_up_move = c_val > yesterday_c; holds_above = c_val > sma65
                
                if (tested_today or tested_yesterday) and holds_above and is_green_candle and is_up_move:
                    support_buy_price = round(today_close_val, 2); support_exit_price = round(sma65 * 0.97, 2) 
                    support_target_price = round(today_close_val * 1.15, 2) 
                    support_confidence = "High (Pullback Support)" if today_close_val > yesterday_row['Close'] else "Medium (Pullback Support)"
                    base_support_rec = (f"Institutional pullback testing critical 65 SMA support (₹{sma65:.2f}). "
                                        f"Buy around ₹{support_buy_price:.2f} with tight stop just below SMA at ₹{support_exit_price:.2f} targeting bounce to ₹{support_target_price:.2f}.")
                    res["support_ma"] = {
                        "symbol": sym.strip().upper(), "company_name": get_company_name(sym), "cmp": today_close_val,
                        "day_change_pct": round(((today_close_val - yesterday_row['Close']) / yesterday_row['Close'] * 100), 2),
                        "dist_65sma_pct": round((today_close_val - sma65) / sma65 * 100, 2), "setup_type": "support_ma",
                        "buy_price": support_buy_price, "exit_price": support_exit_price, "target_price": support_target_price,
                        "confidence": support_confidence, "recommendation": compute_rich_analysis(df_ma, sym, "65 SMA Support", base_support_rec)
                    }
                    
                crossed_golden = (yesterday_row['SMA50'] <= yesterday_row['SMA200']) and (today_row['SMA50'] > today_row['SMA200'])
                crossed_150 = (yesterday_row['SMA50'] <= yesterday_row['SMA150']) and (today_row['SMA50'] > today_row['SMA150'])
                price_crossed_50 = (yesterday_row['Close'] <= yesterday_row['SMA50']) and (today_row['Close'] > today_row['SMA50'])
                price_crossed_150 = (yesterday_row['Close'] <= yesterday_row['SMA150']) and (today_row['Close'] > today_row['SMA150'])
                price_crossed_200 = (yesterday_row['Close'] <= yesterday_row['SMA200']) and (today_row['Close'] > today_row['SMA200'])
                
                if crossed_golden or crossed_150 or price_crossed_50 or price_crossed_150 or price_crossed_200:
                    cross_buy_price = round(today_close_val, 2); cross_exit_price = round(today_close_val * 0.94, 2) 
                    cross_target_price = round(today_close_val * 1.18, 2) 
                    cross_confidence = "High (Golden Cross)" if crossed_golden else "Medium-High (Crossover)"
                    base_cross_rec = (f"Technical moving average crossover signal! Buy near ₹{cross_buy_price:.2f} "
                                      f"to ride the emerging uptrend. Set stop loss at ₹{cross_exit_price:.2f} targeting swing high ₹{cross_target_price:.2f}.")
                    res["crossover_ma"] = {
                        "symbol": sym.strip().upper(), "company_name": get_company_name(sym), "cmp": today_close_val,
                        "day_change_pct": round(((today_close_val - yesterday_row['Close']) / yesterday_row['Close'] * 100), 2),
                        "dist_50sma_pct": round((today_close_val - sma50) / sma50 * 100, 2),
                        "dist_200sma_pct": round((today_close_val - sma200) / sma200 * 100, 2), "setup_type": "crossover_ma",
                        "buy_price": cross_buy_price, "exit_price": cross_exit_price, "target_price": cross_target_price,
                        "confidence": cross_confidence, "recommendation": compute_rich_analysis(df_ma, sym, "MA Crossover", base_cross_rec)
                    }

            if len(df_ma) >= 250:
                today_row = df_ma.iloc[-1]; yesterday_row = df_ma.iloc[-2]; c_val = float(today_row['Close'])
                sma50 = float(today_row['SMA50']); sma150 = float(today_row['SMA150']); sma200 = float(today_row['SMA200'])
                sma200_10d_ago = float(df_ma['SMA200'].iloc[-11]) if len(df_ma) >= 210 else sma200
                high_52w = float(df_ma['High'].iloc[-250:].max()); low_52w = float(df_ma['Low'].iloc[-250:].min())
                
                if c_val > sma150 and c_val > sma200 and sma150 > sma200 and sma200 > sma200_10d_ago and sma50 > sma150 and sma50 > sma200 and c_val > sma50 and c_val >= 1.30 * low_52w and c_val >= 0.75 * high_52w:
                    run_up_200 = round(((c_val - sma200) / sma200 * 100), 2)
                    run_up_52w = round(((c_val - low_52w) / low_52w * 100), 2)
                    is_early = bool(c_val <= 1.20 * sma200)
                    exit_price = round(min(sma200 * 0.98, c_val * 0.94), 2)
                    distance_200 = (c_val - sma200) / sma200
                    target_mult = 1.40 - min(0.15, distance_200 * 0.7) if is_early else 1.18 - min(0.06, (distance_200 - 0.20) * 0.4)
                    target_price = round(max(high_52w * 1.05, c_val * target_mult), 2)
                    min_confidence = "High (Minervini Stage-2)" if is_early else "Medium-High (Minervini Extended)"
                    rem_pct = ((target_price - c_val) / c_val * 100)
                    stage_label = "Early Stage-2 Accumulation" if is_early else "Extended Stage-2 Uptrend"
                    base_minervini_rec = (f"Mark Minervini Stage-2 Trend Template verified! The stock is in an active '{stage_label}' "
                                          f"having run up {run_up_52w:.1f}% from its 52w low and holding {run_up_200:.1f}% above its 200 SMA support. "
                                          f"Buy around CMP ₹{c_val:.2f}. Set stop loss at ₹{exit_price:.2f} (tight support lock) "
                                          f"targeting momentum swing target of ₹{target_price:.2f} (remaining potential +{rem_pct:.1f}%).")
                    res["minervini"] = {
                        "symbol": sym.strip().upper(), "company_name": get_company_name(sym), "cmp": today_close_val,
                        "day_change_pct": round(((today_close_val - yesterday_row['Close']) / yesterday_row['Close'] * 100), 2),
                        "setup_type": "minervini", "run_up_200": run_up_200, "run_up_52w": run_up_52w, "is_early": is_early,
                        "buy_price": round(c_val, 2), "exit_price": exit_price, "target_price": target_price,
                        "confidence": min_confidence, "recommendation": compute_rich_analysis(df_ma, sym, "Minervini Stage-2", base_minervini_rec)
                    }
                    
            scan_res = scan_stock(symbol=sym, df=df, min_dry_days=min_dry, max_dry_days=max_dry, min_volume_ratio=min_vol_ratio, min_price_change=min_price_chg, min_dry_spikes=min_dry_spikes)
            if scan_res is not None:
                scan_res['market_cap_cr'] = 0.0
                if scan_res['signal_strength'] >= min_signal_str:
                    if (not above_50dma_only or scan_res.get('above_50dma', False)) and (not above_200dma_only or scan_res.get('above_200dma', False)):
                        res["flagged"] = scan_res
                        
            coiled_res = scan_coiled_spring(sym, df, max_tightness=vcp_max_tightness)
            if coiled_res is not None:
                coiled_res['market_cap_cr'] = 0.0
                if coiled_res['squeeze_score'] >= min_signal_str:
                    res["coiled"] = coiled_res
                        
            df_wt = df
            if df_wt is not None and len(df_wt) >= 40:
                wt_res = scan_wt_cross(sym, df_wt)
                if wt_res is not None:
                    wt_res['timeframe'] = "Daily"
                    res["wt"] = wt_res
                    
            if df is not None:
                res["vcs"] = scan_vcs(sym, df)
                res["structural_vcp"] = scan_structural_vcp(sym, df)
                res["vpa"] = scan_vpa_trend(sym, df)
                
            return res
"""

run_loop_replacement = """
        import concurrent.futures
        import os
        
        status_box.text(f"Phase 3/3: Scanning {n_stocks} active NSE listed equities (Price > ₹200)...")
        prog_bar.progress(0)
        
        # Parallel Execution Core
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 2 if os.cpu_count() else 8)) as executor:
            future_to_sym = {}
            for sym in scan_symbols:
                df = bulk_data.get(sym.strip().upper())
                if df is None:
                    df = fetch_ohlcv(sym)
                future = executor.submit(process_single_symbol, sym, df, open_price_map, close_price_map, high_price_map, low_price_map, volume_map, min_dry, max_dry, min_vol_ratio, min_price_chg, min_dry_spikes, min_signal_str, above_50dma_only, above_200dma_only, vcp_max_tightness)
                future_to_sym[future] = sym
                
            for i, future in enumerate(concurrent.futures.as_completed(future_to_sym)):
                sym = future_to_sym[future]
                try:
                    res = future.result()
                    if res.get("failed"):
                        failed_count += 1
                        continue
                    if res.get("gapup"): gapup_list.append(res["gapup"])
                    if res.get("above_ma"): above_ma_list.append(res["above_ma"])
                    if res.get("support_ma"): support_ma_list.append(res["support_ma"])
                    if res.get("crossover_ma"): crossover_ma_list.append(res["crossover_ma"])
                    if res.get("minervini"): minervini_list.append(res["minervini"])
                    if res.get("flagged"): flagged_list.append(res["flagged"])
                    if res.get("coiled"): coiled_list.append(res["coiled"])
                    if res.get("wt"): wt_list.append(res["wt"])
                    if res.get("vcs"): vcs_list.append(res["vcs"])
                    if res.get("structural_vcp"): structural_vcp_list.append(res["structural_vcp"])
                    if res.get("vpa"): vpa_list.append(res["vpa"])
                except Exception as exc:
                    print(f"Error processing {sym}: {exc}")
                    failed_count += 1
                    
                # Throttle UI Updates (every 25 iterations or at the end)
                if (i + 1) % 25 == 0 or i + 1 == n_stocks:
                    status_box.text(f"Phase 3/3: Scanning ({i+1}/{n_stocks})")
                    prog_bar.progress((i + 1) / n_stocks)
"""

start_idx = code.find("        for i, sym in enumerate(scan_symbols):")
end_idx = code.find("        # Clean progress assets")

if start_idx == -1 or end_idx == -1:
    print("Could not find start or end index for loop.")
else:
    new_code = code[:start_idx] + process_func + "\n" + run_loop_replacement + "\n" + code[end_idx:]
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(new_code)
    print("Successfully patched app.py!")

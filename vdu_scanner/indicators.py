# indicators.py
# Pre-computation engine: calculates ALL technical indicators for a DataFrame
# in a single pass, eliminating 5-8x redundant recalculations per symbol.

import pandas as pd
import numpy as np
import json


def precompute_indicators(df: pd.DataFrame) -> dict:
    """
    Computes all technical indicators needed by every scanner in one pass.
    Returns a dict containing:
      - 'df': the enriched DataFrame with all computed columns
      - Individual indicator values for the latest bar
      - Pre-computed rich analysis components (RSI status, CCI status, etc.)

    This function is called ONCE per symbol in process_single_symbol(),
    and the result is passed to all scanner functions to avoid redundant work.
    """
    if df is None or len(df) < 5:
        return None

    result = {}
    df_enriched = df.copy()
    close = df_enriched['Close']
    high = df_enriched['High']
    low = df_enriched['Low']
    n = len(df_enriched)

    # =========================================================================
    # 1. SIMPLE MOVING AVERAGES (SMA)
    # =========================================================================
    for period in [20, 50, 65, 150, 200]:
        col_name = f'SMA{period}'
        if n >= period:
            df_enriched[col_name] = close.rolling(window=period).mean()
        else:
            df_enriched[col_name] = np.nan

    # Aliases used by scan_stock
    df_enriched['MA50'] = df_enriched['SMA50']
    df_enriched['MA200'] = df_enriched['SMA200']

    # =========================================================================
    # 2. EXPONENTIAL MOVING AVERAGES (EMA)
    # =========================================================================
    df_enriched['EMA20'] = close.ewm(span=20, adjust=False).mean()
    # Monthly EMAs (used by monthly momentum — not needed here for daily,
    # but pre-compute EMA8/12 for consistency)
    df_enriched['EMA8']  = close.ewm(span=8,  adjust=False).mean()
    df_enriched['EMA12'] = close.ewm(span=12, adjust=False).mean()

    # =========================================================================
    # 2b. MACD (12, 26, 9)
    # =========================================================================
    ema12_macd = close.ewm(span=12, adjust=False).mean()
    ema26_macd = close.ewm(span=26, adjust=False).mean()
    df_enriched['MACD_line']   = ema12_macd - ema26_macd
    df_enriched['MACD_signal'] = df_enriched['MACD_line'].ewm(span=9, adjust=False).mean()
    df_enriched['MACD_hist']   = df_enriched['MACD_line'] - df_enriched['MACD_signal']

    # =========================================================================
    # 2c. BOLLINGER BANDS (20, 2σ)
    # =========================================================================
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df_enriched['BB_upper'] = bb_mid + 2 * bb_std
    df_enriched['BB_lower'] = bb_mid - 2 * bb_std
    # Normalized band width (width / midline) — lower = tighter squeeze
    df_enriched['BB_width'] = (df_enriched['BB_upper'] - df_enriched['BB_lower']) / bb_mid.replace(0, np.nan)
    # BB squeeze: current width <= 115% of the 6-month (126 trading days) minimum width
    bb_width_6m_min = df_enriched['BB_width'].rolling(min(126, n)).min()
    df_enriched['BB_squeeze'] = df_enriched['BB_width'] <= bb_width_6m_min * 1.15

    # =========================================================================
    # 3. RSI (14) — Wilder's smoothing
    # =========================================================================
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ema_gain = gain.ewm(com=13, adjust=False).mean()
    ema_loss = loss.ewm(com=13, adjust=False).mean()
    rs = ema_gain / (ema_loss + 1e-9)
    df_enriched['RSI'] = 100 - (100 / (1 + rs))
    df_enriched['RSI_SMA14'] = df_enriched['RSI'].rolling(window=14).mean()

    # =========================================================================
    # 4. CCI (14)
    # =========================================================================
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=14).mean()
    mad_manual = tp.rolling(window=14).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
    df_enriched['CCI'] = (tp - sma_tp) / (0.015 * mad_manual + 1e-9)

    # =========================================================================
    # 5. WAVETREND (WT1 & WT2) — LazyBear
    # =========================================================================
    ap = (high + low + close) / 3.0
    esa = ap.ewm(span=10, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=10, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d + 1e-10)
    df_enriched['WT1'] = ci.ewm(span=21, adjust=False).mean()
    df_enriched['WT2'] = df_enriched['WT1'].rolling(window=4).mean()

    # =========================================================================
    # 6. ATR & VCS RATIOS (for scan_vcs)
    # =========================================================================
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    df_enriched['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Normalized TR (percent of price)
    df_enriched['TR_pct'] = df_enriched['TR'] / prev_close.replace(0, np.nan)

    # ATR contraction ratios (short vs long)
    lenShort, lenLong, lenVol = 13, 63, 50
    df_enriched['trShort'] = df_enriched['TR_pct'].rolling(lenShort).mean()
    df_enriched['trLong'] = df_enriched['TR_pct'].rolling(lenLong).mean()
    df_enriched['ratioATR'] = df_enriched['trShort'] / df_enriched['trLong']

    # StdDev contraction ratios
    df_enriched['ret'] = close.pct_change()
    df_enriched['stdShort'] = df_enriched['ret'].rolling(lenShort).std()
    df_enriched['stdLong'] = df_enriched['ret'].rolling(lenLong).std()
    df_enriched['ratioStd'] = df_enriched['stdShort'] / df_enriched['stdLong']

    # Volume contraction ratios
    df_enriched['volAvg'] = df_enriched['Volume'].rolling(lenVol).mean()
    df_enriched['volShort'] = df_enriched['Volume'].rolling(5).mean()
    df_enriched['volRatio_vcs'] = df_enriched['volShort'] / df_enriched['volAvg']

    # =========================================================================
    # 7. EXTRACT LATEST VALUES
    # =========================================================================
    latest = df_enriched.iloc[-1]

    result['df'] = df_enriched

    # SMAs
    for period in [20, 50, 65, 150, 200]:
        result[f'sma{period}'] = float(latest[f'SMA{period}']) if not pd.isna(latest[f'SMA{period}']) else None

    # EMAs
    result['ema20'] = float(latest['EMA20'])
    result['ema8']  = float(latest['EMA8'])
    result['ema12'] = float(latest['EMA12'])

    # RSI & CCI
    result['rsi'] = float(latest['RSI']) if not pd.isna(latest['RSI']) else None
    result['rsi_sma14'] = float(latest['RSI_SMA14']) if not pd.isna(latest['RSI_SMA14']) else None
    result['cci'] = float(latest['CCI']) if not pd.isna(latest['CCI']) else None

    # WaveTrend
    result['wt1'] = float(latest['WT1']) if not pd.isna(latest['WT1']) else None
    result['wt2'] = float(latest['WT2']) if not pd.isna(latest['WT2']) else None

    # MACD
    result['macd_line']   = float(latest['MACD_line'])   if not pd.isna(latest['MACD_line'])   else None
    result['macd_signal'] = float(latest['MACD_signal']) if not pd.isna(latest['MACD_signal']) else None
    result['macd_hist']   = float(latest['MACD_hist'])   if not pd.isna(latest['MACD_hist'])   else None
    # Bullish MACD cross-up: histogram just turned from negative to positive (today)
    if n >= 2 and not pd.isna(df_enriched['MACD_hist'].iloc[-1]) and not pd.isna(df_enriched['MACD_hist'].iloc[-2]):
        result['macd_cross_up'] = (
            df_enriched['MACD_hist'].iloc[-1] > 0 and
            df_enriched['MACD_hist'].iloc[-2] <= 0
        )
    else:
        result['macd_cross_up'] = False

    # Bollinger Bands
    result['bb_width']   = float(latest['BB_width'])  if not pd.isna(latest['BB_width'])  else None
    result['bb_squeeze'] = bool(latest['BB_squeeze'])  if not pd.isna(latest['BB_squeeze']) else False


    # OI Interpretation
    oi_status = None
    oi_interp = None
    if 'Open Interest' in df_enriched.columns and n >= 2:
        try:
            prev_close = df_enriched['Close'].iloc[-2]
            curr_close = df_enriched['Close'].iloc[-1]
            prev_oi = df_enriched['Open Interest'].iloc[-2]
            curr_oi = df_enriched['Open Interest'].iloc[-1]
            
            if pd.notna(prev_oi) and pd.notna(curr_oi):
                price_up = curr_close > prev_close
                oi_up = curr_oi > prev_oi
                if price_up and oi_up:
                    oi_status = "Long Build-up"
                    oi_interp = "Bullish – New long positions are being created. Strong buying interest."
                elif not price_up and oi_up:
                    oi_status = "Short Build-up"
                    oi_interp = "Bearish – New short positions are being created. Strong selling interest."
                elif price_up and not oi_up:
                    oi_status = "Short Covering"
                    oi_interp = "Bullish – Existing short sellers are closing positions, causing price to rise."
                else:
                    oi_status = "Long Unwinding"
                    oi_interp = "Bearish – Existing long holders are exiting positions."
        except Exception:
            pass
            
    result['oi_status'] = oi_status
    result['oi_interp'] = oi_interp

    # CMP and price info
    result['cmp'] = float(latest['Close'])
    result['today_volume'] = int(latest['Volume'])

    return result


def build_rich_analysis_from_indicators(indicators: dict, symbol: str, strategy_name: str, base_rec_text: str) -> str:
    """
    Builds the rich analysis JSON payload using PRE-COMPUTED indicator values
    instead of recalculating RSI, CCI, EMA, SMA from scratch.

    This is a drop-in replacement for compute_rich_analysis() when indicators
    are available. The output format is 100% identical.
    """
    if indicators is None:
        return base_rec_text

    try:
        rsi_val = indicators.get('rsi')
        cci_val = indicators.get('cci')
        ema20 = indicators.get('ema20')
        sma50 = indicators.get('sma50')
        sma200 = indicators.get('sma200')
        oi_status = indicators.get('oi_status')
        oi_interp = indicators.get('oi_interp')

        cmp = indicators.get('cmp', 0.0)

        if rsi_val is None or cci_val is None or ema20 is None:
            return base_rec_text

        # Use defaults if SMAs are not available (insufficient data)
        if sma50 is None:
            sma50 = cmp
        if sma200 is None:
            sma200 = cmp

        # ---- RSI Interpretation (identical to compute_rich_analysis) ----
        if rsi_val >= 70:
            rsi_status = "Overbought"
            rsi_interp = "Extremely strong bullish momentum; check for near-term exhaustion."
        elif rsi_val >= 50:
            rsi_status = "Bullish Momentum"
            rsi_interp = "Solid buying interest with strong upward price power."
        elif rsi_val >= 35:
            rsi_status = "Neutral Consolidation"
            rsi_interp = "Price is stable, hovering within a sideways range."
        else:
            rsi_status = "Oversold Bounce"
            rsi_interp = "Deep oversold territory; highly prime for a technical recovery."

        # ---- CCI Interpretation (identical to compute_rich_analysis) ----
        if cci_val >= 100:
            cci_status = "Bullish Breakout"
            cci_interp = "Price velocity is in a strong upward breakout phase."
        elif cci_val >= 0:
            cci_status = "Positive Territory"
            cci_interp = "Short-term momentum is supportive of higher prices."
        elif cci_val >= -100:
            cci_status = "Weak Momentum"
            cci_interp = "Trading in bearish territory, under consolidation."
        else:
            cci_status = "Extremely Oversold"
            cci_interp = "Deep oversold condition; expect a sharp mean-reversion move."

        # ---- Triggers Checklist (identical logic) ----
        triggers = []
        if cmp > sma50 and sma50 > sma200:
            triggers.append("✔️ Golden Trend: Price is supported by long-term institutional SMA structure.")
        elif cmp > sma50:
            triggers.append("✔️ Mid-Term Bullish: Trading comfortably above the 50-day moving average.")

        if rsi_val >= 50 and rsi_val <= 68:
            triggers.append("✔️ RSI Sweet Spot: Healthy buying momentum without being overextended.")
        elif rsi_val < 35:
            triggers.append("✔️ Reversion Alert: Deeply discounted oversold price levels ready to bounce.")

        if cci_val > 100:
            triggers.append("✔️ Momentum Surge: CCI confirms active breakout velocity is backing the move.")
        elif cci_val < -100:
            triggers.append("✔️ Oversold Stretch: CCI indicates institutional selling is exhausted.")

        if cmp >= ema20 * 0.98 and cmp <= ema20 * 1.02:
            triggers.append("✔️ Dynamic Pullback: Price is pulling back perfectly into the 20-day EMA support anchor.")
        elif cmp > ema20:
            triggers.append("✔️ Fast Trend: Sustained buying velocity holding above the short-term 20 EMA.")

        if not triggers:
            triggers.append("✔️ Value Consolidation: Standard technical entry on strategy parameters.")

        # ---- MA Reasoning (identical logic) ----
        if cmp > ema20 and cmp > sma50:
            ma_reasoning = f"Price holds strong above short-term 20 EMA (₹{ema20:,.2f}) and medium-term 50 SMA (₹{sma50:,.2f}), confirming dynamic uptrend support."
        elif cmp > ema20:
            ma_reasoning = f"Price is trading above the fast-moving 20 EMA (₹{ema20:,.2f}), while testing major resistance/consolidation zones."
        elif cmp > sma50:
            ma_reasoning = f"Price holds above structural 50 SMA (₹{sma50:,.2f}) institutional support, despite a short-term dip below 20 EMA."
        else:
            ma_reasoning = f"Price is testing critical structural zones near the 50 SMA (₹{sma50:,.2f}) and 200 SMA (₹{sma200:,.2f}) floors."

        # ---- RSI/CCI Reasoning (identical logic) ----
        if rsi_val >= 70:
            rsi_reasoning = f"RSI is extremely strong at {rsi_val:.1f} ({rsi_status}), showing high bullish momentum."
        elif rsi_val >= 50:
            rsi_reasoning = f"RSI at {rsi_val:.1f} ({rsi_status}) indicates active buying interest and healthy momentum."
        elif rsi_val >= 35:
            rsi_reasoning = f"RSI at {rsi_val:.1f} ({rsi_status}) indicates a healthy cooling consolidation zone."
        else:
            rsi_reasoning = f"RSI at {rsi_val:.1f} ({rsi_status}) indicates deeply oversold levels, highly prime for a technical bounce."

        if cci_val >= 100:
            cci_reasoning = f"CCI is at {cci_val:.1f} ({cci_status}), confirming an active breakout phase."
        elif cci_val >= 0:
            cci_reasoning = f"CCI at {cci_val:.1f} ({cci_status}) indicates supportive short-term momentum."
        elif cci_val >= -100:
            cci_reasoning = f"CCI is at {cci_val:.1f} ({cci_status}), reflecting temporary sideways consolidation."
        else:
            cci_reasoning = f"CCI at {cci_val:.1f} ({cci_status}) shows extreme selling exhaustion, signaling an upward pivot."


        # ---- OI Reasoning ----
        oi_reasoning = ""
        if oi_status and oi_interp:
            oi_reasoning = f"OI Analysis: {oi_status} ({oi_interp})."

        # ---- Extract SL/Target from base text (identical logic) ----
        sl_target_part = ""
        if "Set stop loss" in base_rec_text:
            sl_target_part = base_rec_text[base_rec_text.index("Set stop loss"):]
        elif "Place stop loss" in base_rec_text:
            sl_target_part = base_rec_text[base_rec_text.index("Place stop loss"):]
        elif "stop loss" in base_rec_text.lower():
            try:
                idx = base_rec_text.lower().index("stop loss")
                sl_target_part = base_rec_text[idx-4:]
            except ValueError:
                sl_target_part = base_rec_text
        else:
            sl_target_part = base_rec_text

        final_text = f"{rsi_reasoning} {cci_reasoning} {ma_reasoning} {sl_target_part}"

        analysis_payload = {
            "is_rich": True,
            "text": final_text,
            "rsi": round(rsi_val, 1),
            "rsi_status": rsi_status,
            "rsi_interp": rsi_interp,
            "cci": round(cci_val, 1),
            "cci_status": cci_status,
            "cci_interp": cci_interp,
            "oi_status": oi_status,
            "oi_interp": oi_interp,
            "ema20": round(ema20, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "triggers": triggers
        }
        return json.dumps(analysis_payload)
    except Exception as e:
        print(f"Error building rich analysis from pre-computed indicators for {symbol}: {e}")
        return base_rec_text

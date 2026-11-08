# export_pva_matrix.py
"""
Utility script to generate a standalone Microsoft Excel / Google Sheets workbook
classifying all scanned stocks into the 6 Price + Volume market conditions:

1. Increasing Volume + Increasing Price  -> Bullish (Strong buying interest; possible accumulation or breakout)
2. Increasing Volume + Stable Price      -> Watch (Buyers and sellers competing; possible accumulation)
3. Decreasing Volume + Increasing Price  -> Caution (Price rising with weaker participation; rally may lack strength)
4. Decreasing Volume + Stable Price      -> Neutral (Low participation; consolidation or lack of interest)
5. Decreasing Volume + Decreasing Price  -> Watch reversal (Selling pressure may be weakening; possible selling exhaustion)
6. Increasing Volume + Decreasing Price  -> Bearish (Strong selling pressure; possible distribution or breakdown)
"""

import os
import sys
import io
import pandas as pd
from datetime import datetime

# Add current folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import database

def classify_pva(price_change_pct: float, vol_ratio: float, vol_rising: bool = None) -> dict:
    """
    Classifies a stock into one of the 6 Price + Volume market conditions.
    """
    try:
        price_change_pct = float(price_change_pct or 0.0)
    except (ValueError, TypeError):
        price_change_pct = 0.0

    try:
        vol_ratio = float(vol_ratio or 1.0)
    except (ValueError, TypeError):
        vol_ratio = 1.0

    # Determine volume movement
    if vol_ratio is not None:
        vol_mov = "Increasing" if vol_ratio >= 1.0 else "Decreasing"
    elif vol_rising is not None:
        vol_mov = "Increasing" if vol_rising else "Decreasing"
    else:
        vol_mov = "Increasing" if price_change_pct > 0 else "Decreasing"

    # Determine price movement (with +/- 0.5% threshold for Stable)
    if price_change_pct > 0.5:
        price_mov = "Increasing"
    elif price_change_pct < -0.5:
        price_mov = "Decreasing"
    else:
        price_mov = "Stable"

    matrix = {
        ('Increasing', 'Increasing'): (
            'Bullish',
            'Strong buying interest; possible accumulation or breakout',
            '#00e676', '#E8F5E9', '#1B5E20'
        ),
        ('Increasing', 'Stable'): (
            'Watch',
            'Buyers and sellers competing; possible accumulation',
            '#38bdf8', '#E1F5FE', '#01579B'
        ),
        ('Decreasing', 'Increasing'): (
            'Caution',
            'Price rising with weaker participation; rally may lack strength',
            '#fbbf24', '#FFF8E1', '#B78103'
        ),
        ('Decreasing', 'Stable'): (
            'Neutral',
            'Low participation; consolidation or lack of interest',
            '#94a3b8', '#F1F5F9', '#475569'
        ),
        ('Decreasing', 'Decreasing'): (
            'Watch reversal',
            'Selling pressure may be weakening; possible selling exhaustion',
            '#c084fc', '#F3E8FF', '#6B21A8'
        ),
        ('Increasing', 'Decreasing'): (
            'Bearish',
            'Strong selling pressure; possible distribution or breakdown',
            '#ef4444', '#FFEBEE', '#B71C1C'
        ),
    }

    sig, interp, color, bg, fg = matrix.get(
        (vol_mov, price_mov),
        ('Neutral', 'Low participation; consolidation or lack of interest', '#94a3b8', '#F1F5F9', '#475569')
    )

    return {
        'vol_movement': vol_mov,
        'price_movement': price_mov,
        'signal': sig,
        'interpretation': interp,
        'color': color,
        'excel_bg': bg,
        'excel_fg': fg
    }


def generate_pva_excel_bytes(vpa_data: list[dict], scan_date: str = None) -> bytes:
    """
    Builds a dual-tab Microsoft Excel (.xlsx) workbook compatible with Google Sheets:
      Sheet 1: Price_Volume_Matrix — All stocks with 6-condition classification & interpretations
      Sheet 2: Market_Conditions_Legend — Reference guide with stock counts and percentages
    """
    rows = []
    for r in vpa_data:
        pva = r.get('pva') or classify_pva(
            r.get('day_change_pct', 0.0),
            r.get('vol_ratio_5d', 1.0),
            r.get('vol_rising')
        )
        sym = str(r.get('symbol', '')).strip().upper().replace('.NS', '')
        rows.append({
            'Symbol': sym,
            'Company Name': r.get('company_name', ''),
            'CMP (₹)': round(float(r.get('cmp', 0.0)), 2),
            'Change (%)': round(float(r.get('day_change_pct', 0.0)), 2),
            'Volume Ratio (5d/20d)': round(float(r.get('vol_ratio_5d', 1.0)), 2),
            'Volume Movement': pva['vol_movement'],
            'Price Movement': pva['price_movement'],
            'Signal': pva['signal'],
            'Automatic Interpretation': pva['interpretation'],
            'Blast Score': int(r.get('blast_score', 0)),
            'Blast Tier': r.get('blast_tier', ''),
            'VPA Score': int(r.get('score', 0)),
            'TradingView Link': f"https://in.tradingview.com/chart/?symbol=NSE:{sym}"
        })

    df_matrix = pd.DataFrame(rows)

    # Sort: Bullish first, then Watch, Watch reversal, Caution, Neutral, Bearish
    signal_priority = {
        'Bullish': 1,
        'Watch': 2,
        'Watch reversal': 3,
        'Caution': 4,
        'Neutral': 5,
        'Bearish': 6
    }
    if not df_matrix.empty and 'Signal' in df_matrix:
        df_matrix['_prio'] = df_matrix['Signal'].map(lambda s: signal_priority.get(s, 99))
        df_matrix = df_matrix.sort_values(by=['_prio', 'Blast Score', 'Change (%)'], ascending=[True, False, False])
        df_matrix.drop(columns=['_prio'], inplace=True)

    legend_data = [
        {
            'Volume Movement': 'Increasing',
            'Price Movement': 'Increasing',
            'Interpretation': 'Strong buying interest; possible accumulation or breakout',
            'Signal': 'Bullish'
        },
        {
            'Volume Movement': 'Increasing',
            'Price Movement': 'Stable',
            'Interpretation': 'Buyers and sellers competing; possible accumulation',
            'Signal': 'Watch'
        },
        {
            'Volume Movement': 'Decreasing',
            'Price Movement': 'Increasing',
            'Interpretation': 'Price rising with weaker participation; rally may lack strength',
            'Signal': 'Caution'
        },
        {
            'Volume Movement': 'Decreasing',
            'Price Movement': 'Stable',
            'Interpretation': 'Low participation; consolidation or lack of interest',
            'Signal': 'Neutral'
        },
        {
            'Volume Movement': 'Decreasing',
            'Price Movement': 'Decreasing',
            'Interpretation': 'Selling pressure may be weakening; possible selling exhaustion',
            'Signal': 'Watch reversal'
        },
        {
            'Volume Movement': 'Increasing',
            'Price Movement': 'Decreasing',
            'Interpretation': 'Strong selling pressure; possible distribution or breakdown',
            'Signal': 'Bearish'
        },
    ]

    total_stocks = len(df_matrix)
    for leg in legend_data:
        c = len(df_matrix[df_matrix['Signal'] == leg['Signal']]) if not df_matrix.empty else 0
        leg['Stock Count'] = c
        pct = round(c / total_stocks * 100, 1) if total_stocks > 0 else 0.0
        leg['% of Universe'] = f"{pct}%"

    df_legend = pd.DataFrame(legend_data)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_matrix.to_excel(writer, sheet_name='Price_Volume_Matrix', index=False)
        df_legend.to_excel(writer, sheet_name='Market_Conditions_Legend', index=False)

        workbook = writer.book

        # Styles
        header_fmt = workbook.add_format({
            'bold': True,
            'bg_color': '#0F172A',
            'font_color': '#38BDF8',
            'border': 1,
            'border_color': '#334155',
            'font_size': 11,
            'align': 'center',
            'valign': 'vcenter'
        })

        cell_fmt = workbook.add_format({
            'font_size': 10,
            'valign': 'vcenter',
            'border': 1,
            'border_color': '#E2E8F0'
        })

        signal_formats = {
            'Bullish': workbook.add_format({
                'bg_color': '#DCFCE7', 'font_color': '#15803D', 'bold': True, 'border': 1, 'border_color': '#86EFAC', 'align': 'center'
            }),
            'Watch': workbook.add_format({
                'bg_color': '#E0F2FE', 'font_color': '#0369A1', 'bold': True, 'border': 1, 'border_color': '#7DD3FC', 'align': 'center'
            }),
            'Caution': workbook.add_format({
                'bg_color': '#FEF3C7', 'font_color': '#B45309', 'bold': True, 'border': 1, 'border_color': '#FCD34D', 'align': 'center'
            }),
            'Neutral': workbook.add_format({
                'bg_color': '#F1F5F9', 'font_color': '#475569', 'border': 1, 'border_color': '#CBD5E1', 'align': 'center'
            }),
            'Watch reversal': workbook.add_format({
                'bg_color': '#F3E8FF', 'font_color': '#7E22CE', 'bold': True, 'border': 1, 'border_color': '#D8B4FE', 'align': 'center'
            }),
            'Bearish': workbook.add_format({
                'bg_color': '#FEE2E2', 'font_color': '#B91C1C', 'bold': True, 'border': 1, 'border_color': '#FCA5A5', 'align': 'center'
            }),
        }

        # Sheet 1: Price_Volume_Matrix
        ws1 = writer.sheets['Price_Volume_Matrix']
        ws1.freeze_panes(1, 0)
        ws1.set_row(0, 26)

        for col_idx, col_name in enumerate(df_matrix.columns):
            ws1.write(0, col_idx, col_name, header_fmt)
            max_len = max(
                df_matrix[col_name].astype(str).map(len).max() if not df_matrix.empty else 10,
                len(col_name)
            ) + 4
            ws1.set_column(col_idx, col_idx, max(12, min(max_len, 65)), cell_fmt)

        if not df_matrix.empty:
            sig_col_idx = list(df_matrix.columns).index('Signal')
            for row_idx, sig in enumerate(df_matrix['Signal'], start=1):
                ws1.set_row(row_idx, 20)
                if sig in signal_formats:
                    ws1.write(row_idx, sig_col_idx, sig, signal_formats[sig])

        # Sheet 2: Market_Conditions_Legend
        ws2 = writer.sheets['Market_Conditions_Legend']
        ws2.freeze_panes(1, 0)
        ws2.set_row(0, 26)

        for col_idx, col_name in enumerate(df_legend.columns):
            ws2.write(0, col_idx, col_name, header_fmt)
            max_len = max(
                df_legend[col_name].astype(str).map(len).max() if not df_legend.empty else 10,
                len(col_name)
            ) + 4
            ws2.set_column(col_idx, col_idx, max(14, min(max_len, 70)), cell_fmt)

        leg_sig_col_idx = list(df_legend.columns).index('Signal')
        for row_idx, leg in enumerate(legend_data, start=1):
            ws2.set_row(row_idx, 22)
            sig = leg['Signal']
            if sig in signal_formats:
                ws2.write(row_idx, leg_sig_col_idx, sig, signal_formats[sig])

    return output.getvalue()


if __name__ == "__main__":
    dates = database.get_available_scan_dates()
    print("Available dates in DB:", dates)
    vpa_data = None
    chosen_date = None
    for d in dates:
        cached = database.get_cached_vpa(d)
        if cached:
            vpa_data = cached
            chosen_date = d
            break

    if not vpa_data:
        print("No cached VPA data found. Please run a VPA scan first.")
        sys.exit(0)

    print(f"Loaded {len(vpa_data)} VPA records from {chosen_date}.")
    excel_bytes = generate_pva_excel_bytes(vpa_data, scan_date=chosen_date)
    out_file = f"price_volume_matrix_{chosen_date}.xlsx"
    with open(out_file, "wb") as f:
        f.write(excel_bytes)
    print(f"Generated {out_file} ({len(excel_bytes)} bytes) successfully!")

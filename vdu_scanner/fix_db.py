import sys

content = open('database.py', 'r').read()

content = content.replace('''            SELECT symbol, sector, close, pressure, risk_50d,
                   trend_tpr, rs_rating, vcp_status, vcp_range_pct,
                   entry_signal
            FROM scanned_vcp_minervini''', '''            SELECT symbol, sector, close, pressure, risk_50d,
                   trend_tpr, rs_rating, vcp_status, vcp_range_pct,
                   vcp10_status, vcp10_range_pct, vcp15_status, vcp15_range_pct,
                   entry_signal
            FROM scanned_vcp_minervini''')

content = content.replace('''                "VCP range %": row['vcp_range_pct'],
                "Entry Signal": row['entry_signal']
            })''', '''                "VCP range %": row['vcp_range_pct'],
                "VCP (10d)": row['vcp10_status'],
                "VCP 10d range %": row['vcp10_range_pct'],
                "VCP (15d)": row['vcp15_status'],
                "VCP 15d range %": row['vcp15_range_pct'],
                "Entry Signal": row['entry_signal']
            })''')

open('database.py', 'w').write(content)

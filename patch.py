import re
import os

app_py_path = 'vdu_scanner/app.py'
with open(app_py_path, 'r', encoding='utf-8') as f:
    data = f.read()

new_func = '''
def get_market_date():
    from datetime import datetime, timedelta
    import pytz
    IST_TIMEZONE = pytz.timezone('Asia/Kolkata')
    today = datetime.now(IST_TIMEZONE)
    if today.isoweekday() == 7:
        return (today - timedelta(days=2)).strftime('%Y-%m-%d')
    elif today.isoweekday() == 6:
        return (today - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        return today.strftime('%Y-%m-%d')
'''

if 'get_market_date()' not in data:
    data = data.replace('IST_TIMEZONE = pytz.timezone(\'Asia/Kolkata\')', 'IST_TIMEZONE = pytz.timezone(\'Asia/Kolkata\')\n' + new_func)

data = data.replace('datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")', 'get_market_date()')

with open(app_py_path, 'w', encoding='utf-8') as f:
    f.write(data)

print("Dates patched successfully")

import pandas as pd
import yfinance as yf
from scanner import scan_bb_squeeze
import database
from datetime import datetime
import json

print("Downloading data...")
sym = "63MOONS.NS"
df_daily = yf.download(sym, period="1y", interval="1d", progress=False)
df_weekly = yf.download(sym, period="2y", interval="1wk", progress=False)
df_monthly = yf.download(sym, period="5y", interval="1mo", progress=False)

if isinstance(df_daily.columns, pd.MultiIndex):
    df_daily = df_daily.xs(sym, axis=1, level=1).dropna()
    df_weekly = df_weekly.xs(sym, axis=1, level=1).dropna()
    df_monthly = df_monthly.xs(sym, axis=1, level=1).dropna()

print("Scanning...")
res = scan_bb_squeeze("63MOONS", df_daily, df_weekly, df_monthly)
if res:
    print("Scan OK. Confidence:", res.get('confidence', '').encode('utf-8'))

if res:
    print("Saving to DB...")
    today_str = datetime.now().strftime("%Y-%m-%d")
    success = database.save_bb_squeeze_only(today_str, [res])
    print("Save Success:", success)

    print("Reading from DB...")
    cached = database.get_cached_bb_squeeze(today_str)
    for r in cached:
        if r['symbol'] == "63MOONS":
            print("Cached Score:", r.get('squeeze_score'))
            print("Cached Conf:", r.get('confidence', '').encode('utf-8'))

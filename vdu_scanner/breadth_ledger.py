"""
breadth_ledger.py
Integrates market breadth & sector clustering data from The Breadth Ledger (markets.manasarora.com).
Tracks:
  - Advance/Decline ratio across 994 liquid NSE equities
  - Fresh 52W, 6M, 3M, and 1M Highs vs Lows
  - Broad market participation (% of stocks above 50 DMA and 5 DMA)
  - Leading Industry Clusters (where fresh highs are clustering)
  - Daily breadth history over rolling 60 sessions
"""

import json
import time
import urllib.request
from datetime import datetime

_CACHE_DATA = None
_CACHE_TIMESTAMP = 0
_CACHE_TTL_SECONDS = 600  # 10 minutes


def fetch_breadth_data(force_refresh: bool = False) -> dict:
    """Fetches the latest breadth ledger data with in-memory caching."""
    global _CACHE_DATA, _CACHE_TIMESTAMP

    now = time.time()
    if not force_refresh and _CACHE_DATA is not None and (now - _CACHE_TIMESTAMP) < _CACHE_TTL_SECONDS:
        return _CACHE_DATA

    endpoints = [
        "https://breadth-ledger.manas-arora.workers.dev/data.json",
        "https://breadth-ledger.manas-arora.workers.dev/data-snapshot.json",
    ]

    for url in endpoints:
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if isinstance(data, dict) and "today" in data:
                        _CACHE_DATA = data
                        _CACHE_TIMESTAMP = now
                        return _CACHE_DATA
        except Exception:
            continue

    return _CACHE_DATA or {}


def get_market_breadth_summary() -> dict:
    """Returns formatted market breadth indicators and regime status."""
    data = fetch_breadth_data()
    today = data.get("today", {})
    idx_chg = data.get("indexChg", {})
    nifty = data.get("nifty", {})

    up_n = today.get("upN", 0)
    dn_n = today.get("dnN", 0)
    tot = today.get("n", up_n + dn_n or 994)
    hi_n = today.get("hiN", 0)
    lo_n = today.get("loN", 0)
    h1m = today.get("h1m", 0)
    l1m = today.get("l1m", 0)
    h3m = today.get("h3m", 0)
    l3m = today.get("l3m", 0)
    h6m = today.get("h6m", 0)
    l6m = today.get("l6m", 0)
    u4 = today.get("u4", 0)
    d4 = today.get("d4", 0)
    m50 = today.get("m50", 0.0)
    m5 = today.get("m5", 0.0)

    # Market regime classification based on % above 50 DMA
    if m50 >= 60.0:
        regime = "Strong Bullish Breadth"
        regime_badge = "🟢 AGGRESSIVE BUY"
        regime_desc = "Broad participation: Over 60% of stocks are holding above their 50 DMA. Breakouts have high follow-through."
    elif m50 >= 40.0:
        regime = "Neutral / Stock-Picker"
        regime_badge = "🟡 SELECTIVE ACCUMULATION"
        regime_desc = "Selective environment: 40-60% of stocks above 50 DMA. Focus strictly on top leading industry clusters and tight bases."
    else:
        regime = "Defensive / Distribution"
        regime_badge = "🔴 DEFENSE & CAPITAL PRESERVATION"
        regime_desc = "Weak breadth: Less than 40% of stocks above 50 DMA. High failure rate for breakouts; prioritize stop losses and cash."

    return {
        "as_of": data.get("asOf", ""),
        "universe_size": tot,
        "up_stocks": up_n,
        "down_stocks": dn_n,
        "up_pct": round((up_n / tot * 100), 1) if tot > 0 else 0.0,
        "highs_52w": hi_n,
        "lows_52w": lo_n,
        "highs_1m": h1m,
        "lows_1m": l1m,
        "highs_3m": h3m,
        "lows_3m": l3m,
        "highs_6m": h6m,
        "lows_6m": l6m,
        "surge_up_4pct": u4,
        "drop_dn_4pct": d4,
        "stocks_above_50dma": m50,
        "stocks_above_5dma": m5,
        "nifty_chg": idx_chg.get("nifty", nifty.get("chg", 0.0)),
        "midcap_chg": idx_chg.get("midcap", 0.0),
        "smallcap_chg": idx_chg.get("smallcap", 0.0),
        "vix_chg": idx_chg.get("vix", 0.0),
        "regime": regime,
        "regime_badge": regime_badge,
        "regime_desc": regime_desc,
    }


def get_leading_clusters() -> list[dict]:
    """Returns industry clusters ordered by breadth strength (highest 1M + 3M new highs)."""
    data = fetch_breadth_data()
    raw_clusters = data.get("clusters", [])

    results = []
    for c in raw_clusters:
        industry = c.get("industry", "Unknown")
        total_stocks = c.get("n", 0)
        h1 = c.get("h1", 0)
        h3 = c.get("h3", 0)
        h6 = c.get("h6", 0)
        names = c.get("names", [])
        all_leaders = [item[0] for item in c.get("all", []) if isinstance(item, (list, tuple)) and item]

        # Combine top names
        combined_names = names if names else all_leaders[:8]

        results.append({
            "industry": industry,
            "total_stocks": total_stocks,
            "highs_1m": h1,
            "highs_3m": h3,
            "highs_6m": h6,
            "score": (h1 * 2) + (h3 * 3) + (h6 * 4),
            "leading_stocks": combined_names,
        })

    # Sort descending by score
    results.sort(key=lambda x: (x["score"], x["highs_1m"]), reverse=True)
    return results


def get_cluster_stocks_map() -> dict[str, str]:
    """
    Returns a lookup dictionary mapping ticker symbol -> leading industry cluster.
    E.g. {'PNB': 'PSU Banks', 'MCX': 'Financial Services-Specialty', ...}
    """
    clusters = get_leading_clusters()
    sym_map = {}
    for c in clusters:
        ind = c["industry"]
        for s in c.get("leading_stocks", []):
            clean_s = str(s).strip().upper().replace(".NS", "")
            if clean_s and clean_s not in sym_map:
                sym_map[clean_s] = ind
    return sym_map


def get_stock_cluster_tag(symbol: str) -> str | None:
    """Returns the industry cluster name if this stock is an active leader, else None."""
    if not symbol:
        return None
    sym_clean = str(symbol).strip().upper().replace(".NS", "")
    cluster_map = get_cluster_stocks_map()
    return cluster_map.get(sym_clean)

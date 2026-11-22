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
        names = [str(x).strip().upper().replace(".NS", "") for x in c.get("names", []) if x]

        # In all, items are [ticker, high_flag].
        # flag: 3 = 6M high, 2 = 3M high, 1 = 1M high, 0 = no fresh high
        cluster_stocks = []
        stocks_6m = []
        stocks_3m = []
        stocks_1m = []
        stocks_no_high = []

        all_items = c.get("all", [])
        for item in all_items:
            if isinstance(item, (list, tuple)) and len(item) > 1:
                sym = str(item[0]).strip().upper().replace(".NS", "")
                try:
                    flag = int(item[1])
                except (ValueError, TypeError):
                    flag = 0

                tier_code = "6m" if flag >= 3 else ("3m" if flag == 2 else ("1m" if flag == 1 else "none"))
                tier_name = "6M High" if flag >= 3 else ("3M High" if flag == 2 else ("1M High" if flag == 1 else "No Fresh High"))

                cluster_stocks.append({
                    "symbol": sym,
                    "flag": flag,
                    "tier_code": tier_code,
                    "tier_name": tier_name,
                    "in_1m": flag >= 1,
                    "in_3m": flag >= 2,
                    "in_6m": flag >= 3,
                })

                if flag >= 3:
                    stocks_6m.append(sym)
                if flag >= 2:
                    stocks_3m.append(sym)
                if flag >= 1:
                    stocks_1m.append(sym)
                if flag == 0:
                    stocks_no_high.append(sym)

        # Sort cluster_stocks by flag descending (3, then 2, then 1, then 0), then symbol
        cluster_stocks.sort(key=lambda x: (-x["flag"], x["symbol"]))

        # Combine: keep distinct list preserving order (explicit names first, then remaining 1M+ active stocks)
        combined_set = set(names)
        combined_names = list(names)
        for s in stocks_1m:
            if s not in combined_set:
                combined_names.append(s)
                combined_set.add(s)

        results.append({
            "industry": industry,
            "total_stocks": total_stocks,
            "highs_1m": h1,
            "highs_3m": h3,
            "highs_6m": h6,
            "score": (h1 * 2) + (h3 * 3) + (h6 * 4),
            "leading_stocks": combined_names,
            "cluster_stocks": cluster_stocks,
            "stocks_6m": stocks_6m,
            "stocks_3m": stocks_3m,
            "stocks_1m": stocks_1m,
            "stocks_no_high": stocks_no_high,
        })

    # Sort descending by score, then 1M highs
    results.sort(key=lambda x: (x["score"], x["highs_1m"]), reverse=True)
    return results


def track_cluster_changes() -> dict:
    """
    Compares the current breadth clusters with the previous recorded snapshot in DB.
    Returns:
      - 'added': list of {'symbol': s, 'industry': ind, 'cluster_rank': rank} (newly entered fresh highs)
      - 'removed': list of {'symbol': s, 'industry': ind} (dropped from fresh highs)
      - 'net_change': len(added) - len(removed)
      - 'has_comparison': bool
      - 'prev_date': str
      - 'curr_date': str
      - 'new_symbols_set': set of symbols added (for fast lookup in UI)
      - 'total_leaders': int
    """
    try:
        import database
        data = fetch_breadth_data()
        as_of = data.get("asOf", "")
        # Derive date string YYYY-MM-DD
        curr_date = as_of[:10] if as_of and len(as_of) >= 10 else time.strftime("%Y-%m-%d")

        clusters = get_leading_clusters()
        # Build map: symbol -> dict of details
        curr_leaders: dict[str, dict] = {}
        for rank, c in enumerate(clusters, 1):
            for s in c.get("leading_stocks", []):
                sym = str(s).strip().upper().replace(".NS", "")
                if sym and sym not in curr_leaders:
                    curr_leaders[sym] = {
                        "industry": c["industry"],
                        "cluster_rank": rank,
                        "highs_1m": c["highs_1m"],
                        "highs_3m": c["highs_3m"],
                    }

        # Check existing snapshots before saving to identify previous snapshot
        recent_snapshots = database.get_latest_breadth_clusters(limit=5)
        prev_snapshot = None
        for snap in recent_snapshots:
            if snap.get("scan_date") != curr_date:
                prev_snapshot = snap
                break

        # Persist today's snapshot
        database.save_breadth_clusters(curr_date, as_of, clusters, curr_leaders)

        # Prepare near and rs maps for reasoning generation
        near_map = {x[0]: x for x in data.get("near", []) if isinstance(x, list) and x}
        rs_map = {x[0]: x[1] for x in data.get("rs", {}).get("rows", []) if isinstance(x, list) and len(x) > 1}

        def generate_added_reason(sym: str, ind: str, rank: int | None = None) -> str:
            n = near_map.get(sym)
            rs = rs_map.get(sym)
            parts = []
            if n and len(n) > 2:
                ht = n[2]
                dist = n[1]
                t_str = "52W High" if ht == 0 else ("1M High" if ht == 1 else ("3M High" if ht == 2 else "6M High"))
                if isinstance(dist, (int, float)) and abs(dist) <= 1.5:
                    parts.append(f"Fresh {t_str} Breakout ({dist:+.1f}% from peak)")
                else:
                    parts.append(f"New {t_str} Entry")
            else:
                parts.append("Fresh Cycle High Breakout")

            if rank and ind:
                parts.append(f"accumulating into #{rank} {ind} cluster")
            elif ind:
                parts.append(f"in {ind}")

            if rs:
                parts.append(f"RS: {rs}th percentile")

            return "; ".join(parts) + "."

        def generate_removed_reason(sym: str, ind: str) -> str:
            n = near_map.get(sym)
            if n and len(n) > 1 and isinstance(n[1], (int, float)) and n[1] < -2.0:
                return f"Pulled back {abs(n[1]):.1f}% below cycle peak; failed to sustain fresh-high breakout pace in {ind}."
            return f"Momentum cooled below fresh-high threshold; leadership rotated to stronger momentum peers in {ind}."

        if prev_snapshot and prev_snapshot.get("all_leaders"):
            prev_leaders = prev_snapshot["all_leaders"]
            prev_date = prev_snapshot.get("scan_date", "")

            curr_syms = set(curr_leaders.keys())
            prev_syms = set(prev_leaders.keys())

            added_syms = sorted(curr_syms - prev_syms)
            removed_syms = sorted(prev_syms - curr_syms)

            added = [
                {
                    "symbol": s,
                    "industry": curr_leaders[s]["industry"],
                    "cluster_rank": curr_leaders[s]["cluster_rank"],
                    "reason": generate_added_reason(s, curr_leaders[s]["industry"], curr_leaders[s]["cluster_rank"]),
                }
                for s in added_syms
            ]
            removed = [
                {
                    "symbol": s,
                    "industry": prev_leaders[s].get("industry", "Unknown") if isinstance(prev_leaders[s], dict) else str(prev_leaders[s]),
                    "reason": generate_removed_reason(
                        s,
                        prev_leaders[s].get("industry", "Unknown") if isinstance(prev_leaders[s], dict) else str(prev_leaders[s])
                    ),
                }
                for s in removed_syms
            ]

            return {
                "added": added,
                "removed": removed,
                "net_change": len(added) - len(removed),
                "has_comparison": True,
                "prev_date": prev_date,
                "curr_date": curr_date,
                "new_symbols_set": set(added_syms),
                "new_reasons_map": {x["symbol"]: x["reason"] for x in added},
                "total_leaders": len(curr_leaders),
            }
        else:
            return {
                "added": [],
                "removed": [],
                "net_change": 0,
                "has_comparison": False,
                "prev_date": None,
                "curr_date": curr_date,
                "new_symbols_set": set(),
                "new_reasons_map": {},
                "total_leaders": len(curr_leaders),
            }
    except Exception as e:
        print(f"Error in track_cluster_changes: {e}")
        return {
            "added": [],
            "removed": [],
            "net_change": 0,
            "has_comparison": False,
            "prev_date": None,
            "curr_date": "",
            "new_symbols_set": set(),
            "total_leaders": 0,
        }


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


def get_breadth_stock_picks(
    min_momo_score: int = 36,
    top_n: int = 30,
    prefer_leading_clusters: bool = True,
) -> list[dict]:
    """
    Returns stock picks based on multi-timeframe momentum (momo) data from Breadth Ledger,
    optionally biased toward stocks that are also in the leading industry clusters.

    Scoring system:
      - momo_score: sum of weights across 9 lookback periods (5d to 250d).
        Each period contributes (index+1) if the stock has positive momentum.
        Max possible = 1+2+3+4+5+6+7+8+9 = 45.
      - cluster_boost: +10 if stock is in a Top-5 leading cluster, +5 if Top-10.
      - final_score = momo_score + cluster_boost

    Returns list of dicts sorted by final_score descending.
    """
    data = fetch_breadth_data()
    momo = data.get("momo", {})
    lb = momo.get("lb", [])  # e.g. [5, 10, 20, 30, 60, 90, 125, 180, 250]
    groups = momo.get("groups", [])

    if not groups or not lb:
        return []

    clusters = get_leading_clusters()
    # Build cluster rank map: symbol -> (cluster_rank, cluster_score, industry)
    cluster_rank_map: dict[str, dict] = {}
    for rank, c in enumerate(clusters, 1):
        for s in c.get("leading_stocks", []):
            sym = str(s).strip().upper().replace(".NS", "")
            if sym and sym not in cluster_rank_map:
                cluster_rank_map[sym] = {
                    "cluster_rank": rank,
                    "cluster_score": c["score"],
                    "industry": c["industry"],
                    "highs_1m": c["highs_1m"],
                    "highs_3m": c["highs_3m"],
                    "highs_6m": c["highs_6m"],
                }

    picks = []
    for grp in groups:
        industry = grp.get("industry", "")
        for sym, code in grp.get("stocks", []):
            code_str = str(code)
            if len(code_str) < len(lb):
                continue
            # Calculate weighted momentum score
            momo_score = 0
            positive_tfs = []
            for i in range(len(lb)):
                ch = code_str[i] if i < len(code_str) else "0"
                if ch not in ("0", "-"):
                    weight = i + 1
                    momo_score += weight
                    positive_tfs.append(lb[i])

            if momo_score < min_momo_score:
                continue

            sym_clean = str(sym).strip().upper().replace(".NS", "")
            cl_info = cluster_rank_map.get(sym_clean, {})
            cluster_rank = cl_info.get("cluster_rank", 999)

            # Cluster boost
            if prefer_leading_clusters:
                if cluster_rank <= 5:
                    cluster_boost = 10
                elif cluster_rank <= 10:
                    cluster_boost = 5
                else:
                    cluster_boost = 0
            else:
                cluster_boost = 0

            final_score = momo_score + cluster_boost

            # Trend label based on how many timeframes are green
            n_green = len(positive_tfs)
            if n_green >= 8:
                trend_label = "🟢 Very Strong Uptrend"
                trend_color = "#00e676"
            elif n_green >= 6:
                trend_label = "🟢 Strong Uptrend"
                trend_color = "#4ade80"
            elif n_green >= 4:
                trend_label = "🟡 Moderate Momentum"
                trend_color = "#fbbf24"
            else:
                trend_label = "🟠 Early / Mixed"
                trend_color = "#fb923c"

            picks.append({
                "symbol": sym_clean,
                "industry": cl_info.get("industry", industry),
                "momo_score": momo_score,
                "cluster_boost": cluster_boost,
                "final_score": final_score,
                "in_leading_cluster": cluster_rank <= 10,
                "cluster_rank": cluster_rank,
                "cluster_score": cl_info.get("cluster_score", 0),
                "positive_tfs": positive_tfs,
                "n_green_tfs": n_green,
                "trend_label": trend_label,
                "trend_color": trend_color,
                "highs_1m": cl_info.get("highs_1m", 0),
                "highs_3m": cl_info.get("highs_3m", 0),
                "highs_6m": cl_info.get("highs_6m", 0),
            })

    # Sort by final_score desc, then by momo_score desc
    picks.sort(key=lambda x: (-x["final_score"], -x["momo_score"]))
    return picks[:top_n]

"""Pull bookmaker odds from The Odds API and store them in the odds_snapshots table.

Each pull is saved with a timestamp and a tag ('open', 'close' or 'snap'), so the table builds up
a price history for every match. Line movement and closing-line comparisons are worked out from
that history in engine/odds_analytics.py.

Prices from every bookmaker are kept, not just the best one. The 'eu' region is included mainly
to get Pinnacle, which is widely treated as the most accurate book and is used as the reference
for the "true" market probability elsewhere in the project.

The API's market names are mapped to the ones used here:
  h2h     -> '1x2'  sides home / draw / away
  totals  -> 'ou'   sides over / under, line = total goals (e.g. 2.5)
  spreads -> 'ah'   sides home / away,  line = that side's handicap

Needs an ODDS_API_KEY environment variable (free tier at the-odds-api.com). Without one it prints
a message and does nothing. Each request costs (number of markets) x (number of regions) credits,
so don't run it in a tight loop. The raw JSON responses are kept in data/raw/.

Run:  python soccer.py snapshot [open|close|snap]
"""
import os
import sys
import json
from pathlib import Path
import requests
import pandas as pd

from core import db
from core.names import canonical

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
QUOTA_FILE = RAW.parent / "api_quota.json"   # credits left as of the last call; the poller checks this
BASE = "https://api.the-odds-api.com/v4"


def read_quota():
    """Credits remaining / used as of the last odds request, or None if there hasn't been one."""
    if QUOTA_FILE.exists():
        try:
            return json.loads(QUOTA_FILE.read_text())
        except Exception:
            return None
    return None


def _record_quota(headers):
    """The API reports the remaining credits in its response headers; save them to disk."""
    rem = headers.get("x-requests-remaining")
    try:
        QUOTA_FILE.write_text(json.dumps({
            "remaining": int(rem) if rem is not None else None,
            "used": int(headers.get("x-requests-used", 0) or 0),
            "ts": pd.Timestamp.now("UTC").isoformat()}))
    except Exception:
        pass

# Competitions to pull. The API returns an empty list for any that aren't in season,
# so it's fine to leave them all here year-round.
SPORT_KEYS = [
    "soccer_fifa_world_cup",
    "soccer_uefa_nations_league",
    "soccer_uefa_european_championship",
    "soccer_conmebol_copa_america",
]


def fetch_odds(sport_key, regions="us,us2,eu", markets="h2h,totals,spreads", use_cache_on_fail=True):
    """One odds request for one competition. Returns the API's list of events (each with its
    bookmakers and prices), or an empty list if there is no key or the request fails. If the
    network is down and use_cache_on_fail is set, the last saved response is returned instead.
    """
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("ODDS_API_KEY not set -> no live odds. Set it to ingest.")
        return []
    url = f"{BASE}/sports/{sport_key}/odds/"
    try:
        r = requests.get(url, params={"apiKey": key, "regions": regions, "markets": markets,
                                      "oddsFormat": "decimal"}, timeout=25)
        if r.status_code != 200:
            print(f"[{sport_key}] odds api {r.status_code}: {r.text[:120]}")
            return []
        RAW.mkdir(parents=True, exist_ok=True)
        (RAW / f"odds_{sport_key}.json").write_bytes(r.content)
        _record_quota(r.headers)
        print(f"[{sport_key}] {len(r.json())} events  (requests remaining: {r.headers.get('x-requests-remaining')})")
        return r.json()
    except Exception as e:
        cache = RAW / f"odds_{sport_key}.json"
        if use_cache_on_fail and cache.exists():
            print(f"[{sport_key}] fetch failed ({e}); using cache")
            return json.loads(cache.read_text())
        print(f"[{sport_key}] fetch failed: {e}")
        return []


def _side_for(market_key, outcome, home, away):
    """Translate one outcome from the API into this project's (market, side, line)."""
    name = outcome.get("name")
    pt = outcome.get("point")
    if market_key == "h2h":
        side = "home" if name == home else "away" if name == away else "draw" if name == "Draw" else None
        return ("1x2", side, 0.0)
    if market_key == "totals":
        side = "over" if name == "Over" else "under" if name == "Under" else None
        return ("ou", side, float(pt) if pt is not None else None)
    if market_key == "spreads":
        side = "home" if name == home else "away" if name == away else None
        return ("ah", side, float(pt) if pt is not None else None)
    return (None, None, None)


def flatten_events(events, snapshot_ts, tag):
    """Flatten the nested API response into one row per match / book / market / line / side."""
    rows = []
    for ev in events:
        home, away = ev.get("home_team"), ev.get("away_team")
        ch, ca = canonical(home), canonical(away)
        commence = ev.get("commence_time", "")
        # Kick-off times are UTC, so a late local kick-off can land on the next calendar day.
        # Anything joining odds to results allows +/- one day for that.
        mk = db.match_key(commence, ch, ca)
        for bk in ev.get("bookmakers", []):
            book = bk.get("key")
            for m in bk.get("markets", []):
                for oc in m.get("outcomes", []):
                    market, side, line = _side_for(m.get("key"), oc, home, away)
                    price = oc.get("price")
                    if market and side and price and line is not None:
                        rows.append({
                            "snapshot_ts": snapshot_ts, "tag": tag, "match_key": mk,
                            "market": market, "line": line, "side": side, "book": book,
                            "decimal_odds": float(price),
                        })
    return rows


def snapshot(tag="snap", sport_keys=None, regions="us,us2,eu", markets="h2h,totals,spreads"):
    """Pull odds for every competition in sport_keys and store them under one timestamp and tag."""
    sport_keys = sport_keys or SPORT_KEYS
    ts = pd.Timestamp.now("UTC").isoformat()
    all_rows = []
    for sk in sport_keys:
        all_rows += flatten_events(fetch_odds(sk, regions, markets), ts, tag)
    if not all_rows:
        print("no odds rows to store.")
        return 0
    conn = db.connect()
    db.init_db(conn)
    n = db.upsert(conn, "odds_snapshots", all_rows,
                  ["snapshot_ts", "match_key", "market", "line", "side", "book"])
    # print how many matches we got prices for in each market
    df = pd.DataFrame(all_rows)
    by_market = df.groupby("market")["match_key"].nunique().to_dict()
    print(f"stored {n} odds rows @ {ts} (tag={tag}) | books={df['book'].nunique()} "
          f"| matches by market: {by_market}")
    conn.close()
    return n


def join_coverage():
    """Count how many matches in the odds table can be found in the results table. A low number
    means team names aren't lining up between the two sources (or the fixtures simply haven't
    been ingested yet)."""
    conn = db.connect()
    odds = pd.read_sql("SELECT DISTINCT match_key FROM odds_snapshots", conn)
    matched = 0
    for mk in odds["match_key"]:
        d, h, a = mk.split("|")
        # same teams within a day either side, to allow for the UTC date shift
        q = ("SELECT COUNT(*) FROM matches WHERE home_team=? AND away_team=? "
             "AND ABS(julianday(date)-julianday(?))<=1")
        if conn.execute(q, (h, a, d)).fetchone()[0] > 0:
            matched += 1
    print(f"odds->matches join: {matched}/{len(odds)} distinct odds matches found in results table "
          f"(rest are future fixtures not yet ingested, or name gaps)")
    conn.close()
    return matched, len(odds)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "snap"
    snapshot(tag=tag)
    join_coverage()
    return 0


if __name__ == "__main__":
    sys.exit(main())

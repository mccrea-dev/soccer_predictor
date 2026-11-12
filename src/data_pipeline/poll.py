"""Scheduled odds collection that stays inside the API's free tier.

The Odds API charges credits for odds requests but the fixture list is free. So each run checks
the fixture list first and only pays for an odds request when some match has just come into one
of two windows and hasn't been captured for it yet:

  open   18 to 30 hours before kick-off
  close  30 minutes to 2.5 hours before kick-off

That works out at about two paid requests per match, four credits each, so a whole tournament fits
within the 500 free credits a month. A run that finds nothing due costs nothing, which means it
can be scheduled every hour or two (cron, Task Scheduler) without thinking about it.

After a paid pull it also runs the price comparison in engine/market_ev.py and prints anything
that clears the alert threshold.

Run:  python soccer.py poll [poll|quota|events]
"""
import os
import sys
import requests
import pandas as pd

from core import db
from core.names import canonical
from data_pipeline.ingest_odds import (fetch_odds, flatten_events, SPORT_KEYS, read_quota, BASE)
from engine.market_ev import opportunities

OPEN_WINDOW = (18, 30)        # hours before kick-off in which to take the 'open' snapshot
CLOSE_WINDOW = (0.5, 2.5)     # hours before kick-off in which to take the 'close' snapshot
MIN_RESERVE = 25              # stop making paid requests once credits drop below this
ALERT_EV = 0.03               # print an alert when a price is at least this much better than fair
POLL_MARKETS = "h2h,totals"   # two markets...
POLL_REGIONS = "us,eu"        # ...times two regions (US books plus Pinnacle) = 4 credits a request
ALERTS_LOG = db.DB_PATH.parent / "alerts.log"


def fetch_events(sport_key):
    """Upcoming fixtures for one competition (teams and kick-off time). This endpoint is free."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return []
    try:
        r = requests.get(f"{BASE}/sports/{sport_key}/events",
                         params={"apiKey": key}, timeout=20)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def _has_snapshot(conn, match_key, tag):
    """Have we already stored a snapshot with this tag for this match?"""
    return conn.execute("SELECT 1 FROM odds_snapshots WHERE match_key=? AND tag=? LIMIT 1",
                        (match_key, tag)).fetchone() is not None


def _due(conn, events):
    """Split upcoming fixtures into those that need an 'open' snapshot now and those that need
    a 'close' one. Matches already captured for their window are left out."""
    now = pd.Timestamp.now("UTC")
    open_due, close_due = set(), set()
    for e in events:
        ct = pd.Timestamp(e["commence_time"])
        hrs = (ct - now).total_seconds() / 3600.0
        mk = db.match_key(ct.isoformat(), canonical(e.get("home_team")), canonical(e.get("away_team")))
        if CLOSE_WINDOW[0] <= hrs <= CLOSE_WINDOW[1] and not _has_snapshot(conn, mk, "close"):
            close_due.add(mk)
        elif OPEN_WINDOW[0] <= hrs <= OPEN_WINDOW[1] and not _has_snapshot(conn, mk, "open"):
            open_due.add(mk)
    return open_due, close_due


def _alert(opp):
    """Print (and append to data/alerts.log) any opportunity at or above the alert threshold."""
    hot = opp[opp["ev"] >= ALERT_EV] if not opp.empty else opp
    if hot.empty:
        print("scan: no prices >= {:.0%} better than the sharp line right now.".format(ALERT_EV))
        return
    lines = ["", "ALERT: prices better than the sharp line:"]
    for _, r in hot.iterrows():
        lines.append(f"  {r['home']} v {r['away']} | {r['market']} {r['side']} @ {r['price']} "
                     f"({r['book']})  fair {r['fair_p']*100:.0f}%  EV {r['ev']*100:+.1f}%  ¼K {r['kelly']*100:.2f}%")
    msg = "\n".join(lines)
    print(msg)
    try:
        with open(ALERTS_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"\n[{pd.Timestamp.now('UTC').isoformat()}]{msg}\n")
    except Exception:
        pass
    print(f"  (also appended to {ALERTS_LOG.name})")


def poll(sport_keys=None, markets=POLL_MARKETS, regions=POLL_REGIONS, min_reserve=MIN_RESERVE):
    """One polling run: free fixture check, then a paid odds pull only if something is due and
    there are credits to spare, then the price scan.
    """
    sport_keys = sport_keys or SPORT_KEYS
    conn = db.connect(); db.init_db(conn)

    # 1) free fixture check: does any competition have a match that needs a snapshot?
    due = {}
    for sk in sport_keys:
        evs = fetch_events(sk)
        if not evs:
            continue
        od, cd = _due(conn, evs)
        if od or cd:
            due[sk] = (od, cd)
    if not due:
        print("nothing due (no match in an open/close window lacking a snapshot). 0 credits spent.")
        conn.close(); return 0

    # 2) don't spend credits if we're nearly out
    q = read_quota()
    if q and q.get("remaining") is not None and q["remaining"] < min_reserve:
        print(f"QUOTA GUARD: only {q['remaining']} requests left (< {min_reserve}); skipping paid call.")
        conn.close(); return 0

    # 3) one paid request per competition with something due. The response covers every match in
    #    that competition, so store all of it and tag the due ones 'open' or 'close'.
    ts = pd.Timestamp.now("UTC").isoformat()
    rows = []
    for sk, (od, cd) in due.items():
        events = fetch_odds(sk, regions, markets, use_cache_on_fail=False)
        r = flatten_events(events, ts, "snap")
        for row in r:
            if row["match_key"] in cd:
                row["tag"] = "close"
            elif row["match_key"] in od:
                row["tag"] = "open"
        rows += r
        print(f"[{sk}] due open={len(od)} close={len(cd)} -> snapshot stored "
              f"({markets} x {regions} = {len(markets.split(','))*len(regions.split(','))} credits)")
    if rows:
        db.upsert(conn, "odds_snapshots", rows,
                  ["snapshot_ts", "match_key", "market", "line", "side", "book"])

    # 4) compare prices across books and report anything notable
    _alert(opportunities(conn))
    rem = read_quota()
    print(f"remaining requests: {rem.get('remaining') if rem else '??'}")
    conn.close(); return 0


def quota():
    """Print the API credits left as of the last paid request."""
    q = read_quota()
    if not q:
        print("no quota record yet (make one odds call first).")
        return
    print(f"Odds API: {q.get('remaining')} remaining, {q.get('used')} used  (as of {q.get('ts')})")


def events_preview():
    """List upcoming fixtures with hours to kick-off and whether each is currently in its open
    or close window. Uses only the free endpoint."""
    now = pd.Timestamp.now("UTC")
    for sk in SPORT_KEYS:
        evs = fetch_events(sk)
        for e in sorted(evs, key=lambda x: x["commence_time"])[:20]:
            hrs = (pd.Timestamp(e["commence_time"]) - now).total_seconds() / 3600.0
            win = ("CLOSE" if CLOSE_WINDOW[0] <= hrs <= CLOSE_WINDOW[1]
                   else "OPEN" if OPEN_WINDOW[0] <= hrs <= OPEN_WINDOW[1] else "")
            print(f"  {e['commence_time']}  ({hrs:5.1f}h)  {e['home_team']} v {e['away_team']}  {win}")


def main():
    {"poll": poll, "quota": quota, "events": events_preview}.get(
        sys.argv[1] if len(sys.argv) > 1 else "poll", poll)()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Compare prices across bookmakers, no model involved.

Once it was clear the goal model doesn't out-predict the betting market, the more interesting
question became whether the books disagree with each other. Pinnacle is generally the most
accurate, and the big consumer sportsbooks sometimes lag behind it when a line moves. This script
looks for exactly that.

For each market, line and side:
  1. take Pinnacle's prices and remove the margin to get a "fair" probability
     (median across books if Pinnacle isn't quoting)
  2. for every consumer book offering that side at decimal odds d, compute
     EV per $1 = fair probability * d - 1
  3. list the ones where that comes out positive by at least MIN_EV

Things to keep in mind when reading the output:
  * it's only as good as the reference. Rows marked ref=median are much weaker evidence.
  * a book that's out of line usually corrects within minutes, so most of what shows up here is
    a stale price that is about to move.
  * longshots are filtered out, because small errors in the fair probability turn into large
    apparent edges at long odds.

Run:  python soccer.py scan
"""
import sys
import pandas as pd

from core import db
from core.ev import decimal_to_prob, devig, ev_per_dollar, fractional_kelly

SHARP = "pinnacle"   # reference book for the fair price; never one of the books being compared
MIN_EV = 0.02        # smallest EV worth listing
KF, KCAP = 0.25, 0.05   # quarter Kelly, capped at 5% of bankroll
# The consumer books whose prices get compared with the reference: regulated US sportsbooks, by
# their Odds API keys ('williamhill_us' is Caesars). Exchanges and offshore books are ignored even
# when they show a better price.
US_BOOKS = {"draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers",
              "espnbet", "fanatics", "hardrockbet", "ballybet", "fliff"}
MAX_PRICE = 6.0      # skip anything priced longer than this...
MIN_FAIR = 0.06      # ...or that the reference rates below a 6% chance


def _latest(conn):
    """Every match's most recent snapshot, all books and markets."""
    df = pd.read_sql("SELECT * FROM odds_snapshots", conn)
    if df.empty:
        return df
    latest = df.groupby("match_key")["snapshot_ts"].transform("max")
    return df[df["snapshot_ts"] == latest]


def _pick_sharp(prices):
    """Reference price for one side: Pinnacle's if it has one, otherwise the median across
    books. Returns (price, 'pinnacle' or 'median')."""
    if prices.empty:
        return None, None
    if SHARP in prices.index:
        return float(prices[SHARP]), "pinnacle"
    return float(prices.median()), "median"


def _fair_two_way(a, b):
    """Fair probabilities for the two sides of a market, plus which reference they came from."""
    va, ra = _pick_sharp(a)
    vb, rb = _pick_sharp(b)
    if va is None or vb is None:
        return None
    fair = devig([decimal_to_prob(va), decimal_to_prob(vb)], method="power")
    return float(fair[0]), float(fair[1]), ("pinnacle" if ra == rb == "pinnacle" else "median")


def _fair_three_way(d):
    """Fair (home, draw, away) probabilities, plus which reference they came from."""
    picks = [_pick_sharp(d[s]) for s in ("home", "draw", "away")]
    if any(v is None for v, _ in picks):
        return None
    fair = devig([decimal_to_prob(v) for v, _ in picks], method="power")   # power method, see core.ev.devig
    ref = "pinnacle" if all(r == "pinnacle" for _, r in picks) else "median"
    return [float(x) for x in fair], ref


def _scan_side(rows, match, mkt_label, side, fair_p, ref, out):
    """Add a row to `out` for every consumer book whose price on this side is at least MIN_EV
    better than fair. `rows` maps book -> decimal odds. Longshots are skipped."""
    if fair_p < MIN_FAIR:
        return
    for book, dec in rows.items():
        if book not in US_BOOKS or dec > MAX_PRICE:
            continue
        ev = ev_per_dollar(fair_p, dec)
        if ev >= MIN_EV:
            out.append({**match, "market": mkt_label, "side": side, "book": book,
                        "price": round(float(dec), 2), "fair_p": round(fair_p, 3),
                        "ev": round(ev, 3), "ref": ref,
                        "kelly": round(fractional_kelly(fair_p, dec, KF, KCAP), 4)})


def opportunities(conn, min_ev=MIN_EV, future_only=True):
    """Scan the latest snapshot of every match (upcoming ones only, by default) across
    over/under, match result and Asian handicap. Returns a DataFrame sorted by EV.
    """
    df = _latest(conn)
    if df.empty:
        return pd.DataFrame()
    today = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    out = []
    for mk, m in df.groupby("match_key"):
        date, home, away = mk.split("|")
        if future_only and date < today:      # match already played, its odds are history
            continue
        match = {"match_key": mk, "home": home, "away": away}

        # over/under, each goal line separately
        for line, ml in m[m["market"] == "ou"].groupby("line"):
            over = ml[ml["side"] == "over"].set_index("book")["decimal_odds"]
            under = ml[ml["side"] == "under"].set_index("book")["decimal_odds"]
            fair = _fair_two_way(over, under)
            if fair:
                _scan_side(over, match, f"OU {line} O", "over", fair[0], fair[2], out)
                _scan_side(under, match, f"OU {line} U", "under", fair[1], fair[2], out)

        # match result
        m1 = m[m["market"] == "1x2"]
        if not m1.empty:
            by = {s: m1[m1["side"] == s].set_index("book")["decimal_odds"] for s in ("home", "draw", "away")}
            f3 = _fair_three_way(by)
            if f3:
                fair, ref = f3
                for i, s in enumerate(("home", "draw", "away")):
                    _scan_side(by[s], match, f"1X2 {s}", s, fair[i], ref, out)

        # Asian handicap: home at line L is the other side of away at line -L
        ah = m[m["market"] == "ah"]
        for L in sorted(ah[ah["side"] == "home"]["line"].unique()):
            hp = ah[(ah["side"] == "home") & (ah["line"] == L)].set_index("book")["decimal_odds"]
            ap = ah[(ah["side"] == "away") & (ah["line"] == -L)].set_index("book")["decimal_odds"]
            fair = _fair_two_way(hp, ap)
            if fair:
                _scan_side(hp, match, f"AH {L:+g} H", "home", fair[0], fair[2], out)
                _scan_side(ap, match, f"AH {-L:+g} A", "away", fair[1], fair[2], out)

    cols = ["home", "away", "market", "side", "book", "price", "fair_p", "ev", "kelly", "ref", "match_key"]
    return (pd.DataFrame(out)[cols].sort_values("ev", ascending=False) if out else pd.DataFrame(columns=cols))


def main():
    conn = db.connect()
    present = {r[0] for r in conn.execute("SELECT DISTINCT book FROM odds_snapshots")}
    books_seen = sorted(US_BOOKS & present)
    opp = opportunities(conn)
    print("=" * 104)
    print("PRICE COMPARISON  -- US books priced better than Pinnacle's de-vigged line")
    print(f"US books in the latest data: {books_seen or 'none'}")
    print("EV per $1 = fair probability * price - 1")
    print("=" * 104)
    if opp.empty:
        print("nothing at or above 2% EV in this snapshot, which is the usual result.")
        conn.close(); return 0
    print(f"{'match':30s} {'market':10s} {'book':12s} {'price':>6s} {'fair':>6s} {'EV':>7s} {'¼K':>6s} {'ref':>8s}")
    for _, r in opp.head(30).iterrows():
        print(f"{(r['home'][:13]+' v '+r['away'][:13]):30s} {r['market']:10s} {r['book']:12s} "
              f"{r['price']:6.2f} {r['fair_p']*100:5.1f}% {r['ev']*100:+6.1f}% {r['kelly']*100:5.2f}% {r['ref']:>8s}")
    pin = opp[opp["ref"] == "pinnacle"]
    print("-" * 104)
    print(f"{len(opp)} prices >= 2% EV | {len(pin)} of them against a Pinnacle reference "
          f"| books: {sorted(opp['book'].unique())}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

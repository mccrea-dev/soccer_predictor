"""Paper-trading log: record the over/under picks where the model disagrees with the market, then
see how they did.

No money is involved. The point is to answer one question: when the model thinks the market has
a total wrong, who turns out to be right?

  log      record every over/under pick whose edge clears a threshold, at the best price available
  clv      once later odds have been collected, compare the price taken with the closing price
  results  once the matches are played, check which picks won and what a flat stake would
           have returned

Closing line value needs a second odds snapshot taken near kick-off. With only one snapshot the
"closing" price is the same as the one taken, and CLV comes out as zero for everything.

Run:  python soccer.py [log|clv|results]
"""
import sys
import pandas as pd

from core import db
from engine.odds_analytics import closing_line, clv_bps
from engine.pricing import price_match, load_ou_calibrator, OU_LINE
from models.goals_statistical import DixonColes, fit_from_db


def _model():
    """Load the saved goal model, fitting and saving one first if there isn't one yet."""
    try:
        return DixonColes.load()
    except Exception:
        dc = fit_from_db(); dc.save(); return dc


def log_bet(conn, ts, match_key, market, line, side, book, odds_taken, model_p, fair_p, edge, stake):
    """Insert one hypothetical bet into bets_log."""
    conn.execute(
        "INSERT INTO bets_log (ts, match_key, market, line, side, book, odds_taken, model_p, "
        "fair_p, edge, stake) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (ts, match_key, market, line, side, book, odds_taken, model_p, fair_p, edge, stake))
    conn.commit()


def log_board(min_edge=0.03, bankroll=1000.0):
    """Log an over/under pick for every match where the calibrated model is at least min_edge
    above the market. Stakes are sized by quarter Kelly on a notional bankroll."""
    conn = db.connect()
    dc, cal = _model(), load_ou_calibrator()
    ts = pd.Timestamp.now("UTC").isoformat()
    mks = [r[0] for r in conn.execute("SELECT DISTINCT match_key FROM odds_snapshots WHERE market='ou'")]
    n = 0
    for mk in mks:
        r = price_match(conn, dc, mk, cal)
        if not r or "ou" not in r:
            continue
        o = r["ou"]
        if o["edge"] >= min_edge:
            log_bet(conn, ts, mk, "ou", OU_LINE, o["side"].lower(), "best",
                    o["price"], o["model_p"], o["fair_p"], o["edge"], o["kelly"] * bankroll)
            n += 1
    print(f"logged {n} O/U board picks (edge>={min_edge:.0%}) to bets_log")
    conn.close()
    return n


def grade_clv():
    """Fill in the closing price and CLV for logged bets that don't have them yet."""
    conn = db.connect()
    bets = pd.read_sql("SELECT * FROM bets_log WHERE clv_bps IS NULL", conn)
    if bets.empty:
        print("no ungraded bets."); conn.close(); return
    graded = 0
    for _, b in bets.iterrows():
        cl = closing_line(conn, b["match_key"], b["market"], b["line"], b["side"])
        if cl is None:
            continue
        bps = clv_bps(b["odds_taken"], cl)
        conn.execute("UPDATE bets_log SET closing_odds=?, clv_bps=? WHERE id=?", (cl, bps, b["id"]))
        graded += 1
    conn.commit()
    g = pd.read_sql("SELECT * FROM bets_log WHERE clv_bps IS NOT NULL", conn)
    if len(g):
        print(f"graded {graded} bets vs close | beat-close {(g['clv_bps']>0).mean()*100:.1f}% | "
              f"avg CLV {g['clv_bps'].mean():+.0f} bps")
        print("  (beating the close more than half the time, with average CLV above zero, is the goal)")
        if (g["clv_bps"].abs() < 1).all():
            print("  note: CLV is ~0 for every bet, which means there is only one snapshot per match.")
            print("        Take another near kick-off (`python soccer.py snapshot close`) and re-run.")
    conn.close()


def _actual_total(conn, match_key):
    """Total goals in a played match (date matched to within a day), or None if no result yet."""
    d, h, a = match_key.split("|")
    row = conn.execute(
        "SELECT home_goals, away_goals FROM matches WHERE home_team=? AND away_team=? "
        "AND ABS(julianday(date)-julianday(?))<=1 AND home_goals IS NOT NULL LIMIT 1",
        (h, a, d)).fetchone()
    return (row[0] + row[1]) if row else None


def grade_results():
    """Did the logged picks win? Prints the hit rate and the return from staking one unit on
    each at the price taken. With a few dozen picks this is mostly noise, so read it as a
    sanity check and not as proof of anything."""
    conn = db.connect()
    bets = pd.read_sql("SELECT * FROM bets_log WHERE market='ou'", conn)
    if bets.empty:
        print("no logged O/U bets to grade. Run `log` first."); conn.close(); return
    rows = []
    for _, b in bets.iterrows():
        tot = _actual_total(conn, b["match_key"])
        if tot is None:
            continue
        won = (b["side"] == "under" and tot < b["line"]) or (b["side"] == "over" and tot > b["line"])
        push = tot == b["line"]
        pnl = 0.0 if push else (b["odds_taken"] - 1 if won else -1.0)
        rows.append({"match": b["match_key"].split("|", 1)[1].replace("|", " v "),
                     "pick": b["side"], "edge": b["edge"], "odds": b["odds_taken"],
                     "total": tot, "won": int(won), "pnl": pnl})
    if not rows:
        print("none of the logged picks have a result yet. Run ingest-results and try again.")
        conn.close(); return
    d = pd.DataFrame(rows)
    roi = d["pnl"].sum() / len(d) * 100
    print(f"OUTCOME CHECK on {len(d)} logged O/U picks (small sample):")
    print(f"  hit rate {d['won'].mean()*100:.1f}%  | flat ROI @ taken price {roi:+.1f}%  "
          f"| total staked {len(d)} units, pnl {d['pnl'].sum():+.2f}")
    print(d.sort_values("edge", ascending=False).to_string(index=False))
    print("\nIf the picks with the biggest edge mostly lost, the 'edge' was the model being wrong,")
    print("not the market.")
    conn.close()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "log"
    {"log": log_board, "clv": grade_clv, "results": grade_results}.get(cmd, log_board)()
    return 0


if __name__ == "__main__":
    sys.exit(main())

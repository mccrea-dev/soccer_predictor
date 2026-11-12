"""Model vs. market: line the goal model's probabilities up against bookmaker prices.

For each match with odds stored, this works out

    edge = model probability - market probability

where the market probability has the bookmaker margin removed and comes from Pinnacle when
available (otherwise the median across books). The price shown is the best one on offer across
all books, since that's what you would actually get.

A positive edge only means the model disagrees with the market. It is not evidence the model is
right, and usually it isn't: the market sees team news and a lot of money that the model doesn't.
Treat the board as a way to see where and by how much the two differ.

Run:  python soccer.py board
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from core import db
from core.ev import devig_two_way, devig_multi, ev_per_dollar, fractional_kelly
from models.goals_statistical import DixonColes, fit_from_db

MODELS = Path(__file__).resolve().parents[2] / "models"
OU_LINE = 2.5
KELLY_FRACTION = 0.25
KELLY_CAP = 0.05


def load_ou_calibrator():
    """The saved over/under calibrator (made by `python soccer.py calibrate`), or None if it
    hasn't been fitted, in which case raw model probabilities are used."""
    p = MODELS / "ou_calibrator.pkl"
    return joblib.load(p) if p.exists() else None


def _cal_over(cal, p):
    """Apply the calibrator to one probability, or pass it through if there is no calibrator."""
    return float(cal.predict([p])[0]) if cal is not None else float(p)


def _latest_odds(conn, match_key):
    """All prices for a match from its most recent snapshot."""
    df = pd.read_sql("SELECT * FROM odds_snapshots WHERE match_key=?", conn, params=[match_key])
    if df.empty:
        return df
    return df[df["snapshot_ts"] == df["snapshot_ts"].max()]


def _price(df, market, line, side, book=None, agg="max"):
    """One decimal price for a selection: the best across books by default (agg='max'), the
    median with agg='median', or a single book's if `book` is given. None if nobody quotes it.
    """
    g = df[(df["market"] == market) & (df["line"] == line) & (df["side"] == side)]
    if book is not None:
        g = g[g["book"] == book]
    if g.empty:
        return None
    return float(g["decimal_odds"].max() if agg == "max" else g["decimal_odds"].median())


def _fair_two_way(df, line, over="over", under="under"):
    """Market P(over) with the margin removed, from Pinnacle if it quotes both sides and from
    the median price across books if not."""
    po, pu = _price(df, "ou", line, over, "pinnacle"), _price(df, "ou", line, under, "pinnacle")
    if po is None or pu is None:
        po, pu = _price(df, "ou", line, over, agg="median"), _price(df, "ou", line, under, agg="median")
    if po is None or pu is None:
        return None
    return devig_two_way(po, pu)[0]


def _fair_1x2(df):
    """Market (home, draw, away) probabilities with the margin removed, Pinnacle first, else median."""
    src = "pinnacle" if not df[(df["market"] == "1x2") & (df["book"] == "pinnacle")].empty else None
    o = [(_price(df, "1x2", 0.0, s, src) if src else _price(df, "1x2", 0.0, s, agg="median"))
         for s in ("home", "draw", "away")]
    if any(x is None for x in o):
        return None
    return devig_multi(o)


def get_neutral(conn, date, home, away):
    """Whether a match is at a neutral venue, looked up in the results data (+/- a day). If the
    match isn't there yet, assume neutral, which is right for most tournament games.
    """
    row = conn.execute(
        "SELECT neutral FROM matches WHERE home_team=? AND away_team=? "
        "AND ABS(julianday(date)-julianday(?))<=1 LIMIT 1", (home, away, date)).fetchone()
    return int(row[0]) if row else 1   # default neutral for tournament venues


def price_match(conn, dc, match_key, cal=None):
    """Compare model and market for one match. Returns a dict with an 'ou' entry (over/under 2.5)
    and a '1x2' entry (match result), each holding the side where the model is most above the
    market along with both probabilities, the best price, EV and a suggested stake.
    """
    df = _latest_odds(conn, match_key)
    if df.empty:
        return None
    date, home, away = match_key.split("|")
    neutral = get_neutral(conn, date, home, away)
    out = {"match_key": match_key, "home": home, "away": away, "neutral": neutral}

    # ---- over/under 2.5 ----
    fair_over = _fair_two_way(df, OU_LINE)
    if fair_over is not None:
        model_over = _cal_over(cal, dc.prob_over(home, away, OU_LINE, neutral))
        best_over = _price(df, "ou", OU_LINE, "over")
        best_under = _price(df, "ou", OU_LINE, "under")
        if best_over and best_under:
            cand = []
            for side, mp, fp, price in [("OVER", model_over, fair_over, best_over),
                                        ("UNDER", 1 - model_over, 1 - fair_over, best_under)]:
                cand.append({"side": side, "model_p": mp, "fair_p": fp, "price": price,
                             "edge": mp - fp, "ev": ev_per_dollar(mp, price),
                             "kelly": fractional_kelly(mp, price, KELLY_FRACTION, KELLY_CAP)})
            out["ou"] = max(cand, key=lambda c: c["ev"])

    # ---- match result ----
    fair = _fair_1x2(df)
    if fair is not None:
        mk = dc.markets(home, away, neutral)
        model3 = [mk["home"], mk["draw"], mk["away"]]
        best = [_price(df, "1x2", 0.0, s) for s in ("home", "draw", "away")]
        cand = []
        for i, lab in enumerate(("HOME", "DRAW", "AWAY")):
            if best[i]:
                cand.append({"side": lab, "model_p": model3[i], "fair_p": float(fair[i]),
                             "price": best[i], "edge": model3[i] - float(fair[i]),
                             "ev": ev_per_dollar(model3[i], best[i]),
                             "kelly": fractional_kelly(model3[i], best[i], KELLY_FRACTION, KELLY_CAP)})
        if cand:
            out["1x2"] = max(cand, key=lambda c: c["edge"])
    return out


def main():
    conn = db.connect()
    try:
        dc = DixonColes.load()
    except Exception:
        print("[info] no saved model -- fitting from DB ...")
        dc = fit_from_db(); dc.save()

    cal = load_ou_calibrator()
    matches = [r[0] for r in conn.execute(
        "SELECT DISTINCT match_key FROM odds_snapshots WHERE market='ou'").fetchall()]
    rows = [price_match(conn, dc, mk, cal) for mk in matches]
    rows = [r for r in rows if r and "ou" in r]

    print("=" * 100)
    print("MODEL vs MARKET  -- Over/Under 2.5 goals  (edge = model prob - de-vigged market prob; price = best across books)")
    print(f"calibration: {'isotonic (held-out)' if cal is not None else 'none -- run `python soccer.py calibrate`'}")
    print("A positive edge means the model disagrees with the market, not that the model is right.")
    print("=" * 100)
    print(f"{'match':34s} {'pick':10s} {'model':>6s} {'fair':>6s} {'edge':>7s} {'price':>6s} {'EV/$1':>7s} {'¼Kelly':>7s}")
    for r in sorted(rows, key=lambda x: -x["ou"]["edge"]):
        o = r["ou"]
        tag = "  <<" if o["edge"] > 0.03 else ""
        print(f"{(r['home'][:15]+' v '+r['away'][:15]):34s} {('O/U '+o['side']):10s} "
              f"{o['model_p']*100:5.1f}% {o['fair_p']*100:5.1f}% {o['edge']*100:+6.1f}% "
              f"{o['price']:6.2f} {o['ev']:+6.2f} {o['kelly']*100:5.2f}%{tag}")

    pos = [r for r in rows if r["ou"]["edge"] > 0.03]
    print("-" * 100)
    print(f"{len(rows)} matches priced | {len(pos)} where the model is more than 3% above the market")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Forecast a match: who wins, likely scorelines, and whether it will be high-scoring.

Two models share the work. The winner model supplies the home / draw / away probabilities, and the
Dixon-Coles goal model supplies everything that needs a score distribution: expected goals, the
most likely scorelines, over/under 2.5 and both-teams-to-score. If odds have been collected for a
match, the market's implied probabilities are printed underneath for comparison.

Run:
  python soccer.py predict                         # every upcoming fixture
  python soccer.py predict "Brazil" "Argentina"    # one matchup
  python soccer.py predict "Brazil" "Argentina" 1  # same, at a neutral venue
"""
import sys
import numpy as np
import pandas as pd

from core import db
from core.distributions import prob_1x2, prob_over_under, prob_btts
from core.ev import decimal_to_prob, devig
from core.names import resolve_name
from models.goals_statistical import DixonColes, fit_from_db
from models.winner import fit_and_predict
from engine.pricing import _latest_odds, get_neutral, load_ou_calibrator


# Where the over/under pick flips from Under to Over. You'd expect 0.50, but the calibrated
# probabilities bunch up just below it, and with a 0.50 cut-off the model called Over on only
# 20% of held-out games when about half of them actually went over. On 2,663 held-out games the
# most accurate cut-off was around 0.46 (anything from 0.44 to 0.48 was about as good, 55.8%
# correct). Only the pick uses this; the probability shown is left as it is.
OU_PICK_THRESHOLD = 0.46


def _model():
    """Load the saved goal model, fitting and saving one first if there isn't one yet."""
    try:
        return DixonColes.load()
    except Exception:
        dc = fit_from_db(); dc.save(); return dc


def _market_1x2(df):
    """The market's (home, draw, away) probabilities with the bookmaker margin removed. Uses
    Pinnacle's prices when they are in the snapshot, otherwise the median across books."""
    m1 = df[df["market"] == "1x2"]
    if m1.empty:
        return None
    vals = []
    for s in ("home", "draw", "away"):
        g = m1[m1["side"] == s].set_index("book")["decimal_odds"]
        if g.empty:
            return None
        vals.append(g["pinnacle"] if "pinnacle" in g.index else g.median())
    fair = devig([decimal_to_prob(v) for v in vals], method="power")
    return float(fair[0]), float(fair[1]), float(fair[2])


def forecast(dc, home, away, neutral=0, cal=None, win=None):
    """Everything we predict about one match, as a dict.

    `win` is the winner model's (home, draw, away) probabilities. If it's missing, the goal
    model's own are used as a fallback. `cal` is the over/under calibrator, also optional.
    Scorelines, over/under and both-teams-to-score always come from the goal model `dc`."""
    mtx = dc.score_matrix(home, away, neutral)
    lam, mu = dc.lambda_mu(home, away, neutral)
    h, d, a = win if win is not None else prob_1x2(mtx)
    over = prob_over_under(mtx, 2.5)[0]
    if cal is not None:
        over = float(cal.predict([over])[0])
    btts = prob_btts(mtx)[0]
    # the three most likely scorelines, as (home goals, away goals, probability)
    order = np.argsort(mtx.ravel())[::-1][:3]
    top = [(int(i), int(j), float(mtx[i, j])) for i, j in (np.unravel_index(o, mtx.shape) for o in order)]
    probs = {"home": h, "draw": d, "away": a}
    pick = max(probs, key=probs.get)
    winner = home if pick == "home" else away if pick == "away" else "Draw"
    return {"home": home, "away": away, "neutral": neutral, "lam": lam, "mu": mu,
            "h": h, "d": d, "a": a, "over": over, "btts": btts, "top": top,
            "winner": winner, "conf": max(h, d, a),
            "ou_pick": "Over" if over >= OU_PICK_THRESHOLD else "Under"}


def print_card(f, market=None):
    """Print one forecast as a short block of text. `market` is optional (home, draw, away) probabilities."""
    venue = "  (neutral)" if f["neutral"] else ""
    s1 = f["top"][0]
    print(f"\n{f['home']}  vs  {f['away']}{venue}")
    print(f"  PICK: {f['winner']}   ({f['conf']*100:.0f}% confidence)")
    print(f"  Win:  {f['home']} {f['h']*100:.0f}%   |   Draw {f['d']*100:.0f}%   |   {f['away']} {f['a']*100:.0f}%")
    print(f"  Likely score: {s1[0]}-{s1[1]}"
          + "".join(f"   {t[0]}-{t[1]} ({t[2]*100:.0f}%)" for t in f["top"][:3]))
    print(f"  xGoals: {f['lam']:.1f} - {f['mu']:.1f}   |   O/U 2.5: {f['ou_pick']} "
          f"({f['over']*100:.0f}% over)   |   BTTS: {f['btts']*100:.0f}%")
    if market:
        print(f"  Market:  {f['home']} {market[0]*100:.0f}%   |   Draw {market[1]*100:.0f}%   |   {f['away']} {market[2]*100:.0f}%")


def main():
    args = [a for a in sys.argv[1:]]
    conn = db.connect()
    dc, cal = _model(), load_ou_calibrator()

    if len(args) >= 2:                       # two team names given: forecast just that matchup
        neutral = int(args[2]) if len(args) >= 3 else 0
        # turn whatever was typed ('Ivory Coast') into the name the data uses ('Côte d'Ivoire')
        home, away = resolve_name(args[0], dc.teams), resolve_name(args[1], dc.teams)
        for raw, res in [(args[0], home), (args[1], away)]:
            if res not in dc.teams:
                print(f"[warn] '{raw}' not found in model -> treated as an average team (weak forecast)")
        # add the matchup as a made-up fixture dated today so the winner model builds features for it
        today = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
        mk = db.match_key(today, home, away)
        extra = [{"match_key": mk, "date": today, "tournament": "Friendly",
                  "home_team": home, "away_team": away, "neutral": neutral,
                  "home_goals": None, "away_goals": None}]
        _, wprobs = fit_and_predict(conn, extra=extra)
        print_card(forecast(dc, home, away, neutral, cal, win=wprobs.get(mk)))
        conn.close(); return 0

    today = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    # Upcoming fixtures are the rows in the results data that don't have a score yet.
    sched = pd.read_sql("SELECT date, home_team, away_team, neutral FROM matches "
                        "WHERE home_goals IS NULL AND date>=? ORDER BY date", conn, params=[today])
    pairs = [(str(r["date"])[:10], r["home_team"], r["away_team"], int(r["neutral"]))
             for _, r in sched.iterrows()]
    seen = {(d, h, a) for d, h, a, _ in pairs}
    # the odds feed sometimes lists a match before the results data does; include those too
    for (mk,) in conn.execute("SELECT DISTINCT match_key FROM odds_snapshots WHERE market='1x2'"):
        d, h, a = mk.split("|")
        if d >= today and (d, h, a) not in seen:
            pairs.append((d, h, a, get_neutral(conn, d, h, a))); seen.add((d, h, a))
    if not pairs:
        print('no upcoming matches found. Pass two team names, e.g.:  predict "Brazil" "Spain" 1')
        conn.close(); return 0

    # Fit the winner model once and predict all of them. Matches known only from the odds feed
    # have no row in the matches table, so they have to be passed in as extra fixtures. Without
    # that the winner model has nothing for them and forecast() quietly uses the goal model.
    have = {r[0] for r in conn.execute("SELECT match_key FROM matches")}
    extra = [{"match_key": db.match_key(d, h, a), "date": d, "tournament": "Friendly",
              "home_team": h, "away_team": a, "neutral": nt, "home_goals": None, "away_goals": None}
             for d, h, a, nt in pairs if db.match_key(d, h, a) not in have]
    _, wprobs = fit_and_predict(conn, extra=extra or None)

    print("=" * 78)
    print(f"MATCH PREDICTIONS  ({len(pairs)} upcoming)")
    print("=" * 78)
    cards = []
    for d, h, a, nt in pairs:
        mk = db.match_key(d, h, a)
        df = _latest_odds(conn, mk)
        market = _market_1x2(df) if not df.empty else None
        cards.append((d, forecast(dc, h, a, nt, cal, win=wprobs.get(mk)), market))
    for date, f, market in sorted(cards, key=lambda c: (c[0], -c[1]["conf"])):
        print_card(f, market)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

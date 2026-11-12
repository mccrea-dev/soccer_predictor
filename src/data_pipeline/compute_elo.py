"""Compute Elo ratings for every national team by replaying the full match history in date order.

I compute these rather than scraping an existing ratings site because replaying matches in order
guarantees that every rating only reflects games already played, and there is no scraper to break.

Each team's rating after each match is stored in the ratings table, dated on match day. To get
the rating a team took into a match, use db.rating_before.

The formula is the standard World Football Elo one:

  new = old + K * G * (W - We)

  W    result for the team: 1 win, 0.5 draw, 0 loss (a penalty shootout counts as a draw)
  We   expected result: 1 / (10 ** (-dr / 400) + 1), where dr is the rating difference plus
       100 points for the home side unless the venue is neutral
  K    weight for how much the match matters (60 for a World Cup down to 20 for a friendly)
  G    bonus for winning margin: 1 for a draw or one-goal win, 1.5 for two, (11 + margin) / 8 beyond

New teams start at 1500, and whatever one side gains the other loses.

Run:  python soccer.py compute-elo
"""
import sys
import unicodedata
import pandas as pd

from core import db

BASE = 1500.0
HOME_ADV = 100.0


def _norm(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def elo_k(tournament):
    """K factor: how much a single result moves the ratings, by how much the match matters."""
    t = _norm(tournament)
    if t == "fifa world cup":
        return 60.0
    if t in ("uefa euro", "copa america", "african cup of nations", "afc asian cup",
             "gold cup", "fifa confederations cup", "concacaf championship"):
        return 50.0
    if "qualification" in t or "nations league" in t:
        return 40.0
    if t == "friendly":
        return 20.0
    return 30.0


def gd_multiplier(gd):
    """Winning-margin multiplier G: 1 for a draw or one-goal game, 1.5 for two goals,
    (11 + margin) / 8 for three or more."""
    gd = abs(int(gd))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def expected_home(home_elo, away_elo, neutral):
    """Elo's expected result for the home team, between 0 and 1."""
    dr = (home_elo + (0.0 if neutral else HOME_ADV)) - away_elo
    return 1.0 / (10 ** (-dr / 400.0) + 1.0)


def main():
    conn = db.connect()
    db.init_db(conn)
    m = pd.read_sql("SELECT * FROM matches WHERE home_goals IS NOT NULL ORDER BY date, match_key", conn)
    if m.empty:
        print("no played matches -- run ingest_results first.")
        return 1

    rating = {}            # team -> rating right now, as the replay moves forward
    rows = []              # every post-match rating, written to the DB at the end
    for _, r in m.iterrows():
        h, a = r["home_team"], r["away_team"]
        rh = rating.get(h, BASE)
        ra = rating.get(a, BASE)
        we = expected_home(rh, ra, bool(r["neutral"]))
        hs, as_ = int(r["home_goals"]), int(r["away_goals"])
        w = 1.0 if hs > as_ else 0.5 if hs == as_ else 0.0
        k = elo_k(r["tournament"]) * gd_multiplier(hs - as_)
        delta = k * (w - we)          # what the home side gains and the away side loses
        rating[h] = rh + delta
        rating[a] = ra - delta
        d = r["date"]
        rows.append({"team": h, "date": d, "system": "elo", "value": round(rating[h], 2)})
        rows.append({"team": a, "date": d, "system": "elo", "value": round(rating[a], 2)})

    # start clean so a rebuild never leaves stale ratings behind
    conn.execute("DELETE FROM ratings WHERE system='elo'")
    db.upsert(conn, "ratings", rows, ["team", "date", "system"])
    conn.commit()
    print(f"computed Elo over {len(m)} matches -> {len(rows)} rating rows for {len(rating)} teams")
    top = sorted(rating.items(), key=lambda kv: -kv[1])[:12]
    print("current top 12 (sanity check):")
    for t, v in top:
        print(f"  {t:24s} {v:7.1f}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

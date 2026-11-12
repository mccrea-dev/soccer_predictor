"""Build a table of pre-match features, one row per match, for exploring and modelling goals.

The whole point of this file is that every feature is something you could have known before
kick-off:
  * Elo: each team's rating strictly before the match date
  * form: average goals scored and conceded over each team's previous 10 matches
  * rest days since the last match, number of matches played so far, venue, competition type

It works by walking through history in date order. For each match the features are read off the
running state first, and only then is the state updated with that match's score.

The outcomes (total goals, over 2.5, result, both teams scored) sit alongside as target columns.
The table is written to SQLite as `features` and replaced on every run.

Run:  python soccer.py build-features
"""
import sys
from collections import deque, defaultdict
import numpy as np
import pandas as pd

from core import db
from data_pipeline.ingest_results import competition_of, is_competitive, TARGET_COMPETITIONS

FEATURE_START = "2002-01-01"   # only write rows from here on; earlier matches still feed Elo and form
FORM_N = 10                    # how many previous matches the form averages look back over
DEFAULT_ELO = 1500.0


def _mean(dq):
    """Average of a team's recent values, NaN if it hasn't played yet."""
    return float(np.mean(dq)) if len(dq) else np.nan


def main():
    conn = db.connect()
    db.init_db(conn)
    m = pd.read_sql("SELECT * FROM matches WHERE home_goals IS NOT NULL ORDER BY date, match_key", conn)
    if m.empty:
        print("no played matches -- run ingest_results + compute_elo first.")
        return 1
    if conn.execute("SELECT COUNT(*) FROM ratings WHERE system='elo'").fetchone()[0] == 0:
        print("no Elo ratings -- run compute_elo first.")
        return 1

    # Running state per team. The deques drop the oldest match once they hold FORM_N.
    gf = defaultdict(lambda: deque(maxlen=FORM_N))   # goals scored in recent matches
    ga = defaultdict(lambda: deque(maxlen=FORM_N))   # goals conceded in recent matches
    last_date = {}                                   # date of each team's previous match
    mp = defaultdict(int)                            # matches played so far

    out = []
    for _, r in m.iterrows():
        h, a, date = r["home_team"], r["away_team"], r["date"]
        hs, as_ = int(r["home_goals"]), int(r["away_goals"])
        comp = competition_of(r["tournament"])
        # every match updates the running state, but only these get a row in the output
        emit = comp in TARGET_COMPETITIONS and date >= FEATURE_START

        if emit:
            eh = db.rating_before(conn, h, "elo", date)
            ea = db.rating_before(conn, a, "elo", date)
            eh = DEFAULT_ELO if eh is None else eh
            ea = DEFAULT_ELO if ea is None else ea
            rh = (pd.Timestamp(date) - pd.Timestamp(last_date[h])).days if h in last_date else np.nan
            ra = (pd.Timestamp(date) - pd.Timestamp(last_date[a])).days if a in last_date else np.nan
            total = hs + as_
            out.append({
                "match_key": r["match_key"], "date": date, "tournament": r["tournament"],
                "competition": comp, "neutral": int(r["neutral"]),
                "is_competitive": is_competitive(r["tournament"]),
                "home_team": h, "away_team": a,
                "elo_home": eh, "elo_away": ea, "elo_diff": eh - ea, "elo_sum": eh + ea,
                "home_gf10": _mean(gf[h]), "home_ga10": _mean(ga[h]),
                "away_gf10": _mean(gf[a]), "away_ga10": _mean(ga[a]),
                "home_rest": min(rh, 3650) if pd.notna(rh) else np.nan,
                "away_rest": min(ra, 3650) if pd.notna(ra) else np.nan,
                "home_mp": mp[h], "away_mp": mp[a],
                # what actually happened (targets, never used as inputs)
                "home_goals": hs, "away_goals": as_, "total_goals": total,
                "over_2_5": int(total > 2.5),
                "result": "H" if hs > as_ else ("A" if as_ > hs else "D"),
                "btts": int(hs >= 1 and as_ >= 1),
            })

        # only now fold this match into the running state, so it can't leak into its own features
        gf[h].append(hs); ga[h].append(as_)
        gf[a].append(as_); ga[a].append(hs)
        last_date[h] = date; last_date[a] = date
        mp[h] += 1; mp[a] += 1

    feats = pd.DataFrame(out)
    feats.to_sql("features", conn, if_exists="replace", index=False)
    conn.commit()

    # Sanity check for leakage: no pre-match feature should correlate strongly with the final
    # score. If one does, it has almost certainly been built from the result by mistake.
    num = ["elo_home", "elo_away", "elo_diff", "elo_sum", "home_gf10", "home_ga10",
           "away_gf10", "away_ga10", "home_rest", "away_rest", "home_mp", "away_mp"]
    y = feats["total_goals"].values
    worst_c, worst_r = None, 0.0
    for c in num:
        x = feats[c].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 200 and np.std(x[ok]) > 0:
            rr = abs(np.corrcoef(x[ok], y[ok])[0, 1])
            if rr > worst_r:
                worst_c, worst_r = c, rr
    print(f"built features: {len(feats)} matches, {feats.shape[1]} cols, "
          f"{feats['date'].min()} -> {feats['date'].max()}")
    print(f"  competitions: {feats['competition'].value_counts().to_dict()}")
    print(f"  base rate over 2.5: {feats['over_2_5'].mean()*100:.1f}%  | mean total goals: {feats['total_goals'].mean():.2f}")
    print(f"  [leakage] max |corr(feature, total_goals)| = {worst_r:.3f} ({worst_c})  "
          f"(expect < 0.5; ~1.0 would be a leak)")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

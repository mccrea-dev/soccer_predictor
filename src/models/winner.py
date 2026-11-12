"""Winner model: probabilities for home win / draw / away win.

This is a regularised multinomial logistic regression. It exists because the validation harness
(backtest/walk_forward.py) showed that the Dixon-Coles goal model, for all its structure, was no
better at calling results than a two-feature logistic on Elo. So result prediction got its own
model, and Dixon-Coles is kept for what it's good at: scorelines and goal totals.

The features, all known before kick-off:
  * Elo: the difference, the sum, the absolute gap, and Elo's own expected result. The sum and
    the gap are there for draws, which are more likely between closely matched and
    lower-scoring sides.
  * recent form as exponentially weighted averages: goals for and against, results, and
    results relative to what Elo expected (so beating a strong side counts for more)
  * days of rest and matches played so far
  * which competition it is
  * the Dixon-Coles expected goals for each side
  * a correction for Elo's bias between confederations (see core/geography.py)

Form features are built in one pass through history in date order, reading each team's running
averages before updating them, the same way build_features.py does it.
"""
import bisect
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from data_pipeline.ingest_results import is_competitive, competition_of

MODELS = Path(__file__).resolve().parents[2] / "models"
HOME, DRAW, AWAY = 0, 1, 2
HOME_ADV = 100.0           # Elo points for playing at home; has to match compute_elo.py
EWMA_ALPHA = 0.30          # weight on the newest match in the form averages; anything from
                           # 0.20 to 0.40 scored about the same in validation

# One indicator column per competition type, so the model can learn that, say, home advantage is
# bigger in qualifiers than in friendlies. Anything not listed gets zeros across the board.
COMPS = ["world_cup", "world_cup_qual", "euro", "euro_qual", "nations_league", "copa_america", "friendly"]

# The model's input columns, in order. add_form_features makes most of them, add_dc_features adds
# dc_lam / dc_mu, and conf_bias comes from core.geography. I also tried plain confederation
# indicators; conf_bias did better on both validation setups, so only that one is used.
FEATURES = (["elo_diff", "hfa", "elo_sum", "abs_diff", "we",
             "h_gf", "h_ga", "a_gf", "a_ga", "h_pts", "a_pts", "h_perf", "a_perf",
             "h_rest", "a_rest", "h_mp", "a_mp", "is_competitive"]
            + [f"comp_{c}" for c in COMPS]
            + ["dc_lam", "dc_mu", "conf_bias"])


def _we(elo_h, elo_a, neutral):
    """Elo's expected result for the home team (0 to 1)."""
    dr = (elo_h + (0.0 if neutral else HOME_ADV)) - elo_a
    return 1.0 / (10 ** (-dr / 400.0) + 1.0)


def add_form_features(df, ewma_alpha=EWMA_ALPHA):
    """Add the Elo and form features to a match table.

    Needs date, home_team, away_team, home_goals, away_goals, neutral, tournament, plus elo_h
    and elo_a holding each side's Elo before the match.

    Matches are processed in date order. For each one, the teams' current form averages are
    written out as features first and updated with the result afterwards, so no match sees
    its own score. Fixtures without a score yet get features but leave the averages alone.
    """
    df = df.sort_values("date").reset_index(drop=True)
    eh = df["elo_h"].to_numpy(float); ea = df["elo_a"].to_numpy(float)
    neu = df["neutral"].to_numpy(int)
    ht = df["home_team"].to_numpy(); at = df["away_team"].to_numpy()
    hg = pd.to_numeric(df["home_goals"], errors="coerce").to_numpy(float)
    ag = pd.to_numeric(df["away_goals"], errors="coerce").to_numpy(float)
    dates = df["date"].to_numpy("datetime64[ns]")
    comp = df["tournament"].map(is_competitive).to_numpy(int)

    # Running form per team: goals for, goals against, results, results vs. Elo expectation.
    # None means the team hasn't played yet.
    gf = defaultdict(lambda: None); ga = defaultdict(lambda: None)
    pts = defaultdict(lambda: None); perf = defaultdict(lambda: None)
    last = {}; mp = defaultdict(int)
    n = len(df)
    out = {c: np.full(n, np.nan) for c in
           ("h_gf", "h_ga", "a_gf", "a_ga", "h_pts", "a_pts", "h_perf", "a_perf", "h_rest", "a_rest")}
    hmp = np.zeros(n); amp = np.zeros(n); we_col = np.zeros(n)

    # exponentially weighted average: the newest value gets weight alpha, older ones fade out
    def upd(d, key, v):
        cur = d[key]
        d[key] = v if cur is None else ewma_alpha * v + (1 - ewma_alpha) * cur

    for i in range(n):
        h, a = ht[i], at[i]
        we_col[i] = _we(eh[i], ea[i], neu[i])
        # write out the form each team brings into this match
        out["h_gf"][i] = gf[h] if gf[h] is not None else np.nan
        out["h_ga"][i] = ga[h] if ga[h] is not None else np.nan
        out["a_gf"][i] = gf[a] if gf[a] is not None else np.nan
        out["a_ga"][i] = ga[a] if ga[a] is not None else np.nan
        out["h_pts"][i] = pts[h] if pts[h] is not None else np.nan
        out["a_pts"][i] = pts[a] if pts[a] is not None else np.nan
        out["h_perf"][i] = perf[h] if perf[h] is not None else np.nan
        out["a_perf"][i] = perf[a] if perf[a] is not None else np.nan
        if h in last:
            out["h_rest"][i] = min((dates[i] - last[h]) / np.timedelta64(1, "D"), 365)
        if a in last:
            out["a_rest"][i] = min((dates[i] - last[a]) / np.timedelta64(1, "D"), 365)
        hmp[i] = mp[h]; amp[i] = mp[a]

        if np.isnan(hg[i]) or np.isnan(ag[i]):
            continue                                   # not played yet, nothing to update with
        hs, as_ = int(hg[i]), int(ag[i])
        w = 1.0 if hs > as_ else 0.5 if hs == as_ else 0.0
        upd(gf, h, hs); upd(ga, h, as_); upd(gf, a, as_); upd(ga, a, hs)
        upd(pts, h, w); upd(pts, a, 1 - w)
        # result minus what Elo expected: positive after beating a stronger side
        upd(perf, h, w - we_col[i]); upd(perf, a, (1 - w) - (1 - we_col[i]))
        last[h] = dates[i]; last[a] = dates[i]; mp[h] += 1; mp[a] += 1

    for c, v in out.items():
        df[c] = v
    df["we"] = we_col
    df["h_mp"] = hmp; df["a_mp"] = amp
    df["elo_sum"] = eh + ea
    df["abs_diff"] = np.abs(eh - ea)
    df["is_competitive"] = comp
    comp_cat = df["tournament"].map(competition_of)
    for c in COMPS:
        df[f"comp_{c}"] = (comp_cat == c).astype(float)
    if "elo_diff" not in df:
        df["elo_diff"] = eh - ea
    if "hfa" not in df:
        df["hfa"] = 1 - neu
    return df


def add_dc_features(df, dc):
    """Add the goal model's (log) expected goals for each side as two more features.

    `dc` is a fitted DixonColes. Elo only sees win/draw/loss, so this brings in something it
    can't: how many goals a side tends to score and concede given who it played. When
    backtesting, `dc` must be fitted on the training period only."""
    lm = np.array([dc.lambda_mu(h, a, int(n))
                   for h, a, n in zip(df["home_team"], df["away_team"], df["neutral"])])
    lm = np.log(np.clip(lm, 0.05, None)) if len(lm) else lm.reshape(0, 2)
    df = df.copy()
    df["dc_lam"] = lm[:, 0] if len(lm) else []
    df["dc_mu"] = lm[:, 1] if len(lm) else []
    return df


class WinnerModel:
    """The classifier itself: fill missing values with the median, standardise, then a
    regularised multinomial logistic regression.

    fit() and predict_proba() both take a DataFrame containing the FEATURES columns.
    predict_proba returns an (n, 3) array ordered home / draw / away.
    """

    def __init__(self, C=0.1):
        self.C = C
        self.pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(max_iter=2000, C=C))

    def fit(self, df, y=None):
        X = df[FEATURES].to_numpy(float)
        y = df["y"].to_numpy() if y is None else np.asarray(y)
        self.pipe.fit(X, y)
        self.classes_ = self.pipe.classes_
        return self

    def predict_proba(self, df):
        X = df[FEATURES].to_numpy(float)
        Pl = self.pipe.predict_proba(X)
        # sklearn orders columns by class label; put them explicitly in home/draw/away order
        P = np.zeros((len(X), 3))
        for j, c in enumerate(self.pipe.classes_):
            P[:, int(c)] = Pl[:, j]
        return P

    def save(self, path=None):
        MODELS.mkdir(exist_ok=True)
        joblib.dump(self, path or (MODELS / "winner.pkl"))
        return self

    @classmethod
    def load(cls, path=None):
        return joblib.load(path or (MODELS / "winner.pkl"))


# --------------------------------------------------------------------- features straight from the DB
def _attach_elo(conn, m):
    """Add each side's pre-match Elo (elo_h, elo_a) to a match table.

    Ratings are stored after each match, dated on match day, so the rating a team brings
    into a game is its last one from strictly before that date. Each team's ratings are
    held as a date-sorted list and searched with bisect, which is far quicker than one SQL
    query per row."""
    rat = pd.read_sql("SELECT team, date, value FROM ratings WHERE system='elo' ORDER BY team, date", conn)
    hist = {t: (list(g["date"]), list(g["value"])) for t, g in rat.groupby("team")}
    ds = pd.to_datetime(m["date"]).dt.strftime("%Y-%m-%d").to_numpy()

    def before(team, d):
        dd, vv = hist.get(team, ([], []))
        i = bisect.bisect_left(dd, d)
        return vv[i - 1] if i > 0 else 1500.0

    m["elo_h"] = [before(t, d) for t, d in zip(m["home_team"], ds)]
    m["elo_a"] = [before(t, d) for t, d in zip(m["away_team"], ds)]
    m["elo_diff"] = m["elo_h"] - m["elo_a"]
    m["hfa"] = 1 - m["neutral"].astype(int)
    return m


def feature_frame(conn, extra=None, dc=None):
    """Every match in the database, played or not, with the full feature set attached.

    `extra` adds hypothetical matchups: a list of dicts with date, home_team, away_team,
    neutral and tournament, and no goals. They get features like any upcoming fixture.
    `dc` is a fitted DixonColes; if omitted, the saved one is loaded (or fitted if there
    isn't one)."""
    m = pd.read_sql("SELECT match_key, date, tournament, home_team, away_team, neutral, "
                    "home_goals, away_goals FROM matches", conn)
    if extra:
        m = pd.concat([m, pd.DataFrame(extra)], ignore_index=True)
    m["neutral"] = m["neutral"].fillna(0).astype(int)
    m["tournament"] = m["tournament"].fillna("Friendly")
    m["date"] = pd.to_datetime(m["date"])
    m = _attach_elo(conn, m)
    m = add_form_features(m)
    from core.geography import add_conf_features
    m = add_conf_features(m)
    if dc is None:
        from models.goals_statistical import DixonColes, fit_from_db
        try:
            dc = DixonColes.load()
        except Exception:
            dc = fit_from_db(); dc.save()
    m = add_dc_features(m, dc)
    return m


def fit_and_predict(conn, extra=None, all_rows=False):
    """Train on every played match and predict the ones that haven't been played.

    Returns (model, {match_key: (p_home, p_draw, p_away)}). By default only upcoming
    fixtures and `extra` matchups are predicted, so every number is a genuine forecast.

    all_rows=True also returns probabilities for matches already played. The model was
    trained on those results, so they are not forecasts and shouldn't be scored as if
    they were."""
    fr = feature_frame(conn, extra=extra)
    played = fr[fr["home_goals"].notna()].copy()
    played["y"] = [HOME if h > a else AWAY if a > h else DRAW
                   for h, a in zip(played["home_goals"], played["away_goals"])]
    wm = WinnerModel().fit(played)
    tgt = fr if all_rows else fr[fr["home_goals"].isna()]
    probs = {}
    if len(tgt):
        P = wm.predict_proba(tgt)
        probs = {mk: (float(p[0]), float(p[1]), float(p[2]))
                 for mk, p in zip(tgt["match_key"], P)}
    return wm, probs

"""Validation harness: how well do the models predict matches they haven't seen?

Two tests, and in both the models are refitted from scratch for every fold, so nothing is ever
predicted by a model that trained on it.

  walk-forward by year       for each year Y, train on everything before Y and predict that
                             year's matches
  leave-one-tournament-out   hold out an entire World Cup, Euro or Copa America, train on
                             everything before it, and predict the whole tournament. This is the
                             closest thing to the real use case.

The main metric is the ranked probability score (RPS). It suits football because the outcomes
have an order (home win, draw, away win): predicting a home win when the match is drawn is a
smaller miss than when the away side wins, and RPS scores it that way. Log loss, Brier score and
plain accuracy are reported too.

Four models are compared: historical base rates, a logistic regression on Elo alone,
Dixon-Coles, and the full winner model. The Elo-only one is the bar that matters. If a more
complicated model can't beat it, the complication isn't earning its keep.

Each run appends its numbers to reports/validation/leaderboard.json so a change can be compared
with the best result so far.

Run:  python soccer.py validate [tag]
"""
import sys
import json
import bisect
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from core import db
from models.goals_statistical import DixonColes
from models.winner import WinnerModel, add_form_features, add_dc_features
from data_pipeline.ingest_results import competition_of, TARGET_COMPETITIONS

REPORTS = Path(__file__).resolve().parents[2] / "reports" / "validation"
LEADERBOARD = REPORTS / "leaderboard.json"
WF_YEARS = range(2012, 2026)
HOME, DRAW, AWAY = 0, 1, 2        # class labels, in the order RPS needs them

# tournaments to hold out whole: (label, first day, last day)
TOURNAMENTS = [
    ("WC 2018", "2018-06-14", "2018-07-15"), ("WC 2022", "2022-11-20", "2022-12-18"),
    ("Euro 2016", "2016-06-10", "2016-07-10"), ("Euro 2020", "2021-06-11", "2021-07-11"),
    ("Euro 2024", "2024-06-14", "2024-07-14"), ("Copa 2019", "2019-06-14", "2019-07-07"),
    ("Copa 2021", "2021-06-13", "2021-07-10"), ("Copa 2024", "2024-06-20", "2024-07-14"),
]


# ------------------------------------------------------------------ metrics
def _onehot(y):
    """Outcome labels as an (n, 3) array of 0s with a 1 in the column that happened."""
    O = np.zeros((len(y), 3)); O[np.arange(len(y)), y] = 1; return O


def rps(P, y):
    """Ranked probability score, lower is better. Compares cumulative predicted probabilities
    with cumulative outcomes, so being wrong by one step (home win vs. draw) costs less than
    being wrong by two (home win vs. away win)."""
    P = np.asarray(P, float); O = _onehot(np.asarray(y, int))
    cP, cO = np.cumsum(P, 1), np.cumsum(O, 1)
    return float(np.mean(np.sum((cP[:, :-1] - cO[:, :-1]) ** 2, axis=1) / (P.shape[1] - 1)))


def logloss(P, y, eps=1e-12):
    """Multiclass log loss, lower is better."""
    P = np.clip(np.asarray(P, float), eps, 1); P = P / P.sum(1, keepdims=True)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def brier(P, y):
    """Multiclass Brier score: squared error between predicted probabilities and what happened."""
    return float(np.mean(np.sum((np.asarray(P, float) - _onehot(y)) ** 2, axis=1)))


def accuracy(P, y):
    """Share of matches where the most likely predicted outcome was the actual one."""
    return float(np.mean(np.argmax(P, 1) == y))


def metrics(P, y):
    """All four scores plus the number of matches, as a dict."""
    return {"rps": rps(P, y), "logloss": logloss(P, y), "brier": brier(P, y),
            "acc": accuracy(P, y), "n": int(len(y))}


# ------------------------------------------------------------------ data
def _result(hg, ag):
    """Outcome label from a scoreline."""
    return HOME if hg > ag else AWAY if ag > hg else DRAW


def load_matches():
    """Every played match with its outcome label and the full pre-match feature set: Elo going
    in, form, and the confederation features. Sorted by date.
    """
    conn = db.connect()
    m = pd.read_sql("SELECT * FROM matches WHERE home_goals IS NOT NULL", conn)
    rat = pd.read_sql("SELECT team, date, value FROM ratings WHERE system='elo' ORDER BY team, date", conn)
    conn.close()
    m["date"] = pd.to_datetime(m["date"])
    m["ds"] = m["date"].dt.strftime("%Y-%m-%d")
    m["competition"] = m["tournament"].map(competition_of)
    m["y"] = [_result(h, a) for h, a in zip(m["home_goals"], m["away_goals"])]

    # Elo going into each match. Ratings are stored after each game, dated on match day, so the
    # pre-match value is the last one strictly before that date (found with bisect).
    hist = {t: (list(g["date"]), list(g["value"])) for t, g in rat.groupby("team")}

    def elo_before(team, ds):
        d, v = hist.get(team, ([], []))
        i = bisect.bisect_left(d, ds)
        return v[i - 1] if i > 0 else 1500.0

    m["elo_h"] = [elo_before(t, d) for t, d in zip(m["home_team"], m["ds"])]
    m["elo_a"] = [elo_before(t, d) for t, d in zip(m["away_team"], m["ds"])]
    m["elo_diff"] = m["elo_h"] - m["elo_a"]
    m["hfa"] = 1 - m["neutral"]
    # form, rest and experience features, built in one pass through history
    m = add_form_features(m)
    from core.geography import add_conf_features
    m = add_conf_features(m)
    return m.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------------ per-fold models
def _fit_predict_fold(train, test):
    """Fit all four models on `train` and predict `test`. Returns {model name: (n, 3) array of
    home/draw/away probabilities}."""
    out = {}

    # Dixon-Coles, with match ages measured from the start of the test period
    dc = DixonColes().fit(train, ref_date=test["date"].min())
    P = np.array([[*(lambda mk: (mk["home"], mk["draw"], mk["away"]))(
        dc.markets(r.home_team, r.away_team, int(r.neutral)))] for r in test.itertuples()])
    out["dixon_coles"] = P

    # logistic regression on Elo difference and home advantage only: the baseline to beat
    Xtr = train[["elo_diff", "hfa"]].to_numpy(float)
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(Xtr, train["y"].to_numpy())
    Pe = np.zeros((len(test), 3))
    Pl = lr.predict_proba(test[["elo_diff", "hfa"]].to_numpy(float))
    for j, c in enumerate(lr.classes_):
        Pe[:, c] = Pl[:, j]
    out["elo_logistic"] = Pe

    # the full winner model. Its expected-goals features come from the Dixon-Coles model fitted
    # just above, so they only use this fold's training data too.
    out["winner"] = WinnerModel(C=0.1).fit(add_dc_features(train, dc)).predict_proba(add_dc_features(test, dc))

    # historical frequencies of home win / draw / away win, the same for every match
    br = np.array([(train["y"] == k).mean() for k in (HOME, DRAW, AWAY)])
    out["base_rate"] = np.tile(br, (len(test), 1))
    return out


# ------------------------------------------------------------------ regimes
def walk_forward(m):
    """Year-by-year test. Returns ({model: stacked predictions}, outcomes, row indices into m).
    Years with too little test or training data are skipped.
    """
    preds = defaultdict(list); ys = []
    keep_idx = []
    for Y in WF_YEARS:
        lo, hi = pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y + 1}-01-01")
        train = m[m["date"] < lo]
        test = m[(m["date"] >= lo) & (m["date"] < hi) & m["competition"].isin(TARGET_COMPETITIONS)]
        if len(test) < 30 or len(train) < 2000:
            continue
        fold = _fit_predict_fold(train, test)
        for name, P in fold.items():
            preds[name].append(P)
        ys.append(test["y"].to_numpy())
        keep_idx.append(test.index.to_numpy())
    y = np.concatenate(ys)
    P = {name: np.vstack(parts) for name, parts in preds.items()}
    idx = np.concatenate(keep_idx)
    return P, y, idx


def leave_one_tournament_out(m):
    """Whole-tournament test. Returns the same three things as walk_forward, plus Dixon-Coles
    metrics for each tournament separately.
    """
    preds = defaultdict(list); ys = []; idxs = []
    per_tourn = []
    for name, s, e in TOURNAMENTS:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        train = m[m["date"] < s]
        test = m[(m["date"] >= s) & (m["date"] <= e)]
        if len(test) < 10 or len(train) < 2000:
            continue
        fold = _fit_predict_fold(train, test)
        per_tourn.append((name, metrics(fold["dixon_coles"], test["y"].to_numpy())))
        for k, P in fold.items():
            preds[k].append(P)
        ys.append(test["y"].to_numpy()); idxs.append(test.index.to_numpy())
    y = np.concatenate(ys)
    P = {k: np.vstack(v) for k, v in preds.items()}
    return P, y, np.concatenate(idxs), per_tourn


# ------------------------------------------------------------------ where it goes wrong + leaderboard
def error_breakdown(m, idx, P, y):
    """Print Dixon-Coles RPS split by the Elo of the weaker team and by competition, to show
    which kinds of match the model struggles with.
    """
    df = m.loc[idx].copy().reset_index(drop=True)
    df["rps"] = [rps([P[i]], [y[i]]) for i in range(len(y))]
    df["min_elo"] = df[["elo_h", "elo_a"]].min(axis=1)
    print("\n  DC RPS by weakest-team Elo:")
    for lab, q in [("<1500", df.min_elo < 1500), ("1500-1700", (df.min_elo >= 1500) & (df.min_elo < 1700)),
                   ("1700-1900", (df.min_elo >= 1700) & (df.min_elo < 1900)), (">=1900", df.min_elo >= 1900)]:
        g = df[q]
        if len(g) > 20:
            print(f"    {lab:10s} n={len(g):5d}  RPS={g['rps'].mean():.4f}  acc={(np.argmax(np.array([P[i] for i in g.index]),1)==y[g.index]).mean()*100:4.1f}%")
    print("  DC RPS by competition:")
    for comp, g in df.groupby("competition"):
        if len(g) > 30:
            print(f"    {comp:16s} n={len(g):5d}  RPS={g['rps'].mean():.4f}")


def update_leaderboard(tag, wf_m, loto_m, primary="winner"):
    """Append this run's winner-model metrics to the leaderboard under `tag`. Returns the new
    entry and the previous best (lowest walk-forward RPS), if there was one."""
    board = json.loads(LEADERBOARD.read_text()) if LEADERBOARD.exists() else []
    p = wf_m[primary]; pl = loto_m[primary]
    entry = {"tag": tag, "model": primary, "ts": pd.Timestamp.now("UTC").isoformat(),
             "wf_rps": p["rps"], "wf_logloss": p["logloss"], "wf_acc": p["acc"],
             "loto_rps": pl["rps"], "loto_acc": pl["acc"]}
    prev_best = min((e for e in board), key=lambda e: e["wf_rps"], default=None)
    board.append(entry)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LEADERBOARD.write_text(json.dumps(board, indent=2))
    return entry, prev_best


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "dixon_coles"
    m = load_matches()
    print("=" * 92)
    print(f"VALIDATION HARNESS  (model tag: {tag})   primary metric = RPS (lower is better)")
    print("=" * 92)

    print("\nWALK-FORWARD (train<Y, predict target-comp matches in Y; retrained per year)")
    Pwf, ywf, idxwf = walk_forward(m)
    wf_m = {k: metrics(P, ywf) for k, P in Pwf.items()}
    print(f"  {'model':16s} {'RPS':>8s} {'logloss':>9s} {'Brier':>8s} {'acc':>7s}   (n={len(ywf)})")
    for k in ("base_rate", "elo_logistic", "dixon_coles", "winner"):
        r = wf_m[k]
        print(f"  {k:16s} {r['rps']:.4f}  {r['logloss']:.4f}   {r['brier']:.4f}  {r['acc']*100:5.1f}%")

    print("\nLEAVE-ONE-TOURNAMENT-OUT (hold out whole WC/Euro/Copa editions)")
    Plo, ylo, idxlo, per_t = leave_one_tournament_out(m)
    loto_m = {k: metrics(P, ylo) for k, P in Plo.items()}
    print(f"  {'model':16s} {'RPS':>8s} {'logloss':>9s} {'acc':>7s}   (n={len(ylo)})")
    for k in ("base_rate", "elo_logistic", "dixon_coles", "winner"):
        r = loto_m[k]
        print(f"  {k:16s} {r['rps']:.4f}  {r['logloss']:.4f}   {r['acc']*100:5.1f}%")
    print("  per-tournament (Dixon-Coles):")
    for name, r in per_t:
        print(f"    {name:10s} n={r['n']:3d}  RPS={r['rps']:.4f}  acc={r['acc']*100:4.1f}%")

    # would averaging the winner model with Dixon-Coles help? Costs nothing to check, since both
    # sets of predictions are already computed.
    print("\nBLEND winner x dixon_coles (w=winner weight):")
    for w in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        Bw = w * Pwf["winner"] + (1 - w) * Pwf["dixon_coles"]
        Bl = w * Plo["winner"] + (1 - w) * Plo["dixon_coles"]
        print(f"    w={w:.2f}  wf_RPS={rps(Bw, ywf):.4f}  loto_RPS={rps(Bl, ylo):.4f}")

    error_breakdown(m, idxwf, Pwf["dixon_coles"], ywf)
    print("  WINNER-model RPS by weakest-team Elo:")
    dfw = m.loc[idxwf].copy().reset_index(drop=True)
    dfw["min_elo"] = dfw[["elo_h", "elo_a"]].min(axis=1)
    Pw = Pwf["winner"]
    for lab, q in [("<1500", dfw.min_elo < 1500), ("1500-1700", (dfw.min_elo >= 1500) & (dfw.min_elo < 1700)),
                   ("1700-1900", (dfw.min_elo >= 1700) & (dfw.min_elo < 1900)), (">=1900", dfw.min_elo >= 1900)]:
        g = dfw[q]
        if len(g) > 20:
            gi = g.index.to_numpy()
            print(f"    {lab:10s} n={len(g):5d}  RPS={rps(Pw[gi], ywf[gi]):.4f}  "
                  f"acc={(np.argmax(Pw[gi], 1) == ywf[gi]).mean()*100:4.1f}%")

    entry, prev = update_leaderboard(tag, wf_m, loto_m)
    print("\n" + "-" * 92)
    # the comparison that matters most: did the winner model beat the Elo-only baseline this run?
    dc = wf_m["elo_logistic"]; dl = loto_m["elo_logistic"]
    print(f"winner vs elo_logistic (this run): wf_RPS {dc['rps']:.4f} -> {entry['wf_rps']:.4f} "
          f"({entry['wf_rps']-dc['rps']:+.4f})   loto_RPS {dl['rps']:.4f} -> {entry['loto_rps']:.4f} "
          f"({entry['loto_rps']-dl['rps']:+.4f})")
    if prev:
        d = entry["wf_rps"] - prev["wf_rps"]
        verdict = "BETTER" if d < 0 else "WORSE" if d > 0 else "SAME"
        print(f"vs previous best ('{prev['tag']}'): wf_RPS {prev['wf_rps']:.4f} -> {entry['wf_rps']:.4f} "
              f"({d:+.4f})  => {verdict}")
    else:
        print(f"baseline recorded: wf_RPS={entry['wf_rps']:.4f}  loto_RPS={entry['loto_rps']:.4f}  "
              f"(no prior model to compare)")
    print(f"leaderboard -> {LEADERBOARD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

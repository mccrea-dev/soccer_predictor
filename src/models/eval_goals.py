"""Out-of-sample check of the goal model on its own.

Fit on everything before TEST_START, predict every match after it, and ask two things: does it
beat the dumbest possible baseline (always predicting the historical over rate), and are its
probabilities calibrated? A model that fails either has no business being used.

This looks at over/under 2.5 in detail and at match results briefly. The fuller comparison of
result models is in backtest/walk_forward.py.

Run:  python soccer.py eval
"""
import sys
import numpy as np
import pandas as pd

from core import db
from core.calibration import ece, reliability_table
from models.goals_statistical import DixonColes
from data_pipeline.ingest_results import competition_of, TARGET_COMPETITIONS

TEST_START = "2021-01-01"


def logloss(p, y, eps=1e-12):
    """Log loss for yes/no predictions. Lower is better; it punishes confident misses hard."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, int)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def multiclass_logloss(P, y, eps=1e-12):
    """Log loss for several outcomes. P is (n, classes), y the index of what happened."""
    P = np.clip(np.asarray(P, float), eps, 1)
    P = P / P.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def auc(score, y):
    """AUC from ranks (the Mann-Whitney formula): the chance that a randomly chosen positive
    case was given a higher score than a randomly chosen negative one. 0.5 is no better than
    guessing."""
    score, y = np.asarray(score, float), np.asarray(y, int)
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    conn = db.connect()
    m = pd.read_sql("SELECT * FROM matches WHERE home_goals IS NOT NULL", conn)
    conn.close()
    m["date"] = pd.to_datetime(m["date"])
    m["competition"] = m["tournament"].map(competition_of)

    train = m[m["date"] < TEST_START]
    test = m[(m["date"] >= TEST_START) & (m["competition"].isin(TARGET_COMPETITIONS))].copy()
    print(f"train (all played, time-decayed) < {TEST_START}: {len(train)}  |  "
          f"test (target comps): {len(test)}")

    dc = DixonColes().fit(train, ref_date=TEST_START)

    # predict every test match with the model fitted on the earlier period
    p_over, p_h, p_d, p_a, e_tot = [], [], [], [], []
    for _, r in test.iterrows():
        lam, mu = dc.lambda_mu(r["home_team"], r["away_team"], int(r["neutral"]))
        mk = dc.markets(r["home_team"], r["away_team"], int(r["neutral"]))
        p_over.append(mk["over_2.5"]); p_h.append(mk["home"]); p_d.append(mk["draw"]); p_a.append(mk["away"])
        e_tot.append(lam + mu)
    test["p_over"] = p_over
    test["e_tot"] = e_tot
    over = (test["home_goals"] + test["away_goals"] > 2.5).astype(int).to_numpy()
    res = test.apply(lambda r: 0 if r.home_goals > r.away_goals else (2 if r.away_goals > r.home_goals else 1), axis=1).to_numpy()
    P3 = np.column_stack([p_h, p_d, p_a])

    # ---- Over/Under 2.5 ----
    base = train.eval("home_goals + away_goals > 2.5").mean()
    print("\nOVER/UNDER 2.5")
    print(f"  base rate (train): {base:.3f}  | test actual over-rate: {over.mean():.3f}")
    print(f"  log-loss   model {logloss(p_over, over):.4f}  vs baseline {logloss(np.full(len(over), base), over):.4f}")
    print(f"  brier      model {np.mean((np.array(p_over)-over)**2):.4f}  "
          f"vs baseline {np.mean((base-over)**2):.4f}")
    print(f"  AUC(P_over, over): {auc(p_over, over):.3f}   (discrimination; >0.5 = signal)")
    print(f"  ECE(P_over): {ece(p_over, over):.3f}   (calibration; <0.05 good)")
    print("  reliability (predicted -> realized, n):")
    for row in reliability_table(p_over, over, n_bins=10):
        print(f"    {row['pred']:.2f} -> {row['realized']:.2f}  (n={row['n']})")

    # ---- match result, for context ----
    freq = train["home_goals"].notna()
    hb = (train.home_goals > train.away_goals).mean()
    db_ = (train.home_goals == train.away_goals).mean()
    ab = (train.home_goals < train.away_goals).mean()
    base3 = np.tile([hb, db_, ab], (len(test), 1))
    acc = (P3.argmax(1) == res).mean()
    print("\nMATCH RESULT (home / draw / away)")
    print(f"  log-loss   model {multiclass_logloss(P3, res):.4f}  vs baseline {multiclass_logloss(base3, res):.4f}")
    print(f"  accuracy   model {acc:.3f}  vs always-home {hb:.3f}")
    print(f"  bias check: model E[total goals] {test['e_tot'].mean():.3f}  vs actual "
          f"{((test.home_goals+test.away_goals).mean()):.3f}  "
          f"(mean P(over) {np.mean(p_over):.3f} vs actual over-rate {over.mean():.3f})")

    # print a handful of recent predictions next to what happened
    print("\nsample test predictions:")
    for _, r in test.sort_values("date").tail(5).iterrows():
        tot = int(r.home_goals + r.away_goals)
        print(f"  {r.date.date()} {r.home_team[:14]:14s} v {r.away_team[:14]:14s}  "
              f"P(over2.5)={r.p_over:.2f}  actual total={tot} ({'OVER' if tot>2.5 else 'under'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

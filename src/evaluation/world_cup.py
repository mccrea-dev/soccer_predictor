"""Retrospective on the 2026 World Cup: what the models said before each match vs. what happened.

The tournament is over, so this is the scorecard. For every one of the 104 matches we rebuild the
forecast the models would have made that morning (trained only on matches played before that
day), then grade it against the real result.

Why re-predict instead of just grading the live log? The live tracker only covers the 34 matches
I had odds snapshots for, and some of its rows were filled in after the fact. Replaying the whole
tournament day by day gives a complete, reproducible, strictly out-of-sample record. The live log
is still graded at the end, mainly because it is the only place we have the market's pick to
compare against.

A note on "result": scores are as recorded at the end of play (extra time included). A knockout
game that went to penalties is a draw here, same as in the Elo ratings and the training data.

Outputs (all under reports/world_cup_2026/):
  match_predictions.csv   one row per match: probabilities, picks, actual result
  summary.json            headline numbers, machine readable
  REPORT.md               the tables, readable on GitHub
  figures/*.png           charts used in the README

Run:  python soccer.py worldcup
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from backtest.walk_forward import load_matches, rps, logloss, brier, HOME, DRAW, AWAY
from core import db
from core.calibration import fit_isotonic
from core.distributions import prob_1x2, prob_over_under, prob_btts
from core.geography import conf_of
from core.names import canonical
from engine.predict import OU_PICK_THRESHOLD
from models.calibrate_ou import CAL_END, HOLD_END
from models.goals_statistical import DixonColes
from models.winner import WinnerModel, add_dc_features
from data_pipeline.ingest_results import competition_of, TARGET_COMPETITIONS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "world_cup_2026"
FIGS = OUT / "figures"
LIVE_LOG = ROOT / "reports" / "live_tracker" / "predictions.csv"

WC_START, WC_END = "2026-06-11", "2026-07-19"

# The 48-team format: 72 group games, then 32 knockout games. Stages never share a calendar day,
# so once the matches are sorted by date the stage is just a matter of counting.
STAGES = [("Group stage", 72), ("Round of 32", 16), ("Round of 16", 8),
          ("Quarter-finals", 4), ("Semi-finals", 2), ("Third place", 1), ("Final", 1)]

LABELS = {HOME: "home", DRAW: "draw", AWAY: "away"}


# --------------------------------------------------------------------------- predictions
def _fit_ou_calibrator(m):
    """Same recipe as models/calibrate_ou.py: goal model fit on pre-2020 data, isotonic map fit on
    2020-2022. Refit here so the report doesn't depend on a pickle being present. None of it
    touches 2026."""
    played = m[m["competition"].isin(TARGET_COMPETITIONS)]
    cal = played[(played["date"] >= CAL_END) & (played["date"] < HOLD_END)]
    dc = DixonColes().fit(m[m["date"] < CAL_END], ref_date=CAL_END)
    p = [dc.prob_over(r.home_team, r.away_team, 2.5, int(r.neutral)) for r in cal.itertuples()]
    y = (cal["home_goals"] + cal["away_goals"] > 2.5).astype(int)
    return fit_isotonic(p, y)


def predict_tournament(m, wc, cal):
    """Walk through the tournament one match day at a time. Each day: refit everything on matches
    played before that day, then forecast that day's games. Returns one row per match."""
    rows = []
    days = sorted(wc["date"].unique())
    for k, day in enumerate(days, 1):
        today = wc[wc["date"] == day]
        train = m[m["date"] < day]

        dc = DixonColes().fit(train, ref_date=day)
        train_f, today_f = add_dc_features(train, dc), add_dc_features(today, dc)
        p_winner = WinnerModel(C=0.1).fit(train_f).predict_proba(today_f)

        # Baseline to beat: a two-feature logistic on Elo difference + home advantage.
        elo = LogisticRegression(max_iter=1000, C=1.0)
        elo.fit(train[["elo_diff", "hfa"]].to_numpy(float), train["y"].to_numpy())
        p_elo = np.zeros((len(today), 3))
        p_elo[:, elo.classes_] = elo.predict_proba(today[["elo_diff", "hfa"]].to_numpy(float))

        # And the laziest baseline of all: historical home/draw/away frequencies.
        base = np.array([(train["y"] == c).mean() for c in (HOME, DRAW, AWAY)])

        for i, r in enumerate(today.itertuples()):
            neutral = int(r.neutral)
            mtx = dc.score_matrix(r.home_team, r.away_team, neutral)
            lam, mu = dc.lambda_mu(r.home_team, r.away_team, neutral)
            over_raw = prob_over_under(mtx, 2.5)[0]
            top_i, top_j = np.unravel_index(int(np.argmax(mtx)), mtx.shape)
            pw, pe, pd_ = p_winner[i], p_elo[i], prob_1x2(mtx)
            rows.append({
                "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                "home_team": r.home_team, "away_team": r.away_team, "neutral": neutral,
                "elo_home": r.elo_h, "elo_away": r.elo_a,
                "p_home": pw[0], "p_draw": pw[1], "p_away": pw[2],
                "elo_p_home": pe[0], "elo_p_draw": pe[1], "elo_p_away": pe[2],
                "dc_p_home": pd_[0], "dc_p_draw": pd_[1], "dc_p_away": pd_[2],
                "base_p_home": base[0], "base_p_draw": base[1], "base_p_away": base[2],
                "exp_home_goals": lam, "exp_away_goals": mu,
                "p_over_2_5": float(cal.predict([over_raw])[0]),
                "p_btts": prob_btts(mtx)[0],
                "likely_score": f"{top_i}-{top_j}",
                "home_goals": int(r.home_goals), "away_goals": int(r.away_goals),
                "y": int(r.y), "shootout": int("shootout" in str(r.source)),
            })
        print(f"  [{k:2d}/{len(days)}] {pd.Timestamp(day).date()}  {len(today)} match(es)")
    return pd.DataFrame(rows)


def add_derived(df):
    """Picks, grades and a few convenience columns on top of the raw probabilities."""
    df = df.sort_values(["date", "home_team"]).reset_index(drop=True)
    stage = []
    for name, n in STAGES:
        stage += [name] * n
    df["stage"] = stage

    P = df[["p_home", "p_draw", "p_away"]].to_numpy()
    pick = P.argmax(1)
    names = np.column_stack([df["home_team"], np.full(len(df), "Draw"), df["away_team"]])
    idx = np.arange(len(df))
    df["pick"] = names[idx, pick]
    df["confidence"] = P.max(1)
    df["actual"] = names[idx, df["y"].to_numpy()]
    df["correct"] = (pick == df["y"]).astype(int)
    df["p_actual"] = P[idx, df["y"].to_numpy()]       # probability we gave to what happened

    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["exp_total_goals"] = df["exp_home_goals"] + df["exp_away_goals"]
    df["ou_pick"] = np.where(df["p_over_2_5"] >= OU_PICK_THRESHOLD, "Over", "Under")
    df["ou_actual"] = np.where(df["total_goals"] > 2.5, "Over", "Under")
    df["ou_correct"] = (df["ou_pick"] == df["ou_actual"]).astype(int)
    df["score_correct"] = (df["likely_score"]
                           == df["home_goals"].astype(str) + "-" + df["away_goals"].astype(str)).astype(int)
    # the favourite by pre-match Elo (no home bump: only the three hosts ever play at home)
    df["elo_favourite"] = np.where(df["elo_home"] >= df["elo_away"], df["home_team"], df["away_team"])
    return df


# --------------------------------------------------------------------------- scoring
def _probs(df, prefix):
    return df[[f"{prefix}home", f"{prefix}draw", f"{prefix}away"]].to_numpy()


def score_models(df):
    """Accuracy and the proper scoring rules for each model on the same 104 matches."""
    y = df["y"].to_numpy()
    out = {}
    for name, prefix in [("Winner model", "p_"), ("Dixon-Coles", "dc_p_"),
                         ("Elo logistic", "elo_p_"), ("Base rates", "base_p_")]:
        P = _probs(df, prefix)
        out[name] = {"accuracy": float((P.argmax(1) == y).mean()), "rps": rps(P, y),
                     "log_loss": logloss(P, y), "brier": brier(P, y)}
    # "just pick the higher-rated team" can never call a draw, so it only gets an accuracy
    out["Higher Elo wins"] = {"accuracy": float((df["elo_favourite"] == df["actual"]).mean()),
                              "rps": None, "log_loss": None, "brier": None}
    return out


def bootstrap_ci(correct, n_boot=10000, seed=7):
    """95% interval on accuracy. 104 matches is a small sample and the report should say so."""
    rng = np.random.default_rng(seed)
    c = np.asarray(correct)
    means = c[rng.integers(0, len(c), (n_boot, len(c)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def by_stage(df):
    g = df.groupby("stage", sort=False)
    t = g.agg(matches=("correct", "size"), winner_acc=("correct", "mean"),
              ou_acc=("ou_correct", "mean"), draws=("y", lambda s: int((s == DRAW).sum())),
              goals_per_match=("total_goals", "mean"),
              exp_goals_per_match=("exp_total_goals", "mean")).reset_index()
    return t


def by_confidence(df, edges=(0.0, 0.45, 0.55, 0.65, 0.75, 1.01)):
    """Does a 70% pick actually win about 70% of the time?"""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = df[(df["confidence"] >= lo) & (df["confidence"] < hi)]
        if len(g):
            label = f"{lo*100:.0f}-{min(hi, 1)*100:.0f}%" if lo > 0 else f"under {hi*100:.0f}%"
            rows.append({"confidence": label, "matches": len(g),
                         "avg_confidence": g["confidence"].mean(), "hit_rate": g["correct"].mean()})
    return pd.DataFrame(rows)


def team_table(df):
    """Per-team results against expectation. 'Expected points' uses 1 for a win and 0.5 for a draw,
    so over/under-performance is in the same units as Elo's match score."""
    rows = []
    for side, opp, p_win, p_loss in [("home_team", "away_team", "p_home", "p_away"),
                                     ("away_team", "home_team", "p_away", "p_home")]:
        gf = "home_goals" if side == "home_team" else "away_goals"
        ga = "away_goals" if side == "home_team" else "home_goals"
        won = HOME if side == "home_team" else AWAY
        t = pd.DataFrame({
            "team": df[side], "played": 1,
            "points": np.where(df["y"] == won, 1.0, np.where(df["y"] == DRAW, 0.5, 0.0)),
            "expected": df[p_win] + 0.5 * df["p_draw"],
            "gf": df[gf], "ga": df[ga], "stage": df["stage"]})
        rows.append(t)
    t = pd.concat(rows)
    order = {name: i for i, (name, _) in enumerate(STAGES)}
    out = t.groupby("team").agg(played=("played", "sum"), points=("points", "sum"),
                                expected=("expected", "sum"), gf=("gf", "sum"), ga=("ga", "sum"),
                                reached=("stage", lambda s: max(s, key=order.get))).reset_index()
    out["vs_expected"] = out["points"] - out["expected"]
    return out.sort_values("vs_expected", ascending=False).reset_index(drop=True)


def elo_changes(conn, teams):
    """Elo going into the tournament and coming out of it, for every team that took part."""
    rows = []
    for t in teams:
        before = db.rating_before(conn, t, "elo", WC_START)
        after = db.rating_asof(conn, t, "elo", WC_END)
        rows.append({"team": t, "elo_before": before, "elo_after": after, "change": after - before})
    out = pd.DataFrame(rows).sort_values("elo_before", ascending=False).reset_index(drop=True)
    out["pre_rank"] = np.arange(1, len(out) + 1)
    return out


def grade_live_log(df):
    """Grade the in-tournament tracker log (model pick vs. the betting market's pick).

    The log stores games as the odds feed listed them, which doesn't always match the home/away
    order in the results data, so games are matched on the pair of teams plus the date (+/- a day
    for time zones)."""
    if not LIVE_LOG.exists():
        return None
    log = pd.read_csv(LIVE_LOG)
    results = {}
    for r in df.itertuples():
        results[(frozenset((r.home_team, r.away_team)), r.date)] = r
    graded = []
    for r in log.itertuples():
        a, b = [canonical(x.strip()) for x in r.Game.split(" - ")]
        match = None
        for shift in (0, -1, 1):
            d = (pd.Timestamp(r.Date) + pd.Timedelta(days=shift)).strftime("%Y-%m-%d")
            match = results.get((frozenset((a, b)), d))
            if match is not None:
                break
        if match is None:
            continue
        graded.append({
            "date": r.Date, "game": r.Game, "market_pick": r[3], "model_pick": canonical(r[4]),
            "actual": match.actual, "total_goals": match.total_goals,
            "model_ou": r[6], "ou_actual": match.ou_actual,
            # what the day-by-day replay above picks for the same game, for comparison
            "replay_pick": match.pick})
    g = pd.DataFrame(graded)
    g["market_pick"] = g["market_pick"].map(canonical)
    g["model_correct"] = (g["model_pick"] == g["actual"]).astype(int)
    g["market_correct"] = (g["market_pick"] == g["actual"]).astype(int)
    g["replay_correct"] = (g["replay_pick"] == g["actual"]).astype(int)
    g["ou_correct"] = (g["model_ou"] == g["ou_actual"]).astype(int)
    return g


# --------------------------------------------------------------------------- charts
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12, fontweight="bold")
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(df, scores, stage_tbl, conf_tbl, teams):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    FIGS.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        fig.savefig(FIGS / name, dpi=160, facecolor=SURFACE)
        plt.close(fig)

    # 1. accuracy by model
    order = ["Winner model", "Dixon-Coles", "Elo logistic", "Higher Elo wins", "Base rates"]
    acc = [scores[k]["accuracy"] for k in order]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.bar(order, acc, width=0.55, color=[BLUE] + ["#b7b6b0"] * 4)
    for b, v in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v*100:.1f}%", ha="center",
                fontsize=10, color=INK)
    ax.set_ylim(0, max(acc) + 0.12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    _style(ax, "Correct result picks, 2026 World Cup (104 matches)")
    save(fig, "accuracy_by_model.png")

    # The last four rounds are only eight matches between them, far too few to chart one by one,
    # so the charts lump them together. REPORT.md keeps the full round-by-round table.
    late = {"Quarter-finals", "Semi-finals", "Third place", "Final"}
    stage_tbl = by_stage(df.assign(stage=df["stage"].where(~df["stage"].isin(late), "Quarter-finals on")))

    # 2. accuracy by stage
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(stage_tbl))
    bars = ax.bar(x, stage_tbl["winner_acc"], width=0.55, color=BLUE)
    for b, v, n in zip(bars, stage_tbl["winner_acc"], stage_tbl["matches"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v*100:.0f}%\n(n={n})", ha="center",
                fontsize=9, color=INK)
    ax.set_xticks(x, stage_tbl["stage"])
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    _style(ax, "Winner-model accuracy by stage")
    save(fig, "accuracy_by_stage.png")

    # 3. calibration: stated confidence vs. how often the pick came in
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0.3, 1], [0.3, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
    ax.text(0.94, 0.9, "perfect calibration", color=MUTED, fontsize=8, ha="right", rotation=38)
    ax.plot(conf_tbl["avg_confidence"], conf_tbl["hit_rate"], color=BLUE, linewidth=2, marker="o",
            markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    for r in conf_tbl.itertuples():
        ax.annotate(f"n={r.matches}", (r.avg_confidence, r.hit_rate), textcoords="offset points",
                    xytext=(8, -12), fontsize=8, color=MUTED)
    ax.set_xlim(0.3, 1); ax.set_ylim(0.3, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0)); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Model confidence in its pick", color=MUTED, fontsize=9)
    ax.set_ylabel("Share of picks that were right", color=MUTED, fontsize=9)
    _style(ax, "Calibration of the winner model")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    save(fig, "calibration.png")

    # 4. teams vs. expectation
    top, bottom = teams.head(8), teams.tail(8)
    t = pd.concat([top, bottom]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = [BLUE if v >= 0 else ORANGE for v in t["vs_expected"]]
    ax.barh(t["team"], t["vs_expected"], color=colors, height=0.6)
    for y_, v in enumerate(t["vs_expected"]):
        ax.text(v + (0.03 if v >= 0 else -0.03), y_, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    lim = max(abs(t["vs_expected"])) + 0.35
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Match points above / below model expectation (win = 1, draw = 0.5)",
                  color=MUTED, fontsize=9)
    _style(ax, "Who beat the model's expectations, and who fell short")
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, linewidth=0.8)
    save(fig, "teams_vs_expectation.png")

    # 5. goals: expected vs. actual, by stage
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, stage_tbl["exp_goals_per_match"], width=w, color=BLUE, label="Model expected")
    b2 = ax.bar(x + w / 2 + 0.01, stage_tbl["goals_per_match"], width=w, color=ORANGE, label="Actual")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, f"{b.get_height():.1f}",
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, stage_tbl["stage"])
    ax.set_ylim(0, max(stage_tbl["goals_per_match"].max(), 4) * 1.18)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncols=2)
    _style(ax, "Goals per match: model expectation vs. actual")
    save(fig, "goals_by_stage.png")


# --------------------------------------------------------------------------- report
def _md(df, floats=None):
    """Tiny DataFrame -> GitHub markdown table (saves depending on tabulate)."""
    floats = floats or {}
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(" --- " for _ in df.columns) + "|"
    lines = [head, sep]
    for row in df.itertuples(index=False):
        cells = []
        for c, v in zip(df.columns, row):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("–")
            elif c in floats:
                cells.append(floats[c].format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(df, scores, ci, stage_tbl, conf_tbl, teams, elo, live):
    n = len(df)
    L = []
    L.append("# 2026 World Cup: predictions vs. results\n")
    L.append("Generated by `python soccer.py worldcup`. Every forecast below was made by models "
             "trained only on matches played before that match day. Results are the score at the "
             "end of play; a knockout game decided on penalties counts as a draw.\n")

    L.append("## Headline\n")
    w = scores["Winner model"]
    L.append(f"- Result (home / draw / away) called correctly in **{df['correct'].sum()} of {n} "
             f"matches ({w['accuracy']*100:.1f}%)**, 95% bootstrap interval "
             f"{ci[0]*100:.0f}-{ci[1]*100:.0f}%.")
    L.append(f"- Over/Under 2.5 goals called correctly in **{df['ou_correct'].sum()} of {n} "
             f"({df['ou_correct'].mean()*100:.1f}%)**.")
    L.append(f"- Most likely scoreline was exactly right in {df['score_correct'].sum()} matches "
             f"({df['score_correct'].mean()*100:.1f}%).")
    L.append(f"- Average probability given to the outcome that actually happened: "
             f"{df['p_actual'].mean()*100:.1f}% (a coin-flip across three outcomes would be 33.3%).\n")

    L.append("## Model comparison (same 104 matches)\n")
    t = pd.DataFrame([{"model": k, **v} for k, v in scores.items()])
    t["accuracy"] = t["accuracy"] * 100
    L.append(_md(t.rename(columns={"accuracy": "accuracy %", "rps": "RPS", "log_loss": "log loss",
                                   "brier": "Brier"}),
                 {"accuracy %": "{:.1f}", "RPS": "{:.4f}", "log loss": "{:.4f}", "Brier": "{:.4f}"}))
    L.append("\nRPS (ranked probability score), log loss and Brier are all lower-is-better. "
             "\"Higher Elo wins\" never predicts a draw, so it has no probabilities to score.\n")

    L.append("## By stage\n")
    s = stage_tbl.copy()
    s["winner_acc"] *= 100; s["ou_acc"] *= 100
    L.append(_md(s.rename(columns={"winner_acc": "result acc %", "ou_acc": "O/U acc %",
                                   "goals_per_match": "goals / match",
                                   "exp_goals_per_match": "expected goals / match"}),
                 {"result acc %": "{:.0f}", "O/U acc %": "{:.0f}", "goals / match": "{:.2f}",
                  "expected goals / match": "{:.2f}"}))

    L.append("\n## Calibration: confidence vs. hit rate\n")
    c = conf_tbl.copy()
    c["avg_confidence"] *= 100; c["hit_rate"] *= 100
    L.append(_md(c.rename(columns={"avg_confidence": "avg confidence %", "hit_rate": "hit rate %"}),
                 {"avg confidence %": "{:.0f}", "hit rate %": "{:.0f}"}))

    L.append("\n## Draws\n")
    n_draw, n_pick_draw = int((df["y"] == DRAW).sum()), int((df["pick"] == "Draw").sum())
    L.append(f"- {n_draw} of {n} matches ({n_draw/n*100:.0f}%) were level at the end of play; "
             f"the model's average draw probability was {df['p_draw'].mean()*100:.0f}%.")
    L.append(f"- The model made a draw its top pick {n_pick_draw} time(s). Draws are rarely the "
             f"single most likely outcome, so nearly every draw is a guaranteed miss on accuracy. "
             f"Excluding draws, the model picked the winner in "
             f"{df[df['y'] != DRAW]['correct'].mean()*100:.1f}% of {n - n_draw} decisive matches.\n")

    L.append("## Goals\n")
    L.append(f"- {int(df['total_goals'].sum())} goals, {df['total_goals'].mean():.2f} per match; "
             f"the goal model expected {df['exp_total_goals'].mean():.2f}.")
    L.append(f"- {df['ou_actual'].eq('Over').mean()*100:.0f}% of matches went over 2.5; the model "
             f"called Over in {df['ou_pick'].eq('Over').mean()*100:.0f}%.")
    err = (df["exp_total_goals"] - df["total_goals"]).abs().mean()
    L.append(f"- Mean absolute error on total goals: {err:.2f}.\n")

    cols = ["date", "stage", "home_team", "score", "away_team", "pick", "p_actual"]
    d = df.assign(score=df["home_goals"].astype(str) + "-" + df["away_goals"].astype(str),
                  p_actual=df["p_actual"] * 100)
    L.append("## Biggest surprises (lowest probability given to what happened)\n")
    L.append(_md(d.nsmallest(8, "p_actual")[cols].rename(columns={"p_actual": "p(actual) %"}),
                 {"p(actual) %": "{:.0f}"}))
    L.append("\n## Most confident correct calls\n")
    L.append(_md(d[d["correct"] == 1].nlargest(8, "p_actual")[cols].rename(columns={"p_actual": "p(actual) %"}),
                 {"p(actual) %": "{:.0f}"}))

    L.append("\n## Teams vs. expectation\n")
    L.append("Match points use win = 1, draw = 0.5. Positive means the team did better than the "
             "model expected over the games it played.\n")
    tt = teams[["team", "played", "reached", "points", "expected", "vs_expected", "gf", "ga"]]
    fm = {"points": "{:.1f}", "expected": "{:.2f}", "vs_expected": "{:+.2f}"}
    L.append("**Over-performers**\n")
    L.append(_md(tt.head(8), fm))
    L.append("\n**Under-performers**\n")
    L.append(_md(tt.tail(8).iloc[::-1], fm))

    L.append("\n## Pre-tournament Elo top 10 and how far they went\n")
    e = elo.head(10).merge(teams[["team", "reached"]], on="team")
    L.append(_md(e[["pre_rank", "team", "elo_before", "reached", "elo_after", "change"]],
                 {"elo_before": "{:.0f}", "elo_after": "{:.0f}", "change": "{:+.0f}"}))
    L.append("\n**Biggest Elo movers over the tournament**\n")
    mv = pd.concat([elo.nlargest(5, "change"), elo.nsmallest(5, "change")])
    L.append(_md(mv[["team", "elo_before", "elo_after", "change"]],
                 {"elo_before": "{:.0f}", "elo_after": "{:.0f}", "change": "{:+.0f}"}))

    if live is not None and len(live):
        L.append("\n## Live tracker: model vs. the betting market\n")
        L.append(f"During the tournament I logged the model's pick next to the de-vigged market "
                 f"favourite for {len(live)} matches I had odds snapshots for.\n")
        dis = live[live["model_pick"] != live["market_pick"]]
        L.append(f"- Logged model pick correct: {live['model_correct'].sum()}/{len(live)} "
                 f"({live['model_correct'].mean()*100:.0f}%)")
        L.append(f"- Market favourite correct: {live['market_correct'].sum()}/{len(live)} "
                 f"({live['market_correct'].mean()*100:.0f}%)")
        L.append(f"- Day-by-day replay pick on the same matches: {live['replay_correct'].sum()}/{len(live)} "
                 f"({live['replay_correct'].mean()*100:.0f}%)")
        L.append(f"- Logged Over/Under pick correct: {live['ou_correct'].sum()}/{len(live)} "
                 f"({live['ou_correct'].mean()*100:.0f}%)")
        L.append(f"- The logged pick and the replay pick differ on "
                 f"{int((live['model_pick'] != live['replay_pick']).sum())} of the {len(live)} matches.")
        if len(dis):
            L.append(f"- Model and market disagreed on {len(dis)} matches: model right in "
                     f"{dis['model_correct'].sum()}, market right in {dis['market_correct'].sum()}.")
        L.append("\nCaveat: the tracker was first run part-way through the tournament and back-filled "
                 "picks for games that had already been played, so for those rows the model had seen "
                 "the result. The replay number is the clean one to compare with the market.\n")

    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading matches + building features ...")
    m = load_matches()
    wc = m[(m["tournament"] == "FIFA World Cup") & (m["date"] >= WC_START) & (m["date"] <= WC_END)]
    if len(wc) != sum(n for _, n in STAGES):
        print(f"expected 104 World Cup matches, found {len(wc)} -- run ingest-results + compute-elo first.")
        return 1

    print("fitting Over/Under calibrator (pre-2023 data only) ...")
    cal = _fit_ou_calibrator(m)
    print("replaying the tournament day by day ...")
    df = add_derived(predict_tournament(m, wc, cal))

    scores = score_models(df)
    ci = bootstrap_ci(df["correct"])
    stage_tbl, conf_tbl, teams = by_stage(df), by_confidence(df), team_table(df)
    conn = db.connect()
    elo = elo_changes(conn, teams["team"])
    conn.close()
    live = grade_live_log(df)

    round_cols = [c for c in df.columns if c.startswith(("p_", "elo_p_", "dc_p_", "base_p_", "exp_"))
                  or c == "confidence"]
    df[round_cols] = df[round_cols].round(4)
    df.drop(columns=["y"]).to_csv(OUT / "match_predictions.csv", index=False)
    teams.round(3).to_csv(OUT / "team_performance.csv", index=False)
    if live is not None:
        live.to_csv(LIVE_LOG.with_name("predictions_graded.csv"), index=False)

    summary = {
        "matches": int(len(df)),
        "winner_accuracy": scores["Winner model"]["accuracy"], "winner_accuracy_ci95": ci,
        "over_under_accuracy": float(df["ou_correct"].mean()),
        "exact_score_accuracy": float(df["score_correct"].mean()),
        "draws": int((df["y"] == DRAW).sum()),
        "goals_per_match": float(df["total_goals"].mean()),
        "expected_goals_per_match": float(df["exp_total_goals"].mean()),
        "models": scores,
    }
    if live is not None and len(live):
        summary["live_log"] = {"matches": int(len(live)),
                               "model_accuracy": float(live["model_correct"].mean()),
                               "market_accuracy": float(live["market_correct"].mean()),
                               "replay_accuracy": float(live["replay_correct"].mean())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    make_figures(df, scores, stage_tbl, conf_tbl, teams)
    write_report(df, scores, ci, stage_tbl, conf_tbl, teams, elo, live)

    print(f"\nresult accuracy {scores['Winner model']['accuracy']*100:.1f}%  |  "
          f"O/U accuracy {df['ou_correct'].mean()*100:.1f}%  |  report -> {OUT / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

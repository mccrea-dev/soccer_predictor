"""Prediction tracker: write down what the model and the market each picked before a match, then
grade both once the result is in.

  log    record a prediction for every match that has odds and isn't logged yet: the model's
         winner, the market's favourite, and the model's over/under pick. A match is only
         ever logged once.
  grade  fill in the actual winner and total goals for logged matches that have been played
  show   print the table with accuracy figures and export it to reports/live_tracker/ as
         CSV, HTML and a colour-coded Excel sheet

Running `track` with no sub-command does all three. Nothing here calls the odds API; it only
reads what's already stored. Run ingest-results first so newly finished games can be graded.

One thing to know when reading the log: if a match has already been played by the time it is
first logged, the model has seen its result, so that row is not a real forecast. For clean
numbers, log before kick-off, or use the day-by-day replay in evaluation/world_cup.py.

Run:  python soccer.py track [log|grade|show]
"""
import sys
import numpy as np
import pandas as pd

from core import db
from core.ev import decimal_to_american
from models.goals_statistical import DixonColes, fit_from_db
from models.winner import fit_and_predict
from engine.predict import forecast, _market_1x2, OU_PICK_THRESHOLD
from engine.pricing import _latest_odds, get_neutral, load_ou_calibrator

OUT = db.DB_PATH.parents[1] / "reports" / "live_tracker"
CSV = OUT / "predictions.csv"
HTML = OUT / "predictions.html"
XLSX = OUT / "predictions.xlsx"


def _model():
    """Load the saved goal model, fitting and saving one first if there isn't one yet."""
    try:
        return DixonColes.load()
    except Exception:
        dc = fit_from_db(); dc.save(); return dc


def log(conn):
    """Record a prediction for every match that has odds stored and isn't in the log yet."""
    dc, cal = _model(), load_ou_calibrator()
    mks = [r[0] for r in conn.execute("SELECT DISTINCT match_key FROM odds_snapshots")]
    # Winner-model probabilities for every match with odds. Ones missing from the matches table
    # are passed in as extra fixtures so they still get features.
    have = {r[0] for r in conn.execute("SELECT match_key FROM matches")}
    extra = []
    for mk in mks:
        if mk in have:
            continue
        d, h, a = mk.split("|")
        extra.append({"match_key": mk, "date": d, "tournament": "Friendly", "home_team": h,
                      "away_team": a, "neutral": get_neutral(conn, d, h, a),
                      "home_goals": None, "away_goals": None})
    _, wprobs = fit_and_predict(conn, extra=extra or None, all_rows=True)
    rows = []
    for mk in mks:
        if conn.execute("SELECT 1 FROM predictions WHERE match_key=?", (mk,)).fetchone():
            continue                      # already logged, leave it alone
        date, home, away = mk.split("|")
        neutral = get_neutral(conn, date, home, away)
        f = forecast(dc, home, away, neutral, cal, win=wprobs.get(mk))
        df = _latest_odds(conn, mk)
        mkt = _market_1x2(df)
        if mkt:                           # the market's pick is whichever outcome it rates most likely
            i = int(np.argmax(mkt))
            market_winner, market_p = [home, "Draw", away][i], float(mkt[i])
        else:
            market_winner, market_p = None, None
        rows.append({
            "match_key": mk, "logged_ts": pd.Timestamp.now("UTC").isoformat(), "date": date,
            "game": f"{home} - {away}", "neutral": neutral,
            "model_winner": f["winner"], "model_winner_p": round(f["conf"], 3),
            "market_winner": market_winner,
            "market_winner_p": round(market_p, 3) if market_p else None,
            "model_ou": f["ou_pick"], "model_over_p": round(f["over"], 3),
            "actual_winner": None, "total_goals": None, "actual_ou": None, "graded": 0,
        })
    if rows:
        db.upsert(conn, "predictions", rows, ["match_key"])
    print(f"logged {len(rows)} new prediction(s) ({len(mks)-len(rows)} already on record)")
    return len(rows)


def grade(conn):
    """Fill in the actual winner and total goals for logged matches that now have a result.
    The date is allowed to be a day out either way, since odds feeds use UTC kick-off dates."""
    pend = pd.read_sql("SELECT * FROM predictions WHERE graded=0", conn)
    n = 0
    for _, p in pend.iterrows():
        d, h, a = p["match_key"].split("|")
        row = conn.execute(
            "SELECT home_goals, away_goals FROM matches WHERE home_team=? AND away_team=? "
            "AND ABS(julianday(date)-julianday(?))<=1 AND home_goals IS NOT NULL LIMIT 1",
            (h, a, d)).fetchone()
        if not row:
            continue
        hg, ag = int(row[0]), int(row[1])
        winner = h if hg > ag else a if ag > hg else "Draw"
        tot = hg + ag
        conn.execute(
            "UPDATE predictions SET actual_winner=?, total_goals=?, actual_ou=?, graded=1 "
            "WHERE match_key=?",
            (winner, tot, "Over" if tot > 2.5 else "Under", p["match_key"]))
        n += 1
    conn.commit()
    print(f"graded {n} newly-finished prediction(s)")
    return n


# columns shown in the exported table, and the headers they get
VIEW_COLS = ["date", "game", "market_winner", "model_winner", "actual_winner",
             "model_ou", "model_over_p", "total_goals"]
HEADERS = {"date": "Date", "game": "Game", "market_winner": "Pred Winner (Market)",
           "model_winner": "Pred Winner (Model)", "actual_winner": "Actual Winner",
           "model_ou": "O/U 2.5 (Model)", "model_over_p": "P(Over)", "total_goals": "Total Goals"}


def _accuracy(g):
    """Accuracy figures for a set of graded predictions: model winner, market winner, model
    over/under, and how each did on the matches where model and market picked differently.
    """
    out = {}
    if len(g):
        out["model_winner_acc"] = (g["model_winner"] == g["actual_winner"]).mean()
        mk = g[g["market_winner"].notna()]
        out["market_winner_acc"] = (mk["market_winner"] == mk["actual_winner"]).mean() if len(mk) else np.nan
        out["model_ou_acc"] = (g["model_ou"] == g["actual_ou"]).mean()
        # the interesting subset: matches where the model and the market disagreed
        dis = g[(g["market_winner"].notna()) & (g["model_winner"] != g["market_winner"])]
        out["n_disagree"] = len(dis)
        if len(dis):
            out["model_right_when_disagree"] = (dis["model_winner"] == dis["actual_winner"]).mean()
            out["market_right_when_disagree"] = (dis["market_winner"] == dis["actual_winner"]).mean()
    return out


def show(conn):
    # recompute the over/under pick from the stored probability, so a change to the threshold
    # applies to old rows as well
    """Print the log and accuracy summary, and write the CSV / HTML / Excel exports."""
    conn.execute("UPDATE predictions SET model_ou = CASE WHEN model_over_p >= ? THEN 'Over' ELSE 'Under' END",
                 (OU_PICK_THRESHOLD,))
    conn.commit()
    df = pd.read_sql("SELECT * FROM predictions ORDER BY date, game", conn)
    if df.empty:
        print("no predictions logged yet. Run `python soccer.py track` after an odds snapshot.")
        return
    view = df[VIEW_COLS].rename(columns=HEADERS).fillna("")
    OUT.mkdir(parents=True, exist_ok=True)
    view.to_csv(CSV, index=False)
    _write_html(df)
    export_excel(conn)

    print("\n" + "=" * 110)
    print("PREDICTION LOG")
    print("=" * 110)
    print(view.to_string(index=False))

    g = df[df["graded"] == 1]
    a = _accuracy(g)
    print("\n" + "-" * 110)
    print(f"ACCURACY over {len(g)} graded game(s)   (ungraded: {len(df)-len(g)})")
    if a:
        print(f"  Model winner:  {a['model_winner_acc']*100:.0f}%   |   "
              f"Market winner: {a['market_winner_acc']*100:.0f}%   |   "
              f"Model O/U 2.5: {a['model_ou_acc']*100:.0f}%")
        if a.get("n_disagree"):
            print(f"  When model and market disagreed on the winner ({a['n_disagree']} games): "
                  f"model right {a['model_right_when_disagree']*100:.0f}%, "
                  f"market right {a['market_right_when_disagree']*100:.0f}%")
    print(f"\nsaved -> {XLSX}\n         {CSV}\n         {HTML}")


def _write_html(df):
    """Write the log as a small standalone HTML page, correct picks in green and misses in red."""
    g = df[df["graded"] == 1]
    a = _accuracy(g)
    rows = []
    for _, r in df.iterrows():
        played = r["graded"] == 1
        win_cls = ("ok" if played and r["model_winner"] == r["actual_winner"]
                   else "no" if played else "")
        ou_cls = ("ok" if played and r["model_ou"] == r["actual_ou"] else "no" if played else "")
        rows.append(
            f"<tr><td>{r['date']}</td><td>{r['game']}</td>"
            f"<td>{r['market_winner'] or ''}</td>"
            f"<td class='{win_cls}'>{r['model_winner']}</td>"
            f"<td>{r['actual_winner'] or '—'}</td>"
            f"<td class='{ou_cls}'>{r['model_ou']} ({r['model_over_p']*100:.0f}%)</td>"
            f"<td>{'' if pd.isna(r['total_goals']) else int(r['total_goals'])}</td></tr>")
    summ = ""
    if a:
        summ = (f"Graded {len(g)} — Model winner <b>{a['model_winner_acc']*100:.0f}%</b>, "
                f"Market <b>{a['market_winner_acc']*100:.0f}%</b>, "
                f"Model O/U <b>{a['model_ou_acc']*100:.0f}%</b>")
    html = f"""<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Soccer predictions</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:12px;background:#0f1115;color:#e6e6e6}}
 h2{{margin:.2em 0}} .sub{{color:#9aa;margin-bottom:10px}}
 table{{border-collapse:collapse;width:100%;font-size:14px}}
 th,td{{padding:7px 9px;border-bottom:1px solid #2a2f3a;text-align:left;white-space:nowrap}}
 th{{position:sticky;top:0;background:#1a1e27}}
 td.ok{{color:#5ad17a;font-weight:600}} td.no{{color:#e2706e;font-weight:600}}
 tr:nth-child(even){{background:#151922}}
</style>
<h2>Soccer prediction log</h2><div class=sub>{summ}</div>
<table><tr><th>Date</th><th>Game</th><th>Market</th><th>Model</th><th>Actual</th>
<th>O/U 2.5 (Model)</th><th>Total</th></tr>
{''.join(rows)}
</table>"""
    HTML.write_text(html, encoding="utf-8")


def _market_american(conn, match_key, market_winner):
    """Median American odds across books for the market's pick, so the sheet can show
    'Argentina (-800)'. None if there are no odds stored for it."""
    if not market_winner:
        return None
    date, home, away = match_key.split("|")
    side = "home" if market_winner == home else "away" if market_winner == away else "draw"
    df = _latest_odds(conn, match_key)
    g = df[(df["market"] == "1x2") & (df["line"] == 0.0) & (df["side"] == side)]
    if g.empty:
        return None
    return int(round(decimal_to_american(float(np.median(g["decimal_odds"])))))


def export_excel(conn, path=XLSX):
    """Write the log to Excel in the layout I was originally keeping by hand: Date, Game,
    Market pick, Model pick, Actual, O/U, Total goals. The Model and O/U cells get conditional
    formatting (green when right, red when wrong) that Excel evaluates itself, so a result
    typed into the sheet by hand still colours correctly."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.formatting.rule import FormulaRule
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("(openpyxl not installed -> skipping .xlsx. `pip install openpyxl` to enable)")
        return

    df = pd.read_sql("SELECT * FROM predictions ORDER BY date, game", conn)
    wb = Workbook(); ws = wb.active; ws.title = "Predictions"
    headers = ["Date", "Game", "Predicted Winner (Market)", "Predicted Winner (Model)",
               "Actual Result", "O/U 2.5 (Model)", "Total Goals"]
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        h = ws.cell(1, c); h.font = Font(bold=True); h.alignment = Alignment(horizontal="center")

    for _, r in df.iterrows():
        amer = _market_american(conn, r["match_key"], r["market_winner"])
        market = (f"{r['market_winner']} ({amer:+d})" if r["market_winner"] and amer is not None
                  else (r["market_winner"] or ""))
        model = f"{r['model_winner']} ({r['model_winner_p']*100:.0f}%)"
        ou = "O" if r["model_ou"] == "Over" else "U"
        total = "" if pd.isna(r["total_goals"]) else int(r["total_goals"])
        ws.append([r["date"], r["game"], market, model, r["actual_winner"] or "", ou, total])

    n = ws.max_row
    green = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    red = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    if n >= 2:
        # Model pick (column D) is right when the actual result's text appears inside the cell,
        # e.g. 'Brazil' inside 'Brazil (62%)'.
        ws.conditional_formatting.add(f"D2:D{n}",
            FormulaRule(formula=['AND($E2<>"",ISNUMBER(SEARCH($E2,$D2)))'], fill=green))
        ws.conditional_formatting.add(f"D2:D{n}",
            FormulaRule(formula=['AND($E2<>"",ISNUMBER(SEARCH($E2,$D2))=FALSE)'], fill=red))
        # O/U pick (column F): green for O with 3+ goals or U with 2 or fewer, red otherwise.
        ws.conditional_formatting.add(f"F2:F{n}",
            FormulaRule(formula=['AND(ISNUMBER($G2),OR(AND($F2="O",$G2>2.5),AND($F2="U",$G2<2.5)))'], fill=green))
        ws.conditional_formatting.add(f"F2:F{n}",
            FormulaRule(formula=['AND(ISNUMBER($G2),OR(AND($F2="O",$G2<2.5),AND($F2="U",$G2>2.5)))'], fill=red))

    for i, w in enumerate([12, 34, 26, 26, 16, 16, 12], start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    wb.save(path)


def main():
    conn = db.connect(); db.init_db(conn)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("all", "log"):
        log(conn)
    if cmd in ("all", "grade"):
        grade(conn)
    show(conn)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

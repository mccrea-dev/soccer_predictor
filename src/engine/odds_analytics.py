"""What the stored odds history says about how prices moved.

Three things are computed from the snapshots:
  * line movement: the change in a selection's implied probability between the first and
    last snapshot
  * steam: a large jump between two consecutive snapshots, which usually means informed money
    arrived
  * closing line value (CLV): whether a price taken earlier was better than the final price
    before kick-off. Beating the closing price consistently is the standard sign of real skill
    in betting, and it shows up long before profit and loss stops being noise.

Movements are measured in basis points of implied probability (100 bps = 1 percentage point).

One thing deliberately left out is "reverse line movement", a price moving against the side most
bets are on. That needs data on where the bets are going, which a public odds feed doesn't have,
and faking it from prices alone would just be line movement under another name.
"""
import numpy as np
import pandas as pd

from core import db
from core.ev import decimal_to_prob

KEY = ["match_key", "market", "line", "side"]   # what identifies one selection


def _load(conn, match_key=None):
    """Odds snapshots as a DataFrame with parsed timestamps, optionally for one match only."""
    q = "SELECT * FROM odds_snapshots"
    params = []
    if match_key:
        q += " WHERE match_key=?"
        params = [match_key]
    df = pd.read_sql(q, conn, params=params)
    if df.empty:
        return df
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True, errors="coerce")
    return df


def best_price(conn, match_key, market, line, side, at_ts=None):
    """Best decimal odds across books for one selection, at a given snapshot time or, by
    default, the latest one."""
    df = _load(conn, match_key)
    g = df[(df["market"] == market) & (df["line"] == line) & (df["side"] == side)]
    if g.empty:
        return None
    ts = at_ts or g["snapshot_ts"].max()
    g = g[g["snapshot_ts"] == ts]
    return float(g["decimal_odds"].max()) if len(g) else None


def closing_line(conn, match_key, market, line, side):
    """Best price in the last snapshot before kick-off. This is what CLV is measured against."""
    return best_price(conn, match_key, market, line, side, at_ts=None)


def clv_bps(odds_taken, odds_closing):
    """Closing line value in basis points of implied probability. Positive means the price
    taken was better than the closing price: the market ended up rating the selection as
    more likely than the odds you got implied."""
    return float((decimal_to_prob(odds_closing) - decimal_to_prob(odds_taken)) * 10000)


def line_movement(conn, match_key=None):
    """First vs. last snapshot for each selection, using the median price across books.
    move_bps > 0 means the implied probability went up (the price shortened)."""
    df = _load(conn, match_key)
    if df.empty:
        return pd.DataFrame()
    med = (df.groupby(KEY + ["snapshot_ts"])["decimal_odds"].median().reset_index())
    rows = []
    for k, g in med.groupby(KEY):
        g = g.sort_values("snapshot_ts")
        o, c = g.iloc[0], g.iloc[-1]
        rows.append({**dict(zip(KEY, k)), "n_snaps": len(g),
                     "open_dec": round(o["decimal_odds"], 3), "close_dec": round(c["decimal_odds"], 3),
                     "move_bps": round((decimal_to_prob(c["decimal_odds"]) -
                                        decimal_to_prob(o["decimal_odds"])) * 10000, 1)})
    return pd.DataFrame(rows).sort_values("move_bps", key=lambda s: s.abs(), ascending=False)


def steam_moves(conn, thresh_bps=150.0):
    """Selections whose implied probability (median across books) jumped by at least
    thresh_bps between two consecutive snapshots. The median is used so that one book
    changing its price doesn't register as the market moving. It's a rough proxy: real
    steam detection needs snapshots much closer together than these."""
    df = _load(conn, match_key=None)
    if df.empty:
        return pd.DataFrame()
    med = df.groupby(KEY + ["snapshot_ts"])["decimal_odds"].median().reset_index()
    out = []
    for k, g in med.groupby(KEY):
        g = g.sort_values("snapshot_ts")
        if len(g) < 2:
            continue
        p = g["decimal_odds"].map(decimal_to_prob).to_numpy()
        steps = np.diff(p) * 10000
        j = int(np.argmax(np.abs(steps)))
        if abs(steps[j]) >= thresh_bps:
            out.append({**dict(zip(KEY, k)), "step_bps": round(float(steps[j]), 1),
                        "from_ts": str(g.iloc[j]["snapshot_ts"]), "to_ts": str(g.iloc[j + 1]["snapshot_ts"])})
    return pd.DataFrame(out).sort_values("step_bps", key=lambda s: s.abs(), ascending=False) if out else pd.DataFrame()


def main():
    conn = db.connect()
    n = conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
    snaps = conn.execute("SELECT COUNT(DISTINCT snapshot_ts) FROM odds_snapshots").fetchone()[0]
    print(f"odds_snapshots: {n} rows across {snaps} distinct snapshot(s)")
    mv = line_movement(conn)
    if mv.empty:
        print("no movement yet -- this needs at least two snapshots of the same match taken at different times.")
    else:
        print("\ntop line movements (open->close, cross-book median):")
        print(mv.head(15).to_string(index=False))
        st = steam_moves(conn)
        print(f"\nsteam flags (>=150bps single-step): {len(st)}")
        if not st.empty:
            print(st.head(10).to_string(index=False))
    conn.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

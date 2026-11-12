"""SQLite storage.

One file (data/soccer.db) holds everything. I started with flat CSVs, but a single match picks up
many odds snapshots over time (opening price, a few in between, closing price, across dozens of
books) and team ratings change after every game, so the joins are really time-series lookups.
SQLite gives indexes and date queries while still being a single file with no server to run.

Tables:
  teams           team registry (name, confederation, FIFA code)
  matches         one row per match: date, teams, score, neutral-venue flag
  ratings         team strength over time in long format: (team, date, system) -> value
  odds_snapshots  every price we observed, timestamped, per book / market / line / side
  predictions     the tracker's log: one prediction per match, graded once it's played
  bets_log        hypothetical bets with the price taken and, later, the closing price
"""
import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "soccer.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    name          TEXT PRIMARY KEY,
    confederation TEXT,
    fifa_code     TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    match_key   TEXT PRIMARY KEY,          -- 'YYYY-MM-DD|home|away'
    date        TEXT NOT NULL,
    tournament  TEXT,
    stage       TEXT,                       -- group / R16 / qualifier / friendly ...
    home_team   TEXT NOT NULL,
    away_team   TEXT NOT NULL,
    neutral     INTEGER DEFAULT 0,
    home_goals  INTEGER,
    away_goals  INTEGER,
    home_xg     REAL,
    away_xg     REAL,
    source      TEXT
);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team);

CREATE TABLE IF NOT EXISTS ratings (
    team    TEXT NOT NULL,
    date    TEXT NOT NULL,
    system  TEXT NOT NULL,                  -- elo / spi_off / spi_def / fifa
    value   REAL NOT NULL,
    PRIMARY KEY (team, date, system)
);
CREATE INDEX IF NOT EXISTS idx_ratings_lookup ON ratings(team, system, date);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    snapshot_ts  TEXT NOT NULL,             -- when we pulled it, not when the book posted it
    tag          TEXT,                      -- 'open' / 'close' / 'snap'
    match_key    TEXT NOT NULL,
    market       TEXT NOT NULL,             -- 1x2 / ou / ah / btts / team_total ...
    line         REAL,                      -- e.g. 2.5 for O/U; NULL for 1x2
    side         TEXT NOT NULL,             -- over/under/home/draw/away/yes/no
    book         TEXT,
    decimal_odds REAL NOT NULL,
    PRIMARY KEY (snapshot_ts, match_key, market, line, side, book)
);
CREATE INDEX IF NOT EXISTS idx_odds_match ON odds_snapshots(match_key, market);

CREATE TABLE IF NOT EXISTS predictions (
    match_key       TEXT PRIMARY KEY,        -- one prediction per match, never overwritten
    logged_ts       TEXT,
    date            TEXT,
    game            TEXT,                     -- 'Home - Away'
    neutral         INTEGER,
    model_winner    TEXT,                     -- model 1X2 argmax (team or 'Draw')
    model_winner_p  REAL,
    market_winner   TEXT,                     -- de-vigged market 1X2 argmax
    market_winner_p REAL,
    model_ou        TEXT,                     -- 'Over' / 'Under' (model pick at 2.5)
    model_over_p    REAL,                     -- model P(over 2.5)
    actual_winner   TEXT,                     -- filled at grade time
    total_goals     INTEGER,
    actual_ou       TEXT,
    graded          INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bets_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    match_key     TEXT NOT NULL,
    market        TEXT NOT NULL,
    line          REAL,
    side          TEXT NOT NULL,
    book          TEXT,
    odds_taken    REAL NOT NULL,
    model_p       REAL,
    fair_p        REAL,
    edge          REAL,
    stake         REAL,
    closing_odds  REAL,                      -- filled in at grading time
    result        TEXT,                      -- win/loss/push/void
    pnl           REAL,
    clv_bps       REAL
);
"""


def connect(path=DB_PATH):
    """Open the database, creating the data/ folder on first use."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn):
    """Create any missing tables. Safe to call every time."""
    conn.executescript(SCHEMA)
    conn.commit()


def match_key(date, home, away):
    """The key every table joins on: 'YYYY-MM-DD|home|away'. Team names must already be in their
    canonical spelling (see core.names), otherwise the same match ends up under two keys.
    """
    return f"{str(date)[:10]}|{home}|{away}"


def upsert(conn, table, rows, conflict_cols):
    """Bulk insert-or-update, so re-running an ingest never duplicates rows.

    rows is a list of dicts that all share the same keys; conflict_cols are the primary-key
    columns. If a row with that key exists, its other columns are overwritten.
    """
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ",".join("?" for _ in cols)
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c not in conflict_cols)
    sql = (f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders}) "
           f"ON CONFLICT({','.join(conflict_cols)}) DO UPDATE SET {updates}"
           if updates else
           f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({placeholders})")
    conn.executemany(sql, [[r[c] for c in cols] for r in rows])
    conn.commit()
    return len(rows)


def load_matches(conn, before=None, tournaments=None):
    """Matches as a DataFrame sorted by date. Pass `before` to cut off at a date (exclusive), which
    is how training sets are built without leaking later results; `tournaments` filters by name.
    """
    q = "SELECT * FROM matches WHERE 1=1"
    params = []
    if before:
        q += " AND date < ?"
        params.append(str(before)[:10])
    if tournaments:
        q += f" AND tournament IN ({','.join('?' for _ in tournaments)})"
        params += list(tournaments)
    return pd.read_sql(q + " ORDER BY date", conn, params=params)


def rating_asof(conn, team, system, date):
    """A team's latest rating on or before `date`. Returns None if it has no rating yet.

    Careful: ratings are stored after each match, dated on match day, so on a match day this
    already includes that day's result. For anything predictive use rating_before instead.
    """
    row = conn.execute(
        "SELECT value FROM ratings WHERE team=? AND system=? AND date<=? "
        "ORDER BY date DESC LIMIT 1", (team, system, str(date)[:10])).fetchone()
    return row[0] if row else None


def rating_before(conn, team, system, date):
    """A team's rating strictly before `date`, i.e. what it was walking into a match on that day.
    This is the one to use for features and predictions. None if the team has no earlier match.
    """
    row = conn.execute(
        "SELECT value FROM ratings WHERE team=? AND system=? AND date<? "
        "ORDER BY date DESC LIMIT 1", (team, system, str(date)[:10])).fetchone()
    return row[0] if row else None


if __name__ == "__main__":
    c = connect()
    init_db(c)
    print(f"initialized {DB_PATH}")
    for (name,) in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        print("  table:", name)

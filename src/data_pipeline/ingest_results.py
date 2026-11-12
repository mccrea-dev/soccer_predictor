"""Download international match results and load them into the matches table.

Source: the martj42/international_results dataset on GitHub (CC0). It has every senior men's
international since 1872 and is kept up to date; scheduled games that haven't been played yet
appear with empty scores.

Everything gets loaded, not just the tournaments I care about, because Elo ratings and recent-form
features only make sense when computed over a team's full history. Filtering down to the
competitions being modelled happens later.

Run:  python soccer.py ingest-results
"""
import io
import sys
import unicodedata
from pathlib import Path
import requests
import pandas as pd

from core import db
from core.names import canonical

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SHOOTOUTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def _norm(s):
    """Lowercase with accents stripped, for comparing tournament names."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def competition_of(tournament):
    """Bucket the dataset's tournament names into a handful of competition types. The list is
    short on purpose: these are the ones that behave differently enough to matter as a model
    feature, and everything else is 'other'.
    """
    t = _norm(tournament)
    if t == "fifa world cup":
        return "world_cup"
    if "world cup qualification" in t:
        return "world_cup_qual"
    if t == "uefa euro":
        return "euro"
    if "uefa euro qualification" in t:
        return "euro_qual"
    if "nations league" in t:
        return "nations_league"
    if "copa america" in t:
        return "copa_america"
    if t == "friendly":
        return "friendly"
    if "qualification" in t:
        return "other_qual"
    return "other"


# The competitions the models are evaluated on. Friendlies stay in because there are so many of
# them, but they carry an is_competitive = 0 flag since teams often don't field full-strength sides.
TARGET_COMPETITIONS = {"world_cup", "world_cup_qual", "euro", "euro_qual",
                       "nations_league", "copa_america", "friendly"}


def is_competitive(tournament):
    """1 for anything with something at stake, 0 for friendlies."""
    return 0 if _norm(tournament) == "friendly" else 1


def stage_of(tournament):
    """Rough stage label. The dataset doesn't say whether a finals match was a group game or a
    knockout, so every match at a major tournament is just 'finals' here.
    """
    c = competition_of(tournament)
    if c in ("world_cup", "euro", "copa_america"):
        return "finals"
    if c.endswith("qual"):
        return "qualifier"
    if c == "nations_league":
        return "nations_league"
    if c == "friendly":
        return "friendly"
    return "other"


def fetch(url, cache_name):
    """Download a CSV and keep a copy in data/raw/. If the download fails and there is an
    earlier copy on disk, fall back to that."""
    cache = RAW / cache_name
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        cache.write_bytes(r.content)
        return pd.read_csv(io.BytesIO(r.content), encoding="utf-8")
    except Exception as e:
        if cache.exists():
            print(f"[warn] download failed ({e}); using cached {cache_name}")
            return pd.read_csv(cache, encoding="utf-8")
        raise


def load_shootout_keys():
    """match_keys of games that went to penalties. The recorded score for those is the draw at
    the end of extra time, which is what the models should see anyway; this only tags the row
    so the shootout games can be picked out later.
    """
    try:
        s = fetch(SHOOTOUTS_URL, "shootouts.csv")
    except Exception:
        return set()
    keys = set()
    for _, r in s.iterrows():
        keys.add(db.match_key(r["date"], canonical(r["home_team"]), canonical(r["away_team"])))
    return keys


def main():
    df = fetch(RESULTS_URL, "results.csv")
    shootouts = load_shootout_keys()
    df["home_team"] = df["home_team"].map(canonical)
    df["away_team"] = df["away_team"].map(canonical)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)

    rows = []
    for _, r in df.iterrows():
        hs = r["home_score"]
        as_ = r["away_score"]
        played = pd.notna(hs) and pd.notna(as_)
        mk = db.match_key(r["date"], r["home_team"], r["away_team"])
        # unplayed fixtures keep NULL goals, which is how the rest of the code spots them
        rows.append({
            "match_key": mk, "date": str(r["date"])[:10], "tournament": r["tournament"],
            "stage": stage_of(r["tournament"]), "home_team": r["home_team"],
            "away_team": r["away_team"], "neutral": int(r["neutral"]),
            "home_goals": int(hs) if played else None,
            "away_goals": int(as_) if played else None,
            "home_xg": None, "away_xg": None,
            "source": "martj42" + ("+shootout" if mk in shootouts else ""),
        })

    conn = db.connect()
    db.init_db(conn)
    n = db.upsert(conn, "matches", rows, ["match_key"])
    played = conn.execute("SELECT COUNT(*) FROM matches WHERE home_goals IS NOT NULL").fetchone()[0]
    future = conn.execute("SELECT COUNT(*) FROM matches WHERE home_goals IS NULL").fetchone()[0]
    print(f"ingested {n} matches | played={played} | upcoming(unplayed)={future} | shootouts tagged={len(shootouts)}")
    conn.close()


if __name__ == "__main__":
    sys.exit(main())

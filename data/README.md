# data/

Nothing in this folder is tracked except this note. It is all rebuilt locally:

| File | Created by | Contents |
| --- | --- | --- |
| `raw/results.csv`, `raw/shootouts.csv` | `python soccer.py ingest-results` | Cached download of the [martj42 international results](https://github.com/martj42/international_results) dataset |
| `raw/odds_*.json` | `python soccer.py snapshot` | Raw responses from The Odds API |
| `soccer.db` | `ingest-results`, `compute-elo`, ... | SQLite database: matches, Elo ratings, odds snapshots, prediction log |

See the main README for the order to run things in.

"""Command-line entry point. Run everything from the project root:

  python soccer.py <command> [args]

Rebuild the data and models (in this order):
  ingest-results     download match results into data/soccer.db
  compute-elo        replay history to get Elo ratings
  build-features     pre-match feature table (optional, for exploration)
  fit                fit the Dixon-Coles goal model
  calibrate          fit the over/under calibrator on held-out data

Evaluate:
  validate [tag]     walk-forward and leave-one-tournament-out comparison of the models
  eval               out-of-sample check of the goal model on over/under
  worldcup           2026 World Cup retrospective: predictions vs. results, report and charts

Forecast:
  predict            every upcoming fixture
  predict "Ivory Coast" "Norway" [1]    one matchup (1 = neutral venue)

Odds (needs ODDS_API_KEY in the environment):
  snapshot [open|close|snap]   pull current odds once
  poll [events|quota]          scheduled collection that stays within the free tier
  board              goal model vs. market on over/under
  scan               compare prices across bookmakers
  movement           line movement between snapshots
  track [log|grade|show]       log model and market picks, grade them later
  log | clv | results          paper-trading log for over/under picks
"""
import os
import sys
import runpy

# Windows consoles default to cp1252, which garbles names like Côte d'Ivoire. Force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# The packages live in src/. Putting it on the path here means no PYTHONPATH or install step.
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

# command -> the module whose main() it runs
COMMANDS = {
    "predict": "engine.predict",
    "track": "engine.track",
    "scan": "engine.market_ev",
    "board": "engine.pricing",
    "movement": "engine.odds_analytics",
    "poll": "data_pipeline.poll",
    "snapshot": "data_pipeline.ingest_odds",
    "ingest-odds": "data_pipeline.ingest_odds",
    "ingest-results": "data_pipeline.ingest_results",
    "compute-elo": "data_pipeline.compute_elo",
    "build-features": "data_pipeline.build_features",
    "fit": "models.goals_statistical",
    "calibrate": "models.calibrate_ou",
    "eval": "models.eval_goals",
    "validate": "backtest.walk_forward",
    "worldcup": "evaluation.world_cup",
    "log": "backtest.clv",
    "clv": "backtest.clv",
    "results": "backtest.clv",
}


def main():
    # 'ingest_odds' and 'ingest-odds' both work
    cmd = (sys.argv[1] if len(sys.argv) > 1 else "predict").replace("_", "-")
    mod = COMMANDS.get(cmd)
    if not mod:
        print(f"unknown command '{cmd}'. available:\n  " + "  ".join(sorted(COMMANDS)))
        return 1
    # log / clv / results all live in backtest.clv, which picks between them from its first
    # argument, so hand the command name on to it
    rest = sys.argv[2:]
    if mod == "backtest.clv":
        rest = [cmd] + rest
    # run the module exactly as `python -m <module> <args>` would
    sys.argv = [mod] + rest
    runpy.run_module(mod, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())

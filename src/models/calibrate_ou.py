"""Fit the calibrator for the goal model's over-2.5 probability, and check that it actually helps.

Out of the box the goal model is under-confident at the low end: when it says 27% for over 2.5
goals, the over comes in about 38% of the time. An isotonic calibrator learns that mapping.

To keep this honest the data is cut into three periods that don't overlap:

  before CAL_END           the goal model is fitted here
  CAL_END to HOLD_END      the calibrator is fitted here, on matches the model hasn't seen
  HOLD_END onwards         neither has seen these; calibration error is compared before and after

The saved calibrator is then applied to the goal model fitted on all data. That relies on the
model's miscalibration being a property of the model itself and not of one particular fit, which
the third period is there to check.

Run:  python soccer.py calibrate
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from core import db
from core.calibration import fit_isotonic, ece, reliability_table
from models.goals_statistical import DixonColes
from data_pipeline.ingest_results import competition_of, TARGET_COMPETITIONS

MODELS = Path(__file__).resolve().parents[2] / "models"
CAL_END = "2020-01-01"
HOLD_END = "2023-01-01"


def _oos_over(dc, df):
    """The model's P(over 2.5) and the actual over/under outcome (1/0) for each match in df."""
    ps, ys = [], []
    for _, r in df.iterrows():
        ps.append(dc.prob_over(r["home_team"], r["away_team"], 2.5, int(r["neutral"])))
        ys.append(int(r["home_goals"] + r["away_goals"] > 2.5))
    return np.asarray(ps), np.asarray(ys)


def main():
    conn = db.connect()
    m = pd.read_sql("SELECT * FROM matches WHERE home_goals IS NOT NULL", conn)
    conn.close()
    m["date"] = pd.to_datetime(m["date"])
    m["competition"] = m["tournament"].map(competition_of)

    dc = DixonColes().fit(m[m["date"] < CAL_END], ref_date=CAL_END)
    cal_df = m[(m["date"] >= CAL_END) & (m["date"] < HOLD_END) & m["competition"].isin(TARGET_COMPETITIONS)]
    test_df = m[(m["date"] >= HOLD_END) & m["competition"].isin(TARGET_COMPETITIONS)]

    pc, yc = _oos_over(dc, cal_df)
    pt, yt = _oos_over(dc, test_df)
    iso = fit_isotonic(pc, yc)

    before, after = ece(pt, yt), ece(iso.predict(pt), yt)
    print(f"calibrator fit on {len(cal_df)} held-out matches [{CAL_END},{HOLD_END})")
    print(f"check on matches neither has seen [{HOLD_END}, now)  (n={len(test_df)}):")
    print(f"  ECE  raw {before:.3f}  ->  calibrated {after:.3f}")
    print("  predicted vs. actual over rate, before and after calibration:")
    raw_tbl = reliability_table(pt, yt, 10)
    cal_tbl = reliability_table(iso.predict(pt), yt, 10)
    for rr, cc in zip(raw_tbl, cal_tbl):
        print(f"    raw {rr['pred']:.2f} -> realized {rr['realized']:.2f}   |   "
              f"calibrated {cc['pred']:.2f} -> realized {cc['realized']:.2f}")

    MODELS.mkdir(exist_ok=True)
    joblib.dump(iso, MODELS / "ou_calibrator.pkl")
    print(f"saved -> {MODELS / 'ou_calibrator.pkl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

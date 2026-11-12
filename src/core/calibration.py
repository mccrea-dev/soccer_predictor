"""Checking, and fixing, whether predicted probabilities mean what they say.

A model can be good at ranking matches (it usually knows which side is stronger) and still be
badly calibrated: its "70%" picks might win 60% of the time. Ranking is measured by AUC,
calibration by comparing predicted probabilities with how often things actually happened.

Isotonic regression fixes calibration by learning a monotone map from raw probability to observed
frequency. It can stretch or squeeze the scale but never reorder predictions, so it can't add
skill the model doesn't have.

One rule matters more than anything else here: fit the calibrator on matches the model was not
trained on. I learned this the hard way on an earlier UFC model, where calibrating on training
data made a losing strategy look like a winning one.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(p_val, y_val):
    """Calibrator for a yes/no outcome. p_val are the model's probabilities on held-out matches,
    y_val the 0/1 results. Call .predict(p) on the returned object to calibrate new values.
    """
    return IsotonicRegression(out_of_bounds="clip").fit(np.asarray(p_val, float),
                                                         np.asarray(y_val, int))


def fit_multiclass_cal(P_val, y_val, n_class):
    """One isotonic calibrator per class, each trained as "this class vs. the rest". For
    home/draw/away n_class is 3. P_val is an (n, n_class) array of held-out probabilities and
    y_val the class index that actually happened.
    """
    P_val = np.asarray(P_val, float)
    y_val = np.asarray(y_val, int)
    return [IsotonicRegression(out_of_bounds="clip").fit(P_val[:, k], (y_val == k).astype(int))
            for k in range(n_class)]


def apply_multiclass_cal(cals, P):
    """Run each class through its calibrator, then rescale every row to sum to 1."""
    P = np.asarray(P, float)
    Q = np.column_stack([cals[k].predict(P[:, k]) for k in range(len(cals))])
    s = Q.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return Q / s


def ece(p, y, n_bins=10):
    """Expected calibration error. Sort predictions into probability bins, take the gap between
    the average prediction and the actual hit rate in each bin, and average those gaps weighted
    by how many predictions fall in the bin. 0 is perfect; under about 0.05 is good.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability_table(p, y, n_bins=10):
    """The per-bin numbers behind ECE: for each probability bin, the average prediction, how often
    the event actually happened, and the count. Printing this is the quickest way to see where
    a model is over- or under-confident.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin_lo": float(bins[b]), "bin_hi": float(bins[b + 1]),
                     "pred": float(p[m].mean()), "realized": float(y[m].mean()),
                     "n": int(m.sum())})
    return rows

"""Odds math: converting odds to probabilities, removing the bookmaker's margin, expected value
and Kelly stake sizing.

Decimal odds are the working unit throughout. The American-odds helpers exist because that's how
US books display prices and how I think about them.

The one idea to take from this file: a bookmaker's implied probabilities add up to more than 100%
(the extra is their margin, the "vig"). Before comparing a model to the market you have to strip
that out. Whenever the code says "edge" it means model probability minus the de-vigged market
probability, never model probability against the raw price on offer.
"""
import numpy as np

# ---------------------------------------------------------------- conversions
def decimal_to_prob(d):
    """Implied probability straight from decimal odds, bookmaker margin still included."""
    d = float(d)
    return 1.0 / d if d > 0 else 0.0


def prob_to_decimal(p):
    """The fair decimal odds for a probability (no margin)."""
    p = float(p)
    return 1.0 / p if p > 0 else np.inf


def american_to_decimal(o):
    """American odds (+150, -200) to decimal (2.50, 1.50)."""
    o = float(o)
    return o / 100 + 1 if o > 0 else 100 / (-o) + 1


def decimal_to_american(d):
    """Decimal odds back to American. 2.0 is the crossover between + and - prices."""
    d = float(d)
    return (d - 1) * 100 if d >= 2.0 else -100 / (d - 1)


def american_to_prob(o):
    """Implied probability straight from American odds, margin included."""
    o = float(o)
    return 100 / (o + 100) if o > 0 else (-o) / (-o + 100)


# ---------------------------------------------------------------- de-vig
def devig(probs, method="proportional"):
    """Remove the bookmaker margin from a set of implied probabilities so they sum to 1.

    Two methods:

    'proportional' divides each probability by the total. Simple, but it takes the same share
    of margin off a 5% longshot as off a 60% favourite. Books actually load more of their margin
    onto longshots, so this leaves longshots looking more likely than the book really thinks,
    and that shows up downstream as "value" on big underdogs that isn't real.

    'power' finds the exponent k > 1 for which sum(q_i ** k) = 1 and uses q_i ** k. Raising to
    a power shrinks small probabilities proportionally more, which matches how the margin is
    really spread. Use this for three-way markets with a clear underdog. For an evenly matched
    two-way market the two methods give nearly the same answer.
    """
    q = np.asarray([p if np.isfinite(p) and p > 0 else 0.0 for p in probs], dtype=float)
    s = q.sum()
    if s <= 0:
        return q
    if method == "proportional":
        return q / s
    if method == "power":
        if s <= 1.0:                       # already sums to 1 or less, no margin to remove
            return q / s
        from scipy.optimize import brentq
        nz = q > 0
        try:                               # sum(q**k) - 1 is positive at k=1 and falls as k grows, so there is one root
            k = brentq(lambda kk: np.sum(q[nz] ** kk) - 1.0, 1.0, 50.0)
        except ValueError:
            return q / s
        out = np.where(nz, q ** k, 0.0)
        t = out.sum()
        return out / t if t > 0 else q / s
    raise ValueError(f"unknown de-vig method: {method}")


def devig_two_way(over_odds, under_odds):
    """De-vigged (P(over), P(under)) from the two decimal prices of an over/under market."""
    f = devig([decimal_to_prob(over_odds), decimal_to_prob(under_odds)])
    return float(f[0]), float(f[1])


def devig_multi(decimal_odds_list):
    """De-vigged probabilities for a market with several outcomes (home/draw/away), from decimal
    odds. Missing or invalid prices get probability 0.
    """
    return devig([decimal_to_prob(o) if np.isfinite(o) and o > 0 else 0.0
                  for o in decimal_odds_list])


# ---------------------------------------------------------------- EV / Kelly
def ev_per_dollar(p, decimal_odds):
    """Expected profit per $1 staked if the bet wins with probability p at these decimal odds."""
    d = float(decimal_odds)
    return p * (d - 1) - (1 - p)


def kelly_fraction(p, decimal_odds, cap=0.05):
    """Kelly stake as a fraction of bankroll, limited to between 0 and `cap`.

    Full Kelly is (p*b - (1-p)) / b with b the net odds. It assumes p is exactly right, which
    it never is, and a small overestimate on a long price produces a huge stake. Hence the cap.
    """
    b = float(decimal_odds) - 1
    if b <= 0:
        return 0.0
    f = (p * b - (1 - p)) / b
    return float(np.clip(f, 0.0, cap))


def fractional_kelly(p, decimal_odds, fraction=0.25, cap=0.05):
    """A quarter of the Kelly stake by default. Gives up a little long-run growth for much smaller
    swings, which is the usual trade when you don't fully trust your probabilities.
    """
    return fraction * kelly_fraction(p, decimal_odds, cap)


# ---------------------------------------------------------------- decision helpers
def evaluate_two_way(p_over, over_odds, under_odds, cap=0.05, kelly_fraction_=0.25):
    """Compare a model's P(over) with an over/under market and return whichever side has the
    higher expected value, along with its price, EV, edge and suggested stake. `edge` is the
    model probability minus the de-vigged market probability."""
    mkt_over, mkt_under = devig_two_way(over_odds, under_odds)
    ev_o = ev_per_dollar(p_over, over_odds)
    ev_u = ev_per_dollar(1 - p_over, under_odds)
    if ev_o >= ev_u:
        side, p, odds, ev, mkt = "OVER", p_over, over_odds, ev_o, mkt_over
    else:
        side, p, odds, ev, mkt = "UNDER", 1 - p_over, under_odds, ev_u, mkt_under
    return {"side": side, "model_p": p, "market_p": mkt, "odds": odds, "ev": ev,
            "edge": p - mkt, "kelly": fractional_kelly(p, odds, kelly_fraction_, cap)}


def evaluate_multi(model_probs, decimal_odds, labels, cap=0.05, kelly_fraction_=0.25):
    """Same idea for a market with several outcomes (home/draw/away). model_probs, decimal_odds
    and labels line up by position. Returns the outcome with the largest edge, even if that
    edge is negative, so check the sign before acting on it."""
    fair = devig_multi(decimal_odds)
    best = None
    for i, lab in enumerate(labels):
        o = decimal_odds[i]
        if not (np.isfinite(o) and o > 0):
            continue
        edge = model_probs[i] - fair[i]
        cand = {"side": lab, "model_p": float(model_probs[i]), "market_p": float(fair[i]),
                "odds": float(o), "ev": ev_per_dollar(model_probs[i], o), "edge": float(edge),
                "kelly": fractional_kelly(model_probs[i], o, kelly_fraction_, cap)}
        if best is None or cand["edge"] > best["edge"]:
            best = cand
    return best

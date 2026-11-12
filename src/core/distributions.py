"""From expected goals to probabilities.

Give these functions the goals each side is expected to score and they return a score matrix:
entry [i, j] is the probability of the match finishing i-j. Every question about the match is then
a sum over part of that grid (home win = everything below the diagonal, over 2.5 = every cell where
i + j >= 3, and so on). Because all of it comes from one matrix, the answers can't contradict each
other.

Three ways to build the matrix:
  poisson_score_matrix      each side's goals are independent Poisson draws; the simple baseline
  dixon_coles_matrix        Poisson with a correction for 0-0, 1-0, 0-1 and 1-1, which plain Poisson
                            gets slightly wrong; this is the one the project actually uses
  bivariate_poisson_matrix  lets the two sides' goals be positively correlated

Estimating the expected goals in the first place happens in models/goals_statistical.py. The
negative-binomial helpers at the bottom are for counts that vary more than Poisson allows.
"""
import numpy as np
from scipy.stats import poisson, nbinom


# ---------------------------------------------------------------- score matrices
def poisson_score_matrix(lh, la, max_goals=10):
    """P[i, j] = P(home scores i, away scores j) under independent Poisson."""
    h = poisson.pmf(np.arange(max_goals + 1), max(float(lh), 1e-6))
    a = poisson.pmf(np.arange(max_goals + 1), max(float(la), 1e-6))
    m = np.outer(h, a)
    return m / m.sum()


def dixon_coles_tau(lh, la, rho):
    """The four Dixon-Coles correction factors for scores 0-0, 0-1, 1-0 and 1-1, as a 2x2 array.

    With rho < 0, 0-0 and 1-1 become more likely and 1-0 / 0-1 less likely, which is what real
    results show: low-scoring draws happen more often than independent Poisson predicts.
    rho has to be small enough that every factor stays positive.
    """
    return np.array([[1 - lh * la * rho, 1 + lh * rho],
                     [1 + la * rho,       1 - rho]])


def dixon_coles_matrix(lh, la, rho=-0.05, max_goals=10):
    """Dixon and Coles (1997): start from independent Poisson, multiply the four low-score cells
    by their correction factors, then rescale so the whole matrix sums to 1 again.
    """
    m = poisson_score_matrix(lh, la, max_goals)
    m[:2, :2] = m[:2, :2] * dixon_coles_tau(lh, la, rho)
    m = np.clip(m, 0.0, None)
    return m / m.sum()


def bivariate_poisson_matrix(lh, la, lcov=0.0, max_goals=10):
    """Bivariate Poisson: each side's goals are its own Poisson count plus a shared Poisson count
    with mean `lcov`, which makes the two totals move together. lcov = 0 is plain independent
    Poisson. The expected goals come out as (lh + lcov, la + lcov), so pass lh and la with the
    shared part already taken off.
    """
    lcov = max(float(lcov), 0.0)
    l1, l2 = max(float(lh), 1e-6), max(float(la), 1e-6)
    n = max_goals + 1
    m = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            kmax = min(x, y)
            s = 0.0
            for k in range(kmax + 1):
                s += (poisson.pmf(x - k, l1) * poisson.pmf(y - k, l2) * poisson.pmf(k, lcov))
            m[x, y] = s
    return m / m.sum()


# ---------------------------------------------------------------- market readers
def _margin_grid(n):
    """Goal difference (home minus away) for every cell of an n x n score matrix."""
    idx = np.arange(n)
    return np.subtract.outer(idx, idx)


def _total_grid(n):
    """Total goals for every cell of an n x n score matrix."""
    idx = np.arange(n)
    return np.add.outer(idx, idx)


def prob_1x2(matrix):
    """(P(home win), P(draw), P(away win))."""
    m = _margin_grid(matrix.shape[0])
    return float(matrix[m > 0].sum()), float(matrix[m == 0].sum()), float(matrix[m < 0].sum())


def prob_over_under(matrix, line):
    """(P(over), P(under), P(push)) for a total-goals line. A push (total lands exactly on the
    line) is only possible on whole-number lines like 2 or 3."""
    t = _total_grid(matrix.shape[0])
    return (float(matrix[t > line].sum()), float(matrix[t < line].sum()),
            float(matrix[t == line].sum()))


def prob_btts(matrix):
    """(P(both teams score), P(at least one doesn't))."""
    yes = float(matrix[1:, 1:].sum())
    return yes, 1.0 - yes


def prob_team_total(matrix, side, line):
    """(P(over), P(under), P(push)) for one team's goals. side is 'home' or 'away'."""
    n = matrix.shape[0]
    goals = np.arange(n)
    pmf = matrix.sum(axis=1) if side == "home" else matrix.sum(axis=0)
    return (float(pmf[goals > line].sum()), float(pmf[goals < line].sum()),
            float(pmf[goals == line].sum()))


def prob_correct_score(matrix, i, j):
    """Probability of the exact score i-j (0 if it's off the edge of the matrix)."""
    return float(matrix[i, j]) if i < matrix.shape[0] and j < matrix.shape[1] else 0.0


def _ah_single(matrix, handicap):
    """(P(win), P(push), P(lose)) for a bet on the home side with a goal handicap added to its
    score. -1.0 means the home team has to win by two or more (winning by one is a push);
    +0.5 means the bet wins on a home win or a draw."""
    adj = _margin_grid(matrix.shape[0]) + handicap
    return float(matrix[adj > 0].sum()), float(matrix[adj == 0].sum()), float(matrix[adj < 0].sum())


def asian_handicap_return(matrix, handicap, decimal_odds, side="home"):
    """Expected profit per $1 on an Asian handicap bet at the given decimal odds.

    Whole and half lines settle directly. A quarter line (say -0.75) is really two half-stake
    bets on the lines either side of it (-0.5 and -1.0), so it is priced as the average of
    those. For the away side the handicap's sign and the matrix are both flipped.
    """
    h = float(handicap) if side == "home" else -float(handicap)
    mat = matrix if side == "home" else matrix.T
    frac = abs(h - round(h * 2) / 2)  # non-zero only when h is a quarter line
    legs = [h] if frac < 1e-9 else [h - 0.25, h + 0.25]
    ev = 0.0
    for leg in legs:
        w, push, _ = _ah_single(mat, leg)
        # a win pays odds - 1, a push returns the stake (0), a loss costs the stake (-1)
        ev += (w * (decimal_odds - 1) - (1 - w - push)) / len(legs)
    return float(ev)


# ---------------------------------------------------------------- negative binomial
def nb_params_from_mean_disp(mean, disp_alpha):
    """scipy's (n, p) parameters for a negative binomial with the given mean and dispersion alpha,
    where variance = mean + alpha * mean^2. As alpha goes to 0 this becomes a Poisson.
    """
    mean = np.asarray(mean, dtype=float).clip(1e-6)
    a = max(float(disp_alpha), 1e-6)
    n = 1.0 / a
    return n, n / (n + mean)


def nb_prob_over(line, mean, disp_alpha):
    """P(count > line) under a negative binomial. Useful for a team total when goals are more
    spread out than Poisson expects, since Poisson then underestimates the big scores.
    """
    n, p = nb_params_from_mean_disp(mean, disp_alpha)
    return float(1 - nbinom.cdf(np.floor(line), n, p))


def estimate_alpha(y_true, y_pred):
    """Estimate the dispersion alpha from actual counts and predicted means (method of moments)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float).clip(1e-6)
    num = ((y_true - y_pred) ** 2 - y_pred).sum()
    den = (y_pred ** 2).sum()
    return max(num / den, 1e-6)


def market_summary(matrix, ou_lines=(1.5, 2.5, 3.5)):
    """The common markets from one score matrix, as a dict: home/draw/away, both teams to score,
    expected goals for each side, and over/under at each line in ou_lines.
    """
    h, d, a = prob_1x2(matrix)
    btts_y, btts_n = prob_btts(matrix)
    out = {"home": h, "draw": d, "away": a, "btts_yes": btts_y, "btts_no": btts_n,
           "exp_home_goals": float((matrix.sum(axis=1) * np.arange(matrix.shape[0])).sum()),
           "exp_away_goals": float((matrix.sum(axis=0) * np.arange(matrix.shape[0])).sum())}
    for L in ou_lines:
        o, u, _ = prob_over_under(matrix, L)
        out[f"over_{L}"], out[f"under_{L}"] = o, u
    return out

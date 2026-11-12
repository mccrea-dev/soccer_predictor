"""Dixon-Coles goal model: estimates how many goals each side is expected to score.

Every team gets an attack rating and a defence rating. Expected goals for a match are

  home goals = exp(intercept + attack_home - defence_away + home_advantage)
  away goals = exp(intercept + attack_away - defence_home)

with the home advantage dropped at neutral venues. A larger defence number means the team
concedes less. On top of that there is one extra parameter, rho, for the Dixon-Coles low-score
correction (see core/distributions.py). From the two expected-goal numbers you get a full score
matrix, and from that every probability the project reports.

How it's fitted:

  1. The ratings, home advantage and intercept come from maximum likelihood on a Poisson model,
     solved with L-BFGS using a hand-written gradient (much faster than letting scipy
     estimate it numerically with ~600 parameters). A ridge penalty pulls ratings towards zero,
     which keeps teams with only a few matches from getting extreme values.
  2. rho is then fitted separately with the ratings held fixed. It hardly interacts with them,
     and doing it in a second step keeps step 1 a simple, well-behaved problem.

Recent matches count for more: a match's weight halves every `half_life_days`. The default of two
years is long by club standards, but national teams play only around ten games a year.

Run:  python soccer.py fit
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from core.distributions import dixon_coles_matrix, prob_over_under, prob_1x2, market_summary
from core import db

MODELS = Path(__file__).resolve().parents[2] / "models"
ETA_CLIP = 3.0  # cap on log expected goals (about 20 goals), a guard against absurd mismatches


class DixonColes:
    """Fit with .fit(matches), then ask for expected goals, a score matrix or market probabilities
    for any pair of teams. Teams the model has never seen are treated as average (ratings of 0).
    """
    def __init__(self, half_life_days=730.0, reg=2.0, max_goals=10):
        self.half_life = float(half_life_days)
        self.reg = float(reg)
        self.max_goals = int(max_goals)
        self.attack = {}
        self.defence = {}
        self.gamma = 0.0
        self.intercept = 0.0
        self.rho = 0.0
        self.teams = []

    # ------------------------------------------------------------------ fit
    def fit(self, matches, ref_date=None):
        """Fit on a DataFrame with home_team, away_team, home_goals, away_goals, neutral, date.

        ref_date is "today" for the time weighting: a match's age is measured back from it.
        It defaults to the latest match in the data. When backtesting, pass the first date
        you're about to predict."""
        m = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        m["date"] = pd.to_datetime(m["date"])
        ref = pd.Timestamp(ref_date) if ref_date is not None else m["date"].max()
        age = (ref - m["date"]).dt.days.clip(lower=0).to_numpy()
        w = 0.5 ** (age / self.half_life)
        keep = w > 1e-4                     # matches older than ~25 years contribute nothing; drop them
        m, w = m[keep].reset_index(drop=True), w[keep]

        teams = sorted(set(m["home_team"]) | set(m["away_team"]))
        idx = {t: k for k, t in enumerate(teams)}
        n = len(teams)
        hi = m["home_team"].map(idx).to_numpy()
        ai = m["away_team"].map(idx).to_numpy()
        x = m["home_goals"].to_numpy(float)
        y = m["away_goals"].to_numpy(float)
        h = (1 - m["neutral"].to_numpy(float))     # 1 when the home side really is at home

        # Parameter vector p = [attack per team, defence per team, home advantage, intercept].
        # Returns the penalised negative log-likelihood and its gradient in one pass.
        def negll_grad(p):
            a, b, g, c = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
            eta_l = c + a[hi] - b[ai] + g * h       # log expected home goals
            eta_m = c + a[ai] - b[hi]               # log expected away goals
            lam = np.exp(eta_l)
            mu = np.exp(eta_m)
            # weighted Poisson log-likelihood (the factorial term is constant, so it's left out)
            ll = np.sum(w * (x * eta_l - lam + y * eta_m - mu))
            nll = -ll + self.reg * (a @ a + b @ b)   # ridge on the team ratings only
            # For a Poisson the derivative w.r.t. the log rate is just (expected - actual).
            r_l = -w * (x - lam)
            r_m = -w * (y - mu)
            # A team's attack appears in its own goals, its defence (negatively) in the opponent's.
            # bincount adds up each match's contribution per team.
            ga = np.bincount(hi, r_l, n) + np.bincount(ai, r_m, n) + 2 * self.reg * a
            gb = np.bincount(ai, -r_l, n) + np.bincount(hi, -r_m, n) + 2 * self.reg * b
            gg = np.sum(r_l * h)
            gc = np.sum(r_l) + np.sum(r_m)
            return nll, np.concatenate([ga, gb, [gg, gc]])

        p0 = np.zeros(2 * n + 2)
        p0[2 * n] = 0.3                              # start home advantage near a typical value
        p0[2 * n + 1] = np.log(1.3)                  # and the baseline at about 1.3 goals a side
        res = minimize(negll_grad, p0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-10})
        a, b, g, c = res.x[:n], res.x[n:2 * n], res.x[2 * n], res.x[2 * n + 1]
        # Adding the same constant to every attack and every defence rating changes no prediction
        # (only attack minus defence ever matters), so shift them to make the average attack zero.
        # That way a rating reads directly as "better or worse than average".
        shift = a.mean(); a = a - shift; b = b - shift

        # Step 2: fit rho with the ratings fixed.
        lam = np.exp(c + a[hi] - b[ai] + g * h)
        mu = np.exp(c + a[ai] - b[hi])
        self._fit_rho(x, y, lam, mu, w)

        self.teams = teams
        self.attack = dict(zip(teams, a))
        self.defence = dict(zip(teams, b))
        self.gamma = float(g)
        self.intercept = float(c)
        self._fit_meta = {"n_matches": int(len(m)), "n_teams": n, "ref": str(ref.date()),
                          "converged": bool(res.success)}
        return self

    def _fit_rho(self, x, y, lam, mu, w):
        """Fit rho by maximum likelihood. Only matches that finished 0-0, 1-0, 0-1 or 1-1 are
        affected by the correction, so only those enter the calculation.
        """
        corner = (x <= 1) & (y <= 1)
        xc, yc, lc, mc, wc = x[corner], y[corner], lam[corner], mu[corner], w[corner]

        def nll_rho(rho):
            tau = np.ones_like(lc)
            tau = np.where((xc == 0) & (yc == 0), 1 - lc * mc * rho, tau)
            tau = np.where((xc == 0) & (yc == 1), 1 + lc * rho, tau)
            tau = np.where((xc == 1) & (yc == 0), 1 + mc * rho, tau)
            tau = np.where((xc == 1) & (yc == 1), 1 - rho, tau)
            return -np.sum(wc * np.log(np.clip(tau, 1e-10, None)))

        r = minimize_scalar(nll_rho, bounds=(-0.2, 0.2), method="bounded")
        self.rho = float(r.x)

    # ------------------------------------------------------------------ predict
    def lambda_mu(self, home, away, neutral=0):
        """Expected goals (home, away) for a matchup. neutral=1 removes the home advantage."""
        a_h = self.attack.get(home, 0.0); d_h = self.defence.get(home, 0.0)
        a_a = self.attack.get(away, 0.0); d_a = self.defence.get(away, 0.0)
        eta_l = np.clip(self.intercept + a_h - d_a + self.gamma * (1 - neutral), -ETA_CLIP, ETA_CLIP)
        eta_m = np.clip(self.intercept + a_a - d_h, -ETA_CLIP, ETA_CLIP)
        return float(np.exp(eta_l)), float(np.exp(eta_m))

    def score_matrix(self, home, away, neutral=0):
        """Probability of every scoreline for a matchup."""
        lam, mu = self.lambda_mu(home, away, neutral)
        return dixon_coles_matrix(lam, mu, self.rho, self.max_goals)

    def markets(self, home, away, neutral=0, ou_lines=(1.5, 2.5, 3.5)):
        """Win/draw/win, both-teams-to-score and over/under probabilities for a matchup."""
        return market_summary(self.score_matrix(home, away, neutral), ou_lines)

    def prob_over(self, home, away, line=2.5, neutral=0):
        """P(total goals > line) for a matchup."""
        return prob_over_under(self.score_matrix(home, away, neutral), line)[0]

    # ------------------------------------------------------------------ io
    def save(self, path=None):
        import joblib
        MODELS.mkdir(exist_ok=True)
        path = path or MODELS / "dixon_coles.pkl"
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path=None):
        import joblib
        return joblib.load(path or MODELS / "dixon_coles.pkl")


def fit_from_db(half_life_days=730.0, reg=2.0):
    """Fit on every played match in the database."""
    conn = db.connect()
    m = pd.read_sql("SELECT * FROM matches WHERE home_goals IS NOT NULL", conn)
    conn.close()
    return DixonColes(half_life_days, reg).fit(m)


def main():
    # Run as a script, this file is the module "__main__", and a model pickled from here would be
    # saved as "__main__.DixonColes" and fail to load from anywhere else. Importing the function
    # through the package path makes the saved model point at models.goals_statistical instead.
    from models.goals_statistical import fit_from_db
    dc = fit_from_db()
    print(f"fit: {dc._fit_meta}")
    print(f"intercept c = {dc.intercept:.3f} (baseline {np.exp(dc.intercept):.2f} goals/side, "
          f"avg neutral match {2*np.exp(dc.intercept):.2f} total)")
    print(f"home advantage gamma = {dc.gamma:.3f} (exp={np.exp(dc.gamma):.3f}x home goals)  "
          f"rho = {dc.rho:.4f}")
    # Higher is better for both ratings (a big defence number means fewer goals conceded),
    # so attack + defence is a reasonable one-number ranking.
    rank = sorted(dc.attack, key=lambda t: dc.attack[t] + dc.defence[t], reverse=True)
    print("top 10 by net strength (attack + defence):")
    for t in rank[:10]:
        print(f"  {t:22s} att {dc.attack[t]:+.2f}  def {dc.defence[t]:+.2f}  "
              f"net {dc.attack[t] + dc.defence[t]:+.2f}")
    # eyeball check on a matchup where I know roughly what to expect
    lam, mu = dc.lambda_mu("Brazil", "Argentina", neutral=1)
    mk = dc.markets("Brazil", "Argentina", neutral=1)
    print(f"\nBrazil vs Argentina (neutral): lambda={lam:.2f} mu={mu:.2f}  "
          f"P(home/draw/away)={mk['home']:.2f}/{mk['draw']:.2f}/{mk['away']:.2f}  "
          f"over2.5={mk['over_2.5']:.2f}")
    dc.save()
    print(f"saved -> {MODELS / 'dixon_coles.pkl'}")


if __name__ == "__main__":
    sys.exit(main())

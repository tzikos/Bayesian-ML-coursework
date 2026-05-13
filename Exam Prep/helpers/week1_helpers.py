__all__ = [
    "ObservedData",
    "BetaDistribution",
    "compute_posterior",
    "beta_binomial_log_evidence",
    "plot_beta",
]

import numpy as np
from typing import NamedTuple
from scipy.stats import beta as beta_dist
from scipy.special import betaln


class ObservedData(NamedTuple):
    """Represents an observation from a Binomial distribution Bin(theta|N, y)."""
    y: int
    N: int


class BetaDistribution(NamedTuple):
    a: float
    b: float

    def pdf(self, theta):
        """Returns Beta(theta|a, b)."""
        return beta_dist.pdf(theta, a=self.a, b=self.b)

    @property
    def mean(self):
        return self.a / (self.a + self.b)

    @property
    def variance(self):
        return self.a * self.b / (self.a + self.b) ** 2 / (self.a + self.b + 1)

    @property
    def mode(self):
        """Returns the mode (assumes a > 1 and b > 1)."""
        return (self.a - 1) / (self.a + self.b - 2)

    def get_interval(self, size):
        """Returns a central credibility interval of given size (in percent)."""
        return beta_dist.interval(size / 100, a=self.a, b=self.b)

    def sample(self, n):
        return beta_dist.rvs(a=self.a, b=self.b, size=n)


def compute_posterior(prior: BetaDistribution, data: ObservedData) -> BetaDistribution:
    """Computes posterior Beta distribution given a Beta prior and Binomial data."""
    a = prior.a + data.y
    b = prior.b + data.N - data.y
    return BetaDistribution(a, b)


def beta_binomial_log_evidence(y, N, a, b):
    """Log marginal likelihood log p(y | N, a, b) for the Beta-Binomial model."""
    from scipy.special import comb
    log_binom_coeff = np.log(comb(N, y, exact=False))
    return log_binom_coeff + betaln(a + y, b + N - y) - betaln(a, b)


def plot_beta(ax, dist: BetaDistribution, label='', color=None):
    """Plots the Beta PDF on the given axes."""
    thetas = np.linspace(0, 1, 500)
    kwargs = dict(label=label) if label else {}
    if color is not None:
        kwargs['color'] = color
    ax.plot(thetas, dist.pdf(thetas), **kwargs)
    ax.set(xlabel=r'$\theta$', ylabel='Density')

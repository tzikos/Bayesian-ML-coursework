__all__ = [
    "sigmoid",
    "log_npdf",
    "LogisticRegression",
    "Grid2D",
    "GridApproximation2D",
    "DiscreteDistribution1D",
]

import jax.numpy as jnp
from jax import random
from scipy.stats import binom as binom_dist


def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def log_npdf(x, m, v):
    """Log N(x; m, v) where v is variance."""
    return -0.5 * jnp.log(2 * jnp.pi * v) - (x - m) ** 2 / (2 * v)


class LogisticRegression:
    """Logistic regression model from Week 2.

    log_joint(alpha, beta) accepts broadcastable scalar/array arguments so it
    can be evaluated on a 2-D meshgrid directly.

    Parameters
    ----------
    x : array (N,)  — 1-D feature vector
    y : array (N,)  — binary (0/1) targets
    N : int or array — number of trials per observation (1 for binary)
    sigma2_alpha, sigma2_beta : prior variances
    """

    def __init__(self, x, y, N=1, sigma2_alpha=1.0, sigma2_beta=1.0):
        self.x = jnp.asarray(x, dtype=float)
        self.y = jnp.asarray(y, dtype=float)
        self.N = N
        self.sigma2_alpha = sigma2_alpha
        self.sigma2_beta = sigma2_beta

    def f(self, x, alpha, beta):
        return alpha + beta * x

    def theta(self, x, alpha, beta):
        return sigmoid(self.f(x, alpha, beta))

    def log_prior(self, alpha, beta):
        return log_npdf(alpha, 0, self.sigma2_alpha) + log_npdf(beta, 0, self.sigma2_beta)

    def log_likelihood(self, alpha, beta):
        p = self.theta(self.x, alpha, beta)
        return jnp.sum(
            binom_dist.logpmf(self.y, n=self.N, p=jnp.clip(p, 1e-15, 1 - 1e-15)),
            axis=-1,
            keepdims=True,
        )

    def log_joint(self, alpha, beta):
        return self.log_prior(alpha, beta).squeeze() + self.log_likelihood(alpha, beta).squeeze()

    def grad(self, alpha, beta):
        """Gradient (d/d alpha, d/d beta) of log_joint at scalar (alpha, beta)."""
        from jax import grad as jgrad
        g = jgrad(lambda ab: self.log_joint(ab[0], ab[1]))(jnp.array([alpha, beta]))
        return g[0], g[1]


class Grid2D:
    """Evaluates func on a 2-D (alpha, beta) grid.

    func is called as func(alpha_grid[:, :, None], beta_grid[:, :, None]).squeeze()
    so it should broadcast correctly over two scalar/array arguments.
    """

    def __init__(self, alphas, betas, func, name="Grid2D"):
        self.alphas = jnp.asarray(alphas)
        self.betas  = jnp.asarray(betas)
        self.grid_size = (len(self.alphas), len(self.betas))
        self.alpha_grid, self.beta_grid = jnp.meshgrid(alphas, betas, indexing='ij')
        self.func = func
        self.name = name
        self.values = self.func(
            self.alpha_grid[:, :, None], self.beta_grid[:, :, None]
        ).squeeze()

    def plot_contours(self, ax, color='b', num_contours=10, f=lambda x: x, alpha=1.0, title=None):
        ax.contour(self.alphas, self.betas, f(self.values).T, num_contours, colors=color, alpha=alpha)
        ax.set(xlabel=r'$\alpha$', ylabel=r'$\beta$')
        if title:
            ax.set_title(title, fontweight='bold')

    @property
    def argmax(self):
        idx = jnp.argmax(self.values)
        ai, bi = jnp.unravel_index(idx, self.grid_size)
        return self.alphas[ai], self.betas[bi]


class GridApproximation2D(Grid2D):
    """Grid-based posterior approximation over a 2-D parameter space.

    Parameters
    ----------
    alphas, betas : 1-D arrays defining the grid
    log_joint     : callable log_joint(alpha, beta) accepting broadcastable arrays
    """

    def __init__(self, alphas, betas, log_joint, threshold=1e-8, name="GridApproximation2D"):
        Grid2D.__init__(self, alphas, betas, log_joint, name)
        self.threshold = threshold
        self.prep_approximation()
        self.compute_marginals()

    def prep_approximation(self):
        lj = self.values - jnp.max(self.values)
        self.log_joint_grid = lj
        self.tilde_probabilities_grid = jnp.exp(lj)
        self.Z = jnp.sum(self.tilde_probabilities_grid)
        self.probabilities_grid = self.tilde_probabilities_grid / self.Z
        self.alphas_flat = self.alpha_grid.flatten()
        self.betas_flat  = self.beta_grid.flatten()
        self.num_outcomes = len(self.alphas_flat)
        self.probabilities_flat = self.probabilities_grid.flatten()

    def compute_marginals(self):
        self.pi_alpha = self.probabilities_grid.sum(1)
        self.pi_beta  = self.probabilities_grid.sum(0)
        # expose as DiscreteDistribution1D for convenience
        self.marginal_alpha = DiscreteDistribution1D(self.alphas, self.pi_alpha, name='alpha')
        self.marginal_beta  = DiscreteDistribution1D(self.betas,  self.pi_beta,  name='beta')

    def compute_expectation(self, f):
        """E_q[f(alpha, beta)]."""
        return jnp.sum(f(self.alphas_flat, self.betas_flat) * self.probabilities_flat)

    def sample(self, key, num_samples=1):
        idx = random.choice(key, jnp.arange(self.num_outcomes),
                            p=self.probabilities_flat, shape=(num_samples, 1))
        return self.alphas_flat[idx], self.betas_flat[idx]

    def visualize(self, ax, scaling=8000, title='Grid approximation'):
        mask = self.probabilities_flat > self.threshold
        ax.scatter(self.alphas_flat[mask], self.betas_flat[mask],
                   scaling * self.probabilities_flat[mask], label=r'$\pi_{ij}$')
        ax.set(xlabel=r'$\alpha$', ylabel=r'$\beta$')
        ax.set_title(title, fontweight='bold')


class DiscreteDistribution1D:
    """Discrete 1-D distribution over an array of outcomes.

    Parameters
    ----------
    outcomes      : 1-D array of outcome values
    probabilities : 1-D array of non-negative weights (need not sum to 1)
    """

    def __init__(self, outcomes, probabilities, name='DiscreteDistribution1D'):
        self.outcomes = jnp.asarray(outcomes, dtype=float)
        probs = jnp.asarray(probabilities, dtype=float)
        self.probabilities = probs / jnp.sum(probs)
        self.name = name

    def CDF(self, x):
        return jnp.sum(self.probabilities[self.outcomes <= x])

    def quantile(self, p):
        cdf = jnp.cumsum(self.probabilities)
        idx = jnp.where(jnp.logical_or(p < cdf, jnp.isclose(p, cdf)))[0]
        return jnp.min(self.outcomes[idx])

    @property
    def mean(self):
        return jnp.sum(self.outcomes * self.probabilities)

    @property
    def variance(self):
        return jnp.sum((self.outcomes - self.mean) ** 2 * self.probabilities)

    def central_interval(self, interval_size=95):
        """Returns (lower, upper) for a central credibility interval.

        interval_size is a percentage (default 95 → 95%).
        """
        c = 1.0 - interval_size / 100.0
        return jnp.array([self.quantile(c / 2), self.quantile(1.0 - c / 2)])

    def print_summary(self):
        print(f'Summary for {self.name}')
        print(f'  Mean:        {float(self.mean):.4f}')
        print(f'  Std dev:     {float(jnp.sqrt(self.variance)):.4f}')
        lo, hi = self.central_interval(95)
        print(f'  95% CI:      [{float(lo):.4f}, {float(hi):.4f}]')

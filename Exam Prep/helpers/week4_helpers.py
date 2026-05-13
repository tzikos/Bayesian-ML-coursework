__all__ = [
    "BetaBinomial",
    "LaplaceApproximation1D",
    "LaplaceApproximation",
    "LogisticRegressionWithPrior",
    "PosteriorPredictiveDistribution",
    "accuracy",
    "elpd",
]

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist
from scipy.stats import binom as binom_dist
from scipy.stats import norm as norm_dist
from scipy.stats import multivariate_normal as mvn


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _log_npdf(x, m, v):
    return -(x - m) ** 2 / (2 * v) - 0.5 * np.log(2 * np.pi * v)


def _npdf(x, m, v):
    return np.exp(_log_npdf(x, m, v))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# BetaBinomial — conjugate model used to test the Laplace approximation
# ---------------------------------------------------------------------------

class BetaBinomial:
    """Conjugate Beta-Binomial model.

    The log joint (up to a constant) is:
        log p(theta, y) = (a0 + y - 1) log(theta) + (b0 + N - y - 1) log(1-theta)
    """

    def __init__(self, a, b, y, N):
        self.alpha0, self.beta0 = a, b
        self.y, self.N = y, N
        # true posterior parameters
        self.alpha = a + y
        self.beta = b + N - y

    @property
    def posterior_mean(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def posterior_variance(self):
        s = self.alpha + self.beta
        return self.alpha * self.beta / (s ** 2 * (s + 1))

    def pdf(self, theta):
        return beta_dist.pdf(theta, self.alpha, self.beta)

    def log_joint(self, theta):
        return (
            (self.alpha0 + self.y - 1) * np.log(theta)
            + (self.beta0 + self.N - self.y - 1) * np.log(1 - theta)
        )

    def grad(self, theta):
        return (
            (self.alpha0 + self.y - 1) / theta
            - (self.beta0 + self.N - self.y - 1) / (1 - theta)
        )

    def hessian(self, theta):
        return (
            -(self.alpha0 + self.y - 1) / theta ** 2
            - (self.beta0 + self.N - self.y - 1) / (1 - theta) ** 2
        )


# ---------------------------------------------------------------------------
# 1D Laplace approximation
# ---------------------------------------------------------------------------

class LaplaceApproximation1D:
    """Scalar (1D) Laplace approximation q(theta) = N(theta | MAP, -1/H)."""

    def __init__(self, log_joint_fn):
        self.log_joint_fn = log_joint_fn
        self.mean = None
        self.variance = None

    def construct_approximation(self, theta_init=0.5):
        model = self.log_joint_fn  # expects an object with log_joint, grad, hessian
        result = minimize(
            lambda x: -model.log_joint(x),
            jac=lambda x: -model.grad(x),
            x0=theta_init,
            bounds=[(1e-10, 1 - 1e-10)],
        )
        if not result.success:
            raise ValueError(f"Optimisation failed: {result.message}")
        self.mean = result.x
        self.Hessian = model.hessian(self.mean)
        self.variance = -1.0 / self.Hessian

    def pdf(self, theta):
        return _npdf(theta, self.mean, self.variance)


# ---------------------------------------------------------------------------
# Multivariate Laplace approximation
# ---------------------------------------------------------------------------

class LaplaceApproximation:
    """Multivariate Laplace approximation for a D-dimensional posterior.

    Parameters
    ----------
    log_joint_fn : object with .log_joint(w), .grad(w), .hessian(w), .w_MAP
    D            : dimensionality (used for sanity checks only)
    """

    def __init__(self, log_joint_fn, D):
        self.model = log_joint_fn
        self.D = D
        self.posterior_mean = log_joint_fn.w_MAP
        H = log_joint_fn.hessian(log_joint_fn.w_MAP)
        self.posterior_cov = -np.linalg.inv(H)

    def log_pdf(self, w):
        return mvn.logpdf(w, self.posterior_mean.ravel(), self.posterior_cov)

    def posterior_samples(self, num_samples, seed=0):
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(
            self.posterior_mean.ravel(), self.posterior_cov, size=num_samples
        )


# ---------------------------------------------------------------------------
# Logistic regression with Gaussian prior (generalised, with preprocessing)
# ---------------------------------------------------------------------------

class LogisticRegressionWithPrior:
    """Logistic regression model with isotropic Gaussian prior N(w|0, alpha^{-1} I).

    The design matrix X is standardised internally. w_MAP is computed at
    construction time via scipy.optimize.minimize.

    Parameters
    ----------
    X     : array (N, D_raw)
    y     : binary array (N,) or (N, 1)
    alpha : prior precision
    """

    def __init__(self, X, y, alpha=1.0):
        self.X_raw = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).ravel()
        self.alpha = alpha

        # standardise features
        self.X_mean = np.mean(self.X_raw, axis=0)
        self.X_std = np.std(self.X_raw, axis=0)
        self.X_std[self.X_std == 0] = 1.0
        self.X = (self.X_raw - self.X_mean) / self.X_std

        self.N, self.D = self.X.shape
        self.w_MAP = self._get_MAP()

    def _preprocess(self, X_):
        return (np.asarray(X_, dtype=float) - self.X_mean) / self.X_std

    def _predict(self, X, w):
        """sigma(X @ w)."""
        return _sigmoid(X @ w)

    def log_joint(self, w):
        """Log joint log p(y, w) for w of shape (D,) or (M, D)."""
        w = np.asarray(w, dtype=float)
        if w.ndim == 1:
            p = self._predict(self.X, w)
            log_prior = np.sum(_log_npdf(w, 0.0, 1.0 / self.alpha))
            log_lik = np.sum(binom_dist.logpmf(self.y, n=1, p=np.clip(p, 1e-15, 1 - 1e-15)))
            return log_prior + log_lik
        else:
            # batched: w has shape (M, D)
            log_prior = np.sum(_log_npdf(w, 0.0, 1.0 / self.alpha), axis=1)
            p = _sigmoid(self.X @ w.T)  # (N, M)
            log_lik = np.sum(
                binom_dist.logpmf(self.y[:, None], n=1, p=np.clip(p, 1e-15, 1 - 1e-15)),
                axis=0,
            )
            return log_prior + log_lik

    def grad(self, w):
        """Gradient of log joint w.r.t. w (shape (D,) in, shape (D,) out)."""
        w = np.asarray(w, dtype=float).ravel()
        p = self._predict(self.X, w)
        err = p - self.y
        return -self.X.T @ err - self.alpha * w

    def hessian(self, w):
        """Hessian of log joint (shape (D, D))."""
        w = np.asarray(w, dtype=float).ravel()
        p = self._predict(self.X, w)
        v = p * (1 - p)
        return -self.X.T @ np.diag(v) @ self.X - self.alpha * np.eye(self.D)

    def _get_MAP(self):
        w0 = np.zeros(self.D)
        result = minimize(
            lambda w: -self.log_joint(w),
            jac=lambda w: -self.grad(w),
            x0=w0,
        )
        if not result.success:
            raise ValueError(f"MAP optimisation failed: {result.message}")
        return result.x


# ---------------------------------------------------------------------------
# Posterior predictive distribution
# ---------------------------------------------------------------------------

class PosteriorPredictiveDistribution:
    """Posterior predictive distribution for logistic regression + Laplace approx.

    Parameters
    ----------
    logistic_reg  : LogisticRegressionWithPrior instance
    laplace_approx: LaplaceApproximation instance
    """

    def __init__(self, logistic_reg, laplace_approx):
        self.model = logistic_reg
        self.laplace = laplace_approx

    def posterior_f(self, x):
        """Mean and variance of f* = x_preprocessed @ w under the posterior.

        Parameters
        ----------
        x : array (M, D_raw)

        Returns
        -------
        m : array (M,)
        v : array (M,)
        """
        xp = self.model._preprocess(x)
        m = xp @ self.laplace.posterior_mean
        v = np.diag(xp @ self.laplace.posterior_cov @ xp.T)
        return m, v

    def plugin_approx(self, x):
        """Plug-in approximation using w_MAP."""
        xp = self.model._preprocess(x)
        return _sigmoid(xp @ self.model.w_MAP)

    def montecarlo(self, x, num_samples=1000, seed=0):
        """Monte Carlo estimate of E[sigma(f*)] by sampling w from the posterior."""
        m, v = self.posterior_f(x)
        rng = np.random.default_rng(seed)
        # f_samples shape: (num_samples, M)
        f_samples = m + np.sqrt(v) * rng.standard_normal(size=(num_samples, len(x)))
        return _sigmoid(f_samples).mean(axis=0)

    def probit_approx(self, x):
        """Probit approximation (analytic marginalisation over f*)."""
        m, v = self.posterior_f(x)
        return norm_dist.cdf(m / np.sqrt(8.0 / np.pi + v))


# ---------------------------------------------------------------------------
# Standalone evaluation metrics
# ---------------------------------------------------------------------------

def accuracy(y_true, p_hat, threshold=0.5):
    """Classification accuracy."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    p_hat = np.asarray(p_hat, dtype=float).ravel()
    return float(np.mean(y_true == 1.0 * (p_hat > threshold)))


def elpd(y_true, p_hat):
    """Expected log predictive density (mean log Bernoulli likelihood)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    p_hat = np.asarray(p_hat, dtype=float).ravel()
    return float(np.mean(binom_dist.logpmf(y_true, n=1, p=p_hat)))

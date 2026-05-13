__all__ = [
    "log_B",
    "wishart_log_normalizer",
    "wishart_expected_logdet",
    "wishart_entropy",
    "log_C",
    "log_mv_student_t",
    "VariationalApproximation",
    "VariationalGMM",
    "plot_gmm_contours",
]

import jax.numpy as jnp
from jax import random
from jax.scipy.special import gammaln, logsumexp
from scipy.special import psi
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from time import time


# ---- Wishart helpers ----

def log_B(W, nu):
    D = len(W)
    return (-0.5 * nu * jnp.linalg.slogdet(W)[1]
            - (nu * D / 2) * jnp.log(2)
            - (D * (D - 1) / 4) * jnp.log(jnp.pi)
            - jnp.sum(gammaln((nu + 1 - jnp.arange(1, D + 1)) / 2)))


def _mvt_gammaln(n, alpha):
    return ((n * (n - 1)) / 4) * jnp.log(jnp.pi) + jnp.sum(
        gammaln(alpha + 0.5 * (1 - jnp.arange(1, n + 1))))


def wishart_log_normalizer(W, nu, logdetW=None):
    D = len(W)
    if logdetW is None:
        logdetW = jnp.linalg.slogdet(W)[1]
    return (-(nu / 2) * logdetW
            - (nu * D / 2) * jnp.log(2)
            - _mvt_gammaln(D, nu / 2))


def wishart_expected_logdet(W, nu, logdetW=None):
    D = len(W)
    if logdetW is None:
        logdetW = jnp.linalg.slogdet(W)[1]
    return jnp.sum(psi(0.5 * (nu + 1 - jnp.arange(1, D + 1)))) + D * jnp.log(2) + logdetW


def wishart_entropy(W, nu, logdetW=None):
    D = len(W)
    if logdetW is None:
        logdetW = jnp.linalg.slogdet(W)[1]
    return (-wishart_log_normalizer(W, nu, logdetW)
            - (nu - D - 1) / 2 * wishart_expected_logdet(W, nu, logdetW)
            + nu * D / 2)


# ---- Dirichlet log-normalizer ----

def log_C(alpha):
    return gammaln(jnp.sum(alpha)) - jnp.sum(gammaln(alpha))


# ---- Log multivariate Student-t ----

def log_mv_student_t(x, mu, Lambda, nu):
    D = len(Lambda)
    d = x - mu
    d2 = jnp.sum((d @ Lambda) * d, axis=1)
    return (gammaln(D / 2 + nu / 2)
            - gammaln(nu / 2)
            + 0.5 * jnp.linalg.slogdet(Lambda)[1]
            - 0.5 * D * jnp.log(jnp.pi * nu)
            + (-D / 2 - nu / 2) * jnp.log(1 + d2 / nu))


# ---- Simple mean-field VI for Gaussian+Gamma ----

_log_npdf = lambda x, m, v: -0.5 * jnp.log(2 * jnp.pi * v) - 0.5 * (x - m) ** 2 / v


class VariationalApproximation:
    """Mean-field CAVI for the 1D Gaussian model with unknown mean and precision."""

    def __init__(self, x, mu0=0., lambda0=0., a0=1., b0=1.):
        self.x = x
        self.N = len(x)
        self.mu0 = mu0
        self.lambda0 = lambda0
        self.a0 = a0
        self.b0 = b0
        self.m = self.v = self.a = self.b = None

    def initialize(self, m, v, a, b):
        self.m, self.v, self.a, self.b = m, v, a, b
        return self

    def fit_approx(self, num_iterations):
        """CAVI updates. Returns self; optimal params stored as self.m, .v, .a, .b."""
        xbar = jnp.mean(self.x)
        x2bar = jnp.mean(self.x ** 2)
        m, v, a, b = self.m, self.v, self.a, self.b

        for _ in range(num_iterations):
            # update q(mu)
            tau_mean = a / b
            m = (self.N * xbar + self.lambda0 * self.mu0) / (self.N + self.lambda0)
            v = 1.0 / (tau_mean * (self.N + self.lambda0))

            # update q(tau)
            mu2_mean = m ** 2 + v
            a = self.a0 + (self.N + 1) / 2
            b = (self.b0
                 + 0.5 * self.N * x2bar
                 - m * (self.N * xbar + self.lambda0 * self.mu0)
                 + 0.5 * mu2_mean * (self.N + self.lambda0)
                 + 0.5 * self.lambda0 * self.mu0 ** 2)

        self.m, self.v, self.a, self.b = m, v, a, b
        return self


# ---- Variational Bayesian GMM ----

class VariationalGMM:
    """Variational Bayesian Gaussian Mixture Model (Bishop Ch. 10)."""

    def __init__(self, X=None, K=3, alpha0=0.5, beta0=1., m0=None, W0=None, nu0=None, D=None):
        """
        Can be constructed with D and K explicitly (as in the original code),
        or by providing X together with K. All Bishop priors default to their
        canonical values if not supplied.
        """
        if X is not None and D is None:
            D = X.shape[1]
        self.D = D
        self.K = K

        self.m0 = jnp.zeros(D) if m0 is None else m0
        self.W0 = jnp.identity(D) if W0 is None else W0
        self.W0inv = jnp.linalg.inv(self.W0)
        self.nu0 = D if nu0 is None else nu0
        self.alpha0 = alpha0 * jnp.ones(K)
        self.beta0 = beta0

        self.jitter = 1e-10
        self.convergence_tol = 1e-6
        self.converged = False

        self.lowerbound = None
        self.lowerbound_history = None
        self.pi = None

    def fit(self, X, Xtest=None, max_iter=500, max_itt=None, tol=None, verbose=False, seed=123):
        """CAVI loop. Returns self."""
        # accept both max_iter and max_itt for compatibility
        max_iter = max_itt if max_itt is not None else max_iter

        t0 = time()
        self.X = X
        self.N = len(X)

        self.lowerbound_history = []
        self.pi_history = []
        self.num_active_components = []

        if Xtest is not None:
            self.test_lpd_history = []

        key = random.PRNGKey(seed)
        key, key_rho = random.split(key, 2)

        self.log_rho = 0.01 * random.normal(key_rho, shape=(self.N, self.K))
        self.log_r = self.log_rho - logsumexp(self.log_rho, 1)[:, None]
        self.r = jnp.exp(self.log_r)

        lowerbound_old = -jnp.inf
        self.converged = False

        for itt in range(max_iter):
            self.Nk = jnp.sum(self.r, 0) + 1e-10
            self.Xbar = jnp.sum(self.r[:, :, None] * X[:, None, :], axis=0) / self.Nk[:, None]

            self.S = []
            for k in range(self.K):
                Xdiff = X - self.Xbar[k]
                Xdiff2 = self.r[:, k, None, None] * (Xdiff[:, :, None] @ Xdiff[:, None, :])
                self.S.append(jnp.sum(Xdiff2, axis=0) / self.Nk[k])

            self.alpha_k = self.alpha0 + self.Nk
            self.beta_k = self.beta0 + self.Nk

            self.m = []
            for k in range(self.K):
                self.m.append((self.beta0 * self.m0 + self.Nk[k] * self.Xbar[k]) / self.beta_k[k])

            self.Wk_inv = []
            self.Wk = []
            for k in range(self.K):
                self.Wk_inv.append(
                    self.W0inv
                    + self.Nk[k] * self.S[k]
                    + (self.beta0 * self.Nk[k]) / (self.beta0 + self.Nk[k])
                    * jnp.outer(self.Xbar[k] - self.m0, self.Xbar[k] - self.m0)
                    + self.jitter * jnp.identity(self.D))
                self.Wk.append(jnp.linalg.inv(self.Wk_inv[k]))

            self.nu_k = self.nu0 + self.Nk

            self.logdetWk = []
            self.ln_lam = jnp.zeros(self.K)
            for k in range(self.K):
                self.logdetWk.append(jnp.linalg.slogdet(self.Wk[k])[1])
                val = (self.ln_lam[k]
                       + jnp.sum(psi((self.nu_k[k] + 1 - jnp.arange(1, self.D + 1)) / 2))
                       + self.D * jnp.log(2)
                       + self.logdetWk[k])
                self.ln_lam = self.ln_lam.at[k].set(val)

            self.ln_pi_tilde = psi(self.alpha_k) - psi(jnp.sum(self.alpha_k))

            self.log_rho = jnp.zeros((self.N, self.K))
            for k in range(self.K):
                xdiff = self.X - self.m[k]
                d2 = jnp.sum((xdiff @ self.Wk[k]) * xdiff, 1)
                self.log_rho = self.log_rho.at[:, k].set(
                    self.ln_pi_tilde[k] + 0.5 * self.ln_lam[k]
                    - self.D / (2 * self.beta_k[k])
                    - 0.5 * self.nu_k[k] * d2)

            self.log_r = self.log_rho - logsumexp(self.log_rho, 1)[:, None]
            self.r = jnp.exp(self.log_r)

            self.pi = (self.alpha0 + self.Nk) / (jnp.sum(self.alpha0) + self.N)
            self.pi_history.append(self.pi)
            self.num_active_components.append(jnp.sum(self.Nk >= 1))

            self.compute_lowerbound()
            self.lowerbound_history.append(self.lowerbound)

            if Xtest is not None:
                self.test_lpd_history.append(self.evaluate_log_predictive(Xtest))

            if jnp.abs(lowerbound_old - self.lowerbound_history[-1]) < self.convergence_tol:
                print(f'Converged in {itt:4d} iterations for K={self.K:2d} '
                      f'with lowerbound={self.lowerbound:4.3f}. Time={time()-t0:3.2f}s')
                self.converged = True
                break
            lowerbound_old = self.lowerbound_history[-1]

            if verbose and (itt + 1) % 100 == 0:
                print(f'Itt={itt+1}/{max_iter}, lowerbound={self.lowerbound:3.2f} in {time()-t0:3.2f}s')

        if not self.converged:
            print('Warning: Optimization did not converge')

        return self

    def compute_lowerbound(self):
        t1 = 0
        for k in range(self.K):
            t1 += 0.5 * self.Nk[k] * (
                self.ln_lam[k]
                - self.D / self.beta_k[k]
                - self.nu_k[k] * jnp.trace(self.S[k] @ self.Wk[k])
                - self.nu_k[k] * (self.Xbar[k] - self.m[k]) @ self.Wk[k] @ (self.Xbar[k] - self.m[k]).T
                - self.D * jnp.log(2 * jnp.pi))

        t2 = jnp.sum(self.r * self.ln_pi_tilde)
        t3 = log_C(self.alpha0) + jnp.sum((self.alpha0 - 1) * self.ln_pi_tilde)

        t4 = 0
        for k in range(self.K):
            d = self.m[k] - self.m0
            t4 += 0.5 * (self.D * jnp.log(self.beta0 / (2 * jnp.pi))
                         + self.ln_lam[k]
                         - self.D * self.beta0 / self.beta_k[k]
                         - self.beta0 * self.nu_k[k] * d.T @ self.Wk[k] @ d)
        t4 += self.K * log_B(self.W0, self.nu0) + 0.5 * (self.nu0 - self.D - 1) * jnp.sum(self.ln_lam)
        for k in range(self.K):
            t4 += -0.5 * self.nu_k[k] * jnp.trace(self.W0inv @ self.Wk[k])

        t5 = jnp.sum(self.r * self.log_r)
        t6 = jnp.sum((self.alpha_k - 1) * self.ln_pi_tilde) + log_C(self.alpha_k)

        t7 = 0
        for k in range(self.K):
            t7 += (0.5 * self.ln_lam[k]
                   + 0.5 * self.D * jnp.log(self.beta_k[k] / (2 * jnp.pi))
                   - 0.5 * self.D
                   - wishart_entropy(self.Wk[k], self.nu_k[k], self.logdetWk[k]))

        self.lowerbound = t1 + t2 + t3 + t4 - t5 - t6 - t7

    def compute_component_probs(self, X):
        """Returns responsibilities r_nk of shape (N, K)."""
        Lk = [(self.nu_k[k] + 1 - self.D) * self.beta_k[k] / (1 + self.beta_k[k]) * self.Wk[k]
              for k in range(self.K)]
        log_weights = jnp.log(self.alpha_k) - jnp.log(jnp.sum(self.alpha_k))
        log_contribs = jnp.array([
            log_weights[k] + log_mv_student_t(X, self.m[k], Lk[k], self.nu_k[k] + 1 - self.D)
            for k in range(self.K)])
        log_probs = log_contribs - logsumexp(log_contribs, axis=0)
        return jnp.exp(log_probs.T)

    def evaluate_log_predictive(self, X, pointwise=False):
        """Log marginal likelihood. Sum over points unless pointwise=True."""
        Lk = [(self.nu_k[k] + 1 - self.D) * self.beta_k[k] / (1 + self.beta_k[k]) * self.Wk[k]
              for k in range(self.K)]
        log_weights = jnp.log(self.alpha_k) - jnp.log(jnp.sum(self.alpha_k))
        log_contribs = jnp.array([
            log_weights[k] + log_mv_student_t(X, self.m[k], Lk[k], self.nu_k[k] + 1 - self.D)
            for k in range(self.K)])
        l = logsumexp(log_contribs, axis=0)
        return l if pointwise else l.sum()

    # alias used in original notebooks
    def evaulate_log_predictive(self, X, pointwise=False):
        return self.evaluate_log_predictive(X, pointwise)


# ---- Plotting ----

def plot_gmm_contours(ax, gmm, X=None, n_std=3.0):
    """Plot std-dev ellipses for each component of a fitted VariationalGMM."""
    import matplotlib.pyplot as plt
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if X is not None:
        ax.plot(X[:, 0], X[:, 1], 'k.', markersize=3, alpha=0.4)

    for k in range(gmm.K):
        cov = gmm.S[k]
        m = gmm.m[k]
        pearson = cov[0, 1] / jnp.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = jnp.sqrt(1 + pearson)
        ell_radius_y = jnp.sqrt(1 - pearson)
        color = colors[k % len(colors)]
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor=color, alpha=float(0.3 * gmm.pi[k] / jnp.max(gmm.pi)),
                          edgecolor=color, label=f'k={k+1}')
        scale_x = jnp.sqrt(cov[0, 0]) * n_std
        scale_y = jnp.sqrt(cov[1, 1]) * n_std
        transf = (transforms.Affine2D()
                  .rotate_deg(45)
                  .scale(float(scale_x), float(scale_y))
                  .translate(float(m[0]), float(m[1])))
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)

    ax.autoscale()
    ax.legend()

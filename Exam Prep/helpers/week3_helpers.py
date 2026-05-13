__all__ = [
    "log_npdf",
    "BayesianLinearRegression",
]

import numpy as np
from scipy.optimize import minimize


def log_npdf(x, m, v):
    """Log N(x; m, v) where v is variance (scalar)."""
    return -0.5 * np.log(2 * np.pi * v) - 0.5 * (x - m) ** 2 / v


class BayesianLinearRegression:
    """Bayesian linear regression with isotropic Gaussian prior N(w|0, alpha^{-1} I)
    and Gaussian likelihood N(y|Phi w, beta^{-1} I).

    Parameters
    ----------
    Phi : array of shape (N, D)  — design matrix
    y   : array of shape (N, 1)  — targets
    """

    def __init__(self, Phi, y):
        self.Phi = np.asarray(Phi, dtype=float)
        self.y = np.asarray(y, dtype=float)
        if self.y.ndim == 1:
            self.y = self.y[:, None]
        self.N, self.D = self.Phi.shape

        # placeholders; set by compute_posterior
        self.m_N = None
        self.S_N = None

    # ------------------------------------------------------------------
    # Core posterior computation
    # ------------------------------------------------------------------

    def compute_posterior(self, alpha, beta):
        """Compute and store posterior N(w | m_N, S_N)."""
        inv_S0 = alpha * np.eye(self.D)
        A = inv_S0 + beta * (self.Phi.T @ self.Phi)
        self.S_N = np.linalg.inv(A)
        self.m_N = beta * self.S_N @ self.Phi.T @ self.y  # shape (D, 1)
        self._alpha = alpha
        self._beta = beta
        return self.m_N, self.S_N

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_f(self, Phi_star):
        """Posterior mean and variance of f* = Phi_star @ w.

        Returns
        -------
        mean : array (M,)
        variance : array (M,)
        """
        Phi_star = np.asarray(Phi_star, dtype=float)
        mean = (Phi_star @ self.m_N).ravel()
        variance = np.diag(Phi_star @ self.S_N @ Phi_star.T)
        return mean, variance

    def predict_y(self, Phi_star):
        """Posterior predictive mean and variance of y* (adds noise 1/beta).

        Returns
        -------
        mean : array (M,)
        variance : array (M,)
        """
        mean, var_f = self.predict_f(Phi_star)
        var_y = var_f + 1.0 / self._beta
        return mean, var_y

    # ------------------------------------------------------------------
    # Marginal likelihood
    # ------------------------------------------------------------------

    def compute_marginal_likelihood(self, alpha, beta):
        """Log marginal likelihood log p(y | alpha, beta)."""
        inv_S0 = alpha * np.eye(self.D)
        A = inv_S0 + beta * (self.Phi.T @ self.Phi)
        m = beta * np.linalg.solve(A, self.Phi.T @ self.y)
        Em = (beta / 2) * np.sum((self.y - self.Phi @ m) ** 2) + (alpha / 2) * np.sum(m ** 2)
        sign, logdet = np.linalg.slogdet(A)
        log_ml = (
            self.D / 2 * np.log(alpha)
            + self.N / 2 * np.log(beta)
            - Em
            - 0.5 * logdet
            - self.N / 2 * np.log(2 * np.pi)
        )
        return float(log_ml)

    # ------------------------------------------------------------------
    # Hyperparameter optimisation
    # ------------------------------------------------------------------

    def optimize_hyperparameters(self, alpha0=1.0, beta0=1.0):
        """Maximise marginal likelihood over (alpha, beta); returns optimised values."""
        theta0 = np.array([np.log(alpha0), np.log(beta0)])

        def neg_ml(theta):
            a, b = np.exp(theta[0]), np.exp(theta[1])
            return -self.compute_marginal_likelihood(a, b)

        result = minimize(neg_ml, theta0, method='Nelder-Mead')
        alpha_opt = float(np.exp(result.x[0]))
        beta_opt = float(np.exp(result.x[1]))
        self.compute_posterior(alpha_opt, beta_opt)
        return alpha_opt, beta_opt

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def generate_prior_samples(self, num_samples, alpha):
        """Draw samples from the prior N(w|0, alpha^{-1} I).

        Returns array of shape (num_samples, D).
        """
        return np.random.multivariate_normal(
            np.zeros(self.D), (1.0 / alpha) * np.eye(self.D), size=num_samples
        )

    def generate_posterior_samples(self, num_samples):
        """Draw samples from the posterior N(w | m_N, S_N).

        Returns array of shape (num_samples, D).
        """
        return np.random.multivariate_normal(
            self.m_N.ravel(), self.S_N, size=num_samples
        )

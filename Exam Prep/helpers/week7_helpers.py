__all__ = [
    "softmax",
    "compute_entropy",
    "compute_confidence",
    "compute_expected_utility",
    "BayesianLinearSoftmax",
]

import jax
import jax.numpy as jnp
from jax import value_and_grad, hessian, random
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


def _log_npdf(x, m, v):
    return -0.5 * (x - m) ** 2 / v - 0.5 * jnp.log(2 * jnp.pi * v)


def _to_onehot(y, num_classes):
    return jnp.column_stack([1.0 * (y == c) for c in jnp.arange(num_classes)])


def softmax(a_, axis=1):
    max_val = jnp.max(a_, axis=axis)
    a = a_ - jnp.expand_dims(max_val, axis=axis)
    exp_a = jnp.exp(a)
    return exp_a / jnp.sum(exp_a, axis=axis, keepdims=True)


def compute_entropy(pi):
    """Entropy of categorical distribution. pi shape: [N, K]."""
    log_pi = jnp.where(pi > 0, jnp.log(pi), 0.0)
    return -jnp.sum(pi * log_pi, axis=1)


def compute_confidence(pi):
    """Max class probability. pi shape: [N, K]."""
    return jnp.max(pi, axis=1)


def compute_expected_utility(U, phat):
    """Expected utility: phat @ U. phat shape [P, K], U shape [K, K]."""
    return phat @ U


class BayesianLinearSoftmax(object):
    """Bayesian linear softmax classifier with i.i.d. Gaussian priors."""

    def __init__(self, X_train, y_train, alpha=1.0, num_classes=None):
        self.X = X_train
        self.y = y_train
        self.N, self.D = self.X.shape
        self.alpha = alpha

        if num_classes is None:
            num_classes = int(len(jnp.unique(y_train)))
        self.num_classes = num_classes
        self.num_params = self.num_classes * self.D
        self.y_onehot = _to_onehot(self.y, self.num_classes)

        self.compute_laplace_approximation()

    def log_prior(self, w_flat):
        log_prior_val = jnp.sum(_log_npdf(w_flat, 0.0, 1.0 / self.alpha))
        return log_prior_val

    def log_likelihood(self, w_flat):
        W = w_flat.reshape((self.num_classes, self.D))
        y_all = self.X @ W.T
        p_all = softmax(y_all)
        return jnp.sum(self.y_onehot * jnp.log(p_all))

    def log_joint(self, w_flat):
        return self.log_prior(w_flat) + self.log_likelihood(w_flat)

    def compute_laplace_approximation(self):
        w_init = jnp.zeros(self.num_params)
        cost_fn = lambda w: -self.log_joint(w)
        result = minimize(value_and_grad(cost_fn), w_init, jac=True)
        if result.success:
            w_MAP = result.x
            self.m_flat = w_MAP[:, None]
            self.A_flat = hessian(cost_fn)(w_MAP)
            self.S_flat = jnp.linalg.inv(self.A_flat)
        else:
            print("Warning: optimization failed")

    def predict_f(self, X_star):
        mi = self.m_flat.reshape((self.num_classes, self.D))
        Si = [
            self.S_flat[i * self.D:(i + 1) * self.D, i * self.D:(i + 1) * self.D]
            for i in range(self.num_classes)
        ]
        mu_f = X_star @ mi.T
        var_f = jnp.squeeze(
            jnp.stack([jnp.diag(X_star @ Si[i] @ X_star.T) for i in range(self.num_classes)], axis=1)
        )
        return mu_f, var_f

    def generate_samples_f(self, X_star, num_samples=500, seed=456):
        key = random.PRNGKey(seed)
        w_samples_flat = random.multivariate_normal(
            key, self.m_flat.ravel(), self.S_flat, shape=(num_samples,)
        )
        W_samples = w_samples_flat.T.reshape((self.num_classes, self.D, num_samples))
        f_samples = X_star @ W_samples
        return jnp.swapaxes(f_samples, 0, 1)

    def predict_y(self, X_star, num_samples=500, seed=123):
        f_samples = self.generate_samples_f(X_star, num_samples, seed)
        p_all_samples = softmax(f_samples, axis=1)
        return p_all_samples.mean(axis=2)

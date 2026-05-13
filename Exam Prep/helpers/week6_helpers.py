__all__ = [
    "sigmoid",
    "BernoulliLikelihood",
    "GaussianProcessClassification",
    "probit_approx_binary",
]

import jax
import jax.numpy as jnp
from jax import random
from scipy.optimize import minimize
from scipy.stats import norm
from week5_helpers import Hyperparameters, StationaryIsotropicKernel

jax.config.update("jax_enable_x64", True)


def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def _generate_samples(key, mean, K, M, jitter=1e-8):
    L = jnp.linalg.cholesky(K + jitter * jnp.identity(len(K)))
    zs = random.normal(key, shape=(len(K), M))
    return mean + jnp.dot(L, zs)


class BernoulliLikelihood(object):

    def __init__(self, y):
        self.y = y.ravel()

    def log_lik(self, f):
        p = sigmoid(f)
        return jnp.sum(self.y * jnp.log(p) + (1 - self.y) * jnp.log(1 - p))

    def grad(self, f):
        return self.y - sigmoid(f)

    def hessian(self, f):
        p = sigmoid(f)
        # returns full NxN diagonal matrix
        return jnp.diag(-p * (1 - p))


def probit_approx_binary(mu_f, sigma2_f):
    """Approximate predictive probability P(y=1) using probit approximation."""
    return norm.cdf(mu_f / jnp.sqrt(8.0 / jnp.pi + sigma2_f))


class GaussianProcessClassification(object):

    def __init__(self, X, y, likelihood, kernel, hyperparameters, jitter=1e-8):
        self.X = X
        self.y = y
        self.N = len(X)
        self.likelihood = likelihood(y)
        self.kernel = kernel
        self.jitter = jitter
        self.hyperparameters = hyperparameters

        self.precompute_kernel_matrices()
        self.construct_laplace_approximation()

    def precompute_kernel_matrices(self):
        self.K = self.kernel.construct_kernel(self.X, self.X, self.hyperparameters, self.jitter)
        self.L = jnp.linalg.cholesky(self.K)

    def log_joint_a(self, a):
        f = self.K @ a
        const = -self.N / 2 * jnp.log(2 * jnp.pi)
        logdet = jnp.sum(jnp.log(jnp.diag(self.L)))
        quad_term = 0.5 * jnp.sum(a * f)
        log_prior = const - logdet - quad_term
        log_lik = self.likelihood.log_lik(f)
        return log_prior + log_lik

    def grad_a(self, a):
        f = self.K @ a
        grad_prior = -f
        grad_lik = self.likelihood.grad(f) @ self.K
        return grad_prior + grad_lik

    def compute_f_MAP(self):
        result = minimize(
            lambda a: -self.log_joint_a(a),
            jac=lambda a: -self.grad_a(a),
            x0=jnp.zeros((self.N,)),
        )
        if not result.success:
            print(result)
            raise ValueError("Optimization failed")
        self.a = result.x
        return self.K @ result.x

    def construct_laplace_approximation(self):
        self.m = self.compute_f_MAP()
        Lambda = -self.likelihood.hessian(self.m)
        # Woodbury-stable computation of posterior covariance
        Lsqrt = jnp.sqrt(Lambda)
        B = jnp.identity(len(self.m)) + Lsqrt @ self.K @ Lsqrt
        chol_B = jnp.linalg.cholesky(B)
        e = jnp.linalg.solve(chol_B, Lsqrt @ self.K)
        self.S = self.K - e.T @ e

    def predict_f(self, X_star):
        k = self.kernel.construct_kernel(X_star, self.X, self.hyperparameters, self.jitter)
        Kp = self.kernel.construct_kernel(X_star, X_star, self.hyperparameters, self.jitter)
        h = jnp.linalg.solve(self.K, k.T)
        mu = k @ jnp.linalg.solve(self.K, self.m)
        Sigma = Kp - h.T @ (self.K - self.S) @ h
        return mu, Sigma

    def predict_y(self, X_star):
        mu, Sigma = self.predict_f(X_star)
        p = norm.cdf(mu / jnp.sqrt(8.0 / jnp.pi + jnp.diag(Sigma)))
        return p

    def posterior_samples(self, X_star, num_samples, seed=0):
        key = random.PRNGKey(seed)
        mu, Sigma = self.predict_f(X_star)
        return _generate_samples(key, mu, Sigma, num_samples)

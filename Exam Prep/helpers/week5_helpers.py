__all__ = [
    "Hyperparameters",
    "squared_exponential",
    "matern12",
    "matern32",
    "StationaryIsotropicKernel",
    "GaussianProcessRegression",
    "optimize_marginal_likelihood",
    "generate_gp_samples",
    "plot_gp",
]

import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax import random
from dataclasses import dataclass
from scipy.optimize import minimize
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


@dataclass
class Hyperparameters(object):
    kappa:       float = 1.0
    lengthscale: float = 1.0
    sigma:       float = 1.0

    def to_array(self):
        return jnp.array([self.kappa, self.lengthscale, self.sigma])

    @staticmethod
    def from_array(arr):
        kappa, lengthscale, sigma = arr
        return Hyperparameters(kappa, lengthscale, sigma)

    def __repr__(self):
        return (
            f"Hyperparameters(kappa={self.kappa:3.2f}, "
            f"lengthscale={self.lengthscale:3.2f}, "
            f"sigma={self.sigma:3.2f})"
        )


def squared_exponential(tau, h):
    return h.kappa ** 2 * jnp.exp(-0.5 * tau ** 2 / h.lengthscale ** 2)


def matern12(tau, h):
    return h.kappa ** 2 * jnp.exp(-tau / h.lengthscale)


def matern32(tau, h):
    sqrt3 = jnp.sqrt(3.0)
    return h.kappa ** 2 * (1 + sqrt3 * tau / h.lengthscale) * jnp.exp(-sqrt3 * tau / h.lengthscale)


class StationaryIsotropicKernel(object):

    def __init__(self, kernel_fun):
        self.kernel_fun = kernel_fun

    def construct_kernel(self, X1, X2, h, jitter=1e-6):
        N, M = X1.shape[0], X2.shape[0]
        dists = jnp.sqrt(jnp.sum((jnp.expand_dims(X1, 1) - jnp.expand_dims(X2, 0)) ** 2, axis=-1))
        K = self.kernel_fun(dists, h)
        if len(X1) == len(X2) and jnp.allclose(X1, X2):
            K = K + jitter * jnp.identity(len(X1))
        return K


def generate_gp_samples(mean, K, num_samples, jitter=1e-6):
    key = random.PRNGKey(0)
    N = len(K)
    L = jnp.linalg.cholesky(K + jitter * jnp.identity(N))
    zs = random.normal(key, shape=(N, num_samples))
    return mean[:, None] + jnp.dot(L, zs)


class GaussianProcessRegression(object):

    def __init__(self, X, y, kernel_fn, hyperparameters=None, jitter=1e-8):
        self.X = X
        self.y = y
        self.N = len(X)
        self.kernel = kernel_fn
        self.jitter = jitter
        if hyperparameters is None:
            hyperparameters = Hyperparameters()
        self.hyperparameters = hyperparameters

    def set_hyperparameters(self, h):
        self.hyperparameters = h

    def predict_f(self, X_star):
        h = self.hyperparameters
        if self.N == 0:
            Kstar = self.kernel.construct_kernel(X_star, X_star, h, self.jitter)
            mu = jnp.zeros((len(X_star), 1))
            return mu, Kstar
        k = self.kernel.construct_kernel(X_star, self.X, h, self.jitter)
        K = self.kernel.construct_kernel(self.X, self.X, h, self.jitter)
        Kstar = self.kernel.construct_kernel(X_star, X_star, h, self.jitter)
        C = K + h.sigma ** 2 * jnp.identity(self.N)
        mu = jnp.dot(k, jnp.linalg.solve(C, self.y))
        Sigma = Kstar - jnp.dot(k, jnp.linalg.solve(C, k.T))
        return mu, Sigma

    def predict_y(self, X_star):
        mu, Sigma = self.predict_f(X_star)
        Sigma = Sigma + self.hyperparameters.sigma ** 2 * jnp.identity(len(mu))
        return mu, Sigma

    def posterior_samples(self, X_star, num_samples, seed=0):
        key = random.PRNGKey(seed)
        mu, Sigma = self.predict_f(X_star)
        N = len(X_star)
        L = jnp.linalg.cholesky(Sigma + 1e-8 * jnp.identity(N))
        zs = random.normal(key, shape=(N, num_samples))
        return mu.ravel()[:, None] + jnp.dot(L, zs)

    def log_marginal_likelihood(self, h):
        K = self.kernel.construct_kernel(self.X, self.X, h)
        C = K + h.sigma ** 2 * jnp.identity(self.N)
        L = jnp.linalg.cholesky(C)
        v = jnp.linalg.solve(L, self.y)
        logdet_term = jnp.sum(jnp.log(jnp.diag(L)))
        quad_term = 0.5 * jnp.sum(v ** 2)
        const_term = -0.5 * self.N * jnp.log(2 * jnp.pi)
        return const_term - logdet_term - quad_term


def optimize_marginal_likelihood(gp, X, y, h_init, verbose=True):
    def objective(log_hyp):
        hyp_arr = jnp.exp(log_hyp)
        hyp = Hyperparameters.from_array(hyp_arr)
        return -gp.log_marginal_likelihood(hyp)

    log_hyp0 = jnp.log(h_init.to_array())
    res = minimize(value_and_grad(objective), log_hyp0, jac=True)
    if not res.success and verbose:
        print("Warning: optimization failed!")
    h_opt = Hyperparameters.from_array(jnp.exp(res.x))
    if verbose:
        print("Result of optimization:", h_opt)
    return h_opt


def plot_gp(ax, x, mean, var, samples=None, label=""):
    std = jnp.sqrt(var)
    x = x.ravel()
    mean = mean.ravel()
    std = std.ravel()
    ax.plot(x, mean, label=f"Mean {label}")
    ax.fill_between(x, mean - 2 * std, mean + 2 * std, alpha=0.25, label=f"95% interval {label}")
    if samples is not None:
        ax.plot(x, samples, alpha=0.3)

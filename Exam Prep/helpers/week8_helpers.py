__all__ = [
    "GradientAscentOptimizer",
    "AdamOptimizer",
    "laplace_approximation",
    "create_linear_regression_data",
    "plot_vi_summary",
]

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import inv


class GradientAscentOptimizer:

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grad):
        return params + self.lr * grad


class AdamOptimizer:

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = None
        self.v = None

    def step(self, params, grad):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = (1 - self.beta1) * grad + self.beta1 * self.m
        self.v = (1 - self.beta2) * grad ** 2 + self.beta2 * self.v
        mhat = self.m / (1 - self.beta1 ** self.t)
        vhat = self.v / (1 - self.beta2 ** self.t)
        return params + self.lr * mhat / (np.sqrt(vhat) + self.eps)


def laplace_approximation(log_joint_fn, w_init):
    """Find MAP via scipy minimize, then compute Hessian-based covariance.

    Returns:
        w_map   -- MAP parameter vector (1D numpy array)
        cov     -- posterior covariance matrix (numpy array)
    """
    cost_fn = lambda w: -log_joint_fn(w)

    result = minimize(cost_fn, w_init, method="L-BFGS-B",
                      options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-8})
    if not result.success:
        print("Warning: Laplace approximation optimization failed:", result.message)

    w_map = result.x
    n = len(w_map)
    eps = 1e-5
    H = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        H[:, i] = (cost_fn(w_map + ei) - cost_fn(w_map - ei)) / (2 * eps)
    H = 0.5 * (H + H.T)
    cov = inv(H)
    return w_map, cov


def create_linear_regression_data(N=50, sigma=0.5, seed=0):
    """Generate simple 1D linear regression dataset y = x + noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, 1))
    y = X.ravel() + sigma * rng.standard_normal(N)
    return X, y


def plot_vi_summary(ax, x, mean, std, samples=None):
    """Plot predictive mean with 2-sigma credibility band and optional samples."""
    x = x.ravel()
    mean = np.asarray(mean).ravel()
    std = np.asarray(std).ravel()
    ax.plot(x, mean, label="Mean")
    ax.fill_between(x, mean - 2 * std, mean + 2 * std, alpha=0.25, label="95% interval")
    if samples is not None:
        ax.plot(x, np.asarray(samples), alpha=0.2, color="gray")

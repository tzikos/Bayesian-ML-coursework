__all__ = [
    "log_lik_sigmoid",
    "log_lik_probit",
    "log_lik_robust",
    "VariationalInference",
    "BlackBoxVariationalInference",
    "AdamOptimizer",
]

import jax.numpy as jnp
from jax import random, value_and_grad
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp
from time import time


# ---- helpers ----

_log_npdf = lambda x, m, v: -(x - m) ** 2 / (2 * v) - 0.5 * jnp.log(2 * jnp.pi * v)
_sigmoid = lambda f: 1 / (1 + jnp.exp(-f))
_phi = lambda f: norm.cdf(f)
_log_phi = lambda f: norm.logcdf(f)
_log_bernoulli_logit = lambda x, z: (1 - x) * jnp.log(1 - _sigmoid(z)) + x * jnp.log(_sigmoid(z))
_log_bernoulli_probit = lambda x, z: (1 - x) * jnp.log(1 - _phi(z)) + x * _log_phi(z)


# ---- Log-likelihood functions ----

def log_lik_sigmoid(X, y, w):
    """Log-likelihood for logistic regression with sigmoid link.

    X: (N, D), y: (N,), w: (S, D) -> (S,)
    """
    f = w @ X.T
    logits = _log_bernoulli_logit(y, f)
    return jnp.sum(logits, axis=1)


def log_lik_probit(X, y, w):
    """Log-likelihood for probit regression.

    X: (N, D), y: (N,), w: (S, D) -> (S,)
    """
    f = w @ X.T
    log_probits = _log_bernoulli_probit(y, f)
    return jnp.sum(log_probits, axis=1)


def log_lik_robust(X, y, w, epsilon=0.1):
    """Log-likelihood for robust logistic regression.

    X: (N, D), y: (N,), w: (S, D) -> (S,)
    """
    f = w @ X.T
    logits1 = jnp.log(1 - epsilon) + _log_bernoulli_logit(y, f)
    logits2 = jnp.log(epsilon) + _log_bernoulli_logit(y, -f)
    logits = logsumexp(jnp.stack((logits1, logits2)), axis=0)
    return jnp.sum(logits, axis=1)


# ---- Adam optimizer ----

class AdamOptimizer:

    def __init__(self, initial_param, num_params, step_size):
        self.num_params = num_params
        self.params = initial_param
        self.step_size = step_size
        self.itt = 0
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1e-8
        self.m = jnp.zeros(num_params)
        self.v = jnp.zeros(num_params)

    def step(self, gradient):
        self.m = (1 - self.b1) * gradient + self.b1 * self.m
        self.v = (1 - self.b2) * gradient ** 2 + self.b2 * self.v
        mhat = self.m / (1 - self.b1 ** (self.itt + 1))
        vhat = self.v / (1 - self.b2 ** (self.itt + 1))
        self.params = self.params + self.step_size * mhat / (jnp.sqrt(vhat) + self.eps)
        self.itt += 1
        return self.params


# ---- Analytical VI for linear Gaussian ----

class VariationalInference:
    """Analytical mean-field VI for linear Gaussian model.

    Hyperparameters sigma2 and kappa2 are fixed internally (sigma2=20, kappa2=10).
    """

    def __init__(self, X, y, alpha=None, sigma2=None, num_params=None, step_size=1e-2, max_itt=2000, verbose=False):
        """
        Accepts either (X, y, alpha, sigma2) or the original (num_params, step_size, ...) form.
        When X and y are provided, D is inferred from X.shape[1].
        """
        if X is not None and num_params is None:
            num_params = X.shape[1]

        self.X = X
        self.y = y
        self.num_params = num_params
        self.step_size = step_size
        self.max_itt = max_itt
        self.verbose = verbose
        self.num_var_params = 2 * num_params

        self.m = jnp.zeros(num_params)
        self.v = jnp.ones(num_params) / num_params
        self.lam = self.pack(self.m, self.v)

        self.optimizer = AdamOptimizer(initial_param=self.lam, num_params=self.num_var_params, step_size=self.step_size)
        self._compute_ELBO_and_grad = value_and_grad(self.compute_ELBO)

        self.ELBO = None
        self.ELBO_history = []
        self.lam_history = []

    def pack(self, m, v):
        """Pack mean m and variance v into lam = [m, log(v)]."""
        lam = jnp.zeros(self.num_var_params)
        lam = lam.at[:self.num_params].set(m)
        lam = lam.at[self.num_params:].set(jnp.log(v))
        return lam

    def unpack(self, lam):
        """Unpack lam = [m, log(v)] -> (m, v)."""
        m = lam[:self.num_params]
        v = jnp.exp(lam[self.num_params:])
        return m, v

    def _compute_entropy(self, v):
        return jnp.sum(0.5 * jnp.log(2 * jnp.pi * v) + 0.5)

    def compute_ELBO(self, lam, key=None):
        """Analytical ELBO for linear Gaussian model (sigma2=20, kappa2=10)."""
        m, v = self.unpack(lam)
        sigma2, kappa2 = 20., 10.
        expected_log_lik = (jnp.sum(_log_npdf(self.y, self.X @ m, sigma2))
                            - 1 / (2 * sigma2) * jnp.sum(self.X ** 2 @ v))
        expected_log_prior = jnp.sum(_log_npdf(m, 0, kappa2) - 1 / (2 * kappa2) * v)
        entropy = self._compute_entropy(v)
        return expected_log_lik + expected_log_prior + entropy

    def fit(self, seed=0, num_iterations=None):
        """Optimize ELBO with Adam. Returns self."""
        if num_iterations is not None:
            self.max_itt = num_iterations

        key = random.PRNGKey(seed)
        self.ELBO_history = []
        self.lam_history = []

        t0 = time()
        for itt in range(self.max_itt):
            key, elbo_key = random.split(key)
            self.ELBO, g = self._compute_ELBO_and_grad(self.lam, key=elbo_key)
            self.ELBO_history.append(self.ELBO)
            self.lam_history.append(self.lam)
            self.lam = self.optimizer.step(g)

            if self.verbose and (itt + 1) % 250 == 0:
                print(f'\tItt: {itt+1:5d}, ELBO = {jnp.mean(jnp.array(self.ELBO_history[-250:])):3.2f}')

        print(f'\tOptimization done in {time()-t0:3.2f}s\n')
        self.ELBO_history = jnp.array(self.ELBO_history)
        self.lam_history = jnp.array(self.lam_history)
        self.m = self.lam[:self.num_params]
        self.v = jnp.exp(self.lam[self.num_params:])
        self.m_history = self.lam_history[:, :self.num_params]
        self.v_history = jnp.exp(self.lam_history[:, self.num_params:])
        return self.lam

    def generate_posterior_samples(self, num_samples=1000, seed=0):
        """Sample from q(w) = N(m, diag(v)). Returns (num_samples, D)."""
        key = random.PRNGKey(seed)
        return self.m + jnp.sqrt(self.v) * random.normal(key, shape=(num_samples, self.num_params))


# ---- Black-Box VI ----

class BlackBoxVariationalInference(VariationalInference):
    """BBVI via reparameterisation trick.

    log_likelihood_fn signature: (X, y, w) -> (S,)
    """

    def __init__(self, X, y, log_likelihood_fn, alpha=None, D=None,
                 log_prior_fn=None, step_size=1e-2, max_itt=2000,
                 num_samples=20, batch_size=None, seed=0, verbose=False):

        if D is None:
            D = X.shape[1]

        # default isotropic Gaussian prior
        if log_prior_fn is None:
            prior_var = 10.
            log_prior_fn = lambda w: jnp.sum(_log_npdf(w, 0., prior_var), axis=1)

        self.log_prior_fn = log_prior_fn
        self.log_lik_fn = log_likelihood_fn
        self.num_samples = num_samples
        self.batch_size = batch_size

        super().__init__(X=X, y=y, num_params=D, step_size=step_size, max_itt=max_itt, verbose=verbose)
        self.N = len(X)
        self._compute_ELBO_and_grad = value_and_grad(self.compute_ELBO)

    def compute_ELBO(self, lam, key=None):
        """MC ELBO estimate with reparameterisation."""
        key_eps, key_batch = random.split(key)
        m, v = self.unpack(lam)
        epsilon = random.normal(key_eps, shape=(self.num_samples, self.num_params))
        w_samples = m + jnp.sqrt(v) * epsilon

        expected_log_prior = jnp.mean(self.log_prior_fn(w_samples))

        if self.batch_size:
            batch_idx = random.choice(key_batch, jnp.arange(self.N),
                                      shape=(self.batch_size,), replace=False)
            X_b, y_b = self.X[batch_idx, :], self.y[batch_idx]
            expected_log_lik = self.N / self.batch_size * jnp.mean(self.log_lik_fn(X_b, y_b, w_samples))
        else:
            expected_log_lik = jnp.mean(self.log_lik_fn(self.X, self.y, w_samples))

        return expected_log_lik + expected_log_prior + self._compute_entropy(v)

    def fit(self, seed=0, num_iterations=None, S=None):
        """Optimise ELBO. S overrides num_samples if provided."""
        if S is not None:
            self.num_samples = S
        return super().fit(seed=seed, num_iterations=num_iterations)

    def generate_posterior_samples(self, lam=None, num_samples=1000, seed=0):
        """Sample from q(w). lam optional (uses self.lam if None)."""
        if lam is None:
            lam = self.lam
        key = random.PRNGKey(seed)
        m, v = self.unpack(lam)
        return m + jnp.sqrt(v) * random.normal(key, shape=(num_samples, self.num_params))

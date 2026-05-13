__all__ = [
    "nn_predict",
    "NeuralNetwork",
    "plot_nn_predictions",
]

import jax.numpy as jnp
from jax import jit, grad, random
from jax.flatten_util import ravel_pytree
import optax


@jit
def log_npdf(x, m, v):
    return -0.5 * (x - m) ** 2 / v - 0.5 * jnp.log(2 * jnp.pi * v)


@jit
def nn_predict(params, inputs):
    """Forward pass: tanh hidden layers, linear output.

    params: list of (W, b) tuples.
    inputs: (N, D) array.
    Returns: (N, out_dim) array.
    """
    for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.tanh(outputs)
    return outputs


class NeuralNetwork:
    """Bayesian neural network trained to the MAP via Adam.

    layer_sizes: e.g. [1, 20, 20, 2]
    prior_var / noise_var: kept for interface compatibility; internally alpha = 1/prior_var.
    """

    def __init__(self, layer_sizes, X_train, y_train, prior_var=1., noise_var=None,
                 seed=0, log_lik_fun=None, alpha=None, step_size=0.01,
                 num_iters=1000, batch_size=None, verbose=200):

        self.layer_sizes = layer_sizes
        self.X = X_train
        self.y = y_train
        self.N = len(X_train)
        self.step_size = step_size
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.verbose = verbose
        self.seed = seed
        self.key = random.PRNGKey(seed)

        # alpha (prior precision) can be supplied directly or derived from prior_var
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = 1.0 / prior_var

        # log-likelihood function (default: Gaussian with unit variance)
        if log_lik_fun is not None:
            self.log_lik_fun = log_lik_fun
        else:
            sigma2 = noise_var if noise_var is not None else 1.
            self.log_lik_fun = lambda f, y: jnp.sum(log_npdf(y, f, sigma2))

        self.params = self.init_random_params()
        self.optimize()

    def init_random_params(self):
        """Xavier-style initialisation. Returns list of (W, b) tuples."""
        params = []
        for m, n in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.key, key_w, key_b = random.split(self.key, 3)
            W = jnp.sqrt(2 / n) * random.normal(key_w, shape=(m, n))
            b = jnp.sqrt(2 / n) * random.normal(key_b, shape=(n,))
            params.append((W, b))
        return params

    def predict(self, inputs):
        """Forward pass using current params. Returns (N, out_dim)."""
        return nn_predict(self.params, inputs)

    def first_layers(self, params, inputs):
        """Activations after all but the last layer."""
        for W, b in params[:-1]:
            outputs = jnp.dot(inputs, W) + b
            inputs = jnp.tanh(outputs)
        return inputs

    def last_layer(self, last_params, first_layers_out):
        W, b = last_params
        return jnp.dot(first_layers_out, W) + b

    def log_prior(self, params):
        flattened, _ = ravel_pytree(params)
        return jnp.sum(log_npdf(flattened, 0., 1 / self.alpha))

    def log_likelihood(self, params, key=None, compute_full=False):
        if self.batch_size is None or compute_full:
            f = nn_predict(params, self.X)
            return jnp.sum(self.log_lik_fun(f.reshape((self.N, -1)), self.y))
        else:
            batch_idx = random.choice(key, jnp.arange(self.N),
                                      shape=(self.batch_size,), replace=False)
            X_b, y_b = self.X[batch_idx, :], self.y[batch_idx]
            f = nn_predict(params, X_b)
            return self.N / self.batch_size * jnp.sum(
                self.log_lik_fun(f.reshape((self.batch_size, -1)), y_b))

    def log_joint(self, params, key=None, compute_full=False):
        return self.log_prior(params) + self.log_likelihood(params, key, compute_full)

    def optimize(self, num_iterations=None, batch_size=None, lr=None):
        """Adam training loop. Returns final params."""
        if num_iterations is not None:
            self.num_iters = num_iterations
        if lr is not None:
            self.step_size = lr

        @jit
        def objective(params, key):
            return -self.log_joint(params, key=key)

        objective_grad = grad(objective)
        optimizer = optax.adam(self.step_size)
        key = random.PRNGKey(self.seed)
        params = self.params
        opt_state = optimizer.init(params)

        for i in range(self.num_iters):
            key, key_opt = random.split(key)
            grads = objective_grad(params, key_opt)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if self.verbose and (i + 1) % max(1, int(self.num_iters / 20)) == 0:
                loss = self.log_joint(params, compute_full=True)
                print(f'Itt = {i+1}/{self.num_iters} ({100*(i+1)/self.num_iters:.1f}%), '
                      f'log joint = {loss:3.2f}')

        self.params = params
        return params


def plot_nn_predictions(ax, X_plot, mean, std, X_train=None, y_train=None, color='b', seed=0):
    """Plot mean prediction with 95% credibility band (mean +/- 1.96*std).

    mean, std: 1-D arrays aligned with X_plot.
    """
    X_plot = jnp.ravel(X_plot)
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std
    ax.plot(X_plot, mean, color=color, label='Mean prediction')
    ax.fill_between(X_plot, lower, upper, alpha=0.2, color=color, label='95% interval')
    if X_train is not None and y_train is not None:
        ax.scatter(X_train, y_train, s=15, color='k', label='Training data', zorder=5)
    ax.legend()

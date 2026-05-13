__all__ = [
    "build_energy_functions",
    "leapfrog",
    "HMC",
    "compute_Rhat",
    "compute_effective_sample_size",
    "combine_chains",
    "eval_density_grid",
    "plot_mcmc_summary",
]

import numpy as np
import jax.numpy as jnp
from jax import grad, jit, random
import matplotlib.pyplot as plt
from time import time


def build_energy_functions(log_target):
    """Returns (hamiltonian, potential_energy, kinetic_energy, grad_potential)."""

    def kinetic_energy(nu):
        return 0.5 * jnp.sum(nu ** 2)

    def potential_energy(theta):
        return -log_target(theta).squeeze()

    def hamiltonian(theta, nu):
        return potential_energy(theta) + kinetic_energy(nu)

    return hamiltonian, potential_energy, kinetic_energy, grad(potential_energy)


def leapfrog(theta0, nu0, grad_pot_fn, num_steps, step_size, return_trajectory=False):
    """Leapfrog integrator.

    Returns (theta, nu) or (theta, nu, theta_traj, nu_traj) when return_trajectory=True.
    theta0 and nu0 must have shape (1, D).
    """
    if return_trajectory:
        theta_traj = [theta0]
        nu_traj = [nu0]

    theta, nu = theta0, nu0
    for _ in range(num_steps):
        nu = nu - 0.5 * step_size * grad_pot_fn(theta)
        theta = theta + step_size * nu
        nu = nu - 0.5 * step_size * grad_pot_fn(theta)

        if return_trajectory:
            theta_traj.append(theta)
            nu_traj.append(nu)

    if return_trajectory:
        return theta, nu, jnp.vstack(theta_traj), jnp.vstack(nu_traj)
    return theta, nu


def HMC(log_target, num_iterations, theta0, num_leapfrog_steps=10, step_size=0.1, seed=0):
    """Full HMC sampler.

    Returns samples array of shape (num_iterations+1, D).
    """
    hamiltonian, _, _, grad_pot = build_energy_functions(log_target)
    hamiltonian = jit(hamiltonian)
    grad_pot = jit(grad_pot)

    key = random.PRNGKey(seed)
    param_shape = theta0.shape
    theta = theta0
    thetas = [theta0]
    accept_count = 0
    t0 = time()

    for itt in range(num_iterations):
        key, key_momentum, key_accept = random.split(key, 3)

        nu = random.normal(key_momentum, shape=param_shape)
        theta_star, nu_star = leapfrog(theta, nu, grad_pot, num_leapfrog_steps, step_size)

        log_accept_prob = -hamiltonian(theta_star, nu_star) + hamiltonian(theta, nu)
        u = random.uniform(key_accept)
        if jnp.log(u) < log_accept_prob:
            theta = theta_star
            accept_count += 1

        thetas.append(theta)

        if (itt + 1) % max(1, int(num_iterations / 10)) == 0:
            print(f'Itt {itt+1:5d}/{num_iterations} ({100*(itt+1)/num_iterations:6.2f}%), '
                  f'accept_rate={accept_count/(itt+1):.3f}, time={time()-t0:3.2f}s')

    print(f'HMC done in {time()-t0:.3f}s with avg. accept rate={accept_count/num_iterations:.3f}\n')
    return jnp.vstack(thetas)


# ---- Gelman-Rubin helpers ----

def _gelman_rubin(x):
    m_chains, n_iters = x.shape
    B_over_n = ((jnp.mean(x, axis=1) - jnp.mean(x)) ** 2).sum() / (m_chains - 1)
    W = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum() / (m_chains * (n_iters - 1))
    s2 = W * (n_iters - 1) / n_iters + B_over_n
    return s2


def _ess_single(x):
    m_chains, n_iters = x.shape
    variogram = lambda t: ((x[:, t:] - x[:, :(n_iters - t)]) ** 2).sum() / (m_chains * (n_iters - t))
    post_var = _gelman_rubin(x)
    t = 1
    rho = np.ones(n_iters)
    negative_autocorr = False
    while not negative_autocorr and t < n_iters:
        rho[t] = 1 - variogram(t) / (2 * post_var)
        if not t % 2:
            negative_autocorr = sum(rho[t - 1:t + 1]) < 0
        t += 1
    return int(m_chains * n_iters / (1 + 2 * rho[1:t].sum()))


def compute_Rhat(chains):
    """Gelman-Rubin R-hat. chains shape: (num_chains, num_samples, D). Returns array (D,)."""
    chains = np.array(chains)
    num_chains, num_samples, num_params = chains.shape
    half = int(0.5 * num_samples)
    sub_chains = []
    for i in range(num_chains):
        sub_chains.append(chains[i, :half, :])
        sub_chains.append(chains[i, half:, :])
    m = len(sub_chains)
    chain_means = np.array([np.mean(s, axis=0) for s in sub_chains])
    chain_vars = np.array([1 / (num_samples - 1) * np.sum((s - mu) ** 2, 0)
                           for s, mu in zip(sub_chains, chain_means)])
    global_mean = np.mean(chain_means, axis=0)
    B = num_samples / (m - 1) * np.sum((chain_means - global_mean) ** 2, axis=0)
    W = np.mean(chain_vars, 0)
    var_est = (num_samples - 1) / num_samples * W + (1 / num_samples) * B
    return np.sqrt(var_est / W)


def compute_effective_sample_size(chains):
    """ESS for each parameter. chains shape: (num_chains, num_samples, D). Returns array (D,)."""
    chains = np.array(chains)
    num_chains, num_samples, num_params = chains.shape
    return np.array([_ess_single(chains[:, :, i]) for i in range(num_params)])


def combine_chains(chains):
    """Flatten chains of shape (num_chains, num_samples, D) to (num_chains*num_samples, D)."""
    num_chains, num_samples, D = chains.shape
    return chains.reshape(num_chains * num_samples, D)


def eval_density_grid(density_fn, x_min=-5, x_max=5, P=100):
    """Evaluate a 2D density on a grid. Returns (xx, yy, zz) where zz has shape (P, P)."""
    x_grid = jnp.linspace(x_min, x_max, P)
    X1, X2 = jnp.meshgrid(x_grid, x_grid)
    XX = jnp.column_stack((X1.ravel(), X2.ravel()))
    zz = density_fn(XX).reshape((P, P))
    return x_grid, x_grid, zz


def plot_mcmc_summary(axes, samples, param_names=None):
    """Trace plots and histograms for MCMC samples.

    axes: flat array of 2*D axes (first D for traces, next D for histograms).
    samples: array of shape (num_samples, D).
    """
    D = samples.shape[1]
    if param_names is None:
        param_names = [f'theta_{i}' for i in range(D)]
    for i in range(D):
        axes[i].plot(samples[:, i])
        axes[i].set(xlabel='Iteration', ylabel=param_names[i], title=f'Trace: {param_names[i]}')
        axes[D + i].hist(samples[:, i], bins=30, density=True)
        axes[D + i].set(xlabel=param_names[i], title=f'Histogram: {param_names[i]}')

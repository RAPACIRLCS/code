from __future__ import annotations

from functools import partial
import multiprocessing as mp
from typing import (
    Callable,
    NamedTuple,
)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

from scs.LOGEnvironment1D import (
    ACTIONS,
    D_ACTIONS,
    D_STATES,
    GAMMA,
    MU,
    RESOLUTION,
    SIGMA,
    STATES,
    A,
    B,
    cost,
    initial_state,
    state_transition,
)


class MLBFParams(NamedTuple):
    """Parameters for a monomial linear basis function"""

    weights: jnp.ndarray
    powers: jnp.ndarray


def monomial_basis_function_states(x: jnp.ndarray, params: MLBFParams) -> jnp.ndarray:
    x_powers = jax.vmap(jnp.power, in_axes=(0, None), out_axes=0)(x, params.powers)
    return jnp.matmul(x_powers, params.weights)


def monomial_basis_function_state_actions(
    x: jnp.ndarray,
    a: jnp.ndarray,
    params: MLBFParams,
) -> jnp.ndarray:
    """
    Returns:
        The value of the linear monomial basis function for all state action
        pairs where the states are along axis 0 and the actions along axis 1.
    """
    x_powers = jax.vmap(jnp.power, in_axes=(0, None), out_axes=0)(x, params.powers)
    a_powers = jax.vmap(jnp.power, in_axes=(0, None), out_axes=0)(a, params.powers)
    outer_product = jnp.einsum("ij,lk->iljk", x_powers, a_powers)
    monomials = jnp.reshape(
        outer_product, (x.shape[0], a.shape[0], params.powers.shape[0] ** 2)
    )
    return jnp.matmul(monomials, params.weights)


def cost_function(
    state: np.ndarray, action: np.ndarray, weights: np.ndarray, powers: np.ndarray
) -> np.ndarray:
    x_powers = np.power(state, powers)
    a_powers = np.power(action, powers)
    monomials = np.outer(x_powers, a_powers).flatten()
    return monomials @ weights


def discrete_cost(cost_weights: np.ndarray, cost_powers: np.ndarray) -> np.ndarray:
    c = partial(cost_function, weights=cost_weights, powers=cost_powers)
    return np.array([[c(s, a) for a in D_ACTIONS] for s in D_STATES])


def value_function(
    state: np.ndarray, weights: np.ndarray, powers: np.ndarray
) -> np.ndarray:
    x_powers = np.power(state, powers)
    return x_powers @ weights


def discretized_state_action_transitions(
    subsequent_state: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
) -> np.ndarray:
    expected_value = subsequent_state - A * state - B * action
    standardized_value = (expected_value - MU) / SIGMA
    normalizer = np.sqrt(2 * np.pi) * SIGMA
    density = np.exp(-0.5 * standardized_value**2) / normalizer
    dy = (STATES[-1] - STATES[0]) / RESOLUTION
    return density * dy


def adjoint_bellmann(
    state: np.ndarray,
    action: np.ndarray,
    value_weights: np.ndarray,
    value_powers: np.ndarray,
) -> np.ndarray:
    v = partial(value_function, weights=value_weights, powers=value_powers)
    p_sa = partial(discretized_state_action_transitions, state=state, action=action)
    state_value = v(state)
    approx_integral = scipy.integrate.quad(
        lambda sp: v(sp) * p_sa(sp), STATES[0], STATES[-1]
    )[0]
    return state_value - GAMMA * approx_integral


def discrete_adjoint_bellmann(
    cost_weights: np.ndarray,
    cost_powers: np.ndarray,
    value_weights: np.ndarray,
    value_powers: np.ndarray,
) -> np.ndarray:
    output = np.zeros((len(D_STATES), len(D_ACTIONS)))
    for s, state in tqdm(list(enumerate(D_STATES))):
        for a, action in tqdm(list(enumerate(D_ACTIONS))):
            output[s, a] = adjoint_bellmann(state, action, value_weights, value_powers)
    return output


def expected_value_function(
    state: np.ndarray, action: np.ndarray, cost_powers: np.ndarray
) -> np.ndarray:
    p_sa = partial(discretized_state_action_transitions, state=state, action=action)
    return np.array(
        [
            scipy.integrate.quad(lambda sp: sp**p * p_sa(sp), STATES[0], STATES[-1])[
                0
            ]
            for p in cost_powers
        ]
    )


def det_policy_step(
    policy: jnp.ndarray, state: jnp.ndarray, key: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Performs one step following the determinsitic policy"""
    action = policy * state
    next_state, key = state_transition(state, action, key)
    return next_state, cost(state, action), key


def sample_initial_states(
    key: jnp.ndarray, n_samples: int = 1000
) -> tuple[jnp.ndarray, jnp.ndarray]:
    keys = jax.random.split(key, n_samples + 1)
    initial_states = jax.vmap(initial_state)(keys[1:])[0]
    return initial_states, keys[0]


def generate_state_trajectories_array(
    policy: jnp.ndarray,
    key: jnp.ndarray,
    n_trajectories: int = 1000,
    horizon: int = 1000,
    workers: int = 10,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if workers > 1:
        states, key = generate_state_trajectories_parallel(
            policy, key, n_trajectories, horizon, workers
        )
    else:
        states, key = generate_state_trajectories(policy, key, n_trajectories, horizon)
    return jnp.array(np.array(states)), key  # Faster than direct jnp.array conversion


def generate_truncated_sample_trajectories(
    policy: jnp.ndarray,
    key: jnp.ndarray,
    n_trajectories: int,
    horizon: int = 1000,
    workers: int = 10,
) -> tuple[list[tuple[jnp.ndarray, jnp.ndarray]], jnp.ndarray]:
    if workers > 1:
        samples, key = generate_state_action_trajectories_parallel(
            policy, key, n_trajectories, horizon, workers
        )
    else:
        samples, key = generate_state_action_trajectories(
            policy, key, n_trajectories, horizon
        )
    return samples, key


def generate_state_trajectories(
    policy: jnp.ndarray,
    key: jnp.ndarray,
    n_trajectories: int = 1000,
    horizon: int = 1000,
) -> tuple[list[list[jnp.ndarray]], jnp.ndarray]:
    """Generates trajectories following the deterministic policy"""
    states: list[jnp.ndarray] = []
    for _ in tqdm(range(n_trajectories)):
        state, key = initial_state(key)
        states.append([])
        for _ in range(horizon):
            state, _, key = jax.jit(det_policy_step)(policy, state, key)
            states[-1].append(state)
    return states, key


def _worker_generate_state_trajectory(policy, key, horizon) -> list[jnp.ndarray]:
    states = []
    state, key = initial_state(key)
    for _ in range(horizon):
        state, _, key = jax.jit(det_policy_step)(policy, state, key)
        states.append(state)
    return states


def generate_state_trajectories_parallel(
    policy: jnp.ndarray,
    key: jnp.ndarray,
    n_trajectories: int = 1000,
    horizon: int = 1000,
    workers: int = 10,
) -> tuple[list[list[np.ndarray]], jnp.ndarray]:
    states: list[jnp.ndarray] = []
    keys = jax.random.split(key, n_trajectories + 1)
    tasks = [(policy, key, horizon) for key in keys[1:]]
    with mp.Pool(workers) as pool:
        states = pool.starmap(
            _worker_generate_state_trajectory, tqdm(tasks, total=n_trajectories)
        )
    return states, keys[0]


def generate_state_action_trajectories(
    policy: jnp.ndarray,
    key: jnp.ndarray,
    n_trajectories: int = 1000,
    horizon: int = 1000,
) -> tuple[list[list[jnp.ndarray]], jnp.ndarray]:
    """Generates trajectories following the deterministic policy"""
    states: list[jnp.ndarray] = []
    for _ in tqdm(range(n_trajectories)):
        state, key = initial_state(key)
        states.append([])
        for _ in range(horizon):
            action = policy * state
            state, _, key = jax.jit(det_policy_step)(policy, state, key)
            states[-1].append((state, action))
    return states, key


def _worker_generate_state_action_trajectory(policy, key, horizon) -> list[jnp.ndarray]:
    states = []
    state, key = initial_state(key)
    for _ in range(horizon):
        action = policy * state
        state, _, key = jax.jit(det_policy_step)(policy, state, key)
        states.append((state, action))
    return states


def generate_state_action_trajectories_parallel(
    policy: jnp.ndarray,
    key: jnp.ndarray,
    n_trajectories: int = 1000,
    horizon: int = 1000,
    workers: int = 10,
) -> tuple[list[list[np.ndarray]], jnp.ndarray]:
    states: list[jnp.ndarray] = []
    keys = jax.random.split(key, n_trajectories + 1)
    tasks = [(policy, key, horizon) for key in keys[1:]]
    with mp.Pool(workers) as pool:
        states = pool.starmap(
            _worker_generate_state_action_trajectory, tqdm(tasks, total=n_trajectories)
        )
    return states, keys[0]


def state_frequencies(
    policy: jnp.ndarray,
    key: jnp.ndarray,
    resolution: int,
    workers: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Approximates the discounted state visiting frequencies based on the deterministic
    policy. The statespace is discretized into `resolution` bins, the bins can be
    therefore be treated as the states.
    """
    trajectories, key = generate_state_trajectories_array(policy, key, workers=workers)
    step_distributions: list[jnp.ndarray] = []
    states = jnp.linspace(STATES[0], STATES[1], resolution)
    for t, trajectory in enumerate(trajectories.T):
        hist, _ = jnp.histogram(trajectory, bins=states, range=STATES)
        step_distributions.append((hist / trajectories.shape[0]) * GAMMA**t)
    return jnp.sum(jnp.array(step_distributions), axis=0), states, key


def occupational_measure(
    policy: jnp.ndarray, key: jnp.ndarray, resolution: int = 1000, workers: int = 10
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Constructs the approximated occupational measure by assigning the deterministic
    action, following the optimal deterministic policy, to each state.
    Since the probability of performing an action not covered by states * policy
    is zero, just the actually possible state action pairs are considered.
    This allows for an increased resolution in the action space as its realized
    range is smaller than that of the state space.

    Returns:
        occupational_measure: The approximated state action occupational measure.
        states: The discretized states.
        actions: The discretized actions.
    """
    state_freqs, states, key = state_frequencies(policy, key, resolution, workers)
    actions = jnp.linspace(ACTIONS[0], ACTIONS[1], resolution)
    occupational_measure = np.zeros((resolution, resolution))
    state_actions = jnp.expand_dims(states * policy, axis=1)
    action_indices = jnp.argmin(
        jnp.abs((state_actions - jnp.expand_dims(actions, axis=0))), axis=1
    )
    occupational_measure[
        jnp.arange(state_freqs.shape[0]), action_indices[:-1]
    ] = state_freqs  # Histogram bins are one less than the states linspace
    return (
        jnp.array(occupational_measure),
        states,
        actions,
    )


def get_required_sample_size(n: int, epsilon: float, delta: float) -> int:
    for samples in range(100000):
        value = np.sum(
            [
                scipy.special.comb(samples, i)
                * epsilon**i
                * (1 - epsilon) ** (samples - i)
                for i in range(n)
            ]
        )
        if value <= delta:
            break
    return samples


def plot_compare_cost_functions(true_cost: Callable, estimated_cost: Callable) -> None:
    s_range = np.linspace(STATES[0], STATES[-1], 1000)
    a_range = np.linspace(ACTIONS[0], ACTIONS[-1], 1000)
    s, a = np.meshgrid(s_range, a_range)
    cost_function_true = np.zeros(s.shape)
    cost_function_approx = np.zeros(s.shape)

    for i in tqdm(range(s.shape[0])):
        for j in range(s.shape[1]):
            cost_function_true[i, j] = true_cost(s[i, j], a[i, j])
            cost_function_approx[i, j] = estimated_cost(s[i, j], a[i, j])

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    ax1.plot_surface(s, a, cost_function_true, cmap="viridis")
    ax1.set_title("True Cost Function")
    ax2.plot_surface(s, a, cost_function_approx, cmap="viridis")
    ax2.set_title("Approximate Cost Function")
    plt.show()

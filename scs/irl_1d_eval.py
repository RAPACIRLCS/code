from __future__ import annotations

import multiprocessing as mp
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from tqdm import tqdm

from scs.LOGEnvironment1D import (
    ACTIONS,
    GAMMA,
    STATES,
    discrete_kernel,
    initial_state,
    state_transition,
)


def eval_policy(state: jnp.ndarray, optimal_policy: jnp.ndarray) -> jnp.ndarray:
    return optimal_policy * state


def irl_policy(
    state: jnp.ndarray, states: jnp.ndarray, policy: jnp.ndarray
) -> jnp.ndarray:
    state_index = jnp.argmin(jnp.abs(states - state))
    return policy[state_index]


def evaluate_cost_function(
    policy: Callable,
    cost_function: Callable,
    key: jnp.ndarray,
    n_trajectories: int = 1000,
    horizon: int = 1000,
) -> tuple[np.ndarray, jnp.ndarray]:
    rewards = []
    for _ in tqdm(range(n_trajectories)):
        rewards.append([])
        state, key = initial_state(key)
        for _ in range(horizon):
            action = policy(state)
            reward = cost_function(state, action)
            next_state, key = state_transition(state, action, key)
            rewards[-1].append(reward)
            state = next_state
    return np.array(rewards), key


def evaluate_cost_function_on_initial_states(
    policy: Callable,
    cost_function: Callable,
    key: jnp.ndarray,
    start_states: jnp.ndarray,
    horizon: int = 1000,
) -> tuple[np.ndarray, jnp.ndarray]:
    rewards = []
    for inital_state in tqdm(start_states):
        rewards.append([])
        state = inital_state
        for _ in range(horizon):
            action = policy(state)
            reward = cost_function(state, action)
            next_state, key = state_transition(state, action, key)
            rewards[-1].append(reward)
            state = next_state
    return np.array(rewards), key


def _worker_evaluate_trajectory(
    policy: Callable,
    cost_function: Callable,
    key: jnp.ndarray,
    horizon: int,
    start_state: None | jnp.ndarray = None,
) -> tuple[list[float], jnp.ndarray]:
    cost = []
    if start_state is not None:
        state = start_state
    else:
        state, key = initial_state(key)
    for _ in range(horizon):
        action = policy(state)
        reward = cost_function(state, action)
        state, key = state_transition(state, action, key)
        cost.append(reward)
    return cost


def evaluate_cost_function_parallel(
    policy: Callable,
    cost_function: Callable,
    key: jnp.ndarray,
    n_trajectories: int = 1000,
    horizon: int = 1000,
    workers: int = 10,
    start_states: None | jnp.ndarray = None,
) -> tuple[np.ndarray, jnp.ndarray]:
    keys = jax.random.split(key, n_trajectories + 1)
    if start_states is not None:
        tasks = [
            (policy, cost_function, key, horizon, start_state)
            for key, start_state in zip(keys[1:], start_states)
        ]
    else:
        tasks = [(policy, cost_function, key, horizon) for key in keys[1:]]
    with mp.Pool(workers) as pool:
        rewards = pool.starmap(
            _worker_evaluate_trajectory, tqdm(tasks, total=n_trajectories)
        )
    return np.array(rewards), keys[0]


def value_iteration(
    cost_function: Callable,
    resolution: int = 1000,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    error = []
    states = jnp.linspace(STATES[0], STATES[1], resolution)
    actions = jnp.linspace(ACTIONS[0], ACTIONS[1], resolution)
    state_action_values = jnp.zeros((resolution, resolution))
    kernel = discrete_kernel(states, actions)
    costs = cost_function(states, actions)
    for _ in tqdm(range(max_iterations)):
        update = costs + GAMMA * jnp.einsum(
            "ijk,k->ij", kernel, jnp.min(state_action_values, axis=1)
        )
        error.append(jnp.sum((update - state_action_values) ** 2))
        state_action_values = update
    return np.array(state_action_values), np.array(error)


def success_probabilities(
    deltas: np.ndarray, epsilons: np.ndarray, eps: float
) -> np.ndarray:
    threshold = ((2 - GAMMA) / (1 - GAMMA)) * (epsilons + eps)
    successes = deltas <= threshold
    return np.mean(successes, axis=1)


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


def gaussien_CDF(x: float) -> float:
    return 0.5 * (1 + scipy.special.erf(x / np.sqrt(2)))


def closed_form_N_lower_bound(n: int, epsilon: float, delta: float) -> float:
    n = np.array(n, dtype=np.float128)
    epsilon = np.array(epsilon, dtype=np.float128)
    delta = np.array(delta, dtype=np.float128)
    return (
        2 / epsilon * np.log(1 / delta)
        + 2 * n
        + (2 * n) / epsilon * np.log(2 / epsilon)
    )

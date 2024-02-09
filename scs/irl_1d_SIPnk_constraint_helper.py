from __future__ import annotations

import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import multiprocessing as mp

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from scs.LOGEnvironment1D import (
    ACTIONS,
    D_STATES,
    GAMMA,
    RESOLUTION,
    STATES,
    initial_state,
    state_transition,
)


def get_empiric_kernel(
    states_actions: jnp.ndarray, k: int, key: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    state_transitions = []
    keys = jax.random.split(key, states_actions.shape[0] + 1)
    for i, (state, action) in tqdm(list(enumerate(states_actions))):
        state_transitions.append(
            get_empiric_state_transitions(state, action, k, keys[i])[0]
        )
    return jnp.array(state_transitions), keys[-1]


def get_empiric_state_transitions(
    state: jnp.ndarray, action: jnp.ndarray, k: int, key: jnp.ndarray
) -> tuple[np.ndarray, jnp.ndarray]:
    next_states, keys = get_subsequent_states(state, action, k, key)
    discret_state_indices = jnp.argmin(
        jnp.abs(jnp.expand_dims(next_states, axis=1) - D_STATES), axis=1
    )
    kernel = jnp.zeros(RESOLUTION).at[discret_state_indices].add(1) / k
    return kernel, keys[0]


def get_subsequent_states(
    state: jnp.ndarray, action: jnp.ndarray, k: int, key: jnp.ndarray
) -> tuple[np.ndarray, jnp.ndarray]:
    state_array = jnp.ones(k) * state
    action_array = jnp.ones(k) * action
    keys = jax.random.split(key, k + 1)
    next_states, _ = jax.vmap(state_transition)(state_array, action_array, keys[1:])
    return next_states, keys[0]


def generate_ub_constraint_block_one_mu(
    states_actions: jnp.ndarray,
    value_powers: jnp.ndarray,
    cost_powers: jnp.ndarray,
    mu_k: jnp.ndarray,
    K: int,
    key: jnp.ndarray,
    states: jnp.ndarray,
    actions: jnp.ndarray,
) -> np.ndarray:
    v_shape = value_powers.shape[0]
    c_shape = cost_powers.shape[0] ** 2
    monomial_matrix = np.zeros((states_actions.shape[0], c_shape + v_shape))
    for sample, (s, a) in tqdm(list(enumerate(states_actions))):
        monomial_matrix[sample, :-v_shape] = np.outer(
            np.power(s, cost_powers), np.power(a, cost_powers)
        ).flatten()
        monomial_matrix[sample, -v_shape:] = -np.power(s, value_powers)
        next_states, key = _get_subsequent_states(s, a, K, key)
        next_state_value_monomials = jax.vmap(jnp.power, in_axes=(0, None), out_axes=0)(
            next_states, value_powers
        )
        average_next_state_value_monomial = jnp.mean(next_state_value_monomials, axis=0)
        monomial_matrix[sample, -v_shape:] += average_next_state_value_monomial
        state_index = np.argmin(np.abs(states - s))
        action_index = np.argmin(np.abs(actions - a))
        monomial_matrix[sample, :] *= mu_k[state_index, action_index]
    return np.sum(monomial_matrix, axis=0), key


def _worker_discounted_trajectory_cost_monomial(
    policy: jnp.ndarray,
    cost_powers: jnp.ndarray,
    key: jnp.ndarray,
    horizon: int = 1000,
) -> np.ndarray:
    state, key = initial_state(key)
    monomial = np.outer(
        np.power(state, cost_powers), np.power(state * policy, cost_powers)
    ).flatten()
    for t in range(horizon):
        action = state * policy
        state, key = state_transition(state, action, key)
        monomial += np.outer(
            np.power(state, cost_powers), np.power(action, cost_powers)
        ).flatten() * GAMMA ** (t + 1)
    return monomial


def generate_ub_constraint_block_one(
    value_powers: jnp.ndarray,
    cost_powers: jnp.ndarray,
    policy: jnp.ndarray,
    K: int,
    key: jnp.ndarray,
    workers: int = 10,
) -> np.ndarray:
    v_shape = value_powers.shape[0]
    c_shape = cost_powers.shape[0] ** 2
    monomial = np.zeros((c_shape + v_shape))
    initial_state_value_monomial = np.zeros(v_shape)
    for _ in tqdm(range(K)):
        state, key = initial_state(key)
        initial_state_value_monomial += np.power(state, value_powers) / K
    keys = jax.random.split(key, K + 1)
    tasks = [(policy, cost_powers, key) for key in keys[1:]]
    with mp.Pool(workers) as pool:
        monomials = pool.starmap(
            _worker_discounted_trajectory_cost_monomial, tqdm(tasks, total=K)
        )
    monomial[:-v_shape] = np.mean(monomials, axis=0)
    monomial[-v_shape:] = -initial_state_value_monomial
    return monomial, keys[0]


def _get_subsequent_states(
    state: jnp.ndarray, action: jnp.ndarray, k: int, key: jnp.ndarray
) -> tuple[np.ndarray, jnp.ndarray]:
    state_array = jnp.ones(k) * state
    action_array = jnp.ones(k) * action
    keys = jax.random.split(key, k + 1)
    next_states, _ = jax.vmap(state_transition)(state_array, action_array, keys[1:])
    return next_states, keys[0]


def generate_ub_constraint_block_two(
    value_powers: jnp.ndarray,
    cost_powers: jnp.ndarray,
    N: int,
    K: int,
    key: jnp.ndarray,
) -> np.ndarray:
    v_shape = value_powers.shape[0]
    c_shape = cost_powers.shape[0] ** 2
    sample_key, key = jax.random.split(key, 2)
    states_actions = jax.random.uniform(
        sample_key,
        minval=jnp.array([STATES[0], ACTIONS[0]]),
        maxval=jnp.array([STATES[-1], ACTIONS[-1]]),
        shape=(N, 2),
    )
    monomial_matrix = np.ones((states_actions.shape[0], 1 + c_shape + v_shape))
    keys = jax.random.split(key, states_actions.shape[0] + 1)
    for i, (s, a) in tqdm(list(enumerate(states_actions))):
        monomial_matrix[i, 1 : 1 + c_shape] = np.outer(
            np.power(s, cost_powers), np.power(a, cost_powers)
        ).flatten()
        monomial_matrix[i, -v_shape:] = -np.power(s, value_powers)
        next_states, _ = _get_subsequent_states(s, a, K, keys[i])
        next_state_value_monomials = jax.vmap(jnp.power, in_axes=(0, None), out_axes=0)(
            next_states, value_powers
        )
        average_next_state_value_monomial = jnp.mean(next_state_value_monomials, axis=0)
        monomial_matrix[i, -v_shape:] += average_next_state_value_monomial
    return monomial_matrix, keys[-1]

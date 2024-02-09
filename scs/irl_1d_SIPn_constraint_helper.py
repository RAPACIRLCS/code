from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from scs.LOGEnvironment1D import (
    D_ACTIONS,
    D_STATES,
    GAMMA,
    discrete_kernel,
)
from scs.irl_1d_utils import discretized_state_action_transitions


def generate_constraint_matrix_block_one(
    value_powers: np.ndarray, cost_powers: np.ndarray, states: jnp.ndarray, degree: int
) -> np.ndarray:
    monomial_matrix = np.ones(
        (states.shape[0], D_ACTIONS.shape[0], 1 + degree**2 + degree)
    )
    for s, state in tqdm(list(enumerate(states))):
        for a, action in enumerate(D_ACTIONS):
            s_monomials = np.power(state, cost_powers)
            a_monomials = np.power(action, cost_powers)
            monomial_matrix[s, a, 1 : 1 + degree**2] = np.outer(
                s_monomials, a_monomials
            ).flatten()
            monomial_matrix[s, a, -degree:] = -s_monomials
    state_value_monomials = jax.vmap(jnp.power, in_axes=(0, None), out_axes=0)(
        states, value_powers
    )
    d_kernel = discrete_kernel(states, D_ACTIONS)
    weighted_sv_monomials = np.sum(
        np.einsum("ijk,kl->ijkl", d_kernel, state_value_monomials), axis=2
    )
    monomial_matrix[..., -degree:] += GAMMA * weighted_sv_monomials
    return monomial_matrix


def generate_samples_constraint_matrix(
    value_powers: np.ndarray,
    cost_powers: np.ndarray,
    states_actions: jnp.ndarray,
    degree: int,
) -> np.ndarray:
    monomial_matrix = np.ones((states_actions.shape[0], 1 + degree**2 + degree))
    state_value_monomials = jax.vmap(jnp.power, in_axes=(0, None), out_axes=0)(
        D_STATES, value_powers
    )
    for i, (s, a) in tqdm(list(enumerate(states_actions))):
        s_cost_monomials = np.power(s, cost_powers)
        a_cost_monomials = np.power(a, cost_powers)
        monomial_matrix[i, 1 : 1 + degree**2] = np.outer(
            s_cost_monomials, a_cost_monomials
        ).flatten()
        monomial_matrix[i, -degree:] = -np.power(s, value_powers)
        d_kernel = discretized_state_action_transitions(D_STATES, s, a)
        weighted_sv_monomials = state_value_monomials.T @ d_kernel
        monomial_matrix[i, -degree:] += GAMMA * weighted_sv_monomials
    return monomial_matrix

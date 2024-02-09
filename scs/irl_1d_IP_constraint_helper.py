from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from tqdm import tqdm

from scs.LOGEnvironment1D import (
    ACTIONS,
    D_ACTIONS,
    D_STATES,
    GAMMA,
    STATES,
    discrete_kernel,
)
from scs.irl_1d_utils import discretized_state_action_transitions


def integrated_cost_basis_function(powers: np.ndarray) -> np.ndarray:
    powers = np.stack(np.meshgrid(powers, powers), axis=0).reshape(2, -1).T
    return np.array(
        [
            scipy.integrate.nquad(
                lambda s, a: s**sp * a**ap,
                [[STATES[0], STATES[-1]], [ACTIONS[0], ACTIONS[-1]]],
            )[0]
            for sp, ap in powers
        ]
    )


def integrated_value_basis_function(powers: np.ndarray) -> np.ndarray:
    return np.array(
        [
            scipy.integrate.quad(lambda s: s**p, STATES[0], STATES[-1])[0]
            for p in powers
        ]
    )


def integral_expected_values(value_powers: np.ndarray) -> np.ndarray:
    integrated_values = []
    for p in value_powers:
        wdp = partial(weighted_state_power, value_power=p)
        integrated_values.append(
            scipy.integrate.nquad(
                wdp,
                [[STATES[0], STATES[-1]], [ACTIONS[0], ACTIONS[-1]]],
            )[0]
        )
    return np.array(integrated_values)


def weighted_state_power(
    state: np.ndarray, action: np.ndarray, value_power: np.ndarray
) -> np.ndarray:
    p_sa = partial(discretized_state_action_transitions, state=state, action=action)
    return scipy.integrate.quad(
        lambda sp: sp**value_power * p_sa(sp), STATES[0], STATES[-1]
    )[0]


def generate_constraint_matrix_block_one(
    value_powers: np.ndarray, cost_powers: np.ndarray, degree: int
) -> np.ndarray:
    monomial_matrix = np.ones(
        (D_STATES.shape[0], D_ACTIONS.shape[0], 1 + degree**2 + degree)
    )
    state_value_monomials = jax.vmap(jnp.power, in_axes=(0, None), out_axes=0)(
        D_STATES, value_powers
    )
    for s, state in tqdm(list(enumerate(D_STATES))):
        for a, action in enumerate(D_ACTIONS):
            s_monomials = np.power(state, cost_powers)
            a_monomials = np.power(action, cost_powers)
            monomial_matrix[s, a, 1 : 1 + degree**2] = np.outer(
                s_monomials, a_monomials
            ).flatten()
            monomial_matrix[s, a, -degree:] = -np.power(state, value_powers)
    d_kernel = discrete_kernel(D_STATES, D_ACTIONS)
    weighted_sv_monomials = np.sum(
        np.einsum("ijk,kl->ijkl", d_kernel, state_value_monomials), axis=2
    )
    monomial_matrix[..., -degree:] += GAMMA * weighted_sv_monomials
    return monomial_matrix

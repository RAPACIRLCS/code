from __future__ import annotations

import jax
import jax.numpy as jnp
from scipy.linalg import solve_discrete_are

# Environmnt Parameters
STATES: tuple[int, int] = (-10, 10)
ACTIONS: tuple[int, int] = (-10, 10)
A: float = -1.5
B: float = 1.0
Q: float = 1.0
R: float = 1.0
MU: float = 0.0
SIGMA: float = 1.0
GAMMA: float = 0.99

RESOLUTION: int = 1000

D_STATES: jnp.ndarray = jnp.linspace(*STATES, RESOLUTION)
D_ACTIONS: jnp.ndarray = jnp.linspace(*ACTIONS, RESOLUTION)


def initial_state(key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Samples an initial state, uniformly from the entire state space"""
    aux_key, state_selection_key = jax.random.split(key)
    return (
        jax.random.uniform(
            state_selection_key,
            minval=STATES[0],
            maxval=ACTIONS[1],
        ),
        aux_key,
    )


def state_transition(
    x: jnp.ndarray, a: jnp.ndarray, key: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The transition dynamics of the environment.

    Performs the environment transition based on the state `x`, action `a`, and
    the random normal noise `w`.
    """
    aux_key, noise_key = jax.random.split(key)
    w = MU + SIGMA * jax.random.normal(noise_key)
    next_state = A * x + B * a + w
    return jnp.clip(next_state, *STATES), aux_key


def cost(x: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """The quadratic cost function of the environment"""
    return Q * x**2 + R * a**2


def prob_density(
    y: jnp.ndarray,
    x: jnp.ndarray,
    a: jnp.ndarray,
) -> jnp.ndarray:
    """
    Density function based on the expected transition dynamics of the environment
    """
    expected_value = y - A * x - B * a
    standardized_value = (expected_value - MU) / SIGMA
    normalizer = jnp.sqrt(2 * jnp.pi) * SIGMA
    return jnp.exp(-0.5 * standardized_value**2) / normalizer


def kernel(
    x: jnp.ndarray,
    a: jnp.ndarray,
    states: jnp.ndarray,
) -> jnp.ndarray:
    density = prob_density(states, x, a)
    dy = jnp.diff(states).mean()
    return density * dy


def discrete_kernel(
    states: jnp.ndarray,
    actions: jnp.ndarray,
) -> jnp.ndarray:
    return jax.vmap(
        jax.vmap(kernel, in_axes=(None, 0, None), out_axes=0),
        in_axes=(0, None, None),
        out_axes=0,
    )(
        states, actions, states
    )  # returns dimensions (states, actions, subsequent_states)


def discrete_CDF(
    x: jnp.ndarray,
    a: jnp.ndarray,
    lower: float,
    upper: float,
    resolution: int = 100,
) -> jnp.ndarray:
    """
    Approximates the CDF of the passed subsequent state range based on a state
    action pair.
    """
    integrate_over = jnp.linspace(lower, upper, resolution)
    values = kernel(integrate_over, x, a)
    return jax.scipy.integrate.trapezoid(values, integrate_over, axis=0)


def optimal_deterministic_policy() -> jnp.ndarray:
    """
    Solves for the optimal deterministic policy via the algebraic Riccati equation.
    The optimal deterministic policy for the 1D system at hand is a scalar `k`
    such that the optimal policy is given by `a = k * x`.
    """
    p = solve_discrete_are(
        A,
        B,
        Q,
        R,
    )[0, 0]
    nominator = B * p * A
    denominator = R + B**2 * p
    return -GAMMA * nominator / denominator


def steady_state_distribution(policy: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the parameterization of the random normal noise `w` from the models
    transition dynamics, which is the steady state distribution of the system,
    if the policy results in a stable system.
    """
    stability_measure = jnp.abs(A + B * policy)
    if stability_measure < 1:
        return MU, SIGMA
    else:
        raise ValueError(
            f"Policy is not stable: |A + B * policy| = {stability_measure} >= 1"
        )

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

from scs.LOGEnvironment1D import (
    MU,
    SIGMA,
    STATES,
    A,
    B,
    Q,
    R,
    optimal_deterministic_policy,
)
from scs.irl_1d_eval import (
    closed_form_N_lower_bound,
    gaussien_CDF,
)

if TYPE_CHECKING:
    import jax.numpy as jnp

################################################################################
DEGREE: int = 3
THETA: float = 50
EPSILON: float = 1.001e-4
EPSILON_DELTA: float = 0.001e-4
DELTAS: np.ndarray = np.linspace(0.001, 1, 1000)
OUT_FILE: str = "data/experiment_2_sample_complexity.csv"
SHOW_PLOT: bool = True
################################################################################


key: jnp.ndarray = jax.random.PRNGKey(0)
k_star: jnp.ndarray = optimal_deterministic_policy()
cost_weights = np.ones(DEGREE**2)
cost_powers = np.arange(DEGREE)
value_weights = np.ones(DEGREE)
value_powers = np.arange(DEGREE)


def g(r: float) -> float:
    """
    Lower bound on the probability of ending up in a state not further away than
    r from any state s by performing any action a in s.
    """
    return (r**2 * np.pi) / (16 * STATES[1] ** 2)


# Values required for Theorem 4.1
L = np.max(STATES)
L_cost = 2 * np.max([Q, R]) * L
L_value = 2 * scipy.linalg.solve_discrete_are(A, B, Q, R)[0, 0] * L
L_p = (2 * L * np.max([A, B])) / (
    SIGMA**2
    * np.sqrt(2 * np.pi)
    * (gaussien_CDF((L - MU) / SIGMA) - gaussien_CDF((-L - MU) / SIGMA))
)

L_lambda = THETA * (
    np.sqrt(cost_weights.shape[0]) * L_cost
    + np.sqrt(value_weights.shape[0]) * L_value * (L_p + 1)
)

N_lower = {"minus_delta": [], "no_delta": [], "plus_delta": []}
n = cost_weights.shape[0] + value_weights.shape[0] + 1
for delta in tqdm(DELTAS[::-1]):
    N_lower["minus_delta"].append(
        closed_form_N_lower_bound(n, g((EPSILON - EPSILON_DELTA) / L_lambda), delta)
    )
    N_lower["no_delta"].append(
        closed_form_N_lower_bound(n, g(EPSILON / L_lambda), delta)
    )
    N_lower["plus_delta"].append(
        closed_form_N_lower_bound(n, g((EPSILON + EPSILON_DELTA) / L_lambda), delta)
    )


with open(OUT_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["delta, eps_minus", "eps", "eps_plus"])
    writer.writerows(
        zip(
            DELTAS[::-1],
            N_lower["minus_delta"],
            N_lower["no_delta"],
            N_lower["plus_delta"],
        )
    )

if SHOW_PLOT:
    x_min = np.min([np.min(values) for values in N_lower.values()])
    x_max = np.max([np.max(values) for values in N_lower.values()])
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    for name, values in N_lower.items():
        ax.plot(values, DELTAS, label=name)
    ax.plot([x_min, x_max], [1, 1], color="black", linestyle="--")
    ax.set_xlim([x_min, x_max])
    ax.set_xscale("log")
    plt.title("Sample complexity of Theorem 4.1")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$1-\delta$")
    plt.show()

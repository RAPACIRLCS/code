from __future__ import annotations

import csv
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from scs.LOGEnvironment1D import (
    ACTIONS,
    D_STATES,
    GAMMA,
    RESOLUTION,
    initial_state,
    optimal_deterministic_policy,
)
from scs.irl_1d_eval import (
    eval_policy,
    evaluate_cost_function_parallel,
    irl_policy,
    value_iteration,
)
from scs.irl_1d_utils import (
    MLBFParams,
    cost_function,
    monomial_basis_function_state_actions,
)
from scs.utils import parse_param_files

################################################################################
DEGREE: int = 3
N_VALUES: list[int] = [
    5,
    10,
    20,
    50,
    75,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
]
WORKERS: int = 15
OUT_FILE: str = "data/experiment_2_performance_of_learnt_cost.csv"
SHOW_PLOT: bool = True
################################################################################


key: jnp.ndarray = jax.random.PRNGKey(0)
k_star: jnp.ndarray = optimal_deterministic_policy()

indices_cost = np.arange(DEGREE**2) + 1
cost_powers = np.arange(DEGREE)

parameters = np.array(
    [parse_param_files(f"data/experiment_2/params_{n}.txt") for n in N_VALUES]
)
mean_parameters = np.mean(parameters, axis=1)

keys = jax.random.split(key, 1001)
start_state_keys = keys[1:]
key = keys[0]


start_states, _ = jax.vmap(initial_state)(start_state_keys)

values_eval_policy = {"mean": [], "var": [], "std": [], "se": []}
values_optimal_policy = {"mean": [], "var": [], "std": [], "se": []}
for i, n in enumerate(N_VALUES):
    cost_weights = mean_parameters[i, indices_cost]
    irl_cost = partial(cost_function, weights=cost_weights, powers=cost_powers)
    state_action_values, _ = value_iteration(
        partial(
            monomial_basis_function_state_actions,
            params=MLBFParams(cost_weights, cost_powers),
        ),
        max_iterations=1000,
    )
    irl_policy_sav = jnp.linspace(ACTIONS[0], ACTIONS[-1], RESOLUTION)[
        jnp.argmin(state_action_values, axis=1)
    ]
    evaluation_policy_cost, key = evaluate_cost_function_parallel(
        partial(eval_policy, optimal_policy=k_star),
        irl_cost,
        key,
        workers=WORKERS,
        start_states=start_states,
    )
    irl_policy_cost, key = evaluate_cost_function_parallel(
        partial(irl_policy, states=D_STATES, policy=irl_policy_sav),
        irl_cost,
        key,
        workers=WORKERS,
        start_states=start_states,
    )
    discont = GAMMA ** np.arange(evaluation_policy_cost.shape[1])
    evaluation_values = np.sum(evaluation_policy_cost * discont, axis=1)
    values_eval_policy["mean"].append(np.mean(evaluation_values))
    values_eval_policy["var"].append(np.var(evaluation_values))
    values_eval_policy["std"].append(np.std(evaluation_values))
    values_eval_policy["se"].append(
        np.std(evaluation_values) / np.sqrt(evaluation_values.shape[0])
    )
    optimal_values = np.sum(irl_policy_cost * discont, axis=1)
    values_optimal_policy["mean"].append(np.mean(optimal_values))
    values_optimal_policy["var"].append(np.var(optimal_values))
    values_optimal_policy["std"].append(np.std(optimal_values))
    values_optimal_policy["se"].append(
        np.std(optimal_values) / np.sqrt(optimal_values.shape[0])
    )


with open(OUT_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "N",
            "evalues_eval_mean",
            "evalues_eval_var",
            "evalues_eval_std",
            "evalues_eval_se",
            "evalues_opt_mean",
            "evalues_opt_var",
            "evalues_opt_std",
            "evalues_opt_se",
        ],
    )
    writer.writerows(
        zip(
            N_VALUES,
            values_eval_policy["mean"],
            values_eval_policy["var"],
            values_eval_policy["std"],
            values_eval_policy["se"],
            values_optimal_policy["mean"],
            values_optimal_policy["var"],
            values_optimal_policy["std"],
            values_optimal_policy["se"],
        )
    )

if SHOW_PLOT:
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.plot(N_VALUES, values_eval_policy["mean"], color="red", label=r"$\pi=\pi_{E}$")
    ax.fill_between(
        N_VALUES,
        np.array(values_eval_policy["mean"]) - np.array(values_eval_policy["std"]),
        np.array(values_eval_policy["mean"]) + np.array(values_eval_policy["std"]),
        alpha=0.2,
        color="red",
    )
    ax.plot(
        N_VALUES,
        values_optimal_policy["mean"],
        color="blue",
        label=r"$\pi=\arg\min_{\pi\in\Pi}V^{\pi}_{\tilde{c}_{N}}(\nu_{0})$",
    )
    ax.fill_between(
        N_VALUES,
        np.array(values_optimal_policy["mean"])
        - np.array(values_optimal_policy["std"]),
        np.array(values_optimal_policy["mean"])
        + np.array(values_optimal_policy["std"]),
        alpha=0.2,
        color="blue",
    )
    ax.legend()
    plt.title(r"Policy convergence")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$V^{\pi}_{\bar{\tilde{c}}_{N}}(\nu_{0})$")
    plt.show()

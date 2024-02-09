from __future__ import annotations

from functools import partial
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# Apple M chip
# import multiprocessing as mp
# mp.set_start_method("fork")

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from scs.LOGEnvironment1D import (
    ACTIONS,
    GAMMA,
    RESOLUTION,
    STATES,
    optimal_deterministic_policy,
)
from scs.irl_1d_IP_constraint_helper import (
    integral_expected_values,
    integrated_cost_basis_function,
    integrated_value_basis_function,
)
from scs.irl_1d_SIPn_constraint_helper import (
    generate_constraint_matrix_block_one,
    generate_samples_constraint_matrix,
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
    occupational_measure,
    write_to_file,
)

# Set parameters
################################################################################
DEGREE: int = 3
THETA: float = 50
N: int = 30
EPSILON: float = 0.1
DELTA: float = 0.01
WORKERS: int = 15
################################################################################

key: jnp.ndarray = jax.random.PRNGKey(0)

k_star: jnp.ndarray = optimal_deterministic_policy()
mu_k: jnp.ndarray
states: jnp.ndarray
actions: jnp.ndarray
mu_k_key, key = jax.random.split(key, 2)
mu_k, D_STATES, D_ACTIONS = occupational_measure(
    k_star, mu_k_key, RESOLUTION, workers=WORKERS
)

cost_weights: np.ndarray = np.ones(DEGREE**2)
cost_powers: np.ndarray = np.arange(DEGREE)

value_weights: np.ndarray = np.ones(DEGREE)
value_powers: np.ndarray = np.arange(DEGREE)

for i in range(1000):
    print(f"##############: {i}")

    sample_key, key = jax.random.split(key, 2)
    N_STATES_ACTIONS = jax.random.uniform(
        sample_key,
        minval=jnp.array([STATES[0], ACTIONS[0]]),
        maxval=jnp.array([STATES[-1], ACTIONS[-1]]),
        shape=(N, 2),
    )
    N_STATES = N_STATES_ACTIONS[:, 0].sort()

    # Construct the optimization problem
    (c := np.zeros(1 + 2 * cost_weights.shape[0] + 2 * value_weights.shape[0]))[0] = 1

    # Indices of the optimization variables
    indices_epsilon = np.arange(1)
    indices_c = np.arange(cost_weights.shape[0]) + 1
    indices_v = np.arange(value_weights.shape[0]) + 1 + cost_weights.shape[0]
    # Surrogate variables for L1 norm constraint
    indices_sc = (
        np.arange(cost_weights.shape[0])
        + 1
        + cost_weights.shape[0]
        + value_weights.shape[0]
    )
    indices_sv = (
        np.arange(value_weights.shape[0])
        + 1
        + 2 * cost_weights.shape[0]
        + value_weights.shape[0]
    )

    # Construct equality consteraits of SIPn
    b = integrated_cost_basis_function(cost_powers)
    d = (ACTIONS[-1] - ACTIONS[0]) * integrated_value_basis_function(value_powers)
    e = integral_expected_values(value_powers)
    A_eq = np.concatenate(
        [
            np.zeros(1),
            b,
            -d + GAMMA * e,
            np.zeros(cost_weights.shape[0] + value_weights.shape[0]),
        ]
    )[
        np.newaxis, :
    ]  # For the equality constraint
    b_eq = 1

    # Generate inequality constraints of SIPn
    matrix_block_one = generate_constraint_matrix_block_one(
        value_powers, cost_powers, D_STATES, DEGREE
    )
    constraint_1_inner_product = np.sum(
        matrix_block_one[..., 1:] * mu_k[..., np.newaxis], axis=(0, 1)
    )
    inequality_constraint_1 = np.concatenate(
        [np.array([-1]), constraint_1_inner_product]
    )
    inequality_constraint_2 = -1 * generate_samples_constraint_matrix(
        value_powers, cost_powers, N_STATES_ACTIONS, DEGREE
    )
    inequality_constraint_cl1_1 = np.zeros(
        1 + 2 * cost_weights.shape[0] + 2 * value_weights.shape[0]
    )
    inequality_constraint_cl1_2 = inequality_constraint_cl1_1.copy()
    inequality_constraint_sc = inequality_constraint_cl1_1.copy()
    inequality_constraint_vl1_1 = inequality_constraint_cl1_1.copy()
    inequality_constraint_vl1_2 = inequality_constraint_cl1_1.copy()
    inequality_constraint_sv = inequality_constraint_cl1_1.copy()
    # Block for the L1 norm constraint of the cost weights
    inequality_constraint_cl1_1[indices_c] = 1
    inequality_constraint_cl1_1[indices_sc] = -1
    inequality_constraint_cl1_2[indices_c] = -1
    inequality_constraint_cl1_2[indices_sc] = -1
    # Block for the L1 norm constraint of the value weights
    inequality_constraint_vl1_1[indices_v] = 1
    inequality_constraint_vl1_1[indices_sv] = -1
    inequality_constraint_vl1_2[indices_v] = -1
    inequality_constraint_vl1_2[indices_sv] = -1
    # Block for sum of surrogate variables <= THETA
    inequality_constraint_sc[indices_sc] = 1
    inequality_constraint_sv[indices_sv] = 1

    A_ub = np.concatenate(
        [
            inequality_constraint_1[np.newaxis, :],
            inequality_constraint_2.reshape(-1, 1 + DEGREE**2 + DEGREE),
        ]
    )
    A_ub = np.concatenate(
        [
            A_ub,
            np.zeros((A_ub.shape[0], cost_weights.shape[0] + value_weights.shape[0])),
        ],
        axis=1,
    )
    A_ub = np.concatenate(
        [
            A_ub,
            inequality_constraint_cl1_1[np.newaxis, :],
            inequality_constraint_cl1_2[np.newaxis, :],
            inequality_constraint_vl1_1[np.newaxis, :],
            inequality_constraint_vl1_2[np.newaxis, :],
            inequality_constraint_sc[np.newaxis, :],
            inequality_constraint_sv[np.newaxis, :],
        ],
        axis=0,
    )
    b_ub = np.concatenate([np.zeros(A_ub.shape[0] - 2), np.array([THETA, THETA])])

    # Solve the optimization problem
    result = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq)

    cost_weights_x = result.x[indices_c]
    value_weights_x = result.x[indices_v]

    # Obtain optimal policy of the learned cost function via state-action values
    irl_cost = partial(cost_function, weights=cost_weights_x, powers=cost_powers)
    state_action_values, error = value_iteration(
        partial(
            monomial_basis_function_state_actions,
            params=MLBFParams(cost_weights_x, cost_powers),
        ),
        max_iterations=100,
    )
    irl_policy_sav = jnp.linspace(ACTIONS[0], ACTIONS[-1], RESOLUTION)[
        jnp.argmin(state_action_values, axis=1)
    ]

    evaluation_policy_cost, key = evaluate_cost_function_parallel(
        partial(eval_policy, optimal_policy=k_star), irl_cost, key, workers=WORKERS
    )
    irl_policy_cost, key = evaluate_cost_function_parallel(
        partial(irl_policy, states=D_STATES, policy=irl_policy_sav),
        irl_cost,
        key,
        workers=WORKERS,
    )
    discount = GAMMA ** np.arange(evaluation_policy_cost.shape[1])

    mean_evaluation_policy_cost = np.mean(evaluation_policy_cost, axis=1)
    mean_irl_policy_cost = np.mean(irl_policy_cost, axis=1)

    evaluation_policy_value = np.sum(mean_evaluation_policy_cost * discount)
    irl_policy_value = np.sum(mean_irl_policy_cost * discount)

    threshold = ((2 - GAMMA) / (1 - GAMMA)) * result.x[0]  # For eps = 0

    print(
        f"True policy V: {evaluation_policy_value}, IRL policy V: {irl_policy_value},"
        f" delta: {evaluation_policy_value - irl_policy_value}, eps: {threshold}\n"
        f"Condition check vor eps=0: "
        f"{np.abs(evaluation_policy_value - irl_policy_value) <= threshold}"
    )

    print(result.x)

    write_to_file(f"data/experiment_2/params_{N}_2.txt", result.x)
    write_to_file(
        f"data/experiment_2/values_{N}_2.txt",
        evaluation_policy_value - irl_policy_value,
    )

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
from scs.irl_1d_SIPnk_constraint_helper import (
    generate_ub_constraint_block_one_mu,
    generate_ub_constraint_block_two,
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
)
from scs.utils import write_to_file

# Set parameters
################################################################################
DEGREE: int = 3
THETA: float = 50
N: int = 250
K = 2500
EPSILON: float = 0.1
DELTA: float = 0.01
WORKERS: int = 15
################################################################################


key: jnp.ndarray = jax.random.PRNGKey(0)

k_star: jnp.ndarray = optimal_deterministic_policy()
mu_k: jnp.ndarray
states: jnp.ndarray
actions: jnp.ndarray
mu_k, D_STATES, D_ACTIONS = occupational_measure(
    k_star, key, RESOLUTION, workers=WORKERS
)

cost_weights = np.ones(DEGREE**2)
cost_powers = np.arange(DEGREE)

value_weights = np.ones(DEGREE)
value_powers = np.arange(DEGREE)

c_n = cost_weights.shape[0]
v_n = value_weights.shape[0]

for i in range(1000):
    print(f"##############: {i}")

    sample_key, key = jax.random.split(key, 2)
    N_STATES_ACTIONS = jax.random.uniform(
        sample_key,
        minval=jnp.array([STATES[0], ACTIONS[0]]),
        maxval=jnp.array([STATES[-1], ACTIONS[-1]]),
        shape=(N, 2),
    )

    # Construct the optimization problem
    (c := np.zeros(1 + 2 * c_n + 2 * v_n))[0] = 1

    # Indices of the optimization variables
    indices_epsilon = np.arange(1)
    indices_c = np.arange(c_n) + 1
    indices_v = np.arange(v_n) + 1 + c_n
    # Surrogate variables for L1 norm constraint
    indices_sc = np.arange(c_n) + 1 + c_n + v_n
    indices_sv = np.arange(v_n) + 1 + 2 * c_n + v_n

    # Construct the equality constraint of SIPnk
    A_eq = np.zeros((2, 1 + 2 * c_n + 2 * v_n))
    A_eq[0, indices_c] = 1
    A_eq[1, indices_v] = 1
    b_eq = np.array([1, 1])

    # Construct the inequality constraints of SIPnk
    matrix_block_one, key = generate_ub_constraint_block_one_mu(
        N_STATES_ACTIONS, value_powers, cost_powers, mu_k, K, key, D_STATES, D_ACTIONS
    )
    inequality_constraint_1 = np.concatenate([np.array([-1]), matrix_block_one])
    inequality_constraint_2, key = generate_ub_constraint_block_two(
        value_powers,
        cost_powers,
        N,
        K,
        key,
    )
    inequality_constraint_2 *= -1
    inequality_constraint_simplex = np.zeros(
        (
            c_n + v_n,
            1 + 2 * c_n + 2 * v_n,
        )
    )
    inequality_constraint_simplex[:, np.concatenate([indices_c, indices_v])] = -np.eye(
        c_n + v_n
    )
    inequality_constraint_cl1_1 = np.zeros(1 + 2 * c_n + 2 * v_n)
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

    # Epsilon >= 0
    epsilon_constraint = np.zeros(1 + 2 * c_n + 2 * v_n)
    epsilon_constraint[indices_epsilon] = -1

    A_ub = np.concatenate(
        [
            inequality_constraint_1[np.newaxis, :],
            inequality_constraint_2.reshape(-1, 1 + DEGREE**2 + DEGREE),
        ]
    )
    A_ub = np.concatenate(
        [
            A_ub,
            np.zeros((A_ub.shape[0], c_n + v_n)),
        ],
        axis=1,
    )
    A_ub = np.concatenate(
        [
            epsilon_constraint[np.newaxis, :],
            A_ub,
            inequality_constraint_simplex,
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

    write_to_file(f"data/experiment_3/params_{N}_{K}.txt", result.x)
    write_to_file(
        f"data/experiment_3/values_{N}_{K}.txt",
        evaluation_policy_value - irl_policy_value,
    )

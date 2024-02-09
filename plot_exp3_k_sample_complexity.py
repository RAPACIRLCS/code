from __future__ import annotations

import csv
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
# Apple M chip
# import multiprocessing as mp
# mp.set_start_method("fork")

import matplotlib.pyplot as plt
import numpy as np

from scs.LOGEnvironment1D import GAMMA

# Set parameters
################################################################################
DEGREE: int = 3
THETA: float = 50
N: list[int] = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 500, 1000]
EPSILON: list[float] = [1, 0.5, 0.1]
DELTA: float = 0.01
WORKERS: int = 15
OUT_FILE: str = "data/experiment_3_k_sample_complexity.csv"
SHOW_PLOT: bool = True
################################################################################

cost_weights = np.ones(DEGREE**2)
ost_powers = np.arange(DEGREE)

value_weights = np.ones(DEGREE)

n_u = value_weights.shape[0]
n_c = cost_weights.shape[0]


def k_lower_bound(n_u: int, theta: float, eps: float, gamma: float, N: int) -> float:
    c = 1
    return 8 * c * n_u * theta**2 * np.log((4 * n_u * N) / gamma) / eps**2


k_bounds = [[k_lower_bound(n_u, THETA, eps, GAMMA, n) for n in N] for eps in EPSILON]


with open(OUT_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["N"] + [f"k_{eps}" for eps in EPSILON])
    for i, n in enumerate(N):
        writer.writerow([n] + [k_bounds[j][i] for j, _ in enumerate(EPSILON)])


if SHOW_PLOT:
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    for i, eps in enumerate(EPSILON):
        ax.plot(N, k_bounds[i], label=f"epsilon={eps}")
    ax.set_yscale("log")
    plt.legend()
    plt.title(r"Bound on $k$ as a function of $N$ and $\epsilon$")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$k$")
    plt.show()

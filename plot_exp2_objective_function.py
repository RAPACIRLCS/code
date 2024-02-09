from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from scs.utils import (
    parse_param_files,
    parse_value_files,
)

################################################################################
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
    500,
    1000,
]
OUT_FILE: str = "data/experiment_2_objective_function.csv"
SHOW_PLOT: bool = True
################################################################################


parameters = np.array(
    [parse_param_files(f"data/experiment_2/params_{n}.txt") for n in N_VALUES]
)
values = np.array(
    [parse_value_files(f"data/experiment_2/values_{n}.txt") for n in N_VALUES]
)

epsilons = np.mean(parameters[..., 0], axis=1)
epsilons_var = np.var(parameters[..., 0], axis=1)
epsilons_std = np.std(parameters[..., 0], axis=1)
epsilons_se = epsilons_std / np.sqrt(parameters[..., 0].shape[1])


with open(OUT_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["N", "epsilon", "epsilon_var", "epsilon_std", "epsilon_se"])
    for n, eps, eps_var, eps_std, eps_se in zip(
        N_VALUES, epsilons, epsilons_var, epsilons_std, epsilons_se
    ):
        writer.writerow([n, eps, eps_var, eps_std, eps_se])


if SHOW_PLOT:
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.plot(N_VALUES, epsilons, color="red")
    ax.fill_between(
        N_VALUES,
        epsilons - epsilons_std,
        epsilons + epsilons_std,
        color="red",
        alpha=0.2,
    )
    plt.title(r"Objective function of $SIP_{N}$")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$\tilde{\epsilon}_{N}$")
    plt.show()

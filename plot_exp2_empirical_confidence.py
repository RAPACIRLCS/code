from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from scs.irl_1d_eval import success_probabilities
from scs.utils import (
    parse_param_files,
    parse_value_files,
)

################################################################################
N_VALUES: list[int] = [
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
]
EPS: list[float] = [0, 1e-4, 1e-3, 1e-2]
OUT_FILE: str = "data/experiment_2_empirical_confidence.csv"
SHOW_PLOT: bool = True
################################################################################


parameters = np.array(
    [parse_param_files(f"data/experiment_2/params_{n}.txt") for n in N_VALUES]
)
values = np.array(
    [parse_value_files(f"data/experiment_2/values_{n}.txt") for n in N_VALUES]
)

epsilons = parameters[..., 0]

confidence = {}
for eps in EPS:
    confidence[eps] = success_probabilities(values, epsilons, eps)

with open(OUT_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["N"] + [f"epsilon_{eps}" for eps in EPS])
    for i, n in enumerate(N_VALUES):
        writer.writerow([n] + [confidence[eps][i] for eps in EPS])

if SHOW_PLOT:
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    for eps in EPS:
        ax.plot(N_VALUES, confidence[eps], label=f"epsilon={eps}")
    ax.plot(N_VALUES, [1] * len(N_VALUES), color="black", linestyle="--")
    ax.legend()
    plt.title(r"Empirical confidence of $SIP_{N}$")
    plt.xlabel(r"$N$")
    plt.ylabel(
        r"$\mathbb{P} \left( \tilde{c}_{N}\in\mathcal{C}^"
        r"{\tilde{\varepsilon}_{N}+\epsilon}(\pi_{E}) \right)$"
    )
    plt.show()

from __future__ import annotations

from collections import defaultdict
import csv

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from scs.irl_1d_eval import success_probabilities
from scs.utils import (
    parse_param_files,
    parse_value_files,
)

################################################################################
N_K_VALUES: list[tuple[int, int]] = [
    (25, 25),
    (250, 250),
    (2500, 2500),
    (25000, 25000),
]
EPS: np.ndarray = np.linspace(0, 1, 1000)
OUT_FILE: str = "data/experiment_3_empirical_confidence.csv"
SHOW_PLOT: bool = True
################################################################################


parameters = np.array(
    [parse_param_files(f"data/experiment_3/params_{n}_{k}.txt") for n, k in N_K_VALUES]
)
values = np.array(
    [parse_value_files(f"data/experiment_3/values_{n}_{k}.txt") for n, k in N_K_VALUES]
)

epsilons = parameters[..., 0]

confidence = defaultdict(list)
for eps in tqdm(EPS):
    probabilities = success_probabilities(values, epsilons, eps)
    for i, (n, k) in enumerate(N_K_VALUES):
        confidence[(n, k)].append(probabilities[i])


with open(OUT_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["epsilon"] + [f"confidence_{n}_{k}" for n, k in N_K_VALUES])
    for i, eps in enumerate(EPS):
        writer.writerow([eps] + [confidence[(n, k)][i] for (n, k) in N_K_VALUES])


if SHOW_PLOT:
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    for n, k in N_K_VALUES:
        ax.plot(EPS, confidence[(n, k)], label=f"N={n}, k={k}")
    ax.plot(EPS, [1] * len(EPS), color="black", linestyle="--")
    ax.legend()
    plt.title(r"Empirical confidence of $SIP_{N,m,n,k}$")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(
        r"$\mathbb{P} \left( \tilde{c}_{N,m,n,k}\in\mathcal{C}^"
        r"{\tilde{\varepsilon}_{N,m,n,k}+\epsilon}(\pi_{E}) \right)$"
    )
    plt.show()

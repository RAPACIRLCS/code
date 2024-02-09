from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult


def write_to_file(file_name: str, data: str) -> None:
    with open(file_name, "a") as file:
        file.write(str(data) + "\n")


def parse_value_files(filepath: str) -> np.ndarray:
    with open(filepath, "r") as file:
        data = file.readlines()
    return np.array([float(line.strip()) for line in data])


def parse_param_files(filepath: str) -> np.ndarray:
    arrays = []
    with open(filepath, "r") as file:
        lines = []
        inside_array = False
        for line in file:
            if "[" in line:
                inside_array = True
            if inside_array:
                lines.append(line)
            if "]" in line:
                inside_array = False
                concatenated_lines = (
                    "".join(lines).replace("[", "").replace("]", "").strip()
                )
                arrays.append(np.fromstring(concatenated_lines, sep=" "))
                lines = []
    return np.vstack(arrays)


def append_results_to_file(result: OptimizeResult, filename: str) -> None:
    with open(filename, "a") as file:
        file.write("#" * 30 + " Optimization Result " + "#" * 30 + "\n")
        file.write(f"message: {result.message}\n")
        file.write(f"success: {result.success}\n")
        file.write(f"status: {result.status}\n")
        file.write(f"fun: {result.fun}\n")
        file.write(f"x: {result.x}\n")
        file.write(f"nit: {result.nit}\n")
        file.write(f"slack: {result.slack}\n")
        file.write(f"con: {result.con}\n")
        file.write(f"lower: {result.lower}\n")
        file.write(f"upper: {result.upper}\n")
        file.write(f"ineqlin: {result.ineqlin}\n")
        file.write(f"eqlin: {result.eqlin}\n")
        file.write(f"crossover_nit: {result.crossover_nit}\n")
        file.write(f"mip_node_count: {result.mip_node_count}\n")
        file.write(f"mip_dual_bound: {result.mip_dual_bound}\n")
        file.write(f"mip_gap: {result.mip_gap}\n")
        file.write("#" * 78 + "\n\n")

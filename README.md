# Randomized algorithms and PAC bounds for inverse reinforcement learning in continuous spaces

## Overview
This repository contains the implementation and experimental setup for our research on randomized algorithms and Probably Approximately Correct (PAC) bounds in the context of inverse reinforcement learning within continuous spaces.
Detailed information and results can be found in our accompanying paper.

## Setup Instructions

### Environment Setup
The experiments were run on Unix/Linux and have not been tested on any Windows system.
To get started, create a `Python 3.10` environment and install the required dependencies listed in `requirements.txt`.
We utilize the JAX library for its efficient function vectorization capabilities.
Given the modest size of individual operations, the `JAX[cpu]` package should suffice, eliminating the need for a `CUDA` setup.
However, if you plan to run experiments on a GPU, please ensure adequate resources are available for the number of parallel processes you intend to execute.
If necessary, adjust the environment for CPU usage by uncommenting the following line at the beginning of the experiment scripts:

    os.environ["JAX_PLATFORM_NAME"] = "cpu"

on top of the experiment files.


### Running Experiments on Apple Silicon
For users running experiments on Apple M-chip devices, it is essential to uncomment the specific configuration at the top of the script files to ensure compatibility with the `multiprocessing` module. Additionally, configure the number of parallel processes by setting the `WORKERS` global variable in the `Set Parameters` section according to your requirements.

## Generate Plot Data

To reproduce the plots presented in our paper, execute the `experiment_X.py` scripts, where `X` corresponds to different problem settings:

- `experiment_1.py` for the IP setup,
- `experiment_2.py` for the SIP_N setup,
- `experiment_3.py` for the SIP_NmnK setup.

## Processing Data

After running the experiments, process the generated data using scripts named according to the plots in the paper, i.e., `plot_SUBCAPTION.py`. Here, `SUBCAPTION` should be replaced with the specific sub-caption or identifier used for the plot in the paper.
These scripts will parse the experiment data and save the results in a `.csv` file format as used to generate the plots in the paper using `tikz`.

By default, the parsing scripts also generate a preliminary plot using `matplotlib`.
This feature is intended for quick data inspection or for users who wish to process the data further without using `tikz`.

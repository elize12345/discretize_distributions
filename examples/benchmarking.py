import torch
import numpy as np
import pandas as pd
import time
import discretize_distributions as dd
from shelling_gmms import dbscan_shells, set_grid_locations, quantization_gmm_shells
from examples.Benchmark_tests.gmm_test_cases_dim2 import all_test_cases

results = []

torch.manual_seed(1)

for trial, test_case in enumerate(all_test_cases):
    locs = test_case["locs"]
    covariance_matrix = test_case["covariance_matrix"]
    probs = test_case["probs"]

    num_dims = locs.shape[1]
    num_mix_elems = locs.shape[0]

    gmm = dd.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(probs=probs),
        component_distribution=dd.MultivariateNormal(
            loc=locs,
            covariance_matrix=covariance_matrix * (1 / (np.sqrt(num_dims)))  # scaling
        )
    )

    start = time.time()
    disc_gmm = dd.discretization_generator(gmm, num_locs=100)
    locs_gmm = disc_gmm.locs
    sig_runtime = time.time() - start
    sig_w2 = disc_gmm.w2.item()
    sig_num_locs = len(disc_gmm.locs)

    start = time.time()
    shells, z = dbscan_shells(gmm)
    grid_locs = set_grid_locations(gmm, shells)
    probs_, locs_, w2_, grids = quantization_gmm_shells(
        gmm, shells, z, paddings=[0.01]*len(shells)
    )
    shell_runtime = time.time() - start
    shell_w2 = w2_
    shell_num_locs = len(locs_)
    nr_shells = len(shells)

    results.append({
        "trial": trial,
        "num_dims": num_dims,
        "num_mix_elems": num_mix_elems,
        "sig_runtime": sig_runtime,
        "sig_w2": sig_w2,
        "sig_num_locs": sig_num_locs,
        "shell_runtime": shell_runtime,
        "shell_w2": shell_w2,
        "shell_num_locs": shell_num_locs,
        "nr of shells": nr_shells,
    })

df = pd.DataFrame(results)
df.to_csv("Benchmark_tests/gmm_discretization_results_dim2.csv", index=False)

print("Experiment complete. Results saved to 'gmm_discretization_results.csv'.")


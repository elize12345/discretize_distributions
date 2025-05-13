import torch
import numpy as np
import pandas as pd
import time
import discretize_distributions as dd
from shelling_gmms import dbscan_shells, set_grid_locations, quantization_gmm_shells
from discretize_distributions.grid import Grid
import random

results = []

torch.manual_seed(1)

for trial in range(20):
    num_dims = random.randint(1, 6)  # we won't go higher than 6
    num_mix_elems = random.randint(10, 20)
    batch_size = torch.Size()

    covariance_diag = torch.exp(torch.randn(batch_size + (num_mix_elems, num_dims)))
    covariance_matrix = torch.diag_embed(covariance_diag)
    locs = torch.randn(batch_size + (num_mix_elems, num_dims))
    probs = torch.rand(batch_size + (num_mix_elems,))

    probs = probs / probs.sum(dim=-1, keepdim=True)

    gmm = dd.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(probs=probs),
        component_distribution=dd.MultivariateNormal(
            loc=locs,
            covariance_matrix=covariance_matrix * (1 / (np.sqrt(num_dims)))  # scaling
        )
    )

    start = time.time()
    disc_gmm = dd.discretization_generator(gmm, num_locs=100)
    sig_runtime = time.time() - start
    sig_w2 = disc_gmm.w2
    sig_num_locs = len(disc_gmm.locs)

    start = time.time()
    shells, z = dbscan_shells(gmm)
    grid_locs = set_grid_locations(gmm, shells)
    probs_, locs_, w2_, grids = quantization_gmm_shells(
        gmm, shells, z, paddings=[0.01]*len(shells)
    )
    cluster_runtime = time.time() - start
    cluster_w2 = w2_
    cluster_num_locs = len(locs_)

    results.append({
        "trial": trial,
        "num_dims": num_dims,
        "num_mix_elems": num_mix_elems,
        "sig_runtime": sig_runtime,
        "sig_w2": sig_w2,
        "sig_num_locs": sig_num_locs,
        "cluster_runtime": cluster_runtime,
        "cluster_w2": cluster_w2,
        "cluster_num_locs": cluster_num_locs
    })

df = pd.DataFrame(results)
df.to_csv("gmm_discretization_results.csv", index=False)

print("Experiment complete. Results saved to 'gmm_discretization_results.csv'.")


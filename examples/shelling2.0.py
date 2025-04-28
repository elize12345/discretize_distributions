import torch
import discretize_distributions as dd
from discretize_distributions.utils import calculate_w2_disc_uni_stand_normal
from discretize_distributions.discretize import GRID_CONFIGS, OPTIMAL_1D_GRIDS
from discretize_distributions.grid import Grid
from matplotlib import pyplot as plt, patches
from scipy.spatial import Voronoi, voronoi_plot_2d
from discretize_distributions.distributions import DiscretizedMixtureMultivariateNormal, \
    DiscretizedMixtureMultivariateNormalQuantization
import GMMWas
import numpy as np

if __name__ == "__main__":
    num_dims = 2
    num_mix_elems0 = 1
    batch_size = torch.Size()
    torch.manual_seed(0)

    # covariance_matrix = GMMWas.tensors.generate_pd_mat(batch_size + (num_mix_elems0, num_dims, num_dims))
    # locs = torch.tensor([[1.0, 1.0]])
    # covariance_matrix = torch.tensor([[[0.2, 0.0000],
    #                                    [0.0000, 0.2]]])
    locs = torch.randn(batch_size + (num_mix_elems0, num_dims,))
    # only diagonal and pos def covariance matrices
    covariance_diag = torch.exp(torch.randn(batch_size + (num_mix_elems0, num_dims,)))
    covariance_matrix = torch.diag_embed(covariance_diag)

    probs = torch.rand(batch_size + (num_mix_elems0,))
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # creates gmm with only diagonal covariances
    norm = dd.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(probs=probs),
        component_distribution=dd.MultivariateNormal(
            loc=locs,
            covariance_matrix=covariance_matrix * (1 / (np.sqrt(num_dims)))  # scaling
        )
    )

    # Calculation of region R1 - take Voronoi partitions and grid for inside shell, then use normal method
    # 'grid_discretize_multi_norm_dist' to calculate W2 error - how to generate grid?

    # Calculation of region R2
    # term 1: expectation of truncated gaussian over whole space so (-inf,inf)
    # calculate_w2_disc_uni_stand_normal from utils can be used for the Expectation but setting bounds
    # li,ui to -inf and inf instead for WHOLE SPACE
    edges = torch.Tensor(-float('inf'), float('inf'))
    z = torch.tensor((torch.tensor(1.0), torch.tensor(2.0)))  # location for point z
    w2 = calculate_w2_disc_uni_stand_normal(locs=z)

    # term 2: integral over just boundary box of shell R1 - use 'grid_discretize_multi_norm_dist'
    # for one location, z and one region

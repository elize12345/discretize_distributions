import torch
import discretize_distributions as dd
from discretize_distributions.utils import calculate_w2_disc_uni_stand_normal
from discretize_distributions.discretize import GRID_CONFIGS, OPTIMAL_1D_GRIDS
from discretize_distributions.grid import Grid
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import GMMWas
import numpy as np
def generate_covariance_matrix(eigenvalues, eigenvectors, scale=1.0):
    # cov = V*lambda*VT
    cov_matrix = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    return cov_matrix * scale

if __name__ == "__main__":

    num_dims = 2
    num_mix_elems0 = 5
    batch_size = torch.Size()
    torch.manual_seed(0)
    ref_eigenvectors = torch.eye(num_dims)  # just identity matrix for now
    covariance_matrix_list = []

    for i in range(num_mix_elems0):
        eigenvalues = torch.abs(torch.randn(num_dims))
        cov_matrix = generate_covariance_matrix(eigenvalues, ref_eigenvectors)
        covariance_matrix_list.append(cov_matrix)

    covariance_matrix = torch.stack(covariance_matrix_list)
    loc = torch.randn(batch_size + (num_mix_elems0, num_dims,))
    probs = torch.rand(batch_size + (num_mix_elems0,))
    probs = probs / probs.sum(dim=-1, keepdim=True)

    gmm = dd.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(probs=probs),
        component_distribution=dd.MultivariateNormal(
            loc=loc,
            covariance_matrix=covariance_matrix * (1 / (np.sqrt(num_dims)))  # scaling
        )
    )

    # original method using signature operator
    disc_gmm = dd.discretization_generator(gmm, num_locs=100)
    locs_gmm = disc_gmm.locs
    grid_gmm = locs_gmm.detach().numpy()  # grid for gmm
    print(f'W2 error: {disc_gmm.w2}')

    # voronoi plot
    vor = Voronoi(grid_gmm)
    fig2 = voronoi_plot_2d(vor)
    plt.show()

    # compress GMM to gaussian
    gmm_compressed = dd.compress_mixture_multivariate_normal(gmm, n_max=1)  # gaussian
    # find new grid for gaussian
    disc_g = dd.discretization_generator(gmm_compressed, num_locs=100)  # discretize using optimal grid
    locs_g = disc_g.locs
    grid_g = locs_g.detach().numpy()  # grid for gaussian approx of gmm

    vor = Voronoi(grid_g)
    fig3 = voronoi_plot_2d(vor)
    plt.show()

    # new w2 error using this grid and the compressed gmm ? why not original dist?
    # for discretize_multi_norm_dist must be a gaussian and diag covariance
    locs, probs, w2 = dd.discretize_multi_norm_dist(gmm_compressed, grid=grid_g)
    print(f'W2 error: {disc_g.w2}')

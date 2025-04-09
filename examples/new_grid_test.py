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
    num_mix_elems0 = 3
    batch_size = torch.Size()
    torch.manual_seed(0)

    # only diagonal and pos def covariance matrices
    covariance_diag = torch.exp(torch.randn(batch_size + (num_mix_elems0, num_dims,)))
    covariance_matrix = torch.diag_embed(covariance_diag)

    loc = torch.randn(batch_size + (num_mix_elems0, num_dims,))
    probs = torch.rand(batch_size + (num_mix_elems0,))
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # creates gmm with only diagonal covariances
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
    fig1 = voronoi_plot_2d(vor)
    plt.show()

    # compress GMM to gaussian - but this collapse function does not respect diagonal covariances
    gmm_compressed = dd.compress_mixture_multivariate_normal(gmm, n_max=1)  # gaussian

    # instead, for now: average values for covariance and mean of 3 components approximated by 1 gaussian
    # cov matrix has shape [num_mix_elems0, num_dims, num_dims]
    cov_avg = (probs.view(-1, 1, 1) * covariance_matrix).sum(dim=0)  # [1, num_dims,num_dims]
    loc_avg = (probs.unsqueeze(1) * loc).sum(dim=0)  # [1, num_dims]
    g = dd.MultivariateNormal(loc=loc_avg, covariance_matrix=cov_avg * (1 / (np.sqrt(num_dims))))

    disc_g = dd.discretization_generator(gmm_compressed, num_locs=100)  # discretize using optimal grid
    locs_g = disc_g.locs.squeeze(0)  # (1,locs,dims) --> (locs,dims)
    # grid_g = locs_g.squeeze(0).detach().numpy()  # grid for gaussian approx of gmm
    grid_list = [locs_g[:, i] for i in range(locs_g.shape[1])]
    grid_g = Grid(locs_per_dim=grid_list)

    vor = Voronoi(locs_g.detach().numpy())
    fig2 = voronoi_plot_2d(vor)
    plt.show()

    # need gmm to be described by just 1 covariance matrix
    locs, probs, w2 = dd.discretize_mixture_multi_norm_dist(gmm, grid=grid_g)
    print(f'W2 error: {disc_g.w2}')





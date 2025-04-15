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

    locs = torch.randn(batch_size + (num_mix_elems0, num_dims,))
    # only diagonal and pos def covariance matrices
    covariance_diag = torch.exp(torch.randn(batch_size + (num_mix_elems0, num_dims,)))
    covariance_matrix = torch.diag_embed(covariance_diag)

    probs = torch.rand(batch_size + (num_mix_elems0,))
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # creates gmm with only diagonal covariances
    gmm = dd.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(probs=probs),
        component_distribution=dd.MultivariateNormal(
            loc=locs,
            covariance_matrix=covariance_matrix * (1 / (np.sqrt(num_dims)))  # scaling
        )
    )

    # original method using signature operator
    disc_gmm = dd.discretization_generator(gmm, num_locs=100)
    locs_gmm = disc_gmm.locs
    print(f'W2 error original Signature operation: {disc_gmm.w2}')
    print(f'Number of signature locations: {len(locs_gmm)}')

    # voronoi plot
    vor = Voronoi(locs_gmm.detach().numpy())
    fig1 = voronoi_plot_2d(vor)
    plt.title('Voronoi plot of Union of Signatures of GMM')
    # plt.savefig(f'figures/voronoi_gmm_{user_choice}.svg')
    plt.show()

    locs_gmm = disc_gmm.locs.squeeze(0)  # (1,locs,dims) --> (locs,dims)
    grid_list = [locs_gmm[:, i] for i in range(locs_gmm.shape[1])]
    grid = Grid(locs_per_dim=grid_list)
    boundary = [(locs_gmm[:, i].min(), locs_gmm[:, i].max()) for i in range(locs_gmm.shape[1])]
    shell_input = [(torch.tensor(1.4), torch.tensor(1.8)), (torch.tensor(-2.0), torch.tensor(1.0))]
    shell, core, outer = grid.shell(shell=shell_input)
    vor_shell = grid.voronoi_edge_shell(shell=shell_input)

    grid.plot_shell_2d(shell_input)

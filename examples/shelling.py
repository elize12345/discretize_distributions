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
    num_mix_elems0 = 2
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
    gauss = dd.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(probs=probs),
        component_distribution=dd.MultivariateNormal(
            loc=locs,
            covariance_matrix=covariance_matrix * (1 / (np.sqrt(num_dims)))  # scaling
        )
    )

    # original method using signature operator
    disc_g_sig = dd.discretization_generator(gauss, num_locs=100)  # normal signature
    locs_g_sig = disc_g_sig.locs.detach().numpy()
    probs_g_sig = disc_g_sig.probs.detach().numpy()
    print(f'W2 error original Signature operation: {disc_g_sig.w2}')
    # print(f'Number of signature locations: {len(locs_g_sig)}')

    locs_g = disc_g_sig.locs
    x_min, x_max = locs_g[:, 0].min(), locs_g[:, 0].max()
    y_min, y_max = locs_g[:, 1].min(), locs_g[:, 1].max()
    grid = Grid.from_shape((20, 10), ((x_min, x_max), (y_min, y_max)))

    disc_g_grid = DiscretizedMixtureMultivariateNormalQuantization(gauss, grid)
    locs_g_grid = disc_g_grid.locs.detach().numpy()
    probs_g_grid = disc_g_grid.probs.detach().numpy()
    print(f'W2 error grid operation: {disc_g_grid.w2.item()}')

    # scaling to compare prob mass across grid and signature
    global_min = min(probs_g_grid.min(), probs_g_sig.min())
    global_max = max(probs_g_grid.max(), probs_g_sig.max())
    s_grid = (probs_g_grid - global_min) / (global_max - global_min) * 100
    s_sig = (probs_g_sig - global_min) / (global_max - global_min) * 100

    plt.figure()
    plt.scatter(locs_g_grid[:, 0], locs_g_grid[:, 1], s=s_grid, label="Grid", color='red', alpha=0.6)
    plt.scatter(locs_g_sig[:, 0], locs_g_sig[:, 1], s=s_sig, label="Signature", color='blue', alpha=0.6)
    plt.legend()
    plt.title("Comparison of grid locations and signature locations")
    plt.show()

    # shelling - shell input must be floats
    shell_input = [(torch.tensor(1.0), torch.tensor(2.0)), (torch.tensor(-1.0), torch.tensor(1.0))]
    grid.plot_shell_2d(shell_input)

    # re-calc probs and w2
    probs_total = disc_g_grid.probs
    locs_total = disc_g_grid.locs

    shell_tensor, core_tensor, outer_tensor, core_grid = grid.shell(shell=shell_input)
    # grid not bounded yet ...
    disc_core_grid = DiscretizedMixtureMultivariateNormalQuantization(gauss, core_grid)  # redefine a grid

    # re-scale probs
    core_mask = (locs_total[:, None, :] == core_tensor[None, :, :]).all(dim=-1).any(dim=1)
    probs_core = disc_core_grid.probs
    core_mass = probs_total[core_mask].sum()
    probs_core_scaled = probs_core * core_mass

    # w2 ??
    w2_core = disc_core_grid.w2
    w2_core_scaled = w2_core * core_mass  # feels illegal

    print(f'Sum of prob in core {probs_core_scaled.sum()}')
    print(f'W2 error {w2_core_scaled.item()}')

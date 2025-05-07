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

    # covariance_matrix = GMMWas.tensors.generate_pd_mat(batch_size + (num_mix_elems, num_dims, num_dims))
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

    # original method using signature operator
    disc_norm_sig = dd.discretization_generator(norm, num_locs=100)  # normal signature
    locs_norm_sig = disc_norm_sig.locs.detach().numpy()
    probs_norm_sig = disc_norm_sig.probs.detach().numpy()
    print(f'W2 error original Signature operation: {disc_norm_sig.w2}')
    # print(f'Number of signature locations: {len(locs_g_sig)}')

    locs_g = disc_norm_sig.locs
    x_min, x_max = locs_g[:, 0].min(), locs_g[:, 0].max()
    y_min, y_max = locs_g[:, 1].min(), locs_g[:, 1].max()
    # z_min, z_max = locs_g[:, 2].min(), locs_g[:, 2].max()
    grid = Grid.from_shape((20, 10), ((x_min, x_max), (y_min, y_max)))

    disc_norm_grid = DiscretizedMixtureMultivariateNormalQuantization(norm, grid)
    locs_norm_grid = disc_norm_grid.locs.detach().numpy()
    probs_norm_grid = disc_norm_grid.probs.detach().numpy()
    print(f'W2 error grid operation: {disc_norm_grid.w2.item()}')

    # scaling to compare prob mass across grid and signature
    global_min = min(probs_norm_grid.min(), probs_norm_sig.min())
    global_max = max(probs_norm_grid.max(), probs_norm_sig.max())
    s_grid = (probs_norm_grid - global_min) / (global_max - global_min) * 100
    s_sig = (probs_norm_sig - global_min) / (global_max - global_min) * 100

    plt.figure()
    plt.scatter(locs_norm_grid[:, 0], locs_norm_grid[:, 1], s=s_grid, label="Grid", color='red', alpha=0.6)
    plt.scatter(locs_norm_sig[:, 0], locs_norm_sig[:, 1], s=s_sig, label="Signature", color='blue', alpha=0.6)
    plt.legend()
    plt.title("Comparison of grid locations and signature locations")
    plt.show()

    # shelling - boundary input must be floats
    shell_input = [(torch.tensor(1.0), torch.tensor(2.0)), (torch.tensor(-2.0), torch.tensor(1.0))]

    _, core_tensor, outer_tensor, core_grid, outer_grids, bounds = grid.shell(shell=shell_input)

    unified_grid = grid.unify_grid(core_grid=core_grid, outer_grids=outer_grids, shell=shell_input, bounds=bounds)
    disc_grid = DiscretizedMixtureMultivariateNormalQuantization(norm, unified_grid)
    print(f'Unified grid with new partitioning W2 Error {disc_grid.w2.item()}')  # same w2 as before!

    # plot of core and outer regions
    grid.plot_shell_2d(unified_grid=unified_grid, shell=shell_input)

    probs_grid = disc_grid.probs.detach().numpy()
    locs_grid = disc_grid.locs.detach().numpy()

    # comparing new partitioning prob mass
    global_min = min(probs_norm_grid.min(), probs_grid.min())
    global_max = max(probs_norm_grid.max(), probs_grid.max())
    s_grid = (probs_norm_grid - global_min) / (global_max - global_min) * 100
    s_grid2 = (probs_grid - global_min) / (global_max - global_min) * 100

    plt.figure()
    plt.scatter(locs_norm_grid[:, 0], locs_norm_grid[:, 1], s=s_grid, label="Original Partitioning", color='red', alpha=0.6)
    plt.scatter(locs_grid[:, 0], locs_grid[:, 1], s=s_grid2, label="Redefined Partitioning", color='blue', alpha=0.6)
    plt.legend()
    plt.title("Comparison of grid partitions")
    plt.show()

    # print(f'Total prob mass original grid: {probs_norm_grid.sum()}')
    # print(f'Total prob mass new grid: {probs_grid.sum()}')

    # testing outer region locations
    # locs_outer = [g.get_locs() for g in outer_grids if len(g) > 0]
    # locs_outer_tensor = torch.cat(locs_outer, dim=0) if locs_outer else torch.empty((0, num_dims))
    # print(f'Outer grids locs from outer-grids {len(locs_outer_tensor)}')
    # print(f"Actual outer tensor locs {len(outer_tensor)}")
    # print(f'No. of outer-grids {len(outer_grids)}')
    # set_from_grids = set(map(tuple, locs_outer_tensor.tolist()))
    # set_actual = set(map(tuple, outer_tensor.tolist()))
    # print("Are they equal?", set_from_grids == set_actual)

    # # re-calc probs and w2
    # probs_total = disc_norm_grid.probs
    # locs_total = disc_norm_grid.locs
    #
    # # re-scale probs
    # disc_core_grid = DiscretizedMixtureMultivariateNormalQuantization(norm, core_grid)  # redefine a grid
    # # core grid only has locations of core
    # core_mask = (locs_total[:, None, :] == core_tensor[None, :, :]).all(dim=-1).any(dim=1)
    # probs_core = disc_core_grid.probs
    # core_mass = probs_total[core_mask].sum()
    # probs_core_scaled = probs_core * core_mass
    #
    # w2_core = disc_core_grid.w2
    # print(f'Sum of prob in core {probs_core_scaled.sum()}')
    # print(f'W2 error {w2_core.item()}')
    #
    # # outer regions - same principal
    # w2_outer_list = []
    # probs_outer_list = []
    # for idx, grids in enumerate(outer_grids):
    #     outer_locs = grids.get_locs()
    #     disc_outer_grid = DiscretizedMixtureMultivariateNormalQuantization(norm, grids)
    #     outer_mask = (locs_total[:, None, :] == outer_locs[None, :, :]).all(dim=-1).any(dim=1)
    #     probs_outer = disc_outer_grid.probs
    #     outer_mass = probs_total[outer_mask].sum()
    #     probs_outer_scaled = (probs_outer * outer_mass)
    #     w2 = disc_outer_grid.w2
    #     print(f'Sum of prob in outer region number {idx}: {probs_core_scaled.sum()}')
    #     print(f'W2 error outer region number {idx}: {w2}')
    #     w2_outer_list.append(w2)
    #     probs_outer_list.append(probs_outer_scaled.sum())
    #
    # w2_outer = torch.stack(w2_outer_list).sum()
    # probs_outer_total = torch.stack(probs_outer_list).sum()
    # print(f'W2 error outer regions total {w2_outer.item()}')
    # print(f'Total probs outer regions {probs_outer_total}')

    # test - if bound spans whole space then the w2 error for the core should be equal!
    # shell_inf = [(torch.tensor(-float("inf")), torch.tensor(float("inf"))), (torch.tensor(-float("inf")), torch.tensor(float("inf")))]
    # grid.plot_shell_2d(shell_inf)
    #
    # # re-calc probs and w2
    # probs_total = disc_g_grid.probs
    # locs_total = disc_g_grid.locs
    #
    # _, core_tensor, outer_tensor, core_grid, outer_grids = grid.shell(shell=shell_inf)
    # # but this will evaluate all points in gauss into core_grid which we don't want?
    # disc_core_grid = DiscretizedMixtureMultivariateNormalQuantization(norm, core_grid)  # redefine a grid
    #
    # # re-scale probs
    # core_mask = (locs_total[:, None, :] == core_tensor[None, :, :]).all(dim=-1).any(dim=1)
    # probs_core = disc_core_grid.probs
    # core_mass = probs_total[core_mask].sum()
    # probs_core_scaled = probs_core * core_mass
    #
    # w2_core = disc_core_grid.w2
    #
    # print(f'Sum of prob in core {probs_core_scaled.sum()}')
    # print(f'W2 error {w2_core.item()}')


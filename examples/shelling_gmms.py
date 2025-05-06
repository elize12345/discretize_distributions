import torch
import discretize_distributions as dd
from discretize_distributions.utils import calculate_w2_disc_uni_stand_normal
from discretize_distributions.discretize import GRID_CONFIGS, OPTIMAL_1D_GRIDS, w2_multi_norm_dist_for_set_locations
from discretize_distributions.grid import Grid
from matplotlib import pyplot as plt, patches, cm
from scipy.spatial import Voronoi, voronoi_plot_2d
from discretize_distributions.distributions import DiscretizedMixtureMultivariateNormal, \
    DiscretizedMixtureMultivariateNormalQuantization, DiscretizedMixtureMultivariateNormalQuantizationShell
import GMMWas
import numpy as np

def quantization_gmm_shells(gmm, shell_inputs, z, resolutions, paddings):
    """
    Compute the quantization of a GMM for NON-overlapping grids
    Args:
        gmm:
        shell_inputs:
        resolutions:
        paddings:

    Returns:
        locs:
        probs:
        w2:

    """

    # locs_p = gmm.component_distribution.loc  # shape: [num_components, dim]
    # z = locs_p[2, :]  # arbitrary component location

    total_w2 = 0.0
    all_locs = []
    all_probs = []
    all_R1_grids = []
    all_z_probs = torch.zeros(1)

    for shell_input, shape, pad in zip(shell_inputs, resolutions, paddings):
        R1_grid = Grid.shell(shell_input, shape, pad)
        locs_R1 = R1_grid.get_locs()

        R1_inner = Grid(locs_per_dim=[z[0].unsqueeze(0), z[1].unsqueeze(0)], bounds=shell_input)
        R1_outer = Grid(locs_per_dim=[z[0].unsqueeze(0), z[1].unsqueeze(0)])

        disc_R1 = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_grid)
        disc_R1_inner = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_inner)
        disc_R1_outer = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_outer)

        w2_R1 = disc_R1.w2.item()
        w2_R1_inner = disc_R1_inner.w2.item()
        w2_R1_outer = disc_R1_outer.w2.item()

        w2_shell = w2_R1 + (w2_R1_outer - w2_R1_inner)
        total_w2 += w2_shell

        print(f"Shell bounds: {shell_input}")
        print(f"  - W2 error: {w2_shell}")

        z_probs_R1 = disc_R1.z_probs
        probs_R1 = disc_R1.probs
        print(f'Prob mass of z: {z_probs_R1.item()}')
        print(f'Prob mass of grid: {probs_R1.sum()}')

        # normalize wrt z mass
        z_mass = z_probs_R1.item()
        mass_scale = (1 - z_mass)
        probs_R1 = probs_R1 * mass_scale
        print(f'Prob mass of grid normalized by sum of grid and z: {probs_R1.sum().item()}')

        all_locs.append(locs_R1)
        all_probs.append(probs_R1)
        all_z_probs += z_probs_R1
        all_R1_grids.append(R1_grid)

    all_locs = torch.cat(all_locs, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    locs_all = torch.cat([all_locs, z.unsqueeze(0)], dim=0)
    probs_all = torch.cat([all_probs, all_z_probs], dim=0)

    probs_all = probs_all / probs_all.sum(dim=-1, keepdim=True)

    return probs_all, locs_all, total_w2, all_R1_grids

if __name__ == "__main__":
    num_dims = 2
    num_mix_elems0 = 3
    batch_size = torch.Size()
    torch.manual_seed(1)

    locs = torch.randn(batch_size + (num_mix_elems0, num_dims,))
    # only diagonal and pos def covariance matrices
    covariance_diag = torch.exp(torch.randn(batch_size + (num_mix_elems0, num_dims,)))
    covariance_matrix = torch.diag_embed(covariance_diag)

    probs = torch.rand(batch_size + (num_mix_elems0,))
    probs = probs / probs.sum(dim=-1, keepdim=True)

    gmm = dd.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(probs=probs),
        component_distribution=dd.MultivariateNormal(
            loc=locs,
            covariance_matrix=covariance_matrix * (1 / (np.sqrt(num_dims)))  # scaling
        )
    )

    shell_input = [(torch.tensor(-1.0), torch.tensor(1.0)), (torch.tensor(-1.0), torch.tensor(1.0))]

    # applying to whole GMM instead component wise
    locs_p = gmm.component_distribution.loc  # [num_components, dim]
    w2 = torch.zeros(1)

    R1_grid = Grid.shell(shell_input, (10, 10))
    locs_R1 = R1_grid.get_locs()

    # component
    z = locs_p[2, :]  # can be any arbitrary location
    # z = torch.tensor([2.0, 2.0])  # how does this not change W2 error??

    # calc w2 for R1(k)
    disc_R1 = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_grid)
    probs_R1, w2_R1, z_probs_R1 = disc_R1.probs, disc_R1.w2.item(), disc_R1.z_probs   # probs already weighted by components to add to 1 ...
    print(f"W2 inside R1 grid: {w2_R1}")

    # calc w2 for R^n with z - should just be same as R1_inner with unbounded region
    # w2_Rn = w2_multi_norm_dist_for_set_locations(norm=component_p, signature_locs=z)
    R1_outer = Grid(locs_per_dim=[z[0].unsqueeze(0), z[1].unsqueeze(0)])
    disc_R1_outer = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_outer)
    w2_R1_outer = disc_R1_outer.w2.item()
    print(f'W2 R1 outer: {w2_R1_outer}')

    # calc w2 for R1 with z
    R1_inner = Grid(locs_per_dim=[z[0].unsqueeze(0), z[1].unsqueeze(0)], bounds=shell_input)
    disc_R1_inner = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_inner)
    w2_R1_inner = disc_R1_inner.w2.item()
    print(f'W2 R1 inner: {w2_R1_inner}')

    # total w2 per component
    w2_p = w2_R1 + (w2_R1_outer - w2_R1_inner)
    print(f'W2 error for whole GMM: {w2_p}')

    # z location and mass
    print(f'prob R1 sum: {probs_R1.sum()}')
    print(f'prob mass of z: R1-1 = {z_probs_R1}, at location: {z}')

    grid_mass = probs_R1.sum().item()
    z_mass = z_probs_R1.item()
    mass_scale = (1 - z_mass)  # percentage mass left over in grid
    probs_R1 = probs_R1 * mass_scale
    print(f'Prob mass of grid normalized by sum of grid and z: {probs_R1.sum().item()}')

    locs_ = torch.cat([locs_R1, z.unsqueeze(0)], dim=0)
    probs_ = torch.cat([probs_R1, z_probs_R1], dim=0)  # need to normalize ?
    # probs_ = probs_ / probs_.sum(dim=-1, keepdim=True)
    print(f'Total probs: {probs_.sum()}')

    s = (probs_ - probs_.min()) / (probs_.max() - probs_.min()) * 100

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.scatter(locs_[:, 0], locs_[:, 1], label='Locs', s=s, color='red', alpha=0.6)
    core_lower_vertices_per_dim = R1_grid.lower_vertices_per_dim
    core_upper_vertices_per_dim = R1_grid.upper_vertices_per_dim
    for i in range(len(core_lower_vertices_per_dim[0])):
        for j in range(len(core_lower_vertices_per_dim[1])):
            x0 = core_lower_vertices_per_dim[0][i].item()
            x1 = core_upper_vertices_per_dim[0][i].item()
            y0 = core_lower_vertices_per_dim[1][j].item()
            y1 = core_upper_vertices_per_dim[1][j].item()
            rect = patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                edgecolor='blue',
                facecolor='none',
                linewidth=1.5,
                linestyle='-'
            )
            ax.add_patch(rect)
    plt.legend()
    plt.show()

    # # multiple shells on one gmm
    # shell_inputs = [
    #     [(torch.tensor(-1.0), torch.tensor(-0.6)), (torch.tensor(-1.0), torch.tensor(-0.6))],
    #     [(torch.tensor(-0.5), torch.tensor(1.0)), (torch.tensor(-0.5), torch.tensor(1.0))],
    #     [(torch.tensor(-1.5), torch.tensor(-1.1)), (torch.tensor(-1.5), torch.tensor(-1.1))]
    # ]
    # resolutions = [(3, 3), (10, 10), (4, 4)]
    # padding = [0.05, 0.1, 0.05]
    #
    # locs_p = gmm.component_distribution.loc  # [num_components, dim]
    # z = locs_p[2, :]  # Arbitrary location, or loop over components
    #
    # total_w2 = 0.0
    # all_locs = []
    # all_probs = []
    # all_R1_grids = []
    # all_z_probs = torch.zeros(1)
    #
    # for shell_input, shape, padding in zip(shell_inputs, resolutions, padding):
    #     R1_grid = Grid.shell(shell_input, shape, padding)
    #     locs_R1 = R1_grid.get_locs()
    #
    #     R1_inner = Grid(locs_per_dim=[z[0].unsqueeze(0), z[1].unsqueeze(0)], bounds=shell_input)
    #     R1_outer = Grid(locs_per_dim=[z[0].unsqueeze(0), z[1].unsqueeze(0)])
    #
    #     disc_R1 = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_grid)
    #     disc_R1_inner = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_inner)
    #     disc_R1_outer = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_outer)
    #
    #     w2_R1 = disc_R1.w2.item()
    #     w2_R1_inner = disc_R1_inner.w2.item()
    #     w2_R1_outer = disc_R1_outer.w2.item()
    #
    #     w2_shell = w2_R1 + (w2_R1_outer - w2_R1_inner)
    #     total_w2 += w2_shell
    #
    #     print(f"Shell bounds: {shell_input}")
    #     print(f"  - W2 error: {w2_shell}")
    #
    #     z_probs_R1 = disc_R1.z_probs
    #     probs_R1 = disc_R1.probs
    #     print(f'Prob mass of z: {z_probs_R1.item()}')
    #     print(f'Prob mass of grid: {probs_R1.sum()}')
    #
    #     # normalize wrt z mass
    #     grid_mass = probs_R1.sum().item()
    #     z_mass = z_probs_R1.item()  # already relative to total mass of 1
    #     mass_scale = (1 - z_mass)   # percentage mass left over in grid
    #     probs_R1 = probs_R1 * mass_scale
    #     print(f'Prob mass of grid normalized by sum of grid and z: {probs_R1.sum().item()}')
    #
    #     all_locs.append(locs_R1)
    #     all_probs.append(probs_R1)
    #     all_z_probs += z_probs_R1
    #     all_R1_grids.append(R1_grid)
    #
    # all_locs = torch.cat(all_locs, dim=0)
    # all_probs = torch.cat(all_probs, dim=0)
    # locs_all = torch.cat([all_locs, z.unsqueeze(0)], dim=0)
    # probs_all = torch.cat([all_probs, all_z_probs], dim=0)
    #
    # probs_all = probs_all / probs_all.sum(dim=-1, keepdim=True)  # normalize again wrt to all grids
    #
    # s = (probs_all - probs_all.min()) / (probs_all.max() - probs_all.min()) * 100
    # print(f"Total W2 error over all shells: {total_w2}")
    #
    # plt.figure(figsize=(8, 6))
    # ax = plt.gca()
    # # ax.scatter(locs_p[:, 0], locs_p[:, 1], label='GMM Component Means', s=50, color='red', alpha=0.6)  # means
    # ax.scatter(locs_all[:, 0], locs_all[:, 1], label='Locs', s=s, color='red', alpha=0.6)  # locs
    # cmap = plt.cm.get_cmap('tab10')
    # for idx, R1 in enumerate(all_R1_grids):
    #     core_lower_vertices_per_dim = R1.lower_vertices_per_dim
    #     core_upper_vertices_per_dim = R1.upper_vertices_per_dim
    #     for i in range(len(core_lower_vertices_per_dim[0])):
    #         for j in range(len(core_lower_vertices_per_dim[1])):
    #             x0 = core_lower_vertices_per_dim[0][i].item()
    #             x1 = core_upper_vertices_per_dim[0][i].item()
    #             y0 = core_lower_vertices_per_dim[1][j].item()
    #             y1 = core_upper_vertices_per_dim[1][j].item()
    #             rect = patches.Rectangle(
    #                 (x0, y0),
    #                 x1 - x0,
    #                 y1 - y0,
    #                 edgecolor='blue',
    #                 facecolor='none',
    #                 linewidth=1.5,
    #                 linestyle='-'
    #             )
    #             ax.add_patch(rect)
    # plt.title("Discretization Shells for GMM")
    # plt.legend()
    # plt.grid(True)
    # plt.axis("equal")
    # plt.show()

    # multiple shells on one gmm - function
    shell_inputs = [
        [(torch.tensor(-1.0), torch.tensor(-0.6)), (torch.tensor(-1.0), torch.tensor(-0.6))],
        [(torch.tensor(-0.5), torch.tensor(1.0)), (torch.tensor(-0.5), torch.tensor(1.0))],
        [(torch.tensor(-1.5), torch.tensor(-1.1)), (torch.tensor(-1.5), torch.tensor(-1.1))]
    ]
    resolutions = [(3, 3), (10, 10), (4, 4)]
    padding = [0.05, 0.1, 0.05]

    probs_all, locs_all, total_w2, all_R1_grids = quantization_gmm_shells(
        gmm=gmm,
        z=z,
        shell_inputs=shell_inputs,
        resolutions=resolutions,
        paddings=padding)

    s = (probs_all - probs_all.min()) / (probs_all.max() - probs_all.min()) * 100
    print(f"Total W2 error over all shells: {total_w2}")

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    # ax.scatter(locs_p[:, 0], locs_p[:, 1], label='GMM Component Means', s=50, color='red', alpha=0.6)  # means
    ax.scatter(locs_all[:, 0], locs_all[:, 1], label='Locs', s=s, color='red', alpha=0.6)  # locs
    cmap = plt.cm.get_cmap('tab10')
    for idx, R1 in enumerate(all_R1_grids):
        core_lower_vertices_per_dim = R1.lower_vertices_per_dim
        core_upper_vertices_per_dim = R1.upper_vertices_per_dim
        for i in range(len(core_lower_vertices_per_dim[0])):
            for j in range(len(core_lower_vertices_per_dim[1])):
                x0 = core_lower_vertices_per_dim[0][i].item()
                x1 = core_upper_vertices_per_dim[0][i].item()
                y0 = core_lower_vertices_per_dim[1][j].item()
                y1 = core_upper_vertices_per_dim[1][j].item()
                rect = patches.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    edgecolor='blue',
                    facecolor='none',
                    linewidth=1.5,
                    linestyle='-'
                )
                ax.add_patch(rect)
    plt.title("Discretization Shells for GMM")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
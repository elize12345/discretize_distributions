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
    locs = torch.tensor([[1.0, 1.0]])
    covariance_matrix = torch.tensor([[[0.2, 0.0000],
                                       [0.0000, 0.2]]])
    # locs = torch.randn(batch_size + (num_mix_elems0, num_dims,))
    # # only diagonal and pos def covariance matrices
    # covariance_diag = torch.exp(torch.randn(batch_size + (num_mix_elems0, num_dims,)))
    # covariance_matrix = torch.diag_embed(covariance_diag)

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
    # vor = Voronoi(locs_gmm.detach().numpy())
    # fig1 = voronoi_plot_2d(vor)
    # plt.title('Voronoi plot of Union of Signatures of GMM')
    # # plt.savefig(f'figures/voronoi_gmm_{user_choice}.svg')
    # plt.show()

    locs_gmm = disc_gmm.locs.squeeze(0)  # (1,locs,dims) --> (locs,dims)
    # grid_list = [locs_gmm[:, i] for i in range(locs_gmm.shape[1])]  # could be causing duplicates!
    grid_list = [torch.sort(torch.unique(locs_gmm[:, i]))[0] for i in range(locs_gmm.shape[1])]
    # grid = Grid(locs_per_dim=grid_list)

    grid = Grid.from_shape((10, 10), torch.tensor([[-1, 2], [-1, 2]]))
    shell_input = [(torch.tensor(0.5), torch.tensor(1.5)), (torch.tensor(0.5), torch.tensor(1.5))]
    shell, core, outer = grid.shell(shell=shell_input)
    vor_shell = grid.voronoi_edge_shell(shell=shell_input)

    # grid.plot_shell_2d(shell_input)

    # all mass is at bottom for some reason --> lower w2 !!
    new_loc = torch.tensor([[-0.8, -0.8]])

    print(f'No. of shell points: {len(shell)}, core: {len(core)}, outer: {len(outer)}')

    locs_core, probs_core, locs_outer, probs_outer, w2, w2_sinkhorn = grid.shell_discretize_multi_norm_dist(gmm, shell_input, new_loc)

    # analysis
    print(f'No. of core points: {len(probs_core)}, outer: {len(probs_outer)}')
    print(f'Prob mass inside core {probs_core.sum()}, prob mass outside core {probs_outer.sum()}, total {probs_core.sum()+probs_outer.sum()}')
    print(f'Added W2 error {w2_sinkhorn}')
    # print(f'Probs total {probs.sum()}')
    # combined = torch.cat([probs_core, probs_outer])
    # sorted_combined, _ = torch.sort(combined)
    # sorted_original, _ = torch.sort(probs)
    # assert torch.allclose(sorted_combined, sorted_original, atol=1e-6), "Mismatch in probability values!"

    global_min = min(probs_core.min(), probs_outer.min())
    global_max = max(probs_core.max(), probs_outer.max())
    s_core = (probs_core - global_min) / (global_max - global_min) * 100
    s_outer = (probs_outer - global_min) / (global_max - global_min) * 100

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    # ax.set_facecolor('black')

    ax.scatter(locs_core[:, 0], locs_core[:, 1], s=s_core, color='blue', label='Core', alpha=0.6)
    ax.scatter(locs_outer[:, 0], locs_outer[:, 1], s=s_outer, color='orange', label='Outer', alpha=0.6)
    ax.scatter(new_loc[:, 0], new_loc[:, 1], s=probs_outer.sum()*100, color='red', label='New loc', alpha=0.6)

    x_min, x_max = vor_shell[0][0].item(), vor_shell[0][1].item()
    y_min, y_max = vor_shell[1][0].item(), vor_shell[1][1].item()
    shell_box = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor='red',
        facecolor='none',
        linestyle='--',
        label='Shell boundary'
    )
    ax.add_patch(shell_box)

    ax.set_title('New loc')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)
    plt.show()

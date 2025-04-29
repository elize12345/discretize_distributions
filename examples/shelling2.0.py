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
    torch.manual_seed(1)

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
    # 'grid_discretize_multi_norm_dist' to calculate W2 error - how to generate grid? just choose uniform grid?
    disc_w2_optimal = dd.discretization_generator(norm, num_locs=100)  # w2 optimal grid
    disc_locs = disc_w2_optimal.locs
    disc_probs = disc_w2_optimal.probs.detach().numpy()

    grid_list = [torch.sort(torch.unique(disc_locs[:, i]))[0] for i in range(disc_locs.shape[1])]
    grid_w2_optimal = Grid(locs_per_dim=grid_list)

    # shelling - boundary input must be floats
    shell_input = [(torch.tensor(0.0), torch.tensor(1.5)), (torch.tensor(-1.5), torch.tensor(1.5))]
    # _, _, _, R1, _, _ = grid_w2_optimal.shell(shell=shell_input)  # grid of shell inside
    interval_tensor = torch.tensor([[a.item(), b.item()] for (a, b) in shell_input])  # grid points on shell a problem?
    R1 = Grid.from_shape((10, 10), interval_tensor, bounds=shell_input)
    disc_R1 = DiscretizedMixtureMultivariateNormalQuantization(norm, R1)
    disc_R1_locs = disc_R1.locs
    disc_R1_probs = disc_R1.probs.detach().numpy()  # mass in grid partitions

    global_min = min(disc_R1_probs.min(), disc_probs.min())
    global_max = max(disc_R1_probs.max(), disc_probs.max())
    s1 = (disc_probs - global_min) / (global_max - global_min) * 100
    s2 = (disc_R1_probs - global_min) / (global_max - global_min) * 100

    core_lower_vertices_per_dim, core_upper_vertices_per_dim = R1.lower_vertices_per_dim, R1.upper_vertices_per_dim
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.scatter(disc_locs.detach().numpy()[:, 0], disc_locs.detach().numpy()[:, 1], s=s1, color='red', label='Total', alpha=0.6)
    ax.scatter(disc_R1_locs.detach().numpy()[:, 0], disc_R1_locs.detach().numpy()[:, 1], s=s2, color='blue', label='R1', alpha=0.6)
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
                linewidth=1.2,
                linestyle='-'
            )
            ax.add_patch(rect)

    ax.set_title('Inner shell R1')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)
    plt.show()

    # Calculation of region R2
    # term 1: expectation of truncated gaussian over whole space so (-inf,inf)
    # calculate_w2_disc_uni_stand_normal from utils can be used for the Expectation but setting bounds
    # li,ui to -inf and inf instead for WHOLE SPACE
    # not sure how to get the expectation with unbounded region
    # mean = norm.mean  # [dim]
    # std = norm.stddev  # [dim]
    # # y = [torch.tensor(2.0).unsqueeze(0), torch.tensor(2.0).unsqueeze(0)]
    # y = mean
    # print(f'mean: {mean}')
    # scaled_locs_per_dim = [((y[dim] - mean[dim]) / std[dim]).unsqueeze(0) for dim in range(num_dims)]
    # w2_per_dim = [calculate_w2_disc_uni_stand_normal(dim_locs) for dim_locs in scaled_locs_per_dim]
    # w2 = torch.stack(w2_per_dim).pow(2).sum().sqrt()
    # print(f'W2 error whole space to z: {w2}')

    # Instead just use signature operator for num_locs=1 - not same as solving integral i believe ...
    disc_whole_space = dd.discretization_generator(norm, num_locs=1)
    z = disc_whole_space.locs  # placed z at mean - what if we place it somewhere else?
    print(f'z: {z}')

    # term 2: integral over just boundary box of shell R1 - use 'grid_discretize_multi_norm_dist'
    # for one location, z and one region
    R1_outer = Grid(locs_per_dim=[z[0, 0].unsqueeze(0), z[0, 1].unsqueeze(0)], bounds=shell_input)
    # arbitrary z location
    R1_lower_vertices_per_dim, R1_upper_vertices_per_dim = (R1_outer.lower_vertices_per_dim,
                                                            R1_outer.upper_vertices_per_dim)
    disc_R1_outer = DiscretizedMixtureMultivariateNormalQuantization(norm, R1_outer)
    disc_R1_outer_locs = disc_R1_outer.locs
    disc_R1_outer_probs = disc_R1_outer.probs.detach().numpy()  # mass in grid partitions
    print(f'Prob mass of z: {disc_R1_outer_probs.sum()}')

    # W2 calc
    print(f'W2 error for inner R1(k) locations: {disc_R1.w2.item()}')
    print(f'W2 error whole space to z: {disc_whole_space.w2.item()}')
    print(f'W2 error for all inside mass of R1 to location z: {disc_R1_outer.w2.item()}')
    print(f"Total W2: R1(k) + (expectation - R1) = {disc_R1.w2.item() + (disc_whole_space.w2.item()-disc_R1_outer.w2.item())}")

    global_min = min(disc_R1_outer_probs.min(), disc_probs.min())
    global_max = max(disc_R1_outer_probs.max(), disc_probs.max())
    s1 = (disc_probs - global_min) / (global_max - global_min) * 100
    s3 = (disc_R1_outer_probs - global_min) / (global_max - global_min) * 100

    plt.figure()
    ax = plt.gca()
    ax.scatter(disc_locs.detach().numpy()[:, 0], disc_locs.detach().numpy()[:, 1], s=s1, color='red', label='Total',
               alpha=0.6)
    ax.scatter(disc_R1_outer_locs.detach().numpy()[:, 0], disc_R1_outer_locs.detach().numpy()[:, 1],
               s=s3, color='blue', label='z',
               alpha=0.6)
    x0 = R1_lower_vertices_per_dim[0].item()
    x1 = R1_upper_vertices_per_dim[0].item()
    y0 = R1_lower_vertices_per_dim[1].item()
    y1 = R1_upper_vertices_per_dim[1].item()
    rect = patches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        edgecolor='blue',
        facecolor='none',
        linewidth=1.2,
        linestyle='-'
    )
    ax.add_patch(rect)
    ax.set_title('Outer shell R1 with one location z')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)
    plt.show()

    # redefine prob mass relative to all grids
    global_min = min(disc_R1_probs.min(), disc_probs.min(), disc_R1_outer_probs.min())
    global_max = max(disc_R1_probs.max(), disc_probs.max(), disc_R1_outer_probs.max())
    s1 = (disc_probs - global_min) / (global_max - global_min) * 100
    s2 = (disc_R1_probs - global_min) / (global_max - global_min) * 100
    s3 = (disc_R1_outer_probs - global_min) / (global_max - global_min) * 100

    plt.figure()
    ax = plt.gca()
    ax.scatter(disc_locs.detach().numpy()[:, 0], disc_locs.detach().numpy()[:, 1], s=s1, color='red', label='Total',
               alpha=0.6)
    ax.scatter(disc_R1_outer_locs.detach().numpy()[:, 0], disc_R1_outer_locs.detach().numpy()[:, 1],
               s=s3, color='blue', label='z',
               alpha=0.6)
    ax.scatter(disc_R1_locs.detach().numpy()[:, 0], disc_R1_locs.detach().numpy()[:, 1], s=s2, color='orange', label='R1', alpha=0.6)
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
                linewidth=1.2,
                linestyle='-'
            )
            ax.add_patch(rect)

    x0 = R1_lower_vertices_per_dim[0].item()
    x1 = R1_upper_vertices_per_dim[0].item()
    y0 = R1_lower_vertices_per_dim[1].item()
    y1 = R1_upper_vertices_per_dim[1].item()
    rect = patches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        edgecolor='blue',
        facecolor='none',
        linewidth=1.2,
        linestyle='-'
    )
    ax.add_patch(rect)
    ax.set_title('Shelling R1 inner and outer')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True)
    plt.show()
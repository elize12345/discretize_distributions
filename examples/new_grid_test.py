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


def generate_covariance_matrix(eigenvalues, eigenvectors, scale=1.0):
    # cov = V*lambda*VT
    cov_matrix = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    return cov_matrix * scale


if __name__ == "__main__":

    num_dims = 2
    num_mix_elems0 = 2
    batch_size = torch.Size()
    torch.manual_seed(0)

    user_choice = 'overlap'
    # input(
    #     "Choose GMM mode: type 'spread' for spread apart or 'overlap' for overlapping components: ").strip().lower()
    if user_choice == 'spread':
        locs = torch.tensor([[0.0, 0.0], [1.5, 1.5]])
        covariance_matrix = torch.tensor([[[0.5, 0.0000],
                                           [0.0000, 0.5]],
                                          [[0.2, 0.0000],
                                           [0.0000, 0.2]]])
    elif user_choice == 'overlap':
        locs = torch.tensor([[1.0, 1.0], [1.3, 1.3]])
        covariance_matrix = torch.tensor([[[0.3, 0.0000],
                                           [0.0000, 0.3]],
                                          [[0.5, 0.0000],
                                           [0.0000, 0.5]]])
    else:
        raise ValueError("Invalid choice. Please type 'spread' or 'overlap'.")

    # locs = torch.randn(batch_size + (num_mix_elems0, num_dims,))
    # # only diagonal and pos def covariance matrices
    # covariance_diag = torch.exp(torch.randn(batch_size + (num_mix_elems0, num_dims,)))
    # covariance_matrix = torch.diag_embed(covariance_diag)
    #
    # probs = torch.rand(batch_size + (num_mix_elems0,))
    # probs = probs / probs.sum(dim=-1, keepdim=True)
    probs = torch.tensor([0.3, 0.7])
    # print(probs)
    # creates gmm with only diagonal covariances
    gmm = dd.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(probs=probs),
        component_distribution=dd.MultivariateNormal(
            loc=locs,
            covariance_matrix=covariance_matrix*(1 / (np.sqrt(num_dims)))  # scaling
        )
    )

    # original method using signature operator
    disc_gmm = dd.discretization_generator(gmm, num_locs=100)
    locs_gmm = disc_gmm.locs
    probs_gmm = disc_gmm.probs.detach().numpy()
    # grid_gmm = locs_gmm.detach().numpy()  # grid for gmm
    print(f'W2 error original Signature operation: {disc_gmm.w2}')
    print(f'Number of signature locations: {len(locs_gmm)}')

    # scaling to compare prob mass across grid and signature
    s_gmm = (probs_gmm - probs_gmm.min()) / (probs_gmm.max() - probs_gmm.min()) * 100

    # union of grid locations from both signatures
    x_min, x_max = locs_gmm[:, 0].min(), locs_gmm[:, 0].max()
    y_min, y_max = locs_gmm[:, 1].min(), locs_gmm[:, 1].max()
    grid_union = Grid.from_shape((20, 10), ((x_min, x_max), (y_min, y_max)))

    # approximating signature grids by just one
    mean = (gmm.component_distribution.mean[0] + gmm.component_distribution.mean[1]) / 2
    var = (gmm.component_distribution.variance[0] + gmm.component_distribution.variance[1]) / 2
    avg_cov = torch.diag_embed(var)
    avg_dist = dd.MultivariateNormal(loc=mean, covariance_matrix=avg_cov)
    disc_avg_dist = dd.discretization_generator(avg_dist, num_locs=100)
    locs_avg = disc_avg_dist.locs.squeeze(0)  # (1,locs,dims) --> (locs,dims)
    grid_list = [torch.sort(torch.unique(locs_avg[:, i]))[0] for i in range(locs_avg.shape[1])]
    grid_avg = Grid(locs_per_dim=grid_list)

    user_choice2 = 'grid_union'
    if user_choice2 == 'grid_average':
        grid = grid_avg
    elif user_choice2 == 'grid_union':
        grid = grid_union

    print(f'{user_choice} + {user_choice2}')

    q = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=grid)
    w2 = q.w2
    probs_grid = q.probs.detach().numpy()
    locs_grid = q.locs.detach().numpy()
    print(f'W2 error for Quantization: {w2.item()}')
    print(f'Number of locations: {len(locs_grid)}')

    s_grid = (probs_grid - probs_grid.min()) / (probs_grid.max() - probs_grid.min()) * 100

    global_min = min(probs_grid.min(), probs_gmm.min())
    global_max = max(probs_grid.max(), probs_gmm.max())

    s_grid2 = (probs_grid - global_min) / (global_max - global_min) * 100
    s_gmm2 = (probs_gmm - global_min) / (global_max - global_min) * 100

    # Get axis limits
    x_min, x_max = locs_gmm[:, 0].min().item(), locs_gmm[:, 0].max().item()
    y_min, y_max = locs_gmm[:, 1].min().item(), locs_gmm[:, 1].max().item()

    # Optional padding
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding

    # Compute bounding box for locs_grid
    x_grid_min, x_grid_max = locs_grid[:, 0].min(), locs_grid[:, 0].max()
    y_grid_min, y_grid_max = locs_grid[:, 1].min(), locs_grid[:, 1].max()
    width = x_grid_max - x_grid_min
    height = y_grid_max - y_grid_min

    # Plot 1
    plt.figure()
    plt.scatter(locs_gmm[:, 0], locs_gmm[:, 1], s=s_gmm, label="Locations", color='blue', alpha=0.6)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    # plt.legend()
    # plt.title("Union of optimal locations wrt W2 error")
    # plt.savefig(f"figures/only_w2_optimal_locations_{user_choice}_{user_choice2}.svg")
    plt.show()

    plt.figure()
    plt.scatter(locs_grid[:, 0], locs_grid[:, 1], s=1, label="Possible grid locations", color='black')
    plt.scatter(locs_grid[:, 0], locs_grid[:, 1], s=s_grid, label="Weighted grid locations", color='red', alpha=0.6)

    # Add bounding box
    rect = patches.Rectangle((x_grid_min, y_grid_min), width, height, linewidth=1, edgecolor='black',
                             facecolor='none', label='Grid bounding box')
    plt.gca().add_patch(rect)

    # Set consistent axis limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.legend()
    # plt.title("Unified grid with bounding box")
    # plt.savefig(f"figures/only_grid_locations_{user_choice}_{user_choice2}.svg")
    plt.show()

    # Plot 3
    plt.figure()
    plt.scatter(locs_grid[:, 0], locs_grid[:, 1], s=1, label="Possible grid locations", color='black')
    plt.scatter(locs_grid[:, 0], locs_grid[:, 1], s=s_grid2, label="Weighted grid locations", color='red', alpha=0.6)
    plt.scatter(locs_gmm[:, 0], locs_gmm[:, 1], s=s_gmm2, label="Optimal-W2 locations", color='blue', alpha=0.6)

    # Add bounding box
    rect = patches.Rectangle((x_grid_min, y_grid_min), width, height, linewidth=1, edgecolor='black',
                             facecolor='none', label='Grid bounding box')
    plt.gca().add_patch(rect)

    # Set consistent axis limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.legend()
    # plt.title("Comparison with bounding box")
    # plt.savefig(f"figures/all_locations_{user_choice}_{user_choice2}.svg")
    plt.show()


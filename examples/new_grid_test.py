import torch
import discretize_distributions as dd
from discretize_distributions.utils import calculate_w2_disc_uni_stand_normal
from discretize_distributions.discretize import GRID_CONFIGS, OPTIMAL_1D_GRIDS
from discretize_distributions.grid import Grid
from matplotlib import pyplot as plt
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
        locs = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        covariance_matrix = torch.tensor([[[0.2, 0.0000],
                                           [0.0000, 0.2]],
                                          [[0.1, 0.0000],
                                           [0.0000, 0.1]]])
    elif user_choice == 'overlap':
        locs = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
        covariance_matrix = torch.tensor([[[0.5, 0.0000],
                                           [0.0000, 0.5]],
                                          [[0.5, 0.0000],
                                           [0.0000, 0.5]]])
    else:
        raise ValueError("Invalid choice. Please type 'spread' or 'overlap'.")

    # covariance_matrix = GMMWas.tensors.generate_pd_mat(batch_size + (num_mix_elems0, num_dims, num_dims))
    # locs = torch.randn(batch_size + (num_mix_elems0, num_dims,))
    # only diagonal and pos def covariance matrices
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
    grid_gmm = locs_gmm.detach().numpy()  # grid for gmm
    print(f'W2 error original Signature operation: {disc_gmm.w2}')
    print(f'Number of signature locations: {len(locs_gmm)}')

    # voronoi plot
    vor = Voronoi(grid_gmm)
    fig1 = voronoi_plot_2d(vor)
    plt.title('Voronoi plot of Union of Signatures of GMM')
    # plt.savefig(f'figures/voronoi_gmm_{user_choice}.svg')
    plt.show()

    # take covariance of one of the components, choose one with higher weight
    best_idx = probs.argmax(dim=-1)
    # print(f'Chosen Gaussian component number {best_idx}, prob value of {probs[best_idx]}')
    gaussian = dd.MultivariateNormal(loc=locs[best_idx], covariance_matrix=covariance_matrix[best_idx])
    # using optimal grid from signature operator
    disc_g = dd.discretization_generator(gaussian, num_locs=100)
    locs_g = disc_g.locs.squeeze(0)  # (1,locs,dims) --> (locs,dims)
    grid_list = [locs_g[:, i] for i in range(locs_g.shape[1])]
    # grid for gaussian approximation of GMM
    grid_g = Grid(locs_per_dim=grid_list)

    # uniform grid
    grid1 = Grid.from_shape((10, 10), torch.tensor([[-1, 2], [-1, 2]]))

    # uniform grid based on density of space of GMMs
    std = torch.sqrt(covariance_matrix.diagonal(dim1=-2, dim2=-1) * (1 / (np.sqrt(num_dims))))  # spread
    lower_bounds = locs - 2 * std
    upper_bounds = locs + 2 * std
    overall_lower = lower_bounds.min(dim=0).values
    overall_upper = upper_bounds.max(dim=0).values
    interval_per_dim = torch.stack([overall_lower, overall_upper], dim=1)  # [num_dims, 2]
    grid_uniform = Grid.from_shape((10, 10), interval_per_dim)

    # Voronoi plot of grid
    grid = grid_uniform
    vor = Voronoi(grid.get_locs())
    fig2 = voronoi_plot_2d(vor)
    plt.title('Voronoi plot of grid used in Quantization operator')
    # plt.savefig(f'figures/voronoi_uniform_grid_{user_choice}.svg')
    plt.show()

    q = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=grid)
    w2 = q.w2
    print(f'W2 error for Quantization: {w2.item()}')
    print(f'Number of locations: {len(grid.get_locs())}')

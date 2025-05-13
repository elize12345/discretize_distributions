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
from scipy.spatial import KDTree
import GMMWas
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

def shelling(gmm, grid_type="w2-gaussian-optimal", threshold=2, num_locs=10):
    """
    Combined logic
    Args:
        gmm:
        grid_type:
        threshold:
        num_locs:

    Returns: probs_all, locs_all, total_w2, all_R1_grids from quantization_gmm_shells

    """
    if grid_type == "w2-gaussian-optimal":
        shells, grid_locations_per_shell, z = generate_non_overlapping_shells(
            gmm=gmm, threshold=threshold, grid_type=grid_type, num_locs=num_locs
        )
        return quantization_gmm_shells(gmm, shells, z, resolutions=None, paddings=None,
                                       grid_locs=grid_locations_per_shell)

    elif grid_type == "uniform":
        shells, z = generate_non_overlapping_shells(
            gmm=gmm, threshold=threshold, grid_type=grid_type
        )
        return quantization_gmm_shells(gmm, shells, z,
                                       resolutions=[(10, 10)] * len(shells),
                                       paddings=[0.1] * len(shells),
                                       grid_locs=None)  # default grid settings
    else:
        raise ValueError("grid_type must be either 'w2-gaussian-optimal' or 'uniform'")


def quantization_gmm_shells(gmm, shell_inputs, z, resolutions=None, paddings=None, grid_locs=None):
    """
    Compute the quantization of a GMM for set shell regions
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
    num_shells = len(shell_inputs)
    dim = len(shell_inputs[0])
    if resolutions is None:
        resolutions = [tuple([10] * dim) for _ in range(num_shells)]
    if paddings is None:
        paddings = [0.1] * num_shells

    w2_squared_sum = torch.zeros(1)
    all_locs = []
    all_probs = []
    all_R1_grids = []
    all_z_probs = torch.zeros(1)

    for i, shell_input in enumerate(shell_inputs):
        if grid_locs is not None:
            grid_locs_per_shell = grid_locs[i]
            R1_grid = Grid(locs_per_dim=grid_locs_per_shell, bounds=shell_input)  # w2-optimal-approx-gaussian
        else:
            shape = resolutions[i]
            pad = paddings[i]
            R1_grid = Grid.shell(shell_input, shape, pad)  # uniform

        locs_R1 = R1_grid.get_locs()

        # z for all dims
        z_expanded = [z[j].unsqueeze(0) for j in range(dim)]
        R1_inner = Grid(locs_per_dim=z_expanded, bounds=shell_input)
        R1_outer = Grid(locs_per_dim=z_expanded)

        disc_R1 = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_grid)
        disc_R1_inner = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_inner)
        disc_R1_outer = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_outer)

        w2_R1 = disc_R1.w2
        w2_R1_inner = disc_R1_inner.w2
        w2_R1_outer = disc_R1_outer.w2

        # w2_shell = w2_R1 + (w2_R1_outer - w2_R1_inner)
        # total_w2 += w2_shell
        w2_squared_sum += (w2_R1.pow(2) + (w2_R1_outer.pow(2) - w2_R1_inner.pow(2)))

        print(f"Shell bounds: {shell_input}")
        # print(f"  - W2 error: {w2_shell}")

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
    total_w2 = w2_squared_sum.sqrt().item()
    print(f'Total W2 error: {total_w2}')

    return probs_all, locs_all, total_w2, all_R1_grids


def collapse_into_gaussian(locs, covs, probs):
    weights = probs / probs.sum()
    mean = (weights.unsqueeze(1) * locs).sum(dim=0)

    cov = torch.zeros_like(covs[0])
    for i in range(len(probs)):
        diff = (locs[i] - mean).unsqueeze(0)
        cov += weights[i] * (covs[i] + diff.T @ diff)

    return mean, cov

def shift_shell(shell, dx=0.2, dy=0.2):
    return [
        (shell[0][0] + dx, shell[0][1] + dx),
        (shell[1][0] + dy, shell[1][1] + dy)
    ]

def check_overlap(shell1,shell2, tol=1e-4):
    # return not (
    #     shell1[0][1] <= shell2[0][0] or shell1[0][0] >= shell2[0][1] or
    #     shell1[1][1] <= shell2[1][0] or shell1[1][0] >= shell2[1][1]
    # )
    for (low1, high1), (low2, high2) in zip(shell1, shell2):
        if float(high1) <= float(low2) + tol or float(low1) >= float(high2) - tol:
            return False
    return True

def clip_locations(locs, shell):
    grid_list = [torch.sort(torch.unique(locs[:, i]))[0] for i in range(locs.shape[1])]
    grid_list_clipped = []
    for i, (dim_grid, (lower, upper)) in enumerate(zip(grid_list, shell)):
        in_bounds = (dim_grid >= lower) & (dim_grid <= upper)
        grid_list_clipped.append(dim_grid[in_bounds])
    return grid_list_clipped


def generate_non_overlapping_shells(gmm, threshold=2, grid_type="w2-gaussian-optimal", num_locs=10):
    assert grid_type in ["w2-gaussian-optimal", "uniform"], "grid_type must be 'w2-gaussian-optimal' or 'uniform'"

    num_components = gmm.component_distribution.batch_shape[0]
    locs = gmm.component_distribution.loc
    covs = gmm.component_distribution.covariance_matrix
    probs = gmm.mixture_distribution.probs

    visited = set()
    groups = []
    tree = KDTree(locs)

    # grouping based on location of mean and overlap based on MW2
    if num_components == 1:
        groups = [[0]]
    else:
        for i in range(num_components):
            if i in visited:
                continue
            group = [i]
            _, neighbours = tree.query(locs[i], k=num_components)
            neighbours = np.atleast_1d(neighbours).tolist()

            for j in neighbours:
                if j == i or j in visited:
                    continue
                a = dd.MixtureMultivariateNormal(
                    mixture_distribution=torch.distributions.Categorical(probs=probs[i].unsqueeze(0)),
                    component_distribution=dd.MultivariateNormal(loc=locs[i].unsqueeze(0),
                                                                 covariance_matrix=covs[i].unsqueeze(0)))
                b = dd.MixtureMultivariateNormal(
                    mixture_distribution=torch.distributions.Categorical(probs=probs[j].unsqueeze(0)),
                    component_distribution=dd.MultivariateNormal(loc=locs[j].unsqueeze(0),
                                                                 covariance_matrix=covs[j].unsqueeze(0)))
                w2 = GMMWas.w2(a, b)
                print(f'w2 between multi norm {i} and {j}: {w2}')
                if w2 < threshold:
                    group.append(j)
                    visited.add(j)
            visited.add(i)
            groups.append(group)
            print(f'Grouped components: {group}')

    shells = []
    grid_locations_per_shell = []

    for group in groups:
        group_locs = locs[group]
        group_covs = covs[group]
        group_probs = probs[group]

        # approximating groups of components by one Gaussian
        mean, cov = collapse_into_gaussian(group_locs, group_covs, group_probs)
        std = torch.sqrt(torch.diag(cov))

        if grid_type == "uniform":
            # uniform with one std is ~60% mass
            shell = [
                (mean[0] - std[0], mean[0] + std[0]),
                (mean[1] - std[1], mean[1] + std[1])
            ]
            while any(check_overlap(shell, s) for s in shells):
                print(f'Shells overlapping, shifting shell')
                shell = shift_shell(shell, dx=0.1, dy=0.1)

            shells.append(shell)
        else:
            # 2std ~95% mass
            shell = [
                (mean[0] - std[0], mean[0] + std[0]),
                (mean[1] - std[1], mean[1] + std[1])
            ]
            while any(check_overlap(shell, s) for s in shells):
                print(f'Shells overlapping, shifting shell')
                shell = shift_shell(shell, dx=0.1, dy=0.1)
            shells.append(shell)

            norm = dd.MultivariateNormal(loc=mean, covariance_matrix=cov)
            disc = dd.discretization_generator(norm, num_locs=num_locs)
            locs_norm = disc.locs

            # if grid too tiny then no locations that fit in grid - what to do?
            grid_list_clipped = clip_locations(locs_norm, shell)
            if any(len(g)==0 for g in grid_list_clipped):
                print('Grid too small, no locations fit inside, expand grid!')
            grid_locations_per_shell.append(grid_list_clipped)

    z = (probs.unsqueeze(1) * locs).sum(dim=0)  # same for both methods

    if grid_type == "uniform":
        return shells, z
    else:
        return shells, grid_locations_per_shell, z

def estimate_eps(samples, min_samples=20, plot=False):
    samples_np = samples.detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(samples_np)
    distances, _ = nbrs.kneighbors(samples_np)
    k_distances = distances[:, -1]
    k_distances = np.sort(k_distances)  # sorted in increasing order based on distance
    # x = np.arange(len(k_distances))
    # kl = KneeLocator(x, k_distances, curve='convex', direction='increasing')
    # eps = k_distances[kl.knee]
    eps = np.percentile(k_distances, 95)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(k_distances[::-1])
        plt.axhline(y=eps, color='r', linestyle='--', label=f'$\epsilon$ = {eps:.4f}')
        # plt.title(f"k-distance plot (min_samples={min_samples})")
        plt.xlabel("Sorted index")
        plt.ylabel("k-distance")
        plt.legend()
        plt.grid(True)
        plt.savefig('figures/DBSCAN_epsilon.svg')
        plt.show()

    return eps

def dbscan_shells(gmm, eps=None, min_samples=20):
    # assuming knowledge about gmm to set eps and min_samples
    # gmm stats for z location
    means = gmm.component_distribution.loc
    probs = gmm.mixture_distribution.probs
    # rule of thumb for min_samples
    # dim = len(means[-1])
    # min_samples = dim + 1

    num_components = gmm.component_distribution.batch_shape[0]
    num_samples = torch.tensor([100*num_components])
    samples = gmm.sample((num_samples,))

    if eps is None:  # elbow method for eps
        eps = estimate_eps(samples, min_samples=min_samples, plot=True)

    X = samples.detach().numpy()
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_

    shells = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # dbscan identifies noise, so we can discard it here

    for label in unique_labels:
        mask = torch.tensor(labels == label)
        cluster_points = samples[mask]

        # must be dense enough to form shell around it, negligible complexity as its around O(nd) while
        # eps estimate already has complexity of O(n^2d)
        if len(cluster_points) < min_samples:
            # there can be very small clusters left in dbscan as EVERYTHING is clustered
            continue

        mean = cluster_points.mean(dim=0)

        shell = [(m - eps, m + eps) for m in mean]
        shells.append(shell)

    z = (probs.unsqueeze(1) * means).sum(dim=0)  # z location stays as average of component means

    # return shells, z
    if len(shells) == 1:
        return shells, z
    else:
        final_shells = []
        for shell in shells:
            if all(not check_overlap(shell, existing) for existing in final_shells):
                final_shells.append(shell)
            else:
                print(f'Shells overlap! Change parameters eps and min_samples.')

        if len(final_shells) == 0:
            print(f'No shells found! Increase eps and/or lower min_samples required in cluster.')
    # increase region of eps so more points included or lower amount of points needed in a cluster
        return final_shells, z

def set_grid_locations(gmm, shells, num_locs=100):
    disc = dd.discretization_generator(gmm, num_locs)
    locs_norm = disc.locs
    grid_locations_per_shell = []
    for shell in shells:
        # if grid too tiny then no locations that fit in grid - expand grid manually
        grid_list_clipped = clip_locations(locs_norm, shell)
        if any(len(g) == 0 for g in grid_list_clipped):
            print('Grid too small, no locations fit inside, expand grid!')
        grid_locations_per_shell.append(grid_list_clipped)

    return grid_locations_per_shell


if __name__ == "__main__":
    num_dims = 2
    num_mix_elems = 4
    batch_size = torch.Size()
    torch.manual_seed(1)

    # only diagonal and pos def covariance matrices
    covariance_diag = torch.exp(torch.randn(batch_size + (num_mix_elems, num_dims,)))
    covariance_matrix = torch.diag_embed(covariance_diag)
    locs = torch.randn(batch_size + (num_mix_elems, num_dims,))
    probs = torch.rand(batch_size + (num_mix_elems,))

    # locs = torch.tensor([[0.3, 0.3], [0.4, 0.4], [1.4, 1.4], [1.3, 1.3]])
    # covariance_matrix = torch.tensor([[[0.02, 0.0000],
    #                                    [0.0000, 0.02]],
    #                                   [[0.03, 0.0000],
    #                                    [0.0000, 0.06]],
    #                                   [[0.02, 0.0000],
    #                                    [0.0000, 0.02]],
    #                                   [[0.05, 0.0000],
    #                                    [0.0000, 0.05]]])
    # probs = torch.tensor([0.6, 0.2, 0.2, 0.6])

    # normalize
    probs = probs / probs.sum(dim=-1, keepdim=True)

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
    probs_gmm = disc_gmm.probs.detach().numpy()
    # grid_gmm = locs_gmm.detach().numpy()  # grid for gmm
    print(f'W2 error original Signature operation: {disc_gmm.w2}')
    # print(f'Number of signature locations: {len(locs_gmm)}')
    #
    # scaling to compare prob mass across grid and signature
    s_gmm = (probs_gmm - probs_gmm.min()) / (probs_gmm.max() - probs_gmm.min()) * 100
    plt.figure(figsize=(8, 6))
    plt.scatter(locs_gmm.detach().numpy()[:,0], locs_gmm.detach().numpy()[:,1], color='blue', marker='o', alpha=0.6)
    plt.title(f"Original signature with W2 error {disc_gmm.w2}")
    plt.show()

    # applying to whole GMM instead component wise
    shell_input = [(torch.tensor(-1.0), torch.tensor(1.0)), (torch.tensor(-1.0), torch.tensor(1.0))]

    locs_p = gmm.component_distribution.loc  # [num_components, dim]
    w2 = torch.zeros(1)

    R1_grid = Grid.shell(shell_input, (10, 10))
    locs_R1 = R1_grid.get_locs()

    # component
    # z = locs_p[2, :]  # can be any arbitrary location
    z = torch.tensor([-1.0, -1.0])  # how does this not change W2 error??

    # calc w2 for R1(k)
    disc_R1 = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_grid)
    probs_R1, w2_R1, z_probs_R1 = disc_R1.probs, disc_R1.w2, disc_R1.z_probs   # probs already weighted by components to add to 1 ...
    print(f"W2 inside R1 grid: {w2_R1}")

    # calc w2 for R^n with z - should just be same as R1_inner with unbounded region
    # w2_Rn = w2_multi_norm_dist_for_set_locations(norm=component_p, signature_locs=z)
    R1_outer = Grid(locs_per_dim=[z[0].unsqueeze(0), z[1].unsqueeze(0)])
    disc_R1_outer = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_outer)
    w2_R1_outer = disc_R1_outer.w2
    print(f'W2 R1 outer: {w2_R1_outer}')

    # calc w2 for R1 with z
    R1_inner = Grid(locs_per_dim=[z[0].unsqueeze(0), z[1].unsqueeze(0)], bounds=shell_input)
    disc_R1_inner = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_inner)
    w2_R1_inner = disc_R1_inner.w2
    print(f'W2 R1 inner: {w2_R1_inner}')

    # total w2
    w2_p = (w2_R1.pow(2) + (w2_R1_outer.pow(2) - w2_R1_inner.pow(2))).sqrt().item()
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
    plt.title(f'One general grid for whole GMM, W2: {w2_p}')
    plt.show()

    # uniform disjoint grids
    # probs_all, locs_all, total_w2, all_grids = shelling(gmm, grid_type="uniform")
    # s = (probs_all - probs_all.min()) / (probs_all.max() - probs_all.min()) * 100
    # print(f"Total W2 error over all shells: {total_w2}")
    #
    # plt.figure(figsize=(8, 6))
    # ax = plt.gca()
    # plt.scatter(locs_gmm.detach().numpy()[:,0], locs_gmm.detach().numpy()[:,1], label='original sig locs',
    #             color='blue', marker='o', alpha=0.6)
    # ax.scatter(locs_all[:, 0], locs_all[:, 1], label='Locs', color='red', alpha=0.6)  # locs
    # cmap = plt.colormaps.get_cmap('tab10')
    # for idx, R1 in enumerate(all_grids):
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
    # plt.title(f"Discretization for GMM with uniform shells W2: {total_w2}")
    # plt.legend()
    # plt.grid(True)
    # plt.axis("equal")
    # plt.show()

    # probs_all, locs_all, total_w2, all_grids = shelling(gmm, grid_type="w2-gaussian-optimal")
    # s = (probs_all - probs_all.min()) / (probs_all.max() - probs_all.min()) * 100
    # print(f"Total W2 error over all shells: {total_w2}")
    #
    # plt.figure(figsize=(8, 6))
    # ax = plt.gca()
    # # ax.scatter(locs_p[:, 0], locs_p[:, 1], label='GMM Component Means', s=50, color='red', alpha=0.6)  # means
    # ax.scatter(locs_all[:, 0], locs_all[:, 1], label='Locs', color='red', alpha=0.6)  # locs
    # cmap = plt.colormaps.get_cmap('tab10')
    # for idx, R1 in enumerate(all_grids):
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
    # plt.title(f"Discretization for GMM with heuristic for w2-optimal-shells, W2: {total_w2}")
    # plt.legend()
    # plt.grid(True)
    # plt.axis("equal")
    # plt.show()

    # using clustering
    shells, z = dbscan_shells(gmm)
    grid_locs = set_grid_locations(gmm, shells)
    probs, locs, w2, grids = quantization_gmm_shells(gmm, shells, z, grid_locs=grid_locs, paddings=[0.01]*len(shells))
    s = (probs - probs.min()) / (probs.max() - probs.min()) * 100
    print(f"Total W2 error over all shells: {w2}")

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plt.scatter(locs_gmm.detach().numpy()[:,0], locs_gmm.detach().numpy()[:,1], label='original sig locs',
                color='blue', marker='o', alpha=0.3)
    ax.scatter(locs[:, 0], locs[:, 1], label='Locs', color='red', alpha=0.6)  # locs
    cmap = plt.colormaps.get_cmap('tab10')
    for idx, R1 in enumerate(grids):
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
    plt.title(f"Discretization for GMM with clustering heuristic, W2: {w2}")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.scatter(locs[:, 0], locs[:, 1], label='Locs', s=s, color='red', alpha=0.6)  # locs
    cmap = plt.colormaps.get_cmap('tab10')
    for idx, R1 in enumerate(grids):
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
    plt.title(f"Discretization for GMM with clustering heuristic, W2: {w2}")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

    # take signature locations for grid - one component
    # grid_list = [torch.sort(torch.unique(locs_gmm[:, i]))[0] for i in range(locs_gmm.shape[1])]
    # # applying to whole GMM instead component wise
    # shell_input = [(torch.tensor(-1.0), torch.tensor(2.0)), (torch.tensor(-1.0), torch.tensor(2.0))]
    # locs_p = gmm.component_distribution.loc  # [num_components, dim]
    # w2 = torch.zeros(1)
    #
    # grid_list_clipped = []  # removing locations outside shell region
    # for i, (dim_grid, (lower, upper)) in enumerate(zip(grid_list, shell_input)):
    #     in_bounds = (dim_grid >= lower) & (dim_grid <= upper)
    #     grid_list_clipped.append(dim_grid[in_bounds])
    #
    # R1_grid = Grid(locs_per_dim=grid_list_clipped, bounds=shell_input)
    # locs_R1 = R1_grid.get_locs()
    #
    # # z = locs_p[2, :]  # can be any arbitrary location
    # z = torch.tensor([2.0, 2.0])  # how does this not change W2 error??
    #
    # # calc w2 for R1(k)
    # disc_R1 = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_grid)
    # probs_R1, w2_R1, z_probs_R1 = disc_R1.probs, disc_R1.w2, disc_R1.z_probs  # probs already weighted by components to add to 1 ...
    # print(f"W2 inside R1 grid: {w2_R1}")
    #
    # # calc w2 for R^n with z - should just be same as R1_inner with unbounded region
    # # w2_Rn = w2_multi_norm_dist_for_set_locations(norm=component_p, signature_locs=z)
    # R1_outer = Grid(locs_per_dim=[z[0].unsqueeze(0), z[1].unsqueeze(0)])
    # disc_R1_outer = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_outer)
    # w2_R1_outer = disc_R1_outer.w2
    # print(f'W2 R1 outer: {w2_R1_outer}')
    #
    # # calc w2 for R1 with z
    # R1_inner = Grid(locs_per_dim=[z[0].unsqueeze(0), z[1].unsqueeze(0)], bounds=shell_input)
    # disc_R1_inner = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_inner)
    # w2_R1_inner = disc_R1_inner.w2
    # print(f'W2 R1 inner: {w2_R1_inner}')
    #
    # # total w2
    # w2_p = (w2_R1.pow(2) + (w2_R1_outer.pow(2) - w2_R1_inner.pow(2))).sqrt().item()
    # print(f'W2 error for whole GMM: {w2_p}')
    #
    # # z location and mass
    # print(f'prob R1 sum: {probs_R1.sum()}')
    # print(f'prob mass of z: R1-1 = {z_probs_R1}, at location: {z}')
    #
    # grid_mass = probs_R1.sum().item()
    # z_mass = z_probs_R1.item()
    # mass_scale = (1 - z_mass)  # percentage mass left over in grid
    # probs_R1 = probs_R1 * mass_scale
    # print(f'Prob mass of grid normalized by sum of grid and z: {probs_R1.sum().item()}')
    #
    # locs_ = torch.cat([locs_R1, z.unsqueeze(0)], dim=0)
    # probs_ = torch.cat([probs_R1, z_probs_R1], dim=0)  # need to normalize ?
    # # probs_ = probs_ / probs_.sum(dim=-1, keepdim=True)
    # print(f'Total probs: {probs_.sum()}')
    #
    # s = (probs_ - probs_.min()) / (probs_.max() - probs_.min()) * 100
    #
    # plt.figure(figsize=(8, 6))
    # ax = plt.gca()
    # ax.scatter(locs_[:, 0], locs_[:, 1], label='Locs', s=s, color='red', alpha=0.6)
    # core_lower_vertices_per_dim = R1_grid.lower_vertices_per_dim
    # core_upper_vertices_per_dim = R1_grid.upper_vertices_per_dim
    # for i in range(len(core_lower_vertices_per_dim[0])):
    #     for j in range(len(core_lower_vertices_per_dim[1])):
    #         x0 = core_lower_vertices_per_dim[0][i].item()
    #         x1 = core_upper_vertices_per_dim[0][i].item()
    #         y0 = core_lower_vertices_per_dim[1][j].item()
    #         y1 = core_upper_vertices_per_dim[1][j].item()
    #         rect = patches.Rectangle(
    #             (x0, y0),
    #             x1 - x0,
    #             y1 - y0,
    #             edgecolor='blue',
    #             facecolor='none',
    #             linewidth=1.5,
    #             linestyle='-'
    #         )
    #         ax.add_patch(rect)
    # plt.legend()
    # plt.title(f'W2-optimal signature grid for whole GMM, W2: {w2_p}')
    # plt.show()

    # testing overlap
    # shell_inputs = [
    #     [(torch.tensor(-1.0), torch.tensor(-0.7)), (torch.tensor(-1.0), torch.tensor(-0.7))],
    #     [(torch.tensor(-0.6), torch.tensor(-0.4)), (torch.tensor(-0.6), torch.tensor(-0.4))],
    # ]
    # shell_inputs = [[(torch.tensor(-1.0), torch.tensor(-0.6)), (torch.tensor(-1.0), torch.tensor(-0.6))]]
    # locs_p = gmm.component_distribution.loc  # [num_components, dim]
    # std_devs = gmm.component_distribution.stddev
    # shell_inputs = []
    # for i in range(locs_p.shape[0]):
    #     lower = locs_p[i, 0] - std_devs[i, 0]
    #     upper = locs_p[i, 1] + std_devs[i, 1]
    #     pair = (lower, upper)
    #     shell_inputs.append([pair, pair])
    # z = locs_p[2, :]  # Arbitrary location, or loop over components

    # total_w2 = 0.0
    # w2_squared_sum = torch.zeros(1)
    # all_locs = []
    # all_probs = []
    # all_R1_grids = []
    # all_z_probs = torch.zeros(1)
    #
    # # for shell_input, grid_locs_per_shell in zip(shell_inputs, grid_locs):
    # #     R1_grid = Grid(locs_per_dim=grid_locs_per_shell, bounds=shell_input)
    # #     locs_R1 = R1_grid.get_locs()
    # for shell_input in shell_inputs:
    #     R1_grid = Grid.shell(shell_input, (10, 10), 0.01)
    #     locs_R1 = R1_grid.get_locs()
    #
    #     R1_inner = Grid(locs_per_dim=[z[0].unsqueeze(0), z[1].unsqueeze(0)], bounds=shell_input)
    #     R1_outer = Grid(locs_per_dim=[z[0].unsqueeze(0), z[1].unsqueeze(0)])
    #
    #     disc_R1 = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_grid)
    #     disc_R1_inner = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_inner)
    #     disc_R1_outer = DiscretizedMixtureMultivariateNormalQuantization(gmm, grid=R1_outer)
    #
    #     w2_R1 = disc_R1.w2
    #     print(f'W2 inside R1: {w2_R1.item()}')
    #     w2_R1_inner = disc_R1_inner.w2
    #     print(f'W2 inner R1 {w2_R1_inner.item()}')
    #     w2_R1_outer = disc_R1_outer.w2
    #     print(f'W2 outer R1 {w2_R1_outer.item()}')
    #
    #     w2_squared_sum += (w2_R1.pow(2) + (w2_R1_outer.pow(2) - w2_R1_inner.pow(2)))
    #
    #     # w2_shell = w2_R1 + (w2_R1_outer - w2_R1_inner)
    #     # total_w2 += w2_shell
    #
    #     print(f"Shell bounds: {shell_input}")
    #     # print(f"  - W2 error: {w2_shell}")
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
    # total_w2 = w2_squared_sum.sqrt().item()
    #
    # probs_all = probs_all / probs_all.sum(dim=-1, keepdim=True)  # normalize again wrt to all grids
    # s = (probs_all - probs_all.min()) / (probs_all.max() - probs_all.min()) * 100
    # print(f"Total W2 error over all shells: {total_w2}")
    #
    # plt.figure(figsize=(8, 6))
    # ax = plt.gca()
    # # ax.scatter(locs_p[:, 0], locs_p[:, 1], label='GMM Component Means', s=50, color='red', alpha=0.6)  # means
    # ax.scatter(locs_all[:, 0], locs_all[:, 1], label='Locs', color='red', alpha=0.6)  # locs
    # cmap = plt.colormaps.get_cmap('tab10')
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
    # plt.title(f"Discretization for GMM with overlapping shells W2: {total_w2}")
    # plt.legend()
    # plt.grid(True)
    # plt.axis("equal")
    # plt.show()

    # multiple shells on one gmm - function
    # shell_inputs = [
    #     [(torch.tensor(-1.0), torch.tensor(-0.6)), (torch.tensor(-1.0), torch.tensor(-0.6))],
    #     [(torch.tensor(-0.5), torch.tensor(1.0)), (torch.tensor(-0.5), torch.tensor(1.0))],
    #     [(torch.tensor(-1.5), torch.tensor(-1.1)), (torch.tensor(-1.5), torch.tensor(-1.1))]
    # ]
    # resolutions = [(3, 3), (10, 10), (4, 4)]
    # padding = [0.05, 0.1, 0.05]
    #
    # probs_all, locs_all, total_w2, all_R1_grids = quantization_gmm_shells(
    #     gmm=gmm,
    #     z=z,
    #     shell_inputs=shell_inputs,
    #     resolutions=resolutions,
    #     paddings=padding)
    #
    # s = (probs_all - probs_all.min()) / (probs_all.max() - probs_all.min()) * 100
    # print(f"Total W2 error over all shells: {total_w2}")
    #
    # plt.figure(figsize=(8, 6))
    # ax = plt.gca()
    # # ax.scatter(locs_p[:, 0], locs_p[:, 1], label='GMM Component Means', s=50, color='red', alpha=0.6)  # means
    # ax.scatter(locs_all[:, 0], locs_all[:, 1], label='Locs', s=s, color='red', alpha=0.6)  # locs
    # cmap = plt.colormaps.get_cmap('tab10')
    # for idx, R1 in enumerate(all_R1_grids):
    #     color = cmap(idx % 10)
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
    #                 edgecolor=color,
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


    # # each separate grids
    # disc_gmm = DiscretizedMixtureMultivariateNormalQuantizationShell(gmm)
    #
    # # stats
    # locs = disc_gmm.locs.detach().numpy()
    # probs = disc_gmm.probs.detach().numpy()
    # s = (probs - probs.min()) / (probs.max() - probs.min()) * 100
    # print(f"Total W2 error: {disc_gmm.w2.item()}")
    # cmap = plt.colormaps.get_cmap('tab10')
    #
    # # multiple shells
    # plt.figure(figsize=(8, 6))
    # ax = plt.gca()
    # ax.scatter(locs[:, 0], locs[:, 1], label='Locs', s=s, color='red', alpha=0.6)
    # for idx, R1 in enumerate(disc_gmm.R1_grids):
    #     core_lower_vertices_per_dim = R1.lower_vertices_per_dim
    #     core_upper_vertices_per_dim = R1.upper_vertices_per_dim
    #     color = cmap(idx % 10)
    #     for i in range(len(core_lower_vertices_per_dim[0])):
    #         for j in range(len(core_lower_vertices_per_dim[1])):
    #             x0 = core_lower_vertices_per_dim[0][i].item()
    #             x1 = core_upper_vertices_per_dim[0][i].item()
    #             y0 = core_lower_vertices_per_dim[1][j].item()
    #             y1 = core_upper_vertices_per_dim[1][j].item()
    #
    #             rect = patches.Rectangle(
    #                 (x0, y0),
    #                 x1 - x0,
    #                 y1 - y0,
    #                 edgecolor=color,
    #                 facecolor='none',
    #                 linewidth=1.5,
    #                 linestyle='-'
    #             )
    #             ax.add_patch(rect)
    # plt.legend()
    # plt.title(f"Separate shells for GMM")
    # plt.show()

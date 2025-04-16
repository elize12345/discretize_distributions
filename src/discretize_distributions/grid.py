import torch
from typing import Union
from matplotlib import pyplot as plt, patches
import discretize_distributions.utils as utils
from discretize_distributions.distributions.multivariate_normal import MultivariateNormal
import discretize_distributions.tensors as tensors
import ot

class GridCell:
    def __init__(self, loc: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
        self.loc = loc  # (d,)
        self.lower = lower  # (d,)
        self.upper = upper  # (d,)


class Grid:
    def __init__(self, locs_per_dim: list):
        """
        locs_per_dim: list of 1D torch tensors, each of shape (n_i,)
        Example: [torch.linspace(0, 1, 5), torch.tensor([0., 2., 4.])]
        """
        # self.locs_per_dim = locs_per_dim
        # to ensure sorting so no negative probabilities when taking cdf of upper and lower vertices,
        # so ensures lower vertice < upper vertice in Voronoi partition calculation
        self.locs_per_dim = [locs.sort().values for locs in locs_per_dim]
        self.dim = len(locs_per_dim)
        self.shape = tuple(len(p) for p in locs_per_dim)

        self.lower_vertices_per_dim, self.upper_vertices_per_dim = self._compute_voronoi_edges()

    @staticmethod
    def from_shape(shape, interval_per_dim: torch.Tensor):
        assert len(shape) == len(interval_per_dim), "Shape and interval dimensions do not match."
        locs_per_dim = [torch.linspace(*interval_per_dim[dim], shape[dim]) for dim in range(len(shape))]
        return Grid(locs_per_dim)

    def meshgrid(self, indexing='ij'):
        """Returns meshgrid view (not flattened)."""
        return torch.meshgrid(*self.locs_per_dim, indexing=indexing)

    def get_locs(self):
        """Returns (N, d) tensor of all grid locs, computed lazily."""
        mesh = self.meshgrid()
        stacked = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
        return stacked

    def _compute_voronoi_edges(self):
        """Computes (lower, upper) corners of axis-aligned Voronoi cells with unbounded outer regions."""
        lower_vertices_per_dim = []
        upper_vertices_per_dim = []

        for dim_locs in self.locs_per_dim:
            midlocs = (dim_locs[1:] + dim_locs[:-1]) / 2
            lower = torch.cat([
                torch.full((1,), -torch.inf, device=dim_locs.device, dtype=dim_locs.dtype),
                midlocs
            ])
            upper = torch.cat([
                midlocs,
                torch.full((1,), torch.inf, device=dim_locs.device, dtype=dim_locs.dtype),
            ])
            lower_vertices_per_dim.append(lower)
            upper_vertices_per_dim.append(upper)
        return lower_vertices_per_dim, upper_vertices_per_dim


    def voronoi_edge_shell(self, shell):
        """Finds the closest Voronoi edge to the input shell"""
        vor_shell = []
        # boundary = [
        #     (min_x, max_x),  # for x-dimension (dim=0)
        #     (min_y, max_y)  # for y-dimension (dim=1)
        # ]
        for dim in range(self.dim):
            if (torch.any(torch.isclose(self.locs_per_dim[dim], self.lower_vertices_per_dim[dim])) or
                    torch.any(torch.isclose(self.locs_per_dim[dim], self.upper_vertices_per_dim[dim]))):
                print(f"Grid locs overlap lower Voronoi edges in dim {dim}")

            min_shell, max_shell = shell[dim]
            lower = self.lower_vertices_per_dim[dim]
            upper = self.upper_vertices_per_dim[dim]

            # index for closest edge to input shell
            lower_index = torch.argmin(torch.abs(lower - min_shell))
            upper_index = torch.argmin(torch.abs(upper - max_shell))

            # this is for the largest lower edge ≤ shell_min and smallest upper edge ≥ shell_max

            # this will give all values where boolean is True, if an edge is lower than the shell input
            # lower_index = torch.nonzero(lower <= min_shell, as_tuple=True)[0]
            # # then we select the last index which satisfies, so that is the max value the shell can be
            # lower_index = lower_index[-1] if len(lower_index) > 0 else 0
            # upper_index = torch.nonzero(upper >= max_shell, as_tuple=True)[0]
            # upper_index = upper_index[0] if len(upper_index) > 0 else -1

            vor_shell_min = lower[lower_index]
            vor_shell_max = upper[upper_index]
            vor_shell.append((vor_shell_min, vor_shell_max))

            # check
            # min_matches = torch.any(torch.isclose(lower, vor_shell_min, atol=1e-6))
            # max_matches = torch.any(torch.isclose(upper, vor_shell_max, atol=1e-6))
            #
            # if not (min_matches and max_matches):
            #     print(f"Dimension {dim}: shell ({min_shell.item()}, {max_shell.item()}) not aligned with Voronoi edges.")

        return vor_shell

    def shell(self, shell):
        """Computes the outer and core points based on input shell
        param: input 'shell' list of tensors for max, min values per dim of wanted shell
        """
        grid_points = self.get_locs()  # (N,d) locations
        n_dims = len(grid_points[1])
        shell_points = []
        core = []
        outer_points = []
        visited = set()
        vor_shell = self.voronoi_edge_shell(shell)

        for point in grid_points:
            pt_tuple = tuple(point.tolist())

            if pt_tuple in visited:
                continue  # so no double points added
            visited.add(pt_tuple)  # once processed its added

            is_shell = False
            outer = False

            for dim in range(n_dims):
                min_shell, max_shell = vor_shell[dim]
                value = point[dim]
                # if value <= min_b or value >= max_b:  # this groups outside also to shell
                if torch.isclose(value, min_shell) or torch.isclose(value, max_shell):
                # if value == min_shell or value == max_shell:
                    is_shell = True
                    # print("Points on shell!")
                elif value < min_shell or value > max_shell:
                    outer = True
                    break  # don't need to check other dimensions
                    # shell_points.append(point)
                    # visited_shell.add(pt_tuple)
                    # is_shell = True
                    # break  # so only check one dimension if its in shell, no need to iterate through other dims
            if outer:
                outer_points.append(point)
            elif is_shell:
                shell_points.append(point)
            else:
                core.append(point)

        # stack list into tensor and check if empty
        shell_tensor = torch.stack(shell_points) if shell_points else torch.empty((0, n_dims))
        core_tensor = torch.stack(core) if core else torch.empty((0, n_dims))
        outer_tensor = torch.stack(outer_points) if outer_points else torch.empty((0, n_dims))

        return shell_tensor, core_tensor, outer_tensor  # (N,d)

    def shell_discretize_multi_norm_dist(self, norm, shell, new_loc):

        locs = self.get_locs()
        # probability computation from original func
        probs_per_dim = [utils.cdf(self.upper_vertices_per_dim[dim]) - utils.cdf(self.lower_vertices_per_dim[dim])
                         for dim in range(self.dim)]
        mesh = torch.meshgrid(*probs_per_dim, indexing='ij')
        stacked = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
        probs = stacked.prod(-1)

        scaled_locs_per_dim = [self.locs_per_dim[dim] / norm.variance[dim] for dim in range(self.dim)]
        w2_per_dim = [utils.calculate_w2_disc_uni_stand_normal(dim_locs) for dim_locs in scaled_locs_per_dim]
        w2 = torch.stack(w2_per_dim).pow(2).sum().sqrt()

        # shelling step
        shell, core, outer = self.shell(shell)

        core_mask = (locs[:, None, :] == core[None, :, :]).all(dim=-1).any(dim=1)
        outer_mask = ~core_mask  # as there should ONLY be core or outer
        # print(f'core {core_mask.sum().item()}')
        # print(f'outer {outer_mask.sum().item()}')
        locs_core = locs[core_mask]
        probs_core = probs[core_mask]

        locs_outer = locs[outer_mask]
        probs_outer = probs[outer_mask]
        total_prob = probs_outer.sum()

        N = new_loc.shape[0]
        new_prob = torch.full((N,), total_prob / N)

        # added w2 to move to new location
        M = ot.dist(locs_outer, new_loc, metric='sqeuclidean')  # new loc should be (1,d) size
        M /= M.max()
        w2_added = ot.sinkhorn2(a=probs_outer, b=new_prob.view(-1), M=M, reg=1)

        return locs_core, probs_core, locs_outer, probs_outer, w2, w2_added

    def plot_shell_2d(self, shell):
        """shell is input shell"""

        lower_vertices_per_dim, upper_vertices_per_dim = self._compute_voronoi_edges()

        shell_tensor, core_tensor, outer_tensor = self.shell(shell)

        vor_shell = self.voronoi_edge_shell(shell)

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        ax.scatter(core_tensor[:, 0], core_tensor[:, 1], color='blue', label='Core', alpha=0.6)
        ax.scatter(outer_tensor[:, 0], outer_tensor[:, 1], color='orange', label='Outer', alpha=0.6)

        for i in range(len(lower_vertices_per_dim[0])):
            for j in range(len(lower_vertices_per_dim[1])):
                x0 = lower_vertices_per_dim[0][i].item()
                x1 = upper_vertices_per_dim[0][i].item()
                y0 = lower_vertices_per_dim[1][j].item()
                y1 = upper_vertices_per_dim[1][j].item()
                rect = patches.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    edgecolor='gray',
                    facecolor='none',
                    linewidth=0.5,
                    # linestyle=':'
                )
                ax.add_patch(rect)

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

        ax.set_title('Shell vs Core Points with Voronoi Cells')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True)
        plt.show()

    def __len__(self):
        return int(torch.tensor(self.shape).prod().item())

    def __getitem__(self, idx: Union[tuple, int]):
        """
        Returns a GridCell object for the idx-th loc in the flattened grid.
        """
        if isinstance(idx, tuple):
            multi_idx = list(idx)
        else:
            multi_idx = list(torch.unravel_index(torch.tensor(idx), self.shape))

        loc = torch.stack([self.locs_per_dim[d][i] for d, i in enumerate(multi_idx)])
        lower = torch.stack([self.lower_vertices_per_dim[d][i] for d, i in enumerate(multi_idx)])
        upper = torch.stack([self.upper_vertices_per_dim[d][i] for d, i in enumerate(multi_idx)])
        return GridCell(loc=loc, lower=lower, upper=upper)

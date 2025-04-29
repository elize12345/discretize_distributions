import torch
from typing import Union
from matplotlib import pyplot as plt, patches, cm
import discretize_distributions.utils as utils
from discretize_distributions.distributions.multivariate_normal import MultivariateNormal
import discretize_distributions.tensors as tensors
import ot
import itertools


class GridCell:
    def __init__(self, loc: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
        self.loc = loc  # (d,)
        self.lower = lower  # (d,)
        self.upper = upper  # (d,)


class Grid:
    def __init__(self, locs_per_dim: list, bounds: [] = None):
        """
        locs_per_dim: list of 1D torch tensors, each of shape (n_i,)
        Example: [torch.linspace(0, 1, 5), torch.tensor([0., 2., 4.])]
        """
        # to ensure sorting so no negative probabilities when taking cdf of upper and lower vertices,
        # so ensures lower vertice < upper vertice in Voronoi partition calculation
        # self.locs_per_dim = [locs.sort().values for locs in locs_per_dim]  # use check instead
        assert all((locs[1:] >= locs[:-1]).all() for locs in
                   locs_per_dim), "Each tensor in locs_per_dim must be sorted in ascending order"
        self.locs_per_dim = locs_per_dim
        self.dim = len(locs_per_dim)
        self.shape = tuple(len(p) for p in locs_per_dim)
        self.lower_vertices_per_dim, self.upper_vertices_per_dim = self._compute_voronoi_edges(bounds)

    @staticmethod
    def from_shape(shape, interval_per_dim: torch.Tensor, bounds: [] = None):
        assert len(shape) == len(interval_per_dim), "Shape and interval dimensions do not match."
        locs_per_dim = [torch.linspace(*interval_per_dim[dim], shape[dim]) for dim in range(len(shape))]
        return Grid(locs_per_dim, bounds=bounds)

    def meshgrid(self, indexing='ij'):
        """Returns meshgrid view (not flattened)."""
        return torch.meshgrid(*self.locs_per_dim, indexing=indexing)

    def get_locs(self):
        """Returns (N, d) tensor of all grid locs, computed lazily."""
        mesh = self.meshgrid()
        stacked = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
        return stacked

    @classmethod  # class method - to call function on the class itself - so when you create one with bounds,
    # it uses voronoi edges with bounds
    def from_points(cls, points: torch.Tensor, bounds: [] = None):
        """
        New grid defined by points and possibly set bounds, list of bounds per dimension:
        [bound_lower, bound_upper] = bounds[dim]
        """
        dim = points.shape[1]
        locs_per_dim = []
        for d in range(dim):
            unique_sorted = torch.sort(torch.unique(points[:, d])).values
            locs_per_dim.append(unique_sorted)
        return cls(locs_per_dim, bounds=bounds)

    def _compute_voronoi_edges(self, bounds=None):
        """Computes (lower, upper) corners of axis-aligned bounded Voronoi cells, if bounds are provide,
        else with unbounded outer regions (-inf,inf)."""
        lower_vertices_per_dim = []
        upper_vertices_per_dim = []

        for dim, dim_locs in enumerate(self.locs_per_dim):
            midlocs = (dim_locs[1:] + dim_locs[:-1]) / 2
            if bounds:
                bound_lower, bound_upper = bounds[dim]
                lower = torch.tensor(bound_lower, device=dim_locs.device, dtype=dim_locs.dtype)
                upper = torch.tensor(bound_upper, device=dim_locs.device, dtype=dim_locs.dtype)
            else:
                lower = torch.tensor(-float("inf"), device=dim_locs.device, dtype=dim_locs.dtype)
                upper = torch.tensor(float("inf"), device=dim_locs.device, dtype=dim_locs.dtype)

            lower = torch.cat([lower.unsqueeze(0), midlocs])
            upper = torch.cat([midlocs, upper.unsqueeze(0)])
            lower_vertices_per_dim.append(lower)
            upper_vertices_per_dim.append(upper)

            # lower = torch.cat([
            #     torch.full((1,), -torch.inf, device=dim_locs.device, dtype=dim_locs.dtype),
            #     midlocs
            # ])
            # upper = torch.cat([
            #     midlocs,
            #     torch.full((1,), torch.inf, device=dim_locs.device, dtype=dim_locs.dtype),
            # ])
            # lower_vertices_per_dim.append(lower)
            # upper_vertices_per_dim.append(upper)
        return lower_vertices_per_dim, upper_vertices_per_dim

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

        for point in grid_points:
            pt_tuple = tuple(point.tolist())

            if pt_tuple in visited:
                continue  # so no double points added
            visited.add(pt_tuple)  # once processed its added

            is_shell = False
            outer = False

            for dim in range(n_dims):
                min_shell, max_shell = shell[dim]
                value = point[dim]
                if torch.isclose(value, min_shell) or torch.isclose(value, max_shell):
                    is_shell = True
                    print("Points on shell! Reset boundary")  # assertion if points lie on shell?
                elif value < min_shell or value > max_shell:
                    outer = True
                    break  # don't need to check other dimensions
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

        # then we create a new grid from just the core points
        core_grid = Grid.from_points(core_tensor, bounds=shell) if core_tensor.numel() > 0 else None

        # outer_grids, bounds = self.split_outer_grids(outer_tensor, shell)
        outer_grids = []
        bounds = []

        return shell_tensor, core_tensor, outer_tensor, core_grid, outer_grids, bounds

    def split_outer_grids(self, outer_tensor, shell):
        """
        Args:
            outer_tensor:
            shell:

        Returns:

        """
        grid_points = self.get_locs()  # (N,d) locations
        n_dims = len(grid_points[1])
        # same for outer but ensuring it stays rectangular
        outer_grids = []
        bounds = []
        if outer_tensor.numel() > 0:
            # -1 is less than, 0 is inside, and 1 is greater than core
            all_patterns = list(itertools.product([-1, 0, 1], repeat=n_dims))  # generates all possible combinations
            # between -1, 0 and 1 across each dimension, which is 3^D combinations, e.g. in 2D
            #   (-1, -1), (-1, 0), (-1, 1),
            #   ( 0, -1), ( 0, 0), ( 0, 1),
            #   ( 1, -1), ( 1, 0), ( 1, 1)
            all_patterns.remove((0,) * n_dims)  # this excludes all patterns that is just 0,0,... as it is the core
            print(f'No. of patter combinations: {len(all_patterns)}')
            for pattern in all_patterns:
                bound_per_grid = []  # reset for each grid, for each outer_grid the bound is a list of min, max
                # values per dimension
                mask = torch.ones(len(outer_tensor), dtype=torch.bool)  # mask includes all points first
                for d, direction in enumerate(pattern):
                    min_d, max_d = shell[d]
                    if direction == -1:  # if direction is -1 then it is defined as less than min_core in that dimension
                        mask &= outer_tensor[:, d] < min_d
                        bound_per_grid.append([-float('inf'), torch.clone(min_d)])
                    elif direction == 0:  # same, if direction is 0 then it is defined as inside core in that dim,
                        # we have to do this to ensure the grids are rectangular & disjoint -> aligned with the core!
                        mask &= (outer_tensor[:, d] >= min_d) & (outer_tensor[:, d] <= max_d)
                        bound_per_grid.append([torch.clone(min_d), torch.clone(max_d)])
                    elif direction == 1:  # same, if direction is 1 then it is defined as greater than max_core in that
                        # dimension
                        mask &= outer_tensor[:, d] > max_d
                        bound_per_grid.append([torch.clone(max_d), float('inf')])

                if mask.any():  # if mask has any points that match pattern than make a grid and store it! Should be
                    # 3^D -1 grids (exclude the core)
                    # print(f"\n Pattern: {pattern}")
                    # print(f'Number of points in region: {len(outer_tensor[mask])}')
                    # print("Bound per dimension:")
                    # for i, b in enumerate(bound_per_grid):
                    #     print(f"  Dim {i}: {b}")

                    grid = Grid.from_points(outer_tensor[mask], bounds=bound_per_grid)
                    outer_grids.append(grid)
                    bounds.append(bound_per_grid)

                    outer_lvd, outer_uvd = grid._compute_voronoi_edges(bounds=bound_per_grid)

                    # print("Voronoi Edges (lower, upper):")
                    # for dim in range(len(outer_lvd)):
                    #     print(f"  Dim {dim}:")
                    #     print(f"    Lower: {outer_lvd[dim]}")
                    #     print(f"    Upper: {outer_uvd[dim]}")

        return outer_grids, bounds

    def unify_grid(self, core_grid, outer_grids, shell, bounds):
        """
        Combine grids to one unified grid
        """
        dim = self.dim
        locs_per_dim = self.locs_per_dim

        # all vertices
        lower_vertices_per_dim = [[] for _ in range(dim)]
        upper_vertices_per_dim = [[] for _ in range(dim)]

        # core vertices
        if core_grid is not None:
            lvd_core, uvd_core = core_grid._compute_voronoi_edges(bounds=shell)  # bounded by shell
            for d in range(dim):
                lower_vertices_per_dim[d].append(lvd_core[d])
                upper_vertices_per_dim[d].append(uvd_core[d])

        # outer vertices
        for i, outer_grid in enumerate(outer_grids):
            lvd, uvd = outer_grid._compute_voronoi_edges(bounds=bounds[i])  # bounded by bounds found due to core shell
            for d in range(dim):
                lower_vertices_per_dim[d].append(lvd[d])
                upper_vertices_per_dim[d].append(uvd[d])

        # merge all
        final_lower = []
        final_upper = []
        for d in range(dim):
            lower_d = torch.cat(lower_vertices_per_dim[d])
            upper_d = torch.cat(upper_vertices_per_dim[d])
            final_lower.append(torch.sort(torch.unique(lower_d)).values)
            final_upper.append(torch.sort(torch.unique(upper_d)).values)

        return PartitionedGrid(
            locs_per_dim=locs_per_dim,
            lower_vertices_per_dim=final_lower,
            upper_vertices_per_dim=final_upper
        )  # new partitioned grid!

    def plot_shell_2d(self, unified_grid, shell):
        """shell is input shell"""
        lower_vertices_per_dim, upper_vertices_per_dim = self._compute_voronoi_edges()
        shell_tensor, core_tensor, outer_tensor, core_grid, outer_grids, bounds = self.shell(shell)
        core_lower_vertices_per_dim, core_upper_vertices_per_dim = core_grid._compute_voronoi_edges(bounds=shell)

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        ax.scatter(core_tensor[:, 0], core_tensor[:, 1], color='blue', label='Core', alpha=0.6)
        # ax.scatter(outer_tensor[:, 0], outer_tensor[:, 1], color='orange', label='Outer', alpha=0.6)

        cmap = cm.get_cmap('tab10')  # or 'Set1', 'tab20', etc.

        for idx, grid in enumerate(outer_grids):
            outer_locs = grid.get_locs()
            color = cmap(idx % 10)  # tab10 has 10 unique colors
            ax.scatter(outer_locs[:, 0], outer_locs[:, 1], color=color, alpha=0.6)

        # original partitioning
        for i in range(len(lower_vertices_per_dim[0])):
            for j in range(len(upper_vertices_per_dim[1])):
                x0 = lower_vertices_per_dim[0][i].item()
                x1 = upper_vertices_per_dim[0][i].item()
                y0 = lower_vertices_per_dim[1][j].item()
                y1 = upper_vertices_per_dim[1][j].item()

                rect = patches.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    edgecolor='yellow',
                    facecolor='none',
                    linewidth=1.2,
                    linestyle='-'
                )
                ax.add_patch(rect)

        # unified grid partitions
        lower_vertices = unified_grid.lower_vertices_per_dim
        upper_vertices = unified_grid.upper_vertices_per_dim
        for i in range(len(lower_vertices[0])):
            for j in range(len(lower_vertices[1])):
                x0 = lower_vertices[0][i].item()
                x1 = upper_vertices[0][i].item()
                y0 = lower_vertices[1][j].item()
                y1 = upper_vertices[1][j].item()

                rect = patches.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    edgecolor='purple',
                    facecolor='none',
                    linewidth=1.2,
                    linestyle='-'
                )
                ax.add_patch(rect)

        # core partitioning
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

        ax.set_title('Core of Shell vs Outer-Region Points in 2D')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True)
        plt.show()

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

class PartitionedGrid(Grid):
    def __init__(self, locs_per_dim: list, lower_vertices_per_dim: list, upper_vertices_per_dim: list):
        self.lower_vertices_per_dim = lower_vertices_per_dim
        self.upper_vertices_per_dim = upper_vertices_per_dim
        super().__init__(locs_per_dim)  # as its subclass of Grid, it initializes other attributes using init from Grid

    def _compute_voronoi_edges(self, bounds=None):  # during init it calls the original function but this overrides it
        # the function from Grid
        return self.lower_vertices_per_dim, self.upper_vertices_per_dim
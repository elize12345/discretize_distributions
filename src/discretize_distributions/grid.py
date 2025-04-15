import torch
from typing import Union
from matplotlib import pyplot as plt, patches


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
        lower_vertices_per_dim, upper_vertices_per_dim = self._compute_voronoi_edges()
        # boundary = [
        #     (min_x, max_x),  # for x-dimension (dim=0)
        #     (min_y, max_y)  # for y-dimension (dim=1)
        # ]
        for dim in range(len(shell)):
            min_shell, max_shell = shell[dim]
            lower = lower_vertices_per_dim[dim]
            upper = upper_vertices_per_dim[dim]

            # index for closest edge to input shell
            lower_index = torch.argmin(torch.abs(lower - min_shell))
            upper_index = torch.argmin(torch.abs(upper - max_shell))

            vor_shell_min = lower[lower_index]
            vor_shell_max = upper[upper_index]
            vor_shell.append((vor_shell_min, vor_shell_max))

            # checking
            min_matches = torch.any(torch.isclose(lower, vor_shell_min, atol=1e-6))
            max_matches = torch.any(torch.isclose(upper, vor_shell_max, atol=1e-6))

            if not (min_matches and max_matches):
                print(f"Dimension {dim}: shell ({min_shell.item()}, {max_shell.item()}) not aligned with Voronoi edges.")

        return vor_shell

    def shell(self, shell):
        """Computes the outer and core points based on input shell
        param: input shell list of tensors for max, min values per dim of wanted shell
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
                    is_shell = True
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

        return shell_tensor, core_tensor, outer_tensor

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def plot_shell_2d(self, shell):
        """shell is input shell"""
        # Get Voronoi edges
        lower_vertices_per_dim, upper_vertices_per_dim = self._compute_voronoi_edges()

        # Get shell/core/outer tensors using your custom snapping logic
        shell_tensor, core_tensor, outer_tensor = self.shell(shell)

        vor_shell = self.voronoi_edge_shell(shell)

        # Start figure
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # Plot core and outer points
        ax.scatter(core_tensor[:, 0], core_tensor[:, 1], color='blue', label='Core', alpha=0.6)
        ax.scatter(outer_tensor[:, 0], outer_tensor[:, 1], color='orange', label='Outer', alpha=0.6)

        # Plot all Voronoi-aligned cells as faint boxes
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

        # Plot the snapped Voronoi-aligned shell boundary box in red
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

        # Final formatting
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

import torch
from typing import Union

from discretize_distributions.distributions.multivariate_normal import MultivariateNormal
from discretize_distributions.distributions.categorical_float import CategoricalFloat
from discretize_distributions.distributions.mixture import MixtureMultivariateNormal
from discretize_distributions.discretize import discretize_multi_norm_dist
from discretize_distributions.grid import Grid

__all__ = ['DiscretizedMultivariateNormal',
           'DiscretizedMixtureMultivariateNormal',
           'DiscretizedMixtureMultivariateNormalQuantization',
           'DiscretizedCategoricalFloat',
           'discretization_generator'
           ]


class Discretization(CategoricalFloat):
    def __init__(self,
                 dist: torch.distributions.Distribution,
                 probs: torch.Tensor,
                 locs: torch.Tensor,
                 w2: torch.Tensor):
        self.dist = dist
        self.num_locs = probs.shape[-1]
        self.w2 = w2
        super().__init__(probs, locs)


class DiscretizedMultivariateNormal(Discretization):
    def __init__(self, norm: Union[MultivariateNormal, torch.distributions.MultivariateNormal], num_locs: int):
        assert isinstance(norm, (MultivariateNormal, torch.distributions.MultivariateNormal)), 'distribution not of type MultivariateNormal'

        locs, probs, w2 = discretize_multi_norm_dist(norm, num_locs)

        super().__init__(norm, probs, locs, w2)


class DiscretizedMixtureMultivariateNormal(Discretization):
    def __init__(self, gmm: MixtureMultivariateNormal, num_locs: int, **kwargs):
        assert isinstance(gmm, MixtureMultivariateNormal), 'distribution not of type MixtureMultivariateNormal'

        disc_component_distribution = discretization_generator(gmm.component_distribution, num_locs)

        # Combine the probs, locs and w2 for all components, accounting for their relative weights
        probs = torch.einsum('...ms,...m->...ms',
                             disc_component_distribution.probs,
                             gmm.mixture_distribution.probs)
        probs = probs.flatten(start_dim=-2)
        locs = disc_component_distribution.locs
        locs = locs.reshape(locs.shape[:-3] + (locs.shape[-3:-1].numel(), locs.shape[-1]))

        w2 = torch.einsum('...m,...m->...',
                          gmm.mixture_distribution.probs,
                          disc_component_distribution.w2.pow(2)).sqrt()

        super().__init__(gmm, probs, locs, w2)

class DiscretizedMixtureMultivariateNormalQuantization(Discretization):
    def __init__(self, gmm: MixtureMultivariateNormal, grid: Grid, **kwargs):
        assert isinstance(gmm, MixtureMultivariateNormal), 'distribution not of type MixtureMultivariateNormal'

        probs_mix = gmm.mixture_distribution.probs
        locs = grid.get_locs()  # [num_locs, dim]
        probs = torch.zeros(locs.shape[0])  # [num_locs,]
        locs_p = gmm.component_distribution.loc
        cov_p = gmm.component_distribution.covariance_matrix
        w2 = torch.zeros(1)

        for p in range(len(probs_mix)):
            component_p = MultivariateNormal(
                loc=locs_p[p],
                covariance_matrix=cov_p[p]
            )
            _, probs_p, w2_p = discretize_multi_norm_dist(component_p, None, grid)

            weight_p = probs_mix[p]
            probs += probs_p * weight_p
            w2 += w2_p.pow(2) * weight_p

        # batched version ?
        # probs_mix = gmm.mixture_distribution.probs   # [num_components,]
        # locs = grid.get_locs()  # [num_locs, dim]
        # _, probs_p, w2_p = discretize_multi_norm_dist(gmm.component_distribution, None, grid)
        # # probs_p shape [num_components, num_locs]
        # # w2_p shape [num_components, num_locs]
        # probs = torch.einsum('m,mn->n', probs_mix, probs_p)  # [num_locs]
        # w2 = torch.einsum('m,mn->n', probs_mix, w2_p)  # [num_locs]

        super().__init__(gmm, probs, locs, w2.sqrt())


class DiscretizedCategoricalFloat(Discretization):
    def __init__(self, dist: CategoricalFloat, num_locs: int):
        if num_locs <= dist.num_components:
            raise NotImplementedError
        super().__init__(dist, dist.probs, dist.locs, torch.zeros(dist.batch_shape))


class DiscretizationGenerator:
    def __call__(self, dist, *args, **kwargs):
        """

        :param dist:
        :param num_locs:
        :return:
        """
        if isinstance(dist, (MultivariateNormal, torch.distributions.MultivariateNormal)):
            return DiscretizedMultivariateNormal(dist, *args, **kwargs)
        elif type(dist) is MixtureMultivariateNormal:
            return DiscretizedMixtureMultivariateNormal(dist, *args, **kwargs)
        elif isinstance(dist, CategoricalFloat):
            return DiscretizedCategoricalFloat(dist, *args, **kwargs)
        else:
            raise NotImplementedError

discretization_generator = DiscretizationGenerator()



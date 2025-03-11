import torch
from torch.utils.hipify.hipify_python import InputError

from discretize_distributions.distributions.multivariate_normal import MultivariateNormal
from discretize_distributions.distributions.categorical_float import CategoricalFloat
from discretize_distributions.distributions.mixture import MixtureMultivariateNormal
from discretize_distributions.discretize import discretize_multi_norm_dist

__all__ = ['DiscretizedMultivariateNormal',
           'DiscretizedMixtureMultivariateNormal',
           'discretization_generator'
           ]


class DiscretizedMultivariateNormal(CategoricalFloat):
    def __init__(self, norm: MultivariateNormal, **kwargs):
        if not isinstance(norm, MultivariateNormal):
            raise ValueError('distribution not of type MultivariateNormal')

        self.dist = norm
        locs, probs, self.w2 = discretize_multi_norm_dist(norm=norm, **kwargs)

        self.nr_signature_points_realized = probs.shape[-1]   # \todo rename num_locs

        super().__init__(probs, locs)


class DiscretizedMixtureMultivariateNormal(CategoricalFloat):

    def __init__(self, gmm: MixtureMultivariateNormal, **kwargs):
        if not isinstance(gmm, MixtureMultivariateNormal):
            raise ValueError('distribution not of type MixtureMultivariateNormal')
        discretized_component_distribution = discretization_generator(dist=gmm.component_distribution,
                                                                      **kwargs)

        probs = torch.einsum('...ms,...m->...ms', discretized_component_distribution.probs,
                             gmm.mixture_distribution.probs)
        probs = probs.flatten(start_dim=-2)
        locs = discretized_component_distribution.locs
        locs = locs.reshape(locs.shape[:-3] + (locs.shape[-3:-1].numel(), locs.shape[-1]))
        if discretized_component_distribution.w2 is not None:
            self.w2 = torch.einsum('...m,...m->...', gmm.mixture_distribution.probs,
                                  discretized_component_distribution.w2.pow(2)).sqrt()
        else:
            self.w2 = None
        self.nr_signature_points_realized = discretized_component_distribution.probs.shape[-1]

        super().__init__(probs, locs)


class DiscretizationGenerator:
    def __call__(self, dist, num_locs: int = None, nr_signature_points: int = None,
                            compute_w2: bool=True, **kwargs):
        """

        :param dist:
        :param num_locs:
        :param nr_signature_points: To be replaced by num_locs. Preserved to guarantee compatibility with old code.
        :param compute_w2: Redundancy of old approach
        :return:
        """
        if nr_signature_points is None and num_locs is None:
            raise InputError('Specify num_locs')
        elif num_locs is None:
            num_locs = nr_signature_points

        if type(dist) is MultivariateNormal:
            return DiscretizedMultivariateNormal(dist, num_locs=num_locs, **kwargs)
        elif type(dist) is MixtureMultivariateNormal:
            return DiscretizedMixtureMultivariateNormal(dist, num_locs=num_locs, **kwargs)
        elif isinstance(dist, CategoricalFloat):
            return dist
        else:
            raise NotImplementedError

discretization_generator = DiscretizationGenerator()



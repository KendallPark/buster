from typing import Callable

from IPython import embed

import pandas as pd
#import numpy.typing as npt

from buster.metrics import gowers
from skopt import sampler, optimizer

import skopt.space as sp

from typing import Optional, Text, Union

from sklearn.utils import check_random_state

from sklearn import neighbors

import numpy as np


# TODO: refactor to depend on a mixin--not a subclass of optimizers
class AdaptiveSampler(optimizer.Optimizer):

  def __init__(self,
               dimensions,
               base_estimator="dummy",
               n_random_starts=None,
               n_initial_points=None,
               initial_point_generator="lhs",
               n_jobs=1,
               acq_func="gp_hedge",
               acq_optimizer="auto",
               random_state=None,
               model_queue_size=None,
               acq_func_kwargs=None,
               acq_optimizer_kwargs=None):

    if n_initial_points is None:
      n_initial_points = len(dimensions) * 10

    super().__init__(dimensions,
                     base_estimator=base_estimator,
                     n_random_starts=n_random_starts,
                     n_initial_points=n_initial_points,
                     initial_point_generator=initial_point_generator,
                     n_jobs=n_jobs,
                     acq_func=acq_func,
                     acq_optimizer=acq_optimizer,
                     random_state=random_state,
                     model_queue_size=model_queue_size,
                     acq_func_kwargs=acq_func_kwargs,
                     acq_optimizer_kwargs=acq_optimizer_kwargs)

  def ask(self, n_points=None, strategy="ldn"):

    if len(self.Xi) == 0 and self._initial_samples is not None:
      return self._initial_samples
    elif len(self.Xi) == 0:
      raise ValueError("need initial points to do inference on")

    if n_points is None:
      n_points = self.space.n_dims * 10

    supported_strategies = ["ldn"]

    # embed()

    if not (isinstance(n_points, int) and n_points > 0):
      raise ValueError("n_points should be int > 0, got " + str(n_points))

    if strategy not in supported_strategies:
      raise ValueError("Expected parallel_strategy to be one of " +
                       str(supported_strategies) + ", " + "got %s" % strategy)

    precomputed_dists = gowers.gowers_distance(self.Xi, self.Xi, self.space)

    # TODO: cache distance calculations so they don't repeat
    # self.cache_ = {len(self.Xi): precomputed_dists}  # cache_ the result

    neighborhood = k_largest_diverse_neighborhood(
        precomputed_dists, self.yi, n_neighbors=self.space.n_dims * 2, k=1)[0]

    original_shape = np.shape(neighborhood)

    neighbors = np.array(self.Xi)[neighborhood.flatten()]

    # TODO: find cleaner way to do this
    neighbors[:, self._cat_inds] = 0
    neighbors = neighbors.astype(float)

    mins = neighbors.min(axis=0)
    maxes = neighbors.max(axis=0)

    new_dimensions = []
    for index, dimension in enumerate(self.space.dimensions):
      if isinstance(dimension, sp.Categorical):
        new_dimensions.append(dimension)
        continue
      new_dimensions.append(dimension.__class__(mins[index], maxes[index]))

    return self._initial_point_generator.generate(new_dimensions,
                                                  n_points,
                                                  random_state=self.rng)


def k_largest_diverse_neighborhood(X,
                                   y,
                                   n_neighbors: int = 1,
                                   k: int = 1,
                                   metric: Text = "precomputed",
                                   threshold: float = 0.01):
  # n_neighbors = np.shape(X)[0]/
  if metric == "precomputed":
    nn = neighbors.NearestNeighbors(n_neighbors=n_neighbors,
                                    metric="precomputed")
    nn.fit(X)
    neigh_dist, neigh_ind = nn.kneighbors()

    y = np.array(y)

    original_shape = np.shape(neigh_ind)

    neigh_labels = y[neigh_ind.flatten()].reshape(original_shape)

    diversity = (y[..., np.newaxis] ^
                 neigh_labels.astype(int)).sum(axis=1) / (n_neighbors + 1)

    # average_distance = np.mean(neigh_dist, axis=1)
    max_distance = np.max(neigh_dist, axis=1)

    max_distance[max_distance < threshold] = 0

    score = diversity * max_distance

    top_indices = np.argsort(score)[-k:][::-1]

    return np.append(top_indices[..., np.newaxis], neigh_ind[top_indices], 1)
  else:
    # TODO: refactor with BallTree, waiting on
    # https://github.com/scikit-learn/scikit-learn/pull/16834
    raise NotImplementedError

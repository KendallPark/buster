import numpy.typing as npt
import numpy as np
from pandas.core.arrays import categorical
from scipy import spatial
import pandas as pd
from IPython import embed
import pyDOE as pydoe
from buster import data
from typing import Optional, Text
from skopt import sampler, space
from sklearn import neighbors
import collections

def lhs_scaled(space:data.Space, num_samples:int, criterion:Optional[Text]=None, random_state:Optional[int]=None):
  """Latin hypercube sampling scaled to DatasetSchema"""
  lhs = sampler.Lhs(lhs_type="classic", criterion=criterion)
  design = lhs.generate(space.dimensions, num_samples, random_state)
  return design
  

def intersite_proj_th(p:npt.ArrayLike, samples:npt.ArrayLike, space:data.Space, alpha:float=0.5) -> float:
  """Perform MIPT sampling.

  Crombecq, K., Laermans, E. & Dhaene, T. Efficient space-filling and 
  non-collapsing sequential design strategies for simulation-based modeling. 
  Eur J Oper Res 214, 683–696 (2011).
  """
  d_min = (2 * alpha)/ samples.shape[0]
  gowers_matrix = space.gowers_matrix(p, np.array(samples))
  result = np.linalg.norm(gowers_matrix, ord=np.NINF, axis=1).min()
  # This could present an issue with categorical variables
  # embed()
  if result < d_min:
    return 0
  return np.linalg.norm(gowers_matrix, ord=2, axis=1).min()


def intersite_proj(p:npt.ArrayLike, samples:npt.ArrayLike, space:data.Space) -> float:
  """Perform MIP sampling. (No threshold.)

  Crombecq, K., Laermans, E. & Dhaene, T. Efficient space-filling and 
  non-collapsing sequential design strategies for simulation-based modeling. 
  Eur J Oper Res 214, 683–696 (2011).
  """
  num_dim = space.n_dims
  num_samples = samples.shape[0]
  gowers_matrix = space.gowers_matrix(p, np.array(samples))
  l2_norm_min = np.linalg.norm(gowers_matrix, ord=2, axis=1).min()
  min_norm_min = np.linalg.norm(gowers_matrix, ord=np.NINF, axis=1).min()

  coeff_1 = ((num_samples+1)**(1/num_dim) - 1)/2
  coeff_2 = (num_samples+1)/2
  return coeff_1 * l2_norm_min + coeff_2 * min_norm_min

def label_factor(y:npt.ArrayLike):
  counts = collections.Counter(y)
  return ((np.array([counts[label] for label in y]) * -1) + len(y))/len(y)


def voronoi_volume_fractions(tree:data.GowerBallTree, candidates:npt.ArrayLike):
  data, _, _, _ = tree.get_arrays()
  volumes = np.zeros(len(data))
  _, nn_candidates = tree.query(candidates, k=1)
  nn_candidates = nn_candidates.flatten()
  for i in nn_candidates:
    volumes[i] += 1

  return volumes/len(nn_candidates)


def neighbor_score(X:npt.ArrayLike, y:npt.ArrayLike, tree: data.GowerBallTree, k:Optional[int]=None):
  if k is None:
    n_dim = X.shape[1]
    k = min(2*n_dim+1, len(X))
  _, nn_indices = tree.query(X, k=k)
  neighbor_scores = np.zeros(len(X))
  for i in range(len(nn_indices)):
    counts = collections.Counter([y[j] for j in nn_indices[i]])
    neighbor_scores[i] = k - counts[y[i]]
  return neighbor_scores


def voronoi_rankings(X, y, space, candidates:Optional[npt.ArrayLike]=None, criterion:Optional[Text]=None, random_state:Optional[int]=None):
  if candidates is None:
    sample_size = 500 * X.shape[0] * X.shape[1]
    lhs_scaled(space, sample_size, criterion, random_state)
  
  tree = data.GowerBallTree(X, space)
  vol = voronoi_volume_fractions(tree, candidates)
  ns = neighbor_score(X, y, tree)
  lf = label_factor(y)
  return vol * ns * lf

def _space_filling_metric(X, tree):
  nn_dists, _ = tree.query(X, k=2)
  max_dist = max([dist[1] for dist in nn_dists])
  return 0.5 * max_dist


# def mivor_accept(candidate, v_max, X, tree):
#   t = _space_filling_metric(X, tree)
#   space = tree. 



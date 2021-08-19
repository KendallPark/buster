"""Functions to Gowers distance between sets of points."""
import numpy as np
import numpy.typing as npt
from skopt import space as sp


def gowers_distance(X: npt.ArrayLike, Y: npt.ArrayLike, space: sp.Space):
  """Compute gower distance between two lists of points in space.

  Parameters
  ----------
  X : array of arrays
      First list of points.

  Y : array of arrays
      Second list of points.
  """
  difference_matrix = gowers_difference(X, Y, space)

  distance = difference_matrix.sum(axis=-1) / space.n_dims

  return distance


def gowers_difference(X: npt.ArrayLike, Y: npt.ArrayLike, space: sp.Space):
  space.set_transformer_by_type("normalize", sp.space.Real)
  space.set_transformer_by_type("normalize", sp.space.Integer)
  space.set_transformer_by_type("label", sp.space.Categorical)

  X = space.transform(X)
  Y = space.transform(Y)

  diff = np.abs(X[:, None, :] - Y[None, :, :])

  cat_cols = [isinstance(dim, sp.Categorical) for dim in space.dimensions]

  diff[:, :, cat_cols] = diff[:, :, cat_cols].astype(bool).astype(int)

  return diff

from typing import Union, List, Any, Optional, Text, Dict
import pandas as pd
import numpy as np
import numpy.typing as npt
from IPython.terminal.embed import embed

from skopt import space as sp
from sklearn import neighbors


class Integer(sp.space.Integer):

  def __init__(self,
               low: int,
               high: int,
               prior: Optional[Text] = "uniform",
               base: int = 10,
               transform: Optional[Text] = "normalize",
               name: Optional[Text] = None,
               dtype: npt.DTypeLike = np.int64) -> None:

    super().__init__(low, high, prior, base, transform, name, dtype)

  def transform_distance(self, a, b):
    if not (a in self and b in self):
      raise RuntimeError("Can only compute distance for values within "
                         "the space, not %s and %s." % (a, b))
    return abs(self.transform(a) - self.transform(b))

  def inverse_transform_distance(self, a, b):
    inv_a = self.inverse_transform(a)
    inv_b = self.inverse_transform(b)
    if not (inv_a in self and inv_b in self):
      raise RuntimeError("Can only compute distance for values within "
                         "the space, not %s and %s." % (inv_a, inv_b))
    return abs(a - b)

  @classmethod
  def from_list(cls,
                values: List[int],
                prior: Optional[Text] = "uniform",
                base: int = 10,
                transform: Optional[Text] = "normalize",
                name: Optional[Text] = None,
                dtype: npt.DTypeLike = np.int64):
    return cls(min(values), max(values), prior, base, transform, name, dtype)


class Real(sp.space.Real):

  def __init__(self,
               low: float,
               high: float,
               prior: Optional[Text] = "uniform",
               base: int = 10,
               transform: Optional[Text] = "normalize",
               name: Optional[Text] = None,
               dtype: npt.DTypeLike = float) -> None:
    super().__init__(low, high, prior, base, transform, name, dtype)

  def transform_distance(self, a, b):
    if not (a in self and b in self):
      raise RuntimeError("Can only compute distance for values within "
                         "the space, not %s and %s." % (a, b))
    return abs(self.transform(a) - self.transform(b))

  def inverse_transform_distance(self, a, b):
    inv_a = self.inverse_transform(a)
    inv_b = self.inverse_transform(b)
    if not (inv_a in self and inv_b in self):
      raise RuntimeError("Can only compute distance for values within "
                         "the space, not %s and %s." % (inv_a, inv_b))
    return abs(a - b)

  @classmethod
  def from_list(cls,
                values: List[float],
                prior: Optional[Text] = "uniform",
                base: int = 10,
                transform: Optional[Text] = "normalize",
                name: Optional[Text] = None,
                dtype: npt.DTypeLike = float):
    return cls(min(values), max(values), prior, base, transform, name, dtype)


class Categorical(sp.space.Categorical):

  def __init__(self,
               categories: List[Any],
               prior: Optional[List[float]] = None,
               transform: Optional[Text] = "label",
               name: Optional[Text] = None):
    super().__init__(categories, prior, transform, name)

  def transform(self, X):
    """Transform samples form the original space to a warped space."""
    if np.isscalar(X):
      X = [X]
      return super().transform([X])[0]
    return super().transform(X)

  def inverse_transform(self, Xt):
    """Inverse transform samples from the warped space back into the
        original space.
    """
    if np.isscalar(Xt):
      return super().inverse_transform([Xt])[0]
    return super().inverse_transform(Xt)

  def transform_distance(self, a, b):
    return self.distance(a, b)

  def inverse_transform_distance(self, a, b):
    return self.distance(self.inverse_transform(a), self.inverse_transform(b))

  @classmethod
  def from_list(cls,
                values: List[Any],
                prior: Optional[List[float]] = None,
                transform: Optional[Text] = "label",
                name: Optional[Text] = None):
    # Removes duplicates while preserving order
    categories = list(dict.fromkeys(values))
    return cls(categories, prior, transform, name)


class Space(sp.space.Space):

  def _categorical_columns(self):
    return [isinstance(dim, sp.space.Categorical) for dim in self.dimensions]

  def _numerical_columns(self, rows: Optional[int] = None):
    return [
        isinstance(dim, sp.space.Real) or isinstance(dim, sp.space.Integer)
        for dim in self.dimensions
    ]

  def transform(self, X):
    if isinstance(X, pd.DataFrame):
      X = X.to_numpy()
    elif len(np.shape(X)) == 1:
      X = [X]
    return super().transform(X)

  def gowers_distance(self, X: npt.ArrayLike, Y: npt.ArrayLike, ord=None):
    """Compute gower distance between two points in this space.

    Parameters
    ----------
    X : array
        First point.

    Y : array
        Second point.
    """
    difference_matrix = self.gowers_difference(X, Y)

    distance = difference_matrix.sum(axis=-1) / self.n_dims

    return np.squeeze(distance)

  def gowers_difference(self, X: npt.ArrayLike, Y: npt.ArrayLike):
    X = self.transform(X)
    Y = self.transform(Y)

    diff = np.abs(X[:, None, :] - Y[None, :, :])

    cat_cols = self._categorical_columns()

    diff[:, :, cat_cols] = diff[:, :, cat_cols].astype(bool).astype(int)

    return np.squeeze(diff)

  def inverse_transform_gowers_distance(self, point_a, point_b):
    total_distance = 0.
    for a, b, dim in zip(point_a, point_b, self.dimensions):
      total_distance += dim.inverse_transform_distance(a, b)
    return total_distance / self.n_dims

  @classmethod
  def infer_dimension(cls, series: pd.Series) -> sp.space.Dimension:
    dtype = series.dtype
    if pd.api.types.is_float_dtype(dtype):
      return Real
    elif pd.api.types.is_integer_dtype(dtype):
      return Integer
    elif pd.api.types.is_bool_dtype(dtype):
      return Integer
    else:
      return Categorical

  @classmethod
  def from_df(cls,
              df: pd.DataFrame,
              cat_cols: Optional[List[Text]] = None,
              int_cols: Optional[List[Text]] = None,
              real_cols: Optional[List[Text]] = None,
              priors: Optional[Dict[Text, Union[Text, float]]] = None,
              bases: Optional[Dict[Text, int]] = None):
    """Create DatasetSchema from a Pandas DataFrame"""
    if cat_cols is None:
      cat_cols = []
    if real_cols is None:
      real_cols = []
    if int_cols is None:
      int_cols = []
    if priors is None:
      priors = {}
    if bases is None:
      bases = {}

    cat_set = set(cat_cols)
    real_set = set(real_cols)
    int_set = set(int_cols)

    dimensions = []

    for col_name in df.columns:

      feat_cls = None

      if col_name in cat_set:
        feat_cls = Categorical
      elif col_name in real_set:
        feat_cls = Real
      elif col_name in int_set:
        feat_cls = Integer

      feat_cls = feat_cls or cls.infer_dimension(df[col_name])

      kwargs = {}

      if col_name in priors:
        kwargs['prior'] = priors[col_name]

      if col_name in bases:
        kwargs['base'] = bases[col_name]

      dim_name = str(col_name) if col_name is not None else None

      dimension = feat_cls.from_list(df[col_name], name=dim_name, **kwargs)

      dimensions.append(dimension)

    return Space(dimensions)

  @classmethod
  def from_csv(cls,
               filepath_or_buffer,
               skipinitialspace: bool = True,
               **kwargs):
    """Create DatasetSchema from a csv."""
    return cls.from_df(
        pd.read_csv(filepath_or_buffer, skipinitialspace=skipinitialspace),
        **kwargs)


class GowerBallTree(neighbors.BallTree):

  def __init__(self,
               X: npt.ArrayLike,
               space: Space,
               leaf_size: int = 40,
               **kwargs) -> None:
    self._space = space
    Xt = self._space.transform(np.array(X))
    super().__init__(Xt,
                     leaf_size,
                     metric='pyfunc',
                     func=space.inverse_transform_gowers_distance)

  # TODO: turn this into a wrapper
  def query(self,
            X,
            k=1,
            return_distance=True,
            dualtree=False,
            breadth_first=False,
            sort_results=True):
    if isinstance(X, pd.DataFrame):
      X = X.to_numpy()
    Xt = self._space.transform(X)
    return super().query(Xt, k, return_distance, dualtree, breadth_first,
                         sort_results)

import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Union, List, Any, Optional, Text, Dict

from skopt import space

# DistType = Union[stats.rv_continuous, stats.rv_discrete]

class Integer(space.space.Integer):

  def __init__(self, low:int, high:int, prior:Optional[Text]="uniform", base:int=10, transform:Optional[Text]="normalize", name:Optional[Text]=None, dtype:npt.DTypeLike=np.int64) -> None:

    super().__init__(low, high, prior, base, transform, name, dtype)
  
  @classmethod
  def from_list(cls, values:List[int], prior:Optional[Text]="uniform", base:int=10, transform:Optional[Text]="normalize", name:Optional[Text]=None, dtype:npt.DTypeLike=np.int64):
    return cls(min(values), max(values), prior, base, transform, name, dtype)

class Real(space.space.Real):

  def __init__(self, low:float, high:float, prior:Optional[Text]="uniform", base:int=10, transform:Optional[Text]="normalize", name:Optional[Text]=None, dtype:npt.DTypeLike=float) -> None:
    super().__init__(low, high, prior, base, transform, name, dtype)
  
  @classmethod
  def from_list(cls, values:List[float], prior:Optional[Text]="uniform", base:int=10, transform:Optional[Text]="normalize", name:Optional[Text]=None, dtype:npt.DTypeLike=float):
    return cls(min(values), max(values), prior, base, transform, name, dtype)

class Categorical(space.space.Categorical):

  def __init__(self, categories:List[Any], prior:Optional[List[float]]=None, transform:Optional[Text]="label", name:Optional[Text]=None):
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
  
  @classmethod
  def from_list(cls, values:List[Any], prior:Optional[List[float]]=None, transform:Optional[Text]="label", name:Optional[Text]=None):
    # Removes duplicates while preserving order
    categories = list(dict.fromkeys(values))
    return cls(categories, prior, transform, name)

class Space(space.space.Space):
  @classmethod
  def infer_dimension(cls, series: pd.Series) -> space.space.Dimension:
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
  def from_df(cls, df: pd.DataFrame, cat_cols:Optional[List[Text]]=None, int_cols:Optional[List[Text]]=None, real_cols:Optional[List[Text]]=None, priors:Optional[Dict[Text, Union[Text, float]]]=None, bases:Optional[Dict[Text, int]]=None):
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

      dimension = feat_cls.from_list(df[col_name], name=col_name, **kwargs)

      dimensions.append(dimension)

    return Space(dimensions)
      
  
  @classmethod
  def from_csv(cls, filepath_or_buffer, skipinitialspace:bool=True, **kwargs):
    """Create DatasetSchema from a csv."""
    return cls.from_df(pd.read_csv(filepath_or_buffer, skipinitialspace=skipinitialspace), **kwargs)


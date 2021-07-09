import pandas as pd
import numpy as np
import numpy.typing as npt
from scipy import stats
from typing import Iterable, Union, List, Any, Optional, Text, Dict

DistType = Union[stats.rv_continuous, stats.rv_discrete]

class Feature:

  def __init__(self, dist:DistType, name:Text):
    self._dist = dist
    self._name = name
  
  def transform(self, x:npt.ArrayLike) -> npt.ArrayLike:
    """Transforms the 0,1-uniform distribution to the Feature's distribution."""
    return self._dist.ppf(x)

  @property
  def dist(self):
    return self._dist

  @property
  def name(self):
    return self._name

class Continuous(Feature):

  @classmethod
  def from_list(cls, values:List[Any], name:Optional[Text]=None):
    return Continuous.from_range(min(values), max(values), name)

  @classmethod
  def from_range(cls, low:float, high:float, name:Optional[Text]=None):
    dist = stats.uniform(loc=low, scale=high - low)
    return cls(dist, name)


class Discrete(Feature):

  @classmethod
  def from_list(cls, values:List[Any], name:Optional[Text]=None):
    return Discrete.from_range(min(values), max(values), name)

  @classmethod
  def from_range(cls, low:int, high:int, name:Optional[Text]=None):
    num_int = high-low+1
    dist = stats.rv_discrete(values=(np.arange(low, high+1), np.full(num_int, 1/(num_int))))
    return cls(dist, name)


class Categorical(Feature):

  @classmethod
  def from_list(cls, categories:List[Any], name:Optional[Text]=None):
    num_cat = len(set(categories))
    return Categorical.from_num(num_cat, name)

  @classmethod
  def from_num(cls, num:int, name:Optional[Text]=None):
    dist = stats.rv_discrete(values=(np.arange(num), np.full(num, 1/num)))
    return cls(dist, name)


class DatasetSchema:
  def __init__(self, features: Iterable[Feature]):
    self._features = features

  def __iter__(self):
   yield from self._features

  @property
  def features(self):
    return self._features

  @property
  def num_features(self):
    return len(self._features)

  @classmethod
  def infer_feature(cls, series: pd.Series) -> Feature:
    dtype = series.dtype
    if pd.api.types.is_float_dtype(dtype):
      return Continuous
    elif pd.api.types.is_integer_dtype(dtype):
      return Discrete
    elif pd.api.types.is_bool_dtype(dtype):
      return Discrete
    else:
      return Categorical

  @classmethod
  def from_df(cls, df: pd.DataFrame, cat_names:Optional[List[Text]]=None, cont_names:Optional[List[Text]]=None, disc_names:Optional[List[Text]]=None, dist_override:Optional[Dict[Text, DistType]]=None):
    """Create DatasetSchema from a Pandas DataFrame"""
    if cat_names is None:
      cat_names = []
    if cont_names is None:
      cont_names = []
    if disc_names is None:
      disc_names = []
    if dist_override is None:
      dist_override = {}
    
    cat_set = set(cat_names)
    cont_set = set(cont_names)
    disc_set = set(disc_names)

    features = []
    
    for col_name in df.columns:

      dtype = df[col_name].dtype

      feat_cls = None

      if col_name in cat_set:
        feat_cls = Categorical
      elif col_name in cont_set:
        feat_cls = Continuous
      elif col_name in disc_set:
        feat_cls = Discrete

      feat_cls = feat_cls or cls.infer_feature(df[col_name])

      if col_name in dist_override:
        feature = feat_cls(dist_override[col_name], col_name)
      else:
        feature = feat_cls.from_list(df[col_name], col_name)

      features.append(feature)

    return DatasetSchema(features)
      
  
  @classmethod
  def from_csv(cls, filepath_or_buffer, skipinitialspace:bool=True, **kwargs):
    """Create DatasetSchema from a csv."""
    return cls.from_df(pd.read_csv(filepath_or_buffer, skipinitialspace=skipinitialspace), **kwargs)


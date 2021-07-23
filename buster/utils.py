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
  


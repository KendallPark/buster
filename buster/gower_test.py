import unittest

import numpy as np
import pandas as pd
from buster import gower

from IPython import embed


class TestGower(unittest.TestCase):
  # Modified from https://github.com/wwwjk366/gower
  def test_matrix(self):
    Xd=pd.DataFrame({'age':[21,21,19, 30,21,21,19,30,None],
                     'gender':['M','M','N','M','F','F','F','F',None],
                     'civil_status':['MARRIED','SINGLE','SINGLE','SINGLE','MARRIED','SINGLE','WIDOW','DIVORCED',None],
                     'salary':[3000.0,1200.0 ,32000.0,1800.0 ,2900.0 ,1100.0 ,10000.0,1500.0,None],
                     'has_children':[1,0,1,1,1,0,0,1,None],
                     'available_credit':[2200,100,22000,1100,2000,100,6000,2200,None]})
    Yd = Xd.iloc[1:3,:]
    X = np.asarray(Xd)
    Y = np.asarray(Yd)
    result = gower.gower_matrix(X)
    self.assertAlmostEqual(result[0][1], 0.3590238)

  # Fix https://github.com/wwwjk366/gower/issues/2
  def test_only_integers(self):
    Xd=pd.DataFrame(
      {'age':[21, 21, 30],
      'available_credit': [2200, 100, 500]})

    result = gower.gower_matrix(Xd)
    first_vs_last_expected = ((30 - 21) / 9 + (2200 - 500) / 2100) / 2
    self.assertAlmostEqual(result[0][2], first_vs_last_expected)

  # Fix https://github.com/wwwjk366/gower/pull/1
  def test_range_zero(self):
    Xd=pd.DataFrame(
      {'age':[21, 21, 30],
      'available_credit': [-2200, -100, -500]})
    result = gower.gower_matrix(Xd)
    first_vs_last_expected = ((30 - 21) / 9 + (2200 - 500) / 2100) / 2
    self.assertAlmostEqual(result[0][2], first_vs_last_expected)






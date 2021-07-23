import unittest
from buster import utils, data
import numpy as np
import pandas as pd
from IPython import embed

SAMPLES_2 = np.array([[1, 2, 3],
                      [7, 8, 9],
                      [7, 8, 9]])

SPACE_2 = data.Space.from_df(pd.DataFrame(SAMPLES_2))


DF_1 = pd.DataFrame({ 'int': [-100, 0, 100], 
                      'float': [-100.0, 0.0, 100.0], 
                      'bool': [True, False, True],
                      'cat': ['cat', 'dog', 'rabbit'] })

SPACE_1 = data.Space.from_df(DF_1)

class TestLHS(unittest.TestCase):

  def test_lhs_scaled(self):
    scaled = utils.lhs_scaled(SPACE_1, 3, random_state=0)
    expected = [[98, -52.32070890850537, 1, 'rabbit'],
                [-5, 58.89610125505186, 0, 'cat'],
                [-63, 9.72627420444374, 0, 'dog']]
    self.assertEqual(scaled, expected)

    
class TestIntersiteProjTH(unittest.TestCase):

  def test_less_than_dmin_integer(self):
    result = utils.intersite_proj_th(np.array([1, 3, 7]), SAMPLES_2, SPACE_2)
    self.assertEqual(result, 0)

  def test_greater_than_dmin_integer(self):
    result = utils.intersite_proj_th(np.array([3, 4, 5]), SAMPLES_2, SPACE_2)
    self.assertEqual(result, ((1/3)**2 + (1/3)**2 + (1/3)**2)**0.5)

  def test_less_than_dmin_mixed(self):
    result = utils.intersite_proj_th([0, 0, True, 'cat'], DF_1, SPACE_1)
    self.assertEqual(result, 0)

  def test_greater_than_dmin_mixed(self):
    result = utils.intersite_proj_th([50, 50, False, 'cat'], DF_1, SPACE_1)
    self.assertEqual(result, 0)


class TestIntersiteProj(unittest.TestCase):

  def test_no_distance(self):
    samples = np.array([[0, 0, 0]])
    dimensions = [data.Integer(0, 10), data.Integer(0, 10), data.Integer(0, 10)]
    space = data.Space(dimensions)
    result = utils.intersite_proj(np.array([0, 0, 0]), samples, space)
    self.assertEqual(result, 0)
    
    num_samples = 1
    num_dim = 3
    coeff_1 = ((num_samples+1)**(1/num_dim) - 1)/2
    coeff_2 = (num_samples+1)/2
    l2_norm_min = (0.1**2 + 0.1**2 + 0.1**2)**(1/2)
    min_norm_min = 0.1
    expected_2 = coeff_1 * l2_norm_min + coeff_2 * min_norm_min
    result_2 = utils.intersite_proj(np.array([1, 1, 1]), samples, space)
    self.assertEqual(result_2, expected_2)

  def test_mixed_features(self):
    dimensions = [data.Integer(0, 10), data.Real(0, 10), data.Categorical(['cat', 'dog', 'rabbit'])]
    space = data.Space(dimensions)
    samples = pd.DataFrame([[0, 0.0, 'cat']])
    result = utils.intersite_proj([0, 0.0, 'cat'], samples, space)
    self.assertEqual(result, 0)
    
    num_samples = 1
    num_dim = 3
    coeff_1 = ((num_samples+1)**(1/num_dim) - 1)/2
    coeff_2 = (num_samples+1)/2
    l2_norm_min = (0.1**2 + 0.1**2 + 1**2)**(1/2)
    min_norm_min = 0.1
    expected_2 = coeff_1 * l2_norm_min + coeff_2 * min_norm_min
    result = utils.intersite_proj([1, 1.0, 'dog'], samples, space)
    self.assertEqual(result, expected_2)


if __name__ == '__main__':
  unittest.main()

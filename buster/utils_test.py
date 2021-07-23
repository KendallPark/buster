import unittest
from buster import utils, data
import numpy as np
import pandas as pd
from IPython import embed


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



if __name__ == '__main__':
  unittest.main()

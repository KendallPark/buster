import unittest

from buster import data
import numpy as np
import pandas as pd
from IPython import embed


class TestDimension(unittest.TestCase):

  def test_real_transform(self):
    dimension = data.Real(1, 10)
    self.assertEqual(dimension.inverse_transform(0), 1)
    self.assertEqual(dimension.inverse_transform(0.01), 1.09)
    self.assertEqual(dimension.inverse_transform(0.99), 9.91)
    np.testing.assert_array_equal(dimension.inverse_transform([0.5, 1]), np.array([5.5, 10]))

  def test_integer_transform(self):
    dimension = data.Integer(1, 10)
    self.assertEqual(dimension.inverse_transform(0), 1)
    self.assertEqual(dimension.inverse_transform(0.01), 1)
    self.assertEqual(dimension.inverse_transform(0.99), 10)
    np.testing.assert_array_equal(dimension.inverse_transform([0.5, 1]), np.array([6, 10]))

  def test_categorical_transform(self):
    dimension = data.Categorical.from_list(['cat', 'dog', 'rabbit', 'rabbit'])
    self.assertEqual(dimension.inverse_transform(1), 'dog')
    self.assertEqual(dimension.inverse_transform(0.01), 'cat')
    self.assertEqual(dimension.inverse_transform(0.99), 'dog')
    np.testing.assert_array_equal(dimension.inverse_transform([0.5, 1]), np.array(['cat', 'dog']))


class TestSpace(unittest.TestCase):
  def setUp(self) -> None:
    self._df = pd.DataFrame({ 'int': [-100, 0, 100], 
                              'float': [-100.0, 0.0, 100.0], 
                              'bool': [True, False, True],
                              'cat': ['cat', 'dog', 'rabbit'] })
    return super().setUp()
  
  def test_from_df(self):
    cat_cols = ['cat']
    real_cols = ['float', 'int']
    int_cols = ['bool']
    schema = data.Space.from_df(self._df, cat_cols=cat_cols, int_cols=int_cols, real_cols=real_cols)
    dimensions = [dimension for dimension in schema.dimensions]

    self.assertIsInstance(dimensions[0], data.Real)
    self.assertIsInstance(dimensions[1], data.Real)
    self.assertIsInstance(dimensions[2], data.Integer)
    self.assertIsInstance(dimensions[3], data.Categorical)

    self.assertEqual(dimensions[0].name, 'int')
    self.assertEqual(dimensions[1].name, 'float')
    self.assertEqual(dimensions[2].name, 'bool')
    self.assertEqual(dimensions[3].name, 'cat')

  def test_from_df_with_bases(self):
    schema = data.Space.from_df(self._df, bases={'int': 2})
    dimensions = [dimension for dimension in schema.dimensions]

    self.assertEqual(dimensions[0].base, 2)

  def test_from_df_with_priors(self):
    int_prior = 'log-uniform'
    cat_prior = [0.1, 0.4, 0.5]
    schema = data.Space.from_df(self._df, priors={'int': int_prior, 'cat': cat_prior})
    dimensions = [dimension for dimension in schema.dimensions]

    self.assertEqual(dimensions[0].prior, int_prior)
    self.assertEqual(dimensions[3].prior, cat_prior)

  def test_from_df_with_infer(self):
    schema = data.Space.from_df(self._df)
    dimensions = [dimension for dimension in schema]

    self.assertIsInstance(dimensions[0], data.Integer)
    self.assertIsInstance(dimensions[1], data.Real)
    self.assertIsInstance(dimensions[2], data.Integer)
    self.assertIsInstance(dimensions[3], data.Categorical)

  def test_infer_dimension(self):
    feat_float = data.Space.infer_dimension(pd.Series([1.2]))
    self.assertEqual(feat_float, data.Real)
    feat_int = data.Space.infer_dimension(pd.Series([1]))
    self.assertEqual(feat_int, data.Integer)
    feat_int = data.Space.infer_dimension(pd.Series([False]))
    self.assertEqual(feat_int, data.Integer)
    feat_cat = data.Space.infer_dimension(pd.Series(['cat']))
    self.assertEqual(feat_cat, data.Categorical)






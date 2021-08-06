import unittest

from buster import data
import numpy as np
import pandas as pd
from IPython import embed

DF_1 = pd.DataFrame({ 'int': [-100, 0, 100], 
                      'float': [-100.0, 0.0, 100.0], 
                      'bool': [True, False, True],
                      'cat': ['cat', 'dog', 'rabbit'] })

SPACE_1 = data.Space.from_df(DF_1)


class TestDimension(unittest.TestCase):

  def test_real_transform(self):
    dimension = data.Real(1, 10)
    self.assertEqual(dimension.inverse_transform(0), 1)
    self.assertEqual(dimension.inverse_transform(0.01), 1.09)
    self.assertEqual(dimension.inverse_transform(0.99), 9.91)
    np.testing.assert_array_equal(dimension.inverse_transform([0.5, 1]), np.array([5.5, 10]))

  def test_integer_transform(self):
    dimension = dodeata.Integer(1, 10)
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

  def test_transform_distance(self):
    dimension = data.Real(1, 10)
    self.assertEqual(dimension.transform_distance(1, 10), 1)
    self.assertEqual(dimension.transform_distance(1, 5.5), 0.5)

    dimension = data.Integer(1, 11)
    self.assertEqual(dimension.transform_distance(1, 11), 1)
    self.assertEqual(dimension.transform_distance(1, 6), 0.5)

    dimension = data.Categorical.from_list(['cat', 'dog', 'rabbit', 'rabbit'])
    self.assertEqual(dimension.transform_distance('cat', 'cat'), 0)
    self.assertEqual(dimension.transform_distance('cat', 'dog'), 1)


class TestSpace(unittest.TestCase):

  def test_gowers_distance_min(self):
    distance = SPACE_1.gowers_distance([0, 50, 0, 'cat'], [0, 50, 0, 'cat'])
    self.assertEqual(distance, 0)

  def test_gowers_distance_max(self):
    distance = SPACE_1.gowers_distance([100, 100, 0, 'cat'], [-100, -100, 1, 'dog'])
    self.assertEqual(distance, 1)

  def test_inverse_transform_gowers_distance_min(self):
    point_a = SPACE_1.transform([[0, 50, 0, 'cat']])[0]
    point_b = SPACE_1.transform([[0, 50, 0, 'cat']])[0]
    distance = SPACE_1.inverse_transform_gowers_distance(point_a, point_b)
    self.assertEqual(distance, 0)

  def test_inverse_transform_gowers_distance_max(self):
    point_a = SPACE_1.transform([[100, 100, 0, 'cat']])[0]
    point_b = SPACE_1.transform([[-100, -100, 1, 'dog']])[0]
    distance = SPACE_1.inverse_transform_gowers_distance(point_a, point_b)
    self.assertEqual(distance, 1)

  def test_gowers_matrix(self):
    point_a = [0, 0, 0, 'cat']
    point_b = [100, 0, 0, 'cat']
    point_c = [-100, 50, 1, 'dog']
    result = SPACE_1.gowers_distance(point_a, [point_b, point_c])
    expected = np.array([[0.5 , 0.  , 0.  , 0.  ], [0.5 , 0.25, 1.  , 1.  ]])
    np.testing.assert_array_equal(result, expected)

  def test_gowers_difference(self):
    point_a = [0, 0, 0, 'cat']
    point_b = [100, 0, 0, 'cat']
    point_c = [-100, 50, 1, 'rabbit']
    result = SPACE_1.gowers_difference([point_a, point_c], [point_b, point_c])
    expected = np.array([[[-0.5 ,  0.  ,  0.  ,  0.  ], 
                          [ 0.5 , -0.25, -1.  ,  1.  ]],
                         [[-1.  ,  0.25,  1.  ,  1.  ],
                          [ 0.  ,  0.  ,  0.  ,  0.  ]]])
    np.testing.assert_array_equal(result, expected)

  def test_from_df(self):
    cat_cols = ['cat']
    real_cols = ['float', 'int']
    int_cols = ['bool']
    schema = data.Space.from_df(DF_1, cat_cols=cat_cols, int_cols=int_cols, real_cols=real_cols)
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
    schema = data.Space.from_df(DF_1, bases={'int': 2})
    dimensions = [dimension for dimension in schema.dimensions]

    self.assertEqual(dimensions[0].base, 2)

  def test_from_df_with_priors(self):
    int_prior = 'log-uniform'
    cat_prior = [0.1, 0.4, 0.5]
    schema = data.Space.from_df(DF_1, priors={'int': int_prior, 'cat': cat_prior})
    dimensions = [dimension for dimension in schema.dimensions]

    self.assertEqual(dimensions[0].prior, int_prior)
    self.assertEqual(dimensions[3].prior, cat_prior)

  def test_from_df_with_infer(self):
    schema = data.Space.from_df(DF_1)
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

  # def test_scalar_and_categorical_matrices(self):
  #   result = SPACE_1._scalar_and_categorical_matrices(DF_1)

  def test_categorical_mask(self):
    result = SPACE_1._categorical_mask()
    expected = [True, True, True, False]
    self.assertEqual(result, expected)

  def test_numerical_mask(self):
    result = SPACE_1._numerical_mask()
    expected = [False, False, False, True]
    self.assertEqual(result, expected)
    


class TestGowerBallTree(unittest.TestCase):

  def test_gower_ball_tree(self):
    tree = data.GowerBallTree(DF_1, SPACE_1)
    dist, ind = tree.query(DF_1[:1], k=2)
    np.testing.assert_array_equal(dist, [[0., 0.75]])
    np.testing.assert_array_equal(ind, [[0, 1]])
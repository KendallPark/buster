import unittest
from buster import data
import numpy as np
import pandas as pd
from IPython import embed
from scipy import stats


class TestFeature(unittest.TestCase):

  def test_continuous_transform(self):
    feature = data.Continuous.from_range(1, 11)
    self.assertEqual(feature.transform(0), 1)
    self.assertEqual(feature.transform(0.01), 1.1)
    self.assertEqual(feature.transform(0.99), 10.9)
    np.testing.assert_array_equal(feature.transform([0.5, 1]), np.array([6.0, 11]))

  def test_discrete_transform(self):
    feature = data.Discrete.from_range(1, 10)
    self.assertEqual(feature.transform(0), 0)
    self.assertEqual(feature.transform(0.01), 1)
    self.assertEqual(feature.transform(0.99), 10)
    np.testing.assert_array_equal(feature.transform([0.5, 1]), np.array([5, 10]))

  def test_categorical_transform(self):
    feature = data.Categorical.from_list(['cat', 'dog', 'rabbit', 'rabbit'])
    self.assertEqual(feature.transform(0), -1.0)
    self.assertEqual(feature.transform(0.01), 0)
    self.assertEqual(feature.transform(0.99), 2)
    np.testing.assert_array_equal(feature.transform([0.5, 1]), np.array([1, 2]))


class TestDatasetSchema(unittest.TestCase):
  def setUp(self) -> None:
    self._df = pd.DataFrame({ 'int': [-100, 0, 100], 
                              'float': [-100.0, 0.0, 100.0], 
                              'bool': [True, False, True],
                              'cat': ['cat', 'dog', 'rabbit'] })
    return super().setUp()
  
  def test_from_df(self):
    cat_names = ['cat']
    cont_names = ['float', 'int']
    disc_names = ['bool']
    schema = data.DatasetSchema.from_df(self._df, cat_names=cat_names, cont_names=cont_names, disc_names=disc_names)
    features = [feature for feature in schema]

    self.assertIsInstance(features[0], data.Continuous)
    self.assertIsInstance(features[1], data.Continuous)
    self.assertIsInstance(features[2], data.Discrete)
    self.assertIsInstance(features[3], data.Categorical)

    self.assertEqual(features[0].name, 'int')
    self.assertEqual(features[1].name, 'float')
    self.assertEqual(features[2].name, 'bool')
    self.assertEqual(features[3].name, 'cat')

  def test_from_df_with_override(self):
    dist_custom = stats.rv_discrete(values=([1, 2], [0.2, 0.8]))
    dist_override = {'int': dist_custom}
    schema = data.DatasetSchema.from_df(self._df, dist_override=dist_override)
    features = [feature for feature in schema]

    self.assertEqual(features[0].dist, dist_custom)

  def test_from_df_with_infer(self):
    schema = data.DatasetSchema.from_df(self._df)
    features = [feature for feature in schema]

    self.assertIsInstance(features[0], data.Discrete)
    self.assertIsInstance(features[1], data.Continuous)
    self.assertIsInstance(features[2], data.Discrete)
    self.assertIsInstance(features[3], data.Categorical)

  def test_infer_feature(self):
    feat_float = data.DatasetSchema.infer_feature(pd.Series([1.2]))
    self.assertEqual(feat_float, data.Continuous)
    feat_int = data.DatasetSchema.infer_feature(pd.Series([1]))
    self.assertEqual(feat_int, data.Discrete)
    feat_int = data.DatasetSchema.infer_feature(pd.Series([False]))
    self.assertEqual(feat_int, data.Discrete)
    feat_cat = data.DatasetSchema.infer_feature(pd.Series(['cat']))
    self.assertEqual(feat_cat, data.Categorical)






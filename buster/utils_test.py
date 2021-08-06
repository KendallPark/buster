import unittest
from buster import utils, data
import numpy as np
import pandas as pd
from IPython import embed

SAMPLES_2 = np.array([[1, 2, 3], [7, 8, 9], [7, 8, 9]])

SPACE_2 = data.Space.from_df(pd.DataFrame(SAMPLES_2))

DF_1 = pd.DataFrame({
    'int': [-100, 0, 100],
    'float': [-100.0, 0.0, 100.0],
    'bool': [True, False, True],
    'cat': ['cat', 'dog', 'rabbit']
})

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
    self.assertEqual(result, ((1 / 3)**2 + (1 / 3)**2 + (1 / 3)**2)**0.5)

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
    coeff_1 = ((num_samples + 1)**(1 / num_dim) - 1) / 2
    coeff_2 = (num_samples + 1) / 2
    l2_norm_min = (0.1**2 + 0.1**2 + 0.1**2)**(1 / 2)
    min_norm_min = 0.1
    expected_2 = coeff_1 * l2_norm_min + coeff_2 * min_norm_min
    result_2 = utils.intersite_proj(np.array([1, 1, 1]), samples, space)
    self.assertEqual(result_2, expected_2)

  def test_mixed_features(self):
    dimensions = [
        data.Integer(0, 10),
        data.Real(0, 10),
        data.Categorical(['cat', 'dog', 'rabbit'])
    ]
    space = data.Space(dimensions)
    samples = pd.DataFrame([[0, 0.0, 'cat']])
    result = utils.intersite_proj([0, 0.0, 'cat'], samples, space)
    self.assertEqual(result, 0)

    num_samples = 1
    num_dim = 3
    coeff_1 = ((num_samples + 1)**(1 / num_dim) - 1) / 2
    coeff_2 = (num_samples + 1) / 2
    l2_norm_min = (0.1**2 + 0.1**2 + 1**2)**(1 / 2)
    min_norm_min = 0.1
    expected_2 = coeff_1 * l2_norm_min + coeff_2 * min_norm_min
    result = utils.intersite_proj([1, 1.0, 'dog'], samples, space)
    self.assertEqual(result, expected_2)


class TestVoronoiRankings(unittest.TestCase):

  def test_label_factor(self):
    labels = [True, False, True, True]
    result = utils.label_factor(labels)
    expected = [0.25, 0.75, 0.25, 0.25]
    np.testing.assert_array_equal(result, expected)

    labels_2 = [1, 2, 3, 1]
    result_2 = utils.label_factor(labels_2)
    expected_2 = [0.5, 0.75, 0.75, 0.5]
    np.testing.assert_array_equal(result_2, expected_2)

    labels_3 = ['cat', 'dog', 'rabbit', 'rabbit']
    result_3 = utils.label_factor(labels_3)
    expected_3 = [0.75, 0.75, 0.5, 0.5]
    np.testing.assert_array_equal(result_3, expected_3)

  def test_voronoi_volume_fractions(self):
    tree = data.GowerBallTree(DF_1, SPACE_1)
    candidates = utils.lhs_scaled(SPACE_1, 100, random_state=0)
    volumes = utils.voronoi_volume_fractions(tree, candidates)
    np.testing.assert_array_equal(volumes, [0.15, 0.68, 0.17])

  def test_neighbor_score(self):
    tree = data.GowerBallTree(DF_1, SPACE_1)
    y = [True, False, True]
    scores = utils.neighbor_score(DF_1, y, tree, 2)
    np.testing.assert_array_equal(scores, [1, 1, 0])

  def test_voronoi_ranks(self):
    tree = data.GowerBallTree(DF_1, SPACE_1)
    candidates = utils.lhs_scaled(SPACE_1, 100, random_state=0)
    y = [True, False, True]

    rankings = utils.voronoi_rankings(DF_1, y, SPACE_1, candidates)
    top_rank = np.argmax(rankings)
    self.assertEqual(rankings.shape, (len(DF_1),))
    self.assertEqual(top_rank, 1)


if __name__ == '__main__':
  unittest.main()

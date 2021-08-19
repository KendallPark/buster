import unittest
import math
# from IPython import embed
from skopt import space as sp
import numpy as np

from buster import sampler, metrics


class TestAdaptiveSampler(unittest.TestCase):

  def test_ask_one_dimension_numeric(self):
    X = [[10], [20], [30], [40], [50], [60], [70], [80], [90], [100]]
    y = [False, False, False, False, False, True, True, True, True, True]
    space = sp.Space([(0, 100)])
    opt = sampler.AdaptiveSampler(space.dimensions, random_state=0)
    opt.tell(X, y)
    result = opt.ask()

    expected = [[66], [53], [59], [64], [62], [58], [70], [51], [55], [68]]

    # TODO: rework so that it doesn't change every time the algorithm is altered
    np.testing.assert_array_equal(result, expected)

  def test_ask_two_dimensions_mixed(self):
    X = [[10, 'cat'], [20, 'cat'], [30, 'cat'], [40, 'cat'], [50, 'cat'],
         [60, 'cat'], [70, 'cat'], [80, 'cat'], [90, 'cat'], [100, 'cat']]
    y = [False, False, False, False, False, True, True, True, True, True]

    space = sp.Space([(0, 100), ['cat', 'dog', 'rabbit']])

    apt = sampler.AdaptiveSampler(space.dimensions, random_state=0)
    apt.tell(X, y)
    result = apt.ask()

    expected = [[51, 'cat'], [60, 'dog'], [68, 'dog'], [77, 'rabbit'],
                [54, 'rabbit'], [75, 'dog'], [79, 'cat'], [53, 'dog'],
                [64, 'rabbit'], [42, 'rabbit'], [69, 'cat'], [58, 'rabbit'],
                [50, 'cat'], [44, 'dog'], [72, 'dog'], [57, 'dog'], [47, 'dog'],
                [63, 'dog'], [73, 'cat'], [44, 'dog']]

    # TODO: rework so that it doesn't change every time the algorithm is altered
    np.testing.assert_array_equal(result, expected)

  def test_run_three_dimensions_mixed(self):
    space = sp.Space([(0, 100), (0., 100.), ['cat', 'dog', 'rabbit']])

    opt = sampler.AdaptiveSampler(space.dimensions,
                                  random_state=1,
                                  n_initial_points=1000)

    def func(X):

      def in_radius(c_x, c_y, r, x, y):
        return math.hypot(c_x - x, c_y - y) <= r

      answer = []
      for x, y, a in X:
        if a == 'rabbit':
          answer.append(in_radius(50, 50, 20, x, y))
        elif a == 'cat':
          answer.append(in_radius(20, 20, 10, x, y))
        elif a == 'dog':
          answer.append(in_radius(60, 60, 30, x, y))

      return answer

    opt.run(func, n_iter=20)

    result = opt.get_result()
    # TODO: finish this test


class TestKLargestDiverseNeighborhood(unittest.TestCase):

  def test_one_dimension_numeric(self):
    X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    y = [False, False, False, False, False, True, True, True, True, True]

    space = sp.Space([(0, 10)])

    distances = metrics.gowers.gowers_distance(X, X, space)

    result = sampler.k_largest_diverse_neighborhood(distances,
                                                    y,
                                                    n_neighbors=space.n_dims *
                                                    2,
                                                    k=2)

    expected = np.array([[5, 4, 6], [4, 3, 5]])

    np.testing.assert_array_equal(result, expected)

  def test_one_dimension_categorical(self):
    X = [['cat'], ['dog'], ['rabbit']]
    y = [False, True, False]

    space = sp.Space([['cat', 'dog', 'rabbit']])

    distances = metrics.gowers.gowers_distance(X, X, space)

    result = sampler.k_largest_diverse_neighborhood(distances,
                                                    y,
                                                    n_neighbors=space.n_dims *
                                                    2,
                                                    k=2)

    expected = np.array([[1, 0, 2], [2, 1, 0]])

    np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
  unittest.main()

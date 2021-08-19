"""Tests for data.py."""
import unittest
import numpy as np
from skopt import space as sp

from buster.metrics import gowers

SPACE_1 = sp.Space([
    sp.Integer(-100, 100),
    sp.Real(-100.0, 100.0),
    sp.Categorical([True, False]),
    sp.Categorical(['cat', 'dog', 'rabbit'])
])


class TestGowersDistance(unittest.TestCase):

  def test_gowers_difference_one_to_one(self):
    point_a = [100, 100, 0, 'cat']
    point_b = [-100, -100, 1, 'dog']
    result = gowers.gowers_distance([point_a], [point_b], SPACE_1)
    expected = np.array(1)
    np.testing.assert_array_equal(result, expected)

  def test_gowers_difference_one_to_many(self):
    point_a = [0, 0, 0, 'cat']
    point_b = [100, 0, 0, 'cat']
    point_c = [-100, 50, 1, 'dog']
    result = gowers.gowers_difference([point_a], [point_b, point_c], SPACE_1)
    expected = np.array([[[0.5, 0., 0., 0.], [0.5, 0.25, 1., 1.]]])
    np.testing.assert_array_equal(result, expected)

  def test_gowers_difference_many_to_manys(self):
    point_a = [0, 0, 0, 'cat']
    point_b = [100, 0, 0, 'cat']
    point_c = [-100, 50, 1, 'rabbit']
    result = gowers.gowers_difference([point_a, point_c], [point_b, point_c],
                                      SPACE_1)
    expected = np.array([[[0.5, 0., 0., 0.], [0.5, 0.25, 1., 1.]],
                         [[1., 0.25, 1., 1.], [0., 0., 0., 0.]]])
    np.testing.assert_array_equal(result, expected)

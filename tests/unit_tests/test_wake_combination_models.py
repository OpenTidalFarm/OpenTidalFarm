from opentidalfarm import *
import numpy
import pytest

# To account for precision errors.
TOLERANCE = numpy.array([1e-14]*2)

def setup(combiner, random, zero_flow=False):
    combiner = combiner()

    if random:
        n_turbines = 20
        u_ij = numpy.random.rand(n_turbines, 2)
        u_j = numpy.ones((n_turbines, 2))
    else:
        n_turbines = 2
        u_ij = numpy.array([[0.5, 0.5], [0.2, 1.0]])
        u_j = numpy.ones((n_turbines, 2))

    if zero_flow:
        u_j = numpy.zeros(numpy.shape(u_j))

    for pair in zip(u_ij, u_j):
        combiner.add(pair[0], pair[1])

    return combiner, u_ij, u_j


class TestGeometricSum(object):

    def test_reduce_known(self):
        combiner, u_ij, u_j = setup(GeometricSum, random=False)
        calculated = combiner.reduce()
        # [(0.5/1.0)*(0.2/1.0), (0.5/1.0)*(1.0/1.0)]  = [0.1, 0.5]
        expected = numpy.array([0.1, 0.5])
        assert((abs(calculated-expected) < TOLERANCE).all())

    def test_reduce_known_with_zeros(self):
        combiner, u_ij, u_j = setup(GeometricSum, random=False, zero_flow=True)
        calculated = combiner.reduce()
        # u_j is all zeros, but the combiner should catch the zero division
        # error and set the combined flow to zero.
        expected = numpy.zeros(2)
        assert((abs(calculated-expected) < TOLERANCE).all())

    def test_reduce_random(self):
        combiner, u_ij, u_j = setup(GeometricSum, random=True)
        calculated = combiner.reduce()
        # [1.0, 1.0]
        expected = numpy.ones(2)
        for pair in zip(u_ij, u_j):
            expected *= (pair[0]/pair[1])
        assert((abs(calculated-expected) < TOLERANCE).all())


class TestLinearSuperposition(object):

    def test_reduce_known(self):
        combiner, u_ij, u_j = setup(LinearSuperposition, random=False)
        calculated = combiner.reduce()
        # [(1.0 - 0.5/1.0) + (1.0 - 0.2/1.0),
        #  (1.0 - 0.5/1.0) + (1.0 - 1.0/1.0)] = [1.3, 0.5]
        expected = numpy.array([1.3, 0.5])
        assert((abs(calculated-expected) < TOLERANCE).all())

    def test_reduce_known_with_zeros(self):
        combiner, u_ij, u_j = setup(LinearSuperposition, random=False,
                                    zero_flow=True)
        calculated = combiner.reduce()
        # [(1.0 - 0.5/0) + (1.0 - 0.2/0),
        #  (1.0 - 0.5/0) + (1.0 - 1.0/0)] = [2.0, 2.0]
        # u_j is all zeros, but the combiner should catch the zero division
        # error and set u_ij/u_j to zero.
        expected = numpy.array([2., 2.])
        assert((abs(calculated-expected) < TOLERANCE).all())

    def test_reduce_random(self):
        combiner, u_ij, u_j = setup(LinearSuperposition, random=True)
        calculated = combiner.reduce()
        # [0.0, 0.0]
        expected = numpy.zeros(2)
        for pair in zip(u_ij, u_j):
            expected += (1.0 - (pair[0]/pair[1]))
        assert((abs(calculated-expected) < TOLERANCE).all())


class TestEnergyBalance(object):

    def test_reduce_known(self):
        combiner, u_ij, u_j = setup(EnergyBalance, random=False)
        calculated = combiner.reduce()
        # [(1.0**2 - 0.5**2) + (1.0**2 - 0.2**2),
        #  (1.0**2 - 0.5**2) + (1.0**2 - 1.0**2)] = [0.750+0.960, 0.750+0.00]
        #                                         = [1.71, 0.75]
        expected = numpy.array([1.71, 0.75])
        assert((abs(calculated-expected) < TOLERANCE).all())

    def test_reduce_random(self):
        combiner, u_ij, u_j = setup(EnergyBalance, random=True)
        calculated = combiner.reduce()
        # [0.0, 0.0]
        expected = numpy.zeros(2)
        for pair in zip(u_ij, u_j):
            expected += (pair[1]**2 - pair[0]**2)
        assert((abs(calculated-expected) < TOLERANCE).all())


class TestSumOfSquares(object):

    def test_reduce_known(self):
        combiner, u_ij, u_j = setup(SumOfSquares, random=False)
        calculated = combiner.reduce()
        # [(1.0 - 0.5/1.0)**2 + (1.0 - 0.2/1.0)**2,
        #  (1.0 - 0.5/1.0)**2 + (1.0 - 1.0/1.0)**2] = [0.89, 0.25]
        expected = numpy.array([0.89, 0.25])
        assert((abs(calculated-expected) < TOLERANCE).all())

    def test_reduce_known_with_zeros(self):
        combiner, u_ij, u_j = setup(SumOfSquares, random=False, zero_flow=True)
        calculated = combiner.reduce()
        # [(1.0 - 0.5/0) + (1.0 - 0.2/0),
        #  (1.0 - 0.5/0) + (1.0 - 1.0/0)] = [2.0, 2.0]
        # u_j is all zeros, but the combiner should catch the zero division
        # error and set u_ij/u_j to zero.
        expected = numpy.array([2., 2.])
        assert((abs(calculated-expected) < TOLERANCE).all())

    def test_reduce_random(self):
        combiner, u_ij, u_j = setup(SumOfSquares, random=True)
        calculated = combiner.reduce()
        # [0.0, 0.0]
        expected = numpy.zeros(2)
        for pair in zip(u_ij, u_j):
            expected += (1.0 - (pair[0]/pair[1]))**2
        assert((abs(calculated-expected) < TOLERANCE).all())

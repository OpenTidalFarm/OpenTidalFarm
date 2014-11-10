from opentidalfarm import *
import numpy
import pytest

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
        assert((expected == calculated).all())

    def test_reduce_known_with_zeros(self):
        combiner, u_ij, u_j = setup(GeometricSum, random=False, zero_flow=True)
        calculated = combiner.reduce()
        # u_j is all zeros, but the combiner should catch the zero division
        # error and set the combined flow to zero.
        expected = numpy.zeros(2)
        assert((expected == calculated).all())

    def test_reduce_random(self):
        combiner, u_ij, u_j = setup(GeometricSum, random=True)
        calculated = combiner.reduce()
        # [1.0, 1.0]
        expected = numpy.ones(2)
        for pair in zip(u_ij, u_j):
            expected *= (pair[0]/pair[1])
        assert((expected == calculated).all())



class TestLinearSuperposition(object):

    def calculate_expected(self, setup):
        result = numpy.zeros(2)
        for t1, t2 in zip(setup.turbine, setup.in_wake_of):
            with numpy.errstate(divide="ignore"):
                res = t1/t2
            res[numpy.isinf(res) + numpy.isnan(res)] = 0.0
            result += (1. - res)
        return result

    def test_reduce_known(self):
        setup = GenericSetup(LinearSuperposition, False)
        combiner = setup.combiner
        expected = self.calculate_expected(setup)
        assert((expected == combiner.reduce()).all())

    def test_reduce_random(self):
        setup = GenericSetup(LinearSuperposition, True)
        combiner = setup.combiner
        expected = self.calculate_expected(setup)
        assert((expected == combiner.reduce()).all())


# class TestEnergyBalance(object):
#
#     def calculate_expected(self, setup):
#         result = numpy.zeros(2)
#         for t1, t2 in zip(setup.turbine, setup.in_wake_of):
#             result += t2**2 - t1**2
#         return result
#
#     def test_reduce_known(self):
#         setup = GenericSetup(EnergyBalance, False)
#         combiner = setup.combiner
#         expected = self.calculate_expected(setup)
#         assert((expected == combiner.reduce()).all())
#
#     def test_reduce_random(self):
#         setup = GenericSetup(EnergyBalance, True)
#         combiner = setup.combiner
#         expected = self.calculate_expected(setup)
#         assert((expected == combiner.reduce()).all())
#
#
# class TestSumOfSquares(object):
#
#     def calculate_expected(self, setup):
#         result = numpy.zeros(2)
#         for t1, t2 in zip(setup.turbine, setup.in_wake_of):
#             with numpy.errstate(divide="ignore"):
#                 res = t1/t2
#             res[numpy.isinf(res) + numpy.isnan(res)] = 0.0
#             result += (1. - res)**2
#         return result
#
#
#     def test_reduce_known(self):
#         setup = GenericSetup(SumOfSquares, False)
#         combiner = setup.combiner
#         expected = self.calculate_expected(setup)
#         assert((expected == combiner.reduce()).all())
#
#     def test_reduce_random(self):
#         setup = GenericSetup(SumOfSquares, True)
#         combiner = setup.combiner
#         expected = self.calculate_expected(setup)
#         assert((expected == combiner.reduce()).all())

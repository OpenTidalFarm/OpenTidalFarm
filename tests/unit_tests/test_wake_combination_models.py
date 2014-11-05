from opentidalfarm import *
import numpy
import pytest

class TestGeometricSum(object):

    def test_add(self):
        geometric_sum = GeometricSum((1.0, 1.0))

        geometric_sum.add((10.0, 10.0))
        assert(len(geometric_sum.flow_speed_in_wake)==1)

        geometric_sum.add((5.0, 5.0))
        assert(len(geometric_sum.flow_speed_in_wake)==2)


    def test_reduce(self):
        geometric_sum = GeometricSum((5.0, 5.0))
        for i in xrange(4):
            geometric_sum.add((5.5, 5.5))

        expected = [(5.5**4)/5.0]*2
        assert((geometric_sum.reduce()==expected).all())


class TestLinearSuperposition(object):

    def test_reduce(self):
        at_turbine = [5.0]*2
        reduced = [4.5]*2
        linear_superpos = LinearSuperposition(at_turbine)
        for i in xrange(4):
            linear_superpos.add(reduced)

        reduced = numpy.array(reduced)
        at_turbine = numpy.array(at_turbine)
        expected = 4*(1 - reduced/at_turbine)
        assert((linear_superpos.reduce()==expected).all())


        at_turbine = [1.0]*2
        expected = [0.0]*2
        linear_superpos = LinearSuperposition(at_turbine)
        random_velocities = numpy.random.rand(25, 2)*2.0
        for val in random_velocities:
            linear_superpos.add(val)
            expected += (1.0 - (val/at_turbine))

        expected = [expected]*2
        assert((linear_superpos.reduce()==expected).all())


class TestEnergyBalance(object):

    def test_reduce(self):
        at_turbine = [5.0]*2
        reduced = [4.5]*2
        combiner = EnergyBalance(at_turbine)
        for i in xrange(4):
            combiner.add(reduced)

        reduced = numpy.array(reduced)
        at_turbine = numpy.array(at_turbine)
        expected = 4*(at_turbine**2 - reduced**2)
        assert((combiner.reduce()==expected).all())

        at_turbine = numpy.array([1.0]*2)
        expected = numpy.array([0.0]*2)
        combiner = EnergyBalance(at_turbine)
        random_velocities = numpy.random.rand(25, 2)*2.0
        for val in random_velocities:
            combiner.add(val)
            expected += (at_turbine**2 - val**2)
        assert((combiner.reduce()==expected).all())


class TestSumOfSquares(object):

    def test_reduce(self):
        at_turbine = [5.0]*2
        reduced = [4.5]*2
        combiner = SumOfSquares(at_turbine)
        for i in xrange(4):
            combiner.add(reduced)

        reduced = numpy.array(reduced)
        at_turbine = numpy.array(at_turbine)
        expected = 4*(1. - reduced/at_turbine)**2
        assert((combiner.reduce()==expected).all())

        at_turbine = numpy.array([1.0]*2)
        expected = numpy.array([0.0]*2)
        combiner = SumOfSquares(at_turbine)
        random_velocities = numpy.random.rand(25, 2)*2.0
        for val in random_velocities:
            combiner.add(val)
            expected += (1. - val/at_turbine)**2
        assert((combiner.reduce()==expected).all())

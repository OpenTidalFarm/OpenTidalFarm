from opentidalfarm import *
import numpy
import pytest

class TestWakeModel(object):

    def test_callable_flow_field(self):
        with pytest.raises(TypeError):
            WakeModel(1.0)
            WakeModel(numpy.array([1.0, 2.0]))

    def flow1(self, x):
        return numpy.array([1.0, 0.0])

    def flow2(self, x):
        return numpy.array([1.0, 1.0])


    def test_relative_position_flow1(self):
        wake = WakeModel(self.flow1)

        assert (wake.flow_at(numpy.random.rand(2)) == ((1.0, 0.0))).all()

        point_a = numpy.array([0.0, 0.0])
        point_b = numpy.array([5.0, 5.0])

        assert (wake.relative_position(point_a, point_b) ==
                numpy.array([5.0, 5.0])).all()


        point_a = numpy.array([1.0, 1.0])
        point_b = numpy.array([6.0, 6.0])

        assert (wake.relative_position(point_a, point_b) ==
                numpy.array([5.0, 5.0])).all()


        point_a = numpy.array([0.0, 0.0])
        point_b = numpy.array([5.0, 0.0])

        assert (wake.relative_position(point_a, point_b) ==
                numpy.array([5.0, 0.0])).all()


        point_a = numpy.array([0.0, 0.0])
        point_b = numpy.array([0.0, 5.0])

        assert (wake.relative_position(point_a, point_b) ==
                numpy.array([0.0, 5.0])).all()


    def test_relative_position_flow2(self):
        wake = WakeModel(self.flow2)

        assert (wake.flow_at(numpy.random.rand(2)) == ((1.0, 1.0))).all()

        point_a = numpy.array([0.0, 0.0])
        point_b = numpy.array([5.0, 5.0])
        dist_para = (2*(5.**2))**0.5
        dist_perp = 0.0
        eps = 1e-12

        result_diff = (wake.relative_position(point_a, point_b)
                       -numpy.array([dist_para, dist_perp]))

        assert (result_diff < eps).all()


        point_a = numpy.array([1.0, 1.0])
        point_b = numpy.array([6.0, 6.0])
        result_diff = (wake.relative_position(point_a, point_b)
                       -numpy.array([dist_para, dist_perp]))

        assert (result_diff < eps).all()


        point_a = numpy.array([0.0, 0.0])
        point_b = numpy.array([5.0, 0.0])
        result_diff = (wake.relative_position(point_a, point_b)
                       -numpy.array([dist_para, dist_perp]))

        dist_para = ((5.0**2)*0.5)**0.5
        dist_perp = dist_para

        assert (result_diff < eps).all()


        point_a = numpy.array([0.0, 0.0])
        point_b = numpy.array([0.0, 5.0])

        assert (result_diff < eps).all()

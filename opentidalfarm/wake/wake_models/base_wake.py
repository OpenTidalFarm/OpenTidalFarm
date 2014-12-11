"""
A BaseWakeModel class from which other wake models may be derived.
"""
import numpy

class WakeModel(object):
    """A base class from which wake models may be derived."""
    def __init__(self, flow_field):
        if not hasattr(flow_field, '__call__'):
            raise TypeError("'flow_field' must be a callable object")
        self._flow_field = flow_field


    def flow_at(self, point):
        """Returns the flow field at point."""
        return numpy.asarray(self._flow_field(point))


    def in_wake(self, turbine, another_turbine):
        """Returns if turbine is in the wake of another_turbine."""
        raise NotImplementedError("The 'in_wake' method is not implemented "
                                  "in the base class.")


    def multiplier(self, turbine, another_turbine):
        """Returns flow speed multiplier at turbine due to another_turbine."""
        raise NotImplementedError("The 'multiplier' method is not "
                                  "implemented in the base class.")


    def relative_position(self, turbine, another_turbine):
        """Returns the relative of another_turbine to turbine.

        In the coordinate system where the flow vector at turbine is parallel to
        the x-axis and turbine is at the origin we wish to get x- and
        y-component of another_turbine.
        """
        # We aim to rotate the vector from turbine to another_turbine such that the flow
        # vector at turbine is parallel to the x-axis (the 'target' vector).
        target = numpy.array([1.0, 0.0])
        flow_at_a = self.flow_at(turbine)
        normalised_flow_at_a = flow_at_a/numpy.linalg.norm(flow_at_a, 2)
        sin_theta = numpy.cross(target, normalised_flow_at_a)
        cos_theta = (1 - sin_theta**2)**0.5
        rotation_matrix = numpy.matrix([[cos_theta, -sin_theta],
                                        [sin_theta,  cos_theta]])
        return numpy.array((another_turbine-turbine)*rotation_matrix).flatten()

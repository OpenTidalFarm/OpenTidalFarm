import numpy
from dolfin import *
from dolfin_adjoint import *

class TurbineFunction(object):

    def __init__(self, V, turbine_specification, derivative_index=-1):
        # self.params = ParameterDictionary(params)
        self._turbine_specification = turbine_specification

        # Precompute some turbine parameters for efficiency.
        self.x = interpolate(Expression("x[0]"), V).vector().array()
        self.y = interpolate(Expression("x[1]"), V).vector().array()
        self.V = V


    def __call__(self, cache, name="", derivative_index=None,
                 derivative_var=None, timestep=None):
        """If the derivative selector is i >= 0, the Expression will compute the
        derivative of the turbine with index i with respect to either the x or y
        coorinate or its friction parameter. """


        if derivative_index is None:
            turbine_pos = cache["turbine_pos"]
            if timestep is None:
                turbine_friction = cache["turbine_friction"]
            else:
                turbine_friction = cache["turbine_friction"][timestep]
        else:
            turbine_pos = [cache["turbine_pos"][derivative_index]]
            if timestep is None:
                turbine_friction = [cache["turbine_friction"][derivative_index]]
            else:
                turbine_friction = (
                    [cache["turbine_friction"][timestep][derivative_index]])

        # Infeasible optimisation algorithms (such as SLSQP) may try to evaluate
        # the functional with negative turbine_frictions. Since the forward
        # model would crash in such cases, we project the turbine friction
        # values to positive reals.
        turbine_friction = [max(0, f) for f in turbine_friction]

        ff = numpy.zeros(len(self.x))
        # Ignore division by zero.
        numpy.seterr(divide="ignore")
        eps = 1e-12

        for (x_pos, y_pos), friction in zip(turbine_pos, turbine_friction):
            radius = self._turbine_specification.radius
            x_unit = numpy.minimum(
                numpy.maximum(((self.x-x_pos)/radius), eps-1), 1-eps)
            y_unit = numpy.minimum(
                numpy.maximum(((self.y-y_pos)/radius), eps-1), 1-eps)

            # Apply chain rule to get the derivative with respect to the turbine
            # friction.
            e = numpy.exp(-1/(1-x_unit**2)-1./(1-y_unit**2)+2)
            if derivative_index is None:
                ff += e*friction

            elif derivative_var == "turbine_friction":
                ff += e

            if derivative_var == "turbine_pos_x":
                ff += e*(-2*x_unit/((1.0-x_unit**2)**2))*friction*(-1.0/radius)

            elif derivative_var == "turbine_pos_y":
                ff += e*(-2*y_unit/((1.0-y_unit**2)**2))*friction*(-1.0/radius)

        # Reset numpy to warn for zero division errors.
        numpy.seterr(divide="warn")

        f = Function(self.V, name=name, annotate=False)
        f.vector().set_local(ff)
        f.vector().apply("insert")
        return f




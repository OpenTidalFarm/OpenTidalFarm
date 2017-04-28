from .base_turbine import BaseTurbine
from .controls import Controls
from dolfin import pi, sqrt, dot

class BumpTurbine(BaseTurbine):
    """ Create a turbine that is modelled as a bump of bottom friction """

    # The integral of the unit bump function computed with Wolfram Alpha:
    # "integrate e^(-1/(1-x**2)-1/(1-y**2)+2) dx dy,
    #  x=-0.999..0.999, y=-0.999..0.999"
    # http://www.wolframalpha.com/input/?i=integrate+e%5E%28-1%2F%281-x**2%29-1%2F%281-y**2%29%2B2%29+dx+dy%2C+x%3D-0.999..0.999%2C+y%3D-0.999..0.999
    _unit_bump_int = 1.45661

    def __init__(self, thrust_coefficient=0.8, 
                 diameter=20., minimum_distance=None,
                 controls=Controls(position=True)):

        # Check for a given minimum distance.
        if minimum_distance is None: minimum_distance=diameter*1.5

        self.thrust_coefficient = thrust_coefficient
        swept_area = pi*(diameter/2.)**2

        # in the actual drag term we're using eqn (15) in [1], but the correction is only added in 
        # in the solver as it depends on the depth
        # [1] S.C. Kramer, M.D. Piggott http://doi.org/10.1016/j.renene.2016.02.022
        # so here c_t = friction * bump, friction = A_t*C_t/(2*integral(bump))
        self.c_t_integral = swept_area * thrust_coefficient / 2.
        friction = self.c_t_integral / (self._unit_bump_int * diameter**2/4.)

        # Initialize the base class.
        super(BumpTurbine, self).__init__(friction=friction,
                                          diameter=diameter,
                                          minimum_distance=minimum_distance,
                                          controls=controls,
                                          smeared=False)
    @property
    def integral(self):
        """The integral of the turbine bump function.
        :returns: The integral of the turbine bump function.
        :rtype: float
        """
        return self._unit_bump_int*self._diameter**2/4.



    def force(self, u, tf=None, depth=None):
        """Return the thrust force exerted by a turbine for given velocity u

        Keyword arguments:
        tf    -- turbine friction function containing one or more scaled bump functions. If provided, the function
                 then returns the expression to be integrated over a farm that computes the total power. If not
                 provided, the integrated power of a single turbine is returned.
        depth -- if provided, u is assumed to be the depth-averaged velocity and a correction is made
                 to estimate the free stream and turbine speeds in the force computation. If not provided
                 it is assumed that depth-averaged, free stream and turbine speeds are roughly the same."""
        if tf is None:
            tf = self.c_t_integral
        if depth is None:
            correction = 1
        else:
            # correction from eqn (15) in [1]
            # [1] S.C. Kramer, M.D. Piggott http://doi.org/10.1016/j.renene.2016.02.022
            # where turbine_field = C_t A_t / (2*Integral(bump))
            C_t = self.thrust_coefficient
            # ratio = A_t/\hat{A_t} = pi (D/2)^2 / D*H = pi * D / (4 * H)
            # note that here we always use linear depth to avoid adding unnec. non-linearities
            area_ratio = pi * self.diameter / (4*depth)
            correction = 4/(1+sqrt(1-area_ratio*self.thrust_coefficient))**2
        return correction * tf * sqrt(dot(u, u)) * u


    def power(self, u, tf=None, depth=None):
        """Return the amount of power produced by a turbine for given speed u

        Keyword arguments:
        tf    -- friction function containing one or more scaled bump functions. If provided, the function
                 then returns the expression to be integrated over a farm that computes the total power. If not
                 provided, the integrated power of a single turbine is returned.
        depth -- if provided, u is assumed to be the depth-averaged speed and a correction is made
                 to estimate the free stream and turbine speeds in the power computation. If not provided
                 it is assumed that depth-averaged, free stream and turbine speeds are roughly the same."""

        if tf is None:
            tf = self.c_t_integral
        if depth is None:
            correction = 1
        else:
            # eqn (C.2) from [1]
            # [1] S.C. Kramer, M.D. Piggott http://doi.org/10.1016/j.renene.2016.02.022
            # where turbine_field = C_t A_t / (2*Integral(bump))
            C_t = self.thrust_coefficient
            # ratio = A_t/\hat{A_t} = pi (D/2)^2 / D*H = pi * D / (4 * H)
            area_ratio = pi * self.diameter / (4*depth)
            correction = 4 * (1+sqrt(1-C_t))/(1+sqrt(1-area_ratio*C_t))**3

        return correction * tf * u**3

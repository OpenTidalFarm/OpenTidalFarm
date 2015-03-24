#"""
#.. module:: Power Functionals
#   :synopsis: This module contains the functional classes which compute the
#       power extracted by an array.
#"""

from dolfin import dot, Constant, dx, assemble, conditional
from ..helpers import smooth_uflmin
from prototype_functional import PrototypeFunctional


class PowerFunctional(PrototypeFunctional):
    r""" Implements a simple functional of the form:

    .. math:: J(u, m) = \int g(u) \rho  c_t ||u||^3~ dx,

    where :math:`c_t` defines the friction field due to the turbines, and
    g(u) implements the cut-in/out behaviour of the turbine, i.e.

    .. math:: g(u) =
        \begin{cases}
           eps & \text{if } \|u\| < \textrm{cut_in_speed} \\
           eps & \text{if } \|u\| > \textrm{cut_out_speed} \\
           1 & \text{else.}
        \end{cases}

    :param problem: The problem for which the functional is being computed.
    :type problem: Instance of the problem class.
    :param cut_in_speed: The turbine's cut in speed (Default: None).
    :type cut_in_speed: float
    :param cut_out_speed: The turbine's cut out speed (Default: None).
    :type cut_out_speed: float
    :param eps: The turbine's cut in/out accuracy (Default: 1e-10).
    :type esp: float
    """

    def __init__(self, problem, cut_in_speed=None, cut_out_speed=None, eps=1e-10):

        self.farm = problem.parameters.tidal_farm
        self.rho = problem.parameters.rho
        self.farm.update()

        self._cut_in_speed = cut_in_speed
        self._cut_out_speed = cut_out_speed
        self._eps = eps


    def Jt(self, state, turbine_field):
        """ Computes the power output of the farm.

        :param state: Current solution state
        :type state: UFL
        :param turbine_field: Turbine friction field
        :type turbine_field: UFL

        """
        return self.power(state, turbine_field)*self.farm.site_dx

    def power(self, state, turbine_field):
        """ Computes the power field over the domain.

        :param state: Current solution state.
        :type state: UFL
        :param turbine_field: Turbine friction field
        :type turbine_field: UFL

        """
        return (self._cut_in_out(state) * self.rho *
                turbine_field * (dot(state[0], state[0]) +
                dot(state[1], state[1])) ** 1.5)

    def Jt_individual(self, state, i):
        """ Computes the power output of the i'th turbine.

        :param state: Current solution state
        :type state: UFL
        :param i: refers to the i'th turbine
        :type i: Integer

        """
        turbine_field_individual = \
                self.farm.turbine_cache['turbine_field_individual'][i]
        return assemble(self.power(state, turbine_field_individual) *
                        self.farm.site_dx)

    def force(self, state, turbine_field):
        """ Computes the force field over turbine field

        :param state: Current solution state.
        :type state: UFL
        :param turbine_field: Turbine friction field
        :type turbine_field: UFL

        """
        return (self._cut_in_out(state) * self.rho * turbine_field *
                dot(state[0], state[0]) + dot(state[1], state[1]))

    def force_individual(self, state, i):
        """ Computes the total force on the i'th turbine

        :param state: Current solution state
        :type state: UFL
        :param i: refers to the i'th turbine
        :type i: Integer

        """
        turbine_field_individual = \
                self.farm.turbine_cache['turbine_field_individual'][i]
        return assemble(self.force(state, turbine_field_individual) * self.farm.site_dx(1))

    def _cut_in_out(self, state):
        """ Implements a cut in and out switching function """

        speed_sq = dot(state[0], state[0]) + dot(state[1], state[1])

        factor = 1
        if self._cut_in_speed is not None:
            factor *= conditional(speed_sq < self._cut_in_speed**2, self._eps, 1)

        if self._cut_out_speed is not None:
            factor *= conditional(speed_sq > self._cut_out_speed**2, self._eps, 1)

        return factor

class PowerCurveFunctional(PrototypeFunctional):
#    ''' Implements a functional for the power with a given power curve
#    :math:`J(u, m) = \int_\Omega P(u)`
#    where m controls the strength of each turbine.
    """ TODO: doesn't work yet...
    """
    def __init__(self, farm):
        ''' Constructs a new DefaultFunctional. The turbine settings are
        derived from the settings params. '''
        farm.update()
        self.farm = farm
        # Create a copy of the parameters so that future changes will not
        # affect the definition of this object.
        self.params = dict(farm.params)
        assert(self.params["turbine_thrust_parametrisation"] or \
               self.params["implicit_turbine_thrust_parametrisation"])

    def Jt(self, state, tf):
        up_u = state[3]  # Extract the upstream velocity
        #ux = state[0]
        def power_function(u):
            # A simple power function implementation.
            # Could be replaced with a polynomial approximation.
            fac = Constant(1.5e6 / (3 ** 3))
            return smooth_uflmin(1.5e6, fac * u ** 3)

        P = power_function(up_u)*tf/self.farm.turbine_specification.integral*dx
        return P

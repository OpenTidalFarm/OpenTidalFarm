#"""
#.. module:: Power Functionals
#   :synopsis: This module contains the functional classes which compute the
#       power extracted by an array.
#"""

from dolfin import dot, Constant, dx, sqrt
from ..helpers import smooth_uflmin
from prototype_functional import PrototypeFunctional


class PowerFunctional(PrototypeFunctional):
    r""" Implements a power functional of the form:

    .. math:: J(u, m) = \int \rho  c_t ||sq(u)||^{1.5}~ dx,

    where :math:`c_t` is the friction due to the turbines, and
    :math:`sq(u)` is the squared velocity.

    :param problem: The problem for which the functional is being computed.
    :type problem: Instance of the problem class.
    """

    def __init__(self, problem):

        self.farm = problem.parameters.tidal_farm
        self.rho = problem.parameters.rho
        self.depth = problem.parameters.depth
        self.farm.update()

    def Jt(self, state, turbine_field):
        """ Computes the power output of the farm.

        :param state: Current solution state
        :type state: dolfin.Function
        :param turbine_field: Turbine friction field
        :type turbine_field: dolfin.Function

        """
        u = sqrt(dot(state[0], state[0]) + dot(state[1], state[1]))
        return self.rho * self.farm.power_integral(u, tf=turbine_field, depth=self.depth)
            

    def Jt_individual(self, state, i):
        """ Computes the power output of the i'th turbine.

        :param state: Current solution state
        :type state: dolfin.Function
        :param i: refers to the i'th turbine
        :type i: Integer

        """
        turbine_field_individual = \
                self.farm.turbine_cache['turbine_field_individual'][i]
        u = sqrt(dot(state[0], state[0]) + dot(state[1], state[1]))
        return self.rho * self.farm.power_integral(u, tf=turbine_field_individual, depth=self.depth)

#"""
#.. module:: Cost Functionals
#   :synopsis: This module contains the functional classes which compute the
#       cost of an array.
#"""

from dolfin import dot, Constant, dx, assemble, conditional
from ..helpers import smooth_uflmin
from prototype_functional import PrototypeFunctional


class CostFunctional(PrototypeFunctional):
    r""" Implements a cost functional of the form:

    .. math:: J(u, m) = \int c_t~ dx,

    where :math:`c_t` is the friction due to the turbines.

    :param problem: The problem for which the functional is being computed.
    :type problem: Instance of the problem class.
    """

    def __init__(self, problem):

        self.farm = problem.parameters.tidal_farm
        self.farm.update()

    def Jt(self, state, turbine_field):
        """ Computes the cost of the farm.

        :param state: Current solution state
        :type state: dolfin.Function
        :param turbine_field: Turbine friction field
        :type turbine_field: dolfin.Function

        """
        return self._cost(turbine_field)*self.farm.site_dx

    def _cost(self, turbine_field):
        """ Computes the cost of the turbine farm.

        :param turbine_field: Turbine friction field
        :type turbine_field: dolfin.Function

        """
        return turbine_field

    def Jt_individual(self, state, i):
        """ Computes the cost of the i'th turbine.

        :param state: Current solution state
        :type state: dolfin.Function
        :param i: turbine index
        :type i: Integer

        """
        turbine_field_individual = \
                self.farm.turbine_cache['turbine_field_individual'][i]
        return assemble(self._cost(turbine_field_individual)*self.farm.site_dx)

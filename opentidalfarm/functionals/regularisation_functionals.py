#"""
#.. module:: Regularisation Functionals
#   :synopsis: This module contains the functional classes which compute the
#       cost of an array.
#"""

from dolfin import assemble, inner, grad
from .prototype_functional import PrototypeFunctional


class H01Regularisation(PrototypeFunctional):
    r""" Implements a H01 regularisation of the form:

    .. math:: J(u, m) = \int \grad c_t \cdot \grad c_t ~ dx,

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
        # Note, this will not work properly for DG elements (need to add dS measures)
        mesh = turbine_field.function_space().mesh()
        return inner(grad(turbine_field), grad(turbine_field))*self.farm.site_dx(domain=mesh)

    def Jt_individual(self, state, i):
        """ Computes the cost of the i'th turbine.

        :param state: Current solution state
        :type state: dolfin.Function
        :param i: turbine index
        :type i: Integer

        """
        turbine_field_individual = \
                self.farm.turbine_cache['turbine_field_individual'][i]
        return assemble(self.Jt(None, turbine_field_individual))

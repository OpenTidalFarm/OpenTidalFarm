### TODO... Make work!!!

#"""
#.. module:: Power Functionals
#   :synopsis: This module contains the functional classes which compute the
#       power extracted by an array.
#"""

from dolfin import dot, Constant, dx, assemble
from ..helpers import smooth_uflmin
from prototype_functional import PrototypeFunctional


class FinancialFunctional(PrototypeFunctional):
    """ Implement a simple financial functional.
    """

    def __init__(self, problem=None):

        self.problem = problem
        
        if self.problem is not None:
            self.farm = problem.parameters.tidal_farm
            self.rho = problem.parameters.rho
            self.farm.update()
        # Create a copy of the parameters so that future changes will not
        # affect the definition of this object.
        # self.params = dict(farm.params)

    def late_instantiate(self, problem):
        """ If the functional is being used in advanced turbine placement
        operations, instantiation needs to happen AFTER the problem has been
        defined - hence we do it here...
        """
        self.farm = problem.parameters.tidal_farm
        self.rho = problem.parameters.rho
        self.farm.update()


    def Jt(self, state, turbine_field):
        """ Computes the power output of the farm.

        :param state: Current solution state
        :type state: UFL
        :param turbine_field: Turbine friction field
        :type turbine_field: UFL

        """
        return self.expense + (self.profit(state, turbine_field) * self.farm.site_dx)

    
    def profit(self, state, turbine_field):
        """ Computes the profit (income minus expenditure) of the farm
        """ 
        income = self.income(self.power(state, turbine_field))
        expense = self.expense()
        print '*** Incoming = ', income
        print '*** Outgoing = ', expense
        print '*** Profit   = ', income + expense
        return income + expense


    def power(self, state, turbine_field):
        """ Computes the current power extraction of the array
        """
        return assemble((self.rho * turbine_field * (dot(state[0], state[0]) +
                        dot(state[1], state[1])) ** 1.5) * self.farm.site_dx)

    def income(self, power):
        """ Converts power into income (simply linearly)
        Assumption, each watt of installed capacity will generate 1 penny of
        income over the plant lifetime
        """
        return 1 * power


    def expense(self):
        """ Formulates the expense of the array as directly proportional to the
        number of turbines
        Assumption - each turbine costs /pounds 3.5 m 
        """
        number_of_turbines = len(self.farm.turbine_positions)
        cost_per_turbine = 1500000
        print '*** Number of turbines = ', number_of_turbines
        expense = number_of_turbines * cost_per_turbine
        return -expense


#    def Jt_individual(self, state, i):
#        """ Computes the power output of the i'th turbine.
#
#        :param state: Current solution state
#        :type state: UFL
#        :param i: refers to the i'th turbine
#        :type i: Integer
#
#        """
#        turbine_field_individual = \
#                self.farm.turbine_cache['turbine_field_individual'][i]
#        return self.power(state, turbine_field_individual) * self.farm.site_dx



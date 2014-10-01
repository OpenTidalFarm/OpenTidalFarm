"""
.. module:: Power Functionals
   :synopsis: This module contains the functional classes which compute the
       power extracted by an array.
"""

from dolfin import dot, Constant, dx
from ..turbines import *
from ..helpers import smooth_uflmin
from prototype_functional import PrototypeFunctional


class PowerFunctional(PrototypeFunctional):
    ''' Implements a simple functional of the form:
    J(u, m) = rho * turbines(m) * (||u||**3)
    where turbines(m) defines the friction function due to the turbines.
    '''
### ADD IN RHO ###
    def __init__(self, farm, rho):
        ''' Constructs a new functional for computing the power. The turbine
        settings are derived from the settings params. '''
        farm.turbine_cache.update(farm)
        self.farm = farm
        self.rho = rho
        # Create a copy of the parameters so that future changes will not
        # affect the definition of this object.
        self.params = dict(farm.params)


    def Jt(self, state, tf):
        return self.power(state, tf) * self.farm.site_dx(1)

    def power(self, state, turbines):
        ''' Computes the power field over the domain '''
        return (self.rho * turbines * (dot(state[0], state[0]) + dot(state[1],
                state[1])) ** 1.5)

    def Jt_individual(self, state, i):
        ''' Computes the power output of the i'th turbine. '''
        tf = self.farm.turbine_cache.cache['turbine_field_individual'][i]
        return self.power(state, tf) * self.farm.site_dx(1)

class PowerCurveFunctional(PrototypeFunctional):
    ''' Implements a functional for the power with a given power curve
    J(u, m) = \int_\Omega power(u)
    where m controls the strength of each turbine.
    TODO: doesn't work yet...
    '''
    def __init__(self, farm):
        ''' Constructs a new DefaultFunctional. The turbine settings are
        derived from the settings params. '''
        farm.turbine_cache.update(farm)
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

        P = power_function(up_u) * tf / self.farm.turbine_cache.turbine_integral() * dx
        return P

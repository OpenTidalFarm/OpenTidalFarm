from ..turbines import *
from ..helpers import smooth_uflmin


class FunctionalPrototype(object):
    ''' This prototype class should be overloaded for an actual functional implementation.  '''

    def __init__(self):
        raise NotImplementedError("FunctionalPrototyp.__init__ needs to be overloaded.")

    def Jt(self):
        ''' This function should return the form that computes the functional's contribution for one timelevel.'''
        raise NotImplementedError("FunctionalPrototyp.__Jt__ needs to be overloaded.")


class PowerFunctional(FunctionalPrototype):
    ''' Implements a simple functional of the form:
          J(u, m) = rho * turbines(m) * (||u||**3)
        where turbines(m) defines the friction function due to the turbines.
    '''
    def __init__(self, config):
        config.turbine_cache.update(config)
        self.config = config
        # Create a copy of the parameters so that future changes will not affect the definition of this object.
        self.params = dict(config.params)

    def cost_per_friction(self, turbines):
        if float(self.params['cost_coef']) <= 0:
            return Constant(0)
        # Function spaces with polynomial degree >1 suffer from undershooting which can result in
        # negative cost values.
        if turbines.function_space().ufl_element().degree() > 1:
            raise ValueError('Costing only works if the function space for the turbine friction has polynomial degree < 2.')
        return Constant(self.params['cost_coef']) * turbines

    def power(self, state, turbines):
            return self.params['rho'] * turbines * (dot(state[0], state[0]) + dot(state[1], state[1])) ** 1.5

    def force(self, state, turbines):
            return self.params['rho'] * turbines * dot(state[0], state[0]) + dot(state[1], state[1])

    def Jt(self, state, tf):
        return (self.power(state, tf) - self.cost_per_friction(tf)) * self.config.site_dx(1)

    def Jt_individual(self, state, i):
        ''' Computes the power output of the i'th turbine. '''
        tf = self.config.turbine_cache.cache['turbine_field_individual'][i]
        return (self.power(state, tf) - self.cost_per_friction(tf)) * self.config.site_dx(1)

    def force_individual(self, state, i):
        ''' Computes the total force on the i'th turbine. '''
        tf = self.config.turbine_cache.cache['turbine_field_individual'][i]
        return (self.force(state, tf) - self.cost_per_friction(tf)) * self.config.site_dx(1)


class PowerCurveFunctional(FunctionalPrototype):
    ''' Implements a functional for the power with a given power curve
          J(u, m) = \int_\Omega power(u)
        where m controls the strength of each turbine.
    '''
    def __init__(self, config):
        config.turbine_cache.update(config)
        self.config = config
        # Create a copy of the parameters so that future changes will not affect the definition of this object.
        self.params = dict(config.params)

    def Jt(self, state, tf):
        up_u = state[3]  # Extract the upstream velocity
        #ux = state[0]

        def power_function(u):
            # A simple power function implementation. Could be replaced with a polynomial approximation.
            fac = Constant(1.5e6 / (3 ** 3))
            return smooth_uflmin(1.5e6, fac * u ** 3)

        P = power_function(up_u) * tf / self.config.turbine_cache.turbine_integral() * dx
        return P

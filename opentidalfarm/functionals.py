from dolfin import info
import ufl
from turbines import *
from helpers import info, info_green, info_red, info_blue
from parameter_dict import ParameterDictionary
import shallow_water_model

class FunctionalPrototype(object):
    ''' This prototype class should be overloaded for an actual functional implementation.  '''

    def __init__(self):
        raise NotImplementedError, "FunctionalPrototyp.__init__ needs to be overloaded."

    def Jt(self):
        ''' This function should return the form that computes the functional's contribution for one timelevel.'''
        raise NotImplementedError, "FunctionalPrototyp.__Jt__ needs to be overloaded."

class DefaultFunctional(FunctionalPrototype):
    ''' Implements a simple functional of the form:
          J(u, m) = rho * turbines(m) * (||u||**3)
        where turbines(m) defines the friction function due to the turbines.  
    '''
    def __init__(self, config):
        ''' Constructs a new DefaultFunctional. The turbine settings are derived from the settings params. '''
        config.turbine_cache.update(config)
        self.config = config
        # Create a copy of the parameters so that future changes will not affect the definition of this object.
        self.params = ParameterDictionary(dict(config.params))

    def expr(self, state, turbines):
        return self.params['rho'] * turbines * (dot(state[0], state[0]) + dot(state[1], state[1]))**1.5

    def Jt(self, state, tf):
        return self.expr(state, tf)*self.config.site_dx(1)

class PowerCurveFunctional(FunctionalPrototype):
    ''' Implements a functional for the power with a given power curve 
          J(u, m) = \int_\Omega power(u) 
        where m controls the strength of each turbine.
    '''
    def __init__(self, config):
        ''' Constructs a new DefaultFunctional. The turbine settings are derived from the settings params. '''
        config.turbine_cache.update(config)
        self.config = config
        # Create a copy of the parameters so that future changes will not affect the definition of this object.
        self.params = ParameterDictionary(dict(config.params))
        assert(self.params["turbine_thrust_parametrisation"] or self.params["implicit_turbine_thrust_parametrisation"])

    def Jt(self, state, tf):
        up_u = state[3]
        ux = state[0]

        def power_function(u):
            # A simple power function implementation. Could be replaced with a polynomial approximation. 
            fac = Constant(1.5e6/(3**3))
            return shallow_water_model.smooth_uflmin(1.5e6, fac*u**3)

        P = inner(Constant(1), power_function(up_u)*tf/self.config.turbine_cache.turbine_integral())*dx

        #print "Expected power: %f MW" % (power_function(ux((10, 160)))((0))/1e6)
        print "Estimated power: %f MW" % (assemble(P)/1e6)
        return P

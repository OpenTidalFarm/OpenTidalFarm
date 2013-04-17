from dolfin import info
import ufl
from turbines import *
from helpers import info, info_green, info_red, info_blue
from parameter_dict import ParameterDictionary

class FunctionalPrototype(object):
    ''' This prototype class should be overloaded for an actual functional implementation.  '''

    def __init__(self):
        raise NotImplementedError, "FunctionalPrototyp.__init__ needs to be overloaded."

    def Jt(self):
        ''' This function should return the form that computes the functional's contribution for one timelevel.'''
        raise NotImplementedError, "FunctionalPrototyp.__Jt__ needs to be overloaded."

class DefaultFunctional(FunctionalPrototype):
    ''' Implements a simple functional of the form:
          J(u, m) = 0.5 * rho * turbines(m) * (||u||**3)
        where turbines(m) defines the friction function due to the turbines.  
    '''
    def __init__(self, config):
        ''' Constructs a new DefaultFunctional. The turbine settings are derived from the settings params. '''
        config.turbine_cache.update(config)
        self.config = config
        # Create a copy of the parameters so that future changes will not affect the definition of this object.
        self.params = ParameterDictionary(dict(config.params))

    def expr(self, state, turbines):
        return 0.5 * self.params['rho'] * turbines * (dot(state[0], state[0]) + dot(state[1], state[1]))**1.5

    def Jt(self, state):
        return self.expr(state, self.config.turbine_cache.cache['turbine_field'])*dx 

def smoothmax(r, eps=1e-6):
    return conditional(gt(r, eps), r - eps/2, conditional(lt(r, 0), 0, r**2 / (2*eps)))

def uflmin(a, b):
    return conditional(lt(a, b), a, b)
def uflmax(a, b):
    return conditional(gt(a, b), a, b)

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
        assert(self.params["turbine_thrust_representation"])

    def Jt(self, state):
        up_u = state[3]
        ux = state[0]
        tf = self.config.turbine_cache.cache['turbine_field']

        def power_function(u):
            fac = Constant(1.5e6/27.66)
            return uflmin(1.5e6, fac*u**3)

        P = inner(Constant(1), power_function(up_u)*tf/self.config.turbine_cache.turbine_integral())*dx

        print "Estimated power: %f MW" % (assemble(P)/1e6)
        print "Expected power: %f MW" % (power_function(ux((640./3-1.5*20, 160)))((0))/1e6)
        return assemble(P)

from dolfin import info
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

        class TurbineDomain(SubDomain):
            def __init__(self, turbine_index):
                self.center = config.params["turbine_pos"][turbine_index]
                self.turbine_x = 3*config.params["turbine_x"]
                self.turbine_y = 3*config.params["turbine_y"]
                super(TurbineDomain, self).__init__()

            def inside(self, x, on_boundary):
                return (between(x[0]-self.center[0], (-self.turbine_x, self.turbine_x)) 
                        and between(x[1]-self.center[1], (-self.turbine_y, self.turbine_y)))

        t = Timer("Marking turbine measures")
        self.dx = []
        for i in range(len(config.params["turbine_pos"])):
            domains = CellFunction("size_t", config.domain.mesh)
            domains.set_all(0)

            (TurbineDomain(i)).mark(domains, 1)
            self.dx.append(Measure("dx")[domains])

    def expr(self, state, turbines):
        return 0.5 * self.params['rho'] * turbines * (dot(state[0], state[0]) + dot(state[1], state[1]))**1.5

    def Jt(self, state):
        t = Timer("Time spend in functional evaluation")
        x = 0
        for i in range(len(self.config.params["turbine_pos"])):
            x += min(1.5e6, assemble(self.expr(state, self.config.turbine_cache.cache['turbine_field'])*self.dx[i](1)))
        print "Time spend in functional evaluation", t.stop()
        return x

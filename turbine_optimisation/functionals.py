from dolfin import info
from turbines import *
from helpers import info, info_green, info_red, info_blue

class FunctionalPrototype(object):
    ''' This prototype class should be overloaded for an actual functional implementation.  '''

    def __init__(self):
        raise NotImplementedError, "FunctionalPrototyp.__init__ needs to be overloaded."

    def Jt(self):
        ''' This function should return the form that computes the functional's contribution for one timelevel.'''
        raise NotImplementedError, "FunctionalPrototyp.__Jt__ needs to be overloaded."

    def dJtdm(self):
        ''' This function computes the partial derivatives with respect to controls of the functional's contribution for one timelevel. ''' 
        raise NotImplementedError, "FunctionalPrototyp.__dJdm__ needs to be overloaded."

class DefaultFunctional(FunctionalPrototype):
    ''' Implements a simple functional of the form:
          J(u, m) = dt * turbines(params) * (||u||**3)
        where m controls the strength of each turbine.
    '''
    def __init__(self, config):
        ''' Constructs a new DefaultFunctional. The turbine settings are derived from the settings params. '''
        config.turbine_cache.update(config)
        self.config = config
        # Create a copy of the parameters so that future changes will not affect the definition of this object.
        self.params = configuration.Parameters(dict(config.params))

    def expr(self, state, turbines):
        return self.params['rho'] * turbines * (dot(state[0], state[0]) + dot(state[1], state[1]))**1.5*dx

    def Jt(self, state):
        return self.expr(state, self.config.turbine_cache.cache['turbine_field']) 

    def Jt_individual(self, state, i):
        ''' Computes the power output of the i'th turbine. '''
        return self.expr(state, self.config.turbine_cache.cache['turbine_field_individual'][i]) 

    def dJtdm(self, state):
        djtdm = [] 
        params = self.params

        if "turbine_friction" in params["controls"]:
            # The derivatives with respect to the friction parameter
            for n in range(len(params["turbine_friction"])):
                djtdm.append(self.expr(state, self.config.turbine_cache.cache['turbine_derivative_friction'][n]))

        if "turbine_pos" in params["controls"]:
            # The derivatives with respect to the turbine position
            for n in range(len(params["turbine_pos"])):
                for var in ('turbine_pos_x', 'turbine_pos_y'):
                    djtdm.append(self.expr(state, self.config.turbine_cache.cache['turbine_derivative_pos'][n][var]))
        return djtdm 

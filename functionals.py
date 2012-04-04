from dolfin import info_green
from turbines import *

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
          J(u, m) = dt * turbines(params) * 0.5 * (||u||**3)
        where m controls the strength of each turbine.
    '''
    def scale_turbine_size(self):
        '''Scales the turbine size by the given factor. '''
        params = self.params

        # Scale the turbine size by the given factor.
        if params["functional_turbine_scaling"] != 1.0:
            info("The functional uses turbines which size is scaled by a factor of " + str(params["functional_turbine_scaling"]) + ".")
        params["turbine_x"] *= params["functional_turbine_scaling"]
        params["turbine_y"] *= params["functional_turbine_scaling"] 

    def __init__(self, W, params):
        ''' Constructs a new DefaultFunctional. The turbine settings arederived from the settings params. '''
        # Create a copy of the parameters so that future changes will not affect the definition of this object.
        self.params = configuration.Parameters(dict(params))
        self.scale_turbine_size()
        self.turbine_cache = self.build_turbine_cache(W) 

    def expr(self, state, turbines):
        return turbines*0.5*(dot(state[0], state[0]) + dot(state[1], state[1]))**1.5*dx

    def Jt(self, state):
        return self.expr(state, self.turbine_cache['turbine_field']) 

    def dJtdm(self, state):
        djtdm = [] 
        params = self.params

        if "turbine_friction" in params["controls"]:
            # The derivatives with respect to the friction parameter
            for n in range(len(params["turbine_friction"])):
                djtdm.append(self.expr(state, self.turbine_cache['turbine_derivative_friction'][n]))

        if "turbine_pos" in params["controls"]:
            # The derivatives with respect to the turbine position
            for n in range(len(params["turbine_pos"])):
                for var in ('turbine_pos_x', 'turbine_pos_y'):
                    djtdm.append(self.expr(state, self.turbine_cache['turbine_derivative_pos'][n][var]))
        return djtdm 

    def build_turbine_cache(self, W):
        ''' Creates a list of all turbine function/derivative interpolations. This list is used as a cache 
          to avoid the recomputation of the expensive interpolation of the turbine expression. '''
        params = self.params
        turbine_cache = {}
        U = W.split()[0].sub(0) # extract the first component of the velocity function space 
        U = U.collapse() # recompute the dof map

        # Precompute the interpolation of the friction function
        turbines = Turbines(params)
        tf = Function(U)
        tf.interpolate(turbines)
        turbine_cache["turbine_field"] = tf

        # Precompute the derivatives with respect to the friction
        turbine_cache["turbine_derivative_friction"] = []
        for n in range(len(params["turbine_friction"])):
            tf = Function(U)
            turbines = Turbines(params, derivative_index_selector=n, derivative_var_selector='turbine_friction')
            tf = Function(U)
            tf.interpolate(turbines)
            turbine_cache["turbine_derivative_friction"].append(tf)

        # Precompute the derivatives with respect to the turbine position
        turbine_cache["turbine_derivative_pos"] = []
        for n in range(len(params["turbine_pos"])):
            turbine_cache["turbine_derivative_pos"].append({})
            for var in ('turbine_pos_x', 'turbine_pos_y'):
                tf = Function(U)
                turbines = Turbines(params, derivative_index_selector=n, derivative_var_selector=var)
                tf = Function(U)
                tf.interpolate(turbines)
                turbine_cache["turbine_derivative_pos"][-1][var] = tf

        return turbine_cache

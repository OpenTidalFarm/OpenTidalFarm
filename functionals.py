import sw_lib
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
  def scale_turbine_size(self, params, fac):
    '''Scales the turbine size by the given factor. '''
    params["turbine_x"] *= fac
    params["turbine_y"] *= fac

  def __init__(self, params, turbine_cache):
    ''' Constructs a new DefaultFunctional. The turbine settings arederived from the settings params. 
        '''
    # Create a copy of the parameters so that future changes will not affect the definition of this object.
    self.turbine_cache = turbine_cache
    self.params = sw_config.Parameters(dict(params))

  def expr(self, state, turbines):
    return turbines*0.5*(dot(state[0], state[0]) + dot(state[1], state[1]))**1.5*dx

  def Jt(self, state):
    return self.expr(state, self.turbine_cache['turbine_field']) 

  def dJtdm(self, state):
    djtdm = [] 
    params = self.params

    # The derivatives with respect to the friction parameter
    for n in range(len(params["turbine_friction"])):
      djtdm.append(self.expr(state, self.turbine_cache['turbine_derivative_friction'][n]))

    # The derivatives with respect to the turbine position
    for n in range(len(params["turbine_pos"])):
      for var in ('turbine_pos_x', 'turbine_pos_y'):
        djtdm.append(self.expr(state, self.turbine_cache['turbine_derivative_pos'][n][var]))
    return djtdm 

def build_turbine_cache(params, function_space, turbine_size_scaling = 1.0):
  ''' Creates a list of all turbine function/derivative interpolations. This list is used as a cache 
      to avoid the recomputation of the expensive interpolation of the turbine expression. '''
  turbine_cache = {}

  params = sw_config.Parameters(dict(params))
  # Scale the turbine size by the given factor.
  if turbine_size_scaling != 1.0:
    info_green("The functional uses turbines which size is scaled by a factor of " + str(turbine_size_scaling) + ".")
  params["turbine_x"] *= turbine_size_scaling
  params["turbine_y"] *= turbine_size_scaling

  # Precompute the interpolation of the friction function
  turbines = Turbines(params)
  tf = Function(function_space)
  tf.interpolate(turbines)
  turbine_cache["turbine_field"] = tf

  # Precompute the derivatives with respect to the friction
  turbine_cache["turbine_derivative_friction"] = []
  for n in range(len(params["turbine_friction"])):
    tf = Function(function_space)
    turbines = Turbines(params, derivative_index_selector=n, derivative_var_selector='turbine_friction')
    tf = Function(function_space)
    tf.interpolate(turbines)
    turbine_cache["turbine_derivative_friction"].append(tf)

  # Precompute the derivatives with respect to the turbine position
  turbine_cache["turbine_derivative_pos"] = []
  for n in range(len(params["turbine_pos"])):
    turbine_cache["turbine_derivative_pos"].append({})
    for var in ('turbine_pos_x', 'turbine_pos_y'):
      tf = Function(function_space)
      turbines = Turbines(params, derivative_index_selector=n, derivative_var_selector=var)
      tf = Function(function_space)
      tf.interpolate(turbines)
      turbine_cache["turbine_derivative_pos"][-1][var] = tf

  return turbine_cache

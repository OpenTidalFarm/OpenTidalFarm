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

  def __init__(self, params, turbine_size_scaling=1.0):
    ''' Constructs a new DefaultFunctional. The turbine settings arederived from the settings params. 
        If the optinal turbine_size_scaling argument is provided, then the functional will use a turbine which
        sizes are scaled by that factor.'''
    # Create a copy of the parameters so that future changes will not affect the definition of this object.
    self.params = sw_lib.parameters(dict(params))
    # Divide the turbine size by two in order to get a better turbine model.
    self.scale_turbine_size(self.params, turbine_size_scaling)
    if turbine_size_scaling != 1.0:
      info_green("The functional uses turbines which size is scaled by a factor of " + str(turbine_size_scaling) + ".")

  def expr(self, state, turbines):
    return self.params["dt"]*turbines*0.5*(dot(state[0], state[0]) + dot(state[1], state[1]))**1.5*dx

  def Jt(self, state):
    turbines = Turbines(self.params)
    return self.expr(state, turbines) 

  def dJtdm(self, state):
    djtdm = [] 
    params = self.params

    # The derivatives with respect to the friction parameter
    for n in range(len(params["turbine_friction"])):
      turbines = Turbines(self.params, derivative_index_selector=n, derivative_var_selector='turbine_friction')
      djtdm.append(self.expr(state, turbines))

    # The derivatives with respect to the turbine position
    for n in range(len(params["turbine_pos"])):
      for var in ('turbine_pos_x', 'turbine_pos_y'):
        turbines = Turbines(self.params, derivative_index_selector=n, derivative_var_selector=var)
        djtdm.append(self.expr(state, turbines))
    return djtdm 


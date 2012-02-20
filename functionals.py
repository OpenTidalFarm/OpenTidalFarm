import sw_lib
from turbines import *

class FunctionalPrototype(object):
  ''' Implements a simple functional of the form:
        J(u, m) = dt * turbines(m) * 0.5 * (||u||**3)
      where m controls the strength of each turbine
  '''
  def __init__(self):
    ''' m are the control variables. If None, then the functional does not depend on the controls. '''
    raise NotImplementedError, "FunctionalPrototyp.__init__ needs to be overloaded."

  def Jt(self):
    ''' This function returns the form that computes the functional's contribution for one timelevel with solution 'state'. If m is none then the fucntional does not depend directly on the control parameter.'''
    raise NotImplementedError, "FunctionalPrototyp.__Jt__ needs to be overloaded."

  def dJtdm(self):
    ''' This function computes the partial derivatives with respect to controls of the functional's contribution for one timelevel with solution 'state'. If m is None then the derivative does not depend on the control parmeter. '''
    raise NotImplementedError, "FunctionalPrototyp.__dJdm__ needs to be overloaded."

class DefaultFunctional(FunctionalPrototype):
  ''' Implements a simple functional of the form:
        J(u, m) = dt * turbines(params) * 0.5 * (||u||**3)
      where m controls the strength of each turbine
  '''
  def __init__(self, params):
    # Create a copy of the parameters so that future changes will not affect the definition of this object.
    self.params = sw_lib.parameters(dict(params))

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


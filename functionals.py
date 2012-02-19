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

class DefaultFunctionalWithoutControlDependency(FunctionalPrototype):
  ''' Implements a simple functional of the form:
        J(u, m) = dt * turbines * 0.5 * (||u||**3)
      where m controls the strength of each turbine
  '''
  def __init__(self, params):
    # Create a copy of the parameters so that future changes will not affect the definition of this object.
    self.params = sw_lib.parameters(dict(params))

  def Jt(self, state):
    turbines = self.params['turbine_model'](self.params)
    return self.params["dt"]*turbines*0.5*(dot(state[0], state[0]) + dot(state[1], state[1]))**1.5*dx

  def dJtdm(self, state):
    return []


class DefaultFunctional(FunctionalPrototype):
  ''' Implements a simple functional of the form:
        J(u, m) = dt * turbines(m) * 0.5 * (||u||**3)
      where m controls the strength of each turbine
  '''
  def __init__(self, params, m):
    # Create a copy of the parameters so that future changes will not affect the definition of this object.
    self.params = sw_lib.parameters(dict(params))
    self.m = m

  def Jt(self, state):
    turbine_friction_orig = self.params["turbine_friction"]
    self.params["turbine_friction"] = self.m * self.params["turbine_friction"]
    turbines = self.params['turbine_model'](self.params)
    self.params["turbine_friction"] = turbine_friction_orig 

    return self.params["dt"]*turbines*0.5*(dot(state[0], state[0]) + dot(state[1], state[1]))**1.5*dx

  def dJtdm(self, state):
    djtdm = [] 
    turbine_friction_orig = self.params["turbine_friction"]

    for n in range(len(self.m)):
      dm = numpy.zeros(len(self.m))
      dm[n] = 1.0
      self.params["turbine_friction"] = dm * self.params["turbine_friction"]
      turbines = self.params['turbine_model'](self.params)
      djtdm.append(self.params["dt"]*turbines*0.5*(dot(state[0], state[0]) + dot(state[1], state[1]))**1.5*dx)
      self.params["turbine_friction"] = turbine_friction_orig 
    return djtdm 


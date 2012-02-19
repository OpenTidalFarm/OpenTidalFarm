import sys
import sw_config 
import sw_lib
import numpy
from dolfin import *
from dolfin_adjoint import *
from sw_utils import test_initial_condition_adjoint, test_gradient_array
from turbines import *

def default_config():
  config = sw_config.DefaultConfiguration(nx=20, ny=5)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  config.params["finish_time"] = 2./4*period
  config.params["dt"] = config.params["finish_time"]/5
  print "Wave period (in h): ", period/60/60 
  config.params["dump_period"] = 1000
  config.params["verbose"] = 0

  # Start at rest state
  config.params["start_time"] = period/4 

  # Turbine settings
  config.params["friction"] = 0.0025
  config.params["turbine_pos"] = [[1000., 500.], [2000., 500.]]
  # The turbine friction is the control variable 
  config.params["turbine_friction"] = 12.0*numpy.ones(len(config.params["turbine_pos"]))
  config.params["turbine_length"] = 200
  config.params["turbine_width"] = 400

  # Now create the turbine measure
  config.initialise_turbines_measure()
  return config

def initial_control(config):
  numpy.random.seed(41) 
  return numpy.random.rand(len(config.params['turbine_friction']))

def j_and_dj(m):
  adjointer.reset()
  adj_variables.__init__()
  
  set_log_level(30)
  debugging["record_all"] = True

  W=sw_lib.p1dgp2(config.mesh)

  # Get initial conditions
  state=Function(W)
  state.interpolate(config.get_sin_initial_condition()())

  # Set the control values
  U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
  U = U.collapse() # Recompute the DOF map
  tf = Function(U)
  # Apply the control

  # Set up the turbine friction field using the provided control variable
  turbine_friction_orig = config.params["turbine_friction"]
  config.params["turbine_friction"] = m * turbine_friction_orig
  tf.interpolate(GaussianTurbines(config.params))
  config.params["turbine_friction"] = turbine_friction_orig 

  M,G,rhs_contr,ufl=sw_lib.construct_shallow_water(W, config.ds, config.params, turbine_field = tf)
  class Functional(object):
    def __init__(self, params, m=None):
      ''' m are the control variables. If None, then the functional does not depend on the controls. '''
      # Create a copy of the parameters so that future changes will not affect the definition of this object.
      self.params = sw_lib.parameters(dict(params))
      self.m = m

    def Jt(self, state):
      ''' This function returns the form that computes the functional's contribution for one timelevel with solution 'state'. If m is none then the fucntional does not depend directly on the control parameter.'''
      turbine_friction_orig = self.params["turbine_friction"]
      self.params["turbine_friction"] = self.m * self.params["turbine_friction"]
      turbines = GaussianTurbines(self.params)
      self.params["turbine_friction"] = turbine_friction_orig 

      return self.params["dt"]*turbines*0.5*(dot(state[0], state[0]) + dot(state[1], state[1]))**1.5*dx

    def dJtdm(self, state):
      ''' This function computes the partial derivatives with respect to controls of the functional's contribution for one timelevel with solution 'state'. If m is None then the derivative does not depend on the control parmeter. '''
      djtdm = [] 
      turbine_friction_orig = self.params["turbine_friction"]

      for n in range(len(self.m)):
        dm = numpy.zeros(len(self.m))
        dm[n] = 1.0
        self.params["turbine_friction"] = dm * self.params["turbine_friction"]
        turbines = GaussianTurbines(self.params)
        djtdm.append(self.params["dt"]*turbines*0.5*(dot(state[0], state[0]) + dot(state[1], state[1]))**1.5*dx)
        self.params["turbine_friction"] = turbine_friction_orig 
      return djtdm 

  func_forms = Functional(config.params, m)
  # Solve the shallow water system
  j, djdm, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, state, config.params, time_functional=func_forms)

  J = TimeFunctional(func_forms.Jt(state))
  adj_state = sw_lib.adjoint(state, config.params, J, until=1)

  # Let J be the functional, m the parameter and u the solution of the PDE equation F(u) = 0.
  # Then we have 
  # dJ/dm = (\partial J)/(\partial u) * (d u) / d m + \partial J / \partial m
  #               = adj_state * \partial F / \partial u + \partial J / \partial m
  # In this particular case m = turbine_friction, J = \sum_t(ft) 
  dj = numpy.zeros(len(config.params["turbine_friction"]))
  v = adj_state.vector()
  for n in range(len(dj)):
    turbine_friction_orig = config.params["turbine_friction"]
    m = numpy.zeros(len(dj))
    m[n] = 1.0
    config.params["turbine_friction"] = m * config.params["turbine_friction"]
    tf.interpolate(GaussianTurbines(config.params))
    dj[n] = v.inner(tf.vector()) 
    config.params["turbine_friction"] = turbine_friction_orig 
  
  # Now add the \partial J / \partial m term
  dj += djdm
  return j, dj 

def j(m):
  return j_and_dj(m)[0]

def dj(m):
  return j_and_dj(m)[1]

# run the taylor remainder test 
config = default_config()
m0 = initial_control(config)

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
numpy.random.seed(21) 
p = numpy.random.rand(len(config.params['turbine_friction']))
minconv = test_gradient_array(j, dj, m0, seed=0.0001, perturbation_direction=p)
if minconv < 1.99:
  exit_code = 1
else:
  exit_code = 0
sys.exit(exit_code)

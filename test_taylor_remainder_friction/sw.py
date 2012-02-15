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
  config.params["dt"] = config.params["finish_time"]/20
  print "Wave period (in h): ", period/60/60 
  config.params["dump_period"] = 1000
  config.params["verbose"] = 0

  # Start at rest state
  config.params["start_time"] = period/4 

  # Turbine settings
  config.params["friction"] = 0.0025
  config.params["turbine_pos"] = [[1000., 500.], [2000., 500.]]
  # The turbine friction is the control variable 
  config.params["turbine_friction"] = 12.0
  config.params["turbine_length"] = 200
  config.params["turbine_width"] = 400

  # Now create the turbine measure
  config.initialise_turbines_measure()
  return config

def initial_control(config):
  return numpy.array([0.2145]) # Choose a random starting point

def j_and_dj(x):
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
  config.params["turbine_friction"] = x[0] * turbine_friction_orig
  tf.interpolate(GaussianTurbines(config))
  config.params["turbine_friction"] = turbine_friction_orig 

  M,G,rhs_contr,ufl=sw_lib.construct_shallow_water(W, config.ds, config.params, turbine_field = tf)
  def functional(state):
    turbines = GaussianTurbines(config)
    return config.params["dt"]*turbines*x[0]*0.5*(dot(state[0], state[0]) + dot(state[1], state[1]))**1.5*dx

  # Solve the shallow water system
  j, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, state, config.params, time_functional=functional)

  J = TimeFunctional(functional(state))
  adj_state = sw_lib.adjoint(state, config.params, J, until=1)

  # we have dJ/dx = (\partial J)/(\partial turbine_friction) * (d turbine_friction) / d x +
  #                 + \partial J / \partial x
  #               = adj_state * turbine_friction
  #                 + \partial J / \partial x
  tf.interpolate(GaussianTurbines(config))
  v = adj_state.vector()
  turbines = GaussianTurbines(config)
  # In this case, j = \sum_t(functional) and \partial functional / \partial x = funtional/x. Hence we haev \partial J / \partial x = j/x
  dj = v.inner(tf.vector()) + j/x[0] 
  
  return j, numpy.array([dj])

def j(x):
  return j_and_dj(x)[0]

def dj(x):
  return j_and_dj(x)[1]

# run the taylor remainder test 
config = default_config()
x0 = initial_control(config)

# We set the perturbation_direction, so that it is consistent in a parallel environment.
p = numpy.array([1.])
minconv = test_gradient_array(j, dj, x0, seed=0.0001, perturbation_direction=p)
if minconv < 1.99:
  exit_code = 1
else:
  exit_code = 0
sys.exit(exit_code)

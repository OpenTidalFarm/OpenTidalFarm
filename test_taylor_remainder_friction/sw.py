import sys
import sw_config 
import sw_lib
from functionals import * 
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

def j_and_dj(m ):
  global depend_m
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
  config.params["turbine_friction"] = m 
  tf.interpolate(Turbines(config.params))
  config.params["turbine_friction"] = turbine_friction_orig 

  M,G,rhs_contr,ufl=sw_lib.construct_shallow_water(W, config.ds, config.params, turbine_field = tf)

  if depend_m:
    functional = DefaultFunctional(config.params, m)
  else:
    functional = DefaultFunctionalWithoutControlDependency(config.params)

  # Solve the shallow water system
  j, djdm, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, state, config.params, time_functional=functional)

  J = TimeFunctional(functional.Jt(state))
  adj_state = sw_lib.adjoint(state, config.params, J, until=1)

  # Let J be the functional, m the parameter and u the solution of the PDE equation F(u) = 0.
  # Then we have 
  # dJ/dm = (\partial J)/(\partial u) * (d u) / d m + \partial J / \partial m
  #               = adj_state * \partial F / \partial u + \partial J / \partial m
  # In this particular case m = turbine_friction, J = \sum_t(ft) 
  dj = numpy.zeros(len(config.params["turbine_friction"]))
  v = adj_state.vector()
  for n in range(len(dj)):
    tf.interpolate(Turbines(config.params, derivative_index_selector=n, derivative_var_selector='turbine_friction'))
    dj[n] = v.inner(tf.vector()) 
  
  # Now add the \partial J / \partial m term
  if depend_m:
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
depend_m = None
# Run with a functional that does not depend on m directly
for depend in [False, True]:
  print "Running test with function that depends on control = ", depend
  depend_m = depend
  minconv = test_gradient_array(j, dj, m0, seed=0.0001, perturbation_direction=p)
  if minconv < 1.99:
    sys.exit(1)

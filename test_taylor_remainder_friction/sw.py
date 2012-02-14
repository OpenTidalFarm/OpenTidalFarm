import sys
import sw_config 
import sw_lib
from dolfin import *
from dolfin_adjoint import *
from sw_utils import test_initial_condition_adjoint, test_gradient_array
from turbines import *

def default_config():
  config = sw_config.DefaultConfiguration(nx=20, ny=5)
  period = 1.24*60*60 # Wave period
  config.params["k"]=2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  config.params["finish_time"]=2./4*period
  config.params["dt"]=config.params["finish_time"]/20
  print "Wave period (in h): ", period/60/60 
  config.params["dump_period"]=1
  config.params["verbose"] = 0

  # Start at rest state
  config.params["start_time"] = period/4 

  # Turbine settings
  config.params["friction"]=0.0025
  config.params["turbine_pos"] = [[1000., 500.], [2000., 500.]]
  config.params["turbine_friction"] = 12./config.params["depth"]
  config.params["turbine_length"] = 200
  config.params["turbine_width"] = 400

  # Now create the turbine measure
  config.initialise_turbines_measure()
  return config

def initial_control(config):
  W=sw_lib.p1dgp2(config.mesh)

  U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
  U = U.collapse() # Recompute the DOF map
  tf = Function(U)
  tf.interpolate(GaussianTurbines(config))
  return tf.vector().array()

def j_and_dj(x):
  adjointer.reset()
  adj_variables.__init__()
  
  set_log_level(30)
  debugging["record_all"] = True

  W=sw_lib.p1dgp2(config.mesh)

  state=Function(W)
  state.interpolate(config.get_sin_initial_condition()())

  # Set the control values
  U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
  U = U.collapse() # Recompute the DOF map
  tf = Function(U)
  tf.vector().set_local(x) 

  M,G,rhs_contr,ufl=sw_lib.construct_shallow_water(W, config.ds, config.params, turbine_field = tf)
  def functional(state):
    turbines = GaussianTurbines(config)
    return config.params["dt"]*0.5*turbines*(dot(state[0], state[0]) + dot(state[1], state[1]))**1.5*dx

  j, state = sw_lib.timeloop_theta(M, G, rhs_contr, ufl, state, config.params, time_functional=functional)

  J = TimeFunctional(functional(state))
  adj_state = sw_lib.adjoint(state, config.params, J, until=1)
  return j, adj_state.vector().array()

def j(x):
  return j_and_dj(x)[0]

def dj(x):
  return j_and_dj(x)[1]

# run the taylor remainder test 
config = default_config()
x0 = initial_control(config)

minconv = test_gradient_array(j, dj, x0, seed=0.0001)
if minconv < 1.99:
  exit_code = 1
else:
  exit_code = 0
sys.exit(exit_code)

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
  config = sw_config.DefaultConfiguration(nx=40, ny=20)
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
  config.params["turbine_pos"] = [[1000., 500.]]
  # The turbine friction is the control variable 
  config.params["turbine_friction"] = 12.0*numpy.ones(len(config.params["turbine_pos"]))
  config.params["turbine_length"] = 200
  config.params["turbine_width"] = 400

  # Now create the turbine measure
  config.initialise_turbines_measure()
  return config

def initial_control(config):
  numpy.random.seed(41) 
  res =  numpy.random.rand(len(config.params['turbine_friction']) + len(config.params['turbine_pos']))
  res[0] = 1
  res[1] = 250 # The position parameters have to be in the domain
  #res[-2] = 750 # The position parameters have to be in the domain
  return res

def j_and_dj(m):
  # Change the control variables to the config parameters
  # FIXME: Write a generic algorithm for setting the parameters 
  config.params["turbine_friction"] = m[0:len(config.params["turbine_friction"])]
  i = len(config.params["turbine_friction"])
  config.params["turbine_pos"][0][0] = m[i]

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
  tf = Function(U) # The turbine function
  tfd = Function(U) # The derivative turbine function

  # Set up the turbine friction field using the provided control variable
  tf.interpolate(Turbines(config.params))
  v = tf.vector()
  # The functional of interest is simply the l2 norm of the turbine field
  j = v.inner(v)  

  dj = []
  # Compute the derivatives with respect to the turbine friction
  for n in range(len(config.params["turbine_friction"])):
    tfd.interpolate(Turbines(config.params, derivative_index_selector=n, derivative_var_selector='turbine_friction'))
    dj.append( 2 * v.inner(tfd.vector()) )

  # Compute the derivatives with respect to the turbine x position
  for n in range(len(config.params["turbine_pos"])):
    tf.interpolate(Turbines(config.params, derivative_index_selector=n, derivative_var_selector='turbine_pos_x'))
    dj.append( 2 * v.inner(tfd.vector()) )
  dj = numpy.array(dj)  
  
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
p = numpy.random.rand(len(config.params['turbine_friction']) + len(config.params['turbine_pos']))

# Run with a functional that does not depend on m directly
minconv = test_gradient_array(j, dj, m0, seed=0.0001, perturbation_direction=p)
if minconv < 1.99:
  sys.exit(1)

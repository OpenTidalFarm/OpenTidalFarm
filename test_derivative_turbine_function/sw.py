''' This test checks the correct implemetation of the turbine derivative terms.
    For that, we apply the Taylor remainder test on functional J(u, m) = <turbine_friction(m), turbine_friction(m)>,
    where m contains the turbine positions and the friction magnitude. 
'''

import sys
import sw_config 
import function_spaces
from functionals import * 
import numpy
from dolfin import *
from dolfin_adjoint import *
from sw_utils import test_initial_condition_adjoint, test_gradient_array
from turbines import *

def default_config():
  numpy.random.seed(21) 
  config = sw_config.DefaultConfiguration(nx=40, ny=20)
  config.params["dump_period"] = 1000
  config.params["verbose"] = 0

  # Turbine settings
  config.params["turbine_pos"] = [[1000., 500.], [1600, 300], [2500, 700]]
  # The turbine friction is the control variable 
  config.params["turbine_friction"] = 12.0*numpy.random.rand(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 200
  config.params["turbine_y"] = 400

  return config

def initial_control(config):
  # We use the current turbine settings as the intial control
  res = config.params['turbine_friction'].tolist()
  res += numpy.reshape(config.params['turbine_pos'], -1).tolist()
  return numpy.array(res)

def j_and_dj(m):
  # Change the control variables to the config parameters
  config.params["turbine_friction"] = m[:len(config.params["turbine_friction"])]
  mp = m[len(config.params["turbine_friction"]):]
  config.params["turbine_pos"] = numpy.reshape(mp, (-1, 2))

  set_log_level(30)

  W = function_spaces.p1dgp2(config.mesh)

  # Get initial conditions
  state=Function(W, name = "current_state")
  state.interpolate(config.get_sin_initial_condition()())

  # Set the control values
  U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
  U = U.collapse() # Recompute the DOF map
  tf = Function(U, name = "turbine") # The turbine function
  tfd = Function(U, name = "turbine_derivative") # The derivative turbine function

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

  # Compute the derivatives with respect to the turbine position
  for n in range(len(config.params["turbine_pos"])):
    for var in ('turbine_pos_x', 'turbine_pos_y'):
      tfd.interpolate(Turbines(config.params, derivative_index_selector=n, derivative_var_selector=var))
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
p = numpy.random.rand(len(config.params['turbine_friction']) + 2*len(config.params['turbine_pos']))

# Run with a functional that does not depend on m directly
for model, s in {'GaussianTurbine': {'seed': 100.0, 'tol': 1.9}, 'BumpTurbine': {'seed': 0.001, 'tol': 1.99}}.items():
  print "************* ", model, " ********************"
  config.params["turbine_model"] = model 
  minconv = test_gradient_array(j, dj, m0, s['seed'], perturbation_direction=p)

  if minconv < s['tol']:
    sys.exit(1)

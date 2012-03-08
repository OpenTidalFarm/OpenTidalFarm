'''This tests checks the corrections of the adjoint by using it to compute the 
   derivative of the functional with respect to the friction field.'''

import sys
import sw_config 
import sw_lib
import numpy
import Memoize
from functionals import DefaultFunctional
from dolfin import *
from dolfin_adjoint import *
from sw_utils import test_initial_condition_adjoint, test_gradient_array
from turbines import *

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  numpy.random.seed(21) 
  config = sw_config.DefaultConfiguration(nx=30, ny=15)
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
  adjointer.reset()
  adj_variables.__init__()

  # Change the control variables to the config parameters
  config.params["turbine_friction"] = m[:len(config.params["turbine_friction"])]
  mp = m[len(config.params["turbine_friction"]):]
  config.params["turbine_pos"] = numpy.reshape(mp, (-1, 2))

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

  functional = DefaultFunctional(config.params)

  # Solve the shallow water system
  j, djdm, state = sw_lib.sw_solve(W, config, state, time_functional=functional, turbine_field = tf)
  J = TimeFunctional(functional.Jt(state))
  # Because a turbine field is used, the first equation in the annotation is the initialisation
  # of this turbine field (the second equation will be the initial condition). Hence the adjoint 
  # is computed all the way back to equation 0.
  adj_state = sw_lib.adjoint(state, config.params, J, until=0)

  # Let J be the functional, m the parameter and u the solution of the PDE equation F(u) = 0.
  # Then we have 
  # dJ/dm = (\partial J)/(\partial u) * (d u) / d m + \partial J / \partial m
  #               = adj_state * \partial F / \partial u + \partial J / \partial m
  # In this particular case m = turbine_friction, J = \sum_t(ft) 
  dj = [] 
  v = adj_state.vector()
  # Compute the derivatives with respect to the turbine friction
  for n in range(len(config.params["turbine_friction"])):
    tfd.interpolate(Turbines(config.params, derivative_index_selector=n, derivative_var_selector='turbine_friction'))
    dj.append( v.inner(tfd.vector()) )

  # Compute the derivatives with respect to the turbine position
  for n in range(len(config.params["turbine_pos"])):
    for var in ('turbine_pos_x', 'turbine_pos_y'):
      tfd.interpolate(Turbines(config.params, derivative_index_selector=n, derivative_var_selector=var))
      dj.append( v.inner(tfd.vector()) )
  dj = numpy.array(dj)  
  
  # Now add the \partial J / \partial m term
  dj += djdm

  return j, dj 

j_and_dj_mem = Memoize.MemoizeMutable(j_and_dj)
def j(m):
  return j_and_dj_mem(m)[0]

def dj(m):
  return j_and_dj_mem(m)[1]

# run the taylor remainder test 
config = default_config()
m0 = initial_control(config)

# A random direction
p_rand = numpy.random.rand(len(config.params['turbine_friction']) + 2*len(config.params['turbine_pos']))
p_f = numpy.zeros(len(p_rand))
# Peturb the friction of the first turbine only.
p_f[0] = 1.
i = len(config.params['turbine_friction'])
# Peturb the x position of the first turbine only.
p_x = numpy.zeros(len(p_rand))
p_x[i] = 1.
# Peturb the y position of the first turbine only.
p_y = numpy.zeros(len(p_rand))
p_y[i+1] = 1.

for p in (p_rand, p_f, p_x, p_y):
  print "Running derivative test in direction", p 
  minconv = test_gradient_array(j, dj, m0, seed=0.01, perturbation_direction=p)
  if minconv < 1.98:
    sys.exit(1)

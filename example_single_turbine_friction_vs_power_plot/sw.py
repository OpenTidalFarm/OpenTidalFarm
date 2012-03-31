''' An example that plots the power ouput of a single turbine for different fricition values. (see also test_optimal_friction_single_turbine) 
 - a single turbine (with a bump function as friction distribution)
 - shallow water model with implicit timestepping scheme to avoid oscillations in the turbine areas 
 - the functional is \int C * f * ||u||**3 where C is a constant
 '''

import sys
import sw_config 
import sw_lib
import numpy
import Memoize
import ipopt 
import IPOptUtils
import matplotlib.pyplot as plt
from functionals import DefaultFunctional, build_turbine_cache
from dolfin import *
from dolfin_adjoint import *
from sw_utils import test_initial_condition_adjoint, test_gradient_array, pprint
from turbines import *

# Global counter variable for vtk output
count = 0

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  numpy.random.seed(21) 
  config = sw_config.DefaultConfiguration(nx=20, ny=10)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  # Start at rest state
  config.params["start_time"] = period/4 
  config.params["finish_time"] = period/2
  config.params["dt"] = config.params["finish_time"]/10
  pprint("Wave period (in h): ", period/60/60)
  config.params["dump_period"] = 1
  config.params["verbose"] = 0
  # We need a implicit scheme to avoid oscillations in the turbine areas.
  config.params["theta"] = 1.0

  # Turbine settings
  config.params["friction"] = 0.0025
  config.params["turbine_pos"] = [[1500., 500.]]
  # The turbine friction is the control variable 
  config.params["turbine_friction"] = numpy.ones(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 600
  config.params["turbine_y"] = 600
  # Solver options
  config.params["picard_iterations"] = 4

  return config

def initial_control(config):
  # We use the current turbine settings as the intial control
  res = config.params['turbine_friction'].tolist()
  return numpy.array(res)

def j(m):
  adj_reset()

  # Change the control variables to the config parameters
  config.params["turbine_friction"] = m[:len(config.params["turbine_friction"])]
  mp = m[len(config.params["turbine_friction"]):]

  set_log_level(30)
  debugging["record_all"] = True

  W = sw_lib.p1dgp2(config.mesh)

  # Set initial conditions
  state = Function(W, name = "current_state")
  state.interpolate(config.get_sin_initial_condition()())

  # Set the control values
  U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
  U = U.collapse() # Recompute the DOF map
  tf = Function(U, name = "turbine") # The turbine function
  tfd = Function(U, name = "turbine_derivative") # The derivative turbine function

  # Set up the turbine friction field using the provided control variable
  tf.interpolate(Turbines(config.params))

  global count
  count += 1
  sw_lib.save_to_file_scalar(tf, "turbines_t=."+str(count)+".x")

  turbine_cache = build_turbine_cache(config.params, U, turbine_size_scaling=0.5)
  functional = DefaultFunctional(config.params, turbine_cache)

  # Solve the shallow water system
  j, djdm = sw_lib.sw_solve(W, config, state, turbine_field = tf, time_functional = functional)

  return j

config = default_config()

# Generate the friction values of interest
m0 = initial_control(config)
f = [m0*i for i in numpy.linspace(0., 0.1, 20)]
info_green("Tested friction coefficients: " + str([fx[0] for fx in f]))

# Produce the power values for linear friction
info_green("Compute values for linear friction")
P = []
for fr in f: 
  P.append(j(fr))

# Produce the power values for quadratic friction
info_green("Compute values for quadratic friction")
config.params["quadratic_friction"] = True 
P_quad = []
for fr in f: 
  P_quad.append(j(fr))

# Plot the results
if MPI.process_number() == 0:
  plt.figure(1)
  plt.plot(f, P)
  info_green("Linear friction: The maximum functional value of " + str(max(P)) + " is achieved with a friction coefficient of " + str(f[numpy.argmax(P)]) + ".")
  plt.title('Power output of a single turbine with linear friction.')
  plt.ylabel('Power output')
  plt.xlabel('Linear friction coefficient')

  plt.figure(2)
  plt.plot(f, P_quad)
  info_green("Quadratic friction: The maximum functional value of " + str(max(P_quad)) + " is achieved with a friction coefficient of " + str(f[numpy.argmax(P_quad)]) + ".")
  plt.title('Power output of a single turbine with quadratic friction.')
  plt.ylabel('Power output')
  plt.xlabel('Quadratic friction coefficient')
  plt.show()
  plt.hold()

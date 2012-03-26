''' This example optimises the position of three turbines using the hallow water model. '''

import sys
import cProfile
import pstats
import sw_config 
import sw_lib
import numpy
import Memoize
from animated_plot import *
from functionals import DefaultFunctional
from sw_utils import test_initial_condition_adjoint, test_gradient_array, pprint
from turbines import *
from mini_model import *
from dolfin import *
from dolfin_adjoint import *

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  config = sw_config.DefaultConfiguration(nx=100, ny=30)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  pprint("Wave period (in h): ", period/60/60)
  config.params["dump_period"] = 1
  config.params["verbose"] = 0

  # Start at rest state
  config.params["start_time"] = period/4
  config.params["dt"] = period/20
  config.params["finish_time"] = 2*period/4 
  config.params["theta"] = 0.6

  set_log_level(DEBUG)
  #dolfin.parameters['optimize'] = True
  #dolfin.parameters['optimize_use_dofmap_cache'] = True
  #dolfin.parameters['optimize_use_tensor_cache'] = True
  #dolfin.parameters['form_compiler']['optimize'] = True
  dolfin.parameters['form_compiler']['cpp_optimize'] = True
  dolfin.parameters['form_compiler']['cpp_optimize_flags'] = '-O3'

  # Turbine settings
  config.params["friction"] = 0.0025
  # The turbine position is the control variable 
  config.params["turbine_pos"] = [] 
  border = 100
  for x_r in numpy.linspace(0.+border, config.params["basin_x"]-border, 7):
    for y_r in numpy.linspace(0.+border, config.params["basin_y"]-border, 3):
      config.params["turbine_pos"].append((float(x_r), float(y_r)))

  info_blue("Deployed " + str(len(config.params["turbine_pos"])) + " turbines.")
  # Choosing a friction coefficient of 1.0 ensures that overlapping turbines will lead to
  # less power output.
  config.params["turbine_friction"] = numpy.ones(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 100
  config.params["turbine_y"] = 100

  return config

def initial_control(config):
  # We use the current turbine settings as the intial control
  res = numpy.reshape(config.params['turbine_pos'], -1).tolist()
  return numpy.array(res)

def j(m):
  adjointer.reset()
  adj_variables.__init__()

  # Change the control variables to the config parameters
  config.params["turbine_pos"] = numpy.reshape(m, (-1, 2))

  set_log_level(30)
  debugging["record_all"] = True

  W=sw_lib.p1dgp2(config.mesh)
  state=Function(W)
  state.interpolate(config.get_sin_initial_condition()())

  # Set the control values
  U = W.split()[0].sub(0) # Extract the first component of the velocity function space 
  U = U.collapse() # Recompute the DOF map
  tf = Function(U) # The turbine function

  # Set up the turbine friction field using the provided control variable
  turbines = Turbines(config.params)
  tf.interpolate(turbines)

  sw_lib.save_to_file_scalar(tf, "turbines_t=.0.x")

  # Scale the turbines in the functional for a physically consistent power/friction curve
  functional = DefaultFunctional(config.params, turbine_size_scaling=0.5)

  # Solve the shallow water system
  j, djdm, state = sw_lib.sw_solve(W, config, state, turbine_field = tf, time_functional=functional)

  return j

config = default_config()
m0 = initial_control(config)
j0 = j(m0)
pprint("Power outcome: ", j0)
pprint("Timing summary:")
timer = Timer("NULL")
timer.stop()

list_timings()


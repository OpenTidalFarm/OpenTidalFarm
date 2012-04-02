''' This example optimises the position of three turbines using the hallow water model. '''

import sys
import cProfile
import pstats
import sw_config 
import sw_lib
import numpy
import Memoize
from animated_plot import *
from functionals import DefaultFunctional, build_turbine_cache
from sw_utils import test_initial_condition_adjoint, test_gradient_array, pprint
from turbines import *
from mini_model import *
from dolfin import *
from dolfin_adjoint import *

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  config = sw_config.DefaultConfiguration(nx=100, ny=33)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  pprint("Wave period (in h): ", period/60/60)
  config.params["dump_period"] = 1
  config.params["verbose"] = 0

  # Start at rest state
  config.params["start_time"] = period/4
  config.params["dt"] = period/50
  config.params["finish_time"] = 5.*period/4 
  config.params["theta"] = 0.6
  config.params["include_advection"] = True 
  config.params["include_diffusion"] = True 
  config.params["diffusion_coef"] = 2.0
  config.params["newton_solver"] = False 
  config.params["picard_iterations"] = 20
  config.params['basename'] = "p2p1"
  config.params["run_benchmark"] = False 
  config.params['solver_exclude'] = ['cg', 'lu']
  info_green("Approximate CFL number (assuming a velocity of 2): " +str(2*config.params["dt"]/config.mesh.hmin())) 

  #set_log_level(DEBUG)
  set_log_level(20)
  #dolfin.parameters['optimize'] = True
  #dolfin.parameters['optimize_use_dofmap_cache'] = True
  #dolfin.parameters['optimize_use_tensor_cache'] = True
  #dolfin.parameters['form_compiler']['optimize'] = True
  dolfin.parameters['form_compiler']['cpp_optimize'] = True
  dolfin.parameters['form_compiler']['cpp_optimize_flags'] = '-O3'

  # Turbine settings
  config.params["quadratic_friction"] = True
  config.params["friction"] = 0.0025
  # The turbine position is the control variable 
  config.params["turbine_pos"] = [] 
  border_x = 500
  border_y = 300
  for x_r in numpy.linspace(0.+border_x, config.params["basin_x"]-border_x, 3):
    for y_r in numpy.linspace(0.+border_y, config.params["basin_y"]-border_y, 2):
      config.params["turbine_pos"].append((float(x_r), float(y_r)))

  info_blue("Deployed " + str(len(config.params["turbine_pos"])) + " turbines.")
  # Choosing a friction coefficient of > 0.02 ensures that overlapping turbines will lead to
  # less power output.
  config.params["turbine_friction"] = 0.2*numpy.ones(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 200
  config.params["turbine_y"] = 200

  return config

def initial_control(config):
  # We use the current turbine settings as the intial control
  res = numpy.reshape(config.params['turbine_pos'], -1).tolist()
  return numpy.array(res)

def j(m):
  adj_reset()

  # Change the control variables to the config parameters
  config.params["turbine_pos"] = numpy.reshape(m, (-1, 2))

  debugging["record_all"] = True

  W = sw_lib.p2p1(config.mesh)
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
  turbine_cache = build_turbine_cache(config.params, U, turbine_size_scaling=0.5)
  functional = DefaultFunctional(config.params, turbine_cache)

  # Solve the shallow water system
  j, djdm = sw_lib.sw_solve(W, config, state, turbine_field = tf, time_functional=functional, linear_solver='lu', preconditioner='none')

  return j

config = default_config()
m0 = initial_control(config)
j0 = j(m0)
pprint("Power outcome: ", j0)
pprint("Timing summary:")
timer = Timer("NULL")
timer.stop()

list_timings()


''' This example optimises the position of three turbines using the hallow water model. '''

import configuration 
import function_spaces
import numpy
from reduced_functional import ReducedFunctional
from dolfin import *

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  config = configuration.DefaultConfiguration(nx=100, ny=33)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  info("Wave period (in h): %f" % (period/60/60) )
  config.params["dump_period"] = 1
  config.params["verbose"] = 0

  # Start at rest state
  config.params["element_type"] = function_spaces.p2p1
  config.params["start_time"] = period/4
  config.params["dt"] = period/50
  config.params["finish_time"] = 5.*period/4 
  config.params["theta"] = 0.6
  config.params["include_advection"] = True 
  config.params["include_diffusion"] = True 
  config.params["diffusion_coef"] = 2.0
  config.params["newton_solver"] = False 
  config.params["picard_iterations"] = 20
  config.params["run_benchmark"] = False 
  config.params['solver_exclude'] = ['cg', 'lu']
  info_green("Approximate CFL number (assuming a velocity of 2): " +str(2*config.params["dt"]/config.mesh.hmin())) 


  set_log_level(PROGRESS)
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

config = default_config()
model = ReducedFunctional(config)

m0 = model.initial_control()
j0 = model.j(m0, forward_only = True)
info("Power outcome: %f" % (j0, ))
info("Timing summary:")
timer = Timer("NULL")
timer.stop()

list_timings()


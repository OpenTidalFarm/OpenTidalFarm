''' This tests checks that the power output of a single turbine in a periodic domain is independent of its position ''' 

import sys
import configuration 
import numpy
from dirichlet_bc import DirichletBCSet
import IPOptUtils
import ipopt
from helpers import test_gradient_array
from animated_plot import *
from reduced_functional import ReducedFunctional
from dolfin import *
set_log_level(PROGRESS)

# An animated plot to visualise the development of the functional value
plot = AnimatedPlot(xlabel='Iteration', ylabel='Functional value')

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  numpy.random.seed(21) 
  config = configuration.DefaultConfiguration(nx=100, ny=33)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  info("Wave period (in h): %f" % (period/60/60) )
  config.params["dump_period"] = 1
  config.params["verbose"] = 0

  # Start at rest state
  config.params["start_time"] = period/4
  config.params["dt"] = period/50
  config.params["finish_time"] = 3.*period/4 
  config.params["theta"] = 0.6
  config.params["include_advection"] = True 
  config.params["include_diffusion"] = True 
  config.params["diffusion_coef"] = 2.0
  config.params["newton_solver"] = True 
  config.params["picard_iterations"] = 20
  config.params["linear_solver"] = "default"
  config.params["preconditioner"] = "default"
  config.params["controls"] = ["turbine_pos"]
  info_green("Approximate CFL number (assuming a velocity of 2): " +str(2*config.params["dt"]/config.mesh.hmin())) 

  config.params["bctype"] = "strong_dirichlet"
  bc = DirichletBCSet(config)
  bc.add_analytic_u(config.left)
  bc.add_analytic_u(config.right)
  bc.add_periodic_sides()
  config.params["strong_bc"] = bc

  dolfin.parameters['form_compiler']['cpp_optimize'] = True
  dolfin.parameters['form_compiler']['cpp_optimize_flags'] = '-O3'

  # Turbine settings
  config.params["quadratic_friction"] = True
  config.params["friction"] = 0.0025
  # The turbine position is the control variable 
  config.params["turbine_pos"] = [[1500, 500]] 

  info_blue("Deployed " + str(len(config.params["turbine_pos"])) + " turbines.")
  # Choosing a friction coefficient of > 0.02 ensures that overlapping turbines will lead to
  # less power output.
  config.params["turbine_friction"] = 0.2*numpy.ones(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 200
  config.params["turbine_y"] = 200

  return config

config = default_config()
model = ReducedFunctional(config, scaling_factor = 10**-4, plot = True)
m0 = model.initial_control()
print "Functional value: ", model.j(m0)
print "Derivative value: ", model.dj(m0)

''' This example optimises the position of three turbines using the hallow water model. '''

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
  config.params["finish_time"] = 5.*period/4 
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
  bc.add_noslip_u(config.sides)
  config.params["strong_bc"] = bc

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
model = ReducedFunctional(config, scaling_factor = 10**-4, plot = True)
m0 = model.initial_control()

#p = numpy.random.rand(len(m0))
#minconv = test_gradient_array(model.j, model.dj, m0, seed=0.1, perturbation_direction=p)
#if minconv < 1.9:
#    info_red("The gradient taylor remainder test failed.")
#  sys.exit(1)

g = lambda m: []
dg = lambda m: []

f = IPOptUtils.IPOptFunction()
# Overwrite the functional and gradient function with our implementation
f.objective= model.j 
f.gradient= model.dj 

# Get the upper and lower bounds for the turbine positions
lb, ub = IPOptUtils.position_constraints(config.params)

nlp = ipopt.problem(len(m0), 
                    0, 
                    f, 
                    numpy.array(lb),
                    numpy.array(ub))
nlp.addOption('mu_strategy', 'adaptive')
nlp.addOption('tol', 1e-9)
nlp.addOption('print_level', 5)
nlp.addOption('check_derivatives_for_naninf', 'yes')
# A -1.0 scaling factor transforms the min problem to a max problem.
nlp.addOption('obj_scaling_factor', -1.0)
# Use an approximate Hessian since we do not have second order information.
nlp.addOption('hessian_approximation', 'limited-memory')
nlp.addOption('max_iter', 20)

m, sinfo = nlp.solve(m0)
info(sinfo['status_msg'])
info("Solution of the primal variables: m=%s" % repr(m))
info("Solution of the dual variables: lambda=%s" % repr(sinfo['mult_g']))
info("Objective=%s" % repr(sinfo['obj_val']))

list_timings()

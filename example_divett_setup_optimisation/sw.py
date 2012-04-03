''' This example optimises the position of three turbines using the hallow water model. '''

import sys
import sw_config 
import function_spaces
import numpy
import IPOptUtils
import ipopt
from sw_utils import test_gradient_array
from animated_plot import *
from default_model import DefaultModel
from dolfin import *
from dolfin_adjoint import *

# An animated plot to visualise the development of the functional value
plot = AnimatedPlot(xlabel='Iteration', ylabel='Functional value')

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  numpy.random.seed(21) 
  config = sw_config.DefaultConfiguration(nx=60, ny=20)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  info("Wave period (in h): %f" % (period/60/60) )
  config.params["dump_period"] = 1
  config.params["verbose"] = 0

  # Start at rest state
  config.params["element_type"] = function_spaces.p2p1
  config.params["start_time"] = period/4
  config.params["dt"] = period/2
  config.params["finish_time"] = 5.*period/4 
  config.params["theta"] = 0.6
  config.params["include_advection"] = True 
  config.params["include_diffusion"] = True 
  config.params["diffusion_coef"] = 20.0
  config.params["newton_solver"] = False 
  config.params["picard_iterations"] = 20
  config.params["run_benchmark"] = False 
  config.params['solver_exclude'] = ['cg', 'lu']
  #config.params["controls"] = ["turbine_pos"]
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

config = default_config()
model = DefaultModel(config, scaling_factor = 10**-4)
m0 = model.initial_control()

p = numpy.random.rand(len(m0))
minconv = test_gradient_array(model.j, model.dj, m0, seed=0.00001, perturbation_direction=p)
if minconv < 1.98:
  print "The gradient taylor remainder test failed."
  sys.exit(1)

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
nlp.addOption('max_iter', 12)

m, info = nlp.solve(m0)
pprint(info['status_msg'])
pprint("Solution of the primal variables: m=%s\n" % repr(m))
pprint("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))
pprint("Objective=%s\n" % repr(info['obj_val']))

list_timings()

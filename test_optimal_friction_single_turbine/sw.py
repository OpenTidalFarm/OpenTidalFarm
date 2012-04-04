''' Test description:
 - this test' setup is similar to 'example_single_turbine_friction_vs_power_plot'
 - a single turbine (with a bump function as friction distribution)
 - shallow water model with implicit timestepping scheme to avoid oscillations in the turbine areas 
 - control: turbine friction, initially zero
 - the functional is \int C * f * ||u||**3 where C is a constant
 - in order to avoid the global maximum +oo, the friction coefficient is limited to 0 <= f <= 1.0 
 - the plot in 'example_single_turbine_friction_vs_power_plot' suggestes that the optimal friction coefficient is at about 0.04413 
 '''

import sys
import configuration 
import function_spaces
import numpy
import ipopt 
import IPOptUtils
from dolfin import *
from utils import test_gradient_array 
from dolfin_adjoint import *
from reduced_functional import ReducedFunctional
set_log_level(PROGRESS)

def default_config():
  # We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
  numpy.random.seed(21) 
  config = configuration.DefaultConfiguration(nx=20, ny=10)
  period = 1.24*60*60 # Wave period
  config.params["k"] = 2*pi/(period*sqrt(config.params["g"]*config.params["depth"]))
  # Start at rest state
  config.params["element_type"] = function_spaces.p2p1
  config.params["start_time"] = period/4 
  config.params["finish_time"] = period/2
  config.params["dt"] = config.params["finish_time"]/10
  info_green("Wave period (in h): %f" % (period/60/60, ))
  config.params["dump_period"] = 1
  config.params["verbose"] = 0
  # We need a implicit scheme to avoid oscillations in the turbine areas.
  config.params["theta"] = 1.0

  config.params["include_advection"] = True 
  config.params["include_diffusion"] = True 
  config.params["diffusion_coef"] = 20.0

  # Turbine settings
  config.params["friction"] = 0.0025
  config.params["turbine_pos"] = [[1500., 500.]]
  # The turbine friction is the control variable 
  config.params["turbine_friction"] = numpy.ones(len(config.params["turbine_pos"]))
  config.params["turbine_x"] = 600
  config.params["turbine_y"] = 600
  config.params["controls"] = ["turbine_friction"]
  # Solver options
  config.params["quadratic_friction"] = True
  config.params["picard_iterations"] = 4

  return config

config = default_config()
model = ReducedFunctional(config, scaling_factor = 10**-9)
m0 = model.initial_control()

p = numpy.random.rand(len(m0))
# Note: we choose 0.04*m0 here, because at m0 the problem is almost linear and hence the taylor convergence test will not work.
minconv = test_gradient_array(model.j, model.dj, 0.04*m0, seed=0.001, perturbation_direction=p)
if minconv < 1.98:
  info_red("The gradient taylor remainder test failed.")
  sys.exit(1)

# If this option does not produce any ipopt outputs, delete the ipopt.opt file
g = lambda m: []
dg = lambda m: []

f = IPOptUtils.IPOptFunction()
# Overwrite the functional and gradient function with our implementation
f.objective= model.j 
f.gradient= model.dj 

nlp = ipopt.problem(len(m0), 
                    0, 
                    f, 
                    numpy.zeros(len(m0)), 
                    # Set the maximum friction value to 1.0 to enforce the local minimum at 0.122
                    1.0*numpy.ones(len(m0)))
nlp.addOption('mu_strategy', 'adaptive')
nlp.addOption('tol', 1e-4)
nlp.addOption('print_level', 5)
nlp.addOption('check_derivatives_for_naninf', 'yes')
# Add the internal scaling method so that the first derivtive is arount 1.0
#nlp.addOption('nlp_scaling_max_gradient', 2.0)
# A -1.0 objective scaling factor transforms the min problem to a max problem.
nlp.addOption('obj_scaling_factor', -1.0)
# Use an approximate Hessian since we do not have second order information.
nlp.addOption('hessian_approximation', 'limited-memory')
nlp.addOption('max_iter', 20)

m, sinfo = nlp.solve(m0)
info(sinfo['status_msg'])
info("Solution of the primal variables: m=%s\n" % repr(m))
info("Solution of the dual variables: lambda=%s\n" % repr(sinfo['mult_g']))
info("Objective=%s\n" % repr(sinfo['obj_val']))

if sinfo['status'] != 0 or abs(m-0.04413) > 0.0005: 
  info_red("The optimisation algorithm did not find the correct solution: Expected m = 0.04413, but got m = " + str(m) + ".")
  sys.exit(1) 
else:
  info_green("Test passed")
